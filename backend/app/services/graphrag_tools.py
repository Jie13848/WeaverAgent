"""
GraphRAG 检索工具服务
使用 Microsoft GraphRAG 替代 Zep Cloud 检索工具

完全兼容 zep_tools.py 的所有公共接口和数据类：
  SearchResult, NodeInfo, EdgeInfo, InsightForgeResult, PanoramaResult,
  AgentInterview, InterviewResult, ZepToolsService（→ GraphRAGToolsService 别名）

检索策略对照:
  Zep InsightForge  → GraphRAG local_search  (实体感知深度搜索)
  Zep PanoramaSearch → GraphRAG global_search (社区级宏观搜索)
  Zep QuickSearch   → Parquet 关键词匹配 (轻量快速检索)
"""

import os
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import pandas as pd

from graphrag.api.query import local_search, global_search
from graphrag.config.load_config import load_config
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_covariates,
)
from graphrag.utils.api import get_embedding_store
from graphrag.config.embeddings import entity_description_embedding

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from .graphrag_builder import get_graph_dir, GRAPHRAG_DATA_DIR

logger = get_logger('weaveragent.graphrag_tools')


# ===========================================================
# 数据类（与 zep_tools.py 完全兼容）
# ===========================================================

@dataclass
class SearchResult:
    """搜索结果（兼容 Zep SearchResult）"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count,
        }

    def to_text(self) -> str:
        parts = [f"搜索查询: {self.query}", f"找到 {self.total_count} 条相关信息"]
        if self.facts:
            parts.append("\n### 相关事实:")
            for i, fact in enumerate(self.facts, 1):
                parts.append(f"{i}. {fact}")
        return "\n".join(parts)


@dataclass
class NodeInfo:
    """节点信息（兼容 Zep NodeInfo）"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    source_papers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "source_papers": self.source_papers,
        }

    def to_text(self) -> str:
        entity_type = next(
            (l for l in self.labels if l not in ["Entity", "Node"]), "未知类型"
        )
        src = f"\n来源: {' / '.join(self.source_papers)}" if self.source_papers else ""
        return f"实体: {self.name} (类型: {entity_type}){src}\n摘要: {self.summary}"


@dataclass
class EdgeInfo:
    """边信息（兼容 Zep EdgeInfo）"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    source_papers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at,
            "source_papers": self.source_papers,
        }

    def to_text(self, include_temporal: bool = False) -> str:
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        text = f"关系: {source} --[{self.name}]--> {target}\n事实: {self.fact}"
        if self.source_papers:
            text += f"\n来源: {' / '.join(self.source_papers)}"
        if include_temporal and self.valid_at:
            text += f"\n时效: {self.valid_at} - {self.invalid_at or '至今'}"
        return text

    @property
    def is_expired(self) -> bool:
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """深度洞察检索结果（兼容 Zep InsightForgeResult）"""
    query: str
    analysis_requirement: str
    sub_queries: List[str]
    semantic_facts: List[str] = field(default_factory=list)
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)
    relationship_chains: List[str] = field(default_factory=list)
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "analysis_requirement": self.analysis_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships,
        }

    def to_text(self) -> str:
        parts = [
            f"## 学术文献深度分析",
            f"分析问题: {self.query}",
            f"分析主题: {self.analysis_requirement}",
            f"\n### 文献数据统计",
            f"- 相关文献事实: {self.total_facts}条",
            f"- 涉及实体: {self.total_entities}个",
            f"- 关系链: {self.total_relationships}条",
        ]
        if self.sub_queries:
            parts.append("\n### 分析的子问题")
            for i, sq in enumerate(self.sub_queries, 1):
                parts.append(f"{i}. {sq}")
        if self.semantic_facts:
            parts.append("\n### 【关键文献事实】")
            for i, fact in enumerate(self.semantic_facts, 1):
                parts.append(f'{i}. "{fact}"')
        if self.entity_insights:
            parts.append("\n### 【核心实体】")
            for entity in self.entity_insights:
                parts.append(f"- **{entity.get('name', '未知')}** ({entity.get('type', '实体')})")
                if entity.get("summary"):
                    parts.append(f"  摘要: \"{entity.get('summary')}\"")
        if self.relationship_chains:
            parts.append("\n### 【关系链】")
            for chain in self.relationship_chains:
                parts.append(f"- {chain}")
        return "\n".join(parts)


@dataclass
class PanoramaResult:
    """广度搜索结果（兼容 Zep PanoramaResult）"""
    query: str
    all_nodes: List[NodeInfo] = field(default_factory=list)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    active_facts: List[str] = field(default_factory=list)
    historical_facts: List[str] = field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count,
        }

    def to_text(self) -> str:
        parts = [
            f"## 广度搜索结果（知识图谱全景视图）",
            f"查询: {self.query}",
            f"\n### 统计信息",
            f"- 总节点数: {self.total_nodes}",
            f"- 总边数: {self.total_edges}",
            f"- 当前有效事实: {self.active_count}条",
            f"- 历史/过期事实: {self.historical_count}条",
        ]
        if self.active_facts:
            parts.append("\n### 【当前有效图谱事实】")
            for i, fact in enumerate(self.active_facts, 1):
                parts.append(f'{i}. "{fact}"')
        if self.historical_facts:
            parts.append("\n### 【历史/版本事实】")
            for i, fact in enumerate(self.historical_facts, 1):
                parts.append(f'{i}. "{fact}"')
        if self.all_nodes:
            parts.append("\n### 【涉及实体】")
            for node in self.all_nodes:
                entity_type = next(
                    (l for l in node.labels if l not in ["Entity", "Node"]), "实体"
                )
                parts.append(f"- **{node.name}** ({entity_type})")
        return "\n".join(parts)


@dataclass
class AgentInterview:
    """单个 Agent 采访结果（兼容原接口）"""
    agent_name: str
    agent_role: str
    agent_bio: str
    question: str
    response: str
    key_quotes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes,
        }


@dataclass
class InterviewResult:
    """采访结果集合（兼容原接口）"""
    query: str
    interviews: List[AgentInterview] = field(default_factory=list)
    total_interviews: int = 0
    key_themes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "interviews": [i.to_dict() for i in self.interviews],
            "total_interviews": self.total_interviews,
            "key_themes": self.key_themes,
        }

    def to_text(self) -> str:
        parts = [f"## 采访结果\n查询: {self.query}"]
        for iv in self.interviews:
            parts.append(
                f"\n### {iv.agent_name} ({iv.agent_role})\n"
                f"问: {iv.question}\n"
                f"答: {iv.response}"
            )
        return "\n".join(parts)


# ===========================================================
# 内部 Parquet 读取工具
# ===========================================================

def _load_parquet_safe(path: Path) -> Optional[pd.DataFrame]:
    """安全读取 Parquet 文件，返回 None 如果文件不存在"""
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"读取 Parquet 失败: {path} → {e}")
    return None


def _load_graph_data(graph_id: str) -> Dict[str, Optional[pd.DataFrame]]:
    """加载图谱的所有 Parquet 数据文件"""
    output_dir = get_graph_dir(graph_id) / "output"
    return {
        "entities":          _load_parquet_safe(output_dir / "entities.parquet"),
        "relationships":     _load_parquet_safe(output_dir / "relationships.parquet"),
        "communities":       _load_parquet_safe(output_dir / "communities.parquet"),
        "community_reports": _load_parquet_safe(output_dir / "community_reports.parquet"),
        "text_units":        _load_parquet_safe(output_dir / "text_units.parquet"),
        "covariates":        _load_parquet_safe(output_dir / "covariates.parquet"),
        "documents":         _load_parquet_safe(output_dir / "documents.parquet"),
    }


# ===========================================================
# 论文来源映射（entity/edge → paper title）
# ===========================================================

# 缓存：graph_id → (text_unit_id → [paper_titles])
_PAPER_SOURCE_CACHE: Dict[str, Dict[str, List[str]]] = {}


def _extract_paper_title(doc_text: str, fallback: str) -> str:
    """从 Markdown 文档文本提取真实论文标题（第一个 `# ` 标题）"""
    if not doc_text:
        return fallback
    for line in str(doc_text)[:2000].split("\n"):
        line = line.strip()
        if line.startswith("# ") and len(line) < 250:
            title = line.lstrip("# ").strip()
            # 规范化：多空格合并、去末尾冒号等
            import re as _re
            title = _re.sub(r'\s+', ' ', title).strip(':：')
            if len(title) > 4:
                return title
    return fallback


def _build_paper_source_map(graph_id: str) -> Dict[str, List[str]]:
    """
    构建 text_unit_id → [paper_titles] 的映射，带缓存。

    通过：text_units.document_id → documents.title（或从文本首个 H1 提取真实标题）
    """
    if graph_id in _PAPER_SOURCE_CACHE:
        return _PAPER_SOURCE_CACHE[graph_id]

    data = _load_graph_data(graph_id)
    docs_df = data.get("documents")
    tu_df   = data.get("text_units")

    if docs_df is None or tu_df is None or docs_df.empty or tu_df.empty:
        _PAPER_SOURCE_CACHE[graph_id] = {}
        return {}

    # doc_id → real title
    doc_id_to_title: Dict[str, str] = {}
    for _, r in docs_df.iterrows():
        doc_id = str(r.get("id", ""))
        fallback = str(r.get("title", doc_id) or doc_id)
        real_title = _extract_paper_title(r.get("text", ""), fallback)
        doc_id_to_title[doc_id] = real_title

    # text_unit_id → [title]
    tu_to_titles: Dict[str, List[str]] = {}
    for _, r in tu_df.iterrows():
        tu_id = str(r.get("id", ""))
        doc_id = str(r.get("document_id", "") or "")
        title = doc_id_to_title.get(doc_id)
        if title:
            tu_to_titles[tu_id] = [title]

    _PAPER_SOURCE_CACHE[graph_id] = tu_to_titles
    return tu_to_titles


def _resolve_papers(text_unit_ids, tu_map: Dict[str, List[str]]) -> List[str]:
    """给一组 text_unit_ids 聚合并去重它们对应的论文标题"""
    if text_unit_ids is None:
        return []
    titles: List[str] = []
    seen = set()
    try:
        for tu_id in text_unit_ids:
            for t in tu_map.get(str(tu_id), []):
                if t not in seen:
                    seen.add(t)
                    titles.append(t)
    except TypeError:
        return []
    return titles


def _parquet_to_nodes(
    entities_df: Optional[pd.DataFrame],
    tu_map: Optional[Dict[str, List[str]]] = None,
) -> List[NodeInfo]:
    """将 entities DataFrame 转换为 NodeInfo 列表（附带论文来源）"""
    if entities_df is None or entities_df.empty:
        return []
    tu_map = tu_map or {}
    nodes = []
    for _, row in entities_df.iterrows():
        entity_type = str(row.get("type", "Entity") or "Entity")
        papers = _resolve_papers(row.get("text_unit_ids"), tu_map)
        nodes.append(NodeInfo(
            uuid=str(row.get("id", "")),
            name=str(row.get("title", "")),
            labels=[entity_type],
            summary=str(row.get("description", "") or ""),
            attributes=row.get("attributes") or {},
            source_papers=papers,
        ))
    return nodes


def _parquet_to_edges(
    rels_df: Optional[pd.DataFrame],
    name_to_uuid: Dict[str, str],
    tu_map: Optional[Dict[str, List[str]]] = None,
) -> List[EdgeInfo]:
    """将 relationships DataFrame 转换为 EdgeInfo 列表（附带论文来源）"""
    if rels_df is None or rels_df.empty:
        return []
    tu_map = tu_map or {}
    edges = []
    for _, row in rels_df.iterrows():
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        desc   = str(row.get("description", "") or "")
        papers = _resolve_papers(row.get("text_unit_ids"), tu_map)
        edges.append(EdgeInfo(
            uuid=str(row.get("id", "")),
            name=desc,
            fact=desc,
            source_node_uuid=name_to_uuid.get(source, source),
            target_node_uuid=name_to_uuid.get(target, target),
            source_node_name=source,
            target_node_name=target,
            source_papers=papers,
        ))
    return edges


# ===========================================================
# 主检索服务类
# ===========================================================

class GraphRAGToolsService:
    """
    GraphRAG 检索工具服务（完整替代 ZepToolsService）

    接口与 ZepToolsService 保持完全兼容，包括方法签名和返回类型。
    """

    def __init__(self, api_key: Optional[str] = None):
        # api_key 参数保留以兼容原 ZepToolsService 签名
        self.llm_client = LLMClient()

    # ---------------------------------------------------------
    # 核心检索工具
    # ---------------------------------------------------------

    def search_graph(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        图谱语义搜索（兼容 ZepToolsService.search_graph）

        先尝试 GraphRAG local_search，失败则降级为 Parquet 关键词匹配。
        """
        logger.info(f"图谱搜索: graph_id={graph_id}, query={query[:50]}...")

        try:
            response_text, context_data = self._run_local_search(graph_id, query)

            # 从 local_search 响应中提取事实
            facts = self._extract_facts_from_response(response_text, context_data, limit)

            # 同时做一次 Parquet 关键词匹配补充 edges/nodes
            keyword_result = self._keyword_search(graph_id, query, limit, scope)

            return SearchResult(
                facts=facts or keyword_result.facts,
                edges=keyword_result.edges,
                nodes=keyword_result.nodes,
                query=query,
                total_count=len(facts or keyword_result.facts),
            )
        except Exception as e:
            logger.warning(f"GraphRAG local_search 失败，降级到关键词搜索: {e}")
            return self._keyword_search(graph_id, query, limit, scope)

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        analysis_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        """
        深度洞察检索（对应 Zep InsightForge）

        使用 GraphRAG local_search（实体感知）+ LLM 子问题分解。
        """
        logger.info(f"InsightForge 深度检索: {query[:50]}...")

        # 1. LLM 生成子问题
        sub_queries = self._generate_sub_queries(
            query, analysis_requirement, report_context, max_sub_queries
        )

        # 2. 对主查询 + 子查询分别做 local_search
        all_facts: List[str] = []
        relationship_chains: List[str] = []

        for sq in [query] + sub_queries:
            try:
                response_text, context_data = self._run_local_search(graph_id, sq)
                facts = self._extract_facts_from_response(response_text, context_data, 20)
                all_facts.extend(facts)
            except Exception as e:
                err_str = str(e)
                if "dim" in err_str or "lance" in err_str.lower() or "vector" in err_str.lower():
                    logger.warning(f"GraphRAG local_search 失败，降级到关键词搜索: {err_str[:100]}")
                    fallback = self._keyword_search(graph_id, sq, 10, "edges")
                    all_facts.extend(fallback.facts)
                else:
                    logger.debug(f"子查询失败: {sq[:30]} → {e}")

        # 3. 读取 Parquet 获取实体信息
        data = _load_graph_data(graph_id)
        tu_map = _build_paper_source_map(graph_id)
        nodes = _parquet_to_nodes(data["entities"], tu_map)

        entity_insights = []
        for node in nodes[:30]:
            entity_insights.append({
                "name": node.name,
                "type": node.labels[0] if node.labels else "Entity",
                "summary": node.summary,
                "uuid": node.uuid,
                "source_papers": node.source_papers,
            })

        # 4. 构建关系链
        name_to_uuid = {n.name: n.uuid for n in nodes}
        edges = _parquet_to_edges(data["relationships"], name_to_uuid, tu_map)
        for edge in edges[:50]:
            src_suffix = f" [{'/'.join(edge.source_papers)}]" if edge.source_papers else ""
            chain = f"{edge.source_node_name} → [{edge.name[:40]}] → {edge.target_node_name}{src_suffix}"
            relationship_chains.append(chain)

        # 去重
        all_facts = list(dict.fromkeys(all_facts))

        return InsightForgeResult(
            query=query,
            analysis_requirement=analysis_requirement,
            sub_queries=sub_queries,
            semantic_facts=all_facts,
            entity_insights=entity_insights,
            relationship_chains=relationship_chains,
            total_facts=len(all_facts),
            total_entities=len(entity_insights),
            total_relationships=len(relationship_chains),
        )

    def panorama_search(
        self,
        graph_id: str,
        query: str
    ) -> PanoramaResult:
        """
        广度搜索（对应 Zep PanoramaSearch）

        使用 GraphRAG global_search（社区级宏观搜索）+ Parquet 全量读取。
        """
        logger.info(f"PanoramaSearch 广度搜索: {query[:50]}...")

        # 读取全量 Parquet 数据
        data = _load_graph_data(graph_id)
        tu_map = _build_paper_source_map(graph_id)
        nodes = _parquet_to_nodes(data["entities"], tu_map)
        name_to_uuid = {n.name: n.uuid for n in nodes}
        edges = _parquet_to_edges(data["relationships"], name_to_uuid, tu_map)

        # 尝试 global_search 获取社区摘要级事实
        active_facts: List[str] = []
        try:
            response_text, _ = self._run_global_search(graph_id, query)
            if response_text:
                # 将 global_search 的段落分解为事实列表
                for line in response_text.split("\n"):
                    line = line.strip()
                    if line and len(line) > 20:
                        active_facts.append(line)
        except Exception as e:
            logger.warning(f"GraphRAG global_search 失败: {e}")
            # 降级：从边的 fact 中提取
            active_facts = [e.fact for e in edges if e.fact][:100]

        return PanoramaResult(
            query=query,
            all_nodes=nodes,
            all_edges=edges,
            active_facts=active_facts,
            historical_facts=[],   # GraphRAG 无时间版本概念
            total_nodes=len(nodes),
            total_edges=len(edges),
            active_count=len(active_facts),
            historical_count=0,
        )

    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        """
        快速检索（对应 Zep QuickSearch）

        直接在 Parquet 文件上做关键词匹配，轻量快速。
        """
        logger.info(f"QuickSearch: {query[:50]}...")
        return self._keyword_search(graph_id, query, limit, "edges")

    def deep_entity_query(
        self,
        graph_id: str,
        entity_type: str,
        query: str = "",
        max_entities: int = 5
    ) -> List[NodeInfo]:
        """按实体类型深度查询（兼容原接口）"""
        nodes = self.get_entities_by_type(graph_id, entity_type)
        if query:
            query_lower = query.lower()
            nodes = [n for n in nodes if query_lower in (n.name or "").lower()
                     or query_lower in (n.summary or "").lower()]
        return nodes[:max_entities]

    # ---------------------------------------------------------
    # 节点 / 边 读取方法（与 ZepToolsService 完全兼容）
    # ---------------------------------------------------------

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """获取所有节点"""
        data = _load_graph_data(graph_id)
        tu_map = _build_paper_source_map(graph_id)
        return _parquet_to_nodes(data["entities"], tu_map)

    def get_all_edges(
        self,
        graph_id: str,
        include_temporal: bool = True
    ) -> List[EdgeInfo]:
        """获取所有边"""
        data = _load_graph_data(graph_id)
        tu_map = _build_paper_source_map(graph_id)
        nodes = _parquet_to_nodes(data["entities"], tu_map)
        name_to_uuid = {n.name: n.uuid for n in nodes}
        return _parquet_to_edges(data["relationships"], name_to_uuid, tu_map)

    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """获取单个节点详情（跨所有图谱搜索）"""
        for graph_dir in Path(GRAPHRAG_DATA_DIR).iterdir():
            if not graph_dir.is_dir():
                continue
            entities_file = graph_dir / "output" / "entities.parquet"
            if not entities_file.exists():
                continue
            df = _load_parquet_safe(entities_file)
            if df is None:
                continue
            row = df[df["id"].astype(str) == node_uuid]
            if not row.empty:
                r = row.iloc[0]
                return NodeInfo(
                    uuid=str(r.get("id", "")),
                    name=str(r.get("title", "")),
                    labels=[str(r.get("type", "Entity") or "Entity")],
                    summary=str(r.get("description", "") or ""),
                    attributes=r.get("attributes") or {},
                )
        return None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """获取节点相关的所有边"""
        all_edges = self.get_all_edges(graph_id)
        return [
            e for e in all_edges
            if e.source_node_uuid == node_uuid or e.target_node_uuid == node_uuid
        ]

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str
    ) -> List[NodeInfo]:
        """按类型过滤实体"""
        all_nodes = self.get_all_nodes(graph_id)
        return [n for n in all_nodes if entity_type in n.labels]

    def get_entity_summary(
        self,
        graph_id: str,
        entity_name: str
    ) -> Dict[str, Any]:
        """获取实体关系摘要"""
        search_result = self.search_graph(graph_id, entity_name, limit=20)
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = next(
            (n for n in all_nodes if n.name.lower() == entity_name.lower()), None
        )
        related_edges = self.get_node_edges(graph_id, entity_node.uuid) if entity_node else []
        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges),
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """获取图谱统计信息"""
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)

        entity_types: Dict[str, int] = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1

        relation_types: Dict[str, int] = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1

        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types,
        }

    def get_analysis_context(
        self,
        graph_id: str,
        analysis_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """获取分析上下文"""
        search_result = self.search_graph(graph_id, analysis_requirement, limit=limit)
        stats = self.get_graph_statistics(graph_id)
        all_nodes = self.get_all_nodes(graph_id)

        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary,
                })

        return {
            "analysis_requirement": analysis_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities),
        }

    # 向后兼容别名
    def get_simulation_context(
        self,
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        return self.get_analysis_context(graph_id, simulation_requirement, limit)

    # ---------------------------------------------------------
    # GraphRAG 内部搜索方法
    # ---------------------------------------------------------

    @staticmethod
    def _retry_on_rate_limit(func, max_retries=5, base_delay=2.0):
        """对 429 Rate Limit 错误自动重试（指数退避）"""
        last_exc = None
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower() or "Rate limit" in err_str:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, retry {attempt+1}/{max_retries} after {delay:.1f}s")
                    time.sleep(delay)
                    last_exc = e
                else:
                    raise
        raise last_exc

    def _run_local_search(
        self,
        graph_id: str,
        query: str,
        community_level: int = 2
    ):
        """运行 GraphRAG local_search（同步包装，含 429 重试）"""
        graph_dir = get_graph_dir(graph_id)
        output_dir = graph_dir / "output"

        if not output_dir.exists():
            raise FileNotFoundError(f"图谱尚未索引: {graph_id}")

        config = load_config(str(graph_dir))
        data = _load_graph_data(graph_id)

        if data["entities"] is None or data["entities"].empty:
            raise ValueError("实体数据为空，请先完成图谱索引")

        entities_df          = data["entities"]
        communities_df       = data["communities"]       if data["communities"] is not None else pd.DataFrame()
        community_reports_df = data["community_reports"] if data["community_reports"] is not None else pd.DataFrame()
        text_units_df        = data["text_units"]        if data["text_units"] is not None else pd.DataFrame()
        relationships_df     = data["relationships"]     if data["relationships"] is not None else pd.DataFrame()
        covariates_df        = data["covariates"]

        async def _run():
            return await local_search(
                config=config,
                entities=entities_df,
                communities=communities_df,
                community_reports=community_reports_df,
                text_units=text_units_df,
                relationships=relationships_df,
                covariates=covariates_df,
                community_level=community_level,
                response_type="multiple paragraphs",
                query=query,
            )

        def _sync_run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_run())
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        return self._retry_on_rate_limit(_sync_run)

    def _run_global_search(
        self,
        graph_id: str,
        query: str,
        community_level: int = 1
    ):
        """运行 GraphRAG global_search（同步包装，含 429 重试）"""
        graph_dir = get_graph_dir(graph_id)
        output_dir = graph_dir / "output"

        if not output_dir.exists():
            raise FileNotFoundError(f"图谱尚未索引: {graph_id}")

        config = load_config(str(graph_dir))
        data = _load_graph_data(graph_id)

        if data["entities"] is None or data["entities"].empty:
            raise ValueError("实体数据为空，请先完成图谱索引")

        entities_df          = data["entities"]
        communities_df       = data["communities"]       if data["communities"] is not None else pd.DataFrame()
        community_reports_df = data["community_reports"] if data["community_reports"] is not None else pd.DataFrame()

        async def _run():
            return await global_search(
                config=config,
                entities=entities_df,
                communities=communities_df,
                community_reports=community_reports_df,
                community_level=community_level,
                dynamic_community_selection=False,
                response_type="multiple paragraphs",
                query=query,
            )

        def _sync_run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_run())
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        return self._retry_on_rate_limit(_sync_run)

    def _extract_facts_from_response(
        self,
        response_text: str,
        context_data: Any,
        limit: int
    ) -> List[str]:
        """从 GraphRAG 响应中提取事实列表"""
        facts = []

        # 从 context_data 中提取结构化信息
        if isinstance(context_data, dict):
            for key in ("entities", "relationships", "reports", "sources"):
                if key in context_data:
                    items = context_data[key]
                    if isinstance(items, list):
                        for item in items[:limit]:
                            if isinstance(item, dict):
                                desc = item.get("description") or item.get("content") or ""
                                if desc:
                                    facts.append(str(desc))
                    elif hasattr(items, "iterrows"):
                        for _, row in items.iterrows():
                            desc = str(row.get("description", "") or row.get("content", "") or "")
                            if desc:
                                facts.append(desc)

        # 若无结构化数据，从文本响应中按行提取
        if not facts and response_text:
            for line in response_text.split("\n"):
                line = line.strip().lstrip("•-*123456789. ")
                if len(line) > 20:
                    facts.append(line)

        return list(dict.fromkeys(facts))[:limit]

    # ---------------------------------------------------------
    # 关键词搜索（Parquet 直接搜索，作为降级方案）
    # ---------------------------------------------------------

    def _keyword_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """基于 Parquet 文件的关键词匹配搜索"""
        logger.debug(f"关键词搜索: {query[:30]}...")

        query_lower = query.lower()
        keywords = [
            w.strip()
            for w in query_lower.replace(",", " ").replace("，", " ").split()
            if len(w.strip()) > 1
        ]

        def match_score(text: str) -> int:
            if not text:
                return 0
            text_lower = text.lower()
            if query_lower in text_lower:
                return 100
            return sum(10 for kw in keywords if kw in text_lower)

        data = _load_graph_data(graph_id)
        tu_map = _build_paper_source_map(graph_id)
        nodes = _parquet_to_nodes(data["entities"], tu_map)
        name_to_uuid = {n.name: n.uuid for n in nodes}
        edges = _parquet_to_edges(data["relationships"], name_to_uuid, tu_map)

        facts: List[str] = []
        edges_result: List[Dict] = []
        nodes_result: List[Dict] = []

        if scope in ("edges", "both"):
            scored = sorted(
                [(match_score(e.fact) + match_score(e.name), e) for e in edges if match_score(e.fact) + match_score(e.name) > 0],
                key=lambda x: x[0],
                reverse=True
            )
            for score, edge in scored[:limit]:
                if edge.fact:
                    src = f" 【来源: {' / '.join(edge.source_papers)}】" if edge.source_papers else ""
                    facts.append(f"{edge.source_node_name} → {edge.target_node_name}: {edge.fact}{src}")
                edges_result.append(edge.to_dict())

        if scope in ("nodes", "both"):
            scored = sorted(
                [(match_score(n.name) + match_score(n.summary), n) for n in nodes if match_score(n.name) + match_score(n.summary) > 0],
                key=lambda x: x[0],
                reverse=True
            )
            for score, node in scored[:limit]:
                nodes_result.append(node.to_dict())
                if node.summary:
                    src = f" 【来源: {' / '.join(node.source_papers)}】" if node.source_papers else ""
                    facts.append(f"[{node.name}]: {node.summary}{src}")

        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts),
        )

    # ---------------------------------------------------------
    # LLM 辅助方法
    # ---------------------------------------------------------

    def _generate_sub_queries(
        self,
        query: str,
        analysis_requirement: str,
        context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """使用 LLM 生成子问题（与原 ZepToolsService 行为一致）"""
        prompt = f"""请为以下分析问题生成 {max_queries} 个具体的子问题，以便更全面地检索知识图谱。

主问题: {query}
分析主题: {analysis_requirement}
{f'背景信息: {context[:500]}' if context else ''}

要求:
- 每行一个子问题
- 子问题应从不同维度分析主题
- 聚焦于可从知识图谱中检索的具体信息
- 直接输出问题列表，无需编号或其他格式

子问题列表:"""

        try:
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024
            )
            sub_queries = [
                line.strip().lstrip("•-*0123456789. ")
                for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) > 5
            ]
            return sub_queries[:max_queries]
        except Exception as e:
            logger.warning(f"子问题生成失败: {e}")
            return []


# ===========================================================
# 向后兼容别名（确保 report_agent.py 无需修改）
# ===========================================================

# ZepToolsService → GraphRAGToolsService 别名
ZepToolsService = GraphRAGToolsService
