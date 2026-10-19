"""
Zep检索工具服务
封装图谱搜索、节点读取、边查询等工具，供 Report Agent 使用

完整检索流程：
  LLM抽取实体关系 → Zep Cloud构建GraphRAG → 同步到本地Neo4j → 使用自有检索方法

核心检索工具：
1. InsightForge（深度洞察检索）- 最强大的混合检索，自动生成子问题并多维度检索
2. PanoramaSearch（广度搜索）- 获取知识图谱全貌，包括历史版本
3. QuickSearch（快速检索）- 轻量级直接检索
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger('weaveragent.zep_tools')


@dataclass
class SearchResult:
    """搜索结果"""
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
            "total_count": self.total_count
        }
    
    def to_text(self) -> str:
        """转换为文本格式，供LLM理解"""
        text_parts = [f"搜索查询: {self.query}", f"找到 {self.total_count} 条相关信息"]
        
        if self.facts:
            text_parts.append("\n### 相关事实:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")
        
        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """节点信息"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }
    
    def to_text(self) -> str:
        """转换为文本格式"""
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "未知类型")
        return f"实体: {self.name} (类型: {entity_type})\n摘要: {self.summary}"


@dataclass
class EdgeInfo:
    """边信息"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # 时间信息
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    
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
            "expired_at": self.expired_at
        }
    
    def to_text(self, include_temporal: bool = False) -> str:
        """转换为文本格式"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"关系: {source} --[{self.name}]--> {target}\n事实: {self.fact}"
        
        if include_temporal:
            valid_at = self.valid_at or "未知"
            invalid_at = self.invalid_at or "至今"
            base_text += f"\n时效: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (已过期: {self.expired_at})"
        
        return base_text
    
    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        return self.expired_at is not None
    
    @property
    def is_invalid(self) -> bool:
        """是否已失效"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    深度洞察检索结果 (InsightForge)
    包含多个子问题的检索结果，以及综合分析
    """
    query: str
    analysis_requirement: str        # 分析主题/需求描述
    sub_queries: List[str]
    
    # 各维度检索结果
    semantic_facts: List[str] = field(default_factory=list)  # 语义搜索结果
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)  # 实体洞察
    relationship_chains: List[str] = field(default_factory=list)  # 关系链
    
    # 统计信息
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
            "total_relationships": self.total_relationships
        }
    
    def to_text(self) -> str:
        """转换为详细的文本格式，供LLM理解"""
        text_parts = [
            f"## 学术文献深度分析",
            f"分析问题: {self.query}",
            f"分析主题: {self.analysis_requirement}",
            f"\n### 文献数据统计",
            f"- 相关文献事实: {self.total_facts}条",
            f"- 涉及实体: {self.total_entities}个",
            f"- 关系链: {self.total_relationships}条"
        ]
        
        # 子问题
        if self.sub_queries:
            text_parts.append(f"\n### 分析的子问题")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")
        
        # 语义搜索结果
        if self.semantic_facts:
            text_parts.append(f"\n### 【关键文献事实】(请在报告中引用这些原文)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # 实体洞察
        if self.entity_insights:
            text_parts.append(f"\n### 【核心实体】")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', '未知')}** ({entity.get('type', '实体')})")
                if entity.get('summary'):
                    text_parts.append(f"  摘要: \"{entity.get('summary')}\"")
                if entity.get('related_facts'):
                    text_parts.append(f"  相关文献事实: {len(entity.get('related_facts', []))}条")
        
        # 关系链
        if self.relationship_chains:
            text_parts.append(f"\n### 【关系链】")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")
        
        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    广度搜索结果 (Panorama)
    包含所有相关信息，包括过期内容
    """
    query: str
    
    # 全部节点
    all_nodes: List[NodeInfo] = field(default_factory=list)
    # 全部边（包括过期的）
    all_edges: List[EdgeInfo] = field(default_factory=list)
    # 当前有效的事实
    active_facts: List[str] = field(default_factory=list)
    # 已过期/失效的事实（历史记录）
    historical_facts: List[str] = field(default_factory=list)
    
    # 统计
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
            "historical_count": self.historical_count
        }
    
    def to_text(self) -> str:
        """转换为文本格式（完整版本，不截断）"""
        text_parts = [
            f"## 广度搜索结果（知识图谱全景视图）",
            f"查询: {self.query}",
            f"\n### 统计信息",
            f"- 总节点数: {self.total_nodes}",
            f"- 总边数: {self.total_edges}",
            f"- 当前有效事实: {self.active_count}条",
            f"- 历史/过期事实: {self.historical_count}条"
        ]
        
        # 当前有效的事实（完整输出，不截断）
        if self.active_facts:
            text_parts.append(f"\n### 【当前有效图谱事实】")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # 历史/过期事实（完整输出，不截断）
        if self.historical_facts:
            text_parts.append(f"\n### 【历史/版本事实】(早期版本记录)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # 关键实体（完整输出，不截断）
        if self.all_nodes:
            text_parts.append(f"\n### 【涉及实体】")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "实体")
                text_parts.append(f"- **{node.name}** ({entity_type})")
        
        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """单个Agent的采访结果"""
    agent_name: str
    agent_role: str  # 角色类型（如：学生、教师、媒体等）
    agent_bio: str  # 简介
    question: str  # 采访问题
    response: str  # 采访回答
    key_quotes: List[str] = field(default_factory=list)  # 关键引言
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }
    
    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        # 显示完整的agent_bio，不截断
        text += f"_简介: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**关键引言:**\n"
            for quote in self.key_quotes:
                # 清理各种引号
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                # 去掉开头的标点
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                # 过滤包含问题编号的垃圾内容（问题1-9）
                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                # 截断过长内容（按句号截断，而非硬截断）
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """
    采访结果 (Interview)
    包含多个模拟Agent的采访回答
    """
    interview_topic: str  # 采访主题
    interview_questions: List[str]  # 采访问题列表
    
    # 采访选择的Agent
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    # 各Agent的采访回答
    interviews: List[AgentInterview] = field(default_factory=list)
    
    # 选择Agent的理由
    selection_reasoning: str = ""
    # 整合后的采访摘要
    summary: str = ""
    
    # 统计
    total_agents: int = 0
    interviewed_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }
    
    def to_text(self) -> str:
        """转换为详细的文本格式，供LLM理解和报告引用"""
        text_parts = [
            "## 深度采访报告",
            f"**采访主题:** {self.interview_topic}",
            f"**采访人数:** {self.interviewed_count} / {self.total_agents} 位模拟Agent",
            "\n### 采访对象选择理由",
            self.selection_reasoning or "（自动选择）",
            "\n---",
            "\n### 采访实录",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### 采访 #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("（无采访记录）\n\n---")

        text_parts.append("\n### 采访摘要与核心观点")
        text_parts.append(self.summary or "（无摘要）")

        return "\n".join(text_parts)


class ZepToolsService:
    """
    Zep 检索工具服务

    【完整检索流程】
    LLM抽取实体关系 → Zep Cloud构建GraphRAG → 同步到本地Neo4j → 使用自有检索方法

    【核心检索工具】
    1. insight_forge     - 深度洞察检索（最强大，自动生成子问题，多维度检索）
    2. panorama_search   - 广度搜索（获取知识图谱全貌，含历史版本）
    3. quick_search      - 快速检索（轻量级）
    4. deep_entity_query - 学术实体深度查询（按实体类型展开关联文献）

    【基础工具】
    - search_graph        - 图谱语义搜索（Zep Cloud + 本地Neo4j降级）
    - get_all_nodes       - 获取图谱所有节点
    - get_all_edges       - 获取图谱所有边（含时间信息）
    - get_node_detail     - 获取节点详细信息
    - get_node_edges      - 获取节点相关的边
    - get_entities_by_type- 按类型获取实体
    - get_entity_summary  - 获取实体的关系摘要
    """
    
    # 重试配置
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    
    def __init__(self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY 未配置")
        
        self.client = Zep(api_key=self.api_key)
        # LLM客户端用于InsightForge生成子问题
        self._llm_client = llm_client
        logger.info("ZepToolsService 初始化完成")
    
    @property
    def llm(self) -> LLMClient:
        """延迟初始化LLM客户端"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client
    
    def _call_with_retry(self, func, operation_name: str, max_retries: int = None):
        """带重试机制的API调用"""
        max_retries = max_retries or self.MAX_RETRIES
        last_exception = None
        delay = self.RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Zep {operation_name} 第 {attempt + 1} 次尝试失败: {str(e)[:100]}, "
                        f"{delay:.1f}秒后重试..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Zep {operation_name} 在 {max_retries} 次尝试后仍失败: {str(e)}")
        
        raise last_exception
    
    def search_graph(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        图谱语义搜索
        
        使用混合搜索（语义+BM25）在图谱中搜索相关信息。
        如果Zep Cloud的search API不可用，则降级为本地关键词匹配。
        
        Args:
            graph_id: 图谱ID (Standalone Graph)
            query: 搜索查询
            limit: 返回结果数量
            scope: 搜索范围，"edges" 或 "nodes"
            
        Returns:
            SearchResult: 搜索结果
        """
        logger.info(f"图谱搜索: graph_id={graph_id}, query={query[:50]}...")
        
        # 尝试使用Zep Cloud Search API
        try:
            search_results = self._call_with_retry(
                func=lambda: self.client.graph.search(
                    graph_id=graph_id,
                    query=query,
                    limit=limit,
                    scope=scope,
                    reranker="cross_encoder"
                ),
                operation_name=f"图谱搜索(graph={graph_id})"
            )
            
            facts = []
            edges = []
            nodes = []
            
            # 解析边搜索结果
            if hasattr(search_results, 'edges') and search_results.edges:
                for edge in search_results.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        facts.append(edge.fact)
                    edges.append({
                        "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                        "name": getattr(edge, 'name', ''),
                        "fact": getattr(edge, 'fact', ''),
                        "source_node_uuid": getattr(edge, 'source_node_uuid', ''),
                        "target_node_uuid": getattr(edge, 'target_node_uuid', ''),
                    })
            
            # 解析节点搜索结果
            if hasattr(search_results, 'nodes') and search_results.nodes:
                for node in search_results.nodes:
                    nodes.append({
                        "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                        "name": getattr(node, 'name', ''),
                        "labels": getattr(node, 'labels', []),
                        "summary": getattr(node, 'summary', ''),
                    })
                    # 节点摘要也算作事实
                    if hasattr(node, 'summary') and node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(f"搜索完成: 找到 {len(facts)} 条相关事实")
            
            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts)
            )
            
        except Exception as e:
            logger.warning(f"Zep Search API失败，降级为本地搜索: {str(e)}")
            # 降级：使用本地关键词匹配搜索
            return self._local_search(graph_id, query, limit, scope)
    
    def _local_search(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        本地关键词匹配搜索（作为Zep Search API的降级方案）
        
        获取所有边/节点，然后在本地进行关键词匹配
        
        Args:
            graph_id: 图谱ID
            query: 搜索查询
            limit: 返回结果数量
            scope: 搜索范围
            
        Returns:
            SearchResult: 搜索结果
        """
        logger.info(f"使用本地搜索: query={query[:30]}...")
        
        facts = []
        edges_result = []
        nodes_result = []
        
        # 提取查询关键词（简单分词）
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        
        def match_score(text: str) -> int:
            """计算文本与查询的匹配分数"""
            if not text:
                return 0
            text_lower = text.lower()
            # 完全匹配查询
            if query_lower in text_lower:
                return 100
            # 关键词匹配
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score
        
        try:
            if scope in ["edges", "both"]:
                # 获取所有边并匹配
                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))
                
                # 按分数排序
                scored_edges.sort(key=lambda x: x[0], reverse=True)
                
                for score, edge in scored_edges[:limit]:
                    if edge.fact:
                        facts.append(edge.fact)
                    edges_result.append({
                        "uuid": edge.uuid,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                    })
            
            if scope in ["nodes", "both"]:
                # 获取所有节点并匹配
                all_nodes = self.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.name) + match_score(node.summary)
                    if score > 0:
                        scored_nodes.append((score, node))
                
                scored_nodes.sort(key=lambda x: x[0], reverse=True)
                
                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "labels": node.labels,
                        "summary": node.summary,
                    })
                    if node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(f"本地搜索完成: 找到 {len(facts)} 条相关事实")
            
        except Exception as e:
            logger.error(f"本地搜索失败: {str(e)}")
        
        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )
    
    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        获取图谱的所有节点（分页获取）

        Args:
            graph_id: 图谱ID

        Returns:
            节点列表
        """
        logger.info(f"获取图谱 {graph_id} 的所有节点...")

        nodes = fetch_all_nodes(self.client, graph_id)

        result = []
        for node in nodes:
            node_uuid = getattr(node, 'uuid_', None) or getattr(node, 'uuid', None) or ""
            result.append(NodeInfo(
                uuid=str(node_uuid) if node_uuid else "",
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            ))

        logger.info(f"获取到 {len(result)} 个节点")
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """
        获取图谱的所有边（分页获取，包含时间信息）

        Args:
            graph_id: 图谱ID
            include_temporal: 是否包含时间信息（默认True）

        Returns:
            边列表（包含created_at, valid_at, invalid_at, expired_at）
        """
        logger.info(f"获取图谱 {graph_id} 的所有边...")

        edges = fetch_all_edges(self.client, graph_id)

        result = []
        for edge in edges:
            edge_uuid = getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', None) or ""
            edge_info = EdgeInfo(
                uuid=str(edge_uuid) if edge_uuid else "",
                name=edge.name or "",
                fact=edge.fact or "",
                source_node_uuid=edge.source_node_uuid or "",
                target_node_uuid=edge.target_node_uuid or ""
            )

            # 添加时间信息
            if include_temporal:
                edge_info.created_at = getattr(edge, 'created_at', None)
                edge_info.valid_at = getattr(edge, 'valid_at', None)
                edge_info.invalid_at = getattr(edge, 'invalid_at', None)
                edge_info.expired_at = getattr(edge, 'expired_at', None)

            result.append(edge_info)

        logger.info(f"获取到 {len(result)} 条边")
        return result
    
    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """
        获取单个节点的详细信息
        
        Args:
            node_uuid: 节点UUID
            
        Returns:
            节点信息或None
        """
        logger.info(f"获取节点详情: {node_uuid[:8]}...")
        
        try:
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=node_uuid),
                operation_name=f"获取节点详情(uuid={node_uuid[:8]}...)"
            )
            
            if not node:
                return None
            
            return NodeInfo(
                uuid=getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            )
        except Exception as e:
            logger.error(f"获取节点详情失败: {str(e)}")
            return None
    
    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        获取节点相关的所有边
        
        通过获取图谱所有边，然后过滤出与指定节点相关的边
        
        Args:
            graph_id: 图谱ID
            node_uuid: 节点UUID
            
        Returns:
            边列表
        """
        logger.info(f"获取节点 {node_uuid[:8]}... 的相关边")
        
        try:
            # 获取图谱所有边，然后过滤
            all_edges = self.get_all_edges(graph_id)
            
            result = []
            for edge in all_edges:
                # 检查边是否与指定节点相关（作为源或目标）
                if edge.source_node_uuid == node_uuid or edge.target_node_uuid == node_uuid:
                    result.append(edge)
            
            logger.info(f"找到 {len(result)} 条与节点相关的边")
            return result
            
        except Exception as e:
            logger.warning(f"获取节点边失败: {str(e)}")
            return []
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str
    ) -> List[NodeInfo]:
        """
        按类型获取实体
        
        Args:
            graph_id: 图谱ID
            entity_type: 实体类型（如 Student, PublicFigure 等）
            
        Returns:
            符合类型的实体列表
        """
        logger.info(f"获取类型为 {entity_type} 的实体...")
        
        all_nodes = self.get_all_nodes(graph_id)
        
        filtered = []
        for node in all_nodes:
            # 检查labels是否包含指定类型
            if entity_type in node.labels:
                filtered.append(node)
        
        logger.info(f"找到 {len(filtered)} 个 {entity_type} 类型的实体")
        return filtered
    
    def get_entity_summary(
        self, 
        graph_id: str, 
        entity_name: str
    ) -> Dict[str, Any]:
        """
        获取指定实体的关系摘要
        
        搜索与该实体相关的所有信息，并生成摘要
        
        Args:
            graph_id: 图谱ID
            entity_name: 实体名称
            
        Returns:
            实体摘要信息
        """
        logger.info(f"获取实体 {entity_name} 的关系摘要...")
        
        # 先搜索该实体相关的信息
        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )
        
        # 尝试在所有节点中找到该实体
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break
        
        related_edges = []
        if entity_node:
            # 传入graph_id参数
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)
        
        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }
    
    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        获取图谱的统计信息
        
        Args:
            graph_id: 图谱ID
            
        Returns:
            统计信息
        """
        logger.info(f"获取图谱 {graph_id} 的统计信息...")
        
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)
        
        # 统计实体类型分布
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1
        
        # 统计关系类型分布
        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1
        
        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }
    
    def get_analysis_context(
        self,
        graph_id: str,
        analysis_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """
        获取学术分析相关的上下文信息

        从知识图谱中综合搜索与分析主题相关的所有实体、关系和事实。

        Args:
            graph_id: 图谱ID
            analysis_requirement: 分析主题描述（如"Transformer在NLP中的应用"）
            limit: 每类信息的数量限制

        Returns:
            包含 related_facts、graph_statistics、entities 的上下文字典
        """
        logger.info(f"获取学术分析上下文: {analysis_requirement[:50]}...")

        # 搜索与分析主题相关的信息
        search_result = self.search_graph(
            graph_id=graph_id,
            query=analysis_requirement,
            limit=limit
        )

        # 获取图谱统计
        stats = self.get_graph_statistics(graph_id)

        # 获取所有实体节点
        all_nodes = self.get_all_nodes(graph_id)

        # 筛选有实际类型的实体（非纯Entity节点）
        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })

        return {
            "analysis_requirement": analysis_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities)
        }

    # 向后兼容：旧名称别名
    def get_simulation_context(self, graph_id: str, simulation_requirement: str, limit: int = 30) -> Dict[str, Any]:
        return self.get_analysis_context(graph_id, simulation_requirement, limit)
    
    # ========== 核心检索工具（优化后） ==========
    
    def insight_forge(
        self,
        graph_id: str,
        query: str,
        analysis_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        """
        【InsightForge - 深度洞察检索】

        最强大的混合检索函数，自动分解问题并多维度检索学术知识图谱：
        1. 使用LLM将问题分解为多个子问题
        2. 对每个子问题进行语义搜索（Zep Cloud + 本地Neo4j）
        3. 提取相关实体（Paper/Method/Dataset/Author等）并获取详细信息
        4. 追踪关系链（PROPOSES/EVALUATES_ON/ACHIEVES等）
        5. 整合所有结果，生成深度洞察

        Args:
            graph_id: 图谱ID
            query: 用户问题
            analysis_requirement: 分析主题描述
            report_context: 报告上下文（可选，用于更精准的子问题生成）
            max_sub_queries: 最大子问题数量

        Returns:
            InsightForgeResult: 深度洞察检索结果
        """
        logger.info(f"InsightForge 深度洞察检索: {query[:50]}...")

        result = InsightForgeResult(
            query=query,
            analysis_requirement=analysis_requirement,
            sub_queries=[]
        )

        # Step 1: 使用LLM生成子问题
        sub_queries = self._generate_sub_queries(
            query=query,
            analysis_requirement=analysis_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(f"生成 {len(sub_queries)} 个子问题")
        
        # Step 2: 对每个子问题进行语义搜索
        all_facts = []
        all_edges = []
        seen_facts = set()
        
        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )
            
            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
            
            all_edges.extend(search_result.edges)
        
        # 对原始问题也进行搜索
        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)
        
        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)
        
        # Step 3: 从边中提取相关实体UUID，只获取这些实体的信息（不获取全部节点）
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)
        
        # 获取所有相关实体的详情（不限制数量，完整输出）
        entity_insights = []
        node_map = {}  # 用于后续关系链构建
        
        for uuid in list(entity_uuids):  # 处理所有实体，不截断
            if not uuid:
                continue
            try:
                # 单独获取每个相关节点的信息
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "实体")
                    
                    # 获取该实体相关的所有事实（不截断）
                    related_facts = [
                        f for f in all_facts 
                        if node.name.lower() in f.lower()
                    ]
                    
                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts  # 完整输出，不截断
                    })
            except Exception as e:
                logger.debug(f"获取节点 {uuid} 失败: {e}")
                continue
        
        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)
        
        # Step 4: 构建所有关系链（不限制数量）
        relationship_chains = []
        for edge_data in all_edges:  # 处理所有边，不截断
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')
                
                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]
                
                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)
        
        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)
        
        logger.info(f"InsightForge完成: {result.total_facts}条事实, {result.total_entities}个实体, {result.total_relationships}条关系")
        return result
    
    def _generate_sub_queries(
        self,
        query: str,
        analysis_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """
        使用LLM将学术问题分解为多个可独立检索的子问题

        针对学术论文知识图谱场景，从不同维度展开检索：
        方法、数据集、性能指标、对比基线、作者贡献等
        """
        system_prompt = """你是一个学术文献分析专家。你的任务是将一个关于学术论文知识图谱的复杂问题，
分解为多个可以在知识图谱中独立检索的子问题。

要求：
1. 子问题应覆盖不同维度：方法/模型、数据集、性能指标、对比基线、作者/团队、应用任务等
2. 每个子问题应足够具体，可以直接在图谱中检索到对应的节点或边
3. 子问题应与原始分析主题紧密相关
4. 返回JSON格式：{"sub_queries": ["子问题1", "子问题2", ...]}"""

        user_prompt = f"""分析主题背景：
{analysis_requirement}

{f"报告上下文：{report_context[:500]}" if report_context else ""}

请将以下学术问题分解为{max_queries}个子问题，覆盖不同检索维度：
{query}

返回JSON格式的子问题列表。"""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            sub_queries = response.get("sub_queries", [])
            return [str(sq) for sq in sub_queries[:max_queries]]

        except Exception as e:
            logger.warning(f"生成子问题失败: {str(e)}，使用默认子问题")
            return [
                query,
                f"{query} 涉及的核心方法和模型",
                f"{query} 使用的数据集和评测指标",
                f"{query} 相关的对比基线"
            ][:max_queries]
    
    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        """
        【PanoramaSearch - 广度搜索】
        
        获取全貌视图，包括所有相关内容和历史/过期信息：
        1. 获取所有相关节点
        2. 获取所有边（包括已过期/失效的）
        3. 分类整理当前有效和历史信息
        
        这个工具适用于需要了解事件全貌、追踪演变过程的场景。
        
        Args:
            graph_id: 图谱ID
            query: 搜索查询（用于相关性排序）
            include_expired: 是否包含过期内容（默认True）
            limit: 返回结果数量限制
            
        Returns:
            PanoramaResult: 广度搜索结果
        """
        logger.info(f"PanoramaSearch 广度搜索: {query[:50]}...")
        
        result = PanoramaResult(query=query)
        
        # 获取所有节点
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)
        
        # 获取所有边（包含时间信息）
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)
        
        # 分类事实
        active_facts = []
        historical_facts = []
        
        for edge in all_edges:
            if not edge.fact:
                continue
            
            # 为事实添加实体名称
            source_name = node_map.get(edge.source_node_uuid, NodeInfo('', '', [], '', {})).name or edge.source_node_uuid[:8]
            target_name = node_map.get(edge.target_node_uuid, NodeInfo('', '', [], '', {})).name or edge.target_node_uuid[:8]
            
            # 判断是否过期/失效
            is_historical = edge.is_expired or edge.is_invalid
            
            if is_historical:
                # 历史/过期事实，添加时间标记
                valid_at = edge.valid_at or "未知"
                invalid_at = edge.invalid_at or edge.expired_at or "未知"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                # 当前有效事实
                active_facts.append(edge.fact)
        
        # 基于查询进行相关性排序
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        
        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score
        
        # 排序并限制数量
        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)
        
        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)
        
        logger.info(f"PanoramaSearch完成: {result.active_count}条有效, {result.historical_count}条历史")
        return result
    
    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        """
        【QuickSearch - 简单搜索】
        
        快速、轻量级的检索工具：
        1. 直接调用Zep语义搜索
        2. 返回最相关的结果
        3. 适用于简单、直接的检索需求
        
        Args:
            graph_id: 图谱ID
            query: 搜索查询
            limit: 返回结果数量
            
        Returns:
            SearchResult: 搜索结果
        """
        logger.info(f"QuickSearch 简单搜索: {query[:50]}...")
        
        # 直接调用现有的search_graph方法
        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )
        
        logger.info(f"QuickSearch完成: {result.total_count}条结果")
        return result
    
    def deep_entity_query(
        self,
        graph_id: str,
        entity_type: str,
        query: str = "",
        max_entities: int = 5
    ) -> InterviewResult:
        """
        【DeepEntityQuery - 学术实体深度查询】

        按实体类型展开关联文献的深度查询：
        1. 按类型检索图谱中的节点（Paper/Method/Dataset/Author/Metric等）
        2. 对每个节点做1-hop邻居扩展，收集所有关联边事实
        3. 整合摘要，生成该类型实体的结构化全景分析

        【适用场景】
        - 需要了解某类实体（如所有方法Method）的全貌
        - 需要追踪特定论文引用了哪些数据集
        - 需要展示某个指标Metric被哪些工作评测

        Args:
            graph_id: 图谱ID
            entity_type: 实体类型（Paper/Method/Dataset/Author/Metric/Task/Innovation/Baseline）
            query: 额外过滤条件（可选，留空表示获取该类型全部实体）
            max_entities: 最多展开的实体数量

        Returns:
            InterviewResult: 复用采访结果结构承载实体深度分析
        """
        logger.info(f"DeepEntityQuery: entity_type={entity_type}, query={query[:40]}...")

        result = InterviewResult(
            interview_topic=f"{entity_type} 实体深度分析",
            interview_questions=[query] if query else []
        )

        # Step 1: 按类型获取节点
        typed_nodes = self.get_entities_by_type(graph_id, entity_type)
        result.total_agents = len(typed_nodes)

        if not typed_nodes:
            result.summary = f"图谱中未找到类型为 {entity_type} 的实体"
            return result

        # Step 2: 如有 query，做相关性过滤排序
        if query:
            query_lower = query.lower()
            def relevance(node: NodeInfo) -> int:
                score = 0
                if query_lower in node.name.lower():
                    score += 100
                if node.summary and query_lower in node.summary.lower():
                    score += 50
                return score
            typed_nodes = sorted(typed_nodes, key=relevance, reverse=True)

        # 取前 max_entities 个展开
        nodes_to_expand = typed_nodes[:max_entities]
        result.selected_agents = [{"name": n.name, "type": entity_type, "summary": n.summary} for n in nodes_to_expand]

        # Step 3: 对每个节点做 1-hop 邻居扩展，收集边事实
        for node in nodes_to_expand:
            edges = self.get_node_edges(graph_id, node.uuid)
            facts = [e.fact for e in edges if e.fact]
            relation_list = [
                f"{e.source_node_name or e.source_node_uuid[:8]} --[{e.name}]--> {e.target_node_name or e.target_node_uuid[:8]}"
                for e in edges
            ]
            response_text = f"**摘要:** {node.summary or '（无摘要）'}\n\n"
            if facts:
                response_text += "**关联事实:**\n" + "\n".join(f"- {f}" for f in facts[:10])
            if relation_list:
                response_text += "\n\n**关系链:**\n" + "\n".join(f"- {r}" for r in relation_list[:10])

            interview = AgentInterview(
                agent_name=node.name,
                agent_role=entity_type,
                agent_bio=node.summary or "",
                question=query or f"关于{node.name}的完整信息",
                response=response_text,
                key_quotes=facts[:3]
            )
            result.interviews.append(interview)

        result.interviewed_count = len(result.interviews)

        # Step 4: 生成整合摘要
        try:
            all_facts = []
            for iv in result.interviews:
                all_facts.extend(iv.key_quotes)
            summary_lines = [f"共检索到 {result.total_agents} 个 {entity_type} 类型实体，展开分析 {result.interviewed_count} 个。"]
            if all_facts:
                summary_lines.append("\n**核心事实摘要:**")
                summary_lines.extend([f"- {f}" for f in all_facts[:10]])
            result.summary = "\n".join(summary_lines)
        except Exception as e:
            logger.warning(f"生成摘要失败: {e}")
            result.summary = f"已分析 {result.interviewed_count} 个 {entity_type} 实体"

        logger.info(f"DeepEntityQuery完成: {result.interviewed_count} 个实体")
        return result

