"""
Neo4j 图谱构建服务
用本地 LLM 做 NER + 关系抽取，直接写入 Neo4j，不依赖 Zep Cloud 处理
"""

import uuid
import time
import threading
import json
import re
from typing import Dict, Any, List, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from neo4j import GraphDatabase

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from ..utils.llm_client import LLMClient
from .text_processor import TextProcessor


# ─────────────────────────────────────────────────────────────────
#  并发配置
# ─────────────────────────────────────────────────────────────────
MAX_WORKERS = int(Config.NEO4J_EXTRACT_WORKERS) if hasattr(Config, "NEO4J_EXTRACT_WORKERS") else 5


class Neo4jGraphBuilderService:
    """
    Neo4j 图谱构建服务

    流程：
    1. 创建 Neo4j 图谱（数据库 or label namespace）
    2. 用 LLM 并发提取每个 chunk 的实体 + 关系
    3. 批量写入 Neo4j（MERGE 去重）
    4. 构建全文/向量索引（可选）
    """

    def __init__(self):
        uri  = Config.NEO4J_URI
        user = Config.NEO4J_USER
        pwd  = Config.NEO4J_PASSWORD
        if not uri:
            raise ValueError("NEO4J_URI 未配置")
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
        self.llm = LLMClient()
        self.task_manager = TaskManager()

    # ─── 公开入口 ───────────────────────────────────────────────────

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "WeaverAgent Graph",
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        batch_size: int = 5,          # 并发 LLM 调用数
    ) -> str:
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={"graph_name": graph_name, "chunk_size": chunk_size,
                      "text_length": len(text), "backend": "neo4j"},
        )
        thread = threading.Thread(
            target=self._build_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size),
            daemon=True,
        )
        thread.start()
        return task_id

    # ─── 工作线程 ────────────────────────────────────────────────────

    def _build_worker(self, task_id, text, ontology, graph_name,
                      chunk_size, chunk_overlap, batch_size):
        try:
            self._upd(task_id, 5, "开始构建 Neo4j 图谱...")

            # 1. 确定 graph_id（作为 Neo4j 节点的 graph 标签前缀）
            graph_id = f"graph_{uuid.uuid4().hex[:16]}"
            self._upd(task_id, 8, f"图谱 ID: {graph_id}")

            # 2. 初始化 Neo4j 约束 & 索引
            self._init_schema(graph_id)
            self._upd(task_id, 12, "Schema 初始化完成")

            # 3. 切分文本
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total = len(chunks)
            self._upd(task_id, 15, f"文本切分完成，共 {total} 个 chunk")

            # 4. 并发 LLM 抽取
            entity_types = [e["name"] for e in ontology.get("entity_types", [])]
            edge_types   = [e["name"] for e in ontology.get("edge_types", [])]

            all_nodes: List[Dict] = []
            all_edges: List[Dict] = []
            done = 0

            with ThreadPoolExecutor(max_workers=batch_size) as pool:
                futures = {
                    pool.submit(self._extract_chunk, i, chunk, entity_types, edge_types): i
                    for i, chunk in enumerate(chunks)
                }
                for fut in as_completed(futures):
                    nodes, edges = fut.result()
                    all_nodes.extend(nodes)
                    all_edges.extend(edges)
                    done += 1
                    pct = 15 + int(done / total * 55)   # 15→70%
                    self._upd(task_id, pct,
                              f"LLM 抽取中 {done}/{total}，"
                              f"已得 {len(all_nodes)} 节点 / {len(all_edges)} 关系")

            self._upd(task_id, 65, f"抽取完成：{len(all_nodes)} 节点 / {len(all_edges)} 关系，写入 Neo4j...")

            if not all_nodes:
                raise RuntimeError(
                    f"LLM 抽取结果为空（共 {total} 个 chunk 全部失败），"
                    "请检查 LLM API Key 余额或网络连接"
                )

            # 5. 批量写入 Neo4j
            self._write_nodes(graph_id, all_nodes)
            self._upd(task_id, 85, "节点写入完成，写入关系...")
            self._write_edges(graph_id, all_edges)
            self._upd(task_id, 92, "关系写入完成，统计图谱...")

            # 6. 统计
            stats = self._count(graph_id)
            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": {
                    "graph_id":   graph_id,
                    "node_count": stats["nodes"],
                    "edge_count": stats["edges"],
                    "entity_types": list({n.get("type","Entity") for n in all_nodes}),
                    "backend": "neo4j",
                },
                "chunks_processed": total,
            })

        except Exception as e:
            import traceback
            self.task_manager.fail_task(task_id, f"{e}\n{traceback.format_exc()}")

    # ─── LLM 抽取单 chunk ────────────────────────────────────────────

    def _extract_chunk(
        self,
        idx: int,
        chunk: str,
        entity_types: List[str],
        edge_types: List[str],
    ) -> Tuple[List[Dict], List[Dict]]:
        """调用 LLM 对单个 chunk 做 NER + 关系抽取，返回 (nodes, edges)"""

        entity_list = ", ".join(entity_types) if entity_types else "Entity"
        edge_list   = ", ".join(edge_types)   if edge_types   else "RELATED_TO"

        prompt = f"""You are a knowledge graph extractor for academic papers.

Given the following text chunk, extract:
1. Named entities (only types: {entity_list})
2. Relationships between entities (only types: {edge_list})

Return ONLY valid JSON in this exact format:
{{
  "nodes": [
    {{"id": "unique_slug", "name": "Entity Name", "type": "EntityType", "summary": "brief description"}}
  ],
  "edges": [
    {{"source": "source_id", "target": "target_id", "type": "RELATION_TYPE", "fact": "one-sentence description"}}
  ]
}}

Rules:
- Use snake_case for node id (e.g. "transformer_attention")
- Only extract entities clearly mentioned in the text
- Keep node id stable (same entity = same id across chunks)
- Return empty arrays if nothing found
- Output ONLY the JSON object, no other text

Text:
{chunk[:3000]}
"""
        try:
            raw = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
            )
            # 提取 JSON
            match = re.search(r'\{[\s\S]*\}', raw)
            if not match:
                return [], []
            data = json.loads(match.group())
            nodes = data.get("nodes", []) or []
            edges = data.get("edges", []) or []
            # 过滤格式不合规的条目
            nodes = [n for n in nodes if isinstance(n, dict) and n.get("name") and n.get("type")]
            edges = [e for e in edges if isinstance(e, dict) and e.get("source") and e.get("target")]
            return nodes, edges
        except Exception as e:
            import logging
            logging.getLogger('weaveragent.neo4j_builder').warning(
                f"chunk {idx} LLM 抽取失败: {e}"
            )
            return [], []

    # ─── Neo4j Schema ─────────────────────────────────────────────────

    def _init_schema(self, graph_id: str):
        """建立约束和索引"""
        with self.driver.session() as s:
            # 唯一约束：每个 graph 内，同类型同 id 的节点唯一
            s.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (n:__MFNode) REQUIRE (n.graph_id, n.node_id) IS UNIQUE
            """)
            # 全文索引（用于 RAG 检索）
            try:
                s.run("""
                    CREATE FULLTEXT INDEX mf_node_text IF NOT EXISTS
                    FOR (n:__MFNode) ON EACH [n.name, n.summary]
                """)
            except Exception:
                pass  # 已存在则忽略

    # ─── 批量写入 ─────────────────────────────────────────────────────

    def _write_nodes(self, graph_id: str, nodes: List[Dict]):
        if not nodes:
            return
        # 去重：同 id 同 type 取第一条
        seen: Dict[str, Dict] = {}
        for n in nodes:
            key = f"{n.get('id','')}__{n.get('type','Entity')}"
            if key not in seen:
                seen[key] = n
        unique = list(seen.values())

        BATCH = 200
        with self.driver.session() as s:
            for i in range(0, len(unique), BATCH):
                batch = unique[i:i + BATCH]
                s.run("""
                    UNWIND $rows AS row
                    MERGE (n:__MFNode {graph_id: $graph_id, node_id: row.id})
                    ON CREATE SET
                        n.name     = row.name,
                        n.type     = row.type,
                        n.summary  = coalesce(row.summary, ''),
                        n.graph_id = $graph_id
                    ON MATCH SET
                        n.summary  = CASE WHEN n.summary = '' THEN coalesce(row.summary, '') ELSE n.summary END
                """, rows=batch, graph_id=graph_id)

    def _write_edges(self, graph_id: str, edges: List[Dict]):
        if not edges:
            return
        BATCH = 200
        with self.driver.session() as s:
            for i in range(0, len(edges), BATCH):
                batch = edges[i:i + BATCH]
                s.run("""
                    UNWIND $rows AS row
                    MATCH (src:__MFNode {graph_id: $graph_id, node_id: row.source})
                    MATCH (tgt:__MFNode {graph_id: $graph_id, node_id: row.target})
                    MERGE (src)-[r:RELATED {graph_id: $graph_id, type: row.type}]->(tgt)
                    ON CREATE SET
                        r.fact     = coalesce(row.fact, ''),
                        r.rel_type = row.type,
                        r.graph_id = $graph_id
                """, rows=batch, graph_id=graph_id)

    # ─── 查询 ──────────────────────────────────────────────────────────

    def _count(self, graph_id: str) -> Dict[str, int]:
        with self.driver.session() as s:
            nc = s.run("MATCH (n:__MFNode {graph_id:$g}) RETURN count(n) AS c",
                       g=graph_id).single()["c"]
            ec = s.run("MATCH (:__MFNode {graph_id:$g})-[r:RELATED {graph_id:$g}]->(:__MFNode) RETURN count(r) AS c",
                       g=graph_id).single()["c"]
        return {"nodes": nc, "edges": ec}

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """返回与 ZepGraphBuilderService.get_graph_data 兼容的结构"""
        with self.driver.session() as s:
            raw_nodes = s.run("""
                MATCH (n:__MFNode {graph_id:$g})
                RETURN n.node_id AS id, n.name AS name, n.type AS type, n.summary AS summary
            """, g=graph_id).data()

            raw_edges = s.run("""
                MATCH (src:__MFNode {graph_id:$g})-[r:RELATED {graph_id:$g}]->(tgt:__MFNode {graph_id:$g})
                RETURN src.node_id AS source, tgt.node_id AS target,
                       r.rel_type AS type, r.fact AS fact,
                       src.name AS source_name, tgt.name AS target_name,
                       elementId(r) AS eid
            """, g=graph_id).data()

        nodes = [
            {
                "uuid":    n["id"],
                "name":    n["name"] or "",
                "labels":  [n["type"] or "Entity"],
                "type":    n["type"] or "Entity",
                "summary": n["summary"] or "",
                "attributes": {},
            }
            for n in raw_nodes
        ]
        edges = [
            {
                "uuid":             str(e["eid"]),
                "name":             e["type"] or "RELATED",
                "fact":             e["fact"] or "",
                "fact_type":        e["type"] or "RELATED",
                "source_node_uuid": e["source"],
                "target_node_uuid": e["target"],
                "source_node_name": e["source_name"] or "",
                "target_node_name": e["target_name"] or "",
                "attributes": {},
            }
            for e in raw_edges
        ]
        return {
            "graph_id":    graph_id,
            "nodes":       nodes,
            "edges":       edges,
            "node_count":  len(nodes),
            "edge_count":  len(edges),
            "backend":     "neo4j",
        }

    def append_text_worker(
        self,
        task_id: str,
        graph_id: str,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int = 5,
        ontology: Optional[Dict[str, Any]] = None,
    ):
        """
        向已有图谱追加新文本（复用 LLM 抽取 + Neo4j MERGE 写入，不重建 graph）。
        使用与原始构建相同的 ontology 约束，确保节点类型一致、MERGE 能正确合并。
        """
        try:
            self._upd(task_id, 5, "开始追加文本到图谱...")

            # 统计追加前的数量
            stats_before = self._count(graph_id)

            # 从 ontology 中提取类型约束（与初始构建保持一致）
            ont = ontology or {}
            entity_types = [e["name"] for e in ont.get("entity_types", [])]
            edge_types   = [e["name"] for e in ont.get("edge_types", [])]

            # 切分文本
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total = len(chunks)
            self._upd(task_id, 10, f"文本切分完成，共 {total} 个 chunk，开始 LLM 抽取...")

            # 并发抽取（传入与原始构建相同的类型约束）
            all_nodes: List[Dict] = []
            all_edges: List[Dict] = []
            done = 0

            with ThreadPoolExecutor(max_workers=batch_size) as pool:
                futures = {
                    pool.submit(self._extract_chunk, i, chunk, entity_types, edge_types): i
                    for i, chunk in enumerate(chunks)
                }
                for fut in as_completed(futures):
                    nodes, edges = fut.result()
                    all_nodes.extend(nodes)
                    all_edges.extend(edges)
                    done += 1
                    pct = 10 + int(done / total * 55)  # 10→65%
                    self._upd(task_id, pct,
                              f"LLM 抽取中 {done}/{total}，"
                              f"已得 {len(all_nodes)} 节点 / {len(all_edges)} 关系")

            self._upd(task_id, 65, f"抽取完成：{len(all_nodes)} 节点 / {len(all_edges)} 关系，写入 Neo4j...")

            if not all_nodes:
                raise RuntimeError(
                    f"LLM 抽取结果为空（共 {total} 个 chunk 全部失败），"
                    "请检查 LLM API Key 余额或网络连接"
                )

            # 写入（MERGE 自动去重，与已有节点合并）
            self._write_nodes(graph_id, all_nodes)
            self._upd(task_id, 80, "节点写入完成，写入关系...")
            self._write_edges(graph_id, all_edges)
            self._upd(task_id, 90, "关系写入完成，统计图谱...")

            # 统计追加后总量
            stats_after = self._count(graph_id)

            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": {
                    "graph_id":   graph_id,
                    "node_count": stats_after["nodes"],
                    "edge_count": stats_after["edges"],
                    "backend":    "neo4j",
                },
                "chunks_appended": total,
                "nodes_added": stats_after["nodes"] - stats_before["nodes"],
                "edges_added": stats_after["edges"] - stats_before["edges"],
            })

        except Exception as e:
            import traceback
            self.task_manager.fail_task(task_id, f"{e}\n{traceback.format_exc()}")

    def delete_graph(self, graph_id: str):
        """删除指定 graph_id 的所有节点和关系"""
        with self.driver.session() as s:
            s.run("""
                MATCH (n:__MFNode {graph_id:$g})
                DETACH DELETE n
            """, g=graph_id)

    def close(self):
        self.driver.close()

    # ─── 内部辅助 ──────────────────────────────────────────────────────

    def _upd(self, task_id: str, pct: int, msg: str):
        self.task_manager.update_task(task_id, progress=pct, message=msg)
