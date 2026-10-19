"""
GraphRAG 图谱构建服务
使用 Microsoft GraphRAG 替代 Zep Cloud API

完全兼容 graph_builder.py 的接口，无需修改上层调用代码。

变化对比:
  Zep: 云端API → 调用Zep Cloud → 云端存储
  GraphRAG: 本地索引 → Parquet文件 → 本地lancedb向量存储
"""

import os
import uuid
import json
import asyncio
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

import pandas as pd

from graphrag.api.index import build_index
from graphrag.config.load_config import load_config
from graphrag.config.enums import IndexingMethod

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from .text_processor import TextProcessor
from ..utils.logger import get_logger

logger = get_logger('weaveragent.graphrag_builder')

# GraphRAG数据存储根目录（可通过环境变量覆盖）
GRAPHRAG_DATA_DIR = os.environ.get(
    'GRAPHRAG_DATA_DIR',
    os.path.join(os.path.dirname(__file__), '../../../../graphrag_data')
)


@dataclass
class GraphInfo:
    """图谱信息（与 graph_builder.py 完全兼容）"""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


def get_graph_dir(graph_id: str) -> Path:
    """获取图谱目录路径"""
    return Path(GRAPHRAG_DATA_DIR) / graph_id


def _split_text_by_doc_markers(combined_text: str) -> List[Dict[str, str]]:
    """
    把上传阶段拼接的全文按 '=== filename ===' 分隔符还原成多篇文档。

    返回 [{"name": <文件名>, "text": <正文>}, ...]
    若没有分隔符则返回单篇 [{"name": "document", "text": combined_text}]。
    """
    import re as _re

    text = (combined_text or "").strip()
    if not text:
        return []

    pattern = _re.compile(r'^[ \t]*===\s*(.+?)\s*===[ \t]*$', _re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        return [{"name": "document", "text": text}]

    docs: List[Dict[str, str]] = []
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            docs.append({"name": name, "text": body})
    return docs or [{"name": "document", "text": text}]


CUSTOM_EXTRACT_PROMPT = str(
    Path(__file__).resolve().parent.parent / "prompts" / "extract_graph.txt"
)


def _create_settings_yaml(graph_dir: Path, entity_types: Optional[List[str]] = None) -> str:
    """为指定图谱目录生成 settings.yaml 内容"""
    input_dir   = str(graph_dir / "input")
    output_dir  = str(graph_dir / "output")
    cache_dir   = str(graph_dir / "cache")
    report_dir  = str(graph_dir / "reporting")
    vector_uri  = str(graph_dir / "lancedb")

    default_types = (
        "organization, person, geo, event, concept, "
        "technology, product, work_of_art, location"
    )
    entity_types_str = ", ".join(entity_types) if entity_types else default_types

    # 嵌入模型名称：优先使用 Config.EMBEDDING_MODEL_NAME，否则推断
    embedding_model = getattr(Config, 'EMBEDDING_MODEL_NAME', None)
    if not embedding_model:
        llm_model = (Config.LLM_MODEL_NAME or "").lower()
        if "qwen" in llm_model:
            embedding_model = "text-embedding-v3"
        elif "glm" in llm_model:
            embedding_model = "embedding-3"
        else:
            embedding_model = "text-embedding-3-small"

    # LLM 和 Embedding 可能来自不同供应商（例如 Kimi + DashScope text-embedding-v3）
    llm_api_key = Config.LLM_API_KEY or ''
    llm_base_url = Config.LLM_BASE_URL or 'https://api.openai.com/v1'
    embedding_api_key = getattr(Config, 'EMBEDDING_API_KEY', None) or llm_api_key
    embedding_base_url = getattr(Config, 'EMBEDDING_BASE_URL', None) or llm_base_url

    # 写入 per-graph .env 供 GraphRAG 读取环境变量
    env_file = graph_dir / ".env"
    with open(env_file, "w", encoding="utf-8") as ef:
        ef.write(f"GRAPHRAG_LLM_API_KEY={llm_api_key}\n")
        ef.write(f"GRAPHRAG_LLM_BASE_URL={llm_base_url}\n")
        ef.write(f"GRAPHRAG_EMBEDDING_API_KEY={embedding_api_key}\n")
        ef.write(f"GRAPHRAG_EMBEDDING_BASE_URL={embedding_base_url}\n")

    import re as _re
    llm_model_name = Config.LLM_MODEL_NAME or 'gpt-4o-mini'
    is_qwen3 = bool(_re.match(r'qwen3', llm_model_name, _re.IGNORECASE))
    is_glm = bool(_re.match(r'glm', llm_model_name, _re.IGNORECASE))
    is_kimi = bool(_re.match(r'(kimi|moonshot)', llm_model_name, _re.IGNORECASE))
    is_gpt5 = bool(_re.match(r'gpt-5', llm_model_name, _re.IGNORECASE))

    call_args_block = ""
    if is_qwen3:
        # Qwen3 系列需显式关闭 thinking，否则 response_format=json_object 会报错
        call_args_block = """
    call_args:
      extra_body:
        enable_thinking: false"""
    elif is_glm:
        # 智谱 GLM-5/5.1/5-turbo 等是思考模型，默认会产生数百 reasoning tokens
        # 关掉思考以免每次调用多消耗 10-20 倍 tokens
        call_args_block = """
    call_args:
      extra_body:
        thinking:
          type: disabled"""
    elif is_gpt5:
        # GPT-5 系列是 reasoning 模型，默认 medium 推理每次会多消耗几百 reasoning tokens
        # 用 minimal 几乎关闭推理，速度和单价接近 gpt-4o-mini
        call_args_block = """
    call_args:
      reasoning_effort: minimal"""
    # Kimi 非思考模型（kimi-k2-turbo-preview / kimi-k2-0905-preview）不需特殊参数
    # 注意：kimi-k2.5 是强制思考模型，不建议用于 GraphRAG

    # 并发数：各家 API 速率限制不一样，需要分别适配避免 RateLimitError
    if is_kimi:
        concurrent_requests = 3   # Moonshot 限流严格（~3 RPS）
    elif is_glm:
        concurrent_requests = 5   # 智谱限流较严（~10 RPS）
    elif is_qwen3:
        concurrent_requests = 15  # DashScope 限流宽松
    elif is_gpt5:
        concurrent_requests = 10  # GPT-5 系列默认 Tier-1 限流比 4o-mini 略严
    else:
        concurrent_requests = 25  # OpenAI / 默认

    return f"""\
### WeaverAgent GraphRAG 自动生成的配置文件
### 请勿手动修改 - 由 graphrag_builder.py 管理

concurrent_requests: {concurrent_requests}
async_mode: threaded

completion_models:
  default_chat_model:
    model_provider: openai
    model: {llm_model_name}
    api_key: ${{GRAPHRAG_LLM_API_KEY}}
    api_base: ${{GRAPHRAG_LLM_BASE_URL}}
    auth_method: api_key
    retry:
      type: exponential_backoff{call_args_block}

embedding_models:
  default_embedding_model:
    model_provider: openai
    model: {embedding_model}
    api_key: ${{GRAPHRAG_EMBEDDING_API_KEY}}
    api_base: ${{GRAPHRAG_EMBEDDING_BASE_URL}}
    auth_method: api_key
    retry:
      type: exponential_backoff
    # encoding_format=float 是必需的：DashScope 严格校验，
    # 拒绝 null 默认值（"only support [float, base64]"）
    call_args:
      encoding_format: float

input:
  type: text

chunking:
  type: tokens
  size: 1200
  overlap: 100

input_storage:
  type: file
  base_dir: "{input_dir}"

output_storage:
  type: file
  base_dir: "{output_dir}"

reporting:
  type: file
  base_dir: "{report_dir}"

cache:
  type: json
  storage:
    type: file
    base_dir: "{cache_dir}"

vector_store:
  type: lancedb
  db_uri: "{vector_uri}"

embed_text:
  embedding_model_id: default_embedding_model

extract_graph:
  completion_model_id: default_chat_model
  prompt: "{CUSTOM_EXTRACT_PROMPT}"
  entity_types: [{entity_types_str}]
  max_gleanings: 0

summarize_descriptions:
  completion_model_id: default_chat_model
  max_length: 500

cluster_graph:
  max_cluster_size: 10

community_reports:
  completion_model_id: default_chat_model
  max_length: 2000
  max_input_length: 8000

extract_claims:
  enabled: false

local_search:
  completion_model_id: default_chat_model
  embedding_model_id: default_embedding_model

global_search:
  completion_model_id: default_chat_model

drift_search:
  completion_model_id: default_chat_model
  embedding_model_id: default_embedding_model

basic_search:
  completion_model_id: default_chat_model
  embedding_model_id: default_embedding_model
"""


class GraphBuilderService:
    """
    图谱构建服务（GraphRAG 版本，完整替代 Zep 实现）

    公共接口与原 Zep 版本保持完全兼容：
      - build_graph_async(text, ontology, graph_name, ...)  → task_id
      - create_graph(name)                                  → graph_id
      - set_ontology(graph_id, ontology)
      - add_text_batches(graph_id, chunks, ...)             → episode_uuids
      - get_graph_data(graph_id)                            → dict
      - delete_graph(graph_id)
    """

    def __init__(self, api_key: Optional[str] = None):
        # api_key 参数保留以兼容原接口（GraphRAG 无需它）
        self.task_manager = TaskManager()
        os.makedirs(GRAPHRAG_DATA_DIR, exist_ok=True)

    # =========================================================
    # 对外接口（与 Zep 版本完全兼容）
    # =========================================================

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "WeaverAgent Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 10
    ) -> str:
        """
        异步构建图谱（兼容原接口）

        Returns:
            task_id - 可用于查询进度
        """
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            }
        )

        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size),
            daemon=True
        )
        thread.start()
        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int
    ):
        """图谱构建工作线程"""
        try:
            self.task_manager.update_task(
                task_id, status=TaskStatus.PROCESSING, progress=5, message="开始构建图谱..."
            )

            # 1. 创建图谱目录
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(task_id, progress=10, message=f"图谱目录已创建: {graph_id}")

            # 2. 设置本体（更新 settings.yaml 实体类型）
            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(task_id, progress=15, message="本体已设置")

            # 3. 整篇写入（不预切碎，由 GraphRAG 内部按 token 切块）
            documents = _split_text_by_doc_markers(text)
            self.task_manager.update_task(
                task_id, progress=20,
                message=f"准备写入 {len(documents)} 篇文档（共 {len(text):,} 字符）"
            )

            # 4. 写入完整文档到 input/ 目录
            episode_uuids = self.add_documents(
                graph_id, documents,
                progress_callback=lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.4),
                    message=msg
                )
            )
            total_chunks = len(documents)  # 兼容下游统计

            # 5. 运行 GraphRAG 索引管道（等价于 Zep 处理等待）
            self.task_manager.update_task(task_id, progress=60, message="运行 GraphRAG 索引管道...")
            self._run_indexing_pipeline(
                graph_id,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=60 + int(prog * 0.3),
                    message=msg
                )
            )

            # 6. 读取图谱统计
            self.task_manager.update_task(task_id, progress=90, message="获取图谱信息...")
            graph_info = self._get_graph_info(graph_id)

            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": graph_info.to_dict(),
                "chunks_processed": total_chunks,
            })

        except Exception as e:
            import traceback
            self.task_manager.fail_task(task_id, f"{str(e)}\n{traceback.format_exc()}")

    def create_graph(self, name: str) -> str:
        """
        创建图谱（替代 Zep.graph.create）

        在本地创建 GraphRAG 工作目录和配置文件。
        """
        graph_id = f"weaveragent_{uuid.uuid4().hex[:16]}"
        graph_dir = get_graph_dir(graph_id)

        # 创建必要目录
        for sub in ("input", "output", "cache", "reporting"):
            (graph_dir / sub).mkdir(parents=True, exist_ok=True)

        # 图谱元数据
        meta = {
            "graph_id": graph_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
        }
        with open(graph_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 生成默认 settings.yaml
        with open(graph_dir / "settings.yaml", "w", encoding="utf-8") as f:
            f.write(_create_settings_yaml(graph_dir))

        logger.info(f"图谱目录已创建: {graph_id} → {graph_dir}")
        return graph_id

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """
        设置图谱本体（替代 Zep.graph.set_ontology）

        将本体 JSON 保存到磁盘，并更新 settings.yaml 中的实体类型列表。
        """
        graph_dir = get_graph_dir(graph_id)

        # 持久化本体
        with open(graph_dir / "ontology.json", "w", encoding="utf-8") as f:
            json.dump(ontology, f, ensure_ascii=False, indent=2)

        # 提取实体类型名称，更新 settings.yaml
        entity_types = [et["name"] for et in ontology.get("entity_types", [])]
        with open(graph_dir / "settings.yaml", "w", encoding="utf-8") as f:
            f.write(_create_settings_yaml(graph_dir, entity_types or None))

        logger.info(f"本体已设置: {len(entity_types)} 种实体类型 → {entity_types}")

    def add_documents(
        self,
        graph_id: str,
        documents: List[Dict[str, str]],
        progress_callback: Optional[Callable] = None
    ) -> List[str]:
        """
        将原始文档整篇写入 input/ 目录，由 GraphRAG 自行按 token 切块。

        参数:
            documents: [{"name": "paper1.pdf", "text": "完整论文正文..."}, ...]
                      每个 dict 写成一个 .txt 文件，**不预切碎**
            progress_callback: (msg, ratio) -> None

        Returns:
            doc_ids - 写入的文件 ID 列表

        为何不预切碎：
            - GraphRAG 内置 token 级切块（chunk_size=1200, overlap=100）
            - 字符级预切碎会把句子/公式腰斩，LLM 失去上下文，关系抽取率骤降
            - 整篇写入后，每个 chunk 内容完整，实体共现率高，关系数 ≥ 实体数
            - 同时大幅减少 system prompt 的重复发送（成本下降数倍）
        """
        graph_dir = get_graph_dir(graph_id)
        input_dir = graph_dir / "input"
        input_dir.mkdir(exist_ok=True)

        doc_ids = []
        total = len(documents)

        for i, doc in enumerate(documents):
            text = doc.get("text", "") or ""
            if not text.strip():
                continue

            raw_name = doc.get("name") or f"doc_{i:04d}"
            safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw_name)
            safe_stem = safe_stem[:80] or f"doc_{i:04d}"
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{safe_stem}"

            doc_file = input_dir / f"{doc_id}.txt"
            with open(doc_file, "w", encoding="utf-8") as f:
                f.write(text)

            doc_ids.append(doc_id)

            if progress_callback:
                ratio = (i + 1) / total
                progress_callback(
                    f"已写入文档 {i+1}/{total} ({len(text):,} 字符)",
                    ratio
                )

        total_chars = sum(len(d.get("text", "") or "") for d in documents)
        logger.info(
            f"全部 {len(doc_ids)} 篇文档已写入 {graph_id}/input/，"
            f"共 {total_chars:,} 字符（GraphRAG 将按 1200 token / chunk 切块）"
        )
        return doc_ids

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 10,
        progress_callback: Optional[Callable] = None
    ) -> List[str]:
        """
        [DEPRECATED] 将预切碎的小文本块写入 input/ 目录。

        ⚠️ 此方法会导致字符级预切碎，把句子/公式腰斩，
        建议改用 add_documents() 写入整篇文档，由 GraphRAG 按 token 切块。

        保留此方法仅为兼容旧调用路径。
        """
        graph_dir = get_graph_dir(graph_id)
        input_dir = graph_dir / "input"
        input_dir.mkdir(exist_ok=True)

        chunk_ids = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
            chunk_file = input_dir / f"{chunk_id}.txt"

            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(chunk)

            chunk_ids.append(chunk_id)

            if progress_callback and ((i + 1) % batch_size == 0 or i == total - 1):
                progress = (i + 1) / total
                progress_callback(f"已写入 {i+1}/{total} 个文本块", progress)

        logger.warning(
            f"[DEPRECATED] add_text_batches 写入了 {total} 个预切碎块，"
            f"建议改用 add_documents() 以提升图谱质量"
        )
        return chunk_ids

    def _wait_for_episodes(
        self,
        episode_uuids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600
    ):
        """
        兼容原接口的等待方法（不含 graph_id 参数的旧签名）

        注意：此方法在新的 _build_graph_worker 中不再直接调用，
        保留是为了兼容可能的外部调用。
        """
        logger.warning("_wait_for_episodes 被调用但 graph_id 未传入，跳过索引运行。"
                       "请确保通过 build_graph_async 调用构建流程。")

    @staticmethod
    def _fix_lancedb_dimensions(graph_dir: Path, embedding_model: str):
        """Check LanceDB vector dimensions and rebuild if mismatched."""
        KNOWN_DIMS = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-ada-002': 1536,
            'text-embedding-v3': 1024,
            'embedding-2': 1024,
            'embedding-3': 2048,
        }
        expected_dim = KNOWN_DIMS.get(embedding_model)
        if not expected_dim:
            return

        lancedb_dir = graph_dir / "lancedb"
        if not lancedb_dir.exists():
            return

        try:
            import lancedb as _lancedb
            db = _lancedb.connect(str(lancedb_dir))
            for table_name in db.table_names():
                tbl = db.open_table(table_name)
                schema = tbl.schema
                for field in schema:
                    if hasattr(field.type, 'list_size') and field.type.list_size != expected_dim:
                        logger.warning(
                            f"LanceDB 维度不匹配: {table_name} 现有={field.type.list_size}, "
                            f"期望={expected_dim}（{embedding_model}），删除旧表"
                        )
                        shutil.rmtree(str(lancedb_dir))
                        return
        except Exception as e:
            logger.warning(f"LanceDB 维度检查失败，删除并重建: {e}")
            if lancedb_dir.exists():
                shutil.rmtree(str(lancedb_dir))

    def _run_indexing_pipeline(
        self,
        graph_id: str,
        progress_callback: Optional[Callable] = None,
        is_update_run: bool = False
    ):
        """
        运行 GraphRAG 索引管道（核心替代逻辑）

        对应 Zep 的 _wait_for_episodes：Zep 是等待云端处理完成，
        这里是在本地运行完整的 GraphRAG 索引管道。
        """
        import re as _re
        import time as _time

        graph_dir = get_graph_dir(graph_id)

        embedding_model = getattr(Config, 'EMBEDDING_MODEL_NAME', None) or 'text-embedding-3-small'
        self._fix_lancedb_dimensions(graph_dir, embedding_model)

        if progress_callback:
            progress_callback("加载 GraphRAG 配置...", 0.05)

        config = load_config(str(graph_dir))

        if progress_callback:
            progress_callback("提取实体和关系（LLM 推断中）...", 0.1)

        log_file = graph_dir / "reporting" / "indexing-engine.log"
        stop_monitor = threading.Event()

        def _monitor_log():
            """后台线程：监控索引日志，解析进度并回调"""
            if not progress_callback:
                return
            last_pos = 0
            progress_re = _re.compile(
                r'(?:(?:level \d+ )?)(chunker|extract graph|summarize communities|generate embeddings) progress:\s*(\d+)/(\d+)'
            )
            phase_names = {
                'chunker': '文本分块',
                'extract graph': '提取实体和关系',
                'summarize communities': '生成社区摘要',
                'generate embeddings': '构建向量索引',
            }
            phase_weights = {
                'chunker': (0.10, 0.15),
                'extract graph': (0.15, 0.70),
                'summarize communities': (0.70, 0.90),
                'generate embeddings': (0.90, 0.98),
            }
            workflow_re = _re.compile(r'Workflow started: (\w+)')
            last_workflow = ''

            while not stop_monitor.is_set():
                _time.sleep(2)
                try:
                    if not log_file.exists():
                        continue
                    with open(log_file, 'r', encoding='utf-8') as f:
                        f.seek(last_pos)
                        new_lines = f.readlines()
                        last_pos = f.tell()

                    best_match = None
                    for line in new_lines:
                        wm = workflow_re.search(line)
                        if wm:
                            last_workflow = wm.group(1)
                        m = progress_re.search(line)
                        if m:
                            best_match = m

                    if best_match:
                        phase = best_match.group(1)
                        current, total = int(best_match.group(2)), int(best_match.group(3))
                        phase_cn = phase_names.get(phase, phase)
                        ratio = current / total if total > 0 else 0
                        lo, hi = phase_weights.get(phase, (0.1, 0.95))
                        overall = lo + ratio * (hi - lo)
                        progress_callback(
                            f"{phase_cn}（{current}/{total}）",
                            min(overall, 0.98)
                        )
                    elif last_workflow and new_lines:
                        wf_cn = {
                            'create_base_text_units': '创建文本单元',
                            'create_final_documents': '处理文档',
                            'extract_graph': '提取图谱',
                            'finalize_graph': '合并图谱',
                            'extract_covariates': '提取协变量',
                            'create_communities': '社区检测',
                            'create_community_reports': '生成社区报告',
                            'generate_text_embeddings': '生成向量嵌入',
                        }.get(last_workflow, last_workflow)
                        progress_callback(f"{wf_cn}...", 0.5)
                except Exception:
                    pass

        monitor_thread = threading.Thread(target=_monitor_log, daemon=True)
        monitor_thread.start()

        async def _run():
            return await build_index(
                config=config,
                method=IndexingMethod.Standard,
                is_update_run=is_update_run,
                verbose=False,
            )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(_run())
        finally:
            loop.close()
            asyncio.set_event_loop(None)
            stop_monitor.set()
            monitor_thread.join(timeout=3)

        success_count = sum(1 for r in results if r.error is None)
        error_count = sum(1 for r in results if r.error is not None)

        if error_count > 0:
            logger.warning(f"GraphRAG 索引部分失败: {error_count} 个 workflow 出错")
        else:
            logger.info(f"GraphRAG 索引完成: {success_count} 个 workflow 全部成功")

        if progress_callback:
            progress_callback(
                f"GraphRAG 索引完成（{success_count} 成功，{error_count} 失败）", 1.0
            )

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        """从 Parquet 文件读取图谱统计信息"""
        output_dir = get_graph_dir(graph_id) / "output"

        node_count = 0
        edge_count = 0
        entity_types: List[str] = []

        entities_file = output_dir / "entities.parquet"
        if entities_file.exists():
            df = pd.read_parquet(entities_file)
            node_count = len(df)
            if "type" in df.columns:
                entity_types = df["type"].dropna().unique().tolist()

        rels_file = output_dir / "relationships.parquet"
        if rels_file.exists():
            edge_count = len(pd.read_parquet(rels_file))

        return GraphInfo(
            graph_id=graph_id,
            node_count=node_count,
            edge_count=edge_count,
            entity_types=entity_types,
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        获取完整图谱数据（兼容原接口）

        从 Parquet 文件读取并转换为与 Zep 版本相同的格式。
        """
        output_dir = get_graph_dir(graph_id) / "output"
        nodes_data: List[Dict] = []
        edges_data: List[Dict] = []

        # ---- 读取实体 ----
        entities_file = output_dir / "entities.parquet"
        if entities_file.exists():
            df = pd.read_parquet(entities_file)
            for _, row in df.iterrows():
                entity_type = str(row.get("type", "Entity") or "Entity")
                node_id = str(row.get("id", ""))
                nodes_data.append({
                    "id": node_id,
                    "uuid": node_id,
                    "name": str(row.get("title", "")),
                    "type": entity_type,
                    "label": entity_type,
                    "labels": [entity_type] if entity_type else ["Entity"],
                    "summary": str(row.get("description", "") or ""),
                    "attributes": row.get("attributes") or {},
                    "created_at": None,
                })

        # ---- 读取关系 ----
        rels_file = output_dir / "relationships.parquet"
        if rels_file.exists():
            df = pd.read_parquet(rels_file)
            name_to_uuid = {n["name"]: n["uuid"] for n in nodes_data}

            for _, row in df.iterrows():
                source_name = str(row.get("source", ""))
                target_name = str(row.get("target", ""))
                desc   = str(row.get("description", "") or "")
                rel_type = str(row.get("type", "") or "related").upper()
                source_uuid = name_to_uuid.get(source_name, source_name)
                target_uuid = name_to_uuid.get(target_name, target_name)
                edges_data.append({
                    "uuid": str(row.get("id", "")),
                    "source": source_uuid,
                    "target": target_uuid,
                    "type": rel_type,
                    "relation": rel_type,
                    "name": desc,
                    "fact": desc,
                    "source_node_uuid": source_uuid,
                    "target_node_uuid": target_uuid,
                    "source_node_name": source_name,
                    "target_node_name": target_name,
                    "attributes": row.get("attributes") or {},
                    "created_at": None,
                })

        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }

    def delete_graph(self, graph_id: str):
        """删除图谱（替代 Zep.graph.delete）"""
        graph_dir = get_graph_dir(graph_id)
        if graph_dir.exists():
            shutil.rmtree(str(graph_dir))
            logger.info(f"图谱已删除: {graph_id}")
        else:
            logger.warning(f"图谱目录不存在，跳过删除: {graph_id}")

    def run_incremental_update(self, graph_id: str) -> bool:
        """
        对已有图谱运行增量索引（用于追加文档后更新）

        Returns:
            True 表示成功，False 表示失败
        """
        graph_dir = get_graph_dir(graph_id)
        if not graph_dir.exists():
            logger.error(f"增量更新失败：图谱目录不存在: {graph_id}")
            return False

        try:
            config = load_config(str(graph_dir))

            async def _run():
                return await build_index(
                    config=config,
                    method=IndexingMethod.Standard,
                    is_update_run=True,
                    verbose=False,
                )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(_run())
            finally:
                loop.close()
                asyncio.set_event_loop(None)

            error_count = sum(1 for r in results if r.error is not None)
            logger.info(f"增量索引完成: graph_id={graph_id}, 错误数={error_count}")
            return error_count == 0

        except Exception as e:
            logger.error(f"增量索引失败: {str(e)}")
            return False
