"""
两阶段检索服务：GraphRAG（Zep）+ 普通RAG（ChromaDB）

流程：
  1. Graph检索：用 Zep 找相关 Innovation/PriorArt/Outcome 节点及关系
  2. 定位论文：从节点的 paper_title 属性反查对应文件
  3. 向量检索：在对应论文的文本里做细粒度段落检索
  4. 合并返回：图谱上下文 + 段落上下文，一起送给 LLM 生成最终回答

支持格式：PDF / Markdown / TXT（与图谱构建使用同一套 FileParser）
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..utils.logger import get_logger
from ..utils.file_parser import FileParser

logger = get_logger("weaveragent.rag")

# ChromaDB 向量库目录（持久化到 backend/rag_index/）
RAG_INDEX_DIR = Path(__file__).parent.parent.parent / "rag_index"

# 每个文本块的大小（字符数）
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# 支持的文件扩展名（与 FileParser.SUPPORTED_EXTENSIONS 保持一致）
SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.markdown', '.txt'}


# ─────────────────────────────────────────────
# 懒加载 ChromaDB（避免启动时报错）
# ─────────────────────────────────────────────

_chroma_client = None
_collection = None


def _get_collection():
    """懒加载 ChromaDB collection"""
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        raise ImportError(
            "ChromaDB 未安装，请运行：\n"
            "cd backend && uv add chromadb"
        )

    RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    _chroma_client = chromadb.PersistentClient(path=str(RAG_INDEX_DIR))

    # 使用阿里百炼兼容的 embedding（text-embedding-v3）
    # 如果没有配置则用 ChromaDB 默认的本地 embedding
    from ..config import Config
    api_key = Config.LLM_API_KEY
    base_url = Config.LLM_BASE_URL

    if api_key and "dashscope" in (base_url or ""):
        # 阿里百炼 text-embedding-v3，1536 维，免费额度很大
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            api_base=base_url,
            model_name="text-embedding-v3",
        )
        logger.info("RAG: 使用阿里百炼 text-embedding-v3")
    else:
        # 降级：用 ChromaDB 内置的本地 all-MiniLM（无需 API）
        ef = embedding_functions.DefaultEmbeddingFunction()
        logger.info("RAG: 使用本地 all-MiniLM embedding")

    _collection = _chroma_client.get_or_create_collection(
        name="papers",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


# ─────────────────────────────────────────────
# 索引构建
# ─────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """语义感知切块：复用 file_parser 中的语义分割逻辑"""
    from ..utils.file_parser import split_text_into_chunks
    return split_text_into_chunks(text, chunk_size, overlap)


def _paper_id(filename: str) -> str:
    """从文件名生成稳定 paper_id"""
    return hashlib.md5(filename.encode()).hexdigest()[:16]


def _extract_title(text: str, fallback: str) -> str:
    """从文本中提取标题（支持 MD 的 # 标题，否则取首行）"""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line:
            # 取第一段非空文本的前60字作为标题
            return line[:60]
    return fallback


def build_index(files: Optional[List[Path]] = None, force_rebuild: bool = False) -> Dict[str, Any]:
    """
    构建/更新向量索引

    支持 PDF / Markdown / TXT，通过 FileParser 统一提取文本。

    Args:
        files: 要索引的文件路径列表（Path 对象），None 时不做任何操作
        force_rebuild: 是否强制重建（忽略已有索引）

    Returns:
        {"indexed": N, "skipped": M, "total_chunks": K}
    """
    collection = _get_collection()

    if not files:
        logger.warning("build_index: 未传入文件列表，跳过")
        return {"indexed": 0, "skipped": 0, "total_chunks": 0}

    indexed = 0
    skipped = 0
    total_chunks = 0

    for file_path in files:
        file_path = Path(file_path)

        # 过滤不支持的格式
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.debug(f"跳过不支持的格式: {file_path.name}")
            continue

        if not file_path.exists():
            logger.warning(f"文件不存在，跳过: {file_path}")
            continue

        pid = _paper_id(file_path.name)

        # 已索引则跳过（除非强制重建）
        if not force_rebuild:
            existing = collection.get(where={"paper_id": pid}, limit=1)
            if existing["ids"]:
                skipped += 1
                continue

        # 用 FileParser 提取文本（PDF/MD/TXT 统一处理）
        try:
            text = FileParser.extract_text(str(file_path))
        except Exception as e:
            logger.warning(f"文本提取失败，跳过 {file_path.name}: {e}")
            continue

        if not text.strip():
            logger.warning(f"文件内容为空，跳过: {file_path.name}")
            continue

        chunks = _chunk_text(text)
        title = _extract_title(text, file_path.stem)

        # 批量 upsert
        ids = [f"{pid}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "paper_id": pid,
                "filename": file_path.name,
                "title": title,
                "chunk_index": i,
                "file_type": file_path.suffix.lower().lstrip("."),
            }
            for i in range(len(chunks))
        ]

        BATCH = 500
        for b in range(0, len(chunks), BATCH):
            collection.upsert(
                ids=ids[b : b + BATCH],
                documents=chunks[b : b + BATCH],
                metadatas=metadatas[b : b + BATCH],
            )

        total_chunks += len(chunks)
        indexed += 1
        logger.info(f"RAG 索引: {file_path.name} ({file_path.suffix}) → {len(chunks)} chunks")

    logger.info(f"RAG 索引完成: indexed={indexed}, skipped={skipped}, total_chunks={total_chunks}")
    return {"indexed": indexed, "skipped": skipped, "total_chunks": total_chunks}


def index_status() -> Dict[str, Any]:
    """返回当前索引状态"""
    try:
        collection = _get_collection()
        count = collection.count()
        return {"status": "ready", "total_chunks": count}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}


# ─────────────────────────────────────────────
# 两阶段检索
# ─────────────────────────────────────────────

def retrieve(
    query: str,
    graph_context: Optional[str] = None,
    top_k_graph: int = 5,
    top_k_chunks: int = 6,
    filter_filenames: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    两阶段检索主入口

    Args:
        query: 用户问题
        graph_context: 已从 Zep 获取的图谱上下文（节点/关系文本）
        top_k_graph: 从图谱上下文中定位的论文数（用于缩小向量检索范围）
        top_k_chunks: 返回的段落数
        filter_filenames: 强制只在这些文件里检索

    Returns:
        {
            "graph_context": "...",   # Zep 图谱结果（原样透传）
            "rag_chunks": [...],      # 向量检索段落列表
            "combined_context": "..." # 合并后的完整上下文
        }
    """
    collection = _get_collection()

    # 构建 ChromaDB where 过滤条件
    where = None
    if filter_filenames:
        # 把文件名转成 paper_id
        pids = [_paper_id(f) for f in filter_filenames]
        if len(pids) == 1:
            where = {"paper_id": pids[0]}
        else:
            where = {"paper_id": {"$in": pids}}

    # 向量检索
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k_chunks, collection.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    rag_chunks = []
    if results["ids"] and results["ids"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            rag_chunks.append(
                {
                    "text": doc,
                    "filename": meta.get("filename", ""),
                    "title": meta.get("title", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "score": round(1 - dist, 4),  # cosine similarity
                }
            )

    # 组装合并上下文
    combined_parts = []

    if graph_context:
        combined_parts.append(
            f"【知识图谱检索结果（技术关系）】\n{graph_context}"
        )

    if rag_chunks:
        chunks_text = "\n\n---\n\n".join(
            f"[来源: {c['title']} | 相关度: {c['score']}]\n{c['text']}"
            for c in rag_chunks
        )
        combined_parts.append(f"【论文原文段落检索结果（细节）】\n{chunks_text}")

    combined_context = "\n\n" + "=" * 40 + "\n\n".join(combined_parts)

    return {
        "graph_context": graph_context or "",
        "rag_chunks": rag_chunks,
        "combined_context": combined_context,
    }


def retrieve_by_filenames(
    query: str,
    filenames: List[str],
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """
    在指定论文文件中做向量检索（用于 Graph 定位到论文后的精细检索）

    Args:
        query: 检索问题
        filenames: 限定的论文文件名列表（如 ["25_transformer_....md"]）
        top_k: 返回段落数

    Returns:
        段落列表，每项包含 text/filename/title/score
    """
    result = retrieve(query, top_k_chunks=top_k, filter_filenames=filenames)
    return result["rag_chunks"]
