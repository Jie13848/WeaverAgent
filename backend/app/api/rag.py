"""
RAG API 路由

提供论文向量索引和检索接口（两阶段 Graph+RAG 的第二阶段）

支持格式：PDF / Markdown / TXT

端点：
  POST /api/rag/index          — 构建/更新向量索引
  GET  /api/rag/status         — 查询索引状态
  POST /api/rag/search         — 向量检索（可限定文件范围）
"""

import traceback
from pathlib import Path
from flask import request, jsonify

from . import rag_bp
from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("weaveragent.api.rag")

UPLOAD_FOLDER = Path(Config.UPLOAD_FOLDER)


@rag_bp.route("/index", methods=["POST"])
def build_index():
    """
    构建/更新向量索引

    请求（JSON，可选）：
        {
            "force_rebuild": false,
            "files": ["xxx.pdf", "yyy.md"]  // 相对于 UPLOAD_FOLDER 的文件名，空则跳过
        }
    """
    try:
        from ..services.rag_service import build_index as _build

        data = request.get_json() or {}
        force_rebuild = data.get("force_rebuild", False)
        files_list = data.get("files", [])

        file_paths = None
        if files_list:
            file_paths = [UPLOAD_FOLDER / f for f in files_list if (UPLOAD_FOLDER / f).exists()]
            if not file_paths:
                return jsonify({"success": False, "error": "指定的文件均不存在"}), 400

        result = _build(files=file_paths, force_rebuild=force_rebuild)
        return jsonify({"success": True, "data": result})

    except ImportError as e:
        return jsonify({"success": False, "error": f"ChromaDB 未安装：{e}"}), 503
    except Exception as e:
        logger.error(f"构建索引失败: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@rag_bp.route("/status", methods=["GET"])
def index_status():
    """
    查询当前索引状态

    返回：
        {
            "success": true,
            "data": {
                "status": "ready",
                "total_chunks": 1230
            }
        }
    """
    try:
        from ..services.rag_service import index_status as _status

        return jsonify({"success": True, "data": _status()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@rag_bp.route("/search", methods=["POST"])
def search():
    """
    向量检索论文段落

    请求（JSON）：
        {
            "query": "Transformer attention mechanism",   // 必填
            "top_k": 6,                                  // 可选，默认 6
            "filter_filenames": ["25_transformer....md"] // 可选，限定检索范围
        }

    返回：
        {
            "success": true,
            "data": {
                "chunks": [
                    {
                        "text": "...",
                        "filename": "25_transformer....md",
                        "title": "Attention Is All You Need",
                        "chunk_index": 3,
                        "score": 0.92
                    }
                ]
            }
        }
    """
    try:
        from ..services.rag_service import retrieve_by_filenames, retrieve

        body = request.get_json() or {}
        query = body.get("query", "").strip()
        if not query:
            return jsonify({"success": False, "error": "请提供 query"}), 400

        try:
            top_k = int(body.get("top_k", 6))
        except (ValueError, TypeError):
            top_k = 6
        filter_filenames = body.get("filter_filenames") or []

        if filter_filenames:
            chunks = retrieve_by_filenames(query, filter_filenames, top_k=top_k)
        else:
            result = retrieve(query, top_k_chunks=top_k)
            chunks = result["rag_chunks"]

        return jsonify({"success": True, "data": {"chunks": chunks}})

    except Exception as e:
        logger.error(f"向量检索失败: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500
