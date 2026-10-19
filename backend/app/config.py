"""
配置管理
统一从项目根目录的 .env 文件加载配置
"""

import os
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
# 路径: WeaverAgent/.env (相对于 backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # 如果根目录没有 .env，尝试加载环境变量（用于生产环境）
    load_dotenv(override=True)


class Config:
    """Flask配置类"""
    
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY', 'weaveragent-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # JSON配置 - 禁用ASCII转义，让中文直接显示（而不是 \uXXXX 格式）
    JSON_AS_ASCII = False
    
    # LLM配置（统一使用OpenAI格式）
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    
    # ==============================
    # GraphRAG 配置（替代 Zep）
    # ==============================
    # 本地图谱数据存储目录
    GRAPHRAG_DATA_DIR = os.environ.get(
        'GRAPHRAG_DATA_DIR',
        os.path.join(os.path.dirname(__file__), '../../../graphrag_data')
    )

    # 嵌入模型名称（用于 GraphRAG 向量索引）
    # 如果 LLM 提供商支持嵌入，可以直接使用相同的 API
    EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', 'text-embedding-3-small')

    # Embedding API 独立配置（LLM 和 Embedding 可能来自不同供应商）
    # 如 LLM 用 Moonshot Kimi，Embedding 用 DashScope text-embedding-v3
    # 若未显式设置，则回退到与 LLM 相同的凭证
    EMBEDDING_API_KEY = os.environ.get('EMBEDDING_API_KEY') or os.environ.get('LLM_API_KEY')
    EMBEDDING_BASE_URL = os.environ.get('EMBEDDING_BASE_URL') or os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')

    # ZEP_API_KEY 保留（向后兼容，GraphRAG 不使用）
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY', '')

    # Neo4j 配置（本地图数据库，检索阶段使用）
    NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', '')

    # 文件上传配置
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # 文本处理配置
    DEFAULT_CHUNK_SIZE = 500  # 默认切块大小
    DEFAULT_CHUNK_OVERLAP = 50  # 默认重叠大小
    
    # Report Agent配置
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    
    @classmethod
    def validate(cls):
        """验证必要配置（GraphRAG 版本：不再需要 ZEP_API_KEY）"""
        errors = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY 未配置")
        # ZEP_API_KEY 已不再必须（GraphRAG 本地运行）
        return errors

