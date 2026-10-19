"""
业务服务模块
使用 Microsoft GraphRAG 替代 Zep Cloud API
"""

from .ontology_generator import OntologyGenerator
from .graphrag_builder import GraphBuilderService   # ← graphrag 版本
from .text_processor import TextProcessor
from .graphrag_memory_updater import (              # ← graphrag 版本
    GraphRAGMemoryUpdater,
    GraphRAGMemoryManager,
    AgentActivity,
    # 向后兼容别名
    ZepGraphMemoryUpdater,
    ZepGraphMemoryManager,
)

__all__ = [
    'OntologyGenerator',
    'GraphBuilderService',
    'TextProcessor',
    'GraphRAGMemoryUpdater',
    'GraphRAGMemoryManager',
    'AgentActivity',
    # 向后兼容别名
    'ZepGraphMemoryUpdater',
    'ZepGraphMemoryManager',
]
