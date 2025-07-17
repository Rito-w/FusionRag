"""
FusionRAG - 高效RAG系统工程落地
一个模块化、可扩展的检索增强生成系统

主要特性:
- 多检索器融合 (BM25, Dense Vector, Knowledge Graph)
- 智能查询分类和路由
- 自适应性能优化
- 标准化评测指标
- Neo4j图数据库支持
- 中英文支持
"""

__version__ = "1.0.0"
__author__ = "FusionRAG Team"

from .pipeline import FusionRAGPipeline

__all__ = [
    'FusionRAGPipeline'
]