"""自适应混合索引主类
集成所有自适应组件，提供统一接口
"""

import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .utils.interfaces import BaseRetriever, Document, Query, RetrievalResult
from .analysis.query_feature_analyzer import QueryFeatureAnalyzer, QueryFeatures
from .adaptive.adaptive_router_v2 import AdaptiveRouter
from .fusion.adaptive_fusion_v2 import AdaptiveFusion
from .retriever.efficient_vector_index import EfficientVectorIndex
from .retriever.semantic_bm25 import SemanticBM25
from .retriever.bm25_retriever import BM25Retriever
from .retriever.dense_retriever import DenseRetriever


class AdaptiveHybridIndex(BaseRetriever):
    """自适应混合索引主类
    
    集成查询分析器、自适应路由器、多种检索器和自适应融合引擎
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        name = config.get('name', 'adaptive_hybrid') if config else 'adaptive_hybrid'
        super().__init__(name=name, config=config or {})
        
        # 初始化组件
        self.query_analyzer = None
        self.adaptive_router = None
        self.adaptive_fusion = None
        self.retrievers = {}
        
        # 数据
        self.documents = []
        self.is_built = False
        
        # 初始化各组件
        self._init_components()
    
    def _init_components(self):
        """初始化各组件"""
        # 查询分析器
        analyzer_config = self.config.get('query_analyzer', {})
        self.query_analyzer = QueryFeatureAnalyzer(analyzer_config)
        
        # 自适应路由器
        router_config = self.config.get('adaptive_router', {})
        router_config['available_retrievers'] = ['efficient_vector', 'semantic_bm25']
        self.adaptive_router = AdaptiveRouter(router_config)
        
        # 自适应融合引擎
        fusion_config = self.config.get('adaptive_fusion', {})
        self.adaptive_fusion = AdaptiveFusion(fusion_config)
        
        # 初始化检索器
        self._init_retrievers()
    
    def _init_retrievers(self):
        """初始化检索器"""
        retrievers_config = self.config.get('retrievers', {})
        
        # 高效向量检索器
        vector_config = retrievers_config.get('efficient_vector', {})
        self.retrievers['efficient_vector'] = EfficientVectorIndex("efficient_vector", vector_config)
        
        # 语义增强BM25检索器
        bm25_config = retrievers_config.get('semantic_bm25', {})
        self.retrievers['semantic_bm25'] = SemanticBM25(config=bm25_config)
    
    def build_index(self, documents: List[Document]) -> None:
        """构建索引"""
        print(f"构建自适应混合索引，文档数量: {len(documents)}")
        
        self.documents = documents
        
        # 构建各个检索器的索引
        for name, retriever in self.retrievers.items():
            print(f"构建 {name} 索引...")
            start_time = time.time()
            retriever.build_index(documents)
            end_time = time.time()
            print(f"{name} 索引构建完成，耗时: {end_time - start_time:.2f}秒")
        
        self.is_built = True
        print("自适应混合索引构建完成")
    
    def retrieve(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """检索相关文档"""
        if not self.is_built:
            raise RuntimeError("索引尚未构建，请先调用build_index方法")
        
        # 分析查询特征
        features = self.query_analyzer.analyze(query)
        
        # 获取路由决策
        routing_decision = self.adaptive_router.route(query)
        
        # 执行检索
        retrieval_results = {}
        
        # 主检索器
        primary_retriever = self.retrievers[routing_decision.primary_retriever]
        primary_results = primary_retriever.retrieve(query, top_k)
        retrieval_results[routing_decision.primary_retriever] = primary_results
        
        # 次级检索器
        for secondary_name in routing_decision.secondary_retrievers:
            if secondary_name in self.retrievers:
                secondary_retriever = self.retrievers[secondary_name]
                secondary_results = secondary_retriever.retrieve(query, top_k)
                retrieval_results[secondary_name] = secondary_results
        
        # 融合结果
        if len(retrieval_results) > 1:
            fusion_results = self.adaptive_fusion.fuse(
                retrieval_results, 
                query, 
                features, 
                routing_decision.fusion_weights
            )
            
            # 转换为RetrievalResult格式
            final_results = []
            for fusion_result in fusion_results[:top_k]:
                retrieval_result = RetrievalResult(
                    doc_id=fusion_result.doc_id,
                    score=fusion_result.final_score,
                    document=fusion_result.document,
                    retriever_name=self.name
                )
                final_results.append(retrieval_result)
            
            return final_results
        else:
            # 只有一个检索器，直接返回结果
            return list(retrieval_results.values())[0]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'document_count': len(self.documents),
            'is_built': self.is_built,
            'retrievers': {}
        }
        
        # 各检索器统计
        for name, retriever in self.retrievers.items():
            stats['retrievers'][name] = retriever.get_statistics()
        
        # 路由器统计
        if self.adaptive_router:
            stats['router'] = self.adaptive_router.get_statistics()
        
        return stats
    
    def save_index(self, index_path: str) -> None:
        """保存索引"""
        if not self.is_built:
            raise RuntimeError("索引尚未构建")
        
        index_dir = Path(index_path).parent
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存各检索器索引
        for name, retriever in self.retrievers.items():
            retriever_path = index_dir / f"{name}_index.pkl"
            retriever.save_index(str(retriever_path))
        
        # 保存元数据
        metadata = {
            'documents': self.documents,
            'config': self.config,
            'retrievers': list(self.retrievers.keys())
        }
        
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"自适应混合索引已保存到: {index_path}")
    
    def load_index(self, index_path: str) -> None:
        """加载索引"""
        try:
            # 加载元数据
            with open(index_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.documents = metadata['documents']
            self.config = metadata.get('config', self.config)
            
            # 加载各检索器索引
            index_dir = Path(index_path).parent
            for name in metadata['retrievers']:
                if name in self.retrievers:
                    retriever_path = index_dir / f"{name}_index.pkl"
                    if retriever_path.exists():
                        self.retrievers[name].load_index(str(retriever_path))
            
            self.is_built = True
            print(f"自适应混合索引已从 {index_path} 加载完成")
            
        except Exception as e:
            raise RuntimeError(f"加载索引失败: {e}")


def create_adaptive_hybrid_index(config: Dict[str, Any]) -> AdaptiveHybridIndex:
    """创建自适应混合索引的工厂函数"""
    return AdaptiveHybridIndex(config=config)