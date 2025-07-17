"""优化的自适应混合索引
集成并行处理和缓存机制
"""

import time
import asyncio
from typing import List, Dict, Any, Optional
from modules.adaptive_hybrid_index import AdaptiveHybridIndex
from modules.utils.parallel_retriever import ParallelRetrieverManager
from modules.utils.query_cache import QueryCache, CachedRetriever
from modules.utils.interfaces import Query, RetrievalResult


class OptimizedAdaptiveHybridIndex(AdaptiveHybridIndex):
    """优化的自适应混合索引"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # 并行处理配置
        self.enable_parallel = self.config.get('enable_parallel', True)
        self.max_workers = self.config.get('max_workers', 4)
        
        # 缓存配置
        self.enable_cache = self.config.get('enable_cache', True)
        self.cache_config = self.config.get('cache_config', {})
        
        # 性能监控
        self.performance_stats = {
            'total_queries': 0,
            'avg_latency': 0,
            'cache_hit_rate': 0,
            'parallel_speedup': 0
        }
        
        # 初始化优化组件
        self._init_optimization_components()
    
    def _init_optimization_components(self):
        """初始化优化组件"""
        # 查询缓存
        if self.enable_cache:
            self.query_cache = QueryCache(**self.cache_config)
            # 包装检索器
            self.cached_retrievers = {
                name: CachedRetriever(retriever, self.query_cache)
                for name, retriever in self.retrievers.items()
            }
        else:
            self.cached_retrievers = self.retrievers
        
        # 并行检索管理器
        if self.enable_parallel:
            self.parallel_manager = ParallelRetrieverManager(
                self.cached_retrievers, 
                self.max_workers
            )
    
    def retrieve(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """优化的检索方法"""
        start_time = time.time()
        
        try:
            # 分析查询特征
            features = self.query_analyzer.analyze(query)
            
            # 获取路由决策
            routing_decision = self.adaptive_router.route(query)
            
            # 选择检索策略
            if self.enable_parallel and len(routing_decision.secondary_retrievers) > 0:
                # 并行检索
                results = self._parallel_retrieve(query, routing_decision, top_k)
            else:
                # 串行检索
                results = self._sequential_retrieve(query, routing_decision, top_k)
            
            # 更新性能统计
            latency = time.time() - start_time
            self._update_performance_stats(latency)
            
            return results
            
        except Exception as e:
            print(f"优化检索失败: {e}")
            # 回退到基础检索
            return super().retrieve(query, top_k)
    
    def _parallel_retrieve(self, query: Query, routing_decision, top_k: int) -> List[RetrievalResult]:
        """并行检索"""
        # 确定要使用的检索器
        retrievers_to_use = {
            routing_decision.primary_retriever: self.cached_retrievers[routing_decision.primary_retriever]
        }
        
        for secondary_name in routing_decision.secondary_retrievers:
            if secondary_name in self.cached_retrievers:
                retrievers_to_use[secondary_name] = self.cached_retrievers[secondary_name]
        
        # 并行执行检索
        retrieval_results = self.parallel_manager.retrieve_parallel(query, top_k)
        
        # 过滤结果
        filtered_results = {
            name: results for name, results in retrieval_results.items()
            if name in retrievers_to_use
        }
        
        return self._fuse_results(filtered_results, query, routing_decision, top_k)
    
    def _sequential_retrieve(self, query: Query, routing_decision, top_k: int) -> List[RetrievalResult]:
        """串行检索"""
        retrieval_results = {}
        
        # 主检索器
        primary_retriever = self.cached_retrievers[routing_decision.primary_retriever]
        primary_results = primary_retriever.retrieve(query, top_k)
        retrieval_results[routing_decision.primary_retriever] = primary_results
        
        # 次级检索器
        for secondary_name in routing_decision.secondary_retrievers:
            if secondary_name in self.cached_retrievers:
                secondary_retriever = self.cached_retrievers[secondary_name]
                secondary_results = secondary_retriever.retrieve(query, top_k)
                retrieval_results[secondary_name] = secondary_results
        
        return self._fuse_results(retrieval_results, query, routing_decision, top_k)
    
    def _fuse_results(self, retrieval_results: Dict[str, List[RetrievalResult]], 
                     query: Query, routing_decision, top_k: int) -> List[RetrievalResult]:
        """融合结果"""
        if len(retrieval_results) > 1:
            # 分析查询特征
            features = self.query_analyzer.analyze(query)
            
            # 自适应融合
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
            # 只有一个检索器
            return list(retrieval_results.values())[0]
    
    async def retrieve_async(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """异步检索"""
        if not self.enable_parallel:
            # 在线程池中执行同步检索
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.retrieve, query, top_k)
        
        start_time = time.time()
        
        try:
            # 分析查询特征
            features = self.query_analyzer.analyze(query)
            
            # 获取路由决策
            routing_decision = self.adaptive_router.route(query)
            
            # 异步并行检索
            retrieval_results = await self.parallel_manager.retrieve_async(query, top_k)
            
            # 过滤和融合结果
            filtered_results = {
                name: results for name, results in retrieval_results.items()
                if name in [routing_decision.primary_retriever] + routing_decision.secondary_retrievers
            }
            
            results = self._fuse_results(filtered_results, query, routing_decision, top_k)
            
            # 更新性能统计
            latency = time.time() - start_time
            self._update_performance_stats(latency)
            
            return results
            
        except Exception as e:
            print(f"异步检索失败: {e}")
            # 回退到同步检索
            return self.retrieve(query, top_k)
    
    def _update_performance_stats(self, latency: float):
        """更新性能统计"""
        self.performance_stats['total_queries'] += 1
        
        # 更新平均延迟
        total_queries = self.performance_stats['total_queries']
        prev_avg = self.performance_stats['avg_latency']
        self.performance_stats['avg_latency'] = (
            (prev_avg * (total_queries - 1) + latency) / total_queries
        )
        
        # 更新缓存命中率
        if self.enable_cache:
            total_hits = sum(
                retriever.cache_hits for retriever in self.cached_retrievers.values()
            )
            total_requests = sum(
                retriever.cache_hits + retriever.cache_misses
                for retriever in self.cached_retrievers.values()
            )
            self.performance_stats['cache_hit_rate'] = (
                total_hits / total_requests if total_requests > 0 else 0
            )
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        stats = {
            'performance': self.performance_stats,
            'parallel_enabled': self.enable_parallel,
            'cache_enabled': self.enable_cache,
            'max_workers': self.max_workers
        }
        
        if self.enable_cache:
            stats['cache_stats'] = self.query_cache.get_stats()
            stats['retriever_cache_stats'] = {
                name: retriever.get_cache_stats()
                for name, retriever in self.cached_retrievers.items()
            }
        
        return stats
    
    def clear_cache(self):
        """清空缓存"""
        if self.enable_cache:
            self.query_cache.clear()
            for retriever in self.cached_retrievers.values():
                retriever.cache_hits = 0
                retriever.cache_misses = 0


def create_optimized_adaptive_hybrid_index(config: Dict[str, Any]) -> OptimizedAdaptiveHybridIndex:
    """创建优化的自适应混合索引"""
    return OptimizedAdaptiveHybridIndex(config=config)