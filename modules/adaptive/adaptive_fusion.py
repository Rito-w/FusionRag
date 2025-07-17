"""自适应融合引擎
支持多种融合策略，并能根据查询特征动态调整融合权重
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
from collections import defaultdict

from ..utils.interfaces import BaseFusion, RetrievalResult, FusionResult, Query
from ..utils.common import FileUtils
from ..analysis.query_analyzer import QueryFeatures


class AdaptiveFusion(BaseFusion):
    """自适应融合引擎
    
    支持多种融合策略，并能根据查询特征动态调整融合权重
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.config = config or {}
        self.name = self.config.get('name', 'adaptive_fusion')
        
        # 融合方法配置
        self.default_method = self.config.get('default_method', 'weighted_sum')
        self.available_methods = self.config.get('available_methods', [
            'weighted_sum', 'reciprocal_rank_fusion', 'max_score', 'min_rank', 'score_normalization'
        ])
        
        # RRF参数
        self.rrf_k = self.config.get('rrf_k', 60)  # RRF常数
        
        # 归一化参数
        self.normalize_scores = self.config.get('normalize_scores', True)
        
        # 缓存
        self.fusion_cache = {}
    
    def _normalize_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """归一化检索结果分数"""
        if not results:
            return results
        
        # 按检索器分组
        retriever_groups = defaultdict(list)
        for result in results:
            # 调试：输出 retriever_name 的类型
            if not isinstance(result.retriever_name, str):
                print(f"[调试] 非字符串 retriever_name: {result.retriever_name}, 类型: {type(result.retriever_name)}，result: {result}")
            retriever_groups[result.retriever_name].append(result)
        
        # 对每个检索器的结果进行归一化
        normalized_results = []
        for retriever_name, group in retriever_groups.items():
            # 找出最大和最小分数
            scores = [r.score for r in group]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            score_range = max_score - min_score
            
            # 归一化分数
            for result in group:
                normalized_result = RetrievalResult(
                    doc_id=result.doc_id,
                    score=(result.score - min_score) / score_range if score_range > 0 else result.score,
                    document=result.document,
                    retriever_name=result.retriever_name
                )
                normalized_results.append(normalized_result)
        
        return normalized_results
    
    def _weighted_sum_fusion(self, results: List[RetrievalResult], weights: Dict[str, float] = None) -> List[FusionResult]:
        """加权求和融合"""
        if not results:
            return []
        
        # 如果需要归一化分数
        if self.normalize_scores:
            results = self._normalize_scores(results)
        
        # 默认权重
        if weights is None:
            weights = {}
            retriever_names = set(r.retriever_name for r in results)
            for name in retriever_names:
                weights[name] = 1.0 / len(retriever_names)
        
        # 按文档ID分组
        doc_groups = defaultdict(list)
        for result in results:
            doc_groups[result.doc_id].append(result)
        
        # 计算加权分数
        fusion_results = []
        for doc_id, group in doc_groups.items():
            # 计算加权分数
            weighted_score = 0.0
            source_retrievers = []
            document = None
            
            for result in group:
                retriever_weight = weights.get(result.retriever_name, 1.0)
                weighted_score += result.score * retriever_weight
                source_retrievers.append(result.retriever_name)
                if document is None:
                    document = result.document
            
            # 创建融合结果
            individual_scores = {result.retriever_name: result.score for result in group}
            fusion_result = FusionResult(
                doc_id=doc_id,
                final_score=weighted_score,
                document=document,
                individual_scores=individual_scores
            )
            fusion_results.append(fusion_result)
        
        # 按分数排序
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        return fusion_results
    
    def _reciprocal_rank_fusion(self, results: List[RetrievalResult]) -> List[FusionResult]:
        """倒数排名融合 (RRF)"""
        if not results:
            return []
        
        # 按检索器分组并排序
        retriever_groups = defaultdict(list)
        for result in results:
            retriever_groups[result.retriever_name].append(result)
        
        # 计算每个检索器中文档的排名
        doc_ranks = defaultdict(dict)
        for retriever_name, group in retriever_groups.items():
            # 按分数排序
            sorted_group = sorted(group, key=lambda x: x.score, reverse=True)
            
            # 记录排名
            for rank, result in enumerate(sorted_group):
                doc_ranks[result.doc_id][retriever_name] = rank + 1  # 排名从1开始
        
        # 计算RRF分数
        fusion_results = []
        for doc_id, ranks in doc_ranks.items():
            # 计算RRF分数: sum(1 / (k + rank))
            rrf_score = sum(1.0 / (self.rrf_k + rank) for rank in ranks.values())
            
            # 找到文档对象
            document = None
            individual_scores = {}
            for result in results:
                if result.doc_id == doc_id:
                    document = result.document
                    individual_scores[result.retriever_name] = result.score
            
            # 创建融合结果
            fusion_result = FusionResult(
                doc_id=doc_id,
                final_score=rrf_score,
                document=document,
                individual_scores=individual_scores
            )
            fusion_results.append(fusion_result)
        
        # 按分数排序
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        return fusion_results
    
    def _max_score_fusion(self, results: List[RetrievalResult]) -> List[FusionResult]:
        """最大分数融合"""
        if not results:
            return []
        
        # 如果需要归一化分数
        if self.normalize_scores:
            results = self._normalize_scores(results)
        
        # 按文档ID分组
        doc_groups = defaultdict(list)
        for result in results:
            doc_groups[result.doc_id].append(result)
        
        # 取每个文档的最高分数
        fusion_results = []
        for doc_id, group in doc_groups.items():
            # 找出最高分数的结果
            max_result = max(group, key=lambda x: x.score)
            
            # 创建融合结果
            individual_scores = {r.retriever_name: r.score for r in group}
            fusion_result = FusionResult(
                doc_id=doc_id,
                final_score=max_result.score,
                document=max_result.document,
                individual_scores=individual_scores
            )
            fusion_results.append(fusion_result)
        
        # 按分数排序
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        return fusion_results
    
    def _min_rank_fusion(self, results: List[RetrievalResult]) -> List[FusionResult]:
        """最小排名融合"""
        if not results:
            return []
        
        # 按检索器分组并排序
        retriever_groups = defaultdict(list)
        for result in results:
            retriever_groups[result.retriever_name].append(result)
        
        # 计算每个检索器中文档的排名
        doc_ranks = defaultdict(dict)
        for retriever_name, group in retriever_groups.items():
            # 按分数排序
            sorted_group = sorted(group, key=lambda x: x.score, reverse=True)
            
            # 记录排名
            for rank, result in enumerate(sorted_group):
                doc_ranks[result.doc_id][retriever_name] = rank + 1  # 排名从1开始
        
        # 取每个文档的最小排名
        fusion_results = []
        for doc_id, ranks in doc_ranks.items():
            # 找出最小排名
            min_rank = min(ranks.values())
            
            # 找到文档对象
            document = None
            individual_scores = {}
            for result in results:
                if result.doc_id == doc_id:
                    document = result.document
                    individual_scores[result.retriever_name] = result.score
            
            # 创建融合结果（分数为1/rank，使得排名越小分数越高）
            fusion_result = FusionResult(
                doc_id=doc_id,
                final_score=1.0 / min_rank,
                document=document,
                individual_scores=individual_scores
            )
            fusion_results.append(fusion_result)
        
        # 按分数排序
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        return fusion_results
    
    def _score_normalization_fusion(self, results: List[RetrievalResult]) -> List[FusionResult]:
        """分数归一化融合"""
        if not results:
            return []
        
        # 归一化分数
        normalized_results = self._normalize_scores(results)
        
        # 按文档ID分组
        doc_groups = defaultdict(list)
        for result in normalized_results:
            doc_groups[result.doc_id].append(result)
        
        # 计算平均归一化分数
        fusion_results = []
        for doc_id, group in doc_groups.items():
            # 计算平均分数
            avg_score = sum(r.score for r in group) / len(group)
            
            # 创建融合结果
            individual_scores = {r.retriever_name: r.score for r in group}
            fusion_result = FusionResult(
                doc_id=doc_id,
                final_score=avg_score,
                document=group[0].document,
                individual_scores=individual_scores
            )
            fusion_results.append(fusion_result)
        
        # 按分数排序
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        return fusion_results
    
    def fuse(self, query: Query, results: List[RetrievalResult], method: str = None, weights: Dict[str, float] = None, top_k: int = 10) -> List[FusionResult]:
        """融合多个检索器的结果"""
        if not results:
            return []
        
        # 使用指定方法或默认方法
        fusion_method = method if method in self.available_methods else self.default_method
        
        # 检查缓存
        cache_key = (query.query_id, fusion_method, tuple(sorted(weights.items())) if weights else None)
        if cache_key in self.fusion_cache:
            return self.fusion_cache[cache_key][:top_k]
        
        # 根据方法选择融合策略
        if fusion_method == 'weighted_sum':
            fusion_results = self._weighted_sum_fusion(results, weights)
        elif fusion_method == 'reciprocal_rank_fusion':
            fusion_results = self._reciprocal_rank_fusion(results)
        elif fusion_method == 'max_score':
            fusion_results = self._max_score_fusion(results)
        elif fusion_method == 'min_rank':
            fusion_results = self._min_rank_fusion(results)
        elif fusion_method == 'score_normalization':
            fusion_results = self._score_normalization_fusion(results)
        else:
            # 默认使用加权求和
            fusion_results = self._weighted_sum_fusion(results, weights)
        
        # 缓存结果
        self.fusion_cache[cache_key] = fusion_results
        
        # 返回top-k结果
        return fusion_results[:top_k]
    
    def adaptive_fuse(self, query: Query, results: List[RetrievalResult], query_features: QueryFeatures, 
                      method: str = None, weights: Dict[str, float] = None, top_k: int = 10) -> List[FusionResult]:
        """根据查询特征自适应融合"""
        if not results:
            return []
        
        # 如果没有指定方法，根据查询特征选择方法
        if method is None:
            method = self._select_fusion_method(query_features)
        
        # 如果没有指定权重，根据查询特征计算权重
        if weights is None:
            weights = self._calculate_weights(query_features, results)
        
        # 执行融合
        return self.fuse(query, results, method, weights, top_k)
    
    def _select_fusion_method(self, query_features: QueryFeatures) -> str:
        """根据查询特征选择融合方法"""
        # 根据查询类型选择合适的融合方法
        query_type = query_features.query_type
        
        if query_type.value == 'KEYWORD':
            # 关键词查询适合使用加权求和
            return 'weighted_sum'
        elif query_type.value == 'SEMANTIC':
            # 语义查询适合使用RRF
            return 'reciprocal_rank_fusion'
        elif query_type.value == 'ENTITY':
            # 实体查询适合使用最大分数
            return 'max_score'
        else:  # HYBRID或UNKNOWN
            # 混合查询使用分数归一化
            return 'score_normalization'
    
    def _calculate_weights(self, query_features: QueryFeatures, results: List[RetrievalResult]) -> Dict[str, float]:
        """根据查询特征计算检索器权重"""
        # 获取所有检索器名称
        retriever_names = set(r.retriever_name for r in results)
        
        # 默认平均权重
        weights = {name: 1.0 / len(retriever_names) for name in retriever_names}
        
        # 根据查询类型调整权重
        query_type = query_features.query_type
        
        if query_type.value == 'KEYWORD':
            # 关键词查询优先BM25
            for name in retriever_names:
                if 'bm25' in name.lower():
                    weights[name] = 0.6
                else:
                    weights[name] = 0.4 / (len(retriever_names) - 1) if len(retriever_names) > 1 else 0.4
        
        elif query_type.value == 'SEMANTIC':
            # 语义查询优先向量索引
            for name in retriever_names:
                if any(term in name.lower() for term in ['vector', 'dense', 'hnsw']):
                    weights[name] = 0.7
                else:
                    weights[name] = 0.3 / (len(retriever_names) - 1) if len(retriever_names) > 1 else 0.3
        
        elif query_type.value == 'ENTITY':
            # 实体查询优先图索引或BM25
            has_graph = any('graph' in name.lower() for name in retriever_names)
            
            if has_graph:
                for name in retriever_names:
                    if 'graph' in name.lower():
                        weights[name] = 0.7
                    else:
                        weights[name] = 0.3 / (len(retriever_names) - 1) if len(retriever_names) > 1 else 0.3
            else:
                for name in retriever_names:
                    if 'bm25' in name.lower():
                        weights[name] = 0.6
                    else:
                        weights[name] = 0.4 / (len(retriever_names) - 1) if len(retriever_names) > 1 else 0.4
        
        # 归一化权重，确保总和为1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        
        return weights
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self.fusion_cache = {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取融合引擎统计信息"""
        stats = {
            'name': self.name,
            'default_method': self.default_method,
            'available_methods': self.available_methods,
            'rrf_k': self.rrf_k,
            'normalize_scores': self.normalize_scores,
            'cache_size': len(self.fusion_cache)
        }
        return stats


def create_adaptive_fusion(config: Dict[str, Any] = None) -> AdaptiveFusion:
    """创建自适应融合引擎的工厂函数"""
    return AdaptiveFusion(config=config)