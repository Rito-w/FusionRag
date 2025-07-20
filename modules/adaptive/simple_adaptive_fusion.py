#!/usr/bin/env python3
"""
简化版自适应融合引擎
专门为论文实验设计，集成查询分析器和路由器的输出
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from ..utils.interfaces import RetrievalResult, FusionResult, Query
from ..analysis.simple_query_analyzer import SimpleQueryFeatures
from ..adaptive.simple_adaptive_router import RoutingDecision


class SimpleAdaptiveFusion:
    """简化版自适应融合引擎
    
    根据路由决策执行相应的融合策略
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.config.get('name', 'simple_adaptive_fusion')
        
        # RRF参数
        self.rrf_k = self.config.get('rrf_k', 60)
        
        # 归一化参数
        self.normalize_scores = self.config.get('normalize_scores', True)
        
        # 融合统计
        self.fusion_stats = {
            'total_fusions': 0,
            'method_usage': defaultdict(int),
            'avg_scores': defaultdict(list)
        }
    
    def fuse(self, query: Query, retrieval_results: Dict[str, List[RetrievalResult]], 
             routing_decision: RoutingDecision, top_k: int = 10) -> List[FusionResult]:
        """根据路由决策执行融合"""
        
        # 提取选中检索器的结果
        selected_results = []
        for retriever_name in routing_decision.selected_retrievers:
            if retriever_name in retrieval_results:
                results = retrieval_results[retriever_name]
                selected_results.append(results)
        
        if not selected_results:
            return []
        
        # 根据融合方法执行融合
        fusion_method = routing_decision.fusion_method
        fusion_weights = routing_decision.fusion_weights
        
        if fusion_method == 'RRF':
            fusion_results = self._reciprocal_rank_fusion(selected_results, routing_decision.selected_retrievers)
        elif fusion_method == 'LinearEqual':
            equal_weights = {name: 1.0/len(routing_decision.selected_retrievers) 
                           for name in routing_decision.selected_retrievers}
            fusion_results = self._weighted_fusion(selected_results, routing_decision.selected_retrievers, equal_weights)
        elif fusion_method == 'LinearOptimized':
            fusion_results = self._weighted_fusion(selected_results, routing_decision.selected_retrievers, fusion_weights)
        else:
            # 默认使用RRF
            fusion_results = self._reciprocal_rank_fusion(selected_results, routing_decision.selected_retrievers)
        
        # 更新统计信息
        self.fusion_stats['total_fusions'] += 1
        self.fusion_stats['method_usage'][fusion_method] += 1
        
        # 返回top-k结果
        return fusion_results[:top_k]
    
    def _reciprocal_rank_fusion(self, results_list: List[List[RetrievalResult]], 
                               retriever_names: List[str]) -> List[FusionResult]:
        """倒数排名融合 (RRF)"""
        if not results_list:
            return []
        
        # 收集所有文档和排名
        all_docs = {}
        doc_ranks = defaultdict(dict)
        
        for i, (results, retriever_name) in enumerate(zip(results_list, retriever_names)):
            for rank, result in enumerate(results):
                doc_id = result.doc_id
                all_docs[doc_id] = result.document
                doc_ranks[doc_id][retriever_name] = rank + 1  # 排名从1开始
        
        # 计算RRF分数
        fusion_results = []
        for doc_id, ranks in doc_ranks.items():
            rrf_score = sum(1.0 / (self.rrf_k + rank) for rank in ranks.values())
            
            # 收集各检索器的原始分数
            individual_scores = {}
            for results, retriever_name in zip(results_list, retriever_names):
                for result in results:
                    if result.doc_id == doc_id:
                        individual_scores[retriever_name] = result.score
                        break
            
            fusion_result = FusionResult(
                doc_id=doc_id,
                final_score=rrf_score,
                document=all_docs[doc_id],
                individual_scores=individual_scores
            )
            fusion_results.append(fusion_result)
        
        # 按分数排序
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        return fusion_results
    
    def _weighted_fusion(self, results_list: List[List[RetrievalResult]], 
                        retriever_names: List[str], 
                        weights: Dict[str, float]) -> List[FusionResult]:
        """加权融合"""
        if not results_list:
            return []
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # 收集所有文档和分数
        all_docs = {}
        doc_scores = defaultdict(dict)
        
        for results, retriever_name in zip(results_list, retriever_names):
            if not results:
                continue
            
            weight = weights.get(retriever_name, 0.0)
            
            # 归一化分数到[0,1]
            if self.normalize_scores:
                scores = [r.score for r in results]
                max_score = max(scores) if scores else 1.0
                min_score = min(scores) if scores else 0.0
                score_range = max_score - min_score if max_score > min_score else 1.0
            
            for result in results:
                doc_id = result.doc_id
                all_docs[doc_id] = result.document
                
                # 归一化分数
                if self.normalize_scores:
                    normalized_score = (result.score - min_score) / score_range
                else:
                    normalized_score = result.score
                
                doc_scores[doc_id][retriever_name] = {
                    'normalized': normalized_score,
                    'original': result.score,
                    'weight': weight
                }
        
        # 计算加权分数
        fusion_results = []
        for doc_id, scores_dict in doc_scores.items():
            weighted_score = sum(
                info['normalized'] * info['weight'] 
                for info in scores_dict.values()
            )
            
            # 收集各检索器的原始分数
            individual_scores = {
                retriever: info['original'] 
                for retriever, info in scores_dict.items()
            }
            
            fusion_result = FusionResult(
                doc_id=doc_id,
                final_score=weighted_score,
                document=all_docs[doc_id],
                individual_scores=individual_scores
            )
            fusion_results.append(fusion_result)
        
        # 按分数排序
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        return fusion_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取融合统计信息"""
        stats = self.fusion_stats.copy()
        
        # 计算方法使用比例
        total = stats['total_fusions']
        if total > 0:
            method_percentages = {
                method: count/total * 100 
                for method, count in stats['method_usage'].items()
            }
            stats['method_percentages'] = method_percentages
        
        return stats
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.fusion_stats = {
            'total_fusions': 0,
            'method_usage': defaultdict(int),
            'avg_scores': defaultdict(list)
        }


def create_simple_adaptive_fusion(config: Dict[str, Any] = None) -> SimpleAdaptiveFusion:
    """创建简化自适应融合引擎的工厂函数"""
    return SimpleAdaptiveFusion(config=config)