"""
多路检索结果融合模块
支持多种融合策略：加权融合、RRF、动态权重等
"""

import math
from typing import List, Dict, Any, Optional
from collections import defaultdict

from ..utils.interfaces import BaseFusion, Query, RetrievalResult, FusionResult


class MultiFusion(BaseFusion):
    """多路检索结果融合器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # 融合参数
        self.method = self.config.get('method', 'weighted')
        self.weights = self.config.get('weights', {})
        self.top_k = self.config.get('top_k', 20)
        self.rrf_k = self.config.get('rrf_k', 60)  # RRF参数
        
        # 支持的融合方法
        self.fusion_methods = {
            'weighted': self._weighted_fusion,
            'rrf': self._reciprocal_rank_fusion,
            'combsum': self._combsum_fusion,
            'combmnz': self._combmnz_fusion,
            'dynamic': self._dynamic_fusion
        }
    
    def fuse(self, 
             retrieval_results: Dict[str, List[RetrievalResult]], 
             query: Query) -> List[FusionResult]:
        """融合多个检索器的结果"""
        
        if not retrieval_results:
            return []
        
        # 选择融合方法
        fusion_func = self.fusion_methods.get(self.method, self._weighted_fusion)
        
        # 执行融合
        fused_results = fusion_func(retrieval_results, query)
        
        # 按最终分数排序并返回top-k
        fused_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return fused_results[:self.top_k]
    
    def _weighted_fusion(self, 
                        retrieval_results: Dict[str, List[RetrievalResult]], 
                        query: Query) -> List[FusionResult]:
        """加权线性融合"""
        
        # 收集所有文档
        doc_scores = defaultdict(lambda: {'scores': {}, 'document': None})
        
        for retriever_name, results in retrieval_results.items():
            weight = self.weights.get(retriever_name, 1.0)
            
            for result in results:
                doc_id = result.doc_id
                doc_scores[doc_id]['scores'][retriever_name] = result.score * weight
                doc_scores[doc_id]['document'] = result.document
        
        # 计算融合分数
        fused_results = []
        for doc_id, data in doc_scores.items():
            final_score = sum(data['scores'].values())
            
            if final_score > 0 and data['document']:
                fusion_result = FusionResult(
                    doc_id=doc_id,
                    final_score=final_score,
                    document=data['document'],
                    individual_scores=data['scores']
                )
                fused_results.append(fusion_result)
        
        return fused_results
    
    def _reciprocal_rank_fusion(self, 
                               retrieval_results: Dict[str, List[RetrievalResult]], 
                               query: Query) -> List[FusionResult]:
        """倒数排名融合(RRF)"""
        
        doc_scores = defaultdict(lambda: {'rrf_score': 0.0, 'scores': {}, 'document': None})
        
        for retriever_name, results in retrieval_results.items():
            for rank, result in enumerate(results):
                doc_id = result.doc_id
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                
                doc_scores[doc_id]['rrf_score'] += rrf_score
                doc_scores[doc_id]['scores'][retriever_name] = result.score
                doc_scores[doc_id]['document'] = result.document
        
        # 生成融合结果
        fused_results = []
        for doc_id, data in doc_scores.items():
            if data['rrf_score'] > 0 and data['document']:
                fusion_result = FusionResult(
                    doc_id=doc_id,
                    final_score=data['rrf_score'],
                    document=data['document'],
                    individual_scores=data['scores']
                )
                fused_results.append(fusion_result)
        
        return fused_results
    
    def _combsum_fusion(self, 
                       retrieval_results: Dict[str, List[RetrievalResult]], 
                       query: Query) -> List[FusionResult]:
        """CombSUM融合 - 简单分数相加"""
        
        # 先归一化分数
        normalized_results = self._normalize_scores(retrieval_results)
        
        doc_scores = defaultdict(lambda: {'scores': {}, 'document': None})
        
        for retriever_name, results in normalized_results.items():
            for result in results:
                doc_id = result.doc_id
                doc_scores[doc_id]['scores'][retriever_name] = result.score
                doc_scores[doc_id]['document'] = result.document
        
        # 计算融合分数
        fused_results = []
        for doc_id, data in doc_scores.items():
            final_score = sum(data['scores'].values())
            
            if final_score > 0 and data['document']:
                fusion_result = FusionResult(
                    doc_id=doc_id,
                    final_score=final_score,
                    document=data['document'],
                    individual_scores=data['scores']
                )
                fused_results.append(fusion_result)
        
        return fused_results
    
    def _combmnz_fusion(self, 
                       retrieval_results: Dict[str, List[RetrievalResult]], 
                       query: Query) -> List[FusionResult]:
        """CombMNZ融合 - 分数相加乘以检索器数量"""
        
        # 先进行CombSUM
        combsum_results = self._combsum_fusion(retrieval_results, query)
        
        # 乘以检索器数量
        for result in combsum_results:
            num_retrievers = len(result.individual_scores)
            result.final_score *= num_retrievers
        
        return combsum_results
    
    def _dynamic_fusion(self, 
                       retrieval_results: Dict[str, List[RetrievalResult]], 
                       query: Query) -> List[FusionResult]:
        """动态权重融合 - 基于查询类型或检索器性能"""
        
        # 计算每个检索器的动态权重
        dynamic_weights = self._calculate_dynamic_weights(retrieval_results, query)
        
        # 更新权重并使用加权融合
        original_weights = self.weights.copy()
        self.weights.update(dynamic_weights)
        
        try:
            results = self._weighted_fusion(retrieval_results, query)
        finally:
            # 恢复原始权重
            self.weights = original_weights
        
        return results
    
    def _calculate_dynamic_weights(self, 
                                  retrieval_results: Dict[str, List[RetrievalResult]], 
                                  query: Query) -> Dict[str, float]:
        """计算动态权重"""
        
        weights = {}
        
        # 基于检索结果数量和平均分数计算权重
        for retriever_name, results in retrieval_results.items():
            if not results:
                weights[retriever_name] = 0.0
                continue
            
            # 计算平均分数
            avg_score = sum(r.score for r in results) / len(results)
            
            # 计算覆盖率（返回结果数量）
            coverage = len(results) / max(len(r) for r in retrieval_results.values())
            
            # 动态权重 = 平均分数 * 覆盖率权重
            weights[retriever_name] = avg_score * (0.7 + 0.3 * coverage)
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _normalize_scores(self, 
                         retrieval_results: Dict[str, List[RetrievalResult]]
                         ) -> Dict[str, List[RetrievalResult]]:
        """归一化检索分数到[0,1]区间"""
        
        normalized_results = {}
        
        for retriever_name, results in retrieval_results.items():
            if not results:
                normalized_results[retriever_name] = []
                continue
            
            scores = [r.score for r in results]
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                # 所有分数相同，归一化为1
                normalized_results[retriever_name] = [
                    RetrievalResult(
                        doc_id=r.doc_id,
                        score=1.0,
                        document=r.document,
                        retriever_name=r.retriever_name
                    ) for r in results
                ]
            else:
                # 线性归一化
                normalized_results[retriever_name] = [
                    RetrievalResult(
                        doc_id=r.doc_id,
                        score=(r.score - min_score) / (max_score - min_score),
                        document=r.document,
                        retriever_name=r.retriever_name
                    ) for r in results
                ]
        
        return normalized_results
    
    def get_fusion_statistics(self, 
                             fused_results: List[FusionResult]
                             ) -> Dict[str, Any]:
        """获取融合统计信息"""
        
        if not fused_results:
            return {}
        
        # 统计每个检索器的贡献
        retriever_contributions = defaultdict(int)
        for result in fused_results:
            for retriever_name in result.individual_scores.keys():
                retriever_contributions[retriever_name] += 1
        
        # 计算分数分布
        final_scores = [r.final_score for r in fused_results]
        
        return {
            'total_documents': len(fused_results),
            'fusion_method': self.method,
            'retriever_contributions': dict(retriever_contributions),
            'score_statistics': {
                'mean': sum(final_scores) / len(final_scores),
                'max': max(final_scores),
                'min': min(final_scores)
            },
            'weights_used': self.weights.copy()
        }


def create_fusion(config: Dict[str, Any]) -> MultiFusion:
    """创建融合器的工厂函数"""
    return MultiFusion(config)