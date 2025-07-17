"""自适应融合引擎
根据查询特征和历史性能动态调整融合策略
"""

import math
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter

from ..utils.interfaces import Query, RetrievalResult, BaseFusion, FusionResult
from ..fusion.fusion import MultiFusion
from ..analysis.query_feature_analyzer import QueryFeatures


class AdaptiveFusion(BaseFusion):
    """自适应融合引擎
    
    根据查询特征和历史性能动态调整融合策略和权重
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # 融合方法配置
        self.fusion_methods = self.config.get('fusion_methods', [
            'weighted_sum', 'rrf', 'combsum', 'combmnz', 'borda_count'
        ])
        
        self.default_method = self.config.get('default_method', 'weighted_sum')
        
        # 自适应参数
        self.enable_adaptive_weights = self.config.get('enable_adaptive_weights', True)
        self.enable_adaptive_method = self.config.get('enable_adaptive_method', True)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # 权重学习参数
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.weight_decay = self.config.get('weight_decay', 0.01)
        
        # 历史性能记录
        self.method_performance = defaultdict(list)
        self.weight_performance = defaultdict(list)
        
        # 初始化融合器
        self.fusion_engine = MultiFusion(config=self.config)
        
        # 方法选择规则
        self.method_selection_rules = {
            'factual': 'weighted_sum',      # 事实查询适合加权求和
            'conceptual': 'rrf',            # 概念查询适合倒数排名融合
            'analytical': 'combsum',        # 分析查询适合组合求和
            'procedural': 'borda_count',    # 程序查询适合Borda计数
            'general': 'weighted_sum'       # 一般查询适合加权求和
        }
    
    def fuse(self, 
             retrieval_results: Dict[str, List[RetrievalResult]], 
             query: Query,
             features: Optional[QueryFeatures] = None,
             weights: Optional[Dict[str, float]] = None) -> List[FusionResult]:
        """自适应融合检索结果"""
        
        # 如果没有特征，使用默认融合
        if features is None:
            return self.fusion_engine.fuse(retrieval_results, query, weights)
        
        # 选择融合方法
        fusion_method = self._select_fusion_method(features)
        
        # 计算自适应权重
        if weights is None or self.enable_adaptive_weights:
            adaptive_weights = self._calculate_adaptive_weights(
                retrieval_results, query, features, weights
            )
        else:
            adaptive_weights = weights
        
        # 执行融合
        fused_results = self._execute_fusion(
            retrieval_results, query, fusion_method, adaptive_weights
        )
        
        # 后处理
        final_results = self._post_process_results(fused_results, features)
        
        return final_results
    
    def _select_fusion_method(self, features: QueryFeatures) -> str:
        """选择融合方法"""
        if not self.enable_adaptive_method:
            return self.default_method
        
        # 基于查询类型选择
        method_by_type = self.method_selection_rules.get(features.query_type, self.default_method)
        
        # 基于历史性能调整
        if self.method_performance:
            # 找到相似查询的最佳方法
            best_method = self._find_best_method_for_similar_queries(features)
            
            if best_method and self._get_method_confidence(best_method) > self.confidence_threshold:
                return best_method
        
        # 基于查询特征的规则
        if features.complexity_score > 0.7:
            # 复杂查询使用RRF
            return 'rrf'
        elif features.entity_count > 3:
            # 多实体查询使用CombSUM
            return 'combsum'
        elif features.is_question:
            # 问句查询使用加权求和
            return 'weighted_sum'
        
        return method_by_type
    
    def _calculate_adaptive_weights(self, 
                                  retrieval_results: Dict[str, List[RetrievalResult]], 
                                  query: Query, 
                                  features: QueryFeatures,
                                  initial_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """计算自适应权重"""
        
        retrievers = list(retrieval_results.keys())
        
        # 初始权重
        if initial_weights:
            weights = initial_weights.copy()
        else:
            # 等权重初始化
            weights = {r: 1.0 / len(retrievers) for r in retrievers}
        
        # 基于查询特征调整权重
        weights = self._adjust_weights_by_features(weights, features)
        
        # 基于结果质量调整权重
        weights = self._adjust_weights_by_quality(weights, retrieval_results)
        
        # 基于历史性能调整权重
        weights = self._adjust_weights_by_history(weights, features)
        
        # 权重归一化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {r: w / total_weight for r, w in weights.items()}
        
        return weights
    
    def _adjust_weights_by_features(self, weights: Dict[str, float], features: QueryFeatures) -> Dict[str, float]:
        """基于查询特征调整权重"""
        adjusted_weights = weights.copy()
        
        # 基于查询类型调整
        if features.query_type == 'factual':
            # 事实查询偏向BM25
            if 'semantic_bm25' in adjusted_weights:
                adjusted_weights['semantic_bm25'] *= 1.2
            if 'efficient_vector' in adjusted_weights:
                adjusted_weights['efficient_vector'] *= 0.8
        
        elif features.query_type == 'conceptual':
            # 概念查询偏向向量检索
            if 'efficient_vector' in adjusted_weights:
                adjusted_weights['efficient_vector'] *= 1.2
            if 'semantic_bm25' in adjusted_weights:
                adjusted_weights['semantic_bm25'] *= 0.8
        
        # 基于复杂度调整
        if features.complexity_score > 0.7:
            # 复杂查询偏向向量检索
            if 'efficient_vector' in adjusted_weights:
                adjusted_weights['efficient_vector'] *= (1 + features.complexity_score * 0.3)
        
        # 基于实体数量调整
        if features.entity_count > 2:
            # 多实体查询偏向BM25
            if 'semantic_bm25' in adjusted_weights:
                adjusted_weights['semantic_bm25'] *= (1 + features.entity_count * 0.1)
        
        # 基于查询长度调整
        if features.token_count > 10:
            # 长查询偏向向量检索
            if 'efficient_vector' in adjusted_weights:
                adjusted_weights['efficient_vector'] *= 1.1
        elif features.token_count < 5:
            # 短查询偏向BM25
            if 'semantic_bm25' in adjusted_weights:
                adjusted_weights['semantic_bm25'] *= 1.1
        
        return adjusted_weights
    
    def _adjust_weights_by_quality(self, weights: Dict[str, float], 
                                 retrieval_results: Dict[str, List[RetrievalResult]]) -> Dict[str, float]:
        """基于结果质量调整权重"""
        adjusted_weights = weights.copy()
        
        # 计算每个检索器的质量指标
        quality_scores = {}
        
        for retriever, results in retrieval_results.items():
            if not results:
                quality_scores[retriever] = 0.0
                continue
            
            # 分数分布质量
            scores = [r.score for r in results]
            score_mean = np.mean(scores)
            score_std = np.std(scores)
            
            # 结果数量质量
            result_count_quality = min(1.0, len(results) / 10)  # 假设10个结果为理想
            
            # 分数区分度
            if score_std > 0:
                discrimination = min(1.0, score_std / score_mean)
            else:
                discrimination = 0.0
            
            # 综合质量分数
            quality_scores[retriever] = (
                score_mean * 0.4 + 
                result_count_quality * 0.3 + 
                discrimination * 0.3
            )
        
        # 根据质量分数调整权重
        if quality_scores:
            max_quality = max(quality_scores.values())
            if max_quality > 0:
                for retriever in adjusted_weights:
                    quality_ratio = quality_scores.get(retriever, 0) / max_quality
                    adjusted_weights[retriever] *= (0.7 + quality_ratio * 0.3)
        
        return adjusted_weights
    
    def _adjust_weights_by_history(self, weights: Dict[str, float], features: QueryFeatures) -> Dict[str, float]:
        """基于历史性能调整权重"""
        if not self.weight_performance:
            return weights
        
        adjusted_weights = weights.copy()
        
        # 找到相似查询的历史权重性能
        similar_performances = self._find_similar_weight_performances(features)
        
        if similar_performances:
            # 计算历史最优权重
            best_weights = self._calculate_optimal_weights_from_history(similar_performances)
            
            # 与当前权重融合
            alpha = 0.3  # 历史权重的影响因子
            for retriever in adjusted_weights:
                if retriever in best_weights:
                    adjusted_weights[retriever] = (
                        (1 - alpha) * adjusted_weights[retriever] + 
                        alpha * best_weights[retriever]
                    )
        
        return adjusted_weights
    
    def _execute_fusion(self, 
                       retrieval_results: Dict[str, List[RetrievalResult]], 
                       query: Query,
                       fusion_method: str,
                       weights: Dict[str, float]) -> List[FusionResult]:
        """执行融合"""
        
        if fusion_method == 'weighted_sum':
            return self._weighted_sum_fusion(retrieval_results, weights)
        elif fusion_method == 'rrf':
            return self._rrf_fusion(retrieval_results, weights)
        elif fusion_method == 'combsum':
            return self._combsum_fusion(retrieval_results, weights)
        elif fusion_method == 'combmnz':
            return self._combmnz_fusion(retrieval_results, weights)
        elif fusion_method == 'borda_count':
            return self._borda_count_fusion(retrieval_results, weights)
        else:
            # 默认使用加权求和
            return self._weighted_sum_fusion(retrieval_results, weights)
    
    def _weighted_sum_fusion(self, retrieval_results: Dict[str, List[RetrievalResult]], 
                           weights: Dict[str, float]) -> List[FusionResult]:
        """加权求和融合"""
        doc_scores = defaultdict(float)
        doc_retrievers = defaultdict(set)
        doc_objects = {}
        
        for retriever, results in retrieval_results.items():
            weight = weights.get(retriever, 0.0)
            
            for result in results:
                doc_id = result.doc_id
                doc_scores[doc_id] += result.score * weight
                doc_retrievers[doc_id].add(retriever)
                doc_objects[doc_id] = result.document
        
        # 生成融合结果
        fusion_results = []
        for doc_id, score in doc_scores.items():
            fusion_results.append(FusionResult(
                doc_id=doc_id,
                final_score=score,
                document=doc_objects[doc_id],
                individual_scores={retriever: score for retriever in doc_retrievers[doc_id]}
            ))
        
        return sorted(fusion_results, key=lambda x: x.final_score, reverse=True)
    
    def _rrf_fusion(self, retrieval_results: Dict[str, List[RetrievalResult]], 
                   weights: Dict[str, float], k: int = 60) -> List[FusionResult]:
        """倒数排名融合（RRF）"""
        doc_scores = defaultdict(float)
        doc_retrievers = defaultdict(set)
        doc_objects = {}
        
        for retriever, results in retrieval_results.items():
            weight = weights.get(retriever, 0.0)
            
            for rank, result in enumerate(results):
                doc_id = result.doc_id
                doc_scores[doc_id] += weight / (k + rank + 1)
                doc_retrievers[doc_id].add(retriever)
                doc_objects[doc_id] = result.document
        
        # 生成融合结果
        fusion_results = []
        for doc_id, score in doc_scores.items():
            fusion_results.append(FusionResult(
                doc_id=doc_id,
                final_score=score,
                document=doc_objects[doc_id],
                individual_scores={retriever: score for retriever in doc_retrievers[doc_id]}
            ))
        
        return sorted(fusion_results, key=lambda x: x.final_score, reverse=True)
    
    def _combsum_fusion(self, retrieval_results: Dict[str, List[RetrievalResult]], 
                       weights: Dict[str, float]) -> List[FusionResult]:
        """CombSUM融合"""
        doc_scores = defaultdict(float)
        doc_retrievers = defaultdict(set)
        doc_objects = {}
        
        # 首先归一化每个检索器的分数
        normalized_results = {}
        for retriever, results in retrieval_results.items():
            if not results:
                normalized_results[retriever] = []
                continue
            
            scores = [r.score for r in results]
            max_score = max(scores)
            min_score = min(scores)
            
            if max_score > min_score:
                normalized_results[retriever] = [
                    RetrievalResult(
                        doc_id=r.doc_id,
                        score=(r.score - min_score) / (max_score - min_score),
                        document=r.document,
                        retriever_name=r.retriever_name
                    ) for r in results
                ]
            else:
                normalized_results[retriever] = results
        
        # 计算CombSUM分数
        for retriever, results in normalized_results.items():
            weight = weights.get(retriever, 0.0)
            
            for result in results:
                doc_id = result.doc_id
                doc_scores[doc_id] += result.score * weight
                doc_retrievers[doc_id].add(retriever)
                doc_objects[doc_id] = result.document
        
        # 生成融合结果
        fusion_results = []
        for doc_id, score in doc_scores.items():
            fusion_results.append(FusionResult(
                doc_id=doc_id,
                final_score=score,
                document=doc_objects[doc_id],
                individual_scores={retriever: score for retriever in doc_retrievers[doc_id]}
            ))
        
        return sorted(fusion_results, key=lambda x: x.final_score, reverse=True)
    
    def _combmnz_fusion(self, retrieval_results: Dict[str, List[RetrievalResult]], 
                       weights: Dict[str, float]) -> List[FusionResult]:
        """CombMNZ融合"""
        # 先执行CombSUM
        combsum_results = self._combsum_fusion(retrieval_results, weights)
        
        # 计算每个文档被多少个检索器检索到
        doc_retriever_counts = defaultdict(int)
        for retriever, results in retrieval_results.items():
            retrieved_docs = set(r.doc_id for r in results)
            for doc_id in retrieved_docs:
                doc_retriever_counts[doc_id] += 1
        
        # 应用CombMNZ：分数乘以检索器数量
        for result in combsum_results:
            result.final_score *= doc_retriever_counts[result.doc_id]
        
        return sorted(combsum_results, key=lambda x: x.final_score, reverse=True)
    
    def _borda_count_fusion(self, retrieval_results: Dict[str, List[RetrievalResult]], 
                           weights: Dict[str, float]) -> List[FusionResult]:
        """Borda计数融合"""
        doc_scores = defaultdict(float)
        doc_retrievers = defaultdict(set)
        doc_objects = {}
        
        for retriever, results in retrieval_results.items():
            weight = weights.get(retriever, 0.0)
            n = len(results)
            
            for rank, result in enumerate(results):
                doc_id = result.doc_id
                # Borda分数：(n - rank) * weight
                doc_scores[doc_id] += (n - rank) * weight
                doc_retrievers[doc_id].add(retriever)
                doc_objects[doc_id] = result.document
        
        # 生成融合结果
        fusion_results = []
        for doc_id, score in doc_scores.items():
            fusion_results.append(FusionResult(
                doc_id=doc_id,
                final_score=score,
                document=doc_objects[doc_id],
                individual_scores={retriever: score for retriever in doc_retrievers[doc_id]}
            ))
        
        return sorted(fusion_results, key=lambda x: x.final_score, reverse=True)
    
    def _post_process_results(self, results: List[FusionResult], features: QueryFeatures) -> List[FusionResult]:
        """后处理融合结果"""
        # 基于查询特征调整结果
        
        # 多样性调整
        if features.query_type == 'analytical':
            # 分析性查询需要多样性
            results = self._apply_diversity_filter(results, diversity_threshold=0.8)
        
        # 置信度过滤
        if features.complexity_score > 0.8:
            # 复杂查询需要高置信度结果
            results = self._apply_confidence_filter(results, confidence_threshold=0.3)
        
        return results
    
    def _apply_diversity_filter(self, results: List[FusionResult], diversity_threshold: float = 0.8) -> List[FusionResult]:
        """应用多样性过滤"""
        if not results:
            return results
        
        filtered_results = [results[0]]  # 保留第一个结果
        
        for result in results[1:]:
            # 检查与已选结果的相似度
            is_diverse = True
            for selected in filtered_results:
                # 简单的文本相似度检查
                similarity = self._calculate_text_similarity(
                    result.document.text, selected.document.text
                )
                if similarity > diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                filtered_results.append(result)
        
        return filtered_results
    
    def _apply_confidence_filter(self, results: List[FusionResult], confidence_threshold: float) -> List[FusionResult]:
        """应用置信度过滤"""
        if not results:
            return results
        
        # 计算分数阈值
        max_score = max(r.score for r in results)
        threshold = max_score * confidence_threshold
        
        return [r for r in results if r.score >= threshold]
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 简单的Jaccard相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _find_best_method_for_similar_queries(self, features: QueryFeatures) -> Optional[str]:
        """为相似查询找到最佳方法"""
        # 简化实现：基于查询类型返回最佳方法
        method_performance = defaultdict(list)
        
        for method, performances in self.method_performance.items():
            for perf in performances:
                if perf.get('query_type') == features.query_type:
                    method_performance[method].append(perf.get('score', 0))
        
        if method_performance:
            avg_performances = {
                method: np.mean(scores) 
                for method, scores in method_performance.items()
            }
            return max(avg_performances, key=avg_performances.get)
        
        return None
    
    def _get_method_confidence(self, method: str) -> float:
        """获取方法置信度"""
        if method not in self.method_performance:
            return 0.0
        
        performances = self.method_performance[method]
        if not performances:
            return 0.0
        
        # 基于性能的一致性计算置信度
        scores = [p.get('score', 0) for p in performances]
        if len(scores) < 2:
            return 0.5
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # 置信度 = 平均性能 * (1 - 相对标准差)
        confidence = mean_score * (1 - std_score / mean_score) if mean_score > 0 else 0
        return min(1.0, max(0.0, confidence))
    
    def _find_similar_weight_performances(self, features: QueryFeatures) -> List[Dict[str, Any]]:
        """找到相似查询的权重性能"""
        similar_performances = []
        
        for performances in self.weight_performance.values():
            for perf in performances:
                # 简单的相似度匹配
                if perf.get('query_type') == features.query_type:
                    similar_performances.append(perf)
        
        return similar_performances
    
    def _calculate_optimal_weights_from_history(self, performances: List[Dict[str, Any]]) -> Dict[str, float]:
        """从历史记录计算最优权重"""
        if not performances:
            return {}
        
        # 找到表现最好的权重配置
        best_performance = max(performances, key=lambda p: p.get('score', 0))
        return best_performance.get('weights', {})
    
    def record_performance(self, query: Query, features: QueryFeatures, 
                          method: str, weights: Dict[str, float], 
                          performance_score: float) -> None:
        """记录性能"""
        # 记录方法性能
        self.method_performance[method].append({
            'query_type': features.query_type,
            'complexity': features.complexity_score,
            'score': performance_score,
            'timestamp': time.time()
        })
        
        # 记录权重性能
        weight_key = frozenset(weights.items())
        self.weight_performance[weight_key].append({
            'query_type': features.query_type,
            'complexity': features.complexity_score,
            'weights': weights,
            'score': performance_score,
            'timestamp': time.time()
        })
        
        # 保持历史记录大小
        max_history = 500
        if len(self.method_performance[method]) > max_history:
            self.method_performance[method] = self.method_performance[method][-max_history:]
        
        if len(self.weight_performance[weight_key]) > max_history:
            self.weight_performance[weight_key] = self.weight_performance[weight_key][-max_history:]


def create_adaptive_fusion(config: Dict[str, Any]) -> AdaptiveFusion:
    """创建自适应融合引擎的工厂函数"""
    return AdaptiveFusion(config=config)