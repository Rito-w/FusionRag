"""
高级融合策略 - 论文级别算法改进
包含多种先进的检索结果融合技术
"""

import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import logging

from ..utils.interfaces import RetrievalResult, FusionResult


class AdvancedFusionStrategies:
    """高级融合策略类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 高级参数
        self.alpha = self.config.get('alpha', 0.5)  # 用于自适应权重
        self.beta = self.config.get('beta', 0.1)    # 用于时间衰减
        self.gamma = self.config.get('gamma', 0.8)  # 用于置信度加权
        
    def adaptive_weighted_fusion(self, retrieval_results: Dict[str, List[RetrievalResult]], 
                                query: Any) -> List[FusionResult]:
        """自适应加权融合 - 根据检索器性能动态调整权重"""
        
        # 1. 计算每个检索器的置信度
        confidences = self._calculate_retriever_confidence(retrieval_results)
        
        # 2. 基于查询特征调整权重
        query_weights = self._get_query_adaptive_weights(query, retrieval_results)
        
        # 3. 结合置信度和查询特征的最终权重
        final_weights = {}
        for retriever in retrieval_results.keys():
            confidence = confidences.get(retriever, 0.5)
            query_weight = query_weights.get(retriever, 0.5)
            final_weights[retriever] = confidence * query_weight
        
        # 归一化权重
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {k: v/total_weight for k, v in final_weights.items()}
        
        self.logger.info(f"自适应权重: {final_weights}")
        
        # 4. 执行加权融合
        return self._weighted_score_fusion(retrieval_results, final_weights)
    
    def rank_biased_precision_fusion(self, retrieval_results: Dict[str, List[RetrievalResult]], 
                                   persistence: float = 0.8) -> List[FusionResult]:
        """基于排名偏置精度的融合方法"""
        
        # 收集所有文档
        all_docs = {}
        
        for retriever_name, results in retrieval_results.items():
            for rank, result in enumerate(results):
                doc_id = result.doc_id
                if doc_id not in all_docs:
                    all_docs[doc_id] = {
                        'doc': result.doc,
                        'scores': {},
                        'ranks': {},
                        'rbp_scores': {}
                    }
                
                # 计算RBP权重
                rbp_weight = persistence ** rank
                all_docs[doc_id]['rbp_scores'][retriever_name] = rbp_weight
                all_docs[doc_id]['scores'][retriever_name] = result.score
                all_docs[doc_id]['ranks'][retriever_name] = rank
        
        # 计算融合后的RBP分数
        fusion_results = []
        for doc_id, doc_info in all_docs.items():
            # 计算加权RBP分数
            total_rbp = sum(doc_info['rbp_scores'].values())
            
            fusion_result = FusionResult(
                doc_id=doc_id,
                score=total_rbp,
                doc=doc_info['doc'],
                component_scores=doc_info['scores'],
                fusion_method="rbp_fusion"
            )
            fusion_results.append(fusion_result)
        
        # 按分数排序
        fusion_results.sort(key=lambda x: x.score, reverse=True)
        return fusion_results
    
    def neural_score_fusion(self, retrieval_results: Dict[str, List[RetrievalResult]],
                           query: Any) -> List[FusionResult]:
        """神经网络风格的分数融合（简化版）"""
        
        # 特征提取
        features = self._extract_fusion_features(retrieval_results, query)
        
        # 简化的神经网络计算（可以替换为实际的神经网络）
        fusion_scores = self._simple_neural_combination(features)
        
        # 生成融合结果
        fusion_results = []
        all_docs = self._collect_all_documents(retrieval_results)
        
        for doc_id, doc_info in all_docs.items():
            if doc_id in fusion_scores:
                fusion_result = FusionResult(
                    doc_id=doc_id,
                    score=fusion_scores[doc_id],
                    doc=doc_info['doc'],
                    component_scores=doc_info['scores'],
                    fusion_method="neural_fusion"
                )
                fusion_results.append(fusion_result)
        
        fusion_results.sort(key=lambda x: x.score, reverse=True)
        return fusion_results
    
    def cascade_fusion(self, retrieval_results: Dict[str, List[RetrievalResult]],
                      cascade_threshold: float = 0.8) -> List[FusionResult]:
        """级联融合 - 逐步过滤和重排序"""
        
        if not retrieval_results:
            return []
        
        # 第一阶段：使用最快的检索器（通常是BM25）进行初筛
        primary_retriever = list(retrieval_results.keys())[0]
        primary_results = retrieval_results[primary_retriever]
        
        # 筛选高置信度结果
        high_confidence_docs = []
        low_confidence_docs = []
        
        for result in primary_results:
            if result.score >= cascade_threshold:
                high_confidence_docs.append(result)
            else:
                low_confidence_docs.append(result)
        
        # 第二阶段：对低置信度文档使用更精确的检索器重排序
        if len(retrieval_results) > 1 and low_confidence_docs:
            secondary_retriever = list(retrieval_results.keys())[1]
            secondary_results = retrieval_results[secondary_retriever]
            
            # 重排序低置信度文档
            low_confidence_reranked = self._rerank_with_secondary(
                low_confidence_docs, secondary_results
            )
        else:
            low_confidence_reranked = [
                FusionResult(
                    doc_id=r.doc_id,
                    score=r.score,
                    doc=r.doc,
                    component_scores={primary_retriever: r.score},
                    fusion_method="cascade"
                ) for r in low_confidence_docs
            ]
        
        # 合并结果
        final_results = []
        
        # 高置信度结果
        for result in high_confidence_docs:
            fusion_result = FusionResult(
                doc_id=result.doc_id,
                score=result.score + 0.1,  # 给高置信度结果小幅加分
                doc=result.doc,
                component_scores={primary_retriever: result.score},
                fusion_method="cascade"
            )
            final_results.append(fusion_result)
        
        # 重排序的低置信度结果
        final_results.extend(low_confidence_reranked)
        
        # 最终排序
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results
    
    def uncertainty_aware_fusion(self, retrieval_results: Dict[str, List[RetrievalResult]],
                                query: Any) -> List[FusionResult]:
        """不确定性感知融合"""
        
        # 计算每个检索器的不确定性
        uncertainties = self._calculate_retriever_uncertainty(retrieval_results)
        
        # 收集所有文档
        all_docs = self._collect_all_documents(retrieval_results)
        
        fusion_results = []
        for doc_id, doc_info in all_docs.items():
            
            # 计算不确定性加权分数
            weighted_score = 0.0
            total_weight = 0.0
            
            for retriever, score in doc_info['scores'].items():
                uncertainty = uncertainties.get(retriever, 0.5)
                weight = 1.0 / (1.0 + uncertainty)  # 不确定性越高，权重越低
                
                weighted_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = 0.0
            
            fusion_result = FusionResult(
                doc_id=doc_id,
                score=final_score,
                doc=doc_info['doc'],
                component_scores=doc_info['scores'],
                fusion_method="uncertainty_aware"
            )
            fusion_results.append(fusion_result)
        
        fusion_results.sort(key=lambda x: x.score, reverse=True)
        return fusion_results
    
    def _calculate_retriever_confidence(self, retrieval_results: Dict[str, List[RetrievalResult]]) -> Dict[str, float]:
        """计算检索器置信度"""
        confidences = {}
        
        for retriever_name, results in retrieval_results.items():
            if not results:
                confidences[retriever_name] = 0.0
                continue
            
            # 基于分数分布计算置信度
            scores = [r.score for r in results]
            
            # 方法1：基于分数方差（方差越小，置信度越高）
            if len(scores) > 1:
                score_var = np.var(scores)
                confidence_var = 1.0 / (1.0 + score_var)
            else:
                confidence_var = 0.5
            
            # 方法2：基于最高分数
            max_score = max(scores)
            confidence_max = min(max_score, 1.0)
            
            # 方法3：基于分数的集中度
            mean_score = np.mean(scores)
            top_scores = [s for s in scores if s >= mean_score]
            concentration = len(top_scores) / len(scores)
            
            # 综合置信度
            confidence = (confidence_var + confidence_max + concentration) / 3.0
            confidences[retriever_name] = confidence
        
        return confidences
    
    def _get_query_adaptive_weights(self, query: Any, 
                                  retrieval_results: Dict[str, List[RetrievalResult]]) -> Dict[str, float]:
        """基于查询特征获取自适应权重"""
        
        query_text = getattr(query, 'text', '')
        query_length = len(query_text.split())
        
        weights = {}
        
        # 基于查询长度的启发式规则
        if query_length <= 3:
            # 短查询，偏向BM25
            weights['bm25'] = 0.7
            weights['dense'] = 0.3
            weights['graph'] = 0.2
        elif query_length <= 10:
            # 中等查询，平衡权重
            weights['bm25'] = 0.5
            weights['dense'] = 0.5
            weights['graph'] = 0.4
        else:
            # 长查询，偏向Dense和Graph
            weights['bm25'] = 0.3
            weights['dense'] = 0.6
            weights['graph'] = 0.5
        
        # 基于查询中的实体数量调整图检索权重
        entity_count = self._count_entities_in_query(query_text)
        if entity_count > 2:
            weights['graph'] = min(weights.get('graph', 0.4) + 0.2, 1.0)
        
        # 归一化权重
        available_retrievers = list(retrieval_results.keys())
        final_weights = {}
        total = 0.0
        
        for retriever in available_retrievers:
            weight = weights.get(retriever, 0.33)
            final_weights[retriever] = weight
            total += weight
        
        if total > 0:
            final_weights = {k: v/total for k, v in final_weights.items()}
        
        return final_weights
    
    def _weighted_score_fusion(self, retrieval_results: Dict[str, List[RetrievalResult]], 
                              weights: Dict[str, float]) -> List[FusionResult]:
        """执行加权分数融合"""
        
        all_docs = self._collect_all_documents(retrieval_results)
        
        fusion_results = []
        for doc_id, doc_info in all_docs.items():
            
            weighted_score = 0.0
            for retriever, score in doc_info['scores'].items():
                weight = weights.get(retriever, 0.0)
                weighted_score += score * weight
            
            fusion_result = FusionResult(
                doc_id=doc_id,
                score=weighted_score,
                doc=doc_info['doc'],
                component_scores=doc_info['scores'],
                fusion_method="adaptive_weighted"
            )
            fusion_results.append(fusion_result)
        
        fusion_results.sort(key=lambda x: x.score, reverse=True)
        return fusion_results
    
    def _calculate_retriever_uncertainty(self, retrieval_results: Dict[str, List[RetrievalResult]]) -> Dict[str, float]:
        """计算检索器不确定性"""
        uncertainties = {}
        
        for retriever_name, results in retrieval_results.items():
            if not results:
                uncertainties[retriever_name] = 1.0  # 最大不确定性
                continue
            
            scores = [r.score for r in results]
            
            # 基于分数分布的不确定性
            if len(scores) > 1:
                # 使用熵来衡量不确定性
                # 标准化分数
                min_score, max_score = min(scores), max(scores)
                if max_score > min_score:
                    norm_scores = [(s - min_score) / (max_score - min_score) for s in scores]
                else:
                    norm_scores = [0.5] * len(scores)
                
                # 计算熵
                bins = np.histogram(norm_scores, bins=10, range=(0, 1))[0]
                probs = bins / np.sum(bins) if np.sum(bins) > 0 else np.ones(10) / 10
                probs = probs[probs > 0]  # 避免log(0)
                
                entropy = -np.sum(probs * np.log2(probs))
                uncertainty = entropy / np.log2(len(probs))  # 归一化
            else:
                uncertainty = 0.5
            
            uncertainties[retriever_name] = uncertainty
        
        return uncertainties
    
    def _collect_all_documents(self, retrieval_results: Dict[str, List[RetrievalResult]]) -> Dict[str, Dict]:
        """收集所有文档信息"""
        all_docs = {}
        
        for retriever_name, results in retrieval_results.items():
            for result in results:
                doc_id = result.doc_id
                if doc_id not in all_docs:
                    all_docs[doc_id] = {
                        'doc': result.doc,
                        'scores': {},
                        'ranks': {}
                    }
                
                all_docs[doc_id]['scores'][retriever_name] = result.score
                all_docs[doc_id]['ranks'][retriever_name] = len(all_docs[doc_id]['scores']) - 1
        
        return all_docs
    
    def _count_entities_in_query(self, query_text: str) -> int:
        """简单的实体计数（可以改进为使用NER）"""
        # 简化版本：计算大写字母开头的词
        words = query_text.split()
        entity_count = sum(1 for word in words if word[0].isupper() if word)
        return entity_count
    
    def _extract_fusion_features(self, retrieval_results: Dict[str, List[RetrievalResult]], 
                                query: Any) -> Dict[str, np.ndarray]:
        """提取融合特征"""
        features = {}
        
        # 为每个文档提取特征
        all_docs = self._collect_all_documents(retrieval_results)
        
        for doc_id, doc_info in all_docs.items():
            feature_vector = []
            
            # 特征1-3：各检索器的分数
            for retriever in ['bm25', 'dense', 'graph']:
                score = doc_info['scores'].get(retriever, 0.0)
                feature_vector.append(score)
            
            # 特征4：分数方差
            scores = list(doc_info['scores'].values())
            score_var = np.var(scores) if len(scores) > 1 else 0.0
            feature_vector.append(score_var)
            
            # 特征5：最大最小分数差
            score_range = max(scores) - min(scores) if scores else 0.0
            feature_vector.append(score_range)
            
            features[doc_id] = np.array(feature_vector)
        
        return features
    
    def _simple_neural_combination(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """简化的神经网络组合"""
        
        # 简化的权重矩阵（实际应用中应该通过训练获得）
        weights = np.array([0.4, 0.5, 0.3, -0.2, 0.1])  # 对应5个特征
        bias = 0.1
        
        fusion_scores = {}
        
        for doc_id, feature_vector in features.items():
            # 简单的线性组合 + sigmoid激活
            linear_output = np.dot(feature_vector, weights) + bias
            activated_output = 1 / (1 + np.exp(-linear_output))  # sigmoid
            
            fusion_scores[doc_id] = float(activated_output)
        
        return fusion_scores
    
    def _rerank_with_secondary(self, primary_docs: List[RetrievalResult], 
                              secondary_results: List[RetrievalResult]) -> List[FusionResult]:
        """使用次级检索器重排序"""
        
        # 创建次级结果的分数映射
        secondary_scores = {r.doc_id: r.score for r in secondary_results}
        
        reranked_results = []
        for doc in primary_docs:
            secondary_score = secondary_scores.get(doc.doc_id, 0.0)
            
            # 组合分数：主要检索器分数 + 次级检索器分数
            combined_score = 0.6 * doc.score + 0.4 * secondary_score
            
            fusion_result = FusionResult(
                doc_id=doc.doc_id,
                score=combined_score,
                doc=doc.doc,
                component_scores={
                    'primary': doc.score,
                    'secondary': secondary_score
                },
                fusion_method="cascade_rerank"
            )
            reranked_results.append(fusion_result)
        
        return reranked_results