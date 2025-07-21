#!/usr/bin/env python3
"""
增强版自适应路由器
实现多种融合策略的智能选择和性能反馈学习
"""

import random
import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from pathlib import Path

from ..utils.interfaces import Query, Document, RetrievalResult, FusionResult
from ..analysis.enhanced_query_analyzer import EnhancedQueryFeatures


class EnhancedAdaptiveRouter:
    """增强版自适应路由器
    
    实现多种融合策略的智能选择和性能反馈学习
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 定义融合策略 - 只保留表现良好的策略
        self.strategies = {
            # RRF策略，不同k值
            'rrf_standard': {'type': 'rrf', 'params': {'k': 60}},
            'rrf_conservative': {'type': 'rrf', 'params': {'k': 100}},
            
            # 线性加权策略 - 移除表现差的vector_dominant
            'linear_bm25_dominant': {'type': 'linear', 'params': {'weights': [0.7, 0.3]}},
            'linear_equal': {'type': 'linear', 'params': {'weights': [0.5, 0.5]}},
        }
        
        # 性能跟踪
        self.performance_history = defaultdict(lambda: defaultdict(list))
        self.dataset_preferences = defaultdict(dict)
        
        # 探索率（epsilon-greedy策略的epsilon）- 降低初始探索率
        self.exploration_rate = self.config.get('initial_exploration_rate', 0.1)
        self.min_exploration_rate = self.config.get('min_exploration_rate', 0.05)
        self.exploration_decay = self.config.get('exploration_decay', 0.99)
        
        # 加载历史性能数据（如果存在）
        self._load_performance_data()
    
    def _load_performance_data(self):
        """加载历史性能数据"""
        data_dir = Path(self.config.get('data_dir', 'data/adaptive_router'))
        data_file = data_dir / 'performance_history.json'
        
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                # 转换回原始数据结构
                for dataset, dataset_data in data.items():
                    for key, metrics_list in dataset_data.items():
                        # 键是元组，需要从字符串转回
                        query_type, strategy = eval(key)
                        self.performance_history[dataset][(query_type, strategy)] = metrics_list
                
                # 加载数据集偏好
                pref_file = data_dir / 'dataset_preferences.json'
                if pref_file.exists():
                    with open(pref_file, 'r') as f:
                        self.dataset_preferences = defaultdict(dict, json.load(f))
                
                print(f"已加载历史性能数据: {len(data)} 个数据集")
            except Exception as e:
                print(f"加载历史性能数据失败: {e}")
    
    def _save_performance_data(self):
        """保存性能数据"""
        data_dir = Path(self.config.get('data_dir', 'data/adaptive_router'))
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # 转换数据结构以便JSON序列化
        serializable_data = {}
        for dataset, dataset_data in self.performance_history.items():
            serializable_data[dataset] = {}
            for key, metrics_list in dataset_data.items():
                # 将元组键转换为字符串
                serializable_data[dataset][str(key)] = metrics_list
        
        # 保存性能历史
        with open(data_dir / 'performance_history.json', 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        # 保存数据集偏好
        with open(data_dir / 'dataset_preferences.json', 'w') as f:
            json.dump(dict(self.dataset_preferences), f, indent=2)
    
    def select_strategy(self, query_features: EnhancedQueryFeatures, dataset_name: str) -> Tuple[str, Dict[str, Any], bool]:
        """选择融合策略
        
        Args:
            query_features: 查询特征
            dataset_name: 数据集名称
            
        Returns:
            策略名称, 策略信息, 是否为探索
        """
        query_type = query_features.final_type
        
        # 检查是否应该探索
        if random.random() < self.exploration_rate:
            # 探索：尝试随机策略
            strategy_name = random.choice(list(self.strategies.keys()))
            is_exploration = True
        else:
            # 利用：使用已知最佳策略
            if dataset_name in self.dataset_preferences and query_type in self.dataset_preferences[dataset_name]:
                strategy_name = self.dataset_preferences[dataset_name][query_type]
            else:
                # 如果没有历史记录，基于查询类型和特征的默认策略
                strategy_name = self._get_default_strategy(query_features, dataset_name)
            is_exploration = False
        
        return strategy_name, self.strategies[strategy_name], is_exploration
    
    def _get_default_strategy(self, query_features: EnhancedQueryFeatures, dataset_name: str = None) -> str:
        """获取基于数据集和查询特征的默认策略"""
        
        # 基于数据集的默认策略（来自基线实验结果）
        dataset_defaults = {
            'fiqa': 'rrf_standard',        # RRF表现最好 (MRR=0.278)
            'quora': 'rrf_standard',       # RRF表现最好 (MRR=0.702)
            'scidocs': 'linear_equal',     # LinearEqual表现最好 (MRR=0.323)
            'nfcorpus': 'rrf_standard',    # RRF表现最好 (MRR=0.649)
            'scifact': 'linear_equal',     # LinearEqual表现最好 (MRR=0.736)
            'arguana': 'linear_equal',     # LinearEqual表现最好 (MRR=0.265)
        }
        
        # 优先使用数据集特定的默认策略
        if dataset_name and dataset_name in dataset_defaults:
            return dataset_defaults[dataset_name]
        
        # 如果数据集未知，则使用保守的RRF策略
        return 'rrf_standard'
    
    def apply_strategy(self, strategy_name: str, strategy_info: Dict[str, Any], 
                      bm25_results: List[RetrievalResult], 
                      vector_results: List[RetrievalResult], 
                      top_k: int = 10) -> List[FusionResult]:
        """应用融合策略"""
        strategy_type = strategy_info['type']
        params = strategy_info['params']
        
        if strategy_type == 'rrf':
            return self._apply_rrf(bm25_results, vector_results, params['k'], top_k)
        elif strategy_type == 'linear':
            return self._apply_linear_weighted(bm25_results, vector_results, params['weights'], top_k)
        elif strategy_type == 'max_score':
            return self._apply_max_score(bm25_results, vector_results, top_k)
        else:
            # 默认使用RRF
            return self._apply_rrf(bm25_results, vector_results, 60, top_k)
    
    def _apply_rrf(self, bm25_results: List[RetrievalResult], 
                  vector_results: List[RetrievalResult], 
                  k: int, top_k: int) -> List[FusionResult]:
        """应用RRF融合"""
        all_docs = {}
        doc_ranks = defaultdict(dict)
        
        # BM25结果排名
        for rank, result in enumerate(bm25_results):
            doc_id = result.doc_id
            all_docs[doc_id] = result.document
            doc_ranks[doc_id]['BM25'] = rank + 1
        
        # 向量结果排名
        for rank, result in enumerate(vector_results):
            doc_id = result.doc_id
            all_docs[doc_id] = result.document
            doc_ranks[doc_id]['EfficientVector'] = rank + 1
        
        # 计算RRF分数
        fusion_results = []
        for doc_id, ranks in doc_ranks.items():
            rrf_score = sum(1.0 / (k + rank) for rank in ranks.values())
            
            # 收集原始分数
            individual_scores = {}
            for result in bm25_results:
                if result.doc_id == doc_id:
                    individual_scores['BM25'] = result.score
                    break
            for result in vector_results:
                if result.doc_id == doc_id:
                    individual_scores['EfficientVector'] = result.score
                    break
            
            fusion_result = FusionResult(
                doc_id=doc_id,
                final_score=rrf_score,
                document=all_docs[doc_id],
                individual_scores=individual_scores
            )
            fusion_results.append(fusion_result)
        
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        return fusion_results[:top_k]
    
    def _apply_linear_weighted(self, bm25_results: List[RetrievalResult], 
                              vector_results: List[RetrievalResult], 
                              weights: List[float], top_k: int) -> List[FusionResult]:
        """应用线性加权融合"""
        all_docs = {}
        doc_scores = defaultdict(dict)
        
        # 归一化BM25分数
        if bm25_results:
            bm25_scores = [r.score for r in bm25_results]
            bm25_max = max(bm25_scores) if bm25_scores else 1.0
            bm25_min = min(bm25_scores) if bm25_scores else 0.0
            bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0
            
            for result in bm25_results:
                doc_id = result.doc_id
                all_docs[doc_id] = result.document
                normalized_score = (result.score - bm25_min) / bm25_range if bm25_range > 0 else 0.5
                doc_scores[doc_id]['BM25'] = {
                    'normalized': normalized_score,
                    'original': result.score,
                    'weight': weights[0]
                }
        
        # 归一化向量分数
        if vector_results:
            vector_scores = [r.score for r in vector_results]
            vector_max = max(vector_scores) if vector_scores else 1.0
            vector_min = min(vector_scores) if vector_scores else 0.0
            vector_range = vector_max - vector_min if vector_max > vector_min else 1.0
            
            for result in vector_results:
                doc_id = result.doc_id
                all_docs[doc_id] = result.document
                normalized_score = (result.score - vector_min) / vector_range if vector_range > 0 else 0.5
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {}
                doc_scores[doc_id]['EfficientVector'] = {
                    'normalized': normalized_score,
                    'original': result.score,
                    'weight': weights[1]
                }
        
        # 计算加权分数
        fusion_results = []
        for doc_id, scores_dict in doc_scores.items():
            weighted_score = sum(
                info['normalized'] * info['weight'] 
                for info in scores_dict.values()
            )
            
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
        
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        return fusion_results[:top_k]
    
    def _apply_max_score(self, bm25_results: List[RetrievalResult], 
                        vector_results: List[RetrievalResult], 
                        top_k: int) -> List[FusionResult]:
        """应用最大分数融合"""
        all_docs = {}
        doc_scores = defaultdict(dict)
        
        # 归一化BM25分数
        if bm25_results:
            bm25_scores = [r.score for r in bm25_results]
            bm25_max = max(bm25_scores) if bm25_scores else 1.0
            bm25_min = min(bm25_scores) if bm25_scores else 0.0
            bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0
            
            for result in bm25_results:
                doc_id = result.doc_id
                all_docs[doc_id] = result.document
                normalized_score = (result.score - bm25_min) / bm25_range if bm25_range > 0 else 0.5
                doc_scores[doc_id]['BM25'] = normalized_score
        
        # 归一化向量分数
        if vector_results:
            vector_scores = [r.score for r in vector_results]
            vector_max = max(vector_scores) if vector_scores else 1.0
            vector_min = min(vector_scores) if vector_scores else 0.0
            vector_range = vector_max - vector_min if vector_max > vector_min else 1.0
            
            for result in vector_results:
                doc_id = result.doc_id
                all_docs[doc_id] = result.document
                normalized_score = (result.score - vector_min) / vector_range if vector_range > 0 else 0.5
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {}
                doc_scores[doc_id]['EfficientVector'] = normalized_score
        
        # 计算最大分数
        fusion_results = []
        for doc_id, scores_dict in doc_scores.items():
            max_score = max(scores_dict.values())
            
            # 收集原始分数
            individual_scores = {}
            for result in bm25_results:
                if result.doc_id == doc_id:
                    individual_scores['BM25'] = result.score
                    break
            for result in vector_results:
                if result.doc_id == doc_id:
                    individual_scores['EfficientVector'] = result.score
                    break
            
            fusion_result = FusionResult(
                doc_id=doc_id,
                final_score=max_score,
                document=all_docs[doc_id],
                individual_scores=individual_scores
            )
            fusion_results.append(fusion_result)
        
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        return fusion_results[:top_k]
    
    def update_performance(self, query_features: EnhancedQueryFeatures, 
                          dataset_name: str, strategy_name: str, 
                          metrics: Dict[str, float]) -> None:
        """更新性能历史
        
        Args:
            query_features: 查询特征
            dataset_name: 数据集名称
            strategy_name: 使用的策略名称
            metrics: 检索指标 (mrr, ndcg等)
        """
        query_type = query_features.final_type
        
        # 存储性能指标
        self.performance_history[dataset_name][(query_type, strategy_name)].append(metrics)
        
        # 如果有足够的数据，更新数据集偏好
        if len(self.performance_history[dataset_name][(query_type, strategy_name)]) >= 5:
            self._update_dataset_preferences(dataset_name, query_type)
        
        # 衰减探索率
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)
        
        # 定期保存性能数据
        if random.random() < 0.1:  # 10%的概率保存数据
            self._save_performance_data()
    
    def _update_dataset_preferences(self, dataset_name: str, query_type: str) -> None:
        """更新数据集偏好"""
        best_strategy = None
        best_performance = -1
        
        # 对于该数据集和查询类型的所有策略
        for (qt, strategy), metrics_list in self.performance_history[dataset_name].items():
            if qt != query_type:
                continue
                
            # 使用MRR作为主要指标
            avg_mrr = sum(m.get('mrr', 0) for m in metrics_list) / len(metrics_list)
            
            if avg_mrr > best_performance:
                best_performance = avg_mrr
                best_strategy = strategy
        
        if best_strategy:
            self.dataset_preferences[dataset_name][query_type] = best_strategy


def create_enhanced_adaptive_router(config: Dict[str, Any] = None) -> EnhancedAdaptiveRouter:
    """创建增强自适应路由器的工厂函数"""
    return EnhancedAdaptiveRouter(config)