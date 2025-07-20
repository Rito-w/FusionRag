#!/usr/bin/env python3
"""
简化版自适应路由器
专门为论文实验设计，基于查询特征智能选择检索器和融合策略
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..utils.interfaces import Query, QueryType
from ..analysis.simple_query_analyzer import SimpleQueryFeatures


@dataclass
class RoutingDecision:
    """路由决策"""
    query_id: str
    query_text: str
    query_type: str
    selected_retrievers: List[str]
    fusion_method: str
    fusion_weights: Dict[str, float]
    confidence: float
    reasoning: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.__dict__.copy()


class SimpleAdaptiveRouter:
    """简化版自适应路由器
    
    基于查询特征智能选择检索器组合和融合策略
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 可用检索器
        self.available_retrievers = self.config.get('available_retrievers', [
            'BM25', 'EfficientVector', 'SemanticBM25'
        ])
        
        # 可用融合方法
        self.available_fusion_methods = self.config.get('available_fusion_methods', [
            'RRF', 'LinearEqual', 'LinearOptimized'
        ])
        
        # 路由策略配置
        self.routing_strategy = self.config.get('routing_strategy', 'rule_based')
        self.enable_performance_feedback = self.config.get('enable_performance_feedback', True)
        
        # 性能历史记录
        self.performance_history = {}
        self.routing_history = []
        
        # 数据存储路径
        self.data_dir = Path(self.config.get('data_dir', 'data/adaptive_router'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载历史数据
        self._load_history()
    
    def route(self, query_features: SimpleQueryFeatures) -> RoutingDecision:
        """根据查询特征进行路由决策"""
        
        # 基于规则的路由策略
        if self.routing_strategy == 'rule_based':
            return self._rule_based_routing(query_features)
        else:
            # 默认使用规则策略
            return self._rule_based_routing(query_features)
    
    def _rule_based_routing(self, features: SimpleQueryFeatures) -> RoutingDecision:
        """基于规则的路由策略"""
        query_type = features.query_type
        domain = features.domain_hint
        complexity = features.complexity_level
        is_question = features.is_question
        
        # 初始化决策参数
        selected_retrievers = []
        fusion_method = 'RRF'
        fusion_weights = {}
        confidence = 0.7
        reasoning = ""
        
        # 智能路由策略：基于查询特征选择最优策略
        if query_type == QueryType.SEMANTIC and domain == 'technical':
            # 技术语义查询：向量检索占主导
            selected_retrievers = ['EfficientVector', 'BM25']
            fusion_weights = {'EfficientVector': 0.9, 'BM25': 0.1}
            fusion_method = 'RRF'
            confidence = 0.95
            reasoning = "技术语义查询强化向量检索"
            
        elif query_type == QueryType.SEMANTIC:
            # 一般语义查询：向量检索为主
            selected_retrievers = ['EfficientVector', 'BM25']
            fusion_weights = {'EfficientVector': 0.8, 'BM25': 0.2}
            fusion_method = 'RRF'
            confidence = 0.9
            reasoning = "语义查询向量检索为主"
            
        elif query_type == QueryType.ENTITY and domain == 'financial':
            # 金融实体查询：平衡策略
            selected_retrievers = ['BM25', 'EfficientVector']
            fusion_weights = {'BM25': 0.6, 'EfficientVector': 0.4}
            fusion_method = 'RRF'
            confidence = 0.85
            reasoning = "金融实体查询平衡策略"
            
        elif query_type == QueryType.KEYWORD and domain == 'technical':
            # 技术关键词：向量检索优势
            selected_retrievers = ['EfficientVector', 'BM25']
            fusion_weights = {'EfficientVector': 0.75, 'BM25': 0.25}
            fusion_method = 'RRF'
            confidence = 0.8
            reasoning = "技术关键词向量检索优势"
            
        else:
            # 默认策略：基于基线最佳表现使用RRF等权重
            selected_retrievers = ['BM25', 'EfficientVector']
            fusion_weights = {'BM25': 0.5, 'EfficientVector': 0.5}
            fusion_method = 'RRF'
            confidence = 0.75
            reasoning = "默认RRF等权重策略"
        
        # 根据领域调整策略
        if domain == 'technical':
            # 技术领域：增加向量检索权重
            if 'EfficientVector' in fusion_weights:
                fusion_weights['EfficientVector'] = min(0.9, fusion_weights['EfficientVector'] + 0.1)
                fusion_weights['BM25'] = 1.0 - fusion_weights['EfficientVector']
            reasoning += " + 技术领域增强向量检索"
            
        elif domain == 'medical':
            # 医疗领域：平衡两种检索器
            if len(fusion_weights) == 2:
                fusion_weights = {k: 0.5 for k in fusion_weights.keys()}
            reasoning += " + 医疗领域平衡策略"
            
        elif domain == 'financial':
            # 金融领域：增加BM25权重（实体和数字敏感）
            if 'BM25' in fusion_weights:
                fusion_weights['BM25'] = min(0.8, fusion_weights['BM25'] + 0.1)
                if 'EfficientVector' in fusion_weights:
                    fusion_weights['EfficientVector'] = 1.0 - fusion_weights['BM25']
            reasoning += " + 金融领域增强BM25检索"
        
        # 根据复杂度调整融合方法（优先使用RRF，因为基线表现好）
        if complexity == 'complex':
            fusion_method = 'RRF'  # 复杂查询使用RRF
            reasoning += " + 复杂查询使用RRF融合"
        elif complexity == 'simple':
            fusion_method = 'RRF'  # 简单查询也使用RRF（基线表现好）
            reasoning += " + 简单查询使用RRF融合"
        else:
            fusion_method = 'RRF'  # 默认使用RRF
        
        # 确保权重归一化
        if fusion_weights:
            total_weight = sum(fusion_weights.values())
            fusion_weights = {k: v/total_weight for k, v in fusion_weights.items()}
        
        # 创建路由决策
        decision = RoutingDecision(
            query_id=features.query_id,
            query_text=features.query_text,
            query_type=features.query_type.value,
            selected_retrievers=selected_retrievers,
            fusion_method=fusion_method,
            fusion_weights=fusion_weights,
            confidence=confidence,
            reasoning=reasoning
        )
        
        # 记录决策历史
        self.routing_history.append(decision)
        
        return decision
    
    def update_performance(self, query_id: str, retriever_performance: Dict[str, float]) -> None:
        """更新检索器性能记录"""
        if self.enable_performance_feedback:
            self.performance_history[query_id] = {
                'performance': retriever_performance,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        if not self.performance_history:
            return {}
        
        # 计算各检索器的平均性能
        retriever_stats = {}
        for query_id, data in self.performance_history.items():
            for retriever, metrics in data['performance'].items():
                if retriever not in retriever_stats:
                    retriever_stats[retriever] = []
                retriever_stats[retriever].append(metrics.get('mrr', 0))
        
        # 计算平均值
        avg_performance = {}
        for retriever, scores in retriever_stats.items():
            avg_performance[retriever] = sum(scores) / len(scores) if scores else 0
        
        return {
            'average_performance': avg_performance,
            'total_queries': len(self.performance_history),
            'routing_decisions': len(self.routing_history)
        }
    
    def _load_history(self) -> None:
        """加载历史数据"""
        # 加载性能历史
        perf_file = self.data_dir / 'performance_history.json'
        if perf_file.exists():
            try:
                with open(perf_file, 'r') as f:
                    self.performance_history = json.load(f)
            except Exception as e:
                print(f"加载性能历史失败: {e}")
        
        # 加载路由历史
        routing_file = self.data_dir / 'routing_history.json'
        if routing_file.exists():
            try:
                with open(routing_file, 'r') as f:
                    history_data = json.load(f)
                    self.routing_history = [
                        RoutingDecision(**data) for data in history_data
                    ]
            except Exception as e:
                print(f"加载路由历史失败: {e}")
    
    def save_history(self) -> None:
        """保存历史数据"""
        # 保存性能历史
        perf_file = self.data_dir / 'performance_history.json'
        try:
            with open(perf_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            print(f"保存性能历史失败: {e}")
        
        # 保存路由历史
        routing_file = self.data_dir / 'routing_history.json'
        try:
            history_data = [decision.to_dict() for decision in self.routing_history]
            with open(routing_file, 'w') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            print(f"保存路由历史失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取路由器统计信息"""
        stats = {
            'routing_strategy': self.routing_strategy,
            'available_retrievers': self.available_retrievers,
            'available_fusion_methods': self.available_fusion_methods,
            'total_routing_decisions': len(self.routing_history),
            'performance_records': len(self.performance_history)
        }
        
        # 统计各查询类型的路由决策
        if self.routing_history:
            query_type_stats = {}
            fusion_method_stats = {}
            
            for decision in self.routing_history:
                # 查询类型统计
                qtype = decision.query_type
                if qtype not in query_type_stats:
                    query_type_stats[qtype] = 0
                query_type_stats[qtype] += 1
                
                # 融合方法统计
                fmethod = decision.fusion_method
                if fmethod not in fusion_method_stats:
                    fusion_method_stats[fmethod] = 0
                fusion_method_stats[fmethod] += 1
            
            stats['query_type_distribution'] = query_type_stats
            stats['fusion_method_distribution'] = fusion_method_stats
            stats['average_confidence'] = sum(d.confidence for d in self.routing_history) / len(self.routing_history)
        
        return stats


def create_simple_adaptive_router(config: Dict[str, Any] = None) -> SimpleAdaptiveRouter:
    """创建简化自适应路由器的工厂函数"""
    return SimpleAdaptiveRouter(config=config)