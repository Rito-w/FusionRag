"""
反馈学习模块
基于用户反馈和查询结果质量进行实时学习和优化
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np

from ..utils.interfaces import Query, Document, RetrievalResult, FusionResult


@dataclass
class UserFeedback:
    """用户反馈数据结构"""
    query_id: str
    query_text: str
    doc_id: str
    relevance_score: float  # 0-1之间，1表示完全相关
    feedback_type: str  # 'click', 'like', 'dislike', 'rating'
    timestamp: float
    user_id: Optional[str] = None


@dataclass
class QueryPerformance:
    """查询性能记录"""
    query_id: str
    query_text: str
    retriever_scores: Dict[str, float]  # 各检索器的性能分数
    fusion_method: str
    final_metrics: Dict[str, float]  # 最终评测指标
    timestamp: float


class FeedbackCollector:
    """反馈收集器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feedback_buffer = deque(maxlen=config.get('buffer_size', 1000))
        self.feedback_file = config.get('feedback_file', 'checkpoints/feedback/user_feedback.jsonl')
        
        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
    
    def collect_feedback(self, feedback: UserFeedback) -> None:
        """收集用户反馈"""
        self.feedback_buffer.append(feedback)
        
        # 持久化存储
        with open(self.feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(feedback), ensure_ascii=False) + '\n')
    
    def get_recent_feedback(self, hours: int = 24) -> List[UserFeedback]:
        """获取最近的反馈"""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        return [
            feedback for feedback in self.feedback_buffer
            if feedback.timestamp >= cutoff_time
        ]
    
    def get_query_feedback(self, query_id: str) -> List[UserFeedback]:
        """获取特定查询的反馈"""
        return [
            feedback for feedback in self.feedback_buffer
            if feedback.query_id == query_id
        ]


class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_buffer = deque(maxlen=config.get('buffer_size', 1000))
        self.performance_file = config.get('performance_file', 'checkpoints/feedback/performance.jsonl')
        
        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(self.performance_file), exist_ok=True)
    
    def record_performance(self, performance: QueryPerformance) -> None:
        """记录查询性能"""
        self.performance_buffer.append(performance)
        
        # 持久化存储
        with open(self.performance_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(performance), ensure_ascii=False) + '\n')
    
    def get_retriever_performance(self, retriever_name: str, hours: int = 24) -> List[float]:
        """获取检索器性能历史"""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        scores = []
        for perf in self.performance_buffer:
            if perf.timestamp >= cutoff_time and retriever_name in perf.retriever_scores:
                scores.append(perf.retriever_scores[retriever_name])
        
        return scores


class AdaptiveLearner:
    """自适应学习器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feedback_collector = FeedbackCollector(config.get('feedback', {}))
        self.performance_tracker = PerformanceTracker(config.get('performance', {}))
        
        # 学习参数
        self.learning_rate = config.get('learning_rate', 0.01)
        self.min_feedback_count = config.get('min_feedback_count', 10)
        self.adaptation_threshold = config.get('adaptation_threshold', 0.1)
        
        # 当前权重
        self.current_weights = config.get('initial_weights', {
            'bm25': 0.4,
            'dense': 0.4,
            'graph': 0.2
        })
        
        # 性能历史
        self.performance_history = defaultdict(list)
    
    def update_weights_from_feedback(self, feedback_list: List[UserFeedback]) -> Dict[str, float]:
        """基于用户反馈更新权重"""
        if len(feedback_list) < self.min_feedback_count:
            return self.current_weights
        
        # 计算各检索器的反馈分数
        retriever_feedback = defaultdict(list)
        
        for feedback in feedback_list:
            # 这里需要根据实际情况映射反馈到检索器
            # 简化处理：假设我们能从查询历史中获取检索器信息
            query_performance = self._get_query_performance(feedback.query_id)
            if query_performance:
                for retriever, score in query_performance.retriever_scores.items():
                    # 权重反馈分数和检索器分数
                    weighted_score = feedback.relevance_score * score
                    retriever_feedback[retriever].append(weighted_score)
        
        # 更新权重
        new_weights = self.current_weights.copy()
        total_adjustment = 0
        
        for retriever, scores in retriever_feedback.items():
            if retriever in new_weights and scores:
                avg_score = np.mean(scores)
                current_weight = new_weights[retriever]
                
                # 计算调整量
                adjustment = self.learning_rate * (avg_score - 0.5)  # 0.5为中性分数
                new_weight = max(0.1, min(0.8, current_weight + adjustment))  # 限制权重范围
                
                adjustment_amount = new_weight - current_weight
                new_weights[retriever] = new_weight
                total_adjustment += adjustment_amount
        
        # 归一化权重
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v / total_weight for k, v in new_weights.items()}
        
        # 检查是否需要更新
        weight_change = sum(abs(new_weights[k] - self.current_weights[k]) 
                          for k in new_weights.keys())
        
        if weight_change > self.adaptation_threshold:
            print(f"权重更新: {self.current_weights} -> {new_weights}")
            self.current_weights = new_weights
        
        return self.current_weights
    
    def update_weights_from_performance(self, performance_list: List[QueryPerformance]) -> Dict[str, float]:
        """基于性能指标更新权重"""
        if len(performance_list) < self.min_feedback_count:
            return self.current_weights
        
        # 计算各检索器的平均性能
        retriever_performance = defaultdict(list)
        
        for perf in performance_list:
            for retriever, score in perf.retriever_scores.items():
                retriever_performance[retriever].append(score)
        
        # 更新权重
        new_weights = {}
        total_performance = 0
        
        for retriever, scores in retriever_performance.items():
            avg_performance = np.mean(scores)
            new_weights[retriever] = avg_performance
            total_performance += avg_performance
        
        # 归一化
        if total_performance > 0:
            new_weights = {k: v / total_performance for k, v in new_weights.items()}
        
        # 平滑更新（避免剧烈变化）
        smoothing_factor = 0.8
        for retriever in new_weights:
            if retriever in self.current_weights:
                new_weights[retriever] = (
                    smoothing_factor * self.current_weights[retriever] +
                    (1 - smoothing_factor) * new_weights[retriever]
                )
        
        self.current_weights = new_weights
        return self.current_weights
    
    def _get_query_performance(self, query_id: str) -> Optional[QueryPerformance]:
        """获取查询性能记录"""
        for perf in self.performance_tracker.performance_buffer:
            if perf.query_id == query_id:
                return perf
        return None
    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.current_weights.copy()
    
    def should_adapt(self) -> bool:
        """判断是否应该进行自适应调整"""
        recent_feedback = self.feedback_collector.get_recent_feedback(hours=1)
        return len(recent_feedback) >= self.min_feedback_count


class FeedbackIntegration:
    """反馈集成模块"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.learner = AdaptiveLearner(config)
        self.enabled = config.get('enabled', True)
    
    def process_user_feedback(self, query: Query, results: List[FusionResult], 
                            user_feedback: Dict[str, Any]) -> None:
        """处理用户反馈"""
        if not self.enabled:
            return
        
        # 转换用户反馈格式
        for doc_id, relevance in user_feedback.items():
            feedback = UserFeedback(
                query_id=query.query_id,
                query_text=query.text,
                doc_id=doc_id,
                relevance_score=relevance,
                feedback_type='rating',
                timestamp=time.time()
            )
            self.learner.feedback_collector.collect_feedback(feedback)
    
    def record_query_performance(self, query: Query, retriever_results: Dict[str, List[RetrievalResult]],
                               fusion_method: str, metrics: Dict[str, float]) -> None:
        """记录查询性能"""
        if not self.enabled:
            return
        
        # 计算各检索器分数
        retriever_scores = {}
        for retriever_name, results in retriever_results.items():
            if results:
                avg_score = np.mean([r.score for r in results])
                retriever_scores[retriever_name] = avg_score
        
        performance = QueryPerformance(
            query_id=query.query_id,
            query_text=query.text,
            retriever_scores=retriever_scores,
            fusion_method=fusion_method,
            final_metrics=metrics,
            timestamp=time.time()
        )
        
        self.learner.performance_tracker.record_performance(performance)
    
    def get_adaptive_weights(self) -> Dict[str, float]:
        """获取自适应权重"""
        if not self.enabled or not self.learner.should_adapt():
            return self.learner.get_current_weights()
        
        # 基于反馈更新权重
        recent_feedback = self.learner.feedback_collector.get_recent_feedback(hours=24)
        if recent_feedback:
            self.learner.update_weights_from_feedback(recent_feedback)
        
        # 基于性能更新权重
        recent_performance = list(self.learner.performance_tracker.performance_buffer)[-100:]
        if recent_performance:
            self.learner.update_weights_from_performance(recent_performance)
        
        return self.learner.get_current_weights()
