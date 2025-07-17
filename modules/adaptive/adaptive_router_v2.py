"""自适应路由器
根据查询特征动态选择最优的检索策略
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from ..utils.interfaces import Query, BaseRetriever
from ..utils.common import FileUtils
from ..analysis.query_feature_analyzer import QueryFeatures, QueryFeatureAnalyzer

try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    print("请安装scikit-learn: pip install scikit-learn")


@dataclass
class RoutingDecision:
    """路由决策类"""
    primary_retriever: str                    # 主检索器
    secondary_retrievers: List[str]           # 次级检索器
    fusion_method: str = "weighted"           # 融合方法
    fusion_weights: Dict[str, float] = None   # 融合权重
    confidence_score: float = 0.0             # 置信度
    reasoning: str = ""                       # 决策理由
    
    def __post_init__(self):
        if self.fusion_weights is None:
            self.fusion_weights = {}


@dataclass
class PerformanceRecord:
    """性能记录类"""
    query_id: str
    query_text: str
    query_features: QueryFeatures
    routing_decision: RoutingDecision
    retriever_performances: Dict[str, float]  # 各检索器的性能
    best_retriever: str
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'query_id': self.query_id,
            'query_text': self.query_text,
            'query_features': asdict(self.query_features),
            'routing_decision': asdict(self.routing_decision),
            'retriever_performances': self.retriever_performances,
            'best_retriever': self.best_retriever,
            'timestamp': self.timestamp
        }


class AdaptiveRouter:
    """自适应路由器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 可用的检索器
        self.available_retrievers = self.config.get('available_retrievers', [
            'efficient_vector', 'semantic_bm25'
        ])
        
        # 路由策略
        self.routing_strategy = self.config.get('routing_strategy', 'hybrid')  # 'rule_based', 'learning_based', 'hybrid'
        
        # 学习参数
        self.enable_learning = self.config.get('enable_learning', True)
        self.min_training_samples = self.config.get('min_training_samples', 50)
        self.learning_update_interval = self.config.get('learning_update_interval', 100)  # 查询数
        
        # 性能历史
        self.performance_history = deque(maxlen=self.config.get('history_size', 1000))
        self.query_count = 0
        
        # 查询特征分析器
        self.query_analyzer = QueryFeatureAnalyzer(self.config.get('analyzer_config', {}))
        
        # 学习模型
        self.routing_model = None
        self.model_accuracy = 0.0
        
        # 规则权重
        self.rule_weights = self.config.get('rule_weights', {
            'query_type': 0.3,
            'complexity': 0.2,
            'entity_count': 0.15,
            'token_count': 0.15,
            'is_question': 0.1,
            'domain_specificity': 0.1
        })
        
        # 默认权重
        self.default_weights = self.config.get('default_weights', {
            'efficient_vector': 0.6,
            'semantic_bm25': 0.4
        })
        
        # 缓存路径
        self.cache_dir = Path(self.config.get('cache_dir', 'checkpoints/router'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载历史数据
        self._load_performance_history()
        
        # 初始化学习模型
        if self.enable_learning:
            self._initialize_learning_model()
    
    def route(self, query: Query) -> RoutingDecision:
        """路由决策"""
        # 分析查询特征
        features = self.query_analyzer.analyze(query)
        
        # 根据策略进行路由
        if self.routing_strategy == 'rule_based':
            decision = self._rule_based_routing(features)
        elif self.routing_strategy == 'learning_based' and self.routing_model:
            decision = self._learning_based_routing(features)
        else:
            # 混合策略
            decision = self._hybrid_routing(features)
        
        # 更新查询计数
        self.query_count += 1
        
        # 定期更新学习模型
        if (self.enable_learning and 
            self.query_count % self.learning_update_interval == 0):
            self._update_learning_model()
        
        return decision
    
    def _rule_based_routing(self, features: QueryFeatures) -> RoutingDecision:
        """基于规则的路由"""
        # 计算各检索器的得分
        retriever_scores = {}
        
        for retriever in self.available_retrievers:
            score = self._calculate_retriever_score(retriever, features)
            retriever_scores[retriever] = score
        
        # 选择最佳检索器
        primary_retriever = max(retriever_scores, key=retriever_scores.get)
        
        # 确定次级检索器
        secondary_retrievers = [r for r in self.available_retrievers if r != primary_retriever]
        
        # 计算融合权重
        total_score = sum(retriever_scores.values())
        fusion_weights = {
            r: score / total_score for r, score in retriever_scores.items()
        } if total_score > 0 else self.default_weights
        
        # 生成决策理由
        reasoning = f"规则路由: {primary_retriever} (得分: {retriever_scores[primary_retriever]:.3f})"
        
        return RoutingDecision(
            primary_retriever=primary_retriever,
            secondary_retrievers=secondary_retrievers,
            fusion_method="weighted",
            fusion_weights=fusion_weights,
            confidence_score=retriever_scores[primary_retriever],
            reasoning=reasoning
        )
    
    def _learning_based_routing(self, features: QueryFeatures) -> RoutingDecision:
        """基于学习的路由"""
        if not self.routing_model:
            # 回退到规则路由
            return self._rule_based_routing(features)
        
        # 特征向量化
        feature_vector = self._features_to_vector(features)
        
        # 预测最佳检索器
        try:
            predicted_retriever = self.routing_model.predict([feature_vector])[0]
            prediction_proba = self.routing_model.predict_proba([feature_vector])[0]
            
            # 获取预测概率
            retriever_probs = dict(zip(self.routing_model.classes_, prediction_proba))
            
            # 选择主检索器
            primary_retriever = predicted_retriever
            
            # 次级检索器按概率排序
            secondary_retrievers = sorted(
                [r for r in self.available_retrievers if r != primary_retriever],
                key=lambda r: retriever_probs.get(r, 0),
                reverse=True
            )
            
            # 计算融合权重
            fusion_weights = {r: prob for r, prob in retriever_probs.items()}
            
            # 置信度为最高预测概率
            confidence_score = max(prediction_proba)
            
            reasoning = f"学习路由: {primary_retriever} (置信度: {confidence_score:.3f}, 模型准确率: {self.model_accuracy:.3f})"
            
            return RoutingDecision(
                primary_retriever=primary_retriever,
                secondary_retrievers=secondary_retrievers,
                fusion_method="weighted",
                fusion_weights=fusion_weights,
                confidence_score=confidence_score,
                reasoning=reasoning
            )
            
        except Exception as e:
            print(f"学习路由失败: {e}，回退到规则路由")
            return self._rule_based_routing(features)
    
    def _hybrid_routing(self, features: QueryFeatures) -> RoutingDecision:
        """混合路由策略"""
        # 获取规则路由结果
        rule_decision = self._rule_based_routing(features)
        
        # 如果学习模型可用且准确率足够高，结合学习结果
        if (self.routing_model and 
            self.model_accuracy > 0.6 and 
            len(self.performance_history) >= self.min_training_samples):
            
            learning_decision = self._learning_based_routing(features)
            
            # 混合决策
            if rule_decision.primary_retriever == learning_decision.primary_retriever:
                # 一致时增强置信度
                confidence_score = min(1.0, (rule_decision.confidence_score + learning_decision.confidence_score) / 2)
                
                # 混合权重
                fusion_weights = {}
                for retriever in self.available_retrievers:
                    rule_weight = rule_decision.fusion_weights.get(retriever, 0)
                    learning_weight = learning_decision.fusion_weights.get(retriever, 0)
                    fusion_weights[retriever] = (rule_weight + learning_weight) / 2
                
                reasoning = f"混合路由(一致): {rule_decision.primary_retriever} (置信度: {confidence_score:.3f})"
                
                return RoutingDecision(
                    primary_retriever=rule_decision.primary_retriever,
                    secondary_retrievers=rule_decision.secondary_retrievers,
                    fusion_method="weighted",
                    fusion_weights=fusion_weights,
                    confidence_score=confidence_score,
                    reasoning=reasoning
                )
            else:
                # 不一致时选择置信度更高的
                if rule_decision.confidence_score > learning_decision.confidence_score:
                    selected_decision = rule_decision
                    reason_suffix = "(选择规则路由)"
                else:
                    selected_decision = learning_decision
                    reason_suffix = "(选择学习路由)"
                
                selected_decision.reasoning = f"混合路由(分歧): {selected_decision.primary_retriever} {reason_suffix}"
                return selected_decision
        
        else:
            # 学习模型不可用，使用规则路由
            rule_decision.reasoning = f"混合路由(仅规则): {rule_decision.primary_retriever}"
            return rule_decision
    
    def _calculate_retriever_score(self, retriever: str, features: QueryFeatures) -> float:
        """计算检索器得分"""
        score = 0.0
        
        # 基于查询类型
        if retriever == 'efficient_vector':
            type_scores = {
                'conceptual': 0.9,
                'analytical': 0.8,
                'procedural': 0.7,
                'factual': 0.6,
                'general': 0.7
            }
            score += type_scores.get(features.query_type, 0.5) * self.rule_weights['query_type']
        
        elif retriever == 'semantic_bm25':
            type_scores = {
                'factual': 0.9,
                'general': 0.8,
                'procedural': 0.7,
                'conceptual': 0.6,
                'analytical': 0.5
            }
            score += type_scores.get(features.query_type, 0.5) * self.rule_weights['query_type']
        
        # 基于复杂度
        if retriever == 'efficient_vector':
            # 向量检索适合复杂查询
            complexity_score = features.complexity_score
        else:
            # BM25适合简单查询
            complexity_score = 1.0 - features.complexity_score
        
        score += complexity_score * self.rule_weights['complexity']
        
        # 基于实体数量
        if retriever == 'semantic_bm25':
            # BM25适合多实体查询
            entity_score = min(1.0, features.entity_count / 3.0)
        else:
            # 向量检索在少实体时更好
            entity_score = max(0.0, 1.0 - features.entity_count / 3.0)
        
        score += entity_score * self.rule_weights['entity_count']
        
        # 基于词数
        if retriever == 'efficient_vector':
            # 向量检索适合长查询
            if features.token_count > 8:
                token_score = 1.0
            elif features.token_count > 4:
                token_score = 0.7
            else:
                token_score = 0.4
        else:
            # BM25适合短查询
            if features.token_count <= 4:
                token_score = 1.0
            elif features.token_count <= 8:
                token_score = 0.7
            else:
                token_score = 0.4
        
        score += token_score * self.rule_weights['token_count']
        
        # 基于是否为问句
        if features.is_question:
            if retriever == 'efficient_vector':
                question_score = 0.8  # 向量检索擅长问答
            else:
                question_score = 0.6
        else:
            question_score = 0.7  # 非问句时差异不大
        
        score += question_score * self.rule_weights['is_question']
        
        # 基于领域特异性
        if retriever == 'efficient_vector':
            # 向量检索适合高领域特异性
            domain_score = features.domain_specificity
        else:
            # BM25在一般领域表现稳定
            domain_score = 0.8
        
        score += domain_score * self.rule_weights['domain_specificity']
        
        # 添加历史性能调整
        historical_performance = self._get_historical_performance(retriever, features)
        score = score * 0.7 + historical_performance * 0.3
        
        return min(1.0, score)
    
    def _get_historical_performance(self, retriever: str, features: QueryFeatures) -> float:
        """获取历史性能"""
        if not self.performance_history:
            return 0.5  # 默认值
        
        # 找到相似查询的历史记录
        similar_records = []
        for record in self.performance_history:
            similarity = self._calculate_feature_similarity(features, record.query_features)
            if similarity > 0.7:  # 相似度阈值
                similar_records.append(record)
        
        if not similar_records:
            return 0.5
        
        # 计算该检索器在相似查询上的平均性能
        performances = [
            record.retriever_performances.get(retriever, 0.0) 
            for record in similar_records
        ]
        
        return np.mean(performances) if performances else 0.5
    
    def _calculate_feature_similarity(self, features1: QueryFeatures, features2: QueryFeatures) -> float:
        """计算特征相似度"""
        # 简单的特征相似度计算
        similarities = []
        
        # 查询类型相似度
        similarities.append(1.0 if features1.query_type == features2.query_type else 0.0)
        
        # 复杂度相似度
        similarities.append(1.0 - abs(features1.complexity_score - features2.complexity_score))
        
        # 实体数量相似度
        max_entities = max(features1.entity_count, features2.entity_count, 1)
        similarities.append(1.0 - abs(features1.entity_count - features2.entity_count) / max_entities)
        
        # 词数相似度
        max_tokens = max(features1.token_count, features2.token_count, 1)
        similarities.append(1.0 - abs(features1.token_count - features2.token_count) / max_tokens)
        
        # 问句类型相似度
        similarities.append(1.0 if features1.is_question == features2.is_question else 0.0)
        
        return np.mean(similarities)
    
    def _features_to_vector(self, features: QueryFeatures) -> List[float]:
        """将特征转换为向量"""
        vector = [
            features.query_length / 100.0,  # 规范化
            features.token_count / 20.0,
            features.avg_word_length / 10.0,
            features.complexity_score,
            features.domain_specificity,
            features.entity_count / 10.0,
            1.0 if features.is_question else 0.0,
            1.0 if features.has_numeric else 0.0,
            1.0 if features.has_special_chars else 0.0,
            features.language_complexity,
        ]
        
        # 查询类型one-hot编码
        query_types = ['general', 'factual', 'conceptual', 'analytical', 'procedural']
        for qt in query_types:
            vector.append(1.0 if features.query_type == qt else 0.0)
        
        return vector
    
    def _initialize_learning_model(self) -> None:
        """初始化学习模型"""
        if len(self.performance_history) >= self.min_training_samples:
            self._train_learning_model()
    
    def _train_learning_model(self) -> None:
        """训练学习模型"""
        if len(self.performance_history) < self.min_training_samples:
            return
        
        # 准备训练数据
        X = []
        y = []
        
        for record in self.performance_history:
            feature_vector = self._features_to_vector(record.query_features)
            X.append(feature_vector)
            y.append(record.best_retriever)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        self.routing_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.routing_model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.routing_model.predict(X_test)
        self.model_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"路由模型训练完成，准确率: {self.model_accuracy:.3f}")
    
    def _update_learning_model(self) -> None:
        """更新学习模型"""
        if self.enable_learning:
            self._train_learning_model()
    
    def record_performance(self, query: Query, routing_decision: RoutingDecision, 
                          retriever_performances: Dict[str, float]) -> None:
        """记录性能"""
        # 确定最佳检索器
        best_retriever = max(retriever_performances, key=retriever_performances.get)
        
        # 分析查询特征
        features = self.query_analyzer.analyze(query)
        
        # 创建性能记录
        record = PerformanceRecord(
            query_id=query.query_id,
            query_text=query.text,
            query_features=features,
            routing_decision=routing_decision,
            retriever_performances=retriever_performances,
            best_retriever=best_retriever,
            timestamp=time.time()
        )
        
        # 添加到历史记录
        self.performance_history.append(record)
        
        # 定期保存
        if len(self.performance_history) % 50 == 0:
            self._save_performance_history()
    
    def _save_performance_history(self) -> None:
        """保存性能历史"""
        history_file = self.cache_dir / "performance_history.json"
        
        # 转换为可序列化格式
        history_data = [record.to_dict() for record in self.performance_history]
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
    
    def _load_performance_history(self) -> None:
        """加载性能历史"""
        history_file = self.cache_dir / "performance_history.json"
        
        if not history_file.exists():
            return
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            # 重建性能记录
            for data in history_data:
                features = QueryFeatures(**data['query_features'])
                routing_decision = RoutingDecision(**data['routing_decision'])
                
                record = PerformanceRecord(
                    query_id=data['query_id'],
                    query_text=data['query_text'],
                    query_features=features,
                    routing_decision=routing_decision,
                    retriever_performances=data['retriever_performances'],
                    best_retriever=data['best_retriever'],
                    timestamp=data['timestamp']
                )
                
                self.performance_history.append(record)
            
            print(f"加载了 {len(self.performance_history)} 条性能历史记录")
            
        except Exception as e:
            print(f"加载性能历史失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.performance_history:
            return {}
        
        # 检索器使用统计
        retriever_usage = defaultdict(int)
        retriever_performance = defaultdict(list)
        
        for record in self.performance_history:
            retriever_usage[record.routing_decision.primary_retriever] += 1
            
            for retriever, performance in record.retriever_performances.items():
                retriever_performance[retriever].append(performance)
        
        # 平均性能
        avg_performance = {
            retriever: np.mean(performances) 
            for retriever, performances in retriever_performance.items()
        }
        
        return {
            'total_queries': len(self.performance_history),
            'retriever_usage': dict(retriever_usage),
            'avg_performance': avg_performance,
            'model_accuracy': self.model_accuracy,
            'routing_strategy': self.routing_strategy,
            'learning_enabled': self.enable_learning
        }


def create_adaptive_router(config: Dict[str, Any]) -> AdaptiveRouter:
    """创建自适应路由器的工厂函数"""
    return AdaptiveRouter(config=config)