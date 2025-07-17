"""自适应路由器
根据查询特征选择合适的索引和融合策略
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
from datetime import datetime

from ..utils.interfaces import Query, QueryType
from ..utils.common import FileUtils
from ..analysis.query_analyzer import QueryFeatures

try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
except ImportError:
    print("请安装必要的依赖: pip install pandas scikit-learn")


@dataclass
class IndexPerformanceRecord:
    """索引性能记录"""
    query_id: str
    query_type: str
    index_name: str
    accuracy: float
    recall: float
    mrr: float
    ndcg: float
    latency: float  # 毫秒
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexPerformanceRecord':
        """从字典创建实例"""
        return cls(**data)


@dataclass
class RoutingDecision:
    """路由决策"""
    query_id: str
    query_text: str
    query_type: str
    primary_index: str
    secondary_indices: List[str]
    fusion_method: str
    fusion_weights: Dict[str, float]
    confidence: float
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RoutingDecision':
        """从字典创建实例"""
        return cls(**data)


class AdaptiveRouter:
    """自适应路由器
    
    根据查询特征选择合适的索引和融合策略
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 可用索引配置
        self.available_indices = self.config.get('available_indices', [])
        
        # 融合方法配置
        self.available_fusion_methods = self.config.get('available_fusion_methods', ['weighted_sum', 'reciprocal_rank_fusion', 'max_score'])
        
        # 路由策略配置
        self.routing_strategy = self.config.get('routing_strategy', 'ml')  # 'rule', 'ml', 'hybrid'
        self.default_primary_index = self.config.get('default_primary_index', '')
        self.default_secondary_indices = self.config.get('default_secondary_indices', [])
        self.default_fusion_method = self.config.get('default_fusion_method', 'weighted_sum')
        
        # 机器学习模型配置
        self.model_type = self.config.get('model_type', 'random_forest')  # 'random_forest', 'logistic_regression'
        self.feature_columns = self.config.get('feature_columns', [
            'length', 'token_count', 'keyword_count', 'entity_count', 
            'has_numbers', 'has_special_chars', 'avg_word_length', 
            'is_question', 'complexity_score', 'specificity_score'
        ])
        
        # 性能记录和决策历史
        self.performance_records: List[IndexPerformanceRecord] = []
        self.routing_history: List[RoutingDecision] = []
        
        # 模型和缩放器
        self.index_selector_model = None
        self.fusion_selector_model = None
        self.scaler = None
        
        # 数据路径
        self.data_dir = self.config.get('data_dir', 'data/adaptive_router')
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载历史数据
        self._load_data()
        
        # 如果有足够的数据，初始化模型
        if len(self.performance_records) >= 50:
            self._init_models()
    
    def _load_data(self) -> None:
        """加载历史数据"""
        # 加载性能记录
        performance_path = os.path.join(self.data_dir, 'performance_records.json')
        if os.path.exists(performance_path):
            try:
                with open(performance_path, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                self.performance_records = [IndexPerformanceRecord.from_dict(r) for r in records]
                print(f"已加载 {len(self.performance_records)} 条性能记录")
            except Exception as e:
                print(f"加载性能记录失败: {e}")
        
        # 加载路由历史
        history_path = os.path.join(self.data_dir, 'routing_history.json')
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                self.routing_history = [RoutingDecision.from_dict(h) for h in history]
                print(f"已加载 {len(self.routing_history)} 条路由历史")
            except Exception as e:
                print(f"加载路由历史失败: {e}")
        
        # 加载模型
        model_path = os.path.join(self.data_dir, 'index_selector_model.pkl')
        if os.path.exists(model_path):
            try:
                self.index_selector_model = FileUtils.load_pickle(model_path)
                print("已加载索引选择器模型")
            except Exception as e:
                print(f"加载索引选择器模型失败: {e}")
        
        fusion_model_path = os.path.join(self.data_dir, 'fusion_selector_model.pkl')
        if os.path.exists(fusion_model_path):
            try:
                self.fusion_selector_model = FileUtils.load_pickle(fusion_model_path)
                print("已加载融合选择器模型")
            except Exception as e:
                print(f"加载融合选择器模型失败: {e}")
        
        scaler_path = os.path.join(self.data_dir, 'feature_scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                self.scaler = FileUtils.load_pickle(scaler_path)
                print("已加载特征缩放器")
            except Exception as e:
                print(f"加载特征缩放器失败: {e}")
    
    def _save_data(self) -> None:
        """保存历史数据"""
        # 保存性能记录
        performance_path = os.path.join(self.data_dir, 'performance_records.json')
        try:
            records = [r.to_dict() for r in self.performance_records]
            with open(performance_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存性能记录失败: {e}")
        
        # 保存路由历史
        history_path = os.path.join(self.data_dir, 'routing_history.json')
        try:
            history = [h.to_dict() for h in self.routing_history]
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存路由历史失败: {e}")
        
        # 保存模型
        if self.index_selector_model is not None:
            model_path = os.path.join(self.data_dir, 'index_selector_model.pkl')
            FileUtils.save_pickle(self.index_selector_model, model_path)
        
        if self.fusion_selector_model is not None:
            fusion_model_path = os.path.join(self.data_dir, 'fusion_selector_model.pkl')
            FileUtils.save_pickle(self.fusion_selector_model, fusion_model_path)
        
        if self.scaler is not None:
            scaler_path = os.path.join(self.data_dir, 'feature_scaler.pkl')
            FileUtils.save_pickle(self.scaler, scaler_path)
    
    def _prepare_features(self, query_features: QueryFeatures) -> np.ndarray:
        """准备模型输入特征"""
        # 提取特征
        features = []
        for col in self.feature_columns:
            if col == 'has_numbers':
                features.append(1 if query_features.has_numbers else 0)
            elif col == 'has_special_chars':
                features.append(1 if query_features.has_special_chars else 0)
            elif col == 'is_question':
                features.append(1 if query_features.is_question else 0)
            else:
                features.append(getattr(query_features, col))
        
        # 转换为numpy数组
        features = np.array(features).reshape(1, -1)
        
        # 如果有缩放器，则缩放特征
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        return features
    
    def _init_models(self) -> None:
        """初始化机器学习模型"""
        if len(self.performance_records) < 50:
            print("性能记录数量不足，无法初始化模型")
            return
        
        try:
            # 准备数据
            df = pd.DataFrame([r.to_dict() for r in self.performance_records])
            
            # 找出每个查询的最佳索引
            best_indices = df.loc[df.groupby('query_id')['ndcg'].idxmax()]
            best_indices = best_indices[['query_id', 'index_name']]
            
            # 合并查询特征和最佳索引
            query_features = {}
            for decision in self.routing_history:
                query_features[decision.query_id] = decision
            
            # 创建训练数据
            X = []
            y_index = []
            y_fusion = []
            
            for _, row in best_indices.iterrows():
                query_id = row['query_id']
                if query_id in query_features:
                    decision = query_features[query_id]
                    
                    # 提取特征
                    features = []
                    for col in self.feature_columns:
                        # 这里假设决策对象中有查询特征信息
                        # 实际实现中可能需要从其他地方获取
                        if col in decision.to_dict():
                            features.append(decision.to_dict()[col])
                        else:
                            features.append(0)  # 默认值
                    
                    X.append(features)
                    y_index.append(row['index_name'])
                    y_fusion.append(decision.fusion_method)
            
            if not X:
                print("没有足够的训练数据")
                return
            
            # 转换为numpy数组
            X = np.array(X)
            
            # 特征缩放
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # 分割训练集和测试集
            X_train, X_test, y_index_train, y_index_test = train_test_split(
                X_scaled, y_index, test_size=0.2, random_state=42
            )
            
            _, _, y_fusion_train, y_fusion_test = train_test_split(
                X_scaled, y_fusion, test_size=0.2, random_state=42
            )
            
            # 训练索引选择器模型
            if self.model_type == 'random_forest':
                self.index_selector_model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                self.index_selector_model = LogisticRegression(max_iter=1000, random_state=42)
            
            self.index_selector_model.fit(X_train, y_index_train)
            
            # 评估索引选择器模型
            y_index_pred = self.index_selector_model.predict(X_test)
            index_accuracy = accuracy_score(y_index_test, y_index_pred)
            print(f"索引选择器模型准确率: {index_accuracy:.4f}")
            
            # 训练融合选择器模型
            if self.model_type == 'random_forest':
                self.fusion_selector_model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                self.fusion_selector_model = LogisticRegression(max_iter=1000, random_state=42)
            
            self.fusion_selector_model.fit(X_train, y_fusion_train)
            
            # 评估融合选择器模型
            y_fusion_pred = self.fusion_selector_model.predict(X_test)
            fusion_accuracy = accuracy_score(y_fusion_test, y_fusion_pred)
            print(f"融合选择器模型准确率: {fusion_accuracy:.4f}")
            
        except Exception as e:
            print(f"初始化模型失败: {e}")
    
    def _rule_based_routing(self, query_features: QueryFeatures) -> Tuple[str, List[str], str, Dict[str, float], float]:
        """基于规则的路由策略"""
        query_type = query_features.query_type
        
        # 默认值
        primary_index = self.default_primary_index or (self.available_indices[0] if self.available_indices else "")
        secondary_indices = self.default_secondary_indices[:]
        fusion_method = self.default_fusion_method
        confidence = 0.7  # 默认置信度
        
        # 根据查询类型选择索引
        if query_type == QueryType.KEYWORD:
            # 关键词查询优先使用BM25
            for idx in self.available_indices:
                if 'bm25' in idx.lower():
                    primary_index = idx
                    confidence = 0.8
                    break
            
            # 次级索引可以是向量索引
            for idx in self.available_indices:
                if idx != primary_index and ('vector' in idx.lower() or 'dense' in idx.lower()):
                    secondary_indices.append(idx)
            
            # 限制次级索引数量
            secondary_indices = secondary_indices[:2]
            
            # 融合方法
            fusion_method = 'weighted_sum'
            
        elif query_type == QueryType.SEMANTIC:
            # 语义查询优先使用向量索引
            for idx in self.available_indices:
                if 'vector' in idx.lower() or 'dense' in idx.lower() or 'hnsw' in idx.lower():
                    primary_index = idx
                    confidence = 0.85
                    break
            
            # 次级索引可以是BM25
            for idx in self.available_indices:
                if idx != primary_index and 'bm25' in idx.lower():
                    secondary_indices.append(idx)
            
            # 限制次级索引数量
            secondary_indices = secondary_indices[:1]
            
            # 融合方法
            fusion_method = 'reciprocal_rank_fusion'
            
        elif query_type == QueryType.ENTITY:
            # 实体查询优先使用图索引或BM25
            for idx in self.available_indices:
                if 'graph' in idx.lower():
                    primary_index = idx
                    confidence = 0.9
                    break
            
            # 如果没有图索引，使用BM25
            if primary_index == self.default_primary_index:
                for idx in self.available_indices:
                    if 'bm25' in idx.lower():
                        primary_index = idx
                        confidence = 0.75
                        break
            
            # 次级索引可以是向量索引
            for idx in self.available_indices:
                if idx != primary_index and ('vector' in idx.lower() or 'dense' in idx.lower()):
                    secondary_indices.append(idx)
            
            # 限制次级索引数量
            secondary_indices = secondary_indices[:1]
            
            # 融合方法
            fusion_method = 'max_score'
            
        else:  # HYBRID或UNKNOWN
            # 混合查询使用多种索引
            vector_index = None
            bm25_index = None
            
            # 寻找可用的向量索引和BM25索引
            for idx in self.available_indices:
                if 'vector' in idx.lower() or 'dense' in idx.lower() or 'hnsw' in idx.lower():
                    vector_index = idx
                elif 'bm25' in idx.lower():
                    bm25_index = idx
            
            # 设置主索引和次级索引
            if vector_index and bm25_index:
                primary_index = vector_index
                secondary_indices = [bm25_index]
                confidence = 0.7
            elif vector_index:
                primary_index = vector_index
                confidence = 0.6
            elif bm25_index:
                primary_index = bm25_index
                confidence = 0.6
            
            # 融合方法
            fusion_method = 'weighted_sum'
        
        # 计算融合权重
        fusion_weights = {}
        if primary_index:
            fusion_weights[primary_index] = 0.7
        
        for i, idx in enumerate(secondary_indices):
            # 次级索引权重递减
            fusion_weights[idx] = 0.3 / (i + 1)
        
        return primary_index, secondary_indices, fusion_method, fusion_weights, confidence
    
    def _ml_based_routing(self, query_features: QueryFeatures) -> Tuple[str, List[str], str, Dict[str, float], float]:
        """基于机器学习的路由策略"""
        # 如果模型未初始化，使用规则策略
        if self.index_selector_model is None or self.fusion_selector_model is None:
            return self._rule_based_routing(query_features)
        
        try:
            # 准备特征
            features = self._prepare_features(query_features)
            
            # 预测主索引
            primary_index = self.index_selector_model.predict(features)[0]
            
            # 获取索引概率
            index_probs = self.index_selector_model.predict_proba(features)[0]
            index_classes = self.index_selector_model.classes_
            
            # 按概率排序选择次级索引
            index_prob_pairs = [(idx, prob) for idx, prob in zip(index_classes, index_probs)]
            index_prob_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 选择概率最高的作为主索引（如果与预测不同）
            if index_prob_pairs[0][0] != primary_index:
                primary_index = index_prob_pairs[0][0]
            
            # 选择2-3个概率较高的作为次级索引
            secondary_indices = [idx for idx, _ in index_prob_pairs[1:3] if idx in self.available_indices]
            
            # 预测融合方法
            fusion_method = self.fusion_selector_model.predict(features)[0]
            
            # 计算置信度（使用主索引的预测概率）
            confidence = max(index_probs)
            
            # 计算融合权重
            fusion_weights = {}
            total_weight = sum(prob for idx, prob in index_prob_pairs if idx == primary_index or idx in secondary_indices)
            
            # 主索引权重
            primary_prob = next(prob for idx, prob in index_prob_pairs if idx == primary_index)
            fusion_weights[primary_index] = primary_prob / total_weight
            
            # 次级索引权重
            for idx in secondary_indices:
                idx_prob = next(prob for i, prob in index_prob_pairs if i == idx)
                fusion_weights[idx] = idx_prob / total_weight
            
            return primary_index, secondary_indices, fusion_method, fusion_weights, confidence
            
        except Exception as e:
            print(f"机器学习路由失败: {e}")
            # 失败时回退到规则策略
            return self._rule_based_routing(query_features)
    
    def _hybrid_routing(self, query_features: QueryFeatures) -> Tuple[str, List[str], str, Dict[str, float], float]:
        """混合路由策略（结合规则和机器学习）"""
        # 获取规则策略结果
        rule_primary, rule_secondary, rule_fusion, rule_weights, rule_confidence = \
            self._rule_based_routing(query_features)
        
        # 如果模型未初始化，直接使用规则策略
        if self.index_selector_model is None or self.fusion_selector_model is None:
            return rule_primary, rule_secondary, rule_fusion, rule_weights, rule_confidence
        
        try:
            # 获取机器学习策略结果
            ml_primary, ml_secondary, ml_fusion, ml_weights, ml_confidence = \
                self._ml_based_routing(query_features)
            
            # 根据置信度选择策略
            if ml_confidence >= rule_confidence:
                return ml_primary, ml_secondary, ml_fusion, ml_weights, ml_confidence
            else:
                return rule_primary, rule_secondary, rule_fusion, rule_weights, rule_confidence
                
        except Exception as e:
            print(f"混合路由失败: {e}")
            # 失败时使用规则策略
            return rule_primary, rule_secondary, rule_fusion, rule_weights, rule_confidence
    
    def route(self, query_features: QueryFeatures) -> RoutingDecision:
        """根据查询特征进行路由决策"""
        # 根据策略选择路由方法
        if self.routing_strategy == 'rule':
            primary_index, secondary_indices, fusion_method, fusion_weights, confidence = \
                self._rule_based_routing(query_features)
        elif self.routing_strategy == 'ml':
            primary_index, secondary_indices, fusion_method, fusion_weights, confidence = \
                self._ml_based_routing(query_features)
        else:  # hybrid
            primary_index, secondary_indices, fusion_method, fusion_weights, confidence = \
                self._hybrid_routing(query_features)
        
        # 创建路由决策
        decision = RoutingDecision(
            query_id=query_features.query_id,
            query_text=query_features.query_text,
            query_type=query_features.query_type.value,
            primary_index=primary_index,
            secondary_indices=secondary_indices,
            fusion_method=fusion_method,
            fusion_weights=fusion_weights,
            confidence=confidence
        )
        
        # 记录决策历史
        self.routing_history.append(decision)
        
        # 定期保存数据
        if len(self.routing_history) % 10 == 0:
            self._save_data()
        
        return decision
    
    def update_performance(self, performance_record: IndexPerformanceRecord) -> None:
        """更新索引性能记录"""
        self.performance_records.append(performance_record)
        
        # 定期保存数据
        if len(self.performance_records) % 10 == 0:
            self._save_data()
        
        # 当有足够数据时更新模型
        if len(self.performance_records) % 100 == 0 and len(self.performance_records) >= 50:
            self._init_models()
    
    def batch_update_performance(self, records: List[IndexPerformanceRecord]) -> None:
        """批量更新索引性能记录"""
        self.performance_records.extend(records)
        self._save_data()
        
        # 当有足够数据时更新模型
        if len(self.performance_records) >= 50:
            self._init_models()
    
    def get_fusion_weights(self, indices: List[str]) -> Dict[str, float]:
        """获取索引融合权重"""
        # 如果没有索引，返回空字典
        if not indices:
            return {}
        
        # 如果只有一个索引，给它满权重
        if len(indices) == 1:
            return {indices[0]: 1.0}
        
        # 默认权重分配
        weights = {}
        primary_weight = 0.6
        remaining_weight = 1.0 - primary_weight
        
        weights[indices[0]] = primary_weight
        
        # 为其余索引平均分配剩余权重
        for i, idx in enumerate(indices[1:]):
            weights[idx] = remaining_weight / (len(indices) - 1)
        
        return weights
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取路由器统计信息"""
        stats = {
            'routing_strategy': self.routing_strategy,
            'available_indices': self.available_indices,
            'available_fusion_methods': self.available_fusion_methods,
            'performance_records_count': len(self.performance_records),
            'routing_history_count': len(self.routing_history),
            'model_initialized': self.index_selector_model is not None and self.fusion_selector_model is not None
        }
        
        # 如果有足够的历史数据，计算更多统计信息
        if self.routing_history:
            # 计算各索引被选为主索引的次数
            primary_counts = {}
            for decision in self.routing_history:
                primary_counts[decision.primary_index] = primary_counts.get(decision.primary_index, 0) + 1
            
            # 计算各融合方法被选择的次数
            fusion_counts = {}
            for decision in self.routing_history:
                fusion_counts[decision.fusion_method] = fusion_counts.get(decision.fusion_method, 0) + 1
            
            # 添加到统计信息
            stats['primary_index_distribution'] = primary_counts
            stats['fusion_method_distribution'] = fusion_counts
            stats['average_confidence'] = sum(d.confidence for d in self.routing_history) / len(self.routing_history)
        
        return stats


def create_adaptive_router(config: Dict[str, Any] = None) -> AdaptiveRouter:
    """创建自适应路由器的工厂函数"""
    return AdaptiveRouter(config=config)