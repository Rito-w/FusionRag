"""
问题分类器模块
基于查询类型进行智能路由，决定使用哪些检索器
"""

import re
import json
import pickle
from typing import List, Dict, Any, Set, Optional
from pathlib import Path
from collections import defaultdict
import jieba

from ..utils.interfaces import Query
from ..utils.common import FileUtils, TextProcessor


class QueryClassifier:
    """查询分类器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 分类参数
        self.classes = self.config.get('classes', ['factual', 'analytical', 'procedural'])
        self.threshold = self.config.get('threshold', 0.5)
        
        # 分类特征
        self.feature_patterns = {
            'factual': {
                'keywords': ['what', 'who', 'when', 'where', 'which', 'define', 
                           '什么', '谁', '何时', '哪里', '哪个', '定义'],
                'patterns': [
                    r'\bwhat\s+is\b',
                    r'\bwho\s+is\b',
                    r'\bdefine\b',
                    r'什么是',
                    r'谁是',
                    r'定义'
                ]
            },
            'analytical': {
                'keywords': ['why', 'how', 'compare', 'analyze', 'relationship', 'cause',
                           '为什么', '如何', '比较', '分析', '关系', '原因'],
                'patterns': [
                    r'\bwhy\s+',
                    r'\bhow\s+',
                    r'\bcompare\b',
                    r'\banalyze\b',
                    r'为什么',
                    r'如何',
                    r'比较',
                    r'分析'
                ]
            },
            'procedural': {
                'keywords': ['procedure', 'step', 'process', 'method', 'treatment', 'therapy',
                           '步骤', '过程', '方法', '治疗', '疗法'],
                'patterns': [
                    r'\bstep\s+by\s+step\b',
                    r'\bprocedure\b',
                    r'\bprocess\b',
                    r'\bmethod\b',
                    r'步骤',
                    r'过程',
                    r'方法'
                ]
            }
        }
        
        # 检索器路由规则
        self.routing_rules = {
            'factual': ['dense', 'bm25'],      # 事实性查询：向量+BM25
            'analytical': ['bm25', 'graph'],   # 分析性查询：BM25+图
            'procedural': ['bm25', 'dense']    # 程序性查询：BM25+向量
        }
        
        # 停用词
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                             '的', '是', '在', '有', '和', '与', '对', '等', '中', '了', '也', '而', '为'])
    
    def classify_query(self, query: Query) -> Dict[str, Any]:
        """
        分类查询
        
        返回:
            {
                'predicted_class': str,
                'confidence': float,
                'class_scores': Dict[str, float],
                'recommended_retrievers': List[str]
            }
        """
        text = query.text.lower()
        
        # 计算每个类别的得分
        class_scores = {}
        
        for class_name, features in self.feature_patterns.items():
            score = self._calculate_class_score(text, features)
            class_scores[class_name] = score
        
        # 确定预测类别
        predicted_class = max(class_scores.items(), key=lambda x: x[1])[0]
        confidence = class_scores[predicted_class]
        
        # 如果置信度低，使用默认策略
        if confidence < self.threshold:
            predicted_class = 'factual'  # 默认为事实性查询
            recommended_retrievers = ['bm25', 'dense']
        else:
            recommended_retrievers = self.routing_rules.get(predicted_class, ['bm25', 'dense'])
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_scores': class_scores,
            'recommended_retrievers': recommended_retrievers,
            'query_features': self._extract_features(text)
        }
    
    def _calculate_class_score(self, text: str, features: Dict[str, Any]) -> float:
        """计算类别得分"""
        score = 0.0
        
        # 关键词匹配
        keywords = features.get('keywords', [])
        for keyword in keywords:
            if keyword in text:
                score += 1.0
        
        # 正则模式匹配
        patterns = features.get('patterns', [])
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 2.0  # 模式匹配权重更高
        
        # 归一化
        max_possible_score = len(keywords) + len(patterns) * 2
        if max_possible_score > 0:
            score = score / max_possible_score
        
        return score
    
    def _extract_features(self, text: str) -> Dict[str, Any]:
        """提取查询特征"""
        features = {}
        
        # 基本统计
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        
        # 疑问词
        question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which',
                         '什么', '谁', '何时', '哪里', '为什么', '如何', '哪个']
        features['has_question_word'] = any(word in text.lower() for word in question_words)
        
        # 特殊符号
        features['has_question_mark'] = '?' in text
        
        # 医学术语（简单检测）
        medical_terms = ['cancer', 'disease', 'treatment', 'therapy', 'drug', 'medication',
                        '癌症', '疾病', '治疗', '药物', '疗法']
        features['has_medical_term'] = any(term in text.lower() for term in medical_terms)
        
        return features
    
    def batch_classify(self, queries: List[Query]) -> Dict[str, Dict[str, Any]]:
        """批量分类查询"""
        results = {}
        
        for query in queries:
            results[query.query_id] = self.classify_query(query)
        
        return results
    
    def get_retriever_stats(self, classification_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """获取检索器使用统计"""
        retriever_counts = defaultdict(int)
        class_counts = defaultdict(int)
        
        for result in classification_results.values():
            predicted_class = result['predicted_class']
            class_counts[predicted_class] += 1
            
            for retriever in result['recommended_retrievers']:
                retriever_counts[retriever] += 1
        
        return {
            'class_distribution': dict(class_counts),
            'retriever_usage': dict(retriever_counts),
            'total_queries': len(classification_results)
        }
    
    def save_model(self, model_path: str) -> None:
        """保存分类器模型"""
        model_data = {
            'config': self.config,
            'feature_patterns': self.feature_patterns,
            'routing_rules': self.routing_rules,
            'classes': self.classes,
            'threshold': self.threshold
        }
        
        FileUtils.save_pickle(model_data, model_path)
        print(f"分类器模型已保存到: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """加载分类器模型"""
        try:
            model_data = FileUtils.load_pickle(model_path)
            
            self.config = model_data.get('config', {})
            self.feature_patterns = model_data.get('feature_patterns', self.feature_patterns)
            self.routing_rules = model_data.get('routing_rules', self.routing_rules)
            self.classes = model_data.get('classes', self.classes)
            self.threshold = model_data.get('threshold', self.threshold)
            
            print(f"分类器模型已从 {model_path} 加载完成")
            
        except Exception as e:
            raise RuntimeError(f"加载分类器模型失败: {e}")


class AdaptiveQueryRouter:
    """自适应查询路由器"""
    
    def __init__(self, classifier: QueryClassifier, config: Dict[str, Any] = None):
        self.classifier = classifier
        self.config = config or {}
        
        # 性能追踪
        self.performance_history = defaultdict(list)
        self.route_usage = defaultdict(int)
        
        # 自适应参数
        self.adaptation_enabled = self.config.get('adaptation_enabled', True)
        self.min_samples = self.config.get('min_samples', 10)
        self.performance_threshold = self.config.get('performance_threshold', 0.1)
    
    def route_query(self, query: Query, available_retrievers: List[str]) -> Dict[str, Any]:
        """
        智能路由查询
        
        Args:
            query: 查询对象
            available_retrievers: 可用的检索器列表
            
        Returns:
            路由决策结果
        """
        # 基础分类
        classification = self.classifier.classify_query(query)
        recommended = classification['recommended_retrievers']
        
        # 过滤可用的检索器
        available_recommended = [r for r in recommended if r in available_retrievers]
        
        # 自适应调整（基于历史性能）
        if self.adaptation_enabled:
            adjusted_retrievers = self._adaptive_adjustment(
                query, available_recommended, available_retrievers
            )
        else:
            adjusted_retrievers = available_recommended
        
        # 如果没有推荐的检索器，使用所有可用的
        if not adjusted_retrievers:
            adjusted_retrievers = available_retrievers
        
        # 记录路由使用
        route_key = ','.join(sorted(adjusted_retrievers))
        self.route_usage[route_key] += 1
        
        return {
            'retrievers': adjusted_retrievers,
            'classification': classification,
            'adaptation_applied': adjusted_retrievers != available_recommended,
            'route_confidence': classification['confidence']
        }
    
    def _adaptive_adjustment(self, query: Query, recommended: List[str], 
                           available: List[str]) -> List[str]:
        """基于历史性能进行自适应调整"""
        
        # 简单的自适应策略：如果某个检索器表现不佳，尝试其他组合
        adjusted = recommended.copy()
        
        # 检查是否有足够的历史数据
        query_type = self.classifier.classify_query(query)['predicted_class']
        
        if len(self.performance_history[query_type]) >= self.min_samples:
            avg_performance = sum(self.performance_history[query_type][-self.min_samples:]) / self.min_samples
            
            # 如果性能低于阈值，尝试添加其他检索器
            if avg_performance < self.performance_threshold:
                for retriever in available:
                    if retriever not in adjusted:
                        adjusted.append(retriever)
                        break  # 只添加一个额外的检索器
        
        return adjusted
    
    def update_performance(self, query: Query, retrievers_used: List[str], 
                          performance_score: float) -> None:
        """更新性能历史"""
        query_type = self.classifier.classify_query(query)['predicted_class']
        self.performance_history[query_type].append(performance_score)
        
        # 保持历史记录在合理范围内
        max_history = 100
        if len(self.performance_history[query_type]) > max_history:
            self.performance_history[query_type] = self.performance_history[query_type][-max_history:]
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        return {
            'route_usage': dict(self.route_usage),
            'performance_history': {
                k: {
                    'count': len(v),
                    'avg_performance': sum(v) / len(v) if v else 0,
                    'recent_performance': sum(v[-10:]) / len(v[-10:]) if len(v) >= 10 else None
                }
                for k, v in self.performance_history.items()
            },
            'total_queries_routed': sum(self.route_usage.values())
        }


def create_query_classifier(config: Dict[str, Any] = None) -> QueryClassifier:
    """创建查询分类器的工厂函数"""
    return QueryClassifier(config)


def create_adaptive_router(classifier: QueryClassifier, config: Dict[str, Any] = None) -> AdaptiveQueryRouter:
    """创建自适应路由器的工厂函数"""
    return AdaptiveQueryRouter(classifier, config)