"""
查询特征分析器
分析查询特征，为自适应路由提供依据
优化版本：使用模型缓存，避免重复加载模型
"""

import re
import enum
from typing import Dict, Any, List, Optional
import numpy as np

from ..utils.interfaces import Query
from ..utils.model_cache import model_cache


class QueryType(enum.Enum):
    """查询类型枚举"""
    KEYWORD = "keyword"       # 关键词查询
    SEMANTIC = "semantic"     # 语义查询
    MIXED = "mixed"           # 混合查询
    UNKNOWN = "unknown"       # 未知类型


class QueryFeatures:
    """查询特征类"""
    
    def __init__(self):
        # 基本特征
        self.query_length = 0          # 查询长度（字符数）
        self.token_count = 0           # 词数
        self.entity_count = 0          # 实体数量
        self.avg_word_length = 0.0     # 平均词长度
        
        # 语义特征
        self.complexity_score = 0.0    # 复杂度得分
        self.is_question = False       # 是否为问句
        self.domain_specificity = 0.0  # 领域特异性
        
        # 实体特征
        self.entity_types = {}         # 实体类型及数量
        self.has_numeric = False       # 是否包含数字
        self.has_special_chars = False # 是否包含特殊字符
        
        # 查询类型
        self.query_type = QueryType.UNKNOWN  # 查询类型
        
        # 向量表示
        self.embedding = None          # 查询嵌入向量


class QueryFeatureAnalyzer:
    """查询特征分析器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 模型配置
        self.semantic_model_name = self.config.get('semantic_model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.spacy_model_name = self.config.get('spacy_model_name', 'en_core_web_sm')
        
        # 阈值配置
        self.keyword_query_threshold = self.config.get('keyword_query_threshold', 0.7)
        self.semantic_query_threshold = self.config.get('semantic_query_threshold', 0.6)
        
        # 功能开关
        self.disable_ner = self.config.get('disable_ner', False)
        self.use_simple_features = self.config.get('use_simple_features', False)
        
        # 加载模型（使用缓存）
        self._load_models()
    
    def _load_models(self):
        """加载模型（使用缓存）"""
        # 加载语义模型
        self.semantic_model = model_cache.get_embedding_model(self.semantic_model_name)
        
        # 加载NER模型（如果启用）
        self.nlp = None
        if not self.disable_ner and self.spacy_model_name:
            self.nlp = model_cache.get_spacy_model(self.spacy_model_name)
    
    def analyze_query(self, query: Query) -> QueryFeatures:
        """分析查询特征"""
        features = QueryFeatures()
        
        # 基本特征
        text = query.text
        features.query_length = len(text)
        
        # 分词
        words = re.findall(r'\b\w+\b', text.lower())
        features.token_count = len(words)
        
        if features.token_count > 0:
            features.avg_word_length = sum(len(word) for word in words) / features.token_count
        
        # 检测问句
        features.is_question = any(text.startswith(q) for q in ['what', 'who', 'when', 'where', 'why', 'how', 'which', 'is', 'are', 'do', 'does', 'can']) or '?' in text
        
        # 检测数字和特殊字符
        features.has_numeric = bool(re.search(r'\d', text))
        features.has_special_chars = bool(re.search(r'[^\w\s]', text))
        
        # 实体识别（如果启用）
        if not self.disable_ner and self.nlp:
            doc = self.nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            features.entity_count = len(entities)
            
            # 统计实体类型
            for _, entity_type in entities:
                if entity_type not in features.entity_types:
                    features.entity_types[entity_type] = 0
                features.entity_types[entity_type] += 1
        
        # 计算语义嵌入
        if self.semantic_model:
            features.embedding = self.semantic_model.encode(text, convert_to_numpy=True)
        
        # 计算复杂度得分
        if not self.use_simple_features:
            # 复杂度得分基于多个因素
            complexity_factors = [
                features.token_count / 10,  # 词数（标准化）
                features.avg_word_length / 5,  # 平均词长（标准化）
                features.entity_count / 3 if features.entity_count > 0 else 0,  # 实体数量
                0.5 if features.is_question else 0,  # 是否为问句
                0.3 if features.has_numeric else 0,  # 是否包含数字
                0.2 if features.has_special_chars else 0  # 是否包含特殊字符
            ]
            features.complexity_score = min(1.0, sum(complexity_factors) / 3)
        else:
            # 简化版复杂度计算
            features.complexity_score = min(1.0, features.token_count / 15)
        
        # 确定查询类型
        features.query_type = self._determine_query_type(features)
        
        return features
    
    def _determine_query_type(self, features: QueryFeatures) -> QueryType:
        """确定查询类型"""
        # 简单规则：基于词数和复杂度
        if features.token_count <= 3 and features.complexity_score < self.keyword_query_threshold:
            return QueryType.KEYWORD
        elif features.complexity_score >= self.semantic_query_threshold:
            return QueryType.SEMANTIC
        else:
            return QueryType.MIXED
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "semantic_model": self.semantic_model_name,
            "spacy_model": self.spacy_model_name,
            "disable_ner": self.disable_ner,
            "use_simple_features": self.use_simple_features
        }