#!/usr/bin/env python3
"""
简化版查询分析器
专门为论文实验设计，快速高效的查询特征提取和分类
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..utils.interfaces import Query, QueryType


@dataclass
class SimpleQueryFeatures:
    """简化的查询特征"""
    query_id: str
    query_text: str
    query_type: QueryType
    length: int
    word_count: int
    is_question: bool
    has_numbers: bool
    has_entities: bool
    complexity_level: str  # 'simple', 'medium', 'complex'
    domain_hint: str  # 'medical', 'technical', 'general', 'financial'
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = self.__dict__.copy()
        result['query_type'] = self.query_type.value
        return result


class SimpleQueryAnalyzer:
    """简化版查询分析器
    
    专注于快速、准确的查询分类，不依赖复杂的NLP模型
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 查询类型关键词
        self.question_words = {
            'what', 'who', 'when', 'where', 'why', 'how', 'which', 'whose',
            'is', 'are', 'do', 'does', 'did', 'can', 'could', 'would', 'will', 'should'
        }
        
        self.technical_terms = {
            'algorithm', 'machine learning', 'neural network', 'deep learning', 'ai',
            'artificial intelligence', 'model', 'training', 'optimization', 'classification',
            'regression', 'clustering', 'supervised', 'unsupervised', 'reinforcement',
            'backpropagation', 'gradient', 'loss function', 'overfitting', 'regularization'
        }
        
        self.medical_terms = {
            'disease', 'symptom', 'treatment', 'diagnosis', 'medicine', 'drug', 'therapy',
            'patient', 'clinical', 'medical', 'health', 'healthcare', 'hospital', 'doctor',
            'diabetes', 'cancer', 'covid', 'virus', 'infection', 'vaccine', 'pharmaceutical'
        }
        
        self.financial_terms = {
            'stock', 'price', 'market', 'investment', 'trading', 'finance', 'financial',
            'economy', 'economic', 'bank', 'banking', 'credit', 'loan', 'mortgage',
            'portfolio', 'dividend', 'earnings', 'revenue', 'profit', 'loss'
        }
        
        # 实体模式
        self.entity_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # 人名或公司名
            r'\b[A-Z]{2,}\b',  # 缩写
            r'\b\d{4}\b',  # 年份
            r'\$\d+',  # 金额
            r'\d+%',  # 百分比
        ]
    
    def analyze_query(self, query: Query) -> SimpleQueryFeatures:
        """分析查询特征"""
        text = query.text.lower()
        original_text = query.text
        
        # 基本特征
        length = len(original_text)
        words = text.split()
        word_count = len(words)
        
        # 问句检测
        is_question = self._is_question(text, original_text)
        
        # 数字检测
        has_numbers = bool(re.search(r'\d', text))
        
        # 实体检测
        has_entities = self._has_entities(original_text)
        
        # 复杂度评估
        complexity_level = self._assess_complexity(text, word_count)
        
        # 领域提示
        domain_hint = self._detect_domain(text)
        
        # 查询类型分类
        query_type = self._classify_query_type(text, is_question, has_entities, domain_hint)
        
        return SimpleQueryFeatures(
            query_id=query.query_id,
            query_text=query.text,
            query_type=query_type,
            length=length,
            word_count=word_count,
            is_question=is_question,
            has_numbers=has_numbers,
            has_entities=has_entities,
            complexity_level=complexity_level,
            domain_hint=domain_hint
        )
    
    def _is_question(self, text: str, original_text: str) -> bool:
        """判断是否为问句"""
        # 检查问号
        if '?' in original_text:
            return True
        
        # 检查问句开头词
        words = text.split()
        if words and words[0] in self.question_words:
            return True
        
        return False
    
    def _has_entities(self, text: str) -> bool:
        """检测是否包含实体"""
        for pattern in self.entity_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _assess_complexity(self, text: str, word_count: int) -> str:
        """评估查询复杂度"""
        if word_count <= 3:
            return 'simple'
        elif word_count <= 8:
            return 'medium'
        else:
            return 'complex'
    
    def _detect_domain(self, text: str) -> str:
        """检测查询领域"""
        # 检查技术领域
        for term in self.technical_terms:
            if term in text:
                return 'technical'
        
        # 检查医疗领域
        for term in self.medical_terms:
            if term in text:
                return 'medical'
        
        # 检查金融领域
        for term in self.financial_terms:
            if term in text:
                return 'financial'
        
        return 'general'
    
    def _classify_query_type(self, text: str, is_question: bool, has_entities: bool, domain: str) -> QueryType:
        """分类查询类型"""
        # 基于规则的分类
        if is_question and len(text.split()) > 5:
            return QueryType.SEMANTIC
        elif has_entities:
            return QueryType.ENTITY
        elif domain in ['technical', 'medical'] and not is_question:
            return QueryType.KEYWORD
        elif is_question:
            return QueryType.SEMANTIC
        else:
            return QueryType.KEYWORD
    
    def batch_analyze(self, queries: List[Query]) -> List[SimpleQueryFeatures]:
        """批量分析查询"""
        return [self.analyze_query(query) for query in queries]


def create_simple_query_analyzer(config: Dict[str, Any] = None) -> SimpleQueryAnalyzer:
    """创建简化查询分析器的工厂函数"""
    return SimpleQueryAnalyzer(config=config)