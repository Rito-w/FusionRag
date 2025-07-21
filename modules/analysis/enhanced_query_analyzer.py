#!/usr/bin/env python3
"""
增强版查询分析器
结合规则和嵌入模型的高级查询分析
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
from transformers import AutoModel, AutoTokenizer

from ..utils.interfaces import Query, QueryType


@dataclass
class EnhancedQueryFeatures:
    """增强的查询特征"""
    query_id: str
    query_text: str
    rule_based_type: QueryType
    embedding_based_type: str
    final_type: str
    confidence: float
    complexity: float
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
        result['rule_based_type'] = self.rule_based_type.value
        return result


class EnhancedQueryAnalyzer:
    """增强版查询分析器
    
    结合规则和嵌入模型的高级查询分析
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        from .simple_query_analyzer import SimpleQueryAnalyzer
        
        self.config = config or {}
        self.base_analyzer = SimpleQueryAnalyzer(config)
        
        # 加载嵌入模型
        model_name = self.config.get('semantic_model_name', 'models/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6')
        self.device = 'cuda' if torch.cuda.is_available() and self.config.get('use_gpu', True) else 'cpu'
        
        print(f"加载嵌入模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # 为每种查询类型创建原型查询
        self.prototypes = {
            'entity': [
                "Who is Albert Einstein", 
                "Apple Inc stock price", 
                "COVID-19 symptoms",
                "Barack Obama biography",
                "Tesla Model 3 specifications"
            ],
            'semantic': [
                "What is the meaning of life", 
                "How does photosynthesis work", 
                "Explain quantum physics",
                "Why is the sky blue",
                "How do vaccines work"
            ],
            'keyword': [
                "python tutorial", 
                "cancer research", 
                "machine learning algorithms",
                "climate change data",
                "neural networks introduction"
            ],
            'factual': [
                "capital of France", 
                "height of Mount Everest", 
                "when was World War II",
                "population of Tokyo",
                "distance from Earth to Moon"
            ],
            'procedural': [
                "how to bake bread", 
                "steps to install Python", 
                "way to solve equations",
                "method for training neural networks",
                "process of photosynthesis"
            ]
        }
        
        # 预计算原型嵌入
        print("计算原型查询嵌入...")
        self.prototype_embeddings = self._compute_prototype_embeddings()
        print("原型嵌入计算完成")
    
    def _compute_prototype_embeddings(self) -> Dict[str, np.ndarray]:
        """计算原型查询的嵌入"""
        embeddings = {}
        
        for qtype, examples in self.prototypes.items():
            # 批量编码所有示例
            type_embeddings = self._encode_batch(examples)
            # 计算平均嵌入作为该类型的原型
            embeddings[qtype] = np.mean(type_embeddings, axis=0)
            # 归一化嵌入
            embeddings[qtype] = embeddings[qtype] / np.linalg.norm(embeddings[qtype])
            
        return embeddings
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码文本"""
        # 添加前缀以获得更好的句子嵌入
        texts = [f"query: {text}" for text in texts]
        
        # 分词
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        # 获取嵌入
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 平均池化获取句子嵌入
        attention_mask = inputs['attention_mask']
        embeddings = outputs.last_hidden_state
        
        # 平均池化 (考虑注意力掩码)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        # 转换为numpy数组
        return embeddings.cpu().numpy()
    
    def _encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        return self._encode_batch([text])[0]
    
    def _calculate_complexity(self, text: str) -> float:
        """计算查询复杂度
        
        基于词汇多样性、句法结构和长度
        """
        words = text.lower().split()
        
        if not words:
            return 0.0
        
        # 词汇多样性 (0-1)
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words)
        
        # 长度因子 (0-1)
        max_expected_length = 20  # 预期最大查询长度
        length_factor = min(len(words) / max_expected_length, 1.0)
        
        # 句法复杂度估计 (简单启发式)
        syntax_complexity = 0.0
        if ',' in text:
            syntax_complexity += 0.2
        if 'and' in text or 'or' in text:
            syntax_complexity += 0.2
        if '?' in text:
            syntax_complexity += 0.1
        if any(w in text for w in ['if', 'when', 'while', 'because', 'although']):
            syntax_complexity += 0.3
        if any(w in text for w in ['not', 'except', 'without', 'unless']):
            syntax_complexity += 0.2
            
        # 组合因子
        complexity = (lexical_diversity * 0.4 + length_factor * 0.3 + syntax_complexity * 0.3)
        return complexity
    
    def _combine_classifications(self, rule_based: str, embedding_based: str, confidence: float) -> str:
        """结合基于规则和基于嵌入的分类"""
        # 如果嵌入分类的置信度高，则使用嵌入分类
        if confidence > 0.6:
            return embedding_based
        
        # 如果规则分类是实体，优先使用规则分类
        if rule_based == 'entity':
            return rule_based
        
        # 否则使用嵌入分类
        return embedding_based
    
    def analyze_query(self, query: Query) -> EnhancedQueryFeatures:
        """分析查询特征"""
        # 获取基于规则的基本特征
        base_features = self.base_analyzer.analyze_query(query)
        
        # 获取查询嵌入
        query_embedding = self._encode_single(query.text)
        
        # 计算与原型的相似度
        similarities = {}
        for qtype, prototype_emb in self.prototype_embeddings.items():
            # 计算余弦相似度
            similarity = np.dot(query_embedding, prototype_emb)
            similarities[qtype] = similarity
        
        # 基于嵌入相似度确定最可能的类型
        embedding_type = max(similarities.items(), key=lambda x: x[1])[0]
        
        # 计算置信度分数 (归一化相似度)
        total_sim = sum(max(0, sim) for sim in similarities.values())
        confidence = similarities[embedding_type] / total_sim if total_sim > 0 else 0.5
        
        # 计算查询复杂度
        complexity = self._calculate_complexity(query.text)
        
        # 结合基于规则和基于嵌入的分类
        final_type = self._combine_classifications(
            base_features.query_type.value, 
            embedding_type, 
            confidence
        )
        
        # 创建增强特征
        return EnhancedQueryFeatures(
            query_id=query.query_id,
            query_text=query.text,
            rule_based_type=base_features.query_type,
            embedding_based_type=embedding_type,
            final_type=final_type,
            confidence=confidence,
            complexity=complexity,
            length=base_features.length,
            word_count=base_features.word_count,
            is_question=base_features.is_question,
            has_numbers=base_features.has_numbers,
            has_entities=base_features.has_entities,
            complexity_level=base_features.complexity_level,
            domain_hint=base_features.domain_hint
        )
    
    def batch_analyze(self, queries: List[Query]) -> List[EnhancedQueryFeatures]:
        """批量分析查询"""
        return [self.analyze_query(query) for query in queries]


def create_enhanced_query_analyzer(config: Dict[str, Any] = None) -> EnhancedQueryAnalyzer:
    """创建增强查询分析器的工厂函数"""
    return EnhancedQueryAnalyzer(config=config)