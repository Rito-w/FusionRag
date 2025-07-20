"""查询分析器
分析查询特征，用于自适应路由决策
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from ..utils.interfaces import Query, QueryType

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import spacy
    from spacy.tokens import Doc
except ImportError:
    print("请安装必要的依赖: pip install transformers torch scikit-learn spacy")
    print("并下载spaCy模型: python -m spacy download zh_core_web_sm en_core_web_sm")


@dataclass
class QueryFeatures:
    """查询特征数据模型"""
    query_id: str
    query_text: str
    query_type: QueryType
    length: int
    token_count: int
    keyword_count: int
    entity_count: int
    has_numbers: bool
    has_special_chars: bool
    avg_word_length: float
    is_question: bool
    language: str
    complexity_score: float
    specificity_score: float
    embedding: Optional[np.ndarray] = None
    entities: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = self.__dict__.copy()
        # 将不可序列化的对象转换为可序列化形式
        if self.embedding is not None:
            result['embedding'] = self.embedding.tolist()
        result['query_type'] = self.query_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryFeatures':
        """从字典创建实例"""
        # 转换查询类型
        if 'query_type' in data and isinstance(data['query_type'], str):
            data['query_type'] = QueryType(data['query_type'])
        
        # 转换嵌入向量
        if 'embedding' in data and isinstance(data['embedding'], list):
            data['embedding'] = np.array(data['embedding'])
        
        return cls(**data)


class QueryAnalyzer:
    """查询分析器
    
    分析查询特征，用于自适应路由决策
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 语言模型配置
        self.semantic_model_name = self.config.get('semantic_model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.tokenizer = None
        self.model = None
        
        # NLP模型配置
        self.spacy_model_name = self.config.get('spacy_model_name', 'en_core_web_sm')  # 改为英文模型
        self.nlp = None
        
        # 查询类型阈值
        self.keyword_query_threshold = self.config.get('keyword_query_threshold', 0.7)
        self.semantic_query_threshold = self.config.get('semantic_query_threshold', 0.6)
        self.entity_query_threshold = self.config.get('entity_query_threshold', 0.5)
        
        # 特征提取配置
        self.min_keyword_length = self.config.get('min_keyword_length', 2)
        self.max_keywords = self.config.get('max_keywords', 10)
        self.stopwords = set(self.config.get('stopwords', []))
        
        # 缓存
        self.query_cache = {}
    
    def _load_models(self) -> None:
        """加载必要的模型"""
        # 加载语义模型
        if self.tokenizer is None or self.model is None:
            print(f"加载语义模型: {self.semantic_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.semantic_model_name)
            self.model = AutoModel.from_pretrained(self.semantic_model_name)
            self.model.eval()
            
            # 如果有GPU则使用
            if torch.cuda.is_available():
                self.model = self.model.cuda()
        
        # 加载NLP模型
        if self.nlp is None:
            try:
                print(f"加载NLP模型: {self.spacy_model_name}")
                self.nlp = spacy.load(self.spacy_model_name)
            except OSError:
                # 如果模型不存在，尝试下载
                import subprocess
                print(f"下载NLP模型: {self.spacy_model_name}")
                subprocess.run(["python", "-m", "spacy", "download", self.spacy_model_name])
                self.nlp = spacy.load(self.spacy_model_name)
    
    def _encode_text(self, text: str) -> np.ndarray:
        """使用语义模型编码文本"""
        self._load_models()
        
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        # 如果有GPU则使用
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用平均池化获取句子表示
            attention_mask = inputs['attention_mask']
            embedding = self._mean_pooling(outputs.last_hidden_state, attention_mask)
            return embedding.cpu().numpy()[0]
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """平均池化获取句子表示"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _extract_entities(self, text: str) -> List[str]:
        """提取实体"""
        self._load_models()
        
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents]
        return entities
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        self._load_models()
        
        doc = self.nlp(text)
        
        # 过滤停用词和短词
        keywords = []
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                not token.is_space and
                len(token.text) >= self.min_keyword_length):
                keywords.append(token.text.lower())
        
        # 如果有自定义停用词，进一步过滤
        if self.stopwords:
            keywords = [kw for kw in keywords if kw not in self.stopwords]
        
        # 限制关键词数量
        return keywords[:self.max_keywords]
    
    def _calculate_complexity(self, doc: Doc) -> float:
        """计算查询复杂度分数"""
        # 基于句法树深度、从句数量等计算复杂度
        depth = 0
        for token in doc:
            d = 1
            current = token
            while current.head != current:
                current = current.head
                d += 1
            depth = max(depth, d)
        
        # 计算从句数量
        clause_count = len([t for t in doc if t.dep_ in ('ccomp', 'xcomp', 'advcl', 'acl')])
        
        # 归一化分数 (0-1)
        complexity = min(1.0, (depth * 0.1 + clause_count * 0.2))
        return complexity
    
    def _calculate_specificity(self, doc: Doc, entities: List[str], keywords: List[str]) -> float:
        """计算查询特异性分数"""
        # 基于实体数量、专有名词比例、数字出现等计算特异性
        entity_ratio = len(entities) / max(1, len(doc))
        proper_noun_ratio = len([t for t in doc if t.pos_ == 'PROPN']) / max(1, len(doc))
        keyword_ratio = len(keywords) / max(1, len(doc))
        has_numbers = any(t.like_num for t in doc)
        
        # 归一化分数 (0-1)
        specificity = min(1.0, (entity_ratio * 0.3 + proper_noun_ratio * 0.3 + 
                              keyword_ratio * 0.2 + (0.2 if has_numbers else 0)))
        return specificity
    
    def _detect_language(self, text: str) -> str:
        """检测查询语言"""
        # 简单语言检测，可以根据需要扩展
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        elif re.search(r'[\u3040-\u30ff]', text):
            return 'ja'
        elif re.search(r'[\u1100-\u11ff\uac00-\ud7af]', text):
            return 'ko'
        else:
            return 'en'
    
    def _is_question(self, text: str) -> bool:
        """判断是否为问句"""
        # 检查问号
        if any(q in text for q in ['?', '？']):
            return True
        
        # 检查常见问句开头词
        question_starters = {
            'en': ['what', 'who', 'when', 'where', 'why', 'how', 'is', 'are', 'do', 'does', 'can', 'could', 'would', 'will'],
            'zh': ['什么', '谁', '何时', '何地', '为何', '为什么', '怎么', '怎样', '如何', '是否', '能否', '可否']
        }
        
        lang = self._detect_language(text)
        starters = question_starters.get(lang, question_starters['en'])
        
        # 检查文本是否以问句开头词开始
        text_lower = text.lower()
        for starter in starters:
            if text_lower.startswith(starter):
                return True
        
        return False
    
    def _classify_query_type(self, features: QueryFeatures) -> QueryType:
        """根据查询特征分类查询类型"""
        # 基于特征计算各类型的分数
        keyword_score = features.keyword_count / max(1, features.token_count)
        entity_score = features.entity_count / max(1, features.token_count)
        
        # 语义查询的特征：问句、较长、复杂度高
        semantic_indicators = [
            features.is_question,
            features.complexity_score > 0.5,
            features.length > 10,
            features.avg_word_length > 3
        ]
        semantic_score = sum(1 for ind in semantic_indicators if ind) / len(semantic_indicators)
        
        # 确定查询类型
        if entity_score >= self.entity_query_threshold:
            return QueryType.ENTITY
        elif keyword_score >= self.keyword_query_threshold:
            return QueryType.KEYWORD
        elif semantic_score >= self.semantic_query_threshold:
            return QueryType.SEMANTIC
        else:
            # 默认为混合类型
            return QueryType.HYBRID
    
    def analyze_query(self, query: Query) -> QueryFeatures:
        """分析查询，提取特征"""
        # 检查缓存
        if query.query_id in self.query_cache:
            return self.query_cache[query.query_id]
        
        self._load_models()
        
        # 基本特征
        query_text = query.text
        length = len(query_text)
        
        # NLP处理
        doc = self.nlp(query_text)
        token_count = len(doc)
        
        # 提取实体和关键词
        entities = self._extract_entities(query_text)
        keywords = self._extract_keywords(query_text)
        
        # 计算特征
        has_numbers = any(char.isdigit() for char in query_text)
        has_special_chars = bool(re.search(r'[^\w\s]', query_text))
        avg_word_length = sum(len(token.text) for token in doc) / max(1, token_count)
        is_question = self._is_question(query_text)
        language = self._detect_language(query_text)
        
        # 计算复杂度和特异性分数
        complexity_score = self._calculate_complexity(doc)
        specificity_score = self._calculate_specificity(doc, entities, keywords)
        
        # 编码查询
        embedding = self._encode_text(query_text)
        
        # 创建特征对象
        features = QueryFeatures(
            query_id=query.query_id,
            query_text=query_text,
            query_type=QueryType.UNKNOWN,  # 先设为未知，后面分类
            length=length,
            token_count=token_count,
            keyword_count=len(keywords),
            entity_count=len(entities),
            has_numbers=has_numbers,
            has_special_chars=has_special_chars,
            avg_word_length=avg_word_length,
            is_question=is_question,
            language=language,
            complexity_score=complexity_score,
            specificity_score=specificity_score,
            embedding=embedding,
            entities=entities,
            keywords=keywords
        )
        
        # 分类查询类型
        features.query_type = self._classify_query_type(features)
        
        # 缓存结果
        self.query_cache[query.query_id] = features
        
        return features
    
    def batch_analyze(self, queries: List[Query]) -> List[QueryFeatures]:
        """批量分析查询"""
        return [self.analyze_query(query) for query in queries]
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self.query_cache = {}


def create_query_analyzer(config: Dict[str, Any] = None) -> QueryAnalyzer:
    """创建查询分析器的工厂函数"""
    return QueryAnalyzer(config=config)