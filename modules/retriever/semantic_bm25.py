"""语义增强BM25检索器
结合传统BM25和语义模型，提升检索效果
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter

from ..utils.interfaces import BaseRetriever, Document, Query, RetrievalResult
from ..utils.common import FileUtils, TextProcessor

try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("请安装必要的依赖: pip install rank_bm25 sentence-transformers scikit-learn")


class SemanticBM25(BaseRetriever):
    """语义增强BM25检索器
    
    结合传统BM25和语义模型，通过以下方式提升检索效果：
    1. 查询扩展：使用语义模型扩展原始查询
    2. 文档扩展：为文档添加语义相关的关键词
    3. 混合排序：结合BM25和语义相似度的分数
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        name = config.get('name', 'semantic_bm25') if config else 'semantic_bm25'
        super().__init__(name=name, config=config or {})
        
        # BM25参数
        self.bm25_k1 = self.config.get('bm25_k1', 1.5)
        self.bm25_b = self.config.get('bm25_b', 0.75)
        
        # 语义模型参数
        self.semantic_model_name = self.config.get('semantic_model_name', 'intfloat/e5-large-v2')
        self.semantic_weight = self.config.get('semantic_weight', 0.3)  # 语义分数权重
        
        # 查询扩展参数
        self.enable_query_expansion = self.config.get('enable_query_expansion', True)
        self.query_expansion_terms = self.config.get('query_expansion_terms', 3)
        self.query_expansion_weight = self.config.get('query_expansion_weight', 0.5)
        
        # 文档扩展参数
        self.enable_document_expansion = self.config.get('enable_document_expansion', True)
        self.document_expansion_terms = self.config.get('document_expansion_terms', 5)
        
        # 初始化变量
        self.documents = []
        self.document_texts = []
        self.document_expanded_texts = []
        self.bm25 = None
        self.model = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.is_built = False
        
        # 缓存路径
        self.cache_dir = self.config.get('cache_dir', 'cache')
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_semantic_model(self) -> None:
        """加载语义模型"""
        if self.model is None:
            print(f"加载语义模型: {self.semantic_model_name}")
            self.model = SentenceTransformer(self.semantic_model_name)
    
    def _encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """使用语义模型编码文本"""
        self._load_semantic_model()
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    
    def _build_tfidf(self) -> None:
        """构建TF-IDF向量化器和矩阵"""
        print("构建TF-IDF模型...")
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.document_texts)
    
    def _expand_query(self, query_text: str) -> str:
        """扩展查询，添加语义相关的关键词"""
        if not self.enable_query_expansion:
            return query_text
        
        # 编码查询
        query_embedding = self._encode_texts([query_text])[0]
        
        # 计算查询与所有文档的相似度
        doc_embeddings = self._encode_texts(self.document_texts)
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # 获取最相似的文档
        top_doc_indices = similarities.argsort()[-5:][::-1]
        
        # 从最相似的文档中提取关键词
        expansion_terms = set()
        for idx in top_doc_indices:
            doc_text = self.document_texts[idx]
            # 使用TF-IDF找出文档中的重要词
            doc_vector = self.tfidf_vectorizer.transform([doc_text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # 获取TF-IDF分数最高的词
            tfidf_scores = zip(feature_names, doc_vector.toarray()[0])
            sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
            
            # 添加到扩展词集合
            for term, score in sorted_scores[:self.query_expansion_terms]:
                if term not in query_text.lower().split():
                    expansion_terms.add(term)
            
            if len(expansion_terms) >= self.query_expansion_terms:
                break
        
        # 构建扩展查询
        expanded_query = query_text
        if expansion_terms:
            expansion_str = " " + " ".join(list(expansion_terms)[:self.query_expansion_terms])
            expanded_query += expansion_str
            print(f"查询扩展: '{query_text}' -> '{expanded_query}'")
        
        return expanded_query
    
    def _expand_documents(self) -> List[str]:
        """扩展文档，为每个文档添加语义相关的关键词"""
        if not self.enable_document_expansion:
            return self.document_texts
        
        print("扩展文档内容...")
        expanded_texts = []
        
        # 编码所有文档
        doc_embeddings = self._encode_texts(self.document_texts)
        
        # 计算文档间的相似度矩阵
        similarities = cosine_similarity(doc_embeddings)
        
        # 为每个文档添加相关词
        for i, doc_text in enumerate(self.document_texts):
            # 找出最相似的其他文档
            similar_docs = similarities[i].argsort()[-6:][::-1][1:]  # 排除自身
            
            # 从相似文档中提取关键词
            expansion_terms = set()
            for idx in similar_docs:
                similar_doc = self.document_texts[idx]
                
                # 使用TF-IDF找出文档中的重要词
                doc_vector = self.tfidf_vectorizer.transform([similar_doc])
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                
                # 获取TF-IDF分数最高的词
                tfidf_scores = zip(feature_names, doc_vector.toarray()[0])
                sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
                
                # 添加到扩展词集合
                for term, score in sorted_scores[:self.document_expansion_terms]:
                    if term not in doc_text.lower().split():
                        expansion_terms.add(term)
                
                if len(expansion_terms) >= self.document_expansion_terms:
                    break
            
            # 构建扩展文档
            expanded_text = doc_text
            if expansion_terms:
                expansion_str = " " + " ".join(list(expansion_terms)[:self.document_expansion_terms])
                expanded_text += expansion_str
            
            expanded_texts.append(expanded_text)
        
        return expanded_texts
    
    def build_index(self, documents: List[Document]) -> None:
        """构建索引"""
        print(f"构建语义增强BM25索引，文档数量: {len(documents)}")
        self.documents = documents
        
        # 准备文档文本
        self.document_texts = [TextProcessor.normalize_text(doc.text) for doc in documents]
        
        # 构建TF-IDF模型（用于查询和文档扩展）
        self._build_tfidf()
        
        # 文档扩展
        self.document_expanded_texts = self._expand_documents()
        
        # 对扩展后的文档进行分词
        tokenized_corpus = [doc.split() for doc in self.document_expanded_texts]
        
        # 构建BM25模型
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.bm25_k1, b=self.bm25_b)
        
        self.is_built = True
        print("语义增强BM25索引构建完成")
    
    def retrieve(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """检索相关文档"""
        if not self.is_built:
            raise RuntimeError("索引尚未构建，请先调用build_index方法")
        
        # 规范化查询文本
        query_text = TextProcessor.normalize_text(query.text)
        
        # 查询扩展
        expanded_query = self._expand_query(query_text)
        
        # 对查询进行分词
        tokenized_query = expanded_query.split()
        
        # BM25检索
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 语义检索（如果启用）
        if self.semantic_weight > 0:
            # 编码查询
            query_embedding = self._encode_texts([query_text])[0]
            
            # 编码文档（使用原始文档，不使用扩展文档）
            doc_embeddings = self._encode_texts(self.document_texts)
            
            # 计算余弦相似度
            semantic_scores = cosine_similarity([query_embedding], doc_embeddings)[0]
            
            # 归一化BM25分数和语义分数
            if np.max(bm25_scores) > 0:
                bm25_scores = bm25_scores / np.max(bm25_scores)
            
            # 混合分数
            combined_scores = (1 - self.semantic_weight) * bm25_scores + self.semantic_weight * semantic_scores
        else:
            combined_scores = bm25_scores
        
        # 获取top-k文档
        top_indices = combined_scores.argsort()[-top_k:][::-1]
        
        # 生成检索结果
        results = []
        for rank, idx in enumerate(top_indices):
            if combined_scores[idx] > 0:  # 只返回正分数的结果
                result = RetrievalResult(
                    doc_id=self.documents[idx].doc_id,
                    score=float(combined_scores[idx]),
                    document=self.documents[idx],
                    retriever_name=self.name
                )
                results.append(result)
        
        return results
    
    def save_index(self, index_path: str) -> None:
        """保存索引到文件"""
        if not self.is_built:
            raise RuntimeError("索引尚未构建")
        
        index_dir = Path(index_path).parent
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存数据
        data = {
            'documents': self.documents,
            'document_texts': self.document_texts,
            'document_expanded_texts': self.document_expanded_texts,
            'config': self.config,
            'bm25_k1': self.bm25_k1,
            'bm25_b': self.bm25_b,
            'semantic_model_name': self.semantic_model_name,
            'semantic_weight': self.semantic_weight,
            'enable_query_expansion': self.enable_query_expansion,
            'query_expansion_terms': self.query_expansion_terms,
            'enable_document_expansion': self.enable_document_expansion,
            'document_expansion_terms': self.document_expansion_terms
        }
        
        # 保存BM25模型
        bm25_path = os.path.join(index_dir, "bm25_model.pkl")
        with open(bm25_path, 'wb') as f:
            pickle.dump(self.bm25, f)
        
        # 保存TF-IDF模型
        tfidf_path = os.path.join(index_dir, "tfidf_model.pkl")
        with open(tfidf_path, 'wb') as f:
            pickle.dump((self.tfidf_vectorizer, self.tfidf_matrix), f)
        
        # 更新路径
        data['bm25_path'] = bm25_path
        data['tfidf_path'] = tfidf_path
        
        # 保存元数据
        FileUtils.save_pickle(data, index_path)
        print(f"语义增强BM25索引已保存到: {index_path}")
    
    def load_index(self, index_path: str) -> None:
        """从文件加载索引"""
        try:
            # 加载元数据
            data = FileUtils.load_pickle(index_path)
            
            # 恢复实例变量
            self.documents = data['documents']
            self.document_texts = data['document_texts']
            self.document_expanded_texts = data['document_expanded_texts']
            self.config = data.get('config', self.config)
            self.bm25_k1 = data.get('bm25_k1', self.bm25_k1)
            self.bm25_b = data.get('bm25_b', self.bm25_b)
            self.semantic_model_name = data.get('semantic_model_name', self.semantic_model_name)
            self.semantic_weight = data.get('semantic_weight', self.semantic_weight)
            self.enable_query_expansion = data.get('enable_query_expansion', self.enable_query_expansion)
            self.query_expansion_terms = data.get('query_expansion_terms', self.query_expansion_terms)
            self.enable_document_expansion = data.get('enable_document_expansion', self.enable_document_expansion)
            self.document_expansion_terms = data.get('document_expansion_terms', self.document_expansion_terms)
            
            # 加载BM25模型
            bm25_path = data.get('bm25_path')
            if bm25_path and os.path.exists(bm25_path):
                with open(bm25_path, 'rb') as f:
                    self.bm25 = pickle.load(f)
            else:
                # 如果BM25文件不存在，重新构建
                tokenized_corpus = [doc.split() for doc in self.document_expanded_texts]
                self.bm25 = BM25Okapi(tokenized_corpus, k1=self.bm25_k1, b=self.bm25_b)
            
            # 加载TF-IDF模型
            tfidf_path = data.get('tfidf_path')
            if tfidf_path and os.path.exists(tfidf_path):
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer, self.tfidf_matrix = pickle.load(f)
            else:
                # 如果TF-IDF文件不存在，重新构建
                self._build_tfidf()
            
            self.is_built = True
            print(f"语义增强BM25索引已从 {index_path} 加载完成")
            
        except Exception as e:
            raise RuntimeError(f"加载索引失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        if not self.is_built:
            return {}
        
        stats = {
            'document_count': len(self.documents),
            'bm25_k1': self.bm25_k1,
            'bm25_b': self.bm25_b,
            'semantic_model_name': self.semantic_model_name,
            'semantic_weight': self.semantic_weight,
            'enable_query_expansion': self.enable_query_expansion,
            'query_expansion_terms': self.query_expansion_terms,
            'enable_document_expansion': self.enable_document_expansion,
            'document_expansion_terms': self.document_expansion_terms,
            'vocabulary_size': len(self.bm25.idf) if self.bm25 else 0
        }
        
        return stats


def create_semantic_bm25(config: Dict[str, Any]) -> SemanticBM25:
    """创建语义增强BM25检索器的工厂函数"""
    return SemanticBM25(config=config)