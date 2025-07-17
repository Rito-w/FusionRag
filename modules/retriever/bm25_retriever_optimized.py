"""
BM25检索器优化版本
基于jieba分词的BM25算法，支持中文文本检索
优化了缓存、并行处理和内存使用
"""

import json
import pickle
import math
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from functools import lru_cache
import jieba
import re
import os
import concurrent.futures
import numpy as np

try:
    from pyserini.search import SimpleSearcher
    PYSERINI_AVAILABLE = True
except ImportError:
    PYSERINI_AVAILABLE = False

from ..utils.interfaces import BaseRetriever, Document, Query, RetrievalResult
from ..utils.common import FileUtils, TextProcessor


class OptimizedBM25Retriever(BaseRetriever):
    """优化的BM25检索器"""
    
    def __init__(self, name: str = "bm25", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        
        # BM25参数
        self.k1 = self.config.get('k1', 1.2)
        self.b = self.config.get('b', 0.75)
        self.max_cache_size = self.config.get('max_cache_size', 1000)
        
        # 索引数据
        self.documents: List[Document] = []
        self.doc_frequencies = defaultdict(int)  # 词在多少个文档中出现
        self.idf_scores = {}  # IDF分数
        self.doc_lengths = np.array([])  # 每个文档的长度
        self.avg_doc_length = 0.0  # 平均文档长度
        self.vocabulary = set()  # 词汇表
        
        # 优化的索引结构
        self.inverted_index = defaultdict(list)  # 倒排索引
        self.doc_vectors = {}  # 文档向量表示
        
        # 缓存
        self._text_cache = {}
        self._query_cache = {}
        
        # 初始化jieba（一次性）
        jieba.initialize()
    
    @lru_cache(maxsize=10000)
    def _preprocess_text_cached(self, text: str) -> Tuple[str, ...]:
        """带缓存的文本预处理"""
        # 文本清理
        text = re.sub(r'[^\w\s]', ' ', text)
        text = TextProcessor.normalize_text(text)
        
        # jieba分词
        tokens = list(jieba.cut(text))
        
        # 过滤短词和停用词
        tokens = [token.strip() for token in tokens if len(token.strip()) > 1]
        
        return tuple(tokens)  # 返回元组以便缓存
    
    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理和分词"""
        return list(self._preprocess_text_cached(text))
    
    def _build_inverted_index(self, documents: List[Document]) -> None:
        """构建倒排索引"""
        print("构建倒排索引...")
        
        for doc_id, doc in enumerate(documents):
            full_text = f"{doc.title} {doc.text}"
            tokens = self._preprocess_text(full_text)
            
            # 计算词频
            term_freq = Counter(tokens)
            
            # 构建倒排索引
            for term, freq in term_freq.items():
                self.inverted_index[term].append((doc_id, freq))
                self.vocabulary.add(term)
            
            # 存储文档向量
            self.doc_vectors[doc_id] = term_freq
    
    def _calculate_idf_batch(self) -> None:
        """批量计算IDF分数"""
        print("计算IDF分数...")
        
        N = len(self.documents)
        
        # 并行计算IDF
        def calculate_single_idf(term_data):
            term, postings = term_data
            df = len(postings)
            idf = math.log((N - df + 0.5) / (df + 0.5))
            return term, idf
        
        # 使用线程池并行计算
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            idf_results = list(executor.map(calculate_single_idf, self.inverted_index.items()))
        
        self.idf_scores = dict(idf_results)
    
    def build_index(self, documents: List[Document]) -> None:
        """构建BM25索引"""
        print(f"开始构建优化BM25索引，文档数量: {len(documents)}")
        
        self.documents = documents
        
        # 清空之前的索引
        self.inverted_index.clear()
        self.doc_vectors.clear()
        self.vocabulary.clear()
        
        # 构建倒排索引
        self._build_inverted_index(documents)
        
        # 计算文档长度
        doc_lengths = []
        for doc_id in range(len(documents)):
            doc_length = sum(self.doc_vectors[doc_id].values())
            doc_lengths.append(doc_length)
        
        self.doc_lengths = np.array(doc_lengths)
        self.avg_doc_length = np.mean(self.doc_lengths)
        
        # 批量计算IDF
        self._calculate_idf_batch()
        
        print(f"索引构建完成，词汇表大小: {len(self.vocabulary)}")
    
    def _calculate_bm25_score_optimized(self, query_terms: List[str], doc_id: int) -> float:
        """优化的BM25分数计算"""
        if doc_id not in self.doc_vectors:
            return 0.0
        
        doc_vector = self.doc_vectors[doc_id]
        doc_length = self.doc_lengths[doc_id]
        
        score = 0.0
        
        # 向量化计算
        for term in query_terms:
            if term in doc_vector and term in self.idf_scores:
                tf = doc_vector[term]
                idf = self.idf_scores[term]
                
                # BM25公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                
                score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """搜索相关文档"""
        # 查询缓存
        cache_key = (query.text, top_k)
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        # 预处理查询
        query_terms = self._preprocess_text(query.text)
        
        if not query_terms:
            return []
        
        # 获取候选文档（通过倒排索引）
        candidate_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                for doc_id, _ in self.inverted_index[term]:
                    candidate_docs.add(doc_id)
        
        # 并行计算候选文档的分数
        def calculate_score(doc_id):
            score = self._calculate_bm25_score_optimized(query_terms, doc_id)
            return doc_id, score
        
        # 使用线程池并行计算分数
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            scores = list(executor.map(calculate_score, candidate_docs))
        
        # 排序并返回结果
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in scores[:top_k]:
            if score > 0:
                result = RetrievalResult(
                    doc_id=self.documents[doc_id].doc_id,
                    score=score,
                    doc=self.documents[doc_id]
                )
                results.append(result)
        
        # 缓存结果
        if len(self._query_cache) < self.max_cache_size:
            self._query_cache[cache_key] = results
        
        return results
    
    def save_index(self, save_path: str) -> None:
        """保存索引"""
        print(f"保存BM25索引到: {save_path}")
        
        index_data = {
            'documents': self.documents,
            'inverted_index': dict(self.inverted_index),
            'doc_vectors': self.doc_vectors,
            'idf_scores': self.idf_scores,
            'doc_lengths': self.doc_lengths.tolist(),
            'avg_doc_length': self.avg_doc_length,
            'vocabulary': list(self.vocabulary),
            'config': self.config
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(index_data, f)
    
    def load_index(self, load_path: str) -> bool:
        """加载索引"""
        if not Path(load_path).exists():
            return False
        
        try:
            print(f"加载BM25索引: {load_path}")
            
            with open(load_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.documents = index_data['documents']
            self.inverted_index = defaultdict(list, index_data['inverted_index'])
            self.doc_vectors = index_data['doc_vectors']
            self.idf_scores = index_data['idf_scores']
            self.doc_lengths = np.array(index_data['doc_lengths'])
            self.avg_doc_length = index_data['avg_doc_length']
            self.vocabulary = set(index_data['vocabulary'])
            
            print(f"索引加载完成，文档数量: {len(self.documents)}")
            return True
            
        except Exception as e:
            print(f"索引加载失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return {
            'documents': len(self.documents),
            'vocabulary_size': len(self.vocabulary),
            'avg_doc_length': self.avg_doc_length,
            'total_terms': sum(len(postings) for postings in self.inverted_index.values()),
            'cache_size': len(self._query_cache),
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用量（MB）"""
        import sys
        
        memory_usage = 0
        memory_usage += sys.getsizeof(self.documents)
        memory_usage += sys.getsizeof(self.inverted_index)
        memory_usage += sys.getsizeof(self.doc_vectors)
        memory_usage += sys.getsizeof(self.idf_scores)
        memory_usage += sys.getsizeof(self.doc_lengths)
        memory_usage += sys.getsizeof(self._query_cache)
        
        return memory_usage / 1024 / 1024  # 转换为MB
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._query_cache.clear()
        self._text_cache.clear()
        # 清空LRU缓存
        self._preprocess_text_cached.cache_clear()


# 保持向后兼容
class BM25Retriever(OptimizedBM25Retriever):
    """BM25检索器（向后兼容）"""
    pass


class PyseriniBM25Retriever(BaseRetriever):
    """基于Pyserini的BM25检索器"""
    
    def __init__(self, name: str = "pyserini_bm25", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        
        if not PYSERINI_AVAILABLE:
            raise ImportError("Pyserini not available. Install with: pip install pyserini")
        
        self.index_path = self.config.get('index_path', 'checkpoints/pyserini_index')
        self.searcher = None
        self.documents = {}
    
    def build_index(self, documents: List[Document]) -> None:
        """构建Pyserini索引"""
        print(f"构建Pyserini索引，文档数量: {len(documents)}")
        
        # 保存文档映射
        for doc in documents:
            self.documents[doc.doc_id] = doc
        
        # 这里需要实现Pyserini索引构建逻辑
        # 由于Pyserini索引构建较为复杂，这里提供框架
        print("⚠️  Pyserini索引构建需要额外实现")
    
    def search(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """使用Pyserini搜索"""
        if self.searcher is None:
            print("⚠️  Pyserini搜索器未初始化")
            return []
        
        # 这里需要实现Pyserini搜索逻辑
        return []