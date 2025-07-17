"""
BM25检索器实现
基于jieba分词的BM25算法，支持中文文本检索
"""

import json
import pickle
import math
from collections import defaultdict, Counter
from typing import List, Dict, Any
from pathlib import Path
import jieba
import re
import os
try:
    from pyserini.search import SimpleSearcher
    PYSERINI_AVAILABLE = True
except ImportError:
    PYSERINI_AVAILABLE = False

from ..utils.interfaces import BaseRetriever, Document, Query, RetrievalResult
from ..utils.common import FileUtils, TextProcessor


class BM25Retriever(BaseRetriever):
    """BM25检索器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        name = (config or {}).get('name', 'bm25')
        super().__init__(name, config or {})
        
        # BM25参数
        self.k1 = self.config.get('k1', 1.2)
        self.b = self.config.get('b', 0.75)
        
        # 索引数据
        self.documents: List[Document] = []
        self.doc_frequencies = defaultdict(int)  # 词在多少个文档中出现
        self.idf_scores = {}  # IDF分数
        self.doc_lengths = []  # 每个文档的长度
        self.avg_doc_length = 0  # 平均文档长度
        self.vocabulary = set()  # 词汇表
        
        # 预处理的文档token
        self.doc_tokens = []
    
    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理和分词"""
        # 文本清理
        text = re.sub(r'[^\w\s]', ' ', text)
        text = TextProcessor.normalize_text(text)
        
        # jieba分词
        tokens = list(jieba.cut(text))
        
        # 过滤短词和停用词
        tokens = [token.strip() for token in tokens if len(token.strip()) > 1]
        
        return tokens
    
    def build_index(self, documents: List[Document]) -> None:
        """构建BM25索引"""
        print(f"开始构建BM25索引，文档数量: {len(documents)}")
        
        self.documents = documents
        self.doc_tokens = []
        self.doc_lengths = []
        self.doc_frequencies = defaultdict(int)
        self.vocabulary = set()
        
        # 第一遍：分词和统计
        for doc in documents:
            # 合并标题和内容
            full_text = f"{doc.title} {doc.text}"
            tokens = self._preprocess_text(full_text)
            
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))
            
            # 统计词频
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_frequencies[token] += 1
                self.vocabulary.add(token)
        
        # 计算平均文档长度
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        
        # 计算IDF分数
        N = len(documents)
        for term in self.vocabulary:
            df = self.doc_frequencies[term]
            # 使用平滑的IDF公式
            idf = math.log((N - df + 0.5) / (df + 0.5))
            self.idf_scores[term] = max(idf, 0.01)  # 避免负IDF
        
        self.is_built = True
        print(f"BM25索引构建完成，词汇表大小: {len(self.vocabulary)}")
    
    def _compute_bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """计算单个文档的BM25分数"""
        doc_tokens = self.doc_tokens[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        
        # 计算文档中的词频
        doc_term_freq = Counter(doc_tokens)
        
        score = 0.0
        for term in query_tokens:
            if term not in self.vocabulary:
                continue
            
            # 词频
            tf = doc_term_freq.get(term, 0)
            if tf == 0:
                continue
            
            # IDF
            idf = self.idf_scores[term]
            
            # BM25公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def retrieve(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """检索相关文档"""
        if not self.is_built:
            raise RuntimeError("索引尚未构建，请先调用build_index方法")
        
        # 预处理查询
        query_tokens = self._preprocess_text(query.text)
        
        if not query_tokens:
            return []
        
        # 计算所有文档的BM25分数
        doc_scores = []
        for doc_idx in range(len(self.documents)):
            score = self._compute_bm25_score(query_tokens, doc_idx)
            doc_scores.append((doc_idx, score))
        
        # 按分数排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 生成检索结果
        results = []
        for rank, (doc_idx, score) in enumerate(doc_scores[:top_k]):
            if score > 0:  # 只返回有得分的文档
                result = RetrievalResult(
                    doc_id=self.documents[doc_idx].doc_id,
                    score=score,
                    document=self.documents[doc_idx],
                    retriever_name=self.name
                )
                results.append(result)
        
        return results
    
    def save_index(self, index_path: str) -> None:
        """保存索引到文件"""
        if not self.is_built:
            raise RuntimeError("索引尚未构建")
        
        index_data = {
            'documents': self.documents,
            'doc_frequencies': dict(self.doc_frequencies),
            'idf_scores': self.idf_scores,
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'vocabulary': list(self.vocabulary),
            'doc_tokens': self.doc_tokens,
            'config': self.config,
            'k1': self.k1,
            'b': self.b
        }
        
        FileUtils.save_pickle(index_data, index_path)
        print(f"BM25索引已保存到: {index_path}")
    
    def load_index(self, index_path: str) -> None:
        """从文件加载索引"""
        try:
            index_data = FileUtils.load_pickle(index_path)
            
            self.documents = index_data['documents']
            self.doc_frequencies = defaultdict(int, index_data['doc_frequencies'])
            self.idf_scores = index_data['idf_scores']
            self.doc_lengths = index_data['doc_lengths']
            self.avg_doc_length = index_data['avg_doc_length']
            self.vocabulary = set(index_data['vocabulary'])
            self.doc_tokens = index_data['doc_tokens']
            
            # 更新配置参数
            if 'k1' in index_data:
                self.k1 = index_data['k1']
            if 'b' in index_data:
                self.b = index_data['b']
            
            self.is_built = True
            print(f"BM25索引已从 {index_path} 加载完成")
            
        except Exception as e:
            raise RuntimeError(f"加载索引失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        if not self.is_built:
            return {}
        
        return {
            'document_count': len(self.documents),
            'vocabulary_size': len(self.vocabulary),
            'avg_doc_length': self.avg_doc_length,
            'total_tokens': sum(self.doc_lengths),
            'parameters': {
                'k1': self.k1,
                'b': self.b
            }
        }


class PyseriniBM25Retriever(BaseRetriever):
    """基于Pyserini官方实现的BM25检索器（英文）"""
    def __init__(self, name: str = "pyserini_bm25", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        if not PYSERINI_AVAILABLE:
            raise ImportError("Pyserini未安装，请先 pip install pyserini")
        self.index_dir = self.config.get('index_dir', 'checkpoints/retriever/pyserini_bm25_index')
        self.k1 = self.config.get('k1', 0.82)
        self.b = self.config.get('b', 0.68)
        self.searcher = None
        self.documents = []
        self.is_built = False

    def build_index(self, documents: List[Document]) -> None:
        """用Pyserini构建倒排索引（一次性，需Java环境）"""
        from pyserini.index import IndexReader, IndexWriter
        import json
        import shutil
        from tqdm import tqdm
        # 1. 保存文档为jsonl
        os.makedirs(self.index_dir, exist_ok=True)
        jsonl_path = os.path.join(self.index_dir, 'corpus.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                obj = {'id': doc.doc_id, 'contents': f"{doc.title} {doc.text}".strip()}
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        # 2. 构建Pyserini索引
        cmd = f"python -m pyserini.index.lucene --collection JsonCollection --input {self.index_dir} --index {self.index_dir}/index --generator DefaultLuceneDocumentGenerator --threads 4 --storePositions --storeDocvectors --storeRaw"
        print(f"[PyseriniBM25Retriever] 正在构建索引: {cmd}")
        os.system(cmd)
        self.documents = documents
        self.is_built = True
        print(f"[PyseriniBM25Retriever] 索引构建完成: {self.index_dir}/index")

    def load_index(self, index_dir: str = None) -> None:
        """加载Pyserini索引"""
        idx = index_dir or self.index_dir
        self.searcher = SimpleSearcher(f"{idx}/index")
        self.searcher.set_bm25(self.k1, self.b)
        self.is_built = True
        print(f"[PyseriniBM25Retriever] 已加载索引: {idx}/index")

    def retrieve(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        if not self.is_built:
            raise RuntimeError("Pyserini索引未构建或未加载")
        if self.searcher is None:
            self.load_index()
        hits = self.searcher.search(query.text, k=top_k)
        results = []
        for hit in hits:
            result = RetrievalResult(
                doc_id=hit.docid,
                score=hit.score,
                document=None,  # 可选：可查找原文档
                retriever_name=self.name
            )
            results.append(result)
        return results


def create_bm25_retriever(config: Dict[str, Any]) -> BM25Retriever:
    """创建BM25检索器的工厂函数"""
    return BM25Retriever(config=config)