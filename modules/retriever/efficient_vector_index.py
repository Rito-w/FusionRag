"""高效向量索引实现
支持HNSW和IVF索引结构，提供更高效的向量检索
"""

import os
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..utils.interfaces import BaseRetriever, Document, Query, RetrievalResult
from ..utils.common import FileUtils, TextProcessor
from .dense_retriever import DenseRetriever


class EfficientVectorIndex(DenseRetriever):
    """高效向量索引实现，支持HNSW和IVF索引结构"""
    
    def __init__(self, name: str = "efficient_vector", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        
        # 索引类型配置
        self.index_type = self.config.get('index_type', 'hnsw')  # 'hnsw' 或 'ivf'
        
        # HNSW参数
        self.hnsw_m = self.config.get('hnsw_m', 16)  # 最大出边数
        self.hnsw_ef_construction = self.config.get('hnsw_ef_construction', 200)  # 构建时搜索深度
        self.hnsw_ef_search = self.config.get('hnsw_ef_search', 128)  # 搜索时搜索深度
        
        # IVF参数
        self.ivf_nlist = self.config.get('ivf_nlist', 100)  # 聚类中心数
        self.ivf_nprobe = self.config.get('ivf_nprobe', 10)  # 查询时检查的聚类数
        
        # 自动选择参数
        self.auto_index_selection = self.config.get('auto_index_selection', True)  # 是否自动选择索引类型
        self.large_dataset_threshold = self.config.get('large_dataset_threshold', 1000000)  # 大数据集阈值
    
    def _build_faiss_index(self) -> None:
        """构建FAISS索引，根据配置选择HNSW或IVF"""
        print(f"构建FAISS索引，类型: {self.index_type}...")
        
        # 自动选择索引类型
        if self.auto_index_selection and len(self.documents) > 0:
            self._auto_select_index_type()
        
        # 根据索引类型构建索引
        if self.index_type == 'hnsw':
            self._build_hnsw_index()
        elif self.index_type == 'ivf':
            self._build_ivf_index()
        else:
            # 默认使用FlatIP
            print(f"未知索引类型 {self.index_type}，使用默认FlatIP索引")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(self.document_embeddings)
    
    def _auto_select_index_type(self) -> None:
        """自动选择索引类型"""
        doc_count = len(self.documents)
        
        if doc_count >= self.large_dataset_threshold:
            # 大数据集使用IVF
            self.index_type = 'ivf'
            # 自动调整IVF参数
            self.ivf_nlist = min(4 * int(np.sqrt(doc_count)), 1024)  # 经验公式
            print(f"自动选择IVF索引，文档数: {doc_count}, nlist: {self.ivf_nlist}")
        else:
            # 小数据集使用HNSW
            self.index_type = 'hnsw'
            print(f"自动选择HNSW索引，文档数: {doc_count}")
    
    def _build_hnsw_index(self) -> None:
        """构建HNSW索引"""
        print(f"构建HNSW索引，参数: M={self.hnsw_m}, efConstruction={self.hnsw_ef_construction}")
        
        # 创建HNSW索引
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        
        # 设置构建参数
        self.index.hnsw.efConstruction = self.hnsw_ef_construction
        
        # 添加向量到索引
        self.index.add(self.document_embeddings)
        
        # 设置搜索参数
        self.index.hnsw.efSearch = self.hnsw_ef_search
    
    def _build_ivf_index(self) -> None:
        """构建IVF索引"""
        print(f"构建IVF索引，参数: nlist={self.ivf_nlist}, nprobe={self.ivf_nprobe}")
        
        # 创建量化器
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        
        # 创建IVF索引
        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
        
        # 训练索引（必须步骤）
        if not self.index.is_trained and len(self.document_embeddings) > 0:
            print("训练IVF索引...")
            self.index.train(self.document_embeddings)
        
        # 添加向量到索引
        self.index.add(self.document_embeddings)
        
        # 设置搜索参数
        self.index.nprobe = self.ivf_nprobe
    
    def retrieve(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """检索相关文档"""
        if not self.is_built:
            raise RuntimeError("索引尚未构建，请先调用build_index方法")
        
        if not self.model or not self.index:
            raise RuntimeError("模型或索引未准备就绪")
        
        # 编码查询
        query_text = TextProcessor.normalize_text(query.text)
        query_embedding = self._encode_texts([query_text], show_progress=False)
        
        # 搜索最相似的文档
        scores, indices = self.index.search(query_embedding, top_k)
        
        # 生成检索结果
        results = []
        for rank, (score, doc_idx) in enumerate(zip(scores[0], indices[0])):
            if doc_idx >= 0 and score > 0:  # 有效的索引和正分数
                result = RetrievalResult(
                    doc_id=self.documents[doc_idx].doc_id,
                    score=float(score),
                    document=self.documents[doc_idx],
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
        
        # 保存FAISS索引
        faiss_path = index_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(faiss_path))
        
        # 保存其他数据
        metadata = {
            'documents': self.documents,
            'document_embeddings': self.document_embeddings,
            'config': self.config,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'faiss_path': str(faiss_path),
            'index_type': self.index_type,
            'hnsw_m': self.hnsw_m,
            'hnsw_ef_construction': self.hnsw_ef_construction,
            'hnsw_ef_search': self.hnsw_ef_search,
            'ivf_nlist': self.ivf_nlist,
            'ivf_nprobe': self.ivf_nprobe
        }
        
        FileUtils.save_pickle(metadata, index_path)
        print(f"高效向量索引已保存到: {index_path}")
    
    def load_index(self, index_path: str) -> None:
        """从文件加载索引"""
        try:
            # 加载元数据
            metadata = FileUtils.load_pickle(index_path)
            
            self.documents = metadata['documents']
            self.document_embeddings = metadata['document_embeddings']
            
            # 更新配置
            if 'model_name' in metadata:
                if metadata['model_name'] != self.model_name:
                    print(f"警告: 索引使用的模型({metadata['model_name']})与当前配置({self.model_name})不同")
            
            if 'embedding_dim' in metadata:
                self.embedding_dim = metadata['embedding_dim']
            
            # 加载索引类型和参数
            if 'index_type' in metadata:
                self.index_type = metadata['index_type']
            
            if 'hnsw_m' in metadata:
                self.hnsw_m = metadata['hnsw_m']
            
            if 'hnsw_ef_construction' in metadata:
                self.hnsw_ef_construction = metadata['hnsw_ef_construction']
            
            if 'hnsw_ef_search' in metadata:
                self.hnsw_ef_search = metadata['hnsw_ef_search']
            
            if 'ivf_nlist' in metadata:
                self.ivf_nlist = metadata['ivf_nlist']
            
            if 'ivf_nprobe' in metadata:
                self.ivf_nprobe = metadata['ivf_nprobe']
            
            # 加载FAISS索引
            faiss_path = metadata.get('faiss_path')
            if faiss_path and Path(faiss_path).exists():
                self.index = faiss.read_index(faiss_path)
                
                # 设置搜索参数
                if self.index_type == 'hnsw':
                    self.index.hnsw.efSearch = self.hnsw_ef_search
                elif self.index_type == 'ivf':
                    self.index.nprobe = self.ivf_nprobe
            else:
                # 如果FAISS文件不存在，重新构建索引
                print("FAISS索引文件不存在，重新构建...")
                self._build_faiss_index()
            
            self.is_built = True
            print(f"高效向量索引已从 {index_path} 加载完成，索引类型: {self.index_type}")
            
        except Exception as e:
            raise RuntimeError(f"加载索引失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        if not self.is_built:
            return {}
        
        stats = {
            'document_count': len(self.documents),
            'embedding_dim': self.embedding_dim,
            'model_name': self.model_name,
            'index_type': self.index_type,
            'index_size': self.index.ntotal if self.index else 0,
            'memory_usage_mb': (
                self.document_embeddings.nbytes / (1024 * 1024) 
                if self.document_embeddings is not None else 0
            )
        }
        
        # 添加特定索引类型的统计信息
        if self.index_type == 'hnsw':
            stats.update({
                'hnsw_m': self.hnsw_m,
                'hnsw_ef_construction': self.hnsw_ef_construction,
                'hnsw_ef_search': self.hnsw_ef_search
            })
        elif self.index_type == 'ivf':
            stats.update({
                'ivf_nlist': self.ivf_nlist,
                'ivf_nprobe': self.ivf_nprobe
            })
        
        return stats


def create_efficient_vector_index(config: Dict[str, Any]) -> EfficientVectorIndex:
    """创建高效向量索引的工厂函数"""
    return EfficientVectorIndex(config=config)