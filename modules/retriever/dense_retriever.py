"""
向量检索器实现
基于sentence-transformers和FAISS的密集向量检索
"""

import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pickle
import hashlib
import json

from ..utils.interfaces import BaseRetriever, Document, Query, RetrievalResult
from ..utils.common import FileUtils, TextProcessor


class DenseRetriever(BaseRetriever):
    """密集向量检索器"""
    
    def __init__(self, name: str = "dense", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        
        # 模型配置
        self.model_name = self.config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_dim = self.config.get('embedding_dim', 384)
        self.batch_size = self.config.get('batch_size', 32)
        self.max_length = self.config.get('max_length', 512)
        self.normalize_embeddings = self.config.get('normalize_embeddings', True)  # 是否L2归一化，官方推荐
        
        # 模型和索引
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []
        self.document_embeddings: Optional[np.ndarray] = None

        # 缓存相关
        self.documents_hash: Optional[str] = None
        self.embeddings_cache_path: Optional[str] = None

        # 初始化模型
        self._load_model()
    
    def _load_model(self) -> None:
        """加载sentence-transformers模型"""
        try:
            print(f"加载模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # 验证嵌入维度
            test_embedding = self.model.encode(["测试文本"], show_progress_bar=False)
            actual_dim = test_embedding.shape[1]
            
            if actual_dim != self.embedding_dim:
                print(f"警告: 配置的嵌入维度({self.embedding_dim})与实际维度({actual_dim})不匹配，使用实际维度")
                self.embedding_dim = actual_dim
            
            print(f"模型加载完成，嵌入维度: {self.embedding_dim}")
            
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    
    def _prepare_text(self, document: Document) -> str:
        """准备文档文本用于编码"""
        # 合并标题和内容
        text = f"{document.title} {document.text}".strip()
        
        # 文本预处理
        text = TextProcessor.normalize_text(text)
        
        # 截断长文本
        text = TextProcessor.truncate_text(text, self.max_length)
        
        return text
    
    def _encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """批量编码文本（支持L2归一化，官方BEIR推荐）"""
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings  # 由配置控制
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"文本编码失败: {e}")
    
    def build_index(self, documents: List[Document], force_rebuild: bool = False, dataset_name: str = None) -> None:
        """构建向量索引"""
        print(f"开始构建向量索引，文档数量: {len(documents)}")

        if not self.model:
            raise RuntimeError("模型未加载")

        self.documents = documents

        # 从文档路径推断数据集名称
        if dataset_name is None:
            dataset_name = self._infer_dataset_name(documents)

        # 检查是否可以使用缓存的向量
        print(f"🔍 检查向量缓存 (force_rebuild={force_rebuild}, dataset={dataset_name})")
        if not force_rebuild and self._can_use_cached_embeddings(documents, dataset_name):
            print("✅ 使用缓存的文档向量，跳过重新编码")
            self._build_faiss_index()
        else:
            # 需要重新编码
            print("🔄 需要重新编码文档向量")
            self._encode_and_build_index(documents, dataset_name)

        self.is_built = True
        print(f"向量索引构建完成，索引大小: {self.index.ntotal}")

    def _infer_dataset_name(self, documents: List[Document]) -> str:
        """从文档推断数据集名称"""
        if not documents:
            return "unknown"

        # 首先检查文档的metadata中是否有数据集信息
        first_doc = documents[0]
        if hasattr(first_doc, 'metadata') and first_doc.metadata:
            if 'dataset' in first_doc.metadata:
                dataset_name = first_doc.metadata['dataset']
                print(f"📋 从metadata推断数据集名称: {dataset_name}")
                return dataset_name

        # 从文档ID推断
        first_doc_id = first_doc.doc_id
        print(f"🔍 从文档ID推断数据集: {first_doc_id}")

        if "MED-" in first_doc_id:
            return "nfcorpus"
        elif "covid" in first_doc_id.lower():
            return "trec-covid"
        elif "nq" in first_doc_id.lower():
            return "natural_questions"
        elif first_doc_id.startswith("test-") and any(x in first_doc_id for x in ["pro", "con"]):
            return "arguana"
        elif first_doc_id.isdigit():
            # 纯数字ID，可能是scifact, fiqa, quora等
            if len(documents) > 1000:  # 根据数据集大小推断
                if len(documents) > 100000:
                    return "quora"  # quora有522k文档
                elif len(documents) > 50000:
                    return "fiqa"  # fiqa有57k文档
                elif len(documents) > 20000:
                    return "scidocs"  # scidocs有25k文档
                else:
                    return "scifact"  # scifact有5k文档
            else:
                return "scifact"
        elif len(first_doc_id) == 40 and all(c in '0123456789abcdef' for c in first_doc_id):
            # 40位十六进制哈希，可能是scidocs
            if len(documents) > 20000:
                return "scidocs"
            else:
                return "unknown"
        elif len(first_doc_id) == 8 and first_doc_id.isalnum():
            # 8位字母数字ID，可能是trec-covid
            if len(documents) > 100000:
                return "trec-covid"
            else:
                return "unknown"
        elif len(documents) > 500000:
            return "quora"  # quora是最大的数据集
        else:
            print(f"⚠️ 无法推断数据集名称，文档ID: {first_doc_id}, 文档数量: {len(documents)}")
            return "unknown"

    def _calculate_documents_hash(self, documents: List[Document]) -> str:
        """计算文档集合的哈希值"""
        # 创建文档内容的哈希
        doc_contents = []
        for doc in documents:
            content = f"{doc.doc_id}|{doc.title}|{doc.text}"
            doc_contents.append(content)

        # 计算整体哈希
        combined_content = "\n".join(sorted(doc_contents))
        return hashlib.md5(combined_content.encode('utf-8')).hexdigest()

    def _get_embeddings_cache_path(self, dataset_name: str = None) -> str:
        """获取向量缓存文件路径"""
        cache_dir = Path("checkpoints/embeddings_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 从文档路径推断数据集名称（如果没有提供）
        if dataset_name is None:
            dataset_name = "unknown"
            # 可以从配置或其他地方获取数据集名称

        # 清理模型名称用于文件名
        model_name_safe = self.model_name.replace("/", "_").replace(":", "_").replace("-", "_")

        # 简单直观的命名：数据集_模型名称.pkl
        cache_filename = f"{dataset_name}_{model_name_safe}_embeddings.pkl"

        return str(cache_dir / cache_filename)

    def _can_use_cached_embeddings(self, documents: List[Document], dataset_name: str) -> bool:
        """检查是否可以使用缓存的向量"""
        # 计算当前文档集合的哈希（用于验证）
        current_hash = self._calculate_documents_hash(documents)

        # 检查内存中的缓存
        if (self.document_embeddings is not None and
            self.documents_hash == current_hash and
            len(self.documents) == len(documents)):
            print("✅ 使用内存中的向量缓存")
            return True

        # 检查磁盘缓存
        cache_path = self._get_embeddings_cache_path(dataset_name)
        if Path(cache_path).exists():
            try:
                print(f"📁 从磁盘加载向量缓存: {cache_path}")
                cache_data = FileUtils.load_pickle(cache_path)

                # 详细的验证信息
                cached_model = cache_data.get('model_name', 'unknown')
                cached_dataset = cache_data.get('dataset_name', 'unknown')
                cached_docs = cache_data.get('documents', [])

                print(f"🔍 缓存验证:")
                print(f"   模型匹配: {cached_model} == {self.model_name} -> {cached_model == self.model_name}")
                print(f"   数据集匹配: {cached_dataset} == {dataset_name} -> {cached_dataset == dataset_name}")
                print(f"   文档数量: {len(cached_docs)} vs {len(documents)}")

                # 验证缓存数据
                if (cached_model == self.model_name and
                    cached_dataset == dataset_name and
                    'embeddings' in cache_data and
                    'documents' in cache_data):

                    # 额外验证：检查文档数量是否匹配或是子集关系
                    if len(cached_docs) == len(documents):
                        # 完全匹配情况
                        docs_match = True
                        check_count = min(5, len(documents))
                        for i in range(check_count):
                            if cached_docs[i].doc_id != documents[i].doc_id:
                                docs_match = False
                                print(f"   文档ID不匹配 [{i}]: {cached_docs[i].doc_id} vs {documents[i].doc_id}")
                                break

                        if docs_match:
                            self.document_embeddings = cache_data['embeddings']
                            self.documents = cached_docs
                            self.documents_hash = current_hash
                            self.embeddings_cache_path = cache_path

                            print(f"✅ 成功加载缓存向量，文档数量: {len(self.documents)}")
                            return True
                        else:
                            print(f"⚠️ 文档内容不匹配，需要重新编码")
                    
                    elif len(cached_docs) > len(documents):
                        # 子集匹配情况：当前文档是缓存文档的子集
                        print(f"🔍 检查子集匹配: 缓存{len(cached_docs)}个文档 vs 当前{len(documents)}个文档")
                        
                        # 创建缓存文档ID到索引的映射
                        cached_doc_map = {doc.doc_id: i for i, doc in enumerate(cached_docs)}
                        
                        # 检查当前文档是否都在缓存中
                        subset_indices = []
                        subset_docs = []
                        
                        for doc in documents:
                            if doc.doc_id in cached_doc_map:
                                idx = cached_doc_map[doc.doc_id]
                                subset_indices.append(idx)
                                subset_docs.append(cached_docs[idx])
                            else:
                                print(f"   文档 {doc.doc_id} 不在缓存中")
                                break
                        else:
                            # 所有文档都在缓存中，提取对应的向量
                            import numpy as np
                            subset_embeddings = cache_data['embeddings'][subset_indices]
                            
                            self.document_embeddings = subset_embeddings
                            self.documents = subset_docs
                            self.documents_hash = current_hash
                            self.embeddings_cache_path = cache_path
                            
                            print(f"✅ 成功从缓存提取子集向量，文档数量: {len(self.documents)}")
                            return True
                        
                        print(f"⚠️ 子集匹配失败，需要重新编码")
                    else:
                        print(f"⚠️ 缓存文档数量不足: {len(cached_docs)} < {len(documents)}")
                else:
                    print(f"⚠️ 缓存验证失败")

            except Exception as e:
                print(f"⚠️ 加载向量缓存失败: {e}")

        return False

    def _encode_and_build_index(self, documents: List[Document], dataset_name: str) -> None:
        """编码文档并构建索引"""
        # 准备文档文本
        texts = []
        for doc in documents:
            text = self._prepare_text(doc)
            texts.append(text)

        print("正在编码文档...")
        # 编码文档
        self.document_embeddings = self._encode_texts(texts, show_progress=True)

        # 保存向量到缓存
        self._save_embeddings_cache(documents, dataset_name)

        # 构建FAISS索引
        self._build_faiss_index()

    def _save_embeddings_cache(self, documents: List[Document], dataset_name: str) -> None:
        """保存向量到缓存"""
        try:
            # 计算文档哈希（用于验证）
            documents_hash = self._calculate_documents_hash(documents)
            cache_path = self._get_embeddings_cache_path(dataset_name)

            # 准备缓存数据
            cache_data = {
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'dataset_name': dataset_name,
                'documents_hash': documents_hash,
                'documents': documents,
                'embeddings': self.document_embeddings,
                'document_count': len(documents),
                'created_at': str(Path().cwd()),  # 简单的时间戳
            }

            # 保存到磁盘
            FileUtils.save_pickle(cache_data, cache_path)

            # 更新实例变量
            self.documents_hash = documents_hash
            self.embeddings_cache_path = cache_path

            print(f"💾 向量缓存已保存: {cache_path}")

        except Exception as e:
            print(f"⚠️ 保存向量缓存失败: {e}")
            # 不影响主流程，继续执行

    def _build_faiss_index(self) -> None:
        """构建FAISS索引"""
        print("构建FAISS索引...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # 内积索引(余弦相似度)

        # 添加向量到索引
        self.index.add(self.document_embeddings)
    
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
            'faiss_path': str(faiss_path)
        }
        
        FileUtils.save_pickle(metadata, index_path)
        print(f"向量索引已保存到: {index_path}")
    
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
            
            # 加载FAISS索引
            faiss_path = metadata.get('faiss_path')
            if faiss_path and Path(faiss_path).exists():
                self.index = faiss.read_index(faiss_path)
            else:
                # 如果FAISS文件不存在，重新构建索引
                print("FAISS索引文件不存在，重新构建...")
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.index.add(self.document_embeddings)
            
            self.is_built = True
            print(f"向量索引已从 {index_path} 加载完成")
            
        except Exception as e:
            raise RuntimeError(f"加载索引失败: {e}")
    
    def get_document_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """获取指定文档的向量表示"""
        if not self.is_built:
            return None
        
        for i, doc in enumerate(self.documents):
            if doc.doc_id == doc_id:
                return self.document_embeddings[i]
        
        return None
    
    def find_similar_documents(self, doc_id: str, top_k: int = 10) -> List[RetrievalResult]:
        """查找与指定文档相似的其他文档"""
        embedding = self.get_document_embedding(doc_id)
        if embedding is None:
            return []
        
        # 搜索相似文档
        scores, indices = self.index.search(embedding.reshape(1, -1), top_k + 1)
        
        # 生成结果（排除自身）
        results = []
        for score, doc_idx in zip(scores[0], indices[0]):
            if doc_idx >= 0 and self.documents[doc_idx].doc_id != doc_id:
                result = RetrievalResult(
                    doc_id=self.documents[doc_idx].doc_id,
                    score=float(score),
                    document=self.documents[doc_idx],
                    retriever_name=self.name
                )
                results.append(result)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        if not self.is_built:
            return {}
        
        return {
            'document_count': len(self.documents),
            'embedding_dim': self.embedding_dim,
            'model_name': self.model_name,
            'index_size': self.index.ntotal if self.index else 0,
            'memory_usage_mb': (
                self.document_embeddings.nbytes / (1024 * 1024) 
                if self.document_embeddings is not None else 0
            )
        }


def create_dense_retriever(config: Dict[str, Any]) -> DenseRetriever:
    """创建向量检索器的工厂函数"""
    return DenseRetriever(config=config)