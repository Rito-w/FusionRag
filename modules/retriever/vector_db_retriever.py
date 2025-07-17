"""
向量数据库检索器
支持多种向量数据库：Pinecone, Weaviate, Qdrant等
"""

from typing import List, Dict, Any, Optional
import numpy as np
from abc import ABC, abstractmethod

from ..utils.interfaces import BaseRetriever, Document, Query, RetrievalResult


class BaseVectorDB(ABC):
    """向量数据库基类"""
    
    @abstractmethod
    def create_index(self, dimension: int, metric: str = "cosine"):
        """创建索引"""
        pass
    
    @abstractmethod
    def upsert(self, vectors: List[np.ndarray], metadata: List[Dict]):
        """插入向量"""
        pass
    
    @abstractmethod
    def query(self, vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """查询相似向量"""
        pass


class PineconeVectorDB(BaseVectorDB):
    """Pinecone向量数据库"""
    
    def __init__(self, config: Dict[str, Any]):
        try:
            import pinecone
            self.pinecone = pinecone
        except ImportError:
            raise ImportError("请安装pinecone-client: pip install pinecone-client")
        
        self.api_key = config.get('api_key')
        self.environment = config.get('environment')
        self.index_name = config.get('index_name', 'fusionrag-index')
        
        # 初始化Pinecone
        self.pinecone.init(
            api_key=self.api_key,
            environment=self.environment
        )
        
        self.index = None
    
    def create_index(self, dimension: int, metric: str = "cosine"):
        """创建Pinecone索引"""
        if self.index_name not in self.pinecone.list_indexes():
            self.pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric
            )
        
        self.index = self.pinecone.Index(self.index_name)
    
    def upsert(self, vectors: List[np.ndarray], metadata: List[Dict]):
        """插入向量到Pinecone"""
        if not self.index:
            raise RuntimeError("索引未创建")
        
        # 准备数据
        vectors_to_upsert = [
            (str(i), vector.tolist(), meta)
            for i, (vector, meta) in enumerate(zip(vectors, metadata))
        ]
        
        # 批量插入
        self.index.upsert(vectors=vectors_to_upsert)
    
    def query(self, vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """查询相似向量"""
        if not self.index:
            raise RuntimeError("索引未创建")
        
        results = self.index.query(
            vector=vector.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        return [
            {
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata']
            }
            for match in results['matches']
        ]


class QdrantVectorDB(BaseVectorDB):
    """Qdrant向量数据库"""
    
    def __init__(self, config: Dict[str, Any]):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            self.QdrantClient = QdrantClient
            self.Distance = Distance
            self.VectorParams = VectorParams
        except ImportError:
            raise ImportError("请安装qdrant-client: pip install qdrant-client")
        
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 6333)
        self.collection_name = config.get('collection_name', 'fusionrag')
        
        self.client = self.QdrantClient(host=self.host, port=self.port)
    
    def create_index(self, dimension: int, metric: str = "cosine"):
        """创建Qdrant集合"""
        distance_map = {
            "cosine": self.Distance.COSINE,
            "euclidean": self.Distance.EUCLID,
            "dot": self.Distance.DOT
        }
        
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=self.VectorParams(
                size=dimension,
                distance=distance_map.get(metric, self.Distance.COSINE)
            )
        )
    
    def upsert(self, vectors: List[np.ndarray], metadata: List[Dict]):
        """插入向量到Qdrant"""
        from qdrant_client.models import PointStruct
        
        points = [
            PointStruct(
                id=i,
                vector=vector.tolist(),
                payload=meta
            )
            for i, (vector, meta) in enumerate(zip(vectors, metadata))
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def query(self, vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """查询相似向量"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector.tolist(),
            limit=top_k
        )
        
        return [
            {
                'id': str(result.id),
                'score': result.score,
                'metadata': result.payload
            }
            for result in results
        ]


class VectorDBRetriever(BaseRetriever):
    """向量数据库检索器"""
    
    def __init__(self, name: str = "vectordb", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        
        # 初始化向量数据库
        db_type = self.config.get('db_type', 'qdrant')
        if db_type == 'pinecone':
            self.vector_db = PineconeVectorDB(self.config.get('pinecone', {}))
        elif db_type == 'qdrant':
            self.vector_db = QdrantVectorDB(self.config.get('qdrant', {}))
        else:
            raise ValueError(f"不支持的向量数据库类型: {db_type}")
        
        # 初始化embedding模型
        from sentence_transformers import SentenceTransformer
        model_name = self.config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.encoder = SentenceTransformer(model_name)
        
        self.documents = []
    
    def build_index(self, documents: List[Document]) -> None:
        """构建向量数据库索引"""
        print(f"开始构建向量数据库索引，文档数量: {len(documents)}")
        
        self.documents = documents
        
        # 创建索引
        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.vector_db.create_index(dimension=embedding_dim)
        
        # 准备文档文本和元数据
        texts = [f"{doc.title} {doc.text}" for doc in documents]
        metadata = [
            {
                'doc_id': doc.doc_id,
                'title': doc.title,
                'text': doc.text[:1000]  # 限制长度
            }
            for doc in documents
        ]
        
        # 生成embeddings
        print("生成文档embeddings...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        # 插入向量数据库
        print("插入向量数据库...")
        self.vector_db.upsert(embeddings, metadata)
        
        self.is_built = True
        print("向量数据库索引构建完成")
    
    def retrieve(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """从向量数据库检索"""
        if not self.is_built:
            raise RuntimeError("索引尚未构建")
        
        # 生成查询embedding
        query_embedding = self.encoder.encode([query.text])[0]
        
        # 查询向量数据库
        results = self.vector_db.query(query_embedding, top_k)
        
        # 转换为RetrievalResult
        retrieval_results = []
        for result in results:
            metadata = result['metadata']
            doc = Document(
                doc_id=metadata['doc_id'],
                title=metadata['title'],
                text=metadata['text']
            )
            
            retrieval_results.append(RetrievalResult(
                doc_id=metadata['doc_id'],
                score=result['score'],
                document=doc,
                retriever_name=self.name
            ))
        
        return retrieval_results
    
    def save_index(self, index_path: str) -> None:
        """向量数据库通常不需要本地保存"""
        print("向量数据库索引已保存在云端")
    
    def load_index(self, index_path: str) -> None:
        """向量数据库通常不需要本地加载"""
        self.is_built = True
        print("向量数据库索引已从云端加载")


# 配置示例
EXAMPLE_CONFIG = {
    "pinecone": {
        "db_type": "pinecone",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "pinecone": {
            "api_key": "your-pinecone-api-key",
            "environment": "us-west1-gcp",
            "index_name": "fusionrag-index"
        }
    },
    "qdrant": {
        "db_type": "qdrant",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "fusionrag"
        }
    }
}
