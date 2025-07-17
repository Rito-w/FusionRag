"""
FusionRAG系统统一接口定义
定义了所有模块的基础接口和数据结构
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


@dataclass
class Document:
    """文档数据结构"""
    doc_id: str
    title: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Query:
    """查询数据结构"""
    query_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    doc_id: str
    score: float
    document: Document
    retriever_name: str


@dataclass
class FusionResult:
    """融合后的结果数据结构"""
    doc_id: str
    final_score: float
    document: Document
    individual_scores: Dict[str, float]  # 各检索器的分数


class QueryType(Enum):
    """查询类型枚举"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    ENTITY = "entity"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class BaseRetriever(ABC):
    """检索器基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_built = False
    
    @abstractmethod
    def build_index(self, documents: List[Document]) -> None:
        """构建索引"""
        pass
    
    @abstractmethod
    def retrieve(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """检索相关文档"""
        pass
    
    @abstractmethod
    def save_index(self, index_path: str) -> None:
        """保存索引"""
        pass
    
    @abstractmethod
    def load_index(self, index_path: str) -> None:
        """加载索引"""
        pass


class BaseClassifier(ABC):
    """分类器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_trained = False
    
    @abstractmethod
    def train(self, training_data: List[Tuple[Query, QueryType]]) -> None:
        """训练分类器"""
        pass
    
    @abstractmethod
    def predict(self, query: Query) -> Tuple[QueryType, float]:
        """预测查询类型"""
        pass
    
    @abstractmethod
    def save_model(self, model_path: str) -> None:
        """保存模型"""
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """加载模型"""
        pass


class BaseFusion(ABC):
    """融合器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def fuse(self, retrieval_results: Dict[str, List[RetrievalResult]], 
             query: Query) -> List[FusionResult]:
        """融合多个检索器的结果"""
        pass


class BaseReranker(ABC):
    """重排序器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def rerank(self, query: Query, candidates: List[FusionResult], 
               top_k: int = 10) -> List[FusionResult]:
        """重新排序候选结果"""
        pass


class BaseEvaluator(ABC):
    """评测器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def evaluate(self, predictions: List[List[str]], 
                 ground_truth: List[List[str]]) -> Dict[str, float]:
        """评测检索结果"""
        pass


# 数据加载接口
class DataLoader(ABC):
    """数据加载器基类"""
    
    @abstractmethod
    def load_documents(self, path: str) -> List[Document]:
        """加载文档数据"""
        pass
    
    @abstractmethod
    def load_queries(self, path: str) -> List[Query]:
        """加载查询数据"""
        pass
    
    @abstractmethod
    def load_qrels(self, path: str) -> Dict[str, List[str]]:
        """加载相关性标注数据"""
        pass


# 配置管理接口
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = {}
    
    @abstractmethod
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        pass
    
    @abstractmethod
    def save_config(self, config: Dict[str, Any]) -> None:
        """保存配置文件"""
        pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


# 日志接口
class Logger(ABC):
    """日志记录器基类"""
    
    @abstractmethod
    def log_retrieval(self, query: Query, results: List[RetrievalResult]) -> None:
        """记录检索日志"""
        pass
    
    @abstractmethod
    def log_evaluation(self, metrics: Dict[str, float]) -> None:
        """记录评测日志"""
        pass
    
    @abstractmethod
    def log_system(self, message: str, level: str = "INFO") -> None:
        """记录系统日志"""
        pass