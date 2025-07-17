"""
基础工具类实现
包含配置管理、数据加载、日志记录等功能
"""
import yaml
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .interfaces import (
    Document, Query, ConfigManager, DataLoader, Logger,
    RetrievalResult, FusionResult
)


class YAMLConfigManager(ConfigManager):
    """YAML配置管理器"""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            return self.config
        except FileNotFoundError:
            print(f"配置文件不存在: {self.config_path}")
            self.config = {}
            return self.config
        except yaml.YAMLError as e:
            print(f"配置文件解析错误: {e}")
            self.config = {}
            return self.config
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """保存配置文件"""
        self.config = config
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)


class JSONDataLoader(DataLoader):
    """JSON格式数据加载器"""
    
    def load_documents(self, path: str) -> List[Document]:
        """加载JSONL格式的文档数据"""
        documents = []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        doc = Document(
                            doc_id=data['doc_id'],
                            title=data.get('title', ''),
                            text=data.get('text', ''),
                            metadata=data.get('metadata')
                        )
                        documents.append(doc)
        except FileNotFoundError:
            print(f"文件不存在: {path}")
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
        
        return documents
    
    def load_queries(self, path: str) -> List[Query]:
        """加载JSONL格式的查询数据"""
        queries = []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        query = Query(
                            query_id=data['query_id'],
                            text=data['text'],
                            metadata=data.get('metadata')
                        )
                        queries.append(query)
        except FileNotFoundError:
            print(f"文件不存在: {path}")
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
        
        return queries
    
    def load_qrels(self, path: str) -> Dict[str, List[str]]:
        """加载相关性标注数据，支持TSV和TREC格式"""
        qrels = {}
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 跳过表头
                    if i == 0 and ('query_id' in line or 'query-id' in line):
                        continue
                    
                    # 尝试Tab分隔符，然后是空格分隔符
                    if '\t' in line:
                        parts = line.split('\t')
                    else:
                        parts = line.split()
                    
                    if len(parts) >= 3:
                        query_id = parts[0]
                        doc_id = parts[1] if len(parts) == 3 else parts[2]  # 适应不同格式
                        relevance = int(parts[-1])  # 最后一列是相关性分数
                        
                        if relevance > 0:  # 只保留相关的文档
                            if query_id not in qrels:
                                qrels[query_id] = []
                            qrels[query_id].append(doc_id)
        except FileNotFoundError:
            print(f"文件不存在: {path}")
        except Exception as e:
            print(f"加载qrels文件失败: {e}")
        
        return qrels


class SystemLogger(Logger):
    """系统日志记录器"""
    
    def __init__(self, log_dir: str = "checkpoints/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置系统日志
        self.system_logger = logging.getLogger('fusion_rag')
        self.system_logger.setLevel(logging.INFO)
        
        # 添加文件处理器
        handler = logging.FileHandler(
            self.log_dir / 'system.log', 
            encoding='utf-8'
        )
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s [%(name)s] %(message)s'
        )
        handler.setFormatter(formatter)
        self.system_logger.addHandler(handler)
        
        # 检索日志文件
        self.retrieval_log_path = self.log_dir / 'retrieval.log'
        self.evaluation_log_path = self.log_dir / 'evaluation.log'
    
    def log_retrieval(self, query: Query, results: List[RetrievalResult]) -> None:
        """记录检索日志"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'query_id': query.query_id,
            'query_text': query.text,
            'results': [
                {
                    'doc_id': result.doc_id,
                    'score': result.score,
                    'retriever': result.retriever_name,
                    'rank': i + 1
                }
                for i, result in enumerate(results)
            ]
        }
        
        with open(self.retrieval_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_evaluation(self, metrics: Dict[str, float]) -> None:
        """记录评测日志"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'metrics': metrics
        }
        
        with open(self.evaluation_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_system(self, message: str, level: str = "INFO") -> None:
        """记录系统日志"""
        level = level.upper()
        if level == "DEBUG":
            self.system_logger.debug(message)
        elif level == "INFO":
            self.system_logger.info(message)
        elif level == "WARNING":
            self.system_logger.warning(message)
        elif level == "ERROR":
            self.system_logger.error(message)
        elif level == "CRITICAL":
            self.system_logger.critical(message)


class FileUtils:
    """文件操作工具类"""
    
    @staticmethod
    def save_pickle(obj: Any, filepath: str) -> None:
        """保存pickle文件"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    
    @staticmethod
    def load_pickle(filepath: str) -> Any:
        """加载pickle文件"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_json(obj: Any, filepath: str) -> None:
        """保存JSON文件"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load_json(filepath: str) -> Any:
        """加载JSON文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def ensure_dir(dirpath: str) -> None:
        """确保目录存在"""
        Path(dirpath).mkdir(parents=True, exist_ok=True)


class TextProcessor:
    """文本处理工具类"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """文本规范化"""
        # 移除多余空白字符
        text = ' '.join(text.split())
        
        # 转换为小写
        text = text.lower()
        
        return text
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 512) -> str:
        """截断文本"""
        if len(text) <= max_length:
            return text
        
        # 在单词边界截断
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # 如果空格位置合理
            return truncated[:last_space]
        else:
            return truncated


class MetricsCalculator:
    """评测指标计算工具"""
    
    @staticmethod
    def recall_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
        """计算Recall@K"""
        if not ground_truth:
            return 0.0
        
        pred_at_k = set(predictions[:k])
        relevant = set(ground_truth)
        
        return len(pred_at_k & relevant) / len(relevant)
    
    @staticmethod
    def precision_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
        """计算Precision@K"""
        if k == 0:
            return 0.0
        
        pred_at_k = predictions[:k]
        relevant = set(ground_truth)
        
        return len([p for p in pred_at_k if p in relevant]) / k
    
    @staticmethod
    def average_precision(predictions: List[str], ground_truth: List[str]) -> float:
        """计算平均精确度(AP)"""
        if not ground_truth:
            return 0.0
        
        relevant = set(ground_truth)
        relevant_count = 0
        ap_sum = 0.0
        
        for i, pred in enumerate(predictions):
            if pred in relevant:
                relevant_count += 1
                precision = relevant_count / (i + 1)
                ap_sum += precision
        
        return ap_sum / len(relevant) if relevant else 0.0
    
    @staticmethod
    def dcg_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
        """计算DCG@K"""
        import math
        
        relevant = set(ground_truth)
        dcg = 0.0
        
        for i, pred in enumerate(predictions[:k]):
            if pred in relevant:
                dcg += 1.0 / math.log2(i + 2)
        
        return dcg
    
    @staticmethod
    def ndcg_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
        """计算NDCG@K"""
        import math
        
        dcg = MetricsCalculator.dcg_at_k(predictions, ground_truth, k)
        
        # 计算IDCG
        ideal_order = ground_truth[:k]
        idcg = sum(1.0 / math.log2(i + 2) for i in range(len(ideal_order)))
        
        return dcg / idcg if idcg > 0 else 0.0