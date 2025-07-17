"""
配置管理器
统一管理系统配置，支持YAML格式配置文件
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self.config = {}
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"配置文件加载成功: {self.config_path}")
            else:
                logger.warning(f"配置文件不存在: {self.config_path}")
                self.config = self._get_default_config()
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            self.config = self._get_default_config()
        
        return self.config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "data": {
                "corpus_path": "data/processed/sample_corpus.jsonl",
                "queries_path": "data/processed/sample_queries.jsonl", 
                "qrels_path": "data/processed/sample_qrels.tsv",
                "output_dir": "data/processed/"
            },
            "retrievers": {
                "bm25": {
                    "enabled": True,
                    "index_path": "checkpoints/retriever/bm25_index/",
                    "k1": 1.2,
                    "b": 0.75,
                    "top_k": 100
                },
                "dense": {
                    "enabled": True,
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "index_path": "checkpoints/retriever/dense_index.faiss",
                    "embedding_dim": 384,
                    "top_k": 100
                },
                "graph": {
                    "enabled": False,
                    "graph_path": "checkpoints/retriever/knowledge_graph.pkl",
                    "top_k": 50
                }
            },
            "classifier": {
                "enabled": False,
                "model_path": "checkpoints/cls/classifier.onnx",
                "threshold": 0.5,
                "classes": ["factual", "analytical", "procedural"]
            },
            "fusion": {
                "method": "weighted",
                "weights": {"bm25": 0.4, "dense": 0.5, "graph": 0.1},
                "top_k": 20
            },
            "reranker": {
                "enabled": False,
                "model_path": "checkpoints/reranker/reranker.onnx",
                "top_k": 10
            },
            "evaluation": {
                "metrics": ["recall@5", "recall@10", "ndcg@10", "map"],
                "output_path": "checkpoints/logs/eval_results.json"
            },
            "system": {
                "device": "cpu",
                "batch_size": 32,
                "num_threads": 4,
                "log_level": "INFO",
                "log_path": "checkpoints/logs/system.log"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持点分割的嵌套key"""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """设置配置项，支持点分割的嵌套key"""
        keys = key.split('.')
        config = self.config
        
        # 创建嵌套结构
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """批量更新配置"""
        def deep_update(d: dict, u: dict) -> dict:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        deep_update(self.config, updates)
    
    def save_config(self, path: Optional[str] = None) -> bool:
        """保存配置到文件"""
        save_path = Path(path) if path else self.config_path
        
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            logger.info(f"配置文件保存成功: {save_path}")
            return True
        except Exception as e:
            logger.error(f"配置文件保存失败: {e}")
            return False
    
    def validate_config(self) -> bool:
        """验证配置文件"""
        required_sections = ['data', 'retrievers', 'system']
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"缺少配置节: {section}")
                return False
        
        # 验证路径
        data_paths = [
            self.get('data.corpus_path'),
            self.get('data.queries_path'),
            self.get('data.qrels_path')
        ]
        
        missing_paths = []
        for path in data_paths:
            if path and not Path(path).exists():
                missing_paths.append(path)
        
        if missing_paths:
            logger.warning(f"以下数据文件不存在: {missing_paths}")
        
        return True
    
    def create_directories(self) -> None:
        """创建配置中指定的目录"""
        directories = [
            self.get('data.output_dir'),
            Path(self.get('retrievers.bm25.index_path', '')).parent,
            Path(self.get('retrievers.dense.index_path', '')).parent,
            Path(self.get('system.log_path', '')).parent,
            Path(self.get('evaluation.output_path', '')).parent
        ]
        
        for dir_path in directories:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def __str__(self) -> str:
        """字符串表示"""
        return yaml.dump(self.config, default_flow_style=False, allow_unicode=True)


# 全局配置实例
_global_config = None

def get_config(config_path: str = "configs/config.yaml") -> ConfigManager:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager(config_path)
    return _global_config


def setup_logging(config: ConfigManager) -> None:
    """设置日志"""
    log_level = config.get('system.log_level', 'INFO')
    log_path = config.get('system.log_path', 'checkpoints/logs/system.log')
    
    # 创建日志目录
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )