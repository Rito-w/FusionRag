"""
模型缓存管理器
确保模型只加载一次，避免重复加载浪费资源
"""

import os
from typing import Dict, Any, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelCache:
    """模型缓存管理器"""
    
    _instance = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
            cls._instance._models = {}
            cls._instance._embedding_models = {}
            cls._instance._tokenizers = {}
            cls._instance._nlp_models = {}
            logger.info("初始化模型缓存管理器")
        return cls._instance
    
    def get_transformer_model(self, model_name: str, model_class=None, **kwargs) -> Any:
        """获取Transformer模型（如BERT、RoBERTa等）"""
        if model_name in self._models:
            logger.info(f"从缓存获取模型: {model_name}")
            return self._models[model_name]
        
        logger.info(f"加载模型: {model_name}")
        try:
            from transformers import AutoModel
            model_class = model_class or AutoModel
            model = model_class.from_pretrained(model_name, **kwargs)
            self._models[model_name] = model
            logger.info(f"模型加载完成: {model_name}")
            return model
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return None
    
    def get_embedding_model(self, model_name: str, **kwargs) -> Any:
        """获取嵌入模型（如Sentence-Transformers）"""
        if model_name in self._embedding_models:
            logger.info(f"从缓存获取嵌入模型: {model_name}")
            return self._embedding_models[model_name]
        
        logger.info(f"加载嵌入模型: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, **kwargs)
            self._embedding_models[model_name] = model
            logger.info(f"嵌入模型加载完成: {model_name}, 嵌入维度: {model.get_sentence_embedding_dimension()}")
            return model
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            return None
    
    def get_tokenizer(self, model_name: str, **kwargs) -> Any:
        """获取分词器"""
        if model_name in self._tokenizers:
            logger.info(f"从缓存获取分词器: {model_name}")
            return self._tokenizers[model_name]
        
        logger.info(f"加载分词器: {model_name}")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            self._tokenizers[model_name] = tokenizer
            logger.info(f"分词器加载完成: {model_name}")
            return tokenizer
        except Exception as e:
            logger.error(f"分词器加载失败: {e}")
            return None
    
    def get_spacy_model(self, model_name: str) -> Any:
        """获取spaCy模型"""
        if not model_name or model_name == "null":
            logger.info("spaCy模型名称为空，跳过加载")
            return None
            
        if model_name in self._nlp_models:
            logger.info(f"从缓存获取spaCy模型: {model_name}")
            return self._nlp_models[model_name]
        
        logger.info(f"加载spaCy模型: {model_name}")
        try:
            import spacy
            nlp = spacy.load(model_name)
            self._nlp_models[model_name] = nlp
            logger.info(f"spaCy模型加载完成: {model_name}")
            return nlp
        except Exception as e:
            logger.error(f"spaCy模型加载失败: {e}")
            return None
    
    def clear_cache(self, model_type: Optional[str] = None) -> None:
        """清除缓存"""
        if model_type == "transformer" or model_type is None:
            self._models.clear()
        if model_type == "embedding" or model_type is None:
            self._embedding_models.clear()
        if model_type == "tokenizer" or model_type is None:
            self._tokenizers.clear()
        if model_type == "nlp" or model_type is None:
            self._nlp_models.clear()
        
        # 强制垃圾回收
        import gc
        gc.collect()
        logger.info(f"已清除{'所有' if model_type is None else model_type}模型缓存")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            "transformer_models": list(self._models.keys()),
            "embedding_models": list(self._embedding_models.keys()),
            "tokenizers": list(self._tokenizers.keys()),
            "nlp_models": list(self._nlp_models.keys())
        }


# 全局单例
model_cache = ModelCache()