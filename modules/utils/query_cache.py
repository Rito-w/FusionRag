"""查询缓存系统
缓存查询结果以提升重复查询的响应速度
"""

import hashlib
import json
import time
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
from modules.utils.interfaces import Query, RetrievalResult


class QueryCache:
    """查询缓存系统"""
    
    def __init__(self, 
                 cache_dir: str = "checkpoints/query_cache",
                 max_memory_cache: int = 1000,
                 max_disk_cache: int = 10000,
                 ttl_seconds: int = 3600):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_cache = max_memory_cache
        self.max_disk_cache = max_disk_cache
        self.ttl_seconds = ttl_seconds
        
        # 内存缓存 (LRU)
        self.memory_cache = OrderedDict()
        
        # 磁盘缓存索引
        self.disk_cache_index = self._load_disk_index()
    
    def _generate_cache_key(self, query: Query, retriever_name: str, top_k: int) -> str:
        """生成缓存键"""
        content = f"{query.text}|{retriever_name}|{top_k}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_disk_index(self) -> Dict[str, float]:
        """加载磁盘缓存索引"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_disk_index(self):
        """保存磁盘缓存索引"""
        index_file = self.cache_dir / "cache_index.json"
        with open(index_file, 'w') as f:
            json.dump(self.disk_cache_index, f)
    
    def _is_expired(self, timestamp: float) -> bool:
        """检查是否过期"""
        return time.time() - timestamp > self.ttl_seconds
    
    def get(self, query: Query, retriever_name: str, top_k: int) -> Optional[List[RetrievalResult]]:
        """获取缓存结果"""
        cache_key = self._generate_cache_key(query, retriever_name, top_k)
        
        # 1. 检查内存缓存
        if cache_key in self.memory_cache:
            cached_data = self.memory_cache[cache_key]
            if not self._is_expired(cached_data['timestamp']):
                # 更新LRU顺序
                self.memory_cache.move_to_end(cache_key)
                return cached_data['results']
            else:
                # 过期，删除
                del self.memory_cache[cache_key]
        
        # 2. 检查磁盘缓存
        if cache_key in self.disk_cache_index:
            timestamp = self.disk_cache_index[cache_key]
            if not self._is_expired(timestamp):
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            results = pickle.load(f)
                        
                        # 加载到内存缓存
                        self._add_to_memory_cache(cache_key, results, timestamp)
                        return results
                    except Exception as e:
                        print(f"磁盘缓存加载失败: {e}")
            else:
                # 过期，删除
                del self.disk_cache_index[cache_key]
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
        
        return None
    
    def put(self, query: Query, retriever_name: str, top_k: int, results: List[RetrievalResult]):
        """存储缓存结果"""
        cache_key = self._generate_cache_key(query, retriever_name, top_k)
        timestamp = time.time()
        
        # 1. 存储到内存缓存
        self._add_to_memory_cache(cache_key, results, timestamp)
        
        # 2. 存储到磁盘缓存
        self._add_to_disk_cache(cache_key, results, timestamp)
    
    def _add_to_memory_cache(self, cache_key: str, results: List[RetrievalResult], timestamp: float):
        """添加到内存缓存"""
        # 如果超过最大容量，删除最旧的
        while len(self.memory_cache) >= self.max_memory_cache:
            self.memory_cache.popitem(last=False)
        
        self.memory_cache[cache_key] = {
            'results': results,
            'timestamp': timestamp
        }
    
    def _add_to_disk_cache(self, cache_key: str, results: List[RetrievalResult], timestamp: float):
        """添加到磁盘缓存"""
        # 如果超过最大容量，删除最旧的
        while len(self.disk_cache_index) >= self.max_disk_cache:
            oldest_key = min(self.disk_cache_index.keys(), 
                           key=lambda k: self.disk_cache_index[k])
            del self.disk_cache_index[oldest_key]
            
            cache_file = self.cache_dir / f"{oldest_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
        
        # 保存到磁盘
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
            
            self.disk_cache_index[cache_key] = timestamp
            self._save_disk_index()
        except Exception as e:
            print(f"磁盘缓存保存失败: {e}")
    
    def clear(self):
        """清空缓存"""
        self.memory_cache.clear()
        
        # 清空磁盘缓存
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        
        self.disk_cache_index.clear()
        self._save_disk_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_size': len(self.disk_cache_index),
            'memory_cache_limit': self.max_memory_cache,
            'disk_cache_limit': self.max_disk_cache,
            'ttl_seconds': self.ttl_seconds
        }


class CachedRetriever:
    """带缓存的检索器包装器"""
    
    def __init__(self, retriever, cache: QueryCache):
        self.retriever = retriever
        self.cache = cache
        self.name = retriever.name
        self.cache_hits = 0
        self.cache_misses = 0
    
    def retrieve(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """带缓存的检索"""
        # 尝试从缓存获取
        cached_results = self.cache.get(query, self.name, top_k)
        
        if cached_results is not None:
            self.cache_hits += 1
            return cached_results
        
        # 缓存未命中，执行检索
        self.cache_misses += 1
        results = self.retriever.retrieve(query, top_k)
        
        # 存储到缓存
        self.cache.put(query, self.name, top_k, results)
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def __getattr__(self, name):
        """代理其他方法到原始检索器"""
        return getattr(self.retriever, name)