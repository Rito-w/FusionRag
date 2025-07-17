"""并行检索器优化
通过并行执行多个检索器来提升性能
"""

import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional
from modules.utils.interfaces import BaseRetriever, Query, RetrievalResult


class ParallelRetrieverManager:
    """并行检索器管理器"""
    
    def __init__(self, retrievers: Dict[str, BaseRetriever], max_workers: int = 4):
        self.retrievers = retrievers
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    def retrieve_parallel(self, query: Query, top_k: int = 10) -> Dict[str, List[RetrievalResult]]:
        """并行执行多个检索器"""
        
        def single_retrieve(name_retriever_pair):
            name, retriever = name_retriever_pair
            try:
                return name, retriever.retrieve(query, top_k)
            except Exception as e:
                print(f"检索器 {name} 执行失败: {e}")
                return name, []
        
        # 并行执行所有检索器
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(single_retrieve, (name, retriever)): name
                for name, retriever in self.retrievers.items()
            }
            
            results = {}
            for future in concurrent.futures.as_completed(futures):
                name, retrieval_results = future.result()
                results[name] = retrieval_results
        
        return results
    
    async def retrieve_async(self, query: Query, top_k: int = 10) -> Dict[str, List[RetrievalResult]]:
        """异步并行检索"""
        
        async def async_retrieve(name: str, retriever: BaseRetriever):
            loop = asyncio.get_event_loop()
            try:
                # 在线程池中执行同步检索
                result = await loop.run_in_executor(
                    self.executor, 
                    lambda: retriever.retrieve(query, top_k)
                )
                return name, result
            except Exception as e:
                print(f"异步检索器 {name} 失败: {e}")
                return name, []
        
        # 创建异步任务
        tasks = [
            async_retrieve(name, retriever) 
            for name, retriever in self.retrievers.items()
        ]
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks)
        
        return {name: result for name, result in results}
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)