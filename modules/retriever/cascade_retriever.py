"""级联检索器
实现两阶段检索策略，提高检索效率和准确性
"""

import time
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass

from ..utils.interfaces import BaseRetriever, Document, Query, RetrievalResult
from ..utils.common import FileUtils


class CascadeRetriever(BaseRetriever):
    """级联检索器
    
    实现两阶段检索策略：
    1. 第一阶段：使用高效但可能精度较低的检索器获取候选文档
    2. 第二阶段：使用精度更高但可能较慢的检索器对候选文档进行重新排序
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        name = config.get('name', 'cascade_retriever') if config else 'cascade_retriever'
        super().__init__(name=name, config=config or {})
        
        # 检索器配置
        self.first_stage_retriever = None
        self.second_stage_retriever = None
        
        # 级联参数
        self.first_stage_k = self.config.get('first_stage_k', 100)  # 第一阶段检索的文档数
        self.rerank_all = self.config.get('rerank_all', False)  # 是否对所有候选文档重新排序
        self.min_first_stage_score = self.config.get('min_first_stage_score', 0.0)  # 第一阶段最小分数阈值
        
        # 性能统计
        self.stats = {
            'first_stage_time': 0.0,
            'second_stage_time': 0.0,
            'total_time': 0.0,
            'queries_processed': 0,
            'avg_first_stage_candidates': 0,
            'avg_final_results': 0
        }
    
    def set_first_stage_retriever(self, retriever: BaseRetriever) -> None:
        """设置第一阶段检索器"""
        self.first_stage_retriever = retriever
    
    def set_second_stage_retriever(self, retriever: BaseRetriever) -> None:
        """设置第二阶段检索器"""
        self.second_stage_retriever = retriever
    
    def build_index(self, documents: List[Document]) -> None:
        """构建索引（同时构建两个阶段的检索器索引）"""
        if self.first_stage_retriever is None or self.second_stage_retriever is None:
            raise ValueError("请先设置第一阶段和第二阶段检索器")
        
        print(f"构建级联检索器索引，文档数量: {len(documents)}")
        
        # 构建第一阶段检索器索引
        print("构建第一阶段检索器索引...")
        self.first_stage_retriever.build_index(documents)
        
        # 构建第二阶段检索器索引
        print("构建第二阶段检索器索引...")
        self.second_stage_retriever.build_index(documents)
        
        print("级联检索器索引构建完成")
    
    def retrieve(self, query: Query, top_k: int = 10) -> List[RetrievalResult]:
        """执行级联检索"""
        if self.first_stage_retriever is None or self.second_stage_retriever is None:
            raise ValueError("请先设置第一阶段和第二阶段检索器")
        
        start_time = time.time()
        
        # 第一阶段检索
        first_stage_start = time.time()
        first_stage_results = self.first_stage_retriever.retrieve(query, self.first_stage_k)
        first_stage_end = time.time()
        
        # 过滤低分结果
        if self.min_first_stage_score > 0:
            first_stage_results = [r for r in first_stage_results if r.score >= self.min_first_stage_score]
        
        # 如果第一阶段没有结果，直接返回空列表
        if not first_stage_results:
            return []
        
        # 第二阶段检索
        second_stage_start = time.time()
        
        if self.rerank_all:
            # 对所有候选文档重新排序
            candidate_docs = [r.document for r in first_stage_results]
            second_stage_results = self.second_stage_retriever.retrieve(query, top_k)
            
            # 只保留第一阶段检索到的文档
            candidate_doc_ids = {doc.doc_id for doc in candidate_docs}
            second_stage_results = [r for r in second_stage_results if r.doc_id in candidate_doc_ids]
        else:
            # 只对第一阶段检索到的文档重新排序
            # 这里假设第二阶段检索器支持直接对指定文档进行排序
            # 如果不支持，需要修改实现方式
            if hasattr(self.second_stage_retriever, 'rerank'):
                second_stage_results = self.second_stage_retriever.rerank(query, first_stage_results, top_k)
            else:
                # 如果没有rerank方法，使用通用方法
                candidate_docs = [r.document for r in first_stage_results]
                # 创建一个临时检索器，只包含候选文档
                temp_retriever = self.second_stage_retriever.__class__(self.second_stage_retriever.config)
                temp_retriever.build_index(candidate_docs)
                second_stage_results = temp_retriever.retrieve(query, top_k)
        
        second_stage_end = time.time()
        end_time = time.time()
        
        # 更新统计信息
        self.stats['first_stage_time'] += (first_stage_end - first_stage_start)
        self.stats['second_stage_time'] += (second_stage_end - second_stage_start)
        self.stats['total_time'] += (end_time - start_time)
        self.stats['queries_processed'] += 1
        self.stats['avg_first_stage_candidates'] = (
            (self.stats['avg_first_stage_candidates'] * (self.stats['queries_processed'] - 1) + 
             len(first_stage_results)) / self.stats['queries_processed']
        )
        self.stats['avg_final_results'] = (
            (self.stats['avg_final_results'] * (self.stats['queries_processed'] - 1) + 
             len(second_stage_results)) / self.stats['queries_processed']
        )
        
        # 返回结果（最多top_k个）
        return second_stage_results[:top_k]
    
    def save_index(self, index_path: str) -> None:
        """保存索引（同时保存两个阶段的检索器索引）"""
        if self.first_stage_retriever is None or self.second_stage_retriever is None:
            raise ValueError("请先设置第一阶段和第二阶段检索器")
        
        # 保存第一阶段检索器索引
        first_stage_path = f"{index_path}.first_stage"
        self.first_stage_retriever.save_index(first_stage_path)
        
        # 保存第二阶段检索器索引
        second_stage_path = f"{index_path}.second_stage"
        self.second_stage_retriever.save_index(second_stage_path)
        
        # 保存级联检索器配置
        config = {
            'name': self.name,
            'first_stage_k': self.first_stage_k,
            'rerank_all': self.rerank_all,
            'min_first_stage_score': self.min_first_stage_score,
            'first_stage_retriever_type': self.first_stage_retriever.__class__.__name__,
            'second_stage_retriever_type': self.second_stage_retriever.__class__.__name__,
            'first_stage_path': first_stage_path,
            'second_stage_path': second_stage_path,
            'stats': self.stats
        }
        
        FileUtils.save_json(config, index_path)
        print(f"级联检索器配置已保存到: {index_path}")
    
    def load_index(self, index_path: str) -> None:
        """加载索引（同时加载两个阶段的检索器索引）"""
        # 加载级联检索器配置
        config = FileUtils.load_json(index_path)
        
        # 更新配置
        self.name = config.get('name', self.name)
        self.first_stage_k = config.get('first_stage_k', self.first_stage_k)
        self.rerank_all = config.get('rerank_all', self.rerank_all)
        self.min_first_stage_score = config.get('min_first_stage_score', self.min_first_stage_score)
        self.stats = config.get('stats', self.stats)
        
        # 检查检索器是否已设置
        if self.first_stage_retriever is None or self.second_stage_retriever is None:
            raise ValueError("请先设置第一阶段和第二阶段检索器")
        
        # 加载第一阶段检索器索引
        first_stage_path = config.get('first_stage_path')
        if first_stage_path:
            self.first_stage_retriever.load_index(first_stage_path)
        
        # 加载第二阶段检索器索引
        second_stage_path = config.get('second_stage_path')
        if second_stage_path:
            self.second_stage_retriever.load_index(second_stage_path)
        
        print(f"级联检索器配置已从 {index_path} 加载完成")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        stats = self.stats.copy()
        
        # 添加检索器信息
        if self.first_stage_retriever:
            stats['first_stage_retriever'] = {
                'type': self.first_stage_retriever.__class__.__name__,
                'stats': self.first_stage_retriever.get_statistics()
            }
        
        if self.second_stage_retriever:
            stats['second_stage_retriever'] = {
                'type': self.second_stage_retriever.__class__.__name__,
                'stats': self.second_stage_retriever.get_statistics()
            }
        
        # 添加配置信息
        stats['config'] = {
            'first_stage_k': self.first_stage_k,
            'rerank_all': self.rerank_all,
            'min_first_stage_score': self.min_first_stage_score
        }
        
        # 计算平均时间
        if stats['queries_processed'] > 0:
            stats['avg_first_stage_time'] = stats['first_stage_time'] / stats['queries_processed']
            stats['avg_second_stage_time'] = stats['second_stage_time'] / stats['queries_processed']
            stats['avg_total_time'] = stats['total_time'] / stats['queries_processed']
        
        return stats


class TwoStageRetriever(CascadeRetriever):
    """两阶段检索器（级联检索器的别名）"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = self.config.get('name', 'two_stage_retriever')


def create_cascade_retriever(config: Dict[str, Any] = None) -> CascadeRetriever:
    """创建级联检索器的工厂函数"""
    return CascadeRetriever(config=config)


def create_two_stage_retriever(config: Dict[str, Any] = None) -> TwoStageRetriever:
    """创建两阶段检索器的工厂函数"""
    return TwoStageRetriever(config=config)