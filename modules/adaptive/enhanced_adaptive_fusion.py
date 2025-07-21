#!/usr/bin/env python3
"""
增强版自适应融合引擎
整合增强的查询分析器和自适应路由器
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..utils.interfaces import Query, Document, RetrievalResult, FusionResult
from ..analysis.enhanced_query_analyzer import EnhancedQueryFeatures, create_enhanced_query_analyzer
from .enhanced_adaptive_router import create_enhanced_adaptive_router


class EnhancedAdaptiveFusion:
    """增强版自适应融合引擎
    
    整合增强的查询分析器和自适应路由器，实现智能融合策略选择
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 创建查询分析器
        self.query_analyzer = create_enhanced_query_analyzer(self.config.get('query_analyzer', {}))
        
        # 创建自适应路由器
        self.router = create_enhanced_adaptive_router(self.config.get('adaptive_router', {}))
        
        # 设置日志
        self.enable_logging = self.config.get('enable_logging', True)
        if self.enable_logging:
            self._setup_logging()
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"adaptive_fusion_{time.strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('adaptive_fusion')
        self.logger.info("初始化增强版自适应融合引擎")
    
    def fuse(self, query: Query, bm25_results: List[RetrievalResult], 
            vector_results: List[RetrievalResult], 
            dataset_name: str, top_k: int = 10) -> Tuple[List[FusionResult], Dict[str, Any]]:
        """融合多个检索器的结果
        
        Args:
            query: 查询
            bm25_results: BM25检索结果
            vector_results: 向量检索结果
            dataset_name: 数据集名称
            top_k: 返回的结果数量
            
        Returns:
            融合结果, 融合元数据
        """
        start_time = time.time()
        
        # 分析查询
        query_features = self.query_analyzer.analyze_query(query)
        
        # 选择融合策略
        strategy_name, strategy_info, is_exploration = self.router.select_strategy(
            query_features, dataset_name
        )
        
        # 应用融合策略
        fusion_results = self.router.apply_strategy(
            strategy_name, strategy_info, bm25_results, vector_results, top_k
        )
        
        # 记录元数据
        metadata = {
            'query_features': query_features.__dict__,
            'strategy': strategy_name,
            'strategy_info': strategy_info,
            'is_exploration': is_exploration,
            'processing_time': time.time() - start_time
        }
        
        # 记录日志
        if self.enable_logging:
            self.logger.info(
                f"查询: '{query.text[:50]}...' | "
                f"类型: {query_features.final_type} | "
                f"策略: {strategy_name} | "
                f"结果数: {len(fusion_results)}"
            )
        
        return fusion_results, metadata
    
    def update_performance(self, query: Query, dataset_name: str, 
                          strategy_name: str, metrics: Dict[str, float]) -> None:
        """更新性能反馈
        
        Args:
            query: 查询
            dataset_name: 数据集名称
            strategy_name: 使用的策略名称
            metrics: 检索指标
        """
        # 分析查询
        query_features = self.query_analyzer.analyze_query(query)
        
        # 更新路由器的性能历史
        self.router.update_performance(query_features, dataset_name, strategy_name, metrics)
        
        # 记录日志
        if self.enable_logging:
            self.logger.info(
                f"更新性能: 数据集={dataset_name} | "
                f"查询类型={query_features.final_type} | "
                f"策略={strategy_name} | "
                f"MRR={metrics.get('mrr', 0):.3f}"
            )


def create_enhanced_adaptive_fusion(config: Dict[str, Any] = None) -> EnhancedAdaptiveFusion:
    """创建增强自适应融合引擎的工厂函数"""
    return EnhancedAdaptiveFusion(config)