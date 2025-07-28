#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FusionRAG: 自适应混合检索系统

该系统结合了多种检索技术，通过自适应路由和融合策略提高检索效果。
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from modules.utils.interfaces import Document, Query, BaseRetriever
from modules.utils.common import YAMLConfigManager, JSONDataLoader, FileUtils, SystemLogger

# 导入检索器
from modules.retriever.dense_retriever import DenseRetriever
from modules.retriever.bm25_retriever import BM25Retriever
from modules.retriever.efficient_vector_index import EfficientVectorIndex
from modules.retriever.semantic_bm25 import SemanticBM25
from modules.retriever.cascade_retriever import CascadeRetriever
from modules.retriever.graph_retriever import GraphRetriever

# 导入分析和自适应组件
from modules.analysis.query_analyzer import QueryAnalyzer
from modules.adaptive.adaptive_router import AdaptiveRouter
from modules.adaptive.adaptive_fusion import AdaptiveFusion

# 导入评估组件
from modules.evaluation.evaluator import RetrievalEvaluator


class FusionRAGSystem:
    """FusionRAG 系统主类
    
    集成了所有组件，提供统一的接口。
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化 FusionRAG 系统
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config_manager = YAMLConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
        # 处理自动设备配置
        if self.config.get("system", {}).get("device") == "auto":
            import torch
            self.config["system"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"自动选择设备: {self.config['system']['device']}")
            
            # 更新配置文件
            self.config_manager.save_config(self.config)
        
        # 设置日志
        log_config = self.config.get("system", {})
        self.logger = SystemLogger(log_config.get("log_path", "checkpoints/logs/system.log"))
        self.logger.log_system(f"初始化 FusionRAG 系统，配置文件: {config_path}")
        
        # 创建数据目录
        self.data_dir = self.config.get("data", {}).get("output_dir", "data/processed")
        self.index_dir = "checkpoints/retriever"
        self.output_dir = "reports"
        
        for directory in [self.data_dir, self.index_dir, self.output_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self._init_components()
        
        self.logger.log_system("FusionRAG 系统初始化完成")
    
    def _init_components(self):
        """初始化系统组件"""
        # 初始化数据加载器
        data_config = self.config.get("data", {})
        self.data_loader = JSONDataLoader()
        
        # 初始化查询分析器
        analyzer_config = self.config.get("classifier", {})
        self.query_analyzer = QueryAnalyzer(config=analyzer_config)
        
        # 初始化检索器
        self.retrievers = {}
        self._init_retrievers()
        
        # 初始化自适应路由器
        router_config = self.config.get("classifier", {})
        self.router = AdaptiveRouter(config=router_config)
        
        # 初始化自适应融合引擎
        fusion_config = self.config.get("fusion", {})
        self.fusion = AdaptiveFusion(config=fusion_config)
        
        # 初始化评估器
        evaluator_config = self.config.get("evaluation", {})
        self.evaluator = RetrievalEvaluator(config=evaluator_config)
    
    def _init_retrievers(self):
        """初始化检索器"""
        retriever_configs = self.config.get("retrievers", {})
        
        # 创建BM25检索器
        if retriever_configs.get("bm25", {}).get("enabled", False):
            bm25_config = retriever_configs["bm25"]
            self.retrievers["bm25"] = BM25Retriever(name="bm25", config=bm25_config)
        
        # 创建稠密检索器
        if retriever_configs.get("dense", {}).get("enabled", False):
            dense_config = retriever_configs["dense"]
            self.retrievers["dense"] = DenseRetriever(name="dense", config=dense_config)
        
        # 创建图检索器
        if retriever_configs.get("graph", {}).get("enabled", False):
            graph_config = retriever_configs["graph"]
            self.retrievers["graph"] = GraphRetriever(name="graph", config=graph_config)
        
        # 创建高效向量索引
        efficient_config = {
            "model_name": retriever_configs.get("dense", {}).get("model_name", "intfloat/e5-large-v2"),
            "index_path": os.path.join(self.index_dir, "efficient_vector_index.faiss"),
            "index_type": "hnsw",
            "top_k": 100
        }
        self.retrievers["efficient_vector"] = EfficientVectorIndex(config=efficient_config)
        
        # 创建语义增强BM25
        semantic_bm25_config = {
            "model_name": retriever_configs.get("dense", {}).get("model_name", "intfloat/e5-large-v2"),
            "index_path": os.path.join(self.index_dir, "semantic_bm25_index.pkl"),
            "top_k": 100,
            "name": "semantic_bm25"
        }
        self.retrievers["semantic_bm25"] = SemanticBM25(config=semantic_bm25_config)
        
        # 创建级联检索器
        cascade_config = {
            "first_stage_top_k": 100,
            "second_stage_top_k": 20,
            "score_threshold": 0.5,
            "name": "cascade"
        }
        self.retrievers["cascade"] = CascadeRetriever(
            config=cascade_config
        )
        
        # 设置级联检索器的两个阶段
        if isinstance(self.retrievers["cascade"], CascadeRetriever):
            self.retrievers["cascade"].set_first_stage_retriever(self.retrievers.get("bm25") or self.retrievers.get("efficient_vector"))
            self.retrievers["cascade"].set_second_stage_retriever(self.retrievers.get("dense") or self.retrievers.get("semantic_bm25"))
        
        self.logger.log_system(f"已初始化 {len(self.retrievers)} 个检索器: {', '.join(self.retrievers.keys())}")
    
    def index_documents(self, documents: List[Document], force_rebuild: bool = False):
        """为所有检索器构建索引
        
        Args:
            documents: 文档列表
            force_rebuild: 是否强制重建索引
        """
        self.logger.log_system(f"开始为 {len(documents)} 个文档构建索引")
        
        for name, retriever in self.retrievers.items():
            self.logger.log_system(f"为检索器 '{name}' 构建索引")
            start_time = time.time()
            
            # 检查检索器的 build_index 方法是否支持 force_rebuild 参数
            import inspect
            sig = inspect.signature(retriever.build_index)
            if 'force_rebuild' in sig.parameters:
                retriever.build_index(documents, force_rebuild=force_rebuild)
            else:
                retriever.build_index(documents)
                
            end_time = time.time()
            self.logger.log_system(f"检索器 '{name}' 索引构建完成，耗时: {end_time - start_time:.2f}秒")
    
    def retrieve(self, query: Union[str, Query], top_k: int = 10, use_adaptive: bool = True) -> List[Document]:
        """检索文档
        
        Args:
            query: 查询字符串或查询对象
            top_k: 返回的文档数量
            use_adaptive: 是否使用自适应路由和融合
            
        Returns:
            检索到的文档列表
        """
        # 转换查询字符串为查询对象
        if isinstance(query, str):
            query = Query(query_id="q1", text=query)
        
        self.logger.info(f"处理查询: {query.text}")
        
        # 分析查询
        query_features = self.query_analyzer.analyze(query)
        query_type = self.query_analyzer.classify(query_features)
        
        self.logger.info(f"查询类型: {query_type.name}")
        
        if use_adaptive:
            # 使用自适应路由
            routing_decision = self.router.route(query, query_features, query_type)
            
            self.logger.log_system(f"路由决策: 主索引={routing_decision.primary_index}, "
                           f"次级索引={routing_decision.secondary_index}, "
                           f"融合方法={routing_decision.fusion_method}")
            
            # 获取检索结果
            primary_results = self.retrievers[routing_decision.primary_index].retrieve(query, top_k)
            
            if routing_decision.secondary_index:
                secondary_results = self.retrievers[routing_decision.secondary_index].retrieve(query, top_k)
                
                # 融合结果
                fusion_results = self.fusion.fuse(
                    query=query,
                    results_list=[primary_results, secondary_results],
                    weights=routing_decision.fusion_weights,
                    method=routing_decision.fusion_method,
                    top_k=top_k
                )
                
                # 更新路由器
                self.router.update(query, query_features, query_type, fusion_results)
                
                return [result.document for result in fusion_results.results[:top_k]]
            else:
                # 更新路由器
                self.router.update(query, query_features, query_type, primary_results)
                
                return [result.document for result in primary_results[:top_k]]
        else:
            # 使用默认检索器
            default_retriever = "dense" if "dense" in self.retrievers else list(self.retrievers.keys())[0]
            results = self.retrievers[default_retriever].retrieve(query, top_k)
            return [result.document for result in results[:top_k]]
    
    def evaluate(self, dataset_name: str = None, retriever_names: List[str] = None):
        """评估检索器性能
        
        Args:
            dataset_name: 数据集名称，如果为None则使用所有数据集
            retriever_names: 检索器名称列表，如果为None则评估所有检索器
        
        Returns:
            评估结果
        """
        # 加载数据集
        if dataset_name:
            datasets = {dataset_name: self._load_dataset(dataset_name)}
        else:
            # 使用默认数据集
            datasets = {"default": self._load_dataset()}
        
        # 选择检索器
        if retriever_names:
            retrievers = [self.retrievers[name] for name in retriever_names if name in self.retrievers]
        else:
            retrievers = list(self.retrievers.values())
        
        results = {}
        
        # 对每个数据集评估
        for dataset_name, (queries, qrels) in datasets.items():
            self.logger.log_system(f"在数据集 '{dataset_name}' 上评估 {len(retrievers)} 个检索器")
            dataset_results = self.evaluator.compare(retrievers, queries, qrels, dataset_name)
            results[dataset_name] = dataset_results
        
        # 生成报告
        for dataset_name, dataset_results in results.items():
            self.evaluator.generate_report(dataset_results)
        
        return results
    
    def _load_dataset(self, dataset_name: str = None) -> Tuple[List[Query], Dict[str, List[str]]]:
        """加载数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            查询列表和相关性标注
        """
        data_config = self.config.get("data", {})
        
        # 使用配置中的路径或构建路径
        if dataset_name:
            queries_path = f"data/processed/{dataset_name}_queries.jsonl"
            qrels_path = f"data/processed/{dataset_name}_qrels.tsv"
        else:
            queries_path = data_config.get("queries_path")
            qrels_path = data_config.get("qrels_path")
        
        if not os.path.exists(queries_path) or not os.path.exists(qrels_path):
            raise ValueError(f"数据集文件不存在: {queries_path} 或 {qrels_path}")
        
        # 加载查询和相关性标注
        queries = self.data_loader.load_queries(queries_path)
        qrels = self.data_loader.load_qrels(qrels_path)
        
        dataset_name = dataset_name or "default"
        self.logger.log_system(f"已加载数据集 '{dataset_name}': {len(queries)} 个查询, {len(qrels)} 个相关性标注")
        
        return queries, qrels
    
    def load_documents(self, corpus_path: str = None) -> List[Document]:
        """加载文档
        
        Args:
            corpus_path: 文档路径，如果为None则使用配置中的路径
            
        Returns:
            文档列表
        """
        data_config = self.config.get("data", {})
        corpus_path = corpus_path or data_config.get("corpus_path")
        
        if not os.path.exists(corpus_path):
            raise ValueError(f"文档文件不存在: {corpus_path}")
        
        documents = self.data_loader.load_documents(corpus_path)
        self.logger.log_system(f"已加载 {len(documents)} 个文档")
        
        return documents
    
    def save_state(self, path: str = None):
        """保存系统状态
        
        Args:
            path: 保存路径，如果为None则使用默认路径
        """
        if path is None:
            path = os.path.join(self.output_dir, "system_state.json")
        
        # 保存路由器状态
        router_state = self.router.save_state(os.path.join(self.output_dir, "router_state.json"))
        
        # 保存融合引擎状态
        fusion_state = self.fusion.save_state(os.path.join(self.output_dir, "fusion_state.json"))
        
        # 保存系统状态
        state = {
            "router_state": router_state,
            "fusion_state": fusion_state,
            "timestamp": time.time()
        }
        
        FileUtils.save_json(state, path)
        self.logger.log_system(f"系统状态已保存到: {path}")
        
        return path
    
    def load_state(self, path: str = None):
        """加载系统状态
        
        Args:
            path: 加载路径，如果为None则使用默认路径
        """
        if path is None:
            path = os.path.join(self.output_dir, "system_state.json")
        
        if not os.path.exists(path):
            self.logger.log_system(f"系统状态文件不存在: {path}", level="WARNING")
            return False
        
        try:
            state = FileUtils.load_json(path)
            
            # 加载路由器状态
            if "router_state" in state and os.path.exists(state["router_state"]):
                self.router.load_state(state["router_state"])
            
            # 加载融合引擎状态
            if "fusion_state" in state and os.path.exists(state["fusion_state"]):
                self.fusion.load_state(state["fusion_state"])
            
            self.logger.log_system(f"系统状态已从 {path} 加载")
            return True
        except Exception as e:
            self.logger.log_system(f"加载系统状态失败: {e}", level="ERROR")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "retrievers": {},
            "router": self.router.get_statistics(),
            "fusion": self.fusion.get_statistics(),
            "query_analyzer": self.query_analyzer.get_statistics()
        }
        
        # 获取各检索器统计信息
        for name, retriever in self.retrievers.items():
            stats["retrievers"][name] = retriever.get_statistics()
        
        return stats


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="FusionRAG: 自适应混合检索系统")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 索引命令
    index_parser = subparsers.add_parser("index", help="构建索引")
    index_parser.add_argument("--documents", "-d", help="文档文件路径")
    index_parser.add_argument("--force", "-f", action="store_true", help="强制重建索引")
    
    # 检索命令
    retrieve_parser = subparsers.add_parser("retrieve", help="检索文档")
    retrieve_parser.add_argument("--query", "-q", required=True, help="查询文本")
    retrieve_parser.add_argument("--top-k", "-k", type=int, default=10, help="返回的文档数量")
    retrieve_parser.add_argument("--no-adaptive", action="store_true", help="不使用自适应路由和融合")
    retrieve_parser.add_argument("--retriever", "-r", help="指定使用的检索器")
    
    # 评估命令
    evaluate_parser = subparsers.add_parser("evaluate", help="评估检索器性能")
    evaluate_parser.add_argument("--dataset", "-d", help="数据集名称")
    evaluate_parser.add_argument("--retrievers", "-r", nargs="*", help="检索器名称列表")
    
    # 统计命令
    stats_parser = subparsers.add_parser("stats", help="显示系统统计信息")
    
    # 通用参数
    parser.add_argument("--config", "-c", default="configs/config.yaml", help="配置文件路径")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 初始化系统
    system = FusionRAGSystem(config_path=args.config)
    
    # 执行命令
    if args.command == "index":
        # 加载文档
        documents = system.load_documents(args.documents)
        # 构建索引
        system.index_documents(documents, force_rebuild=args.force)
    
    elif args.command == "retrieve":
        # 检索文档
        use_adaptive = not args.no_adaptive
        
        if args.retriever and not use_adaptive:
            # 使用指定检索器
            if args.retriever not in system.retrievers:
                print(f"错误: 检索器 '{args.retriever}' 不存在")
                print(f"可用检索器: {', '.join(system.retrievers.keys())}")
                return
            
            query = Query(query_id="q1", text=args.query)
            results = system.retrievers[args.retriever].retrieve(query, args.top_k)
            documents = [result.document for result in results]
        else:
            # 使用自适应检索
            documents = system.retrieve(
                query=args.query,
                top_k=args.top_k,
                use_adaptive=use_adaptive
            )
        
        # 打印结果
        print(f"\n查询: {args.query}")
        print(f"找到 {len(documents)} 个相关文档:\n")
        
        for i, doc in enumerate(documents):
            print(f"{i+1}. {doc.title}")
            print(f"   ID: {doc.doc_id}")
            print(f"   内容: {doc.text[:200]}..." if len(doc.text) > 200 else f"   内容: {doc.text}")
            print()
    
    elif args.command == "evaluate":
        # 评估检索器
        system.evaluate(dataset_name=args.dataset, retriever_names=args.retrievers)
    
    elif args.command == "stats":
        # 显示统计信息
        stats = system.get_statistics()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    else:
        print("请指定命令: index, retrieve, evaluate 或 stats")
        print("使用 --help 查看帮助信息")


if __name__ == "__main__":
    main()