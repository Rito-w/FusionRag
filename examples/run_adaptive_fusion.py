#!/usr/bin/env python3
"""
自适应融合方法完整实验
集成查询分析器、自适应路由器和融合引擎的端到端系统
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from modules.retriever.bm25_retriever import BM25Retriever
from modules.retriever.efficient_vector_index import EfficientVectorIndex
from modules.analysis.simple_query_analyzer import SimpleQueryAnalyzer, create_simple_query_analyzer
from modules.adaptive.simple_adaptive_router import SimpleAdaptiveRouter, create_simple_adaptive_router
from modules.adaptive.simple_adaptive_fusion import SimpleAdaptiveFusion, create_simple_adaptive_fusion
from modules.evaluation.evaluator import IndexEvaluator
from modules.utils.interfaces import Query, Document, RetrievalResult, FusionResult


def load_dataset(dataset_name: str) -> Tuple[List[Document], List[Query], Dict[str, Dict[str, int]]]:
    """加载数据集"""
    print(f"加载数据集: {dataset_name}")
    
    # 加载文档
    documents = []
    corpus_path = f"data/processed/{dataset_name}_corpus.jsonl"
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc = Document(
                doc_id=data['doc_id'],
                title=data.get('title', ''),
                text=data.get('text', '')
            )
            documents.append(doc)
    
    # 加载查询
    queries = []
    queries_path = f"data/processed/{dataset_name}_queries.jsonl"
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query = Query(
                query_id=data['query_id'],
                text=data['text']
            )
            queries.append(query)
    
    # 加载相关性判断
    relevance_judgments = {}
    qrels_path = f"data/processed/{dataset_name}_qrels.tsv"
    with open(qrels_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头
        for line in f:
            query_id, doc_id, relevance = line.strip().split('\t')
            if query_id not in relevance_judgments:
                relevance_judgments[query_id] = {}
            relevance_judgments[query_id][doc_id] = int(relevance)
    
    print(f"数据集加载完成: 文档数量={len(documents)}, 查询数量={len(queries)}")
    return documents, queries, relevance_judgments


class AdaptiveFusionPipeline:
    """自适应融合完整流水线"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 创建组件
        self.query_analyzer = create_simple_query_analyzer(config.get('query_analyzer', {}))
        self.adaptive_router = create_simple_adaptive_router(config.get('adaptive_router', {}))
        self.adaptive_fusion = create_simple_adaptive_fusion(config.get('adaptive_fusion', {}))
        
        # 检索器
        self.retrievers = {}
        
        # 统计信息
        self.pipeline_stats = {
            'total_queries': 0,
            'avg_processing_time': 0,
            'routing_decisions': [],
            'fusion_results': []
        }
    
    def setup_retrievers(self, documents: List[Document], dataset_name: str):
        """设置检索器"""
        print("设置检索器...")
        
        # BM25检索器
        bm25_retriever = BM25Retriever(self.config.get('bm25', {}))
        cache_dir = Path("checkpoints/retriever_cache")
        bm25_cache_path = cache_dir / f"bm25_{dataset_name}_index.pkl"
        
        if bm25_cache_path.exists():
            bm25_retriever.load_index(str(bm25_cache_path))
            print("✅ BM25索引从缓存加载")
        else:
            print("🔄 构建BM25索引...")
            bm25_retriever.build_index(documents)
        
        self.retrievers['BM25'] = bm25_retriever
        
        # 向量检索器
        vector_retriever = EfficientVectorIndex("EfficientVector", self.config.get('efficient_vector', {}))
        vector_cache_path = cache_dir / f"efficientvector_{dataset_name}_index.pkl"
        
        if vector_cache_path.exists():
            vector_retriever.load_index(str(vector_cache_path))
            print("✅ 向量索引从缓存加载")
        else:
            print("🔄 构建向量索引...")
            vector_retriever.build_index(documents)
        
        self.retrievers['EfficientVector'] = vector_retriever
        
        print(f"检索器设置完成: {list(self.retrievers.keys())}")
    
    def process_query(self, query: Query, top_k: int = 10) -> Tuple[List[FusionResult], Dict[str, Any]]:
        """处理单个查询"""
        start_time = time.time()
        
        # 1. 查询分析
        query_features = self.query_analyzer.analyze_query(query)
        
        # 2. 自适应路由
        routing_decision = self.adaptive_router.route(query_features)
        
        # 3. 执行检索
        retrieval_results = {}
        for retriever_name in routing_decision.selected_retrievers:
            if retriever_name in self.retrievers:
                results = self.retrievers[retriever_name].retrieve(query, top_k * 2)  # 获取更多结果用于融合
                retrieval_results[retriever_name] = results
        
        # 4. 自适应融合
        fusion_results = self.adaptive_fusion.fuse(query, retrieval_results, routing_decision, top_k)
        
        # 5. 记录统计信息
        processing_time = (time.time() - start_time) * 1000  # 毫秒
        
        query_stats = {
            'query_id': query.query_id,
            'query_text': query.text,
            'query_features': query_features.to_dict(),
            'routing_decision': routing_decision.to_dict(),
            'processing_time': processing_time,
            'num_results': len(fusion_results)
        }
        
        return fusion_results, query_stats
    
    def run_experiment(self, queries: List[Query], relevance_judgments: Dict[str, Dict[str, int]], 
                      sample_size: Optional[int] = None, top_k: int = 10) -> Dict[str, Any]:
        """运行完整实验"""
        
        # 采样查询
        if sample_size and sample_size < len(queries):
            import random
            random.seed(42)
            queries = random.sample(queries, sample_size)
            print(f"随机抽样 {sample_size} 个查询")
        
        print(f"开始处理 {len(queries)} 个查询...")
        
        all_fusion_results = []
        all_query_stats = []
        
        for i, query in enumerate(queries):
            if (i + 1) % 20 == 0:
                print(f"处理进度: {i + 1}/{len(queries)}")
            
            try:
                fusion_results, query_stats = self.process_query(query, top_k)
                all_fusion_results.append(fusion_results)
                all_query_stats.append(query_stats)
                
            except Exception as e:
                print(f"处理查询 {query.query_id} 失败: {e}")
                continue
        
        # 评估结果
        print("评估自适应融合结果...")
        metrics = self._evaluate_results(all_fusion_results, queries[:len(all_fusion_results)], relevance_judgments, top_k)
        
        # 生成统计报告
        stats_report = self._generate_stats_report(all_query_stats)
        
        return {
            'metrics': metrics,
            'statistics': stats_report,
            'query_details': all_query_stats
        }
    
    def _evaluate_results(self, fusion_results_list: List[List[FusionResult]], 
                         queries: List[Query], 
                         relevance_judgments: Dict[str, Dict[str, int]], 
                         top_k: int = 10) -> Dict[str, float]:
        """评估融合结果"""
        import numpy as np
        
        all_mrr = []
        all_ndcg = []
        all_precision = []
        all_recall = []
        
        for query, fusion_results in zip(queries, fusion_results_list):
            query_id = query.query_id
            
            if query_id not in relevance_judgments:
                continue
            
            relevant_docs = set(relevance_judgments[query_id].keys())
            retrieved_docs = [r.doc_id for r in fusion_results[:top_k]]
            
            if retrieved_docs:
                # Precision@k
                relevant_retrieved = len(set(retrieved_docs) & relevant_docs)
                precision = relevant_retrieved / len(retrieved_docs)
                all_precision.append(precision)
                
                # Recall@k
                recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
                all_recall.append(recall)
                
                # MRR
                mrr = 0
                for i, doc_id in enumerate(retrieved_docs):
                    if doc_id in relevant_docs:
                        mrr = 1.0 / (i + 1)
                        break
                all_mrr.append(mrr)
                
                # NDCG@k (简化版本)
                dcg = 0
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_docs), top_k)))
                
                for i, doc_id in enumerate(retrieved_docs):
                    if doc_id in relevant_docs:
                        dcg += 1.0 / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0
                all_ndcg.append(ndcg)
        
        return {
            'precision': np.mean(all_precision) if all_precision else 0,
            'recall': np.mean(all_recall) if all_recall else 0,
            'mrr': np.mean(all_mrr) if all_mrr else 0,
            'ndcg': np.mean(all_ndcg) if all_ndcg else 0
        }
    
    def _generate_stats_report(self, query_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成统计报告"""
        if not query_stats:
            return {}
        
        # 处理时间统计
        processing_times = [stat['processing_time'] for stat in query_stats]
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        # 查询类型分布
        query_types = [stat['query_features']['query_type'] for stat in query_stats]
        type_distribution = {}
        for qtype in query_types:
            type_distribution[qtype] = type_distribution.get(qtype, 0) + 1
        
        # 融合方法使用统计
        fusion_methods = [stat['routing_decision']['fusion_method'] for stat in query_stats]
        method_distribution = {}
        for method in fusion_methods:
            method_distribution[method] = method_distribution.get(method, 0) + 1
        
        # 检索器选择统计
        retriever_usage = {}
        for stat in query_stats:
            for retriever in stat['routing_decision']['selected_retrievers']:
                retriever_usage[retriever] = retriever_usage.get(retriever, 0) + 1
        
        return {
            'total_queries': len(query_stats),
            'avg_processing_time_ms': avg_processing_time,
            'query_type_distribution': type_distribution,
            'fusion_method_distribution': method_distribution,
            'retriever_usage': retriever_usage,
            'router_stats': self.adaptive_router.get_statistics(),
            'fusion_stats': self.adaptive_fusion.get_statistics()
        }


def run_adaptive_fusion_experiment(dataset_name: str, config: Dict[str, Any], 
                                 sample_size: Optional[int] = None, top_k: int = 10) -> Dict[str, Any]:
    """运行自适应融合实验"""
    print(f"\n=== 运行 {dataset_name} 自适应融合实验 ===")
    
    # 加载数据集
    documents, queries, relevance_judgments = load_dataset(dataset_name)
    
    # 创建流水线
    pipeline = AdaptiveFusionPipeline(config)
    
    # 设置检索器
    pipeline.setup_retrievers(documents, dataset_name)
    
    # 运行实验
    results = pipeline.run_experiment(queries, relevance_judgments, sample_size, top_k)
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行自适应融合实验")
    parser.add_argument("--config", type=str, default="configs/paper_experiments.json", help="配置文件路径")
    parser.add_argument("--datasets", type=str, nargs="+", default=["fiqa", "quora", "scidocs"], help="数据集列表")
    parser.add_argument("--sample", type=int, default=50, help="查询样本大小")
    parser.add_argument("--top_k", type=int, default=10, help="检索文档数量")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print("🚀 开始运行自适应融合实验")
    print("=" * 50)
    
    all_results = {}
    
    for dataset in args.datasets:
        try:
            results = run_adaptive_fusion_experiment(dataset, config, args.sample, args.top_k)
            all_results[dataset] = results
            
            # 显示结果摘要
            metrics = results['metrics']
            stats = results['statistics']
            
            print(f"\n{dataset} 自适应融合结果:")
            print(f"  MRR: {metrics.get('mrr', 0):.3f}")
            print(f"  NDCG: {metrics.get('ndcg', 0):.3f}")
            print(f"  Precision: {metrics.get('precision', 0):.3f}")
            print(f"  Recall: {metrics.get('recall', 0):.3f}")
            print(f"  平均处理时间: {stats.get('avg_processing_time_ms', 0):.1f}ms")
            print(f"  查询类型分布: {stats.get('query_type_distribution', {})}")
            print(f"  融合方法分布: {stats.get('fusion_method_distribution', {})}")
            
        except Exception as e:
            print(f"数据集 {dataset} 实验失败: {e}")
            continue
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"reports/adaptive_fusion_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n自适应融合实验完成! 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()