#!/usr/bin/env python3
"""
融合方法基线实验
实现并测试标准的融合方法：RRF和线性加权融合
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
from collections import defaultdict

from modules.retriever.bm25_retriever import BM25Retriever
from modules.retriever.efficient_vector_index import EfficientVectorIndex
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


class SimpleFusionMethods:
    """简单融合方法实现"""
    
    @staticmethod
    def reciprocal_rank_fusion(results_list: List[List[RetrievalResult]], k: int = 60) -> List[FusionResult]:
        """倒数排名融合 (RRF)"""
        if not results_list:
            return []
        
        # 收集所有文档
        all_docs = {}
        doc_ranks = defaultdict(dict)
        
        # 为每个检索器的结果计算排名
        for retriever_idx, results in enumerate(results_list):
            retriever_name = results[0].retriever_name if results else f"retriever_{retriever_idx}"
            
            for rank, result in enumerate(results):
                doc_id = result.doc_id
                all_docs[doc_id] = result.document
                doc_ranks[doc_id][retriever_name] = rank + 1  # 排名从1开始
        
        # 计算RRF分数
        fusion_results = []
        for doc_id, ranks in doc_ranks.items():
            rrf_score = sum(1.0 / (k + rank) for rank in ranks.values())
            
            # 收集各检索器的原始分数
            individual_scores = {}
            for results in results_list:
                for result in results:
                    if result.doc_id == doc_id:
                        individual_scores[result.retriever_name] = result.score
                        break
            
            fusion_result = FusionResult(
                doc_id=doc_id,
                final_score=rrf_score,
                document=all_docs[doc_id],
                individual_scores=individual_scores
            )
            fusion_results.append(fusion_result)
        
        # 按分数排序
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        return fusion_results
    
    @staticmethod
    def linear_weighted_fusion(results_list: List[List[RetrievalResult]], weights: List[float] = None) -> List[FusionResult]:
        """线性加权融合"""
        if not results_list:
            return []
        
        # 默认等权重
        if weights is None:
            weights = [1.0 / len(results_list)] * len(results_list)
        
        # 归一化权重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # 收集所有文档和分数
        all_docs = {}
        doc_scores = defaultdict(dict)
        
        for results, weight in zip(results_list, weights):
            if not results:
                continue
                
            retriever_name = results[0].retriever_name
            
            # 归一化分数到[0,1]
            scores = [r.score for r in results]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            for result in results:
                doc_id = result.doc_id
                all_docs[doc_id] = result.document
                
                # 归一化分数
                normalized_score = (result.score - min_score) / score_range
                doc_scores[doc_id][retriever_name] = {
                    'normalized': normalized_score,
                    'original': result.score,
                    'weight': weight
                }
        
        # 计算加权分数
        fusion_results = []
        for doc_id, scores_dict in doc_scores.items():
            weighted_score = sum(
                info['normalized'] * info['weight'] 
                for info in scores_dict.values()
            )
            
            # 收集各检索器的原始分数
            individual_scores = {
                retriever: info['original'] 
                for retriever, info in scores_dict.items()
            }
            
            fusion_result = FusionResult(
                doc_id=doc_id,
                final_score=weighted_score,
                document=all_docs[doc_id],
                individual_scores=individual_scores
            )
            fusion_results.append(fusion_result)
        
        # 按分数排序
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        return fusion_results


def run_fusion_experiment(dataset_name: str, config: Dict[str, Any], 
                         sample_size: Optional[int] = None, top_k: int = 10) -> Dict[str, Dict[str, float]]:
    """运行融合方法实验"""
    print(f"\n=== 运行 {dataset_name} 融合实验 ===")
    
    # 加载数据集
    documents, queries, relevance_judgments = load_dataset(dataset_name)
    
    # 采样查询
    if sample_size and sample_size < len(queries):
        import random
        random.seed(42)
        queries = random.sample(queries, sample_size)
        print(f"随机抽样 {sample_size} 个查询")
    
    # 创建检索器
    print("创建检索器...")
    bm25_retriever = BM25Retriever(config.get('bm25', {}))
    vector_retriever = EfficientVectorIndex("EfficientVector", config.get('efficient_vector', {}))
    
    # 加载索引
    cache_dir = Path("checkpoints/retriever_cache")
    
    # 加载BM25索引
    bm25_cache_path = cache_dir / f"bm25_{dataset_name}_index.pkl"
    if bm25_cache_path.exists():
        bm25_retriever.load_index(str(bm25_cache_path))
        print(f"✅ BM25索引从缓存加载")
    else:
        print("🔄 构建BM25索引...")
        bm25_retriever.build_index(documents)
    
    # 加载向量索引
    vector_cache_path = cache_dir / f"efficientvector_{dataset_name}_index.pkl"
    if vector_cache_path.exists():
        vector_retriever.load_index(str(vector_cache_path))
        print(f"✅ 向量索引从缓存加载")
    else:
        print("🔄 构建向量索引...")
        vector_retriever.build_index(documents)
    
    # 执行检索
    print("执行检索...")
    all_results = {}
    
    # 单一检索器结果
    bm25_results = []
    vector_results = []
    
    for i, query in enumerate(queries):
        if (i + 1) % 20 == 0:
            print(f"检索进度: {i + 1}/{len(queries)}")
        
        # BM25检索
        bm25_query_results = bm25_retriever.retrieve(query, top_k * 2)  # 获取更多结果用于融合
        bm25_results.append(bm25_query_results)
        
        # 向量检索
        vector_query_results = vector_retriever.retrieve(query, top_k * 2)
        vector_results.append(vector_query_results)
    
    # 融合方法
    fusion_methods = SimpleFusionMethods()
    
    print("执行融合...")
    fusion_results = {}
    
    # RRF融合
    rrf_results = []
    for i in range(len(queries)):
        query_results = [bm25_results[i], vector_results[i]]
        fused = fusion_methods.reciprocal_rank_fusion(query_results)
        rrf_results.append(fused[:top_k])  # 只取top-k
    
    # 线性加权融合 (等权重)
    linear_equal_results = []
    for i in range(len(queries)):
        query_results = [bm25_results[i], vector_results[i]]
        fused = fusion_methods.linear_weighted_fusion(query_results, [0.5, 0.5])
        linear_equal_results.append(fused[:top_k])
    
    # 线性加权融合 (优化权重，基于之前的结果给向量检索更高权重)
    linear_optimized_results = []
    for i in range(len(queries)):
        query_results = [bm25_results[i], vector_results[i]]
        fused = fusion_methods.linear_weighted_fusion(query_results, [0.3, 0.7])
        linear_optimized_results.append(fused[:top_k])
    
    # 评估结果
    print("评估融合结果...")
    evaluator = IndexEvaluator(config.get('evaluator', {}))
    
    # 转换融合结果为检索结果格式进行评估
    def fusion_to_retrieval_results(fusion_results_list: List[List[FusionResult]], method_name: str) -> List[List[RetrievalResult]]:
        converted = []
        for query_results in fusion_results_list:
            query_converted = []
            for fusion_result in query_results:
                retrieval_result = RetrievalResult(
                    doc_id=fusion_result.doc_id,
                    score=fusion_result.final_score,
                    document=fusion_result.document,
                    retriever_name=method_name
                )
                query_converted.append(retrieval_result)
            converted.append(query_converted)
        return converted
    
    # 简化评估：直接计算基本指标
    final_results = {}
    
    # 评估RRF
    rrf_metrics = evaluate_fusion_results(rrf_results, queries, relevance_judgments, top_k)
    final_results['RRF'] = rrf_metrics
    print(f"RRF: MRR={rrf_metrics.get('mrr', 0):.3f}, NDCG={rrf_metrics.get('ndcg', 0):.3f}")
    
    # 评估线性融合(等权重)
    linear_equal_metrics = evaluate_fusion_results(linear_equal_results, queries, relevance_judgments, top_k)
    final_results['LinearEqual'] = linear_equal_metrics
    print(f"LinearEqual: MRR={linear_equal_metrics.get('mrr', 0):.3f}, NDCG={linear_equal_metrics.get('ndcg', 0):.3f}")
    
    # 评估线性融合(优化权重)
    linear_opt_metrics = evaluate_fusion_results(linear_optimized_results, queries, relevance_judgments, top_k)
    final_results['LinearOptimized'] = linear_opt_metrics
    print(f"LinearOptimized: MRR={linear_opt_metrics.get('mrr', 0):.3f}, NDCG={linear_opt_metrics.get('ndcg', 0):.3f}")
    
    return final_results


def evaluate_fusion_results(fusion_results_list: List[List[FusionResult]], 
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
        
        # 计算指标
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行融合方法基线实验")
    parser.add_argument("--config", type=str, default="configs/paper_experiments.json", help="配置文件路径")
    parser.add_argument("--datasets", type=str, nargs="+", default=["fiqa", "quora", "scidocs"], help="数据集列表")
    parser.add_argument("--sample", type=int, default=100, help="查询样本大小")
    parser.add_argument("--top_k", type=int, default=10, help="检索文档数量")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print("🚀 开始运行融合方法基线实验")
    print("=" * 50)
    
    all_results = {}
    
    for dataset in args.datasets:
        try:
            results = run_fusion_experiment(dataset, config, args.sample, args.top_k)
            all_results[dataset] = results
        except Exception as e:
            print(f"数据集 {dataset} 实验失败: {e}")
            continue
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"reports/fusion_baseline_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n融合基线实验完成! 结果已保存到: {output_file}")
    
    # 显示总结
    print("\n实验结果总结:")
    for dataset, methods in all_results.items():
        print(f"\n{dataset}:")
        for method, metrics in methods.items():
            mrr = metrics.get('mrr', 0)
            ndcg = metrics.get('ndcg', 0)
            print(f"  {method}: MRR={mrr:.3f}, NDCG={ndcg:.3f}")


if __name__ == "__main__":
    main()