#!/usr/bin/env python3
"""
融合策略对比实验
对比不同融合方法的效果，分析动态权重vs静态权重的优势
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
from modules.analysis.simple_query_analyzer import SimpleQueryAnalyzer, create_simple_query_analyzer
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


class FusionStrategyComparator:
    """融合策略对比器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.query_analyzer = create_simple_query_analyzer(config.get('query_analyzer', {}))
        
        # 统计信息
        self.stats = {
            'total_queries': 0,
            'strategy_usage': defaultdict(int),
            'query_type_distribution': defaultdict(int)
        }
    
    def apply_fusion_strategy(self, strategy: str, query: Query, 
                            bm25_results: List[RetrievalResult], 
                            vector_results: List[RetrievalResult], 
                            top_k: int = 10) -> List[FusionResult]:
        """应用指定的融合策略"""
        
        self.stats['total_queries'] += 1
        self.stats['strategy_usage'][strategy] += 1
        
        # 分析查询特征（用于某些策略）
        features = self.query_analyzer.analyze_query(query)
        self.stats['query_type_distribution'][features.query_type.value] += 1
        
        if strategy == "rrf_standard":
            return self._rrf_fusion(bm25_results, vector_results, k=60)[:top_k]
            
        elif strategy == "rrf_optimized":
            return self._rrf_fusion(bm25_results, vector_results, k=30)[:top_k]
            
        elif strategy == "linear_equal":
            return self._weighted_fusion(bm25_results, vector_results, 0.5, 0.5)[:top_k]
            
        elif strategy == "linear_bm25_dominant":
            return self._weighted_fusion(bm25_results, vector_results, 0.7, 0.3)[:top_k]
            
        elif strategy == "linear_vector_dominant":
            return self._weighted_fusion(bm25_results, vector_results, 0.3, 0.7)[:top_k]
            
        elif strategy == "adaptive_by_query_type":
            return self._adaptive_by_query_type(query, features, bm25_results, vector_results)[:top_k]
            
        elif strategy == "adaptive_by_length":
            return self._adaptive_by_length(query, bm25_results, vector_results)[:top_k]
            
        elif strategy == "max_score":
            return self._max_score_fusion(bm25_results, vector_results)[:top_k]
            
        else:
            # 默认使用标准RRF
            return self._rrf_fusion(bm25_results, vector_results, k=60)[:top_k]
    
    def _rrf_fusion(self, bm25_results: List[RetrievalResult], 
                   vector_results: List[RetrievalResult], k: int = 60) -> List[FusionResult]:
        """RRF融合"""
        all_docs = {}
        doc_ranks = defaultdict(dict)
        
        # BM25结果排名
        for rank, result in enumerate(bm25_results):
            doc_id = result.doc_id
            all_docs[doc_id] = result.document
            doc_ranks[doc_id]['BM25'] = rank + 1
        
        # 向量结果排名
        for rank, result in enumerate(vector_results):
            doc_id = result.doc_id
            all_docs[doc_id] = result.document
            doc_ranks[doc_id]['EfficientVector'] = rank + 1
        
        # 计算RRF分数
        fusion_results = []
        for doc_id, ranks in doc_ranks.items():
            rrf_score = sum(1.0 / (k + rank) for rank in ranks.values())
            
            # 收集原始分数
            individual_scores = {}
            for result in bm25_results:
                if result.doc_id == doc_id:
                    individual_scores['BM25'] = result.score
                    break
            for result in vector_results:
                if result.doc_id == doc_id:
                    individual_scores['EfficientVector'] = result.score
                    break
            
            fusion_result = FusionResult(
                doc_id=doc_id,
                final_score=rrf_score,
                document=all_docs[doc_id],
                individual_scores=individual_scores
            )
            fusion_results.append(fusion_result)
        
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        return fusion_results
    
    def _weighted_fusion(self, bm25_results: List[RetrievalResult], 
                        vector_results: List[RetrievalResult], 
                        bm25_weight: float, vector_weight: float) -> List[FusionResult]:
        """加权融合"""
        all_docs = {}
        doc_scores = defaultdict(dict)
        
        # 归一化BM25分数
        if bm25_results:
            bm25_scores = [r.score for r in bm25_results]
            bm25_max = max(bm25_scores)
            bm25_min = min(bm25_scores)
            bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0
            
            for result in bm25_results:
                doc_id = result.doc_id
                all_docs[doc_id] = result.document
                normalized_score = (result.score - bm25_min) / bm25_range
                doc_scores[doc_id]['BM25'] = {
                    'normalized': normalized_score,
                    'original': result.score,
                    'weight': bm25_weight
                }
        
        # 归一化向量分数
        if vector_results:
            vector_scores = [r.score for r in vector_results]
            vector_max = max(vector_scores)
            vector_min = min(vector_scores)
            vector_range = vector_max - vector_min if vector_max > vector_min else 1.0
            
            for result in vector_results:
                doc_id = result.doc_id
                all_docs[doc_id] = result.document
                normalized_score = (result.score - vector_min) / vector_range
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {}
                doc_scores[doc_id]['EfficientVector'] = {
                    'normalized': normalized_score,
                    'original': result.score,
                    'weight': vector_weight
                }
        
        # 计算加权分数
        fusion_results = []
        for doc_id, scores_dict in doc_scores.items():
            weighted_score = sum(
                info['normalized'] * info['weight'] 
                for info in scores_dict.values()
            )
            
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
        
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        return fusion_results
    
    def _adaptive_by_query_type(self, query: Query, features, 
                               bm25_results: List[RetrievalResult], 
                               vector_results: List[RetrievalResult]) -> List[FusionResult]:
        """根据查询类型自适应融合"""
        if features.query_type.value == 'entity':
            # 实体查询偏向BM25
            return self._weighted_fusion(bm25_results, vector_results, 0.6, 0.4)
        elif features.query_type.value == 'semantic':
            # 语义查询偏向向量
            return self._weighted_fusion(bm25_results, vector_results, 0.4, 0.6)
        else:
            # 关键词查询使用RRF
            return self._rrf_fusion(bm25_results, vector_results, k=60)
    
    def _adaptive_by_length(self, query: Query, 
                           bm25_results: List[RetrievalResult], 
                           vector_results: List[RetrievalResult]) -> List[FusionResult]:
        """根据查询长度自适应融合"""
        word_count = len(query.text.split())
        
        if word_count <= 3:
            # 短查询使用RRF
            return self._rrf_fusion(bm25_results, vector_results, k=60)
        elif word_count <= 8:
            # 中等查询偏向向量
            return self._weighted_fusion(bm25_results, vector_results, 0.4, 0.6)
        else:
            # 长查询更偏向向量
            return self._weighted_fusion(bm25_results, vector_results, 0.3, 0.7)
    
    def _max_score_fusion(self, bm25_results: List[RetrievalResult], 
                         vector_results: List[RetrievalResult]) -> List[FusionResult]:
        """最大分数融合"""
        all_docs = {}
        doc_scores = defaultdict(dict)
        
        # 归一化BM25分数
        if bm25_results:
            bm25_scores = [r.score for r in bm25_results]
            bm25_max = max(bm25_scores)
            bm25_min = min(bm25_scores)
            bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0
            
            for result in bm25_results:
                doc_id = result.doc_id
                all_docs[doc_id] = result.document
                normalized_score = (result.score - bm25_min) / bm25_range
                doc_scores[doc_id]['BM25'] = normalized_score
        
        # 归一化向量分数
        if vector_results:
            vector_scores = [r.score for r in vector_results]
            vector_max = max(vector_scores)
            vector_min = min(vector_scores)
            vector_range = vector_max - vector_min if vector_max > vector_min else 1.0
            
            for result in vector_results:
                doc_id = result.doc_id
                all_docs[doc_id] = result.document
                normalized_score = (result.score - vector_min) / vector_range
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {}
                doc_scores[doc_id]['EfficientVector'] = normalized_score
        
        # 计算最大分数
        fusion_results = []
        for doc_id, scores_dict in doc_scores.items():
            max_score = max(scores_dict.values())
            
            fusion_result = FusionResult(
                doc_id=doc_id,
                final_score=max_score,
                document=all_docs[doc_id],
                individual_scores=scores_dict
            )
            fusion_results.append(fusion_result)
        
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        return fusion_results


def run_fusion_strategy_comparison(dataset_name: str, config: Dict[str, Any], 
                                  sample_size: Optional[int] = None, 
                                  top_k: int = 10) -> Dict[str, Any]:
    """运行融合策略对比实验"""
    print(f"\n=== 运行 {dataset_name} 融合策略对比实验 ===")
    
    # 加载数据集
    documents, queries, relevance_judgments = load_dataset(dataset_name)
    
    # 采样查询
    if sample_size and sample_size < len(queries):
        import random
        random.seed(42)
        queries = random.sample(queries, sample_size)
        print(f"随机抽样 {sample_size} 个查询")
    
    # 创建检索器
    print("设置检索器...")
    bm25_retriever = BM25Retriever(config.get('bm25', {}))
    vector_retriever = EfficientVectorIndex("EfficientVector", config.get('efficient_vector', {}))
    
    # 加载索引
    cache_dir = Path("checkpoints/retriever_cache")
    
    bm25_cache_path = cache_dir / f"bm25_{dataset_name}_index.pkl"
    if bm25_cache_path.exists():
        bm25_retriever.load_index(str(bm25_cache_path))
        print("✅ BM25索引从缓存加载")
    else:
        print("🔄 构建BM25索引...")
        bm25_retriever.build_index(documents)
    
    vector_cache_path = cache_dir / f"efficientvector_{dataset_name}_index.pkl"
    if vector_cache_path.exists():
        vector_retriever.load_index(str(vector_cache_path))
        print("✅ 向量索引从缓存加载")
    else:
        print("🔄 构建向量索引...")
        vector_retriever.build_index(documents)
    
    # 融合策略列表
    strategies = [
        "rrf_standard",           # 标准RRF (k=60)
        "rrf_optimized",          # 优化RRF (k=30)
        "linear_equal",           # 等权重线性融合
        "linear_bm25_dominant",   # BM25主导
        "linear_vector_dominant", # 向量主导
        "adaptive_by_query_type", # 根据查询类型自适应
        "adaptive_by_length",     # 根据查询长度自适应
        "max_score"               # 最大分数融合
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\n测试融合策略: {strategy}")
        
        # 创建策略对比器
        comparator = FusionStrategyComparator(config)
        
        # 执行检索和融合
        all_fusion_results = []
        
        for i, query in enumerate(queries):
            if (i + 1) % 20 == 0:
                print(f"处理进度: {i + 1}/{len(queries)}")
            
            try:
                # 执行检索
                bm25_results = bm25_retriever.retrieve(query, top_k * 2)
                vector_results = vector_retriever.retrieve(query, top_k * 2)
                
                # 检查检索结果
                if not bm25_results and not vector_results:
                    all_fusion_results.append([])
                    continue
                
                # 应用融合策略
                fusion_results = comparator.apply_fusion_strategy(
                    strategy, query, bm25_results, vector_results, top_k
                )
                all_fusion_results.append(fusion_results)
                
            except Exception as e:
                print(f"处理查询 {query.query_id} 时出错: {e}")
                all_fusion_results.append([])
                continue
        
        # 评估结果
        metrics = evaluate_fusion_results(all_fusion_results, queries, relevance_judgments, top_k)
        
        results[strategy] = {
            'metrics': metrics,
            'statistics': comparator.stats
        }
        
        print(f"  {strategy}: MRR={metrics.get('mrr', 0):.3f}, NDCG={metrics.get('ndcg', 0):.3f}")
    
    return results


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
    
    for i, (query, fusion_results) in enumerate(zip(queries, fusion_results_list)):
        query_id = query.query_id
        
        if query_id not in relevance_judgments:
            continue
        
        relevant_docs = set(relevance_judgments[query_id].keys())
        
        if not fusion_results:
            continue
            
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
            for j, doc_id in enumerate(retrieved_docs):
                if doc_id in relevant_docs:
                    mrr = 1.0 / (j + 1)
                    break
            all_mrr.append(mrr)
            
            # NDCG@k
            dcg = 0
            idcg = sum(1.0 / np.log2(j + 2) for j in range(min(len(relevant_docs), top_k)))
            
            for j, doc_id in enumerate(retrieved_docs):
                if doc_id in relevant_docs:
                    dcg += 1.0 / np.log2(j + 2)
            
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
    parser = argparse.ArgumentParser(description="运行融合策略对比实验")
    parser.add_argument("--config", type=str, default="configs/paper_experiments.json", help="配置文件路径")
    parser.add_argument("--datasets", type=str, nargs="+", default=["fiqa", "quora", "scidocs"], help="数据集列表")
    parser.add_argument("--sample", type=int, default=50, help="查询样本大小")
    parser.add_argument("--top_k", type=int, default=10, help="检索文档数量")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print("🚀 开始运行融合策略对比实验")
    print("=" * 50)
    
    all_results = {}
    
    for dataset in args.datasets:
        try:
            results = run_fusion_strategy_comparison(dataset, config, args.sample, args.top_k)
            all_results[dataset] = results
            
        except Exception as e:
            print(f"数据集 {dataset} 实验失败: {e}")
            continue
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"reports/fusion_strategy_comparison_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n融合策略对比实验完成! 结果已保存到: {output_file}")
    
    # 分析结果
    print("\n=== 融合策略对比结果分析 ===")
    for dataset in all_results:
        print(f"\n{dataset} 数据集:")
        strategies_performance = []
        for strategy, result in all_results[dataset].items():
            metrics = result['metrics']
            mrr = metrics.get('mrr', 0)
            ndcg = metrics.get('ndcg', 0)
            strategies_performance.append((strategy, mrr, ndcg))
            
        # 按MRR排序
        strategies_performance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (strategy, mrr, ndcg) in enumerate(strategies_performance):
            print(f"  {i+1:2d}. {strategy:25s}: MRR={mrr:.3f}, NDCG={ndcg:.3f}")


if __name__ == "__main__":
    main()