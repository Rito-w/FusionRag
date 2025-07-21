#!/usr/bin/env python3
"""
增强版消融实验
使用增强的查询分析器和自适应路由器进行消融实验
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
from modules.analysis.enhanced_query_analyzer import create_enhanced_query_analyzer
from modules.adaptive.enhanced_adaptive_router import create_enhanced_adaptive_router
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


class EnhancedAblationBaseline:
    """增强版消融实验基线类
    
    实现不同的消融版本来分析各组件的贡献
    """
    
    def __init__(self, config: Dict[str, Any], ablation_type: str = "full"):
        self.config = config
        self.ablation_type = ablation_type
        
        # 根据消融类型决定是否使用查询分析器
        if ablation_type != "no_query_analyzer":
            self.query_analyzer = create_enhanced_query_analyzer(config.get('query_analyzer', {}))
        else:
            self.query_analyzer = None
        
        # 根据消融类型决定是否使用自适应路由器
        if ablation_type != "no_adaptive_routing":
            self.router = create_enhanced_adaptive_router(config.get('adaptive_router', {}))
        else:
            self.router = None
        
        # 统计信息
        self.stats = {
            'total_queries': 0,
            'strategy_usage': defaultdict(int),
            'query_type_distribution': defaultdict(int),
            'ablation_type': ablation_type
        }
    
    def fusion(self, query: Query, bm25_results: List[RetrievalResult], 
               vector_results: List[RetrievalResult], dataset_name: str, 
               top_k: int = 10) -> List[FusionResult]:
        """根据消融类型选择融合策略"""
        
        self.stats['total_queries'] += 1
        
        if self.ablation_type == "full":
            # 完整方法：智能路由 + 查询分析
            return self._smart_fusion(query, bm25_results, vector_results, dataset_name, top_k)
            
        elif self.ablation_type == "no_query_analyzer":
            # 无查询分析器：使用固定策略
            strategy = 'rrf_standard'
            fusion_results = self._rrf_fusion(bm25_results, vector_results, 60)
            self.stats['strategy_usage'][strategy] += 1
            return fusion_results[:top_k]
            
        elif self.ablation_type == "no_adaptive_routing":
            # 无自适应路由：只使用查询分析，但固定使用RRF
            if self.query_analyzer:
                features = self.query_analyzer.analyze_query(query)
                self.stats['query_type_distribution'][features.final_type] += 1
            
            strategy = 'rrf_standard'
            fusion_results = self._rrf_fusion(bm25_results, vector_results, 60)
            self.stats['strategy_usage'][strategy] += 1
            return fusion_results[:top_k]
            
        elif self.ablation_type == "static_weights":
            # 静态权重：使用查询分析但固定权重
            if self.query_analyzer:
                features = self.query_analyzer.analyze_query(query)
                self.stats['query_type_distribution'][features.final_type] += 1
            
            strategy = 'linear_equal'
            fusion_results = self._linear_weighted_fusion(bm25_results, vector_results, [0.5, 0.5])
            self.stats['strategy_usage'][strategy] += 1
            return fusion_results[:top_k]
            
        else:
            # 默认使用RRF
            strategy = 'rrf_default'
            fusion_results = self._rrf_fusion(bm25_results, vector_results, 60)
            self.stats['strategy_usage'][strategy] += 1
            return fusion_results[:top_k]
    
    def _smart_fusion(self, query: Query, bm25_results: List[RetrievalResult], 
                     vector_results: List[RetrievalResult], dataset_name: str,
                     top_k: int = 10) -> List[FusionResult]:
        """完整的智能融合策略"""
        
        # 分析查询特征
        features = self.query_analyzer.analyze_query(query)
        self.stats['query_type_distribution'][features.final_type] += 1
        
        # 使用自适应路由器选择策略
        strategy_name, strategy_info, _ = self.router.select_strategy(features, dataset_name)
        self.stats['strategy_usage'][strategy_name] += 1
        
        # 应用选择的策略
        fusion_results = self.router.apply_strategy(
            strategy_name, strategy_info, bm25_results, vector_results, top_k
        )
        
        return fusion_results
    
    def _rrf_fusion(self, bm25_results: List[RetrievalResult], 
                   vector_results: List[RetrievalResult], k: int) -> List[FusionResult]:
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
    
    def _linear_weighted_fusion(self, bm25_results: List[RetrievalResult], 
                               vector_results: List[RetrievalResult], 
                               weights: List[float]) -> List[FusionResult]:
        """线性加权融合"""
        all_docs = {}
        doc_scores = defaultdict(dict)
        
        # 归一化BM25分数
        if bm25_results:
            bm25_scores = [r.score for r in bm25_results]
            bm25_max = max(bm25_scores) if bm25_scores else 1.0
            bm25_min = min(bm25_scores) if bm25_scores else 0.0
            bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0
            
            for result in bm25_results:
                doc_id = result.doc_id
                all_docs[doc_id] = result.document
                normalized_score = (result.score - bm25_min) / bm25_range if bm25_range > 0 else 0.5
                doc_scores[doc_id]['BM25'] = {
                    'normalized': normalized_score,
                    'original': result.score,
                    'weight': weights[0]
                }
        
        # 归一化向量分数
        if vector_results:
            vector_scores = [r.score for r in vector_results]
            vector_max = max(vector_scores) if vector_scores else 1.0
            vector_min = min(vector_scores) if vector_scores else 0.0
            vector_range = vector_max - vector_min if vector_max > vector_min else 1.0
            
            for result in vector_results:
                doc_id = result.doc_id
                all_docs[doc_id] = result.document
                normalized_score = (result.score - vector_min) / vector_range if vector_range > 0 else 0.5
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {}
                doc_scores[doc_id]['EfficientVector'] = {
                    'normalized': normalized_score,
                    'original': result.score,
                    'weight': weights[1]
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


def run_enhanced_ablation_experiment(dataset_name: str, config: Dict[str, Any], 
                                    ablation_type: str, sample_size: Optional[int] = None, 
                                    top_k: int = 10) -> Dict[str, Any]:
    """运行增强版消融实验"""
    print(f"\n=== 运行 {dataset_name} 增强版消融实验 ({ablation_type}) ===")
    
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
    
    # 创建消融基线
    ablation_baseline = EnhancedAblationBaseline(config, ablation_type)
    
    # 执行检索和融合
    print(f"执行增强版消融实验 ({ablation_type})...")
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
                print(f"警告: 查询 {query.query_id} 没有任何检索结果")
                all_fusion_results.append([])
                continue
            
            # 消融融合
            fusion_results = ablation_baseline.fusion(query, bm25_results, vector_results, dataset_name, top_k)
            all_fusion_results.append(fusion_results)
            
        except Exception as e:
            print(f"处理查询 {query.query_id} 时出错: {e}")
            all_fusion_results.append([])
            continue
    
    # 评估结果
    print(f"评估增强版消融实验结果 ({ablation_type})...")
    metrics = evaluate_fusion_results(all_fusion_results, queries, relevance_judgments, top_k)
    
    return {
        'metrics': metrics,
        'statistics': ablation_baseline.stats
    }


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
        
        # 添加调试信息
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
    parser = argparse.ArgumentParser(description="运行增强版消融实验")
    parser.add_argument("--config", type=str, default="configs/paper_experiments.json", help="配置文件路径")
    parser.add_argument("--datasets", type=str, nargs="+", default=["fiqa", "quora", "scidocs"], help="数据集列表")
    parser.add_argument("--sample", type=int, default=100, help="查询样本大小")
    parser.add_argument("--top_k", type=int, default=10, help="检索文档数量")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print("🚀 开始运行增强版消融实验")
    print("=" * 50)
    
    # 消融实验类型
    ablation_types = [
        "full",                    # 完整方法
        "no_query_analyzer",       # 无查询分析器
        "no_adaptive_routing",     # 无自适应路由
        "static_weights"           # 静态权重
    ]
    
    all_results = {}
    
    for dataset in args.datasets:
        all_results[dataset] = {}
        
        for ablation_type in ablation_types:
            try:
                results = run_enhanced_ablation_experiment(dataset, config, ablation_type, args.sample, args.top_k)
                all_results[dataset][ablation_type] = results
                
                # 显示结果摘要
                metrics = results['metrics']
                stats = results['statistics']
                
                print(f"\n{dataset} - {ablation_type} 结果:")
                print(f"  MRR: {metrics.get('mrr', 0):.3f}")
                print(f"  NDCG: {metrics.get('ndcg', 0):.3f}")
                print(f"  Precision: {metrics.get('precision', 0):.3f}")
                print(f"  Recall: {metrics.get('recall', 0):.3f}")
                
            except Exception as e:
                print(f"数据集 {dataset} 消融实验 {ablation_type} 失败: {e}")
                continue
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"reports/enhanced_ablation_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n增强版消融实验完成! 结果已保存到: {output_file}")
    
    # 分析结果
    print("\n=== 增强版消融实验结果分析 ===")
    for dataset in all_results:
        print(f"\n{dataset} 数据集:")
        for ablation_type in ablation_types:
            if ablation_type in all_results[dataset]:
                metrics = all_results[dataset][ablation_type]['metrics']
                print(f"  {ablation_type:20s}: MRR={metrics.get('mrr', 0):.3f}, NDCG={metrics.get('ndcg', 0):.3f}")


if __name__ == "__main__":
    main()