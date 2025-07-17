"""
实验运行脚本
用于在多个数据集上评估自适应混合索引的性能
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

from modules.adaptive_hybrid_index import AdaptiveHybridIndex, create_adaptive_hybrid_index
from modules.retriever.bm25_retriever import BM25Retriever
from modules.retriever.dense_retriever import DenseRetriever
from modules.retriever.efficient_vector_index import EfficientVectorIndex
from modules.retriever.semantic_bm25 import SemanticBM25
# from modules.retriever.graph_retriever import GraphRetriever
from modules.retriever.cascade_retriever import CascadeRetriever
from modules.evaluation.evaluator import IndexEvaluator
from modules.utils.interfaces import Query, Document


def load_dataset(dataset_name: str) -> Tuple[List[Document], List[Query], Dict[str, Dict[str, int]]]:
    """加载数据集"""
    print(f"加载数据集: {dataset_name}")
    
    # 加载文档
    documents = []
    corpus_path = f"data/processed/{dataset_name}_corpus.jsonl"
    print(f"加载文档: {corpus_path}")
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
    print(f"加载查询: {queries_path}")
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
    print(f"加载相关性判断: {qrels_path}")
    with open(qrels_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头
        for line in f:
            query_id, doc_id, relevance = line.strip().split('\t')
            if query_id not in relevance_judgments:
                relevance_judgments[query_id] = {}
            relevance_judgments[query_id][doc_id] = int(relevance)
    
    print(f"数据集加载完成: 文档数量={len(documents)}, 查询数量={len(queries)}, 相关性判断数量={len(relevance_judgments)}")
    return documents, queries, relevance_judgments


def create_retrievers(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """创建检索器"""
    config = config or {}
    
    # 只保留 SemanticBM25 检索器
    retrievers = {
        "SemanticBM25": SemanticBM25(config.get('semantic_bm25', {})),
    }
    
    return retrievers

def run_experiment(dataset_name: str, config: Dict[str, Any] = None, 
                top_k: int = 10, sample_size: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """运行实验"""
    # 加载数据集
    documents, queries, relevance_judgments = load_dataset(dataset_name)
    
    # 如果指定了样本大小，随机抽样
    if sample_size and sample_size < len(queries):
        import random
        random.seed(42)  # 固定随机种子，确保可重复性
        queries = random.sample(queries, sample_size)
        print(f"随机抽样 {sample_size} 个查询")
    
    # 创建检索器
    print("创建检索器...")
    retrievers = create_retrievers(config)
    
    # 构建索引
    print("构建索引...")
    for name, retriever in retrievers.items():
        print(f"构建 {name} 索引...")
        start_time = time.time()
        retriever.build_index(documents)
        end_time = time.time()
        print(f"{name} 索引构建完成，耗时: {end_time - start_time:.2f}秒")
    
    # 创建评估器
    evaluator_config = config.get('evaluator', {})
    evaluator = IndexEvaluator(evaluator_config)
    
    # 评估性能
    print("评估性能...")
    results = evaluator.evaluate_multiple_retrievers(retrievers, queries, relevance_judgments, top_k)
    
    # 生成报告
    report_dir = f"reports/{dataset_name}"
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{report_dir}/evaluation_{timestamp}.json"
    
    evaluator.generate_report(results, dataset_name, report_file)
    
    return results


def run_ablation_study(dataset_name: str, config: Dict[str, Any] = None, 
                     top_k: int = 10, sample_size: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """运行消融实验"""
    # 加载数据集
    documents, queries, relevance_judgments = load_dataset(dataset_name)
    
    # 如果指定了样本大小，随机抽样
    if sample_size and sample_size < len(queries):
        import random
        random.seed(42)
        queries = random.sample(queries, sample_size)
        print(f"随机抽样 {sample_size} 个查询")
    
    # 创建基础检索器
    print("创建检索器...")
    base_retriever = create_adaptive_hybrid_index(config)
    
    # 创建消融实验检索器
    component_retrievers = {
        "NoQueryAnalysis": create_adaptive_hybrid_index({
            **config,
            'query_analyzer': {'disabled': True}
        }),
        "NoAdaptiveRouting": create_adaptive_hybrid_index({
            **config,
            'adaptive_router': {'disabled': True}
        }),
        "NoAdaptiveFusion": create_adaptive_hybrid_index({
            **config,
            'adaptive_fusion': {'disabled': True}
        }),
        "OnlyBM25": BM25Retriever(config.get('bm25', {})),
        "OnlyVector": EfficientVectorIndex(config.get('efficient_vector', {}))
    }
    
    # 构建索引
    print("构建基础索引...")
    base_retriever.build_index(documents)
    
    print("构建消融实验索引...")
    for name, retriever in component_retrievers.items():
        print(f"构建 {name} 索引...")
        retriever.build_index(documents)
    
    # 创建评估器
    evaluator_config = config.get('evaluator', {})
    evaluator = IndexEvaluator(evaluator_config)
    
    # 运行消融实验
    print("运行消融实验...")
    results = evaluator.run_ablation_study(
        base_retriever, component_retrievers, queries, relevance_judgments, top_k
    )
    
    # 生成报告
    report_dir = f"reports/{dataset_name}/ablation"
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{report_dir}/ablation_{timestamp}.json"
    
    evaluator.generate_report(results, f"{dataset_name}_ablation", report_file)
    
    return results

def run_query_type_analysis(dataset_name: str, config: Dict[str, Any] = None,
                          top_k: int = 10, sample_size: Optional[int] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """运行查询类型分析"""
    # 加载数据集
    documents, queries, relevance_judgments = load_dataset(dataset_name)
    
    # 如果指定了样本大小，随机抽样
    if sample_size and sample_size < len(queries):
        import random
        random.seed(42)
        queries = random.sample(queries, sample_size)
        print(f"随机抽样 {sample_size} 个查询")
    
    # 创建检索器
    print("创建检索器...")
    retrievers = create_retrievers(config)
    
    # 构建索引
    print("构建索引...")
    for name, retriever in retrievers.items():
        print(f"构建 {name} 索引...")
        retriever.build_index(documents)
    
    # 分析查询类型
    print("分析查询类型...")
    query_analyzer = retrievers["AdaptiveHybrid"].query_analyzer
    query_types = {}
    
    for query in queries:
        features = query_analyzer.analyze_query(query)
        query_types[query.query_id] = features.query_type.value
    
    # 创建评估器
    evaluator_config = config.get('evaluator', {})
    evaluator = IndexEvaluator({**evaluator_config, 'query_types': query_types})
    
    # 按查询类型评估
    print("按查询类型评估...")
    results = evaluator.evaluate_by_query_type(retrievers, queries, relevance_judgments, top_k)
    
    # 生成报告
    report_dir = f"reports/{dataset_name}/query_types"
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{report_dir}/query_types_{timestamp}.json"
    
    # 保存结果
    with open(report_file, 'w') as f:
        json.dump({
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "query_type_distribution": {k: list(query_types.values()).count(k) for k in set(query_types.values())},
            "results": {k: {r: v for r, v in v.items()} for k, v in results.items()}
        }, f, indent=2)
    
    print(f"查询类型分析报告已保存到: {report_file}")
    
    return results


def run_all_experiments(config_path: str = "configs/experiments.json", 
                      datasets: List[str] = None,
                      top_k: int = 10,
                      sample_size: Optional[int] = None) -> None:
    """运行所有实验"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 如果没有指定数据集，使用配置中的数据集
    if datasets is None:
        datasets = config.get('datasets', ['nfcorpus', 'scifact', 'trec-covid'])
    
    all_results = {}
    
    # 运行基本实验
    for dataset in datasets:
        print(f"\n=== 运行 {dataset} 基本实验 ===\n")
        results = run_experiment(dataset, config, top_k, sample_size)
        all_results[dataset] = results
    
    # 运行消融实验
    for dataset in datasets:
        print(f"\n=== 运行 {dataset} 消融实验 ===\n")
        ablation_results = run_ablation_study(dataset, config, top_k, sample_size)
        all_results[f"{dataset}_ablation"] = ablation_results
    
    # 运行查询类型分析
    for dataset in datasets:
        print(f"\n=== 运行 {dataset} 查询类型分析 ===\n")
        query_type_results = run_query_type_analysis(dataset, config, top_k, sample_size)
        # 查询类型结果结构较复杂，不合并到all_results
    
    # 保存总体结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"reports/all_results_{timestamp}.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n所有实验完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行自适应混合索引实验")
    parser.add_argument("--config", type=str, default="configs/experiments.json", help="实验配置文件路径")
    parser.add_argument("--datasets", type=str, nargs="+", help="要评估的数据集")
    parser.add_argument("--top_k", type=int, default=10, help="检索的文档数量")
    parser.add_argument("--sample", type=int, help="查询样本大小，用于快速测试")
    
    args = parser.parse_args()
    
    run_all_experiments(args.config, args.datasets, args.top_k, args.sample)