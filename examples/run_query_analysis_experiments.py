"""
查询类型分析脚本
专门用于分析不同查询类型下各检索方法的性能
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

from modules.adaptive_hybrid_index import AdaptiveHybridIndex, create_adaptive_hybrid_index
from modules.retriever.bm25_retriever import BM25Retriever
from modules.retriever.efficient_vector_index import EfficientVectorIndex
from modules.retriever.semantic_bm25 import SemanticBM25
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
    
    retrievers = {
        "BM25": BM25Retriever(config.get('bm25', {})),
        "EfficientVector": EfficientVectorIndex(config.get('efficient_vector', {})),
        "SemanticBM25": SemanticBM25(config.get('semantic_bm25', {})),
    }
    
    # 创建自适应混合索引
    adaptive_config = {
        'retrievers': {
            'bm25': config.get('bm25', {}),
            'efficient_vector': config.get('efficient_vector', {}),
            'semantic_bm25': config.get('semantic_bm25', {})
        },
        'query_analyzer': config.get('query_analyzer', {}),
        'adaptive_router': config.get('adaptive_router', {}),
        'adaptive_fusion': config.get('adaptive_fusion', {})
    }
    
    retrievers["AdaptiveHybrid"] = create_adaptive_hybrid_index(adaptive_config)
    
    return retrievers


def run_query_type_analysis(dataset_name: str, config: Dict[str, Any] = None,
                          top_k: int = 10, sample_size: Optional[int] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """运行查询类型分析"""
    print(f"\n=== 运行 {dataset_name} 查询类型分析 ===")
    
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
        start_time = time.time()
        retriever.build_index(documents)
        end_time = time.time()
        print(f"{name} 索引构建完成，耗时: {end_time - start_time:.2f}秒")
        
        # 强制垃圾回收
        import gc
        gc.collect()
    
    # 分析查询类型
    print("分析查询类型...")
    adaptive_retriever = retrievers.get("AdaptiveHybrid")
    if not adaptive_retriever:
        print("错误: 未找到AdaptiveHybrid检索器，无法进行查询类型分析")
        return {}
    
    query_analyzer = adaptive_retriever.query_analyzer
    query_types = {}
    
    for query in queries:
        features = query_analyzer.analyze_query(query)
        query_types[query.query_id] = features.query_type.value
    
    # 统计查询类型分布
    type_distribution = defaultdict(int)
    for query_type in query_types.values():
        type_distribution[query_type] += 1
    
    print("查询类型分布:")
    for query_type, count in type_distribution.items():
        print(f"  {query_type}: {count} ({count/len(queries)*100:.1f}%)")
    
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
            "query_type_distribution": {k: v for k, v in type_distribution.items()},
            "results": {k: {r: v for r, v in v.items()} for k, v in results.items()}
        }, f, indent=2)
    
    print(f"查询类型分析报告已保存到: {report_file}")
    
    # 显示简要结果
    print("\n查询类型分析结果摘要:")
    for query_type, retrievers_results in results.items():
        print(f"\n查询类型: {query_type} (数量: {type_distribution[query_type]})")
        for retriever_name, metrics in retrievers_results.items():
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            mrr = metrics.get('mrr', 0.0)
            print(f"  {retriever_name}: P={precision:.3f}, R={recall:.3f}, MRR={mrr:.3f}")
    
    return results


def run_multiple_datasets(config_path: str = "configs/lightweight_config.json", 
                        datasets: List[str] = None,
                        top_k: int = 10,
                        sample_size: Optional[int] = None) -> None:
    """在多个数据集上运行查询类型分析"""
    # 加载配置
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        config = {
            "bm25": {"k1": 1.2, "b": 0.75},
            "efficient_vector": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "index_type": "hnsw",
                "batch_size": 8
            },
            "semantic_bm25": {
                "semantic_model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "semantic_weight": 0.3,
                "batch_size": 8
            },
            "query_analyzer": {
                "semantic_model_name": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "adaptive_router": {
                "routing_strategy": "hybrid"
            },
            "adaptive_fusion": {
                "default_method": "weighted_sum"
            },
            "evaluator": {
                "metrics": ["precision", "recall", "mrr", "ndcg", "latency"],
                "report_dir": "reports"
            }
        }
    
    # 如果没有指定数据集，使用配置中的数据集或默认数据集
    if datasets is None:
        datasets = config.get('datasets', ['nfcorpus'])
    
    # 运行查询类型分析
    for dataset in datasets:
        try:
            run_query_type_analysis(dataset, config, top_k, sample_size)
        except Exception as e:
            print(f"数据集 {dataset} 查询类型分析失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n所有查询类型分析完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行查询类型分析")
    parser.add_argument("--config", type=str, default="configs/lightweight_config.json", 
                       help="实验配置文件路径")
    parser.add_argument("--datasets", type=str, nargs="+", 
                       help="要评估的数据集")
    parser.add_argument("--top_k", type=int, default=10, 
                       help="检索的文档数量")
    parser.add_argument("--sample", type=int, 
                       help="查询样本大小，用于快速测试")
    
    args = parser.parse_args()
    
    print("🔍 开始运行查询类型分析")
    print("=" * 50)
    
    run_multiple_datasets(
        config_path=args.config,
        datasets=args.datasets, 
        top_k=args.top_k, 
        sample_size=args.sample
    )