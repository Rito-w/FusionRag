"""
优化版实验脚本
使用模型缓存，避免重复加载模型
支持批处理和内存优化
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import argparse
import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from modules.utils.model_cache import model_cache
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


def create_retrievers(config: Dict[str, Any] = None, lightweight: bool = False) -> Dict[str, Any]:
    """创建检索器（优化版本）"""
    config = config or {}
    
    retrievers = {}
    
    # 基础BM25检索器（总是包含，因为内存占用小）
    retrievers["BM25"] = BM25Retriever(config.get('bm25', {}))
    
    if not lightweight:
        # 高效向量检索器
        vector_config = config.get('efficient_vector', {})
        retrievers["EfficientVector"] = EfficientVectorIndex(vector_config)
        
        # 语义增强BM25（可选，因为内存占用较大）
        if config.get('include_semantic_bm25', True):
            semantic_config = config.get('semantic_bm25', {})
            retrievers["SemanticBM25"] = SemanticBM25(semantic_config)
    
    return retrievers


def run_experiment(dataset_name: str, config: Dict[str, Any] = None, 
                top_k: int = 10, sample_size: Optional[int] = None,
                lightweight: bool = False) -> Dict[str, Dict[str, float]]:
    """运行实验（优化版本）"""
    print(f"\n=== 运行 {dataset_name} 实验 ===")
    
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
    retrievers = create_retrievers(config, lightweight=lightweight)
    
    # 构建索引
    print("构建索引...")
    for name, retriever in retrievers.items():
        print(f"构建 {name} 索引...")
        start_time = time.time()
        retriever.build_index(documents)
        end_time = time.time()
        print(f"{name} 索引构建完成，耗时: {end_time - start_time:.2f}秒")
        
        # 强制垃圾回收
        gc.collect()
    
    # 创建评估器
    evaluator_config = config.get('evaluator', {})
    evaluator = IndexEvaluator(evaluator_config)
    
    # 评估性能
    print("评估性能...")
    results = evaluator.evaluate_multiple_retrievers(retrievers, queries, relevance_judgments, top_k)
    
    # 生成报告
    report_dir = f"reports/{dataset_name}/optimized"
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{report_dir}/optimized_evaluation_{timestamp}.json"
    
    evaluator.generate_report(results, f"{dataset_name}_optimized", report_file)
    
    # 显示简要结果
    print(f"\n{dataset_name} 实验结果摘要:")
    for retriever_name, metrics in results.items():
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        mrr = metrics.get('mrr', 0.0)
        latency = metrics.get('latency', 0.0)
        print(f"  {retriever_name}: P={precision:.3f}, R={recall:.3f}, MRR={mrr:.3f}, 延迟={latency:.1f}ms")
    
    return results


def run_multiple_datasets(config_path: str = "configs/optimized_e5_experiments.json", 
                        datasets: List[str] = None,
                        top_k: int = 10,
                        sample_size: Optional[int] = None,
                        lightweight: bool = False) -> None:
    """在多个数据集上运行实验（优化版本）"""
    # 加载配置
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        config = {
            "bm25": {"k1": 1.2, "b": 0.75},
            "efficient_vector": {
                "model_name": "intfloat/e5-large-v2",
                "index_type": "hnsw",
                "batch_size": 8
            },
            "semantic_bm25": {
                "semantic_model_name": "intfloat/e5-large-v2",
                "semantic_weight": 0.3,
                "batch_size": 8
            },
            "evaluator": {
                "metrics": ["precision", "recall", "mrr", "ndcg", "latency"],
                "report_dir": "reports"
            }
        }
    
    # 如果没有指定数据集，使用配置中的数据集或默认数据集
    if datasets is None:
        datasets = config.get('datasets', ['nfcorpus'])
    
    all_results = {}
    
    # 运行实验
    for dataset in datasets:
        try:
            results = run_experiment(dataset, config, top_k, sample_size, lightweight)
            all_results[dataset] = results
        except Exception as e:
            print(f"数据集 {dataset} 实验失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 每个数据集完成后清理缓存
        model_cache.clear_cache("embedding")
        gc.collect()
    
    # 保存总体结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"reports/optimized_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n所有实验完成! 结果已保存到: {output_file}")


def monitor_memory():
    """监控内存使用情况"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        return memory_mb
    except ImportError:
        return -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行优化版实验")
    parser.add_argument("--config", type=str, default="configs/optimized_e5_experiments.json", 
                       help="实验配置文件路径")
    parser.add_argument("--datasets", type=str, nargs="+", 
                       help="要评估的数据集")
    parser.add_argument("--top_k", type=int, default=10, 
                       help="检索的文档数量")
    parser.add_argument("--sample", type=int, default=5,
                       help="查询样本大小，用于快速测试")
    parser.add_argument("--lightweight", action="store_true", 
                       help="使用轻量级模式（仅BM25）")
    
    args = parser.parse_args()
    
    # 显示初始内存使用
    initial_memory = monitor_memory()
    if initial_memory > 0:
        print(f"初始内存使用: {initial_memory:.1f} MB")
    
    print("🚀 开始运行优化版实验")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"数据集: {args.datasets or '使用配置文件中的数据集'}")
    print(f"样本大小: {args.sample or '全部查询'}")
    print(f"轻量级模式: {'是' if args.lightweight else '否'}")
    print("=" * 60)
    
    # 运行实验
    start_time = time.time()
    run_multiple_datasets(
        config_path=args.config,
        datasets=args.datasets, 
        top_k=args.top_k, 
        sample_size=args.sample,
        lightweight=args.lightweight
    )
    end_time = time.time()
    
    # 显示最终内存使用和总耗时
    final_memory = monitor_memory()
    if final_memory > 0:
        print(f"最终内存使用: {final_memory:.1f} MB")
    print(f"总耗时: {end_time - start_time:.2f}秒")