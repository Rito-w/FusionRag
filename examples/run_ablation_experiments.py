"""
消融实验脚本
专门用于评估各个组件对系统性能的贡献
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


def run_ablation_study(dataset_name: str, config: Dict[str, Any] = None, 
                     top_k: int = 10, sample_size: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """运行消融实验"""
    print(f"\n=== 运行 {dataset_name} 消融实验 ===")
    
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
    base_config = config.copy() if config else {}
    base_retriever = create_adaptive_hybrid_index(base_config)
    
    # 创建消融实验检索器
    component_retrievers = {
        "NoQueryAnalysis": create_adaptive_hybrid_index({
            **base_config,
            'query_analyzer': {'disabled': True}
        }),
        "NoAdaptiveRouting": create_adaptive_hybrid_index({
            **base_config,
            'adaptive_router': {'disabled': True}
        }),
        "NoAdaptiveFusion": create_adaptive_hybrid_index({
            **base_config,
            'adaptive_fusion': {'disabled': True}
        }),
        "OnlyBM25": BM25Retriever(base_config.get('bm25', {})),
        "OnlyVector": EfficientVectorIndex(base_config.get('efficient_vector', {}))
    }
    
    # 构建索引
    print("构建基础索引...")
    start_time = time.time()
    base_retriever.build_index(documents)
    end_time = time.time()
    print(f"基础索引构建完成，耗时: {end_time - start_time:.2f}秒")
    
    # 强制垃圾回收
    import gc
    gc.collect()
    
    print("构建消融实验索引...")
    for name, retriever in component_retrievers.items():
        print(f"构建 {name} 索引...")
        start_time = time.time()
        retriever.build_index(documents)
        end_time = time.time()
        print(f"{name} 索引构建完成，耗时: {end_time - start_time:.2f}秒")
        
        # 每个索引构建后进行垃圾回收
        gc.collect()
    
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
    
    # 显示简要结果
    print(f"\n{dataset_name} 消融实验结果摘要:")
    for retriever_name, metrics in results.items():
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        mrr = metrics.get('mrr', 0.0)
        latency = metrics.get('latency', 0.0)
        print(f"  {retriever_name}: P={precision:.3f}, R={recall:.3f}, MRR={mrr:.3f}, 延迟={latency:.1f}ms")
    
    return results


def run_multiple_datasets(config_path: str = "configs/lightweight_config.json", 
                        datasets: List[str] = None,
                        top_k: int = 10,
                        sample_size: Optional[int] = None) -> None:
    """在多个数据集上运行消融实验"""
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
    
    all_results = {}
    
    # 运行消融实验
    for dataset in datasets:
        try:
            results = run_ablation_study(dataset, config, top_k, sample_size)
            all_results[dataset] = results
        except Exception as e:
            print(f"数据集 {dataset} 消融实验失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存总体结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"reports/ablation_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n所有消融实验完成! 结果已保存到: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行消融实验")
    parser.add_argument("--config", type=str, default="configs/lightweight_config.json", 
                       help="实验配置文件路径")
    parser.add_argument("--datasets", type=str, nargs="+", 
                       help="要评估的数据集")
    parser.add_argument("--top_k", type=int, default=10, 
                       help="检索的文档数量")
    parser.add_argument("--sample", type=int, 
                       help="查询样本大小，用于快速测试")
    
    args = parser.parse_args()
    
    print("🔬 开始运行消融实验")
    print("=" * 50)
    
    run_multiple_datasets(
        config_path=args.config,
        datasets=args.datasets, 
        top_k=args.top_k, 
        sample_size=args.sample
    )