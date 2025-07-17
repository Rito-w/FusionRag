#!/usr/bin/env python
"""
FusionRAG 快速性能测试
使用较小数据规模进行快速验证
"""

import sys
import time
import json

sys.path.append('.')

from modules.utils.interfaces import Document, Query
from modules.retriever.bm25_retriever import BM25Retriever
from modules.retriever.dense_retriever import DenseRetriever
from modules.retriever.graph_retriever import GraphRetriever
from modules.fusion.fusion import MultiFusion
from modules.evaluator.evaluator import IRMetricsEvaluator

def load_sample_data(doc_limit: int = 100, query_limit: int = 10):
    """加载小规模测试数据"""
    print(f"📁 加载测试数据 (文档:{doc_limit}, 查询:{query_limit})")
    
    # 加载文档
    documents = []
    with open('data/processed/nfcorpus_corpus.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= doc_limit:
                break
            data = json.loads(line)
            doc = Document(
                doc_id=data['doc_id'],
                title=data.get('title', ''),
                text=data['text'],
                metadata=data.get('metadata', {})
            )
            documents.append(doc)
    
    # 加载查询
    queries = []
    with open('data/processed/nfcorpus_queries.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= query_limit:
                break
            data = json.loads(line)
            query = Query(
                query_id=data['query_id'],
                text=data['text'],
                metadata=data.get('metadata', {})
            )
            queries.append(query)
    
    print(f"✅ 数据加载完成: {len(documents)}文档, {len(queries)}查询")
    return documents, queries

def quick_performance_test():
    """快速性能测试"""
    print("⚡ FusionRAG 快速性能测试")
    print("=" * 50)
    
    # 加载小规模数据
    documents, queries = load_sample_data(doc_limit=100, query_limit=10)
    
    # 优化配置
    configs = {
        'bm25': {'k1': 1.5, 'b': 0.75, 'top_k': 50},
        'dense': {'model_name': 'sentence-transformers/all-MiniLM-L6-v2', 'top_k': 50},
        'graph': {'entity_threshold': 2, 'max_walk_length': 2, 'top_k': 30}
    }
    
    # 初始化检索器
    print("\n🔧 初始化检索器...")
    retrievers = {}
    
    # BM25检索器
    retrievers['bm25'] = BM25Retriever(name="bm25", config=configs['bm25'])
    print("✅ BM25检索器初始化完成")
    
    # Dense检索器
    retrievers['dense'] = DenseRetriever(name="dense", config=configs['dense'])
    print("✅ Dense检索器初始化完成")
    
    # Graph检索器
    retrievers['graph'] = GraphRetriever(name="graph", config=configs['graph'])
    print("✅ Graph检索器初始化完成")
    
    # 构建索引
    print("\n🔨 构建索引...")
    build_start = time.time()
    
    for name, retriever in retrievers.items():
        index_start = time.time()
        if name == 'graph':
            retriever.build_index(documents, dataset_name="quick_test")
        else:
            retriever.build_index(documents)
        index_time = time.time() - index_start
        print(f"  ✅ {name}: {index_time:.2f}s")
    
    total_build_time = time.time() - build_start
    print(f"📊 总构建时间: {total_build_time:.2f}s")
    
    # 融合器
    fusion_config = {
        'method': 'weighted',
        'weights': {'bm25': 0.5, 'dense': 0.35, 'graph': 0.15},
        'top_k': 20
    }
    fusion = MultiFusion(config=fusion_config)
    
    # 执行检索测试
    print("\n🎯 执行检索测试...")
    
    all_results = []
    retrieval_times = []
    retriever_stats = {'bm25': 0, 'dense': 0, 'graph': 0}
    
    for i, query in enumerate(queries):
        print(f"\n🔍 查询 {i+1}: {query.text[:40]}...")
        
        query_start = time.time()
        
        # 各检索器检索
        retriever_results = {}
        for name, retriever in retrievers.items():
            try:
                results = retriever.retrieve(query, top_k=configs[name]['top_k'])
                retriever_results[name] = results
                retriever_stats[name] += len(results)
                print(f"  {name}: {len(results)} 结果")
            except Exception as e:
                print(f"  ⚠️ {name} 检索失败: {e}")
                retriever_results[name] = []
        
        # 融合结果
        if retriever_results:
            fused_results = fusion.fuse(retriever_results, query)
            all_results.append((query, fused_results))
            print(f"  融合: {len(fused_results)} 结果")
        
        query_time = time.time() - query_start
        retrieval_times.append(query_time)
        print(f"  ⏱️ 时间: {query_time:.3f}s")
    
    # 性能统计
    print("\n📊 性能统计")
    print("=" * 40)
    
    avg_query_time = sum(retrieval_times) / len(retrieval_times)
    print(f"平均查询时间: {avg_query_time:.3f}s")
    print(f"查询吞吐量: {1/avg_query_time:.1f} queries/sec")
    
    print(f"\n检索器结果统计:")
    for name, count in retriever_stats.items():
        avg_per_query = count / len(queries)
        print(f"  {name}: 平均 {avg_per_query:.1f} 结果/查询")
    
    # 融合结果分析
    print(f"\n融合结果分析:")
    fusion_contributions = {'bm25': 0, 'dense': 0, 'graph': 0}
    total_fused = 0
    
    for _, fused_results in all_results:
        total_fused += len(fused_results)
        for result in fused_results[:10]:  # 看前10个结果
            # FusionResult包含individual_scores，显示各检索器的贡献
            if hasattr(result, 'individual_scores') and result.individual_scores:
                # 找到贡献最大的检索器
                max_score_retriever = max(result.individual_scores.items(), key=lambda x: x[1])[0]
                if max_score_retriever in fusion_contributions:
                    fusion_contributions[max_score_retriever] += 1
    
    print(f"前10融合结果中主要贡献检索器:")
    total_contributions = sum(fusion_contributions.values())
    for name, count in fusion_contributions.items():
        percentage = count / total_contributions * 100 if total_contributions > 0 else 0
        print(f"  {name}: {count} ({percentage:.1f}%)")
    
    # 图检索器质量分析
    print(f"\n🔗 图检索器分析:")
    graph_retriever = retrievers['graph']
    stats = graph_retriever.get_statistics()
    
    if stats:
        print(f"  节点数: {stats.get('nodes', 'N/A')}")
        print(f"  边数: {stats.get('edges', 'N/A')}")
        print(f"  平均度: {stats.get('avg_degree', 'N/A'):.1f}")
        print(f"  数据集: {stats.get('dataset', 'N/A')}")
    
    # 示例结果展示
    print(f"\n🎯 示例检索结果:")
    if all_results:
        query, results = all_results[0]
        print(f"查询: {query.text}")
        print(f"前5个结果:")
        for i, result in enumerate(results[:5], 1):
            print(f"  {i}. [{result.final_score:.3f}] {result.document.title[:50]}...")
            # 显示各检索器的贡献分数
            if hasattr(result, 'individual_scores') and result.individual_scores:
                scores_str = ", ".join([f"{k}:{v:.3f}" for k, v in result.individual_scores.items()])
                print(f"     分数分布: {scores_str}")
    
    print(f"\n🎉 快速测试完成!")
    print(f"测试规模: {len(documents)}文档, {len(queries)}查询")
    print(f"总耗时: {time.time() - build_start + total_build_time:.2f}s")

if __name__ == "__main__":
    quick_performance_test()