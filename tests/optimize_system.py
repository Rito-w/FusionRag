#!/usr/bin/env python
"""
优化的FusionRAG系统
基于诊断结果进行的性能优化
"""

import sys
import os
import time
sys.path.append('.')

def create_optimized_config():
    """创建优化的配置"""
    optimized_config = """
# 优化的FusionRAG配置

data:
  corpus_path: "data/processed/nfcorpus_corpus.jsonl"
  queries_path: "data/processed/nfcorpus_queries.jsonl"
  qrels_path: "data/processed/nfcorpus_qrels.tsv"
  output_dir: "data/processed/"

retrievers:
  bm25:
    enabled: true
    index_path: "checkpoints/retriever/optimized_bm25_index.pkl"
    k1: 1.2
    b: 0.75
    top_k: 200  # 增加检索数量
    
  dense:
    enabled: true
    model_name: "sentence-transformers/all-mpnet-base-v2"  # 更强的模型
    index_path: "checkpoints/retriever/optimized_dense_index.faiss"
    embedding_dim: 768  # 更高维度
    top_k: 200  # 增加检索数量
    batch_size: 16  # 减小batch size提高质量

fusion:
  method: "rrf"  # 使用RRF融合，对排名更敏感
  rrf_k: 60
  top_k: 50  # 增加最终结果数量

reranker:
  enabled: false

evaluation:
  metrics: ["recall@5", "recall@10", "recall@20", "ndcg@10", "ndcg@20", "map"]
  output_path: "checkpoints/logs/optimized_eval_results.json"

system:
  device: "cpu"
  batch_size: 16
  num_threads: 4
  log_level: "INFO"
  log_path: "checkpoints/logs/optimized_system.log"
"""
    
    with open("configs/optimized_config.yaml", 'w', encoding='utf-8') as f:
        f.write(optimized_config)
    
    print("✅ 优化配置已创建: configs/optimized_config.yaml")

def test_optimized_system():
    """测试优化后的系统"""
    print("🚀 测试优化后的FusionRAG系统")
    print("=" * 50)
    
    from pipeline import FusionRAGPipeline
    
    # 使用优化配置
    pipeline = FusionRAGPipeline("configs/optimized_config.yaml")
    
    # 加载数据
    print("加载数据...")
    pipeline.load_data()
    
    print(f"数据统计:")
    print(f"  文档数: {len(pipeline.documents):,}")
    print(f"  查询数: {len(pipeline.queries):,}")
    print(f"  标注数: {len(pipeline.qrels):,}")
    
    # 构建优化索引
    print("\n构建优化索引...")
    start_time = time.time()
    pipeline.build_indexes(force_rebuild=True)
    index_time = time.time() - start_time
    print(f"索引构建时间: {index_time:.2f}s")
    
    # 测试性能
    print("\n测试检索性能...")
    
    # 选择有标注的查询进行测试
    test_queries = [q for q in pipeline.queries if q.query_id in pipeline.qrels][:20]
    
    all_results = {}
    retrieval_times = []
    
    for i, query in enumerate(test_queries):
        start = time.time()
        results = pipeline.search(query, top_k=50)  # 获取更多结果
        retrieval_time = time.time() - start
        retrieval_times.append(retrieval_time)
        
        all_results[query.query_id] = results
        
        if i < 5:  # 显示前5个查询的详细结果
            relevant_docs = set(pipeline.qrels[query.query_id])
            result_docs = [r.doc_id for r in results]
            
            recall_5 = len(set(result_docs[:5]) & relevant_docs) / len(relevant_docs)
            recall_10 = len(set(result_docs[:10]) & relevant_docs) / len(relevant_docs)
            recall_20 = len(set(result_docs[:20]) & relevant_docs) / len(relevant_docs)
            
            print(f"\n  查询 {i+1}: {query.text[:50]}...")
            print(f"    相关文档数: {len(relevant_docs)}")
            print(f"    Recall@5: {recall_5:.3f}")
            print(f"    Recall@10: {recall_10:.3f}")
            print(f"    Recall@20: {recall_20:.3f}")
            print(f"    检索时间: {retrieval_time:.3f}s")
            
            # 显示前几个结果
            print(f"    前5个结果:")
            for j, result in enumerate(results[:5]):
                is_relevant = "✓" if result.doc_id in relevant_docs else "✗"
                print(f"      {j+1}. [{result.final_score:.3f}] {is_relevant} {result.document.title[:40]}...")
    
    # 全面评测
    print(f"\n全面评测 ({len(test_queries)} 个查询)...")
    
    query_predictions = {}
    query_ground_truth = {}
    
    for query_id, results in all_results.items():
        query_predictions[query_id] = [r.doc_id for r in results]
        query_ground_truth[query_id] = pipeline.qrels[query_id]
    
    metrics = pipeline.evaluator.evaluate_retrieval(query_predictions, query_ground_truth)
    
    print(f"\n📊 优化后性能:")
    print(f"  平均检索时间: {sum(retrieval_times)/len(retrieval_times):.3f}s")
    print(f"  评测结果:")
    for metric, score in metrics.items():
        if metric not in ['num_queries', 'timestamp']:
            print(f"    {metric}: {score:.4f}")
    
    return metrics

def compare_with_baseline():
    """与基线系统对比"""
    print(f"\n📈 与基线系统对比")
    print("=" * 50)
    
    # 基线结果 (从之前的测试)
    baseline_metrics = {
        'recall@5': 0.0189,
        'recall@10': 0.0189, 
        'ndcg@10': 0.1158,
        'map': 0.0189
    }
    
    # 优化系统结果
    optimized_metrics = test_optimized_system()
    
    print(f"\n🏆 性能对比:")
    print(f"{'指标':<12} {'基线':<10} {'优化后':<10} {'提升':<10}")
    print("-" * 45)
    
    for metric in ['recall@5', 'recall@10', 'ndcg@10', 'map']:
        if metric in optimized_metrics:
            baseline = baseline_metrics.get(metric, 0)
            optimized = optimized_metrics.get(metric, 0)
            improvement = ((optimized - baseline) / baseline * 100) if baseline > 0 else 0
            
            print(f"{metric:<12} {baseline:<10.4f} {optimized:<10.4f} {improvement:>+7.1f}%")

def test_query_expansion():
    """测试查询扩展技术"""
    print(f"\n🔍 查询扩展优化")
    print("=" * 50)
    
    def expand_query(query_text):
        """简单的查询扩展"""
        # 添加同义词和相关词
        expansions = {
            'cancer': ['cancer', 'tumor', 'carcinoma', 'malignancy'],
            'breast': ['breast', 'mammary'],
            'statin': ['statin', 'cholesterol drug', 'HMG-CoA reductase inhibitor'],
            'treatment': ['treatment', 'therapy', 'medication'],
            'diet': ['diet', 'nutrition', 'food'],
            'obesity': ['obesity', 'overweight', 'BMI'],
        }
        
        expanded_terms = []
        words = query_text.lower().split()
        
        for word in words:
            expanded_terms.append(word)
            for key, synonyms in expansions.items():
                if key in word:
                    expanded_terms.extend([s for s in synonyms if s != word])
        
        return ' '.join(expanded_terms)
    
    from pipeline import FusionRAGPipeline
    from modules.utils.interfaces import Query
    
    pipeline = FusionRAGPipeline("configs/optimized_config.yaml")
    pipeline.load_data()
    
    # 使用已构建的索引
    test_query_text = "breast cancer statin treatment"
    original_query = Query("test", test_query_text)
    expanded_query = Query("test_expanded", expand_query(test_query_text))
    
    print(f"原始查询: {original_query.text}")
    print(f"扩展查询: {expanded_query.text}")
    
    # 对比结果
    original_results = pipeline.search(original_query, top_k=20)
    expanded_results = pipeline.search(expanded_query, top_k=20)
    
    print(f"\n结果对比:")
    print(f"原始查询找到 {len(original_results)} 个结果")
    print(f"扩展查询找到 {len(expanded_results)} 个结果")
    
    print(f"\n原始查询前3个结果:")
    for i, result in enumerate(original_results[:3]):
        print(f"  {i+1}. [{result.final_score:.3f}] {result.document.title[:50]}...")
    
    print(f"\n扩展查询前3个结果:")
    for i, result in enumerate(expanded_results[:3]):
        print(f"  {i+1}. [{result.final_score:.3f}] {result.document.title[:50]}...")

def main():
    """主函数"""
    print("⚡ FusionRAG系统性能优化")
    print("=" * 60)
    
    try:
        # 1. 创建优化配置
        create_optimized_config()
        
        # 2. 对比测试
        compare_with_baseline()
        
        # 3. 测试查询扩展
        test_query_expansion()
        
        print(f"\n✅ 优化完成!")
        print(f"主要改进:")
        print(f"  1. 使用更强的向量模型 (all-mpnet-base-v2)")
        print(f"  2. 增加检索数量 (top_k=200)")
        print(f"  3. 使用RRF融合策略")
        print(f"  4. 扩展评测指标")
        print(f"  5. 查询扩展技术")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中断")
    except Exception as e:
        print(f"\n❌ 优化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()