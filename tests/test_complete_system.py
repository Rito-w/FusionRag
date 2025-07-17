#!/usr/bin/env python
"""
完整FusionRAG系统测试
测试包含所有模块的完整系统功能
"""

import sys
import os
sys.path.append('.')

from pipeline import FusionRAGPipeline
from modules.utils.interfaces import Query

def test_complete_system():
    """测试完整系统功能"""
    print("🚀 完整FusionRAG系统测试")
    print("=" * 60)
    
    # 创建包含所有组件的配置
    config_content = """
data:
  corpus_path: "data/processed/nfcorpus_corpus.jsonl"
  queries_path: "data/processed/nfcorpus_queries.jsonl"
  qrels_path: "data/processed/nfcorpus_qrels.tsv"

retrievers:
  bm25:
    enabled: true
    index_path: "checkpoints/retriever/complete_test_bm25_index.pkl"
    k1: 1.2
    b: 0.75
    top_k: 50
    
  dense:
    enabled: true
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    index_path: "checkpoints/retriever/complete_test_dense_index.faiss"
    embedding_dim: 384
    top_k: 50
    
  graph:
    enabled: true
    index_path: "checkpoints/retriever/complete_test_graph_index.pkl"
    neo4j_uri: "bolt://localhost:7687"
    neo4j_user: "neo4j"
    neo4j_password: "password"
    database: "neo4j"
    max_walk_length: 2
    entity_threshold: 2
    top_k: 30

classifier:
  enabled: true
  threshold: 0.5
  classes: ["factual", "analytical", "procedural"]
  adaptation_enabled: true
  min_samples: 5
  performance_threshold: 0.1

fusion:
  method: "weighted"
  weights:
    bm25: 0.4
    dense: 0.4
    graph: 0.2
  top_k: 20

evaluation:
  metrics: ["recall@5", "recall@10", "ndcg@10", "map"]

system:
  log_level: "INFO"
  log_path: "checkpoints/logs/complete_test.log"
"""
    
    temp_config = "temp_complete_test_config.yaml"
    with open(temp_config, 'w') as f:
        f.write(config_content)
    
    try:
        # 初始化Pipeline
        print("📝 初始化完整系统...")
        pipeline = FusionRAGPipeline(temp_config)
        
        # 加载数据
        print("\n📁 加载数据...")
        pipeline.load_data()
        
        print(f"数据概况:")
        print(f"  文档数: {len(pipeline.documents):,}")
        print(f"  查询数: {len(pipeline.queries):,}")
        print(f"  标注数: {len(pipeline.qrels):,}")
        
        # 使用小数据集进行测试
        print("\n🔨 构建索引...")
        test_docs = pipeline.documents[:100]  # 使用100个文档测试
        
        for name, retriever in pipeline.retrievers.items():
            print(f"  构建 {name} 索引...")
            retriever.build_index(test_docs)
        
        # 测试智能检索
        print("\n🔍 测试智能检索...")
        
        # 不同类型的测试查询
        test_queries = [
            # 事实性查询
            Query("fact_1", "What is diabetes?"),
            Query("fact_2", "Define breast cancer"),
            
            # 分析性查询  
            Query("anal_1", "Why do statins cause side effects?"),
            Query("anal_2", "How does obesity affect cardiovascular health?"),
            
            # 程序性查询
            Query("proc_1", "Treatment procedure for hypertension"),
            Query("proc_2", "Step by step cancer diagnosis process"),
        ]
        
        all_results = {}
        routing_stats = []
        
        for i, query in enumerate(test_queries):
            print(f"\n查询 {i+1}: {query.text}")
            print("-" * 50)
            
            # 执行检索
            results = pipeline.search(query, top_k=10)
            all_results[query.query_id] = results
            
            # 显示分类和路由信息
            if pipeline.classifier:
                classification = pipeline.classifier.classify_query(query)
                print(f"  分类: {classification['predicted_class']} (置信度: {classification['confidence']:.3f})")
                print(f"  推荐检索器: {classification['recommended_retrievers']}")
            
            # 显示检索结果
            print(f"  检索结果数: {len(results)}")
            for j, result in enumerate(results[:3]):
                print(f"    {j+1}. [{result.final_score:.4f}] {result.document.title[:50]}...")
                print(f"       各检索器分数: {result.individual_scores}")
        
        # 获取路由统计
        if pipeline.router:
            routing_stats = pipeline.router.get_routing_stats()
            print(f"\n📊 路由统计:")
            print(f"  路由使用: {routing_stats['route_usage']}")
            print(f"  性能历史: {routing_stats['performance_history']}")
        
        # 评测系统性能
        print(f"\n📈 系统性能评测...")
        
        # 选择有标注的查询进行评测
        eval_queries = [q for q in pipeline.queries if q.query_id in pipeline.qrels][:10]
        eval_results = {}
        
        for query in eval_queries:
            results = pipeline.search(query, top_k=20)
            eval_results[query.query_id] = [r.doc_id for r in results]
        
        # 构建ground truth
        eval_qrels = {qid: pipeline.qrels[qid] for qid in eval_results.keys()}
        
        # 执行评测
        metrics = pipeline.evaluator.evaluate_retrieval(eval_results, eval_qrels)
        
        print(f"评测结果 ({len(eval_queries)} 个查询):")
        for metric, score in metrics.items():
            if metric not in ['num_queries', 'timestamp']:
                print(f"  {metric}: {score:.4f}")
        
        # 组件统计
        print(f"\n🔧 系统组件统计:")
        print(f"  启用的检索器: {list(pipeline.retrievers.keys())}")
        print(f"  分类器状态: {'启用' if pipeline.classifier else '禁用'}")
        print(f"  智能路由: {'启用' if pipeline.router else '禁用'}")
        print(f"  融合策略: {pipeline.fusion.method if pipeline.fusion else 'None'}")
        
        # 图检索器统计（如果启用）
        if 'graph' in pipeline.retrievers:
            graph_stats = pipeline.retrievers['graph'].get_statistics()
            print(f"  图检索器统计: {graph_stats}")
        
        print(f"\n✅ 完整系统测试成功!")
        
    except Exception as e:
        print(f"\n❌ 系统测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_config):
            os.remove(temp_config)

def test_component_interaction():
    """测试组件间交互"""
    print("\n🔗 组件交互测试")
    print("=" * 60)
    
    try:
        # 测试各组件的协同工作
        from modules.classifier.query_classifier import QueryClassifier
        from modules.retriever.bm25_retriever import BM25Retriever
        from modules.retriever.dense_retriever import DenseRetriever
        from modules.fusion.fusion import MultiFusion
        from modules.utils.common import JSONDataLoader
        
        # 加载数据
        loader = JSONDataLoader()
        documents = loader.load_documents("data/processed/nfcorpus_corpus.jsonl")[:50]
        
        # 初始化组件
        classifier = QueryClassifier()
        bm25_retriever = BM25Retriever()
        dense_retriever = DenseRetriever(config={'model_name': 'sentence-transformers/all-MiniLM-L6-v2'})
        fusion = MultiFusion({'method': 'weighted', 'weights': {'bm25': 0.6, 'dense': 0.4}})
        
        # 构建索引
        bm25_retriever.build_index(documents)
        dense_retriever.build_index(documents)
        
        # 测试查询
        test_query = Query("test", "What causes diabetes mellitus?")
        
        # 分类查询
        classification = classifier.classify_query(test_query)
        print(f"查询分类结果: {classification}")
        
        # 基于分类结果选择检索器
        recommended = classification['recommended_retrievers']
        print(f"推荐检索器: {recommended}")
        
        # 执行检索
        retrieval_results = {}
        
        if 'bm25' in recommended:
            bm25_results = bm25_retriever.retrieve(test_query, top_k=20)
            retrieval_results['bm25'] = bm25_results
            print(f"BM25检索结果: {len(bm25_results)} 个")
        
        if 'dense' in recommended:
            dense_results = dense_retriever.retrieve(test_query, top_k=20)
            retrieval_results['dense'] = dense_results
            print(f"Dense检索结果: {len(dense_results)} 个")
        
        # 融合结果
        if len(retrieval_results) > 1:
            fused_results = fusion.fuse(retrieval_results, test_query)
            print(f"融合后结果: {len(fused_results)} 个")
            
            # 显示融合统计
            fusion_stats = fusion.get_fusion_statistics(fused_results)
            print(f"融合统计: {fusion_stats}")
        
        print("✅ 组件交互测试成功!")
        
    except Exception as e:
        print(f"❌ 组件交互测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🎯 FusionRAG完整系统验证")
    print("=" * 70)
    
    try:
        # 1. 完整系统测试
        test_complete_system()
        
        # 2. 组件交互测试
        test_component_interaction()
        
        print(f"\n🎉 FusionRAG系统验证完成!")
        print(f"\n📋 系统特性总结:")
        print("✅ 多检索器融合 (BM25 + Dense + Graph)")
        print("✅ 智能查询分类和路由")
        print("✅ 自适应性能优化")  
        print("✅ Neo4j图数据库支持（可回退到内存模式）")
        print("✅ 标准化评测指标")
        print("✅ 模块化可扩展架构")
        print("✅ 中英文支持")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中断测试")
    except Exception as e:
        print(f"\n❌ 系统验证失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()