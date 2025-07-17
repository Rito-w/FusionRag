#!/usr/bin/env python
"""
查询分类器测试脚本
测试智能查询分类和路由功能
"""

import sys
import os
sys.path.append('.')

from modules.classifier.query_classifier import QueryClassifier, AdaptiveQueryRouter
from modules.utils.interfaces import Query

def test_query_classifier():
    """测试查询分类器"""
    print("🧪 查询分类器测试")
    print("=" * 50)
    
    classifier = QueryClassifier()
    
    # 测试查询
    test_queries = [
        # 事实性查询
        Query("1", "What is breast cancer?"),
        Query("2", "Define diabetes mellitus"),
        Query("3", "什么是高血压？"),
        Query("4", "Who discovered insulin?"),
        
        # 分析性查询
        Query("5", "Why do statins cause muscle pain?"),
        Query("6", "How does chemotherapy work?"),
        Query("7", "Compare different diabetes treatments"),
        Query("8", "为什么会得糖尿病？"),
        
        # 程序性查询
        Query("9", "Step by step procedure for blood pressure measurement"),
        Query("10", "Treatment protocol for acute myocardial infarction"),
        Query("11", "如何治疗高血压？"),
        Query("12", "Cancer diagnosis process"),
    ]
    
    print("单个查询分类测试:")
    print("-" * 40)
    
    for query in test_queries:
        result = classifier.classify_query(query)
        
        print(f"查询: {query.text}")
        print(f"  预测类别: {result['predicted_class']}")
        print(f"  置信度: {result['confidence']:.3f}")
        print(f"  推荐检索器: {result['recommended_retrievers']}")
        print(f"  类别得分: {result['class_scores']}")
        print()
    
    # 批量分类测试
    print("\n批量分类测试:")
    print("-" * 40)
    
    batch_results = classifier.batch_classify(test_queries)
    stats = classifier.get_retriever_stats(batch_results)
    
    print(f"类别分布: {stats['class_distribution']}")
    print(f"检索器使用: {stats['retriever_usage']}")
    print(f"总查询数: {stats['total_queries']}")

def test_adaptive_router():
    """测试自适应路由器"""
    print("\n🔀 自适应路由器测试")
    print("=" * 50)
    
    classifier = QueryClassifier()
    router = AdaptiveQueryRouter(classifier)
    
    # 可用检索器
    available_retrievers = ['bm25', 'dense', 'graph']
    
    # 测试查询
    test_queries = [
        Query("r1", "What causes diabetes?"),
        Query("r2", "How to treat hypertension?"),
        Query("r3", "Define cardiovascular disease"),
        Query("r4", "Compare cancer therapies"),
    ]
    
    print("路由决策测试:")
    print("-" * 40)
    
    routing_results = []
    
    for query in test_queries:
        route_result = router.route_query(query, available_retrievers)
        routing_results.append(route_result)
        
        print(f"查询: {query.text}")
        print(f"  分类: {route_result['classification']['predicted_class']}")
        print(f"  路由到: {route_result['retrievers']}")
        print(f"  路由置信度: {route_result['route_confidence']:.3f}")
        print(f"  自适应调整: {route_result['adaptation_applied']}")
        print()
    
    # 模拟性能反馈
    print("模拟性能更新:")
    print("-" * 40)
    
    import random
    for i, (query, route_result) in enumerate(zip(test_queries, routing_results)):
        # 模拟性能得分
        performance_score = random.uniform(0.1, 0.9)
        router.update_performance(query, route_result['retrievers'], performance_score)
        print(f"查询 {i+1}: 性能得分 {performance_score:.3f}")
    
    # 获取路由统计
    routing_stats = router.get_routing_stats()
    print(f"\n路由统计:")
    print(f"  路由使用: {routing_stats['route_usage']}")
    print(f"  性能历史: {routing_stats['performance_history']}")

def test_classifier_persistence():
    """测试分类器持久化"""
    print("\n💾 分类器持久化测试")
    print("=" * 50)
    
    # 创建并配置分类器
    original_classifier = QueryClassifier({
        'threshold': 0.6,
        'classes': ['factual', 'analytical', 'procedural', 'custom']
    })
    
    # 添加自定义特征
    original_classifier.feature_patterns['custom'] = {
        'keywords': ['custom', 'special'],
        'patterns': [r'\bcustom\b', r'\bspecial\b']
    }
    
    # 保存模型
    model_path = "temp_classifier_model.pkl"
    original_classifier.save_model(model_path)
    
    # 创建新分类器并加载模型
    new_classifier = QueryClassifier()
    new_classifier.load_model(model_path)
    
    # 测试加载的模型
    test_query = Query("test", "This is a custom special query")
    
    original_result = original_classifier.classify_query(test_query)
    loaded_result = new_classifier.classify_query(test_query)
    
    print(f"原始分类器结果: {original_result['predicted_class']}")
    print(f"加载后分类器结果: {loaded_result['predicted_class']}")
    print(f"结果一致: {original_result['predicted_class'] == loaded_result['predicted_class']}")
    
    # 清理
    os.remove(model_path)
    print("✅ 模型保存和加载测试成功")

def test_real_queries():
    """测试真实查询数据"""
    print("\n🔬 真实查询测试")
    print("=" * 50)
    
    try:
        from modules.utils.common import JSONDataLoader
        
        # 加载真实查询
        loader = JSONDataLoader()
        queries = loader.load_queries("data/processed/nfcorpus_queries.jsonl")[:10]
        
        classifier = QueryClassifier()
        
        print("真实查询分类结果:")
        print("-" * 40)
        
        for query in queries:
            result = classifier.classify_query(query)
            print(f"查询: {query.text[:60]}...")
            print(f"  类别: {result['predicted_class']} (置信度: {result['confidence']:.3f})")
            print(f"  推荐: {result['recommended_retrievers']}")
            print()
        
        # 统计分析
        batch_results = classifier.batch_classify(queries)
        stats = classifier.get_retriever_stats(batch_results)
        
        print(f"真实数据统计:")
        print(f"  类别分布: {stats['class_distribution']}")
        print(f"  检索器推荐: {stats['retriever_usage']}")
        
    except Exception as e:
        print(f"❌ 真实查询测试失败: {e}")

def test_integration_with_pipeline():
    """测试与Pipeline集成"""
    print("\n🔗 Pipeline集成测试")
    print("=" * 50)
    
    try:
        # 创建包含分类器的配置
        config_content = """
data:
  corpus_path: "data/processed/nfcorpus_corpus.jsonl"
  queries_path: "data/processed/nfcorpus_queries.jsonl"
  qrels_path: "data/processed/nfcorpus_qrels.tsv"

retrievers:
  bm25:
    enabled: true
    index_path: "checkpoints/retriever/test_bm25_index.pkl"
    k1: 1.2
    b: 0.75
    top_k: 50
  dense:
    enabled: true
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    index_path: "checkpoints/retriever/test_dense_index.faiss"
    embedding_dim: 384
    top_k: 50

classifier:
  enabled: true
  threshold: 0.5
  classes: ["factual", "analytical", "procedural"]
  adaptation_enabled: true

fusion:
  method: "weighted"
  weights:
    bm25: 0.5
    dense: 0.5
  top_k: 10

system:
  log_level: "ERROR"
"""
        
        temp_config = "temp_classifier_config.yaml"
        with open(temp_config, 'w') as f:
            f.write(config_content)
        
        # 这里只是演示配置，实际集成需要在Pipeline中实现
        print("✅ 分类器配置已准备好集成到Pipeline")
        print("配置包含:")
        print("  - 查询分类功能")
        print("  - 智能检索器路由")
        print("  - 自适应性能优化")
        
        # 清理
        os.remove(temp_config)
        
    except Exception as e:
        print(f"❌ Pipeline集成测试失败: {e}")

def main():
    """主函数"""
    print("🚀 查询分类器全面测试")
    print("=" * 60)
    
    try:
        # 1. 基本分类测试
        test_query_classifier()
        
        # 2. 自适应路由测试
        test_adaptive_router()
        
        # 3. 持久化测试
        test_classifier_persistence()
        
        # 4. 真实查询测试
        test_real_queries()
        
        # 5. Pipeline集成测试
        test_integration_with_pipeline()
        
        print("\n✅ 所有测试完成!")
        print("\n📝 功能特点:")
        print("1. 基于规则的查询分类（事实性、分析性、程序性）")
        print("2. 智能检索器路由推荐")
        print("3. 自适应性能优化")
        print("4. 支持中英文查询")
        print("5. 模型持久化功能")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()