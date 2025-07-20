#!/usr/bin/env python3
"""
测试查询分析器
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from modules.analysis.simple_query_analyzer import SimpleQueryAnalyzer, create_simple_query_analyzer
from modules.adaptive.simple_adaptive_router import SimpleAdaptiveRouter, create_simple_adaptive_router
from modules.utils.interfaces import Query, QueryType

def test_query_analyzer():
    """测试查询分析器和自适应路由器"""
    print("🧪 测试查询分析器和自适应路由器")
    
    # 创建配置
    analyzer_config = {
        'semantic_model_name': 'models/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6',
        'spacy_model_name': 'en_core_web_sm',
        'use_local_model': True
    }
    
    router_config = {
        'available_retrievers': ['BM25', 'EfficientVector'],
        'routing_strategy': 'rule_based',
        'enable_performance_feedback': True
    }
    
    # 创建查询分析器和路由器
    analyzer = create_simple_query_analyzer(analyzer_config)
    router = create_simple_adaptive_router(router_config)
    
    # 测试查询
    test_queries = [
        Query("q1", "What is diabetes?"),  # 问句，语义查询
        Query("q2", "machine learning algorithms"),  # 关键词查询
        Query("q3", "Apple Inc stock price"),  # 实体查询
        Query("q4", "How does neural network backpropagation work in deep learning?"),  # 复杂语义查询
        Query("q5", "COVID-19 symptoms treatment"),  # 混合查询
    ]
    
    print("\n分析查询特征和路由决策:")
    for query in test_queries:
        try:
            # 分析查询特征
            features = analyzer.analyze_query(query)
            print(f"\n查询: {query.text}")
            print(f"  类型: {features.query_type.value}")
            print(f"  长度: {features.length}")
            print(f"  词数: {features.word_count}")
            print(f"  是问句: {features.is_question}")
            print(f"  有数字: {features.has_numbers}")
            print(f"  有实体: {features.has_entities}")
            print(f"  复杂度: {features.complexity_level}")
            print(f"  领域: {features.domain_hint}")
            
            # 路由决策
            decision = router.route(features)
            print(f"  路由决策:")
            print(f"    选择检索器: {decision.selected_retrievers}")
            print(f"    融合方法: {decision.fusion_method}")
            print(f"    融合权重: {decision.fusion_weights}")
            print(f"    置信度: {decision.confidence:.2f}")
            print(f"    推理: {decision.reasoning}")
            
        except Exception as e:
            print(f"分析查询 '{query.text}' 失败: {e}")
    
    # 显示路由器统计信息
    print(f"\n路由器统计信息:")
    stats = router.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ 查询分析器和自适应路由器测试完成")

if __name__ == "__main__":
    test_query_analyzer()