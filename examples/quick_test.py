"""
快速测试脚本
用于测试自适应混合索引的基本功能
"""

import json
import time
import sys
import os

from typing import List

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from modules.adaptive_hybrid_index import create_adaptive_hybrid_index
from modules.utils.interfaces import Document, Query


def create_test_documents() -> List[Document]:
    """创建测试文档"""
    
    documents = [
        Document(
            doc_id="doc1",
            title="Python编程语言",
            text="Python是一种高级编程语言，以其简洁、易读的语法和丰富的库而闻名。它支持多种编程范式，包括面向对象、命令式和函数式编程。"
        ),
        Document(
            doc_id="doc2",
            title="机器学习简介",
            text="机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。它主要关注开发能够从数据中学习并做出预测的算法。"
        ),
        Document(
            doc_id="doc3",
            title="自然语言处理",
            text="自然语言处理(NLP)是计算机科学和人工智能的一个领域，关注计算机与人类语言之间的交互。它包括文本分类、情感分析、机器翻译等任务。"
        ),
        Document(
            doc_id="doc4",
            title="深度学习基础",
            text="深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑的学习过程。它在图像识别、语音识别和自然语言处理等领域取得了突破性进展。"
        ),
        Document(
            doc_id="doc5",
            title="检索增强生成",
            text="检索增强生成(RAG)是一种结合检索系统和生成模型的方法，通过从外部知识库检索相关信息来增强生成模型的输出质量和准确性。"
        )
    ]
    return documents


def create_test_queries() -> List[Query]:
    """创建测试查询"""
    
    queries = [
        Query(query_id="q1", text="Python是什么编程语言？"),
        Query(query_id="q2", text="机器学习的定义"),
        Query(query_id="q3", text="NLP技术"),
        Query(query_id="q4", text="深度学习与机器学习的区别"),
        Query(query_id="q5", text="RAG系统如何工作")
    ]
    return queries


def test_adaptive_hybrid_index():
    """测试自适应混合索引"""
    print("创建测试数据...")
    documents = create_test_documents()
    queries = create_test_queries()
    
    print("创建自适应混合索引...")
    config = {
        'retrievers': {
            'bm25': {},
            'efficient_vector': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'index_type': 'hnsw'
            },
            'semantic_bm25': {
                'semantic_model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'semantic_weight': 0.3
            }
        }
    }
    
    index = create_adaptive_hybrid_index(config)
    
    print("构建索引...")
    index.build_index(documents)
    
    print("\n执行查询测试:")
    for query in queries:
        print(f"\n查询: {query.text}")
        
        # 分析查询特征
        features = index.query_analyzer.analyze_query(query)
        print(f"查询类型: {features.query_type.value}")
        print(f"查询特征: 长度={features.length}, 词数={features.token_count}, 实体数={features.entity_count}")
        
        # 路由决策
        decision = index.adaptive_router.route(features)
        print(f"路由决策: 主索引={decision.primary_index}, 次级索引={decision.secondary_indices}")
        print(f"融合方法: {decision.fusion_method}, 置信度: {decision.confidence:.2f}")
        
        # 检索结果
        start_time = time.time()
        results = index.retrieve(query, top_k=3)
        end_time = time.time()
        
        print(f"检索耗时: {(end_time - start_time) * 1000:.2f}毫秒")
        print("检索结果:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.doc_id}: {result.document.title} (得分: {result.score:.4f})")
    
    print("\n获取统计信息:")
    stats = index.get_statistics()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    test_adaptive_hybrid_index()