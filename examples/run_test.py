"""
测试评估框架
"""

import sys
import os
import json
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.retriever.bm25_retriever import BM25Retriever
from modules.retriever.efficient_vector_index import EfficientVectorIndex
from modules.evaluation.evaluator import IndexEvaluator
from modules.utils.interfaces import Document, Query


def create_test_data():
    """创建测试数据"""
    # 创建文档
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
    
    # 创建查询
    queries = [
        Query(query_id="q1", text="Python是什么编程语言？"),
        Query(query_id="q2", text="机器学习的定义"),
        Query(query_id="q3", text="NLP技术"),
        Query(query_id="q4", text="深度学习与机器学习的区别"),
        Query(query_id="q5", text="RAG系统如何工作")
    ]
    
    # 创建相关性判断
    relevance_judgments = {
        "q1": {"doc1": 3, "doc3": 1},
        "q2": {"doc2": 3, "doc4": 1},
        "q3": {"doc3": 3, "doc4": 1},
        "q4": {"doc2": 2, "doc4": 3},
        "q5": {"doc5": 3}
    }
    
    return documents, queries, relevance_judgments


def test_evaluator():
    """测试评估框架"""
    print("创建测试数据...")
    documents, queries, relevance_judgments = create_test_data()
    
    print("创建检索器...")
    retrievers = {
        "BM25": BM25Retriever(),
        "Vector": EfficientVectorIndex()
    }
    
    print("构建索引...")
    for name, retriever in retrievers.items():
        print(f"构建 {name} 索引...")
        start_time = time.time()
        retriever.build_index(documents)
        end_time = time.time()
        print(f"{name} 索引构建完成，耗时: {end_time - start_time:.2f}秒")
    
    print("创建评估器...")
    evaluator = IndexEvaluator({
        'metrics': ['precision', 'recall', 'mrr', 'ndcg', 'latency'],
        'report_dir': 'reports/test'
    })
    
    print("评估检索器性能...")
    results = evaluator.evaluate_multiple_retrievers(retrievers, queries, relevance_judgments, top_k=3)
    
    print("\n评估结果:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # 生成报告
    Path("reports/test").mkdir(parents=True, exist_ok=True)
    evaluator.generate_report(results, "test", "reports/test/evaluation_test.json")
    
    print("\n评估完成!")


if __name__ == "__main__":
    test_evaluator()