"""
简化版测试脚本
不使用查询分析器和自适应路由，只测试基本检索功能
"""

import sys
import os
import time
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.retriever.bm25_retriever import BM25Retriever
from modules.retriever.efficient_vector_index import EfficientVectorIndex
from modules.utils.interfaces import Document, Query


def create_test_documents():
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


def create_test_queries():
    """创建测试查询"""
    queries = [
        Query(query_id="q1", text="Python是什么编程语言？"),
        Query(query_id="q2", text="机器学习的定义"),
        Query(query_id="q3", text="NLP技术"),
        Query(query_id="q4", text="深度学习与机器学习的区别"),
        Query(query_id="q5", text="RAG系统如何工作")
    ]
    return queries


def test_bm25_retriever():
    """测试BM25检索器"""
    print("\n=== 测试BM25检索器 ===")
    documents = create_test_documents()
    queries = create_test_queries()
    
    # 创建BM25检索器
    retriever = BM25Retriever()
    
    # 构建索引
    print("构建BM25索引...")
    retriever.build_index(documents)
    
    # 执行查询
    for query in queries:
        print(f"\n查询: {query.text}")
        
        start_time = time.time()
        results = retriever.retrieve(query, top_k=3)
        end_time = time.time()
        
        print(f"检索耗时: {(end_time - start_time) * 1000:.2f}毫秒")
        print("检索结果:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.doc_id}: {result.document.title} (得分: {result.score:.4f})")


def test_vector_retriever():
    """测试向量检索器"""
    print("\n=== 测试向量检索器 ===")
    documents = create_test_documents()
    queries = create_test_queries()
    
    # 创建向量检索器
    retriever = EfficientVectorIndex()
    
    # 构建索引
    print("构建向量索引...")
    retriever.build_index(documents)
    
    # 执行查询
    for query in queries:
        print(f"\n查询: {query.text}")
        
        start_time = time.time()
        results = retriever.retrieve(query, top_k=3)
        end_time = time.time()
        
        print(f"检索耗时: {(end_time - start_time) * 1000:.2f}毫秒")
        print("检索结果:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.doc_id}: {result.document.title} (得分: {result.score:.4f})")


if __name__ == "__main__":
    print("开始简化测试...")
    test_bm25_retriever()
    test_vector_retriever()
    print("\n测试完成!")