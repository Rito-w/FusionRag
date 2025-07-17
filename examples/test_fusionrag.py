#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FusionRAG系统测试脚本

该脚本演示了如何使用FusionRAG系统的主要功能，包括：
1. 初始化系统
2. 加载文档
3. 构建索引
4. 执行检索
5. 评估性能
"""

import os
import sys
import time
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fusionrag import FusionRAGSystem
from modules.utils.interfaces import Query, Document


def test_basic_functionality(config_path):
    """测试基本功能"""
    print("\n===== 测试基本功能 =====")
    
    # 初始化系统
    print("\n1. 初始化FusionRAG系统...")
    system = FusionRAGSystem(config_path=config_path)
    
    # 加载示例文档
    print("\n2. 创建示例文档...")
    documents = [
        Document(
            doc_id="doc1",
            title="机器学习简介",
            text="机器学习是人工智能的一个分支，它使用统计学方法让计算机系统能够从数据中学习和改进。"
        ),
        Document(
            doc_id="doc2",
            title="深度学习技术",
            text="深度学习是机器学习的一个子领域，它使用多层神经网络来模拟人脑的学习过程，已在图像识别、自然语言处理等领域取得突破。"
        ),
        Document(
            doc_id="doc3",
            title="自然语言处理",
            text="自然语言处理(NLP)是计算机科学和人工智能的一个领域，专注于使计算机能够理解、解释和生成人类语言。"
        ),
        Document(
            doc_id="doc4",
            title="计算机视觉",
            text="计算机视觉是一个跨学科领域，研究如何使计算机能够从图像或视频中获取高层次的理解，模拟人类视觉系统的功能。"
        ),
        Document(
            doc_id="doc5",
            title="强化学习",
            text="强化学习是机器学习的一种方法，它通过让智能体在环境中采取行动并获得奖励或惩罚来学习最优策略。"
        )
    ]
    
    # 构建索引
    print("\n3. 为文档构建索引...")
    system.index_documents(documents, force_rebuild=True)
    
    # 执行检索
    print("\n4. 执行检索测试...")
    queries = [
        "什么是机器学习？",
        "深度学习和传统机器学习有什么区别？",
        "计算机如何理解人类语言？",
        "人工智能在图像识别中的应用"
    ]
    
    for i, query_text in enumerate(queries):
        print(f"\n查询 {i+1}: {query_text}")
        start_time = time.time()
        results = system.retrieve(query_text, top_k=3, use_adaptive=True)
        end_time = time.time()
        
        print(f"检索耗时: {(end_time - start_time)*1000:.2f}毫秒")
        print("检索结果:")
        for j, doc in enumerate(results):
            print(f"  {j+1}. {doc.title} (ID: {doc.doc_id})")
            print(f"     {doc.text[:100]}..." if len(doc.text) > 100 else f"     {doc.text}")
    
    # 获取系统统计信息
    print("\n5. 获取系统统计信息...")
    stats = system.get_statistics()
    print(f"检索器数量: {len(stats['retrievers'])}")
    print(f"可用检索器: {', '.join(stats['retrievers'].keys())}")
    
    return system


def test_adaptive_routing(system):
    """测试自适应路由功能"""
    print("\n===== 测试自适应路由 =====")
    
    # 不同类型的查询
    query_types = {
        "factual": "什么是强化学习？",
        "analytical": "比较深度学习和传统机器学习的优缺点",
        "procedural": "如何实现一个基本的神经网络模型？"
    }
    
    for query_type, query_text in query_types.items():
        print(f"\n查询类型: {query_type}")
        print(f"查询文本: {query_text}")
        
        # 使用自适应路由
        start_time = time.time()
        adaptive_results = system.retrieve(query_text, top_k=3, use_adaptive=True)
        adaptive_time = time.time() - start_time
        
        # 不使用自适应路由
        start_time = time.time()
        default_results = system.retrieve(query_text, top_k=3, use_adaptive=False)
        default_time = time.time() - start_time
        
        print(f"自适应路由耗时: {adaptive_time*1000:.2f}毫秒")
        print(f"默认检索耗时: {default_time*1000:.2f}毫秒")
        
        print("自适应路由结果:")
        for i, doc in enumerate(adaptive_results[:2]):
            print(f"  {i+1}. {doc.title} (ID: {doc.doc_id})")
        
        print("默认检索结果:")
        for i, doc in enumerate(default_results[:2]):
            print(f"  {i+1}. {doc.title} (ID: {doc.doc_id})")


def test_retriever_comparison(system):
    """测试不同检索器的性能比较"""
    print("\n===== 检索器性能比较 =====")
    
    query_text = "机器学习和人工智能的关系是什么？"
    print(f"查询: {query_text}")
    
    # 获取所有检索器
    retrievers = system.retrievers
    
    # 对每个检索器执行检索
    for name, retriever in retrievers.items():
        start_time = time.time()
        results = retriever.retrieve(Query(query_id="q1", text=query_text), top_k=3)
        end_time = time.time()
        
        print(f"\n检索器: {name}")
        print(f"耗时: {(end_time - start_time)*1000:.2f}毫秒")
        print("结果:")
        for i, result in enumerate(results[:2]):
            print(f"  {i+1}. {result.document.title} (分数: {result.score:.4f})")


def main():
    parser = argparse.ArgumentParser(description="FusionRAG系统测试脚本")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    print("\n========== FusionRAG系统测试 ==========\n")
    print(f"使用配置文件: {args.config}")
    
    # 测试基本功能
    system = test_basic_functionality(args.config)
    
    # 测试自适应路由
    test_adaptive_routing(system)
    
    # 测试检索器比较
    test_retriever_comparison(system)
    
    print("\n========== 测试完成 ==========\n")


if __name__ == "__main__":
    main()