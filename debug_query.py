#!/usr/bin/env python3
"""
调试查询处理问题
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
from modules.utils.interfaces import Query
from modules.analysis.simple_query_analyzer import create_simple_query_analyzer
from modules.retriever.bm25_retriever import BM25Retriever
from modules.retriever.efficient_vector_index import EfficientVectorIndex

def test_query_analyzer():
    """测试查询分析器"""
    print("=== 测试查询分析器 ===")
    
    # 创建查询分析器
    analyzer = create_simple_query_analyzer()
    
    # 测试查询
    test_queries = [
        Query("PLAIN-319", "Is licorice tea also harmful?"),
        Query("PLAIN-454", "Breast Cancer Cells Feed on Cholesterol"),
        Query("389", "Some scientific query"),
    ]
    
    for query in test_queries:
        try:
            print(f"分析查询: {query.query_id} - {query.text}")
            features = analyzer.analyze_query(query)
            print(f"  结果: {features.query_type.value}, 复杂度: {features.complexity_level}")
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

def test_retriever():
    """测试检索器"""
    print("\n=== 测试检索器 ===")
    
    # 加载配置
    with open("configs/paper_experiments.json", 'r') as f:
        config = json.load(f)
    
    # 加载一个简单的查询
    query = Query("PLAIN-319", "Is licorice tea also harmful?")
    
    try:
        # 测试BM25检索器
        print("测试BM25检索器...")
        bm25_retriever = BM25Retriever(config.get('bm25', {}))
        bm25_cache_path = "checkpoints/retriever_cache/bm25_nfcorpus_index.pkl"
        if os.path.exists(bm25_cache_path):
            bm25_retriever.load_index(bm25_cache_path)
            print("BM25索引加载成功")
            
            results = bm25_retriever.retrieve(query, 10)
            print(f"BM25检索结果数量: {len(results)}")
            if results:
                print(f"第一个结果: {results[0].doc_id}, 分数: {results[0].score}")
        else:
            print("BM25索引文件不存在")
            
    except Exception as e:
        print(f"BM25检索器错误: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # 测试向量检索器 - 使用正确的配置
        print("\n测试向量检索器...")
        vector_config = config.get('efficient_vector', {})
        print(f"向量检索器配置: {vector_config}")
        
        vector_retriever = EfficientVectorIndex("EfficientVector", vector_config)
        vector_cache_path = "checkpoints/retriever_cache/efficientvector_nfcorpus_index.pkl"
        if os.path.exists(vector_cache_path):
            vector_retriever.load_index(vector_cache_path)
            print("向量索引加载成功")
            
            # 添加调试信息
            print(f"文档数量: {len(vector_retriever.documents)}")
            print(f"索引大小: {vector_retriever.index.ntotal}")
            
            results = vector_retriever.retrieve(query, 10)
            print(f"向量检索结果数量: {len(results)}")
            if results:
                print(f"第一个结果: {results[0].doc_id}, 分数: {results[0].score}")
        else:
            print("向量索引文件不存在")
            
    except Exception as e:
        print(f"向量检索器错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_query_analyzer()
    test_retriever()