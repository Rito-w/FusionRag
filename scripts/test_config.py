#!/usr/bin/env python
"""
配置文件测试脚本
用于验证配置文件的有效性和运行快速测试
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_config_validity(config_path: str) -> bool:
    """测试配置文件有效性"""
    try:
        from pipeline import FusionRAGPipeline
        
        print(f"🔍 测试配置文件: {config_path}")
        
        # 尝试加载配置
        pipeline = FusionRAGPipeline(config_path)
        print("✅ 配置文件加载成功")
        
        # 检查数据文件是否存在
        config = pipeline.config
        corpus_path = config.get('data.corpus_path')
        queries_path = config.get('data.queries_path')
        qrels_path = config.get('data.qrels_path')
        
        missing_files = []
        if not Path(corpus_path).exists():
            missing_files.append(corpus_path)
        if not Path(queries_path).exists():
            missing_files.append(queries_path)
        if not Path(qrels_path).exists():
            missing_files.append(qrels_path)
        
        if missing_files:
            print("⚠️  缺少数据文件:")
            for file in missing_files:
                print(f"   - {file}")
            return False
        else:
            print("✅ 所有数据文件存在")
        
        # 显示配置摘要
        print("\n📋 配置摘要:")
        metadata = config.get('metadata', {})
        if metadata:
            print(f"   数据集: {metadata.get('dataset', 'unknown')}")
            print(f"   模板: {metadata.get('template', 'unknown')}")
            print(f"   描述: {metadata.get('description', 'unknown')}")
            print(f"   创建时间: {metadata.get('created_at', 'unknown')}")
        
        # 显示检索器配置
        retrievers = config.get('retrievers', {})
        enabled_retrievers = [name for name, cfg in retrievers.items() if cfg.get('enabled', False)]
        print(f"   启用检索器: {', '.join(enabled_retrievers)}")
        
        # 显示融合配置
        fusion = config.get('fusion', {})
        print(f"   融合方法: {fusion.get('method', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False

def run_quick_test(config_path: str, num_docs: int = 50, num_queries: int = 5) -> bool:
    """运行快速测试"""
    try:
        from pipeline import FusionRAGPipeline
        from modules.utils.interfaces import Query
        
        print(f"\n🚀 运行快速测试 (文档数: {num_docs}, 查询数: {num_queries})")
        
        # 初始化pipeline
        pipeline = FusionRAGPipeline(config_path)
        
        # 加载数据
        print("📁 加载数据...")
        pipeline.load_data()
        
        # 使用部分数据进行测试
        test_docs = pipeline.documents[:num_docs]
        test_queries = pipeline.queries[:num_queries]
        
        print(f"   使用 {len(test_docs)} 个文档, {len(test_queries)} 个查询")
        
        # 构建索引
        print("🔨 构建索引...")
        for name, retriever in pipeline.retrievers.items():
            if hasattr(retriever, 'build_index'):
                print(f"   构建 {name} 索引...")
                retriever.build_index(test_docs)
        
        # 测试查询
        print("🔍 测试查询...")
        for i, query in enumerate(test_queries):
            print(f"   查询 {i+1}/{len(test_queries)}: {query.text[:50]}...")
            results = pipeline.search(query, top_k=10)
            print(f"      返回 {len(results)} 个结果")
        
        print("✅ 快速测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 快速测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="配置文件测试工具")
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--quick-test', action='store_true', help='运行快速测试')
    parser.add_argument('--num-docs', type=int, default=50, help='快速测试使用的文档数量')
    parser.add_argument('--num-queries', type=int, default=5, help='快速测试使用的查询数量')
    
    args = parser.parse_args()
    
    config_path = args.config
    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    print("🎯 FusionRAG配置文件测试工具")
    print("=" * 50)
    
    # 测试配置有效性
    if not test_config_validity(config_path):
        print("❌ 配置文件测试失败，停止执行")
        return
    
    # 运行快速测试
    if args.quick_test:
        if not run_quick_test(config_path, args.num_docs, args.num_queries):
            print("❌ 快速测试失败")
            return
    
    print("\n✅ 所有测试通过！")

if __name__ == "__main__":
    main()
