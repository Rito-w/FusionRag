"""
运行所有实验的主脚本
按顺序执行标准实验、消融实验和查询类型分析
支持分阶段运行和内存优化
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入各个实验脚本
from run_standard_experiments import run_multiple_datasets as run_standard
from run_ablation_experiments import run_multiple_datasets as run_ablation
from run_query_analysis_experiments import run_multiple_datasets as run_query_analysis


def monitor_memory():
    """监控内存使用情况"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        return memory_mb
    except ImportError:
        return -1


def run_experiments(config_path: str = "configs/lightweight_config.json",
                  datasets: List[str] = None,
                  top_k: int = 10,
                  sample_size: Optional[int] = None,
                  run_standard_exp: bool = True,
                  run_ablation_exp: bool = True,
                  run_query_analysis_exp: bool = True,
                  lightweight: bool = False):
    """运行所有实验"""
    start_time = time.time()
    
    # 创建报告目录
    Path("reports").mkdir(exist_ok=True)
    
    # 记录初始内存使用
    initial_memory = monitor_memory()
    if initial_memory > 0:
        print(f"初始内存使用: {initial_memory:.1f} MB")
    
    # 1. 运行标准实验
    if run_standard_exp:
        print("\n" + "=" * 60)
        print("🚀 第1阶段: 运行标准实验")
        print("=" * 60)
        
        standard_start = time.time()
        run_standard(config_path, datasets, top_k, sample_size, lightweight)
        standard_time = time.time() - standard_start
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        if monitor_memory() > 0:
            print(f"标准实验后内存使用: {monitor_memory():.1f} MB")
        print(f"标准实验完成，耗时: {standard_time:.1f}秒")
    
    # 2. 运行消融实验
    if run_ablation_exp:
        print("\n" + "=" * 60)
        print("🔬 第2阶段: 运行消融实验")
        print("=" * 60)
        
        ablation_start = time.time()
        run_ablation(config_path, datasets, top_k, sample_size)
        ablation_time = time.time() - ablation_start
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        if monitor_memory() > 0:
            print(f"消融实验后内存使用: {monitor_memory():.1f} MB")
        print(f"消融实验完成，耗时: {ablation_time:.1f}秒")
    
    # 3. 运行查询类型分析
    if run_query_analysis_exp:
        print("\n" + "=" * 60)
        print("🔍 第3阶段: 运行查询类型分析")
        print("=" * 60)
        
        query_analysis_start = time.time()
        run_query_analysis(config_path, datasets, top_k, sample_size)
        query_analysis_time = time.time() - query_analysis_start
        
        if monitor_memory() > 0:
            print(f"查询类型分析后内存使用: {monitor_memory():.1f} MB")
        print(f"查询类型分析完成，耗时: {query_analysis_time:.1f}秒")
    
    # 总结
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ 所有实验完成! 总耗时: {total_time:.1f}秒")
    print("=" * 60)
    
    # 显示各阶段耗时
    if run_standard_exp:
        print(f"标准实验耗时: {standard_time:.1f}秒")
    if run_ablation_exp:
        print(f"消融实验耗时: {ablation_time:.1f}秒")
    if run_query_analysis_exp:
        print(f"查询类型分析耗时: {query_analysis_time:.1f}秒")
    
    print(f"\n结果已保存到 reports/ 目录")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行所有实验")
    parser.add_argument("--config", type=str, default="configs/lightweight_config.json", 
                       help="实验配置文件路径")
    parser.add_argument("--datasets", type=str, nargs="+", 
                       help="要评估的数据集")
    parser.add_argument("--top_k", type=int, default=10, 
                       help="检索的文档数量")
    parser.add_argument("--sample", type=int, default=5,
                       help="查询样本大小，用于快速测试")
    parser.add_argument("--no-standard", action="store_true",
                       help="跳过标准实验")
    parser.add_argument("--no-ablation", action="store_true",
                       help="跳过消融实验")
    parser.add_argument("--no-query-analysis", action="store_true",
                       help="跳过查询类型分析")
    parser.add_argument("--lightweight", action="store_true",
                       help="使用轻量级模式（仅BM25）")
    
    args = parser.parse_args()
    
    print("🌟 自适应混合索引实验框架")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"数据集: {args.datasets or '使用配置文件中的数据集'}")
    print(f"样本大小: {args.sample or '全部查询'}")
    print(f"轻量级模式: {'是' if args.lightweight else '否'}")
    print("=" * 60)
    
    run_experiments(
        config_path=args.config,
        datasets=args.datasets,
        top_k=args.top_k,
        sample_size=args.sample,
        run_standard_exp=not args.no_standard,
        run_ablation_exp=not args.no_ablation,
        run_query_analysis_exp=not args.no_query_analysis,
        lightweight=args.lightweight
    )