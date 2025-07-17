#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FusionRAG多数据集实验示例

该脚本展示了如何使用run_experiments.py在多个数据集上运行FusionRAG系统的实验。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入实验运行脚本
from scripts.run_experiments import run_multi_dataset_experiments, compare_retrievers


def run_example():
    """运行多数据集实验示例"""
    print("\n===== FusionRAG多数据集实验示例 =====\n")
    
    # 定义要测试的数据集
    datasets = ["nfcorpus", "scifact"]
    
    # 基础配置文件
    base_config_path = "configs/fusionrag_config.yaml"
    
    # 运行实验
    print(f"在 {len(datasets)} 个数据集上运行实验: {', '.join(datasets)}\n")
    results = run_multi_dataset_experiments(
        datasets=datasets,
        base_config_path=base_config_path,
        force_rebuild=False,  # 如果索引已存在，不重新构建
        auto_download=True    # 如果数据不存在，自动下载
    )
    
    # 比较结果
    print("\n比较不同数据集上的检索器性能:\n")
    compare_retrievers("reports/multi_dataset_results.json")
    
    print("\n===== 示例完成 =====\n")


def main():
    """主函数"""
    run_example()


if __name__ == "__main__":
    main()