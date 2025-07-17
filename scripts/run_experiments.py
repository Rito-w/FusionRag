#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FusionRAG多数据集实验运行脚本

该脚本用于在多个数据集上运行FusionRAG系统的实验，支持：
1. 自动加载和处理数据集
2. 构建索引
3. 运行检索实验
4. 评估性能
5. 生成详细报告
"""

import os
import sys
import time
import json
import argparse
import yaml
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置环境变量避免段错误
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# 导入FusionRAG系统
from fusionrag import FusionRAGSystem
from modules.utils.interfaces import Document, Query

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_data_availability(dataset_name: str) -> bool:
    """检查数据集是否可用
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        数据集是否可用
    """
    # 构建数据文件路径
    # 首先检查新格式路径（直接在processed目录下）
    corpus_path = f"data/processed/{dataset_name}_corpus.jsonl"
    queries_path = f"data/processed/{dataset_name}_queries.jsonl"
    qrels_path = f"data/processed/{dataset_name}_qrels.tsv"
    
    # 如果新格式路径不存在，则检查旧格式路径（在dataset_name子目录下）
    if not (Path(corpus_path).exists() and Path(queries_path).exists() and Path(qrels_path).exists()):
        corpus_path = f"data/processed/{dataset_name}/corpus.jsonl"
        queries_path = f"data/processed/{dataset_name}/queries.jsonl"
        qrels_path = f"data/processed/{dataset_name}/qrels.tsv"
    
    # 检查文件是否存在
    missing_files = []
    for file_path in [corpus_path, queries_path, qrels_path]:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"数据集 {dataset_name} 缺少必要的数据文件:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    
    return True


def download_and_preprocess_data(dataset_name: str) -> bool:
    """自动下载和预处理指定数据集
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        是否成功
    """
    logger.info(f"🔄 自动下载和预处理{dataset_name}数据集...")
    
    try:
        # 下载数据
        logger.info(f"📥 下载{dataset_name}数据集...")
        from scripts.download_data import DataDownloader
        downloader = DataDownloader()
        success = downloader.download_beir_dataset(dataset_name)
        
        if not success:
            logger.error("数据下载失败")
            return False
        
        # 预处理数据
        logger.info("⚙️ 预处理数据...")
        from scripts.preprocess_data import DataProcessor
        processor = DataProcessor()
        success = processor.process_beir_dataset(dataset_name)
        
        if not success:
            logger.error("数据预处理失败")
            return False
        
        logger.info("✅ 数据准备完成")
        return True
        
    except Exception as e:
        logger.error(f"数据准备失败: {e}")
        return False


def create_dataset_config(dataset_name: str, base_config_path: str) -> str:
    """为数据集创建配置文件
    
    Args:
        dataset_name: 数据集名称
        base_config_path: 基础配置文件路径
        
    Returns:
        新配置文件路径
    """
    # 加载基础配置
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 处理自动设备配置
    if config.get("system", {}).get("device") == "auto":
        import torch
        config["system"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"自动选择设备: {config['system']['device']}")
    
    # 更新数据路径
    # 检查新格式路径（直接在processed目录下）
    corpus_path = f"data/processed/{dataset_name}_corpus.jsonl"
    queries_path = f"data/processed/{dataset_name}_queries.jsonl"
    qrels_path = f"data/processed/{dataset_name}_qrels.tsv"
    
    # 如果新格式路径不存在，则使用旧格式路径（在dataset_name子目录下）
    if not (Path(corpus_path).exists() and Path(queries_path).exists() and Path(qrels_path).exists()):
        corpus_path = f"data/processed/{dataset_name}/corpus.jsonl"
        queries_path = f"data/processed/{dataset_name}/queries.jsonl"
        qrels_path = f"data/processed/{dataset_name}/qrels.tsv"
    
    config['data']['corpus_path'] = corpus_path
    config['data']['queries_path'] = queries_path
    config['data']['qrels_path'] = qrels_path
    
    # 更新元数据
    if 'metadata' not in config:
        config['metadata'] = {}
    config['metadata']['dataset'] = dataset_name
    
    # 保存新配置
    output_dir = Path("configs/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{dataset_name}_config.yaml"
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"已为数据集 {dataset_name} 创建配置文件: {output_path}")
    return str(output_path)


def run_experiment(config_path: str, force_rebuild: bool = False, auto_download: bool = True) -> Dict[str, Any]:
    """使用指定配置文件运行实验
    
    Args:
        config_path: 配置文件路径
        force_rebuild: 是否强制重建索引
        auto_download: 是否自动下载数据
        
    Returns:
        实验结果
    """
    logger.info("🚀 开始配置化实验")
    logger.info("=" * 60)
    
    # 检查配置文件
    if not Path(config_path).exists():
        logger.error(f"配置文件不存在: {config_path}")
        return {}
    
    # 加载配置获取数据集信息
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 处理自动设备配置
    if config.get("system", {}).get("device") == "auto":
        import torch
        config["system"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"自动选择设备: {config['system']['device']}")
        
        # 保存更新后的配置
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    dataset_name = config.get('metadata', {}).get('dataset', 'unknown')
    
    logger.info(f"📋 配置文件: {config_path}")
    logger.info(f"📊 数据集: {dataset_name}")
    
    # 检查数据
    if not check_data_availability(dataset_name):
        if auto_download:
            logger.info("尝试自动下载数据...")
            if not download_and_preprocess_data(dataset_name):
                logger.error("无法获取测试数据，测试终止")
                return {}
        else:
            logger.error("数据文件缺失，测试终止")
            return {}
    
    # 初始化FusionRAG系统
    start_time = time.time()
    system = FusionRAGSystem(config_path=config_path)
    init_time = time.time() - start_time
    logger.info(f"系统初始化完成 ({init_time:.2f}s)")
    
    # 加载文档
    logger.info("📁 加载文档...")
    start_time = time.time()
    documents = system.load_documents()
    load_time = time.time() - start_time
    logger.info(f"文档加载完成 ({load_time:.2f}s), 共 {len(documents)} 个文档")
    
    # 构建索引
    logger.info("🔨 构建索引...")
    start_time = time.time()
    system.index_documents(documents, force_rebuild=force_rebuild)
    build_time = time.time() - start_time
    logger.info(f"索引构建完成 ({build_time:.2f}s)")
    
    # 运行评估
    logger.info("📊 运行性能评估...")
    start_time = time.time()
    evaluation_results = system.evaluate(dataset_name=dataset_name)
    eval_time = time.time() - start_time
    logger.info(f"评估完成 ({eval_time:.2f}s)")
    
    # 显示结果
    if evaluation_results:
        logger.info("🎯 实验结果:")
        logger.info("=" * 60)
        
        # 显示主要指标
        for dataset, results in evaluation_results.items():
            logger.info(f"数据集: {dataset}")
            
            for retriever_name, metrics in results.items():
                logger.info(f"检索器: {retriever_name}")
                
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {metric_name}: {value:.4f}")
    
    # 性能总结
    total_time = init_time + load_time + build_time + eval_time
    logger.info(f"\n⏱️ 性能总结:")
    logger.info(f"  系统初始化: {init_time:.2f}s")
    logger.info(f"  数据加载: {load_time:.2f}s")
    logger.info(f"  索引构建: {build_time:.2f}s") 
    logger.info(f"  性能评估: {eval_time:.2f}s")
    logger.info(f"  总耗时: {total_time:.2f}s")
    
    # 保存结果
    results_summary = {
        "config": config_path,
        "dataset": dataset_name,
        "performance": {
            "init_time": init_time,
            "load_time": load_time,
            "build_time": build_time,
            "eval_time": eval_time,
            "total_time": total_time
        },
        "evaluation": evaluation_results
    }
    
    # 生成结果文件名
    config_name = Path(config_path).stem
    output_file = f"reports/{config_name}_results.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 详细结果已保存到: {output_file}")
    
    return results_summary


def run_multi_dataset_experiments(datasets: List[str], base_config_path: str, force_rebuild: bool = False, auto_download: bool = True) -> Dict[str, Any]:
    """在多个数据集上运行实验
    
    Args:
        datasets: 数据集名称列表
        base_config_path: 基础配置文件路径
        force_rebuild: 是否强制重建索引
        auto_download: 是否自动下载数据
        
    Returns:
        所有实验结果
    """
    logger.info(f"🚀 开始在 {len(datasets)} 个数据集上运行实验")
    logger.info("=" * 60)
    
    all_results = {}
    
    for dataset_name in datasets:
        logger.info(f"\n📊 数据集: {dataset_name}")
        logger.info("-" * 40)
        
        # 为数据集创建配置
        config_path = create_dataset_config(dataset_name, base_config_path)
        
        # 运行实验
        results = run_experiment(config_path, force_rebuild, auto_download)
        all_results[dataset_name] = results
    
    # 保存汇总结果
    output_file = f"reports/multi_dataset_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📄 所有数据集的汇总结果已保存到: {output_file}")
    
    return all_results


def compare_retrievers(results_file: str):
    """比较不同检索器的性能
    
    Args:
        results_file: 结果文件路径
    """
    logger.info("🔄 比较检索器性能...")
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if not results:
            logger.warning("结果文件为空")
            return
        
        # 提取评估结果
        if isinstance(results, dict) and 'evaluation' in results:
            # 单数据集结果
            evaluation_results = results['evaluation']
        elif isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
            # 多数据集结果
            evaluation_results = {}
            for dataset, dataset_results in results.items():
                if 'evaluation' in dataset_results:
                    evaluation_results[dataset] = dataset_results['evaluation']
        else:
            logger.warning("无法解析结果文件格式")
            return
        
        # 显示比较结果
        logger.info("\n📊 检索器性能比较:")
        logger.info("=" * 60)
        
        for dataset, dataset_results in evaluation_results.items():
            logger.info(f"\n数据集: {dataset}")
            logger.info("-" * 40)
            
            # 提取所有检索器和指标
            retrievers = set()
            metrics = set()
            
            for retriever_results in dataset_results.values():
                for retriever, retriever_metrics in retriever_results.items():
                    retrievers.add(retriever)
                    metrics.update(retriever_metrics.keys())
            
            # 按指标比较
            for metric in sorted(metrics):
                logger.info(f"\n指标: {metric}")
                
                # 收集所有检索器在此指标上的性能
                retriever_scores = {}
                
                for retriever_results in dataset_results.values():
                    for retriever, retriever_metrics in retriever_results.items():
                        if metric in retriever_metrics:
                            retriever_scores[retriever] = retriever_metrics[metric]
                
                # 按性能排序
                sorted_retrievers = sorted(retriever_scores.items(), key=lambda x: x[1], reverse=True)
                
                # 显示排序结果
                for i, (retriever, score) in enumerate(sorted_retrievers):
                    logger.info(f"  {i+1}. {retriever}: {score:.4f}")
    
    except Exception as e:
        logger.error(f"比较检索器性能失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FusionRAG多数据集实验运行脚本")
    parser.add_argument("--config", type=str, default="configs/fusionrag_config.yaml", help="基础配置文件路径")
    parser.add_argument("--datasets", type=str, nargs="+", help="要运行的数据集列表")
    parser.add_argument("--all-datasets", action="store_true", help="运行所有可用的数据集")
    parser.add_argument("--force-rebuild", action="store_true", help="强制重建索引")
    parser.add_argument("--no-auto-download", action="store_true", help="不自动下载数据")
    parser.add_argument("--compare", type=str, help="比较指定结果文件中的检索器性能")
    
    args = parser.parse_args()
    
    logger.info("🎯 FusionRAG多数据集实验运行脚本")
    logger.info("=" * 60)
    
    if args.compare:
        # 比较检索器性能
        compare_retrievers(args.compare)
        return
    
    # 确定要运行的数据集
    if args.all_datasets:
        # 获取所有可用的数据集
        datasets = [
            "nfcorpus",
            "scifact",
            "scidocs",
            "fiqa",
            "trec-covid",
            "arguana",
            "webis-touche2020"
        ]
        logger.info(f"将在所有 {len(datasets)} 个数据集上运行实验")
    elif args.datasets:
        datasets = args.datasets
        logger.info(f"将在指定的 {len(datasets)} 个数据集上运行实验")
    else:
        # 默认使用NFCorpus数据集
        datasets = ["nfcorpus"]
        logger.info("未指定数据集，将使用默认的NFCorpus数据集")
    
    # 运行多数据集实验
    run_multi_dataset_experiments(
        datasets=datasets,
        base_config_path=args.config,
        force_rebuild=args.force_rebuild,
        auto_download=not args.no_auto_download
    )
    
    logger.info("\n✅ 所有实验完成!")


if __name__ == "__main__":
    main()