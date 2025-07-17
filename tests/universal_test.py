#!/usr/bin/env python
"""
FusionRAG通用测试框架
支持动态加载配置文件，适用于任何数据集和配置的完整测试
"""

import sys
import os
import time
import json
import argparse
import yaml
from pathlib import Path
sys.path.append('.')

# 设置环境变量避免段错误
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from pipeline import FusionRAGPipeline
from modules.utils.interfaces import Query
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_data_availability(config_path: str):
    """检查配置文件指定的数据是否可用"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        data_config = config.get('data', {})
        required_files = [
            data_config.get('corpus_path'),
            data_config.get('queries_path'),
            data_config.get('qrels_path')
        ]

        missing_files = []
        for file_path in required_files:
            if file_path and not Path(file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            logger.error("缺少必要的数据文件:")
            for file_path in missing_files:
                logger.error(f"  - {file_path}")

            # 从配置推断数据集名称
            dataset_name = "unknown"
            metadata = config.get('metadata', {})
            if 'dataset' in metadata:
                dataset_name = metadata['dataset']
            else:
                # 从文件路径推断
                corpus_path = data_config.get('corpus_path', '')
                if 'nfcorpus' in corpus_path:
                    dataset_name = 'nfcorpus'
                elif 'trec-covid' in corpus_path:
                    dataset_name = 'trec-covid'
                elif 'natural-questions' in corpus_path:
                    dataset_name = 'natural-questions'

            logger.info("请先运行数据下载和预处理:")
            logger.info(f"  python scripts/download_data.py --dataset {dataset_name}")
            logger.info(f"  python scripts/preprocess_data.py --dataset {dataset_name}")
            return False

        return True

    except Exception as e:
        logger.error(f"检查数据文件时出错: {e}")
        return False

def download_and_preprocess_data(dataset_name: str):
    """自动下载和预处理指定数据集"""
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

def test_with_config():
    """使用指定配置文件进行测试"""
    config_path = "configs/ms_marco_config.yaml"
    auto_download = True
    
    logger.info("🚀 开始配置化测试")
    logger.info("=" * 60)

    # 检查配置文件
    if not Path(config_path).exists():
        logger.error(f"配置文件不存在: {config_path}")
        return None

    # 加载配置获取数据集信息
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dataset_name = config.get('metadata', {}).get('dataset', 'unknown')
    template_name = config.get('metadata', {}).get('template', 'unknown')

    logger.info(f"📋 配置文件: {config_path}")
    logger.info(f"📊 数据集: {dataset_name}")
    logger.info(f"🎯 模板: {template_name}")

    # 检查数据
    if not check_data_availability(config_path):
        if auto_download:
            logger.info("尝试自动下载数据...")
            if not download_and_preprocess_data(dataset_name):
                logger.error("无法获取测试数据，测试终止")
                return None
        else:
            logger.error("数据文件缺失，测试终止")
            return None

    # 初始化pipeline
    pipeline = FusionRAGPipeline(config_path)
    
    # 加载数据
    logger.info("📁 加载完整数据集...")
    start_time = time.time()
    pipeline.load_data()
    load_time = time.time() - start_time
    
    logger.info(f"数据加载完成 ({load_time:.2f}s):")
    logger.info(f"  📄 文档数量: {len(pipeline.documents):,}")
    logger.info(f"  🔍 查询数量: {len(pipeline.queries):,}")
    logger.info(f"  🏷️ 标注数量: {len(pipeline.qrels):,}")
    
    # 构建索引
    logger.info("🔨 构建索引...")
    start_time = time.time()
    pipeline.build_indexes(force_rebuild=False)  # 允许使用缓存
    build_time = time.time() - start_time
    
    logger.info(f"索引构建完成 ({build_time:.2f}s)")
    
    # 运行完整评测
    logger.info("📊 运行完整性能评测...")
    start_time = time.time()
    
    # 批量检索
    search_results = pipeline.batch_search()
    search_time = time.time() - start_time
    
    logger.info(f"批量检索完成 ({search_time:.2f}s)")
    logger.info(f"  平均每查询: {search_time/len(pipeline.queries)*1000:.2f}ms")
    
    # 评测结果
    logger.info("📈 计算评测指标...")
    evaluation_results = pipeline.evaluate(search_results)
    
    # 显示结果
    if evaluation_results:
        logger.info("🎯 高性能模型测试结果:")
        logger.info("=" * 60)
        
        # 主要指标
        for metric_name, metric_data in evaluation_results.items():
            if isinstance(metric_data, dict):
                logger.info(f"{metric_name.upper()}:")
                for k, v in metric_data.items():
                    if isinstance(v, (int, float)):
                        logger.info(f"  {k}: {v:.4f}")
                    elif isinstance(v, dict) and 'mean' in v:
                        logger.info(f"  {k}: {v['mean']:.4f} (±{v.get('std', 0):.4f})")
        
        # 生成详细报告
        report = pipeline.evaluator.generate_report(evaluation_results)
        logger.info("\n📋 详细评测报告:")
        logger.info("=" * 60)
        for line in report.split('\n'):
            if line.strip():
                logger.info(line)
    
    # 性能总结
    total_time = load_time + build_time + search_time
    logger.info(f"\n⏱️ 性能总结:")
    logger.info(f"  数据加载: {load_time:.2f}s")
    logger.info(f"  索引构建: {build_time:.2f}s") 
    logger.info(f"  批量检索: {search_time:.2f}s")
    logger.info(f"  总耗时: {total_time:.2f}s")
    
    # 保存结果
    results_summary = {
        "config": config_path,
        "dataset": dataset_name,
        "template": template_name,
        "dataset_stats": {
            "documents": len(pipeline.documents),
            "queries": len(pipeline.queries),
            "qrels": len(pipeline.qrels)
        },
        "performance": {
            "load_time": load_time,
            "build_time": build_time,
            "search_time": search_time,
            "total_time": total_time,
            "avg_query_time": search_time / len(pipeline.queries)
        },
        "evaluation": evaluation_results
    }

    # 生成结果文件名（基于配置文件名）
    config_name = Path(config_path).stem
    output_file = f"checkpoints/logs/{config_name}_test_results.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"📄 详细结果已保存到: {output_file}")

    return results_summary

def compare_results(result1_file: str, result2_file: str):
    """比较两个测试结果"""
    logger.info("🔄 比较测试结果...")

    try:
        with open(result1_file, 'r') as f:
            results1 = json.load(f)
        with open(result2_file, 'r') as f:
            results2 = json.load(f)

        logger.info("📊 结果对比:")
        logger.info(f"  配置1: {results1.get('config', 'unknown')}")
        logger.info(f"  配置2: {results2.get('config', 'unknown')}")

        # 比较主要指标
        eval1 = results1.get('evaluation', {}).get('metrics', {})
        eval2 = results2.get('evaluation', {}).get('metrics', {})

        for metric in ['recall@5', 'recall@10', 'ndcg@10', 'map']:
            if metric in eval1 and metric in eval2:
                val1, val2 = eval1[metric], eval2[metric]
                improvement = ((val2 - val1) / val1 * 100) if val1 > 0 else 0
                logger.info(f"  {metric}: {val1:.4f} → {val2:.4f} ({improvement:+.1f}%)")

    except Exception as e:
        logger.warning(f"无法比较结果: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FusionRAG通用测试框架")
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--no-auto-download', action='store_true', help='不自动下载数据')
    parser.add_argument('--compare-with', type=str, help='与指定结果文件对比')

    args = parser.parse_args()

    logger.info("🎯 FusionRAG通用测试框架")
    logger.info("=" * 60)

    try:
        # 运行测试
        results = test_with_config(args.config, not args.no_auto_download)

        if results is None:
            logger.error("测试失败")
            return

        # 对比结果
        if args.compare_with:
            if Path(args.compare_with).exists():
                config_name = Path(args.config).stem
                current_result = f"checkpoints/logs/{config_name}_test_results.json"
                compare_results(args.compare_with, current_result)
            else:
                logger.warning(f"对比文件不存在: {args.compare_with}")

        logger.info("\n✅ 测试完成!")
        logger.info("系统已完成指定配置的完整测试")

        return results

    except KeyboardInterrupt:
        logger.info("\n⏹️ 用户中断测试")
    except Exception as e:
        logger.error(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
