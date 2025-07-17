#!/usr/bin/env python3
"""
FusionRAG实验运行器
用于运行完整的论文级别实验
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

sys.path.append('.')

from tests.universal_test import test_with_config


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 数据集配置
        self.datasets = {
            'nfcorpus': 'configs/datasets/nfcorpus/high_performance.yaml',
            'trec-covid': 'configs/datasets/trec-covid/high_performance.yaml',
            'natural-questions': 'configs/datasets/natural-questions/high_performance.yaml'
        }
        
        self.baseline_configs = {
            'nfcorpus_baseline': 'configs/datasets/nfcorpus/baseline.yaml',
        }
    
    def run_single_experiment(self, config_path: str, experiment_name: str) -> dict:
        """运行单个实验"""
        print(f"🚀 运行实验: {experiment_name}")
        print(f"📋 配置文件: {config_path}")
        
        start_time = time.time()
        
        try:
            # 运行测试
            results = test_with_config(config_path, auto_download=True)
            
            if results:
                # 添加实验元信息
                results['experiment_info'] = {
                    'name': experiment_name,
                    'config_path': config_path,
                    'duration': time.time() - start_time,
                    'timestamp': time.time()
                }
                
                # 保存结果
                result_file = self.output_dir / f"{experiment_name}_results.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"✅ 实验完成: {experiment_name}")
                print(f"📄 结果保存: {result_file}")
                
                # 打印关键指标
                if 'evaluation' in results:
                    metrics = results['evaluation'].get('metrics', {})
                    if 'ndcg@10' in metrics:
                        print(f"📊 NDCG@10: {metrics['ndcg@10']:.4f}")
                    if 'recall@10' in metrics:
                        print(f"📊 Recall@10: {metrics['recall@10']:.4f}")
                
                return results
            else:
                print(f"❌ 实验失败: {experiment_name}")
                return None
                
        except Exception as e:
            print(f"❌ 实验异常: {experiment_name} - {e}")
            return None
    
    def run_all_datasets(self) -> dict:
        """运行所有数据集实验"""
        print("🎯 开始运行所有数据集实验")
        print("=" * 60)
        
        all_results = {}
        
        for dataset_name, config_path in self.datasets.items():
            if Path(config_path).exists():
                results = self.run_single_experiment(config_path, dataset_name)
                if results:
                    all_results[dataset_name] = results
            else:
                print(f"⚠️ 配置文件不存在: {config_path}")
        
        # 保存综合结果
        summary_file = self.output_dir / "all_datasets_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 综合结果保存: {summary_file}")
        
        return all_results
    
    def run_baseline_comparison(self) -> dict:
        """运行基线对比实验"""
        print("📈 开始基线对比实验")
        print("=" * 60)
        
        baseline_results = {}
        
        for exp_name, config_path in self.baseline_configs.items():
            if Path(config_path).exists():
                results = self.run_single_experiment(config_path, f"baseline_{exp_name}")
                if results:
                    baseline_results[exp_name] = results
        
        return baseline_results
    
    def generate_comparison_table(self, results: dict) -> str:
        """生成对比表格"""
        
        table = "\n📊 实验结果对比表\n"
        table += "=" * 80 + "\n"
        table += f"{'数据集':<20} {'NDCG@10':<10} {'Recall@10':<12} {'MAP':<10} {'响应时间(s)':<12}\n"
        table += "-" * 80 + "\n"
        
        for dataset_name, result in results.items():
            if 'evaluation' in result:
                metrics = result['evaluation'].get('metrics', {})
                perf = result.get('performance', {})
                
                ndcg = metrics.get('ndcg@10', 0.0)
                recall = metrics.get('recall@10', 0.0)
                map_score = metrics.get('map', 0.0)
                response_time = perf.get('avg_query_time', 0.0)
                
                table += f"{dataset_name:<20} {ndcg:<10.4f} {recall:<12.4f} {map_score:<10.4f} {response_time:<12.4f}\n"
        
        return table
    
    def run_complete_evaluation(self) -> None:
        """运行完整评估"""
        print("🎓 FusionRAG完整实验评估")
        print("=" * 80)
        
        start_time = time.time()
        
        # 1. 运行所有数据集
        all_results = self.run_all_datasets()
        
        # 2. 运行基线对比
        baseline_results = self.run_baseline_comparison()
        
        # 3. 生成对比报告
        if all_results:
            comparison_table = self.generate_comparison_table(all_results)
            print(comparison_table)
            
            # 保存对比表格
            with open(self.output_dir / "comparison_table.txt", 'w', encoding='utf-8') as f:
                f.write(comparison_table)
        
        # 4. 生成最终报告
        total_time = time.time() - start_time
        
        final_report = {
            'experiment_summary': {
                'total_experiments': len(all_results) + len(baseline_results),
                'successful_experiments': len([r for r in all_results.values() if r is not None]),
                'total_duration': total_time,
                'datasets_tested': list(all_results.keys()),
                'timestamp': time.time()
            },
            'results': all_results,
            'baselines': baseline_results
        }
        
        with open(self.output_dir / "final_report.json", 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 完整实验评估完成！")
        print(f"⏱️ 总耗时: {total_time:.2f} 秒")
        print(f"📁 结果目录: {self.output_dir}")
        print(f"📊 最终报告: {self.output_dir}/final_report.json")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FusionRAG实验运行器")
    parser.add_argument('--dataset', choices=['nfcorpus', 'trec-covid', 'natural-questions', 'all'], 
                       default='all', help='要运行的数据集')
    parser.add_argument('--output', '-o', default='experiment_results', help='输出目录')
    parser.add_argument('--baseline', action='store_true', help='运行基线对比')
    
    args = parser.parse_args()
    
    # 创建实验运行器
    runner = ExperimentRunner(args.output)
    
    if args.dataset == 'all':
        # 运行完整评估
        runner.run_complete_evaluation()
    else:
        # 运行单个数据集
        config_path = runner.datasets.get(args.dataset)
        if config_path and Path(config_path).exists():
            runner.run_single_experiment(config_path, args.dataset)
        else:
            print(f"❌ 数据集配置不存在: {args.dataset}")


if __name__ == "__main__":
    main()