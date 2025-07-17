"""
评估报告生成器
用于生成详细的评估报告，包括性能指标、对比分析和可视化图表
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import numpy as np


class ReportGenerator:
    """评估报告生成器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.report_dir = self.config.get('report_dir', 'reports')
        self.metrics = self.config.get('metrics', ['precision', 'recall', 'mrr', 'ndcg', 'latency'])
        
        # 创建报告目录
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, results: Dict[str, Dict[str, float]], 
                      dataset_name: str = "unknown",
                      output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成评估报告"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.report_dir}/{dataset_name}_evaluation_{timestamp}.json"
        
        # 准备报告数据
        report_data = {
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "results": results
        }
        
        # 保存JSON报告
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # 生成文本报告
        text_report_file = output_file.replace('.json', '.txt')
        self._generate_text_report(report_data, text_report_file)
        
        print(f"评估报告已保存到: {output_file} 和 {text_report_file}")
        
        return report_data 
   def _generate_text_report(self, report_data: Dict[str, Any], output_file: str) -> None:
        """生成文本格式的评估报告"""
        with open(output_file, 'w') as f:
            f.write(f"# 检索器评估报告 - {report_data['dataset']}\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 写入总体结果
            f.write("## 总体结果\n\n")
            f.write("| 检索器 | 准确率 | 召回率 | MRR | NDCG | 延迟(ms) |\n")
            f.write("|--------|--------|--------|-----|------|----------|\n")
            
            for name, metrics in report_data['results'].items():
                precision = metrics.get('precision', 0.0)
                recall = metrics.get('recall', 0.0)
                mrr = metrics.get('mrr', 0.0)
                ndcg = metrics.get('ndcg', 0.0)
                latency = metrics.get('latency', 0.0)
                
                f.write(f"| {name} | {precision:.4f} | {recall:.4f} | {mrr:.4f} | {ndcg:.4f} | {latency:.2f} |\n")
            
            f.write("\n")
            
            # 写入详细指标
            f.write("## 详细指标\n\n")
            for name, metrics in report_data['results'].items():
                f.write(f"### {name}\n\n")
                f.write("| 指标 | 值 |\n")
                f.write("|------|----|\n")
                
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"| {metric} | {value:.4f} |\n")
                
                f.write("\n")
    
    def generate_comparison_report(self, results_list: List[Dict[str, Dict[str, float]]],
                                 names: List[str],
                                 dataset_name: str = "unknown",
                                 output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成对比报告"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.report_dir}/{dataset_name}_comparison_{timestamp}.json"
        
        # 准备报告数据
        report_data = {
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "names": names,
            "results": results_list
        }
        
        # 保存JSON报告
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # 生成文本报告
        text_report_file = output_file.replace('.json', '.txt')
        self._generate_comparison_text_report(report_data, text_report_file)
        
        print(f"对比报告已保存到: {output_file} 和 {text_report_file}")
        
        return report_data
    
    def _generate_comparison_text_report(self, report_data: Dict[str, Any], output_file: str) -> None:
        """生成文本格式的对比报告"""
        with open(output_file, 'w') as f:
            f.write(f"# 检索器对比报告 - {report_data['dataset']}\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 写入对比结果
            f.write("## 对比结果\n\n")
            
            # 获取所有检索器名称
            all_retrievers = set()
            for result in report_data['results']:
                all_retrievers.update(result.keys())
            
            # 对每个检索器进行对比
            for retriever in sorted(all_retrievers):
                f.write(f"### {retriever}\n\n")
                f.write("| 配置 | 准确率 | 召回率 | MRR | NDCG | 延迟(ms) |\n")
                f.write("|------|--------|--------|-----|------|----------|\n")
                
                for i, result in enumerate(report_data['results']):
                    if retriever in result:
                        metrics = result[retriever]
                        precision = metrics.get('precision', 0.0)
                        recall = metrics.get('recall', 0.0)
                        mrr = metrics.get('mrr', 0.0)
                        ndcg = metrics.get('ndcg', 0.0)
                        latency = metrics.get('latency', 0.0)
                        
                        name = report_data['names'][i] if i < len(report_data['names']) else f"配置{i+1}"
                        f.write(f"| {name} | {precision:.4f} | {recall:.4f} | {mrr:.4f} | {ndcg:.4f} | {latency:.2f} |\n")
                
                f.write("\n")
    
    def generate_ablation_report(self, base_results: Dict[str, float],
                               ablation_results: Dict[str, Dict[str, float]],
                               dataset_name: str = "unknown",
                               output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成消融实验报告"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.report_dir}/{dataset_name}_ablation_{timestamp}.json"
        
        # 准备报告数据
        report_data = {
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "base_results": base_results,
            "ablation_results": ablation_results
        }
        
        # 保存JSON报告
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # 生成文本报告
        text_report_file = output_file.replace('.json', '.txt')
        self._generate_ablation_text_report(report_data, text_report_file)
        
        print(f"消融实验报告已保存到: {output_file} 和 {text_report_file}")
        
        return report_data
    
    def _generate_ablation_text_report(self, report_data: Dict[str, Any], output_file: str) -> None:
        """生成文本格式的消融实验报告"""
        with open(output_file, 'w') as f:
            f.write(f"# 消融实验报告 - {report_data['dataset']}\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 写入基准结果
            f.write("## 基准结果\n\n")
            f.write("| 指标 | 值 |\n")
            f.write("|------|----|\n")
            
            for metric, value in report_data['base_results'].items():
                if isinstance(value, (int, float)):
                    f.write(f"| {metric} | {value:.4f} |\n")
            
            f.write("\n")
            
            # 写入消融结果
            f.write("## 消融结果\n\n")
            f.write("| 组件 | 准确率 | 召回率 | MRR | NDCG | 延迟(ms) | 准确率变化 | 召回率变化 | MRR变化 | NDCG变化 |\n")
            f.write("|------|--------|--------|-----|------|----------|------------|------------|---------|----------|\n")
            
            base_precision = report_data['base_results'].get('precision', 0.0)
            base_recall = report_data['base_results'].get('recall', 0.0)
            base_mrr = report_data['base_results'].get('mrr', 0.0)
            base_ndcg = report_data['base_results'].get('ndcg', 0.0)
            
            for name, metrics in report_data['ablation_results'].items():
                precision = metrics.get('precision', 0.0)
                recall = metrics.get('recall', 0.0)
                mrr = metrics.get('mrr', 0.0)
                ndcg = metrics.get('ndcg', 0.0)
                latency = metrics.get('latency', 0.0)
                
                # 计算变化百分比
                precision_change = (precision - base_precision) / base_precision * 100 if base_precision > 0 else 0
                recall_change = (recall - base_recall) / base_recall * 100 if base_recall > 0 else 0
                mrr_change = (mrr - base_mrr) / base_mrr * 100 if base_mrr > 0 else 0
                ndcg_change = (ndcg - base_ndcg) / base_ndcg * 100 if base_ndcg > 0 else 0
                
                f.write(f"| {name} | {precision:.4f} | {recall:.4f} | {mrr:.4f} | {ndcg:.4f} | {latency:.2f} | {precision_change:+.1f}% | {recall_change:+.1f}% | {mrr_change:+.1f}% | {ndcg_change:+.1f}% |\n")
            
            f.write("\n")
            
            # 写入结论
            f.write("## 结论\n\n")
            f.write("根据消融实验结果，我们可以得出以下结论：\n\n")
            
            # 分析各组件的贡献
            component_contributions = {}
            for name, metrics in report_data['ablation_results'].items():
                if name.startswith("No"):
                    component_name = name[2:]  # 去掉"No"前缀
                    precision = metrics.get('precision', 0.0)
                    base_precision = report_data['base_results'].get('precision', 0.0)
                    contribution = (base_precision - precision) / base_precision * 100 if base_precision > 0 else 0
                    component_contributions[component_name] = contribution
            
            # 按贡献排序
            sorted_contributions = sorted(component_contributions.items(), key=lambda x: x[1], reverse=True)
            
            for component, contribution in sorted_contributions:
                if contribution > 0:
                    f.write(f"- {component}组件对准确率的贡献约为{contribution:.1f}%\n")
                else:
                    f.write(f"- 移除{component}组件对准确率没有显著影响\n")
            
            f.write("\n")


def create_report_generator(config: Dict[str, Any] = None) -> ReportGenerator:
    """创建报告生成器的工厂函数"""
    return ReportGenerator(config=config)