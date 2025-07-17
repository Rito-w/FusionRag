"""
评测模块实现
支持多种IR评测指标：Recall@K, Precision@K, NDCG@K, MAP等
"""

import json
import math
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime

from ..utils.interfaces import BaseEvaluator, Query, FusionResult
from ..utils.common import FileUtils, MetricsCalculator


class IRMetricsEvaluator(BaseEvaluator):
    """信息检索评测器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        self.metrics = self.config.get('metrics', ['recall@5', 'recall@10', 'ndcg@10', 'map'])
        self.output_path = self.config.get('output_path', 'checkpoints/logs/eval_results.json')
        
        # 支持的评测指标
        self.metric_functions = {
            'recall@k': self._recall_at_k,
            'precision@k': self._precision_at_k,
            'ndcg@k': self._ndcg_at_k,
            'map': self._mean_average_precision,
            'mrr': self._mean_reciprocal_rank,
            'success@k': self._success_at_k
        }
    
    def evaluate(self, 
                 predictions: List[List[str]], 
                 ground_truth: List[List[str]]) -> Dict[str, float]:
        """评测检索结果（简化版接口）"""
        
        # 转换为标准格式
        query_predictions = {}
        query_ground_truth = {}
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            query_id = f"q_{i:03d}"
            query_predictions[query_id] = pred
            query_ground_truth[query_id] = gt
        
        return self.evaluate_retrieval(query_predictions, query_ground_truth)
    
    def evaluate_retrieval(self, 
                          query_predictions: Dict[str, List[str]], 
                          query_ground_truth: Dict[str, List[str]]) -> Dict[str, float]:
        """评测检索结果（完整版）"""
        
        results = {}
        query_results = defaultdict(dict)
        
        # 逐查询计算指标
        for query_id in query_predictions:
            if query_id not in query_ground_truth:
                continue
            
            predictions = query_predictions[query_id]
            ground_truth = query_ground_truth[query_id]
            
            # 计算各项指标
            for metric in self.metrics:
                score = self._compute_metric(metric, predictions, ground_truth)
                query_results[query_id][metric] = score
        
        # 计算平均指标
        if query_results:
            for metric in self.metrics:
                scores = [query_results[qid].get(metric, 0.0) for qid in query_results]
                results[metric] = sum(scores) / len(scores)
        
        # 添加统计信息
        results['num_queries'] = len(query_results)
        results['timestamp'] = datetime.now().isoformat()
        
        return results
    
    def _compute_metric(self, metric: str, predictions: List[str], ground_truth: List[str]) -> float:
        """计算单个指标"""
        
        # 解析指标名称和参数
        if '@' in metric:
            metric_name, k_str = metric.split('@')
            k = int(k_str)
        else:
            metric_name = metric
            k = len(predictions)
        
        # 调用对应的计算函数
        if metric_name == 'recall':
            return self._recall_at_k(predictions, ground_truth, k)
        elif metric_name == 'precision':
            return self._precision_at_k(predictions, ground_truth, k)
        elif metric_name == 'ndcg':
            return self._ndcg_at_k(predictions, ground_truth, k)
        elif metric_name == 'map':
            return self._mean_average_precision(predictions, ground_truth)
        elif metric_name == 'mrr':
            return self._mean_reciprocal_rank(predictions, ground_truth)
        elif metric_name == 'success':
            return self._success_at_k(predictions, ground_truth, k)
        else:
            return 0.0
    
    def _recall_at_k(self, predictions: List[str], ground_truth: List[str], k: int) -> float:
        """计算Recall@K"""
        return MetricsCalculator.recall_at_k(predictions, ground_truth, k)
    
    def _precision_at_k(self, predictions: List[str], ground_truth: List[str], k: int) -> float:
        """计算Precision@K"""
        return MetricsCalculator.precision_at_k(predictions, ground_truth, k)
    
    def _ndcg_at_k(self, predictions: List[str], ground_truth: List[str], k: int) -> float:
        """计算NDCG@K"""
        return MetricsCalculator.ndcg_at_k(predictions, ground_truth, k)
    
    def _mean_average_precision(self, predictions: List[str], ground_truth: List[str]) -> float:
        """计算平均精确度(AP)"""
        return MetricsCalculator.average_precision(predictions, ground_truth)
    
    def _mean_reciprocal_rank(self, predictions: List[str], ground_truth: List[str]) -> float:
        """计算倒数排名(RR)"""
        if not ground_truth:
            return 0.0
        
        relevant = set(ground_truth)
        
        for rank, pred in enumerate(predictions, 1):
            if pred in relevant:
                return 1.0 / rank
        
        return 0.0
    
    def _success_at_k(self, predictions: List[str], ground_truth: List[str], k: int) -> float:
        """计算Success@K (至少命中一个相关文档)"""
        if not ground_truth:
            return 0.0
        
        relevant = set(ground_truth)
        pred_at_k = set(predictions[:k])
        
        return 1.0 if len(pred_at_k & relevant) > 0 else 0.0
    
    def evaluate_fusion_results(self, 
                               query_results: Dict[str, List[FusionResult]], 
                               ground_truth: Dict[str, List[str]]) -> Dict[str, Any]:
        """评测融合结果"""
        
        # 转换融合结果为预测列表
        query_predictions = {}
        for query_id, results in query_results.items():
            predictions = [result.doc_id for result in results]
            query_predictions[query_id] = predictions
        
        # 基础评测
        metrics = self.evaluate_retrieval(query_predictions, ground_truth)
        
        # 添加融合特定的分析
        fusion_analysis = self._analyze_fusion_results(query_results)
        
        return {
            'metrics': metrics,
            'fusion_analysis': fusion_analysis
        }
    
    def _analyze_fusion_results(self, query_results: Dict[str, List[FusionResult]]) -> Dict[str, Any]:
        """分析融合结果"""
        
        if not query_results:
            return {}
        
        # 统计每个检索器的贡献
        retriever_contributions = defaultdict(int)
        total_results = 0
        
        score_distributions = defaultdict(list)
        
        for query_id, results in query_results.items():
            for result in results:
                total_results += 1
                
                # 统计检索器贡献
                for retriever_name in result.individual_scores.keys():
                    retriever_contributions[retriever_name] += 1
                
                # 收集分数分布
                for retriever_name, score in result.individual_scores.items():
                    score_distributions[retriever_name].append(score)
        
        # 计算统计信息
        analysis = {
            'total_results': total_results,
            'retriever_contributions': dict(retriever_contributions),
            'retriever_statistics': {}
        }
        
        # 每个检索器的统计
        for retriever_name, scores in score_distributions.items():
            if scores:
                analysis['retriever_statistics'][retriever_name] = {
                    'mean_score': sum(scores) / len(scores),
                    'max_score': max(scores),
                    'min_score': min(scores),
                    'contribution_rate': retriever_contributions[retriever_name] / total_results
                }
        
        return analysis
    
    def save_results(self, results: Dict[str, Any], filepath: Optional[str] = None) -> None:
        """保存评测结果"""
        save_path = filepath or self.output_path
        FileUtils.save_json(results, save_path)
        print(f"评测结果已保存到: {save_path}")
    
    def load_results(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """加载评测结果"""
        load_path = filepath or self.output_path
        return FileUtils.load_json(load_path)
    
    def compare_results(self, 
                       results1: Dict[str, Any], 
                       results2: Dict[str, Any],
                       label1: str = "Result1",
                       label2: str = "Result2") -> Dict[str, Any]:
        """比较两组评测结果"""
        
        comparison = {
            'comparison_info': {
                'label1': label1,
                'label2': label2,
                'timestamp': datetime.now().isoformat()
            },
            'metric_comparison': {},
            'improvements': {}
        }
        
        # 比较指标
        metrics1 = results1.get('metrics', {})
        metrics2 = results2.get('metrics', {})
        
        for metric in set(metrics1.keys()) | set(metrics2.keys()):
            if metric in ['num_queries', 'timestamp']:
                continue
                
            score1 = metrics1.get(metric, 0.0)
            score2 = metrics2.get(metric, 0.0)
            
            comparison['metric_comparison'][metric] = {
                label1: score1,
                label2: score2,
                'difference': score2 - score1,
                'relative_improvement': ((score2 - score1) / score1 * 100) if score1 > 0 else 0.0
            }
        
        # 总结改进情况
        improvements = []
        regressions = []
        
        for metric, comp in comparison['metric_comparison'].items():
            if comp['difference'] > 0:
                improvements.append((metric, comp['relative_improvement']))
            elif comp['difference'] < 0:
                regressions.append((metric, comp['relative_improvement']))
        
        comparison['improvements'] = {
            'improved_metrics': sorted(improvements, key=lambda x: x[1], reverse=True),
            'regressed_metrics': sorted(regressions, key=lambda x: x[1])
        }
        
        return comparison
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成评测报告"""
        
        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("检索系统评测报告")
        report_lines.append("=" * 50)
        
        # 基础信息
        metrics = results.get('metrics', {})
        report_lines.append(f"评测时间: {metrics.get('timestamp', 'N/A')}")
        report_lines.append(f"查询数量: {metrics.get('num_queries', 'N/A')}")
        report_lines.append("")
        
        # 主要指标
        report_lines.append("主要评测指标:")
        report_lines.append("-" * 30)
        
        for metric, score in metrics.items():
            if metric not in ['num_queries', 'timestamp']:
                report_lines.append(f"{metric:15s}: {score:.4f}")
        
        # 融合分析
        if 'fusion_analysis' in results:
            analysis = results['fusion_analysis']
            report_lines.append("")
            report_lines.append("融合结果分析:")
            report_lines.append("-" * 30)
            
            # 检索器贡献
            contributions = analysis.get('retriever_contributions', {})
            total = analysis.get('total_results', 1)
            
            for retriever, count in contributions.items():
                rate = count / total * 100
                report_lines.append(f"{retriever:15s}: {count:4d} ({rate:5.1f}%)")
        
        return "\n".join(report_lines)


def create_evaluator(config: Dict[str, Any]) -> IRMetricsEvaluator:
    """创建评测器的工厂函数"""
    return IRMetricsEvaluator(config)