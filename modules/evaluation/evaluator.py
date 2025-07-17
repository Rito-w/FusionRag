"""
评估框架
用于评估不同索引方法的性能
"""

import time
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from ..utils.interfaces import BaseRetriever, Document, Query, RetrievalResult


class IndexEvaluator:
    """索引评估器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics = self.config.get('metrics', ['precision', 'recall', 'mrr', 'ndcg', 'latency'])
        self.report_dir = self.config.get('report_dir', 'reports')
        self.top_k_values = self.config.get('top_k_values', [5, 10, 20, 50, 100])
        self.query_types = self.config.get('query_types', {})  # 查询类型映射
        
        # 创建报告目录
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)
        
        # 评估结果缓存
        self.results_cache = {}
    
    def evaluate_retriever(self, retriever: BaseRetriever, queries: List[Query], 
                          relevance_judgments: Dict[str, Dict[str, int]], 
                          top_k: int = 10) -> Dict[str, float]:
        """评估单个检索器的性能"""
        results = {}
        latencies = []
        all_metrics = []
        
        # 对每个查询进行检索
        for i, query in enumerate(queries):
            # 进度日志，每10个查询打印一次
            if i % 10 == 0 or i == len(queries) - 1:
                print(f"检索器[{retriever.name}] 进度: {i+1}/{len(queries)} 查询ID: {query.query_id}")
            # 检查缓存
            retriever_name = str(retriever.name)  # 确保是字符串
            cache_key = (retriever_name, query.query_id, top_k)
            if cache_key in self.results_cache:
                query_results = self.results_cache[cache_key]['metrics']
                latency = self.results_cache[cache_key]['latency']
            else:
                # 执行检索
                start_time = time.time()
                retrieved_docs = retriever.retrieve(query, top_k)
                end_time = time.time()
                
                # 记录延迟
                latency = (end_time - start_time) * 1000  # 毫秒
                
                # 计算评估指标
                query_results = self._calculate_metrics(query.query_id, retrieved_docs, relevance_judgments, top_k)
                
                # 缓存结果
                self.results_cache[cache_key] = {
                    'metrics': query_results,
                    'latency': latency,
                    'retrieved_docs': retrieved_docs
                }
            
            latencies.append(latency)
            all_metrics.append(query_results)
        
        # 计算平均指标
        for metric in self.metrics:
            if metric != 'latency':
                values = [m.get(metric, 0.0) for m in all_metrics]
                results[metric] = np.mean(values)
        
        # 添加延迟指标
        results['latency'] = np.mean(latencies)
        results['latency_p95'] = np.percentile(latencies, 95)
        
        return results
    
    def evaluate_multiple_retrievers(self, retrievers: Dict[str, BaseRetriever], 
                                   queries: List[Query],
                                   relevance_judgments: Dict[str, Dict[str, int]], 
                                   top_k: int = 10) -> Dict[str, Dict[str, float]]:
        """评估多个检索器的性能"""
        results = {}
        
        for name, retriever in retrievers.items():
            print(f"评估检索器: {name}")
            results[name] = self.evaluate_retriever(retriever, queries, relevance_judgments, top_k)
        
        return results
    
    def evaluate_by_query_type(self, retrievers: Dict[str, BaseRetriever], 
                             queries: List[Query],
                             relevance_judgments: Dict[str, Dict[str, int]], 
                             top_k: int = 10) -> Dict[str, Dict[str, Dict[str, float]]]:
        """按查询类型评估检索器性能"""
        # 按查询类型分组
        query_groups = defaultdict(list)
        
        for query in queries:
            query_type = self.query_types.get(query.query_id, 'unknown')
            query_groups[query_type].append(query)
        
        # 对每个查询类型进行评估
        results = {}
        for query_type, group_queries in query_groups.items():
            if group_queries:
                print(f"评估查询类型: {query_type} (查询数量: {len(group_queries)})")
                results[query_type] = self.evaluate_multiple_retrievers(
                    retrievers, group_queries, relevance_judgments, top_k
                )
        
        return results
    
    def evaluate_at_multiple_k(self, retrievers: Dict[str, BaseRetriever], 
                             queries: List[Query],
                             relevance_judgments: Dict[str, Dict[str, int]]) -> Dict[int, Dict[str, Dict[str, float]]]:
        """在多个k值上评估检索器性能"""
        results = {}
        
        for k in self.top_k_values:
            print(f"评估 top-{k} 性能")
            results[k] = self.evaluate_multiple_retrievers(retrievers, queries, relevance_judgments, k)
        
        return results
    
    def _calculate_metrics(self, query_id: str, retrieved_docs: List[RetrievalResult],
                         relevance_judgments: Dict[str, Dict[str, int]], 
                         top_k: int) -> Dict[str, float]:
        """计算评估指标"""
        results = {}
        
        # 获取相关文档
        relevant_docs = relevance_judgments.get(query_id, {})
        
        # 准确率
        if 'precision' in self.metrics:
            precision = self._calculate_precision(retrieved_docs, relevant_docs, top_k)
            results['precision'] = precision
        
        # 召回率
        if 'recall' in self.metrics:
            recall = self._calculate_recall(retrieved_docs, relevant_docs)
            results['recall'] = recall
        
        # MRR
        if 'mrr' in self.metrics:
            mrr = self._calculate_mrr(retrieved_docs, relevant_docs)
            results['mrr'] = mrr
        
        # NDCG
        if 'ndcg' in self.metrics:
            ndcg = self._calculate_ndcg(retrieved_docs, relevant_docs, top_k)
            results['ndcg'] = ndcg
        
        # F1
        if 'f1' in self.metrics and 'precision' in results and 'recall' in results:
            precision = results['precision']
            recall = results['recall']
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            results['f1'] = f1
        
        return results
    
    def _calculate_precision(self, retrieved_docs: List[RetrievalResult],
                           relevant_docs: Dict[str, int], top_k: int) -> float:
        """计算准确率"""
        if not retrieved_docs:
            return 0.0
        
        # 计算相关文档数量
        relevant_count = sum(1 for doc in retrieved_docs[:top_k] if doc.doc_id in relevant_docs)
        
        # 计算准确率
        precision = relevant_count / min(top_k, len(retrieved_docs))
        
        return precision
    
    def _calculate_recall(self, retrieved_docs: List[RetrievalResult],
                        relevant_docs: Dict[str, int]) -> float:
        """计算召回率"""
        if not relevant_docs:
            return 1.0  # 如果没有相关文档，召回率为1
        
        # 计算检索到的相关文档数量
        retrieved_relevant = sum(1 for doc in retrieved_docs if doc.doc_id in relevant_docs)
        
        # 计算召回率
        recall = retrieved_relevant / len(relevant_docs)
        
        return recall
    
    def _calculate_mrr(self, retrieved_docs: List[RetrievalResult],
                     relevant_docs: Dict[str, int]) -> float:
        """计算平均倒数排名 (MRR)"""
        if not retrieved_docs or not relevant_docs:
            return 0.0
        
        # 找到第一个相关文档的排名
        for i, doc in enumerate(retrieved_docs):
            if doc.doc_id in relevant_docs:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _calculate_ndcg(self, retrieved_docs: List[RetrievalResult],
                      relevant_docs: Dict[str, int], top_k: int) -> float:
        """计算归一化折损累积增益 (NDCG)"""
        if not retrieved_docs or not relevant_docs:
            return 0.0
        
        # 计算DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:top_k]):
            if doc.doc_id in relevant_docs:
                relevance = relevant_docs[doc.doc_id]
                dcg += (2 ** relevance - 1) / np.log2(i + 2)
        
        # 计算理想DCG
        ideal_relevances = sorted(relevant_docs.values(), reverse=True)[:top_k]
        idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        # 计算NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return ndcg
    
    def generate_report(self, results: Dict[str, Dict[str, float]], 
                       dataset_name: str = "unknown",
                       output_file: str = None) -> None:
        """生成评估报告"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.report_dir}/evaluation_{dataset_name}_{timestamp}.json"
        
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
        with open(text_report_file, 'w') as f:
            f.write(f"# 检索器评估报告 - {dataset_name}\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 写入总体结果
            f.write("## 总体结果\n\n")
            f.write("| 检索器 | 准确率 | 召回率 | MRR | NDCG | 延迟(ms) |\n")
            f.write("|--------|--------|--------|-----|------|----------|\n")
            
            for name, metrics in results.items():
                precision = metrics.get('precision', 0.0)
                recall = metrics.get('recall', 0.0)
                mrr = metrics.get('mrr', 0.0)
                ndcg = metrics.get('ndcg', 0.0)
                latency = metrics.get('latency', 0.0)
                
                f.write(f"| {name} | {precision:.4f} | {recall:.4f} | {mrr:.4f} | {ndcg:.4f} | {latency:.2f} |\n")
            
            f.write("\n")
        
        print(f"评估报告已保存到: {output_file} 和 {text_report_file}")
        
        return report_data
    
    def run_ablation_study(self, base_retriever: BaseRetriever, 
                         component_retrievers: Dict[str, BaseRetriever],
                         queries: List[Query],
                         relevance_judgments: Dict[str, Dict[str, int]], 
                         top_k: int = 10) -> Dict[str, Dict[str, float]]:
        """运行消融实验"""
        results = {}
        
        # 评估基础检索器
        print(f"评估基础检索器: {base_retriever.name}")
        results["base"] = self.evaluate_retriever(base_retriever, queries, relevance_judgments, top_k)
        
        # 评估各组件检索器
        for name, retriever in component_retrievers.items():
            print(f"评估组件: {name}")
            results[name] = self.evaluate_retriever(retriever, queries, relevance_judgments, top_k)
        
        return results


def create_evaluator(config: Dict[str, Any] = None) -> IndexEvaluator:
    """创建评估器的工厂函数"""
    return IndexEvaluator(config=config)