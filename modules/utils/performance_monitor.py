#!/usr/bin/env python
"""
FusionRAG 性能监控框架
实时监控和记录系统性能，为参数优化提供反馈
"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import psutil
import os

@dataclass
class PerformanceMetrics:
    """性能指标数据结构"""
    timestamp: str
    dataset_name: str
    retriever_name: str
    query_id: str
    
    # 时间指标
    response_time: float  # 响应时间(ms)
    index_build_time: float  # 索引构建时间(s)
    
    # 质量指标
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float
    map_score: float
    
    # 系统资源指标
    memory_usage: float  # MB
    cpu_usage: float     # %
    
    # 检索器特定指标
    results_count: int
    avg_score: float
    score_variance: float
    
    # 配置参数
    parameters: Dict[str, Any]

@dataclass
class SystemStatus:
    """系统状态"""
    timestamp: str
    total_queries_processed: int
    avg_response_time: float
    current_memory_usage: float
    current_cpu_usage: float
    active_retrievers: List[str]
    error_count: int
    warning_count: int

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, log_dir: str = "logs/performance", buffer_size: int = 1000):
        self.log_dir = log_dir
        self.buffer_size = buffer_size
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 性能数据缓冲区
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.system_status_buffer = deque(maxlen=100)
        
        # 统计数据
        self.dataset_stats = defaultdict(list)
        self.retriever_stats = defaultdict(list)
        self.performance_history = defaultdict(list)
        
        # 线程安全锁
        self.lock = threading.Lock()
        
        # 日志配置
        self._setup_logging()
        
        # 系统资源监控
        self.start_time = time.time()
        self.query_count = 0
        self.error_count = 0
        self.warning_count = 0
        
        print("📊 性能监控框架初始化完成")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = os.path.join(self.log_dir, f"performance_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def start_query_monitoring(self, query_id: str, dataset_name: str, retriever_name: str) -> str:
        """开始查询监控"""
        monitor_id = f"{dataset_name}_{retriever_name}_{query_id}_{int(time.time())}"
        start_time = time.time()
        
        with self.lock:
            self.query_count += 1
        
        self.logger.info(f"🔍 开始监控查询: {monitor_id}")
        return monitor_id
    
    def end_query_monitoring(self, monitor_id: str, results: List, parameters: Dict[str, Any] = None) -> PerformanceMetrics:
        """结束查询监控并记录指标"""
        parts = monitor_id.split('_')
        
        # 最后一个部分是时间戳
        start_timestamp = int(parts[-1])
        
        # 倒数第二个部分是query_id
        query_id = parts[-2]
        
        # 第一个部分是dataset_name
        dataset_name = parts[0]
        
        # 中间部分组成retriever_name
        retriever_name = '_'.join(parts[1:-2])
        
        end_time = time.time()
        response_time = (end_time - start_timestamp) * 1000  # 转换为毫秒
        
        # 计算检索质量指标
        results_count = len(results)
        avg_score = sum(r.score for r in results) / results_count if results else 0
        score_variance = sum((r.score - avg_score) ** 2 for r in results) / results_count if results else 0
        
        # 获取系统资源使用情况
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        cpu_usage = psutil.Process().cpu_percent()
        
        # 创建性能指标对象
        metrics = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            dataset_name=dataset_name,
            retriever_name=retriever_name,
            query_id=query_id,
            response_time=response_time,
            index_build_time=0.0,  # 需要外部设置
            recall_at_5=0.0,       # 需要外部计算
            recall_at_10=0.0,      # 需要外部计算
            ndcg_at_10=0.0,        # 需要外部计算
            map_score=0.0,         # 需要外部计算
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            results_count=results_count,
            avg_score=avg_score,
            score_variance=score_variance,
            parameters=parameters or {}
        )
        
        # 添加到缓冲区
        with self.lock:
            self.metrics_buffer.append(metrics)
            self.dataset_stats[dataset_name].append(metrics)
            self.retriever_stats[retriever_name].append(metrics)
        
        self.logger.info(f"✅ 查询监控完成: {monitor_id}, 响应时间: {response_time:.2f}ms")
        return metrics
    
    def update_evaluation_metrics(self, monitor_id: str, recall_5: float, recall_10: float, 
                                  ndcg_10: float, map_score: float):
        """更新评估指标"""
        # 查找对应的性能指标
        with self.lock:
            for metrics in reversed(self.metrics_buffer):
                expected_id = f"{metrics.dataset_name}_{metrics.retriever_name}_{metrics.query_id}"
                if monitor_id.startswith(expected_id):
                    metrics.recall_at_5 = recall_5
                    metrics.recall_at_10 = recall_10
                    metrics.ndcg_at_10 = ndcg_10
                    metrics.map_score = map_score
                    break
    
    def record_index_build_time(self, dataset_name: str, retriever_name: str, build_time: float):
        """记录索引构建时间"""
        self.logger.info(f"🔨 索引构建完成: {retriever_name}@{dataset_name}, 时间: {build_time:.2f}s")
        
        # 查找最近的相关指标并更新
        with self.lock:
            for metrics in reversed(self.metrics_buffer):
                if (metrics.dataset_name == dataset_name and 
                    metrics.retriever_name == retriever_name and 
                    metrics.index_build_time == 0.0):
                    metrics.index_build_time = build_time
                    break
    
    def record_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """记录错误"""
        with self.lock:
            self.error_count += 1
        
        self.logger.error(f"❌ 错误: {error_type} - {error_message}")
        if context:
            self.logger.error(f"   上下文: {context}")
    
    def record_warning(self, warning_message: str, context: Dict[str, Any] = None):
        """记录警告"""
        with self.lock:
            self.warning_count += 1
        
        self.logger.warning(f"⚠️ 警告: {warning_message}")
        if context:
            self.logger.warning(f"   上下文: {context}")
    
    def get_current_system_status(self) -> SystemStatus:
        """获取当前系统状态"""
        uptime = time.time() - self.start_time
        
        with self.lock:
            # 计算平均响应时间
            recent_metrics = list(self.metrics_buffer)[-50:]  # 最近50个查询
            avg_response_time = (sum(m.response_time for m in recent_metrics) / 
                               len(recent_metrics)) if recent_metrics else 0
            
            # 获取活跃的检索器
            active_retrievers = list(set(m.retriever_name for m in recent_metrics))
        
        # 当前系统资源
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        current_cpu = psutil.Process().cpu_percent()
        
        status = SystemStatus(
            timestamp=datetime.now().isoformat(),
            total_queries_processed=self.query_count,
            avg_response_time=avg_response_time,
            current_memory_usage=current_memory,
            current_cpu_usage=current_cpu,
            active_retrievers=active_retrievers,
            error_count=self.error_count,
            warning_count=self.warning_count
        )
        
        with self.lock:
            self.system_status_buffer.append(status)
        
        return status
    
    def get_performance_summary(self, dataset_name: str = None, retriever_name: str = None) -> Dict[str, Any]:
        """获取性能摘要"""
        with self.lock:
            metrics = list(self.metrics_buffer)
        
        # 过滤数据
        if dataset_name:
            metrics = [m for m in metrics if m.dataset_name == dataset_name]
        if retriever_name:
            metrics = [m for m in metrics if m.retriever_name == retriever_name]
        
        if not metrics:
            return {}
        
        # 计算统计数据
        response_times = [m.response_time for m in metrics]
        recall_5_scores = [m.recall_at_5 for m in metrics if m.recall_at_5 > 0]
        recall_10_scores = [m.recall_at_10 for m in metrics if m.recall_at_10 > 0]
        ndcg_scores = [m.ndcg_at_10 for m in metrics if m.ndcg_at_10 > 0]
        
        summary = {
            'total_queries': len(metrics),
            'avg_response_time': sum(response_times) / len(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'avg_results_count': sum(m.results_count for m in metrics) / len(metrics),
        }
        
        if recall_5_scores:
            summary['avg_recall_at_5'] = sum(recall_5_scores) / len(recall_5_scores)
        if recall_10_scores:
            summary['avg_recall_at_10'] = sum(recall_10_scores) / len(recall_10_scores)
        if ndcg_scores:
            summary['avg_ndcg_at_10'] = sum(ndcg_scores) / len(ndcg_scores)
        
        return summary
    
    def export_metrics(self, filename: str = None) -> str:
        """导出性能指标到JSON文件"""
        if not filename:
            filename = f"{self.log_dir}/metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with self.lock:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_queries': self.query_count,
                'system_uptime': time.time() - self.start_time,
                'metrics': [asdict(m) for m in self.metrics_buffer],
                'system_status_history': [asdict(s) for s in self.system_status_buffer],
                'dataset_summaries': {
                    dataset: self.get_performance_summary(dataset_name=dataset)
                    for dataset in self.dataset_stats.keys()
                },
                'retriever_summaries': {
                    retriever: self.get_performance_summary(retriever_name=retriever)
                    for retriever in self.retriever_stats.keys()
                }
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📊 性能指标已导出: {filename}")
        return filename
    
    def generate_performance_report(self) -> str:
        """生成性能报告"""
        status = self.get_current_system_status()
        uptime_hours = (time.time() - self.start_time) / 3600
        
        report = f"""
# FusionRAG 性能监控报告

## 📊 系统概览
- **监控时间**: {status.timestamp}
- **系统运行时间**: {uptime_hours:.2f} 小时
- **总查询数**: {status.total_queries_processed}
- **平均响应时间**: {status.avg_response_time:.2f} ms
- **错误数**: {status.error_count}
- **警告数**: {status.warning_count}

## 💻 系统资源
- **当前内存使用**: {status.current_memory_usage:.1f} MB
- **当前CPU使用**: {status.current_cpu_usage:.1f}%

## 🔍 活跃检索器
{', '.join(status.active_retrievers)}

## 📈 数据集性能统计
"""
        
        for dataset_name in self.dataset_stats.keys():
            summary = self.get_performance_summary(dataset_name=dataset_name)
            if summary:
                report += f"""
### {dataset_name}
- 查询数: {summary['total_queries']}
- 平均响应时间: {summary['avg_response_time']:.2f} ms
- 平均结果数: {summary['avg_results_count']:.1f}
"""
                if 'avg_recall_at_10' in summary:
                    report += f"- 平均Recall@10: {summary['avg_recall_at_10']:.4f}\n"
        
        report += f"""
## 🔧 检索器性能对比
"""
        
        for retriever_name in self.retriever_stats.keys():
            summary = self.get_performance_summary(retriever_name=retriever_name)
            if summary:
                report += f"""
### {retriever_name}
- 查询数: {summary['total_queries']}
- 平均响应时间: {summary['avg_response_time']:.2f} ms
- 响应时间范围: {summary['min_response_time']:.2f} - {summary['max_response_time']:.2f} ms
"""
        
        return report
    
    def clear_metrics(self):
        """清空性能指标"""
        with self.lock:
            self.metrics_buffer.clear()
            self.system_status_buffer.clear()
            self.dataset_stats.clear()
            self.retriever_stats.clear()
            self.query_count = 0
            self.error_count = 0
            self.warning_count = 0
            self.start_time = time.time()
        
        self.logger.info("🧹 性能指标已清空")

# 全局监控器实例
_global_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def test_performance_monitor():
    """测试性能监控框架"""
    print("🧪 测试性能监控框架")
    print("=" * 50)
    
    # 获取监控器
    monitor = get_performance_monitor()
    
    # 模拟查询监控
    from modules.utils.interfaces import RetrievalResult, Document
    
    for i in range(5):
        # 开始监控
        monitor_id = monitor.start_query_monitoring(f"test_query_{i}", "test_dataset", "test_retriever")
        
        # 模拟检索过程
        time.sleep(0.1)
        
        # 模拟结果
        results = [
            RetrievalResult(
                doc_id=f"doc_{j}",
                score=0.8 - j*0.1,
                document=Document(doc_id=f"doc_{j}", title="Test", text="Test"),
                retriever_name="test_retriever"
            )
            for j in range(3)
        ]
        
        # 结束监控
        metrics = monitor.end_query_monitoring(monitor_id, results, {"param1": "value1"})
        
        # 更新评估指标
        monitor.update_evaluation_metrics(monitor_id, 0.2, 0.3, 0.15, 0.1)
    
    # 记录索引构建时间
    monitor.record_index_build_time("test_dataset", "test_retriever", 5.2)
    
    # 获取系统状态
    status = monitor.get_current_system_status()
    print(f"系统状态: {status.total_queries_processed} 查询, {status.avg_response_time:.2f}ms 平均响应")
    
    # 获取性能摘要
    summary = monitor.get_performance_summary()
    print(f"性能摘要: {summary}")
    
    # 生成报告
    report = monitor.generate_performance_report()
    print("性能报告:")
    print(report[:500] + "...")
    
    # 导出指标
    export_file = monitor.export_metrics()
    print(f"指标已导出: {export_file}")
    
    print("\n🎉 性能监控测试完成!")

if __name__ == "__main__":
    test_performance_monitor()