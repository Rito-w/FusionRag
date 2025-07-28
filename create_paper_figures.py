#!/usr/bin/env python3
"""
为论文创建可视化图表
基于论文中的实验数据生成高质量的图表
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd

# 设置字体和样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 设置颜色方案
colors = {
    'simple': '#2E8B57',      # 海绿色 - 简单方法
    'complex': '#CD5C5C',     # 印度红 - 复杂方法
    'rrf': '#4682B4',         # 钢蓝色 - RRF基线
    'improvement': '#32CD32'   # 酸橙绿 - 提升
}

def create_performance_comparison():
    """创建性能对比图"""
    # 数据来自表2
    datasets = ['FIQA', 'Quora', 'SciDocs', 'SciFact', 'NFCorpus', 'ArguAna']
    
    # 最佳方法的MRR值
    best_mrr = [0.343, 0.717, 0.326, 0.596, 0.583, 0.283]
    best_std = [0.015, 0.019, 0.016, 0.022, 0.025, 0.014]
    
    # RRF基线的MRR值
    rrf_mrr = [0.317, 0.669, 0.294, 0.583, 0.583, 0.283]
    rrf_std = [0.012, 0.018, 0.013, 0.016, 0.025, 0.014]
    
    # 提升幅度
    improvements = [8.2, 7.2, 10.9, 2.2, 0, 0]
    
    # 方法类型
    method_types = ['简单', '简单', '简单', '简单', '复杂', '并列']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1: MRR性能对比
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, rrf_mrr, width, yerr=rrf_std, 
                   label='RRF基线', color=colors['rrf'], alpha=0.8, capsize=5)
    bars2 = ax1.bar(x + width/2, best_mrr, width, yerr=best_std,
                   label='最佳方法', color=colors['simple'], alpha=0.8, capsize=5)
    
    ax1.set_xlabel('Dataset', fontsize=12)
    ax1.set_ylabel('MRR', fontsize=12)
    ax1.set_title('MRR Performance Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + rrf_std[i],
                f'{height1:.3f}', ha='center', va='bottom', fontsize=9)
        ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + best_std[i],
                f'{height2:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 子图2: 提升幅度
    colors_by_type = [colors['simple'] if t == '简单' else colors['complex'] if t == '复杂' else colors['rrf'] 
                     for t in method_types]
    
    bars3 = ax2.bar(datasets, improvements, color=colors_by_type, alpha=0.8)
    ax2.set_xlabel('数据集', fontsize=12)
    ax2.set_ylabel('提升幅度 (%)', fontsize=12)
    ax2.set_title('简单方法相对于RRF的性能提升', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(datasets, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, imp in zip(bars3, improvements):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'+{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['simple'], label='简单方法最佳'),
                      Patch(facecolor=colors['complex'], label='复杂方法最佳'),
                      Patch(facecolor=colors['rrf'], label='并列/无提升')]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('figures/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/performance_comparison.pdf', bbox_inches='tight')
    plt.show()

def create_ablation_study():
    """创建消融实验图"""
    datasets = ['Quora', 'SciDocs', 'FIQA']
    
    # 消融实验数据
    rrf_baseline = [0.669, 0.294, 0.317]
    no_query_analyzer = [0.671, 0.326, 0.324]
    no_adaptive_routing = [0.669, 0.310, 0.320]
    static_weights = [0.663, 0.290, 0.316]
    
    x = np.arange(len(datasets))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - 1.5*width, rrf_baseline, width, label='RRF基线', 
                  color=colors['rrf'], alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, no_query_analyzer, width, label='无查询分析', 
                  color=colors['simple'], alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, no_adaptive_routing, width, label='无自适应路由', 
                  color='#FFA500', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, static_weights, width, label='静态权重', 
                  color='#9370DB', alpha=0.8)
    
    ax.set_xlabel('数据集', fontsize=12)
    ax.set_ylabel('MRR', fontsize=12)
    ax.set_title('消融实验结果：移除复杂组件的影响', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    all_bars = [bars1, bars2, bars3, bars4]
    all_values = [rrf_baseline, no_query_analyzer, no_adaptive_routing, static_weights]
    
    for bars, values in zip(all_bars, all_values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figures/ablation_study.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/ablation_study.pdf', bbox_inches='tight')
    plt.show()

def create_baseline_comparison():
    """创建基线对比图"""
    datasets = ['FIQA', 'Quora', 'SciDocs', 'NFCorpus', 'SciFact', 'ArguAna']
    
    # 表1数据
    bm25_scores = [0.253, 0.652, 0.267, 0.589, 0.501, 0.248]
    dense_scores = [0.241, 0.631, 0.285, 0.543, 0.553, 0.231]
    rrf_scores = [0.317, 0.669, 0.294, 0.583, 0.583, 0.283]
    linear_equal_scores = [0.316, 0.663, 0.290, 0.585, 0.596, 0.280]
    linear_opt_scores = [0.343, 0.717, 0.326, 0.556, 0.521, 0.283]
    
    x = np.arange(len(datasets))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.bar(x - 2*width, bm25_scores, width, label='BM25', 
                  color='#8B4513', alpha=0.8)
    bars2 = ax.bar(x - width, dense_scores, width, label='Dense Vector', 
                  color='#4169E1', alpha=0.8)
    bars3 = ax.bar(x, rrf_scores, width, label='RRF', 
                  color=colors['rrf'], alpha=0.8)
    bars4 = ax.bar(x + width, linear_equal_scores, width, label='Linear Equal', 
                  color=colors['simple'], alpha=0.8)
    bars5 = ax.bar(x + 2*width, linear_opt_scores, width, label='Linear Optimized', 
                  color='#228B22', alpha=0.8)
    
    ax.set_xlabel('数据集', fontsize=12)
    ax.set_ylabel('MRR', fontsize=12)
    ax.set_title('基线方法性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/baseline_comparison.pdf', bbox_inches='tight')
    plt.show()

def create_query_type_distribution():
    """创建查询类型分布图"""
    datasets = ['FIQA', 'Quora', 'SciDocs', 'NFCorpus', 'SciFact', 'ArguAna']
    
    # 表5数据
    entity_ratios = [10, 0, 78, 26, 34, 78]
    keyword_ratios = [14, 0, 22, 66, 66, 22]
    semantic_ratios = [76, 100, 0, 8, 0, 0]
    
    # 最佳策略
    best_strategies = ['Linear-BM25-Dom', 'Linear-BM25-Dom', 'Linear-Vector-Dom', 
                      'RRF', 'Linear-Equal', 'RRF/Linear-BM25']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 子图1: 堆叠柱状图显示查询类型分布
    width = 0.6
    x = np.arange(len(datasets))
    
    bars1 = ax1.bar(x, entity_ratios, width, label='实体查询', 
                   color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x, keyword_ratios, width, bottom=entity_ratios, 
                   label='关键词查询', color='#4ECDC4', alpha=0.8)
    bars3 = ax1.bar(x, semantic_ratios, width, 
                   bottom=np.array(entity_ratios) + np.array(keyword_ratios),
                   label='语义查询', color='#45B7D1', alpha=0.8)
    
    ax1.set_xlabel('数据集', fontsize=12)
    ax1.set_ylabel('查询类型比例 (%)', fontsize=12)
    ax1.set_title('各数据集的查询类型分布', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 最佳策略分布
    strategy_colors = []
    for strategy in best_strategies:
        if 'Linear' in strategy:
            strategy_colors.append(colors['simple'])
        elif 'RRF' in strategy and '/' not in strategy:
            strategy_colors.append(colors['complex'])
        else:
            strategy_colors.append(colors['rrf'])
    
    bars4 = ax2.bar(datasets, [1]*len(datasets), color=strategy_colors, alpha=0.8)
    ax2.set_xlabel('数据集', fontsize=12)
    ax2.set_ylabel('最佳策略', fontsize=12)
    ax2.set_title('各数据集的最佳融合策略', fontsize=14, fontweight='bold')
    ax2.set_yticks([])
    
    # 添加策略标签
    for i, (bar, strategy) in enumerate(zip(bars4, best_strategies)):
        ax2.text(bar.get_x() + bar.get_width()/2., 0.5,
                strategy.replace('-', '\n'), ha='center', va='center', 
                fontsize=9, fontweight='bold', color='white')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['simple'], label='简单方法'),
                      Patch(facecolor=colors['complex'], label='复杂方法'),
                      Patch(facecolor=colors['rrf'], label='并列/混合')]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('figures/query_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/query_type_distribution.pdf', bbox_inches='tight')
    plt.show()

def create_summary_figure():
    """创建论文核心发现总结图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 简单vs复杂方法胜出次数
    methods = ['简单方法', '复杂方法', '并列']
    wins = [4, 1, 1]  # 基于表2统计
    colors_wins = [colors['simple'], colors['complex'], colors['rrf']]
    
    wedges, texts, autotexts = ax1.pie(wins, labels=methods, colors=colors_wins, 
                                      autopct='%1.0f次', startangle=90)
    ax1.set_title('最佳策略分布\n(6个数据集)', fontsize=12, fontweight='bold')
    
    # 2. 平均提升幅度
    datasets_with_improvement = ['FIQA', 'Quora', 'SciDocs', 'SciFact']
    improvements = [8.2, 7.2, 10.9, 2.2]
    
    bars = ax2.bar(datasets_with_improvement, improvements, 
                  color=colors['improvement'], alpha=0.8)
    ax2.set_ylabel('提升幅度 (%)', fontsize=12)
    ax2.set_title('简单方法的性能提升', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{imp:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. 计算效率对比
    methods_efficiency = ['简单方法', '复杂方法']
    latency = [1.25, 100]  # ms
    
    bars_eff = ax3.bar(methods_efficiency, latency, 
                      color=[colors['simple'], colors['complex']], alpha=0.8)
    ax3.set_ylabel('平均推理时间 (ms)', fontsize=12)
    ax3.set_title('计算效率对比', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    for bar, lat in zip(bars_eff, latency):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                f'{lat}ms', ha='center', va='bottom', fontweight='bold')
    
    # 4. 消融实验关键发现
    ablation_datasets = ['SciDocs', 'Quora', 'FIQA']
    baseline_scores = [0.294, 0.669, 0.317]
    improved_scores = [0.326, 0.671, 0.324]
    
    x_abl = np.arange(len(ablation_datasets))
    width_abl = 0.35
    
    bars_base = ax4.bar(x_abl - width_abl/2, baseline_scores, width_abl, 
                       label='RRF基线', color=colors['rrf'], alpha=0.8)
    bars_imp = ax4.bar(x_abl + width_abl/2, improved_scores, width_abl,
                      label='移除查询分析器', color=colors['simple'], alpha=0.8)
    
    ax4.set_xlabel('数据集', fontsize=12)
    ax4.set_ylabel('MRR', fontsize=12)
    ax4.set_title('消融实验：简化的效果', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_abl)
    ax4.set_xticklabels(ablation_datasets)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Less is More: 简单线性融合优于复杂方法 - 核心发现', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('figures/summary_figure.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/summary_figure.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 创建输出目录
    import os
    os.makedirs('figures', exist_ok=True)
    
    print("正在生成论文图表...")
    
    # 生成所有图表
    create_performance_comparison()
    print("✓ 性能对比图已生成")
    
    create_ablation_study()
    print("✓ 消融实验图已生成")
    
    create_baseline_comparison()
    print("✓ 基线对比图已生成")
    
    create_query_type_distribution()
    print("✓ 查询类型分布图已生成")
    
    create_summary_figure()
    print("✓ 总结图已生成")
    
    print("\n所有图表已保存到 figures/ 目录")
    print("包含 PNG (高分辨率) 和 PDF (矢量) 格式")
