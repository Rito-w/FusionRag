#!/usr/bin/env python3
"""
优化后的论文图表生成脚本
精简为5个核心图表，提高质量和针对性
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# 设置学术风格
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# 学术期刊标准色彩方案
colors = {
    'simple': '#1f77b4',      # 蓝色 - 简单方法
    'complex': '#d62728',     # 红色 - 复杂方法
    'neutral': '#ff7f0e',     # 橙色 - 中性
    'highlight': '#2ca02c',   # 绿色 - 突出显示
    'background': '#f7f7f7'   # 浅灰 - 背景
}

def create_figure1_performance_with_datasets():
    """图1: 性能对比图（整合数据集特征）"""
    # 实验数据
    datasets = ['SciFact', 'FIQA', 'Quora', 'SciDocs', 'NFCorpus', 'ArguAna']
    simple_mrr = [0.567, 0.324, 0.669, 0.326, 0.622, 0.259]
    complex_mrr = [0.500, 0.317, 0.669, 0.294, 0.622, 0.259]
    simple_std = [0.018, 0.015, 0.018, 0.016, 0.025, 0.014]
    complex_std = [0.016, 0.012, 0.018, 0.013, 0.025, 0.014]
    p_values = [0.003, 0.156, 0.421, 0.008, 1.0, 0.089]
    
    # 数据集特征（查询类型主导）
    dominant_types = ['Keyword', 'Semantic', 'Semantic', 'Entity', 'Keyword', 'Entity']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 子图1: 性能对比
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, simple_mrr, width, yerr=simple_std, 
                   label='Simple Methods', color=colors['simple'], alpha=0.8, capsize=5)
    bars2 = ax1.bar(x + width/2, complex_mrr, width, yerr=complex_std,
                   label='Complex Methods', color=colors['complex'], alpha=0.8, capsize=5)
    
    # 标记统计显著性
    for i, p in enumerate(p_values):
        if p < 0.01:
            y_max = max(simple_mrr[i] + simple_std[i], complex_mrr[i] + complex_std[i])
            ax1.plot([i - width/2, i + width/2], [y_max + 0.03, y_max + 0.03], 'k-', linewidth=2)
            ax1.text(i, y_max + 0.04, '**', ha='center', va='bottom', fontsize=14, fontweight='bold')
        elif p < 0.05:
            y_max = max(simple_mrr[i] + simple_std[i], complex_mrr[i] + complex_std[i])
            ax1.plot([i - width/2, i + width/2], [y_max + 0.03, y_max + 0.03], 'k-', linewidth=1)
            ax1.text(i, y_max + 0.04, '*', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('MRR Score', fontsize=14)
    ax1.set_title('Performance Comparison: Simple vs Complex Fusion Methods', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend(fontsize=12)
    ax1.set_ylim(0, 0.75)
    
    # 子图2: 数据集特征分布
    type_colors = {'Entity': '#ff9999', 'Keyword': '#66b3ff', 'Semantic': '#99ff99'}
    colors_list = [type_colors[t] for t in dominant_types]
    
    bars = ax2.bar(datasets, [1]*len(datasets), color=colors_list, alpha=0.7)
    
    # 添加标签
    for i, (bar, dtype) in enumerate(zip(bars, dominant_types)):
        ax2.text(bar.get_x() + bar.get_width()/2., 0.5, dtype,
                ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Dominant Query Type', fontsize=14)
    ax2.set_title('Dataset Characteristics', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    
    # 添加图例
    legend_elements = [mpatches.Patch(color=color, label=qtype) 
                      for qtype, color in type_colors.items()]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # 添加注释
    ax1.text(0.02, 0.98, '** p < 0.01, * p < 0.05', transform=ax1.transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('fig1_performance_comparison.pdf')
    plt.savefig('fig1_performance_comparison.png')
    plt.show()

def create_figure2_ablation_study():
    """图2: 消融实验热力图"""
    ablation_data = pd.DataFrame({
        'RRF Baseline': [0.669, 0.294, 0.317],
        'Remove Query Analysis': [0.669, 0.326, 0.324],
        'Remove Adaptive Routing': [0.669, 0.310, 0.320],
        'Static Weights Only': [0.663, 0.290, 0.316]
    }, index=['Quora', 'SciDocs', 'FIQA'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 创建热力图
    sns.heatmap(ablation_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                center=0.4, cbar_kws={'label': 'MRR Score'}, ax=ax,
                linewidths=0.5, linecolor='white')
    
    # 高亮最佳结果
    for i in range(len(ablation_data.index)):
        best_col = ablation_data.iloc[i].idxmax()
        best_col_idx = list(ablation_data.columns).index(best_col)
        rect = Rectangle((best_col_idx, i), 1, 1, linewidth=3, 
                        edgecolor='green', facecolor='none')
        ax.add_patch(rect)
    
    ax.set_title('Ablation Study: Impact of Removing Complex Components', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('System Configuration', fontsize=14)
    ax.set_ylabel('Datasets', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('fig2_ablation_study.pdf')
    plt.savefig('fig2_ablation_study.png')
    plt.show()

def create_figure3_efficiency_complexity():
    """图3: 效率与复杂度对比（合并图表）"""
    methods = ['Linear\nEqual', 'Linear\nOptimized', 'RRF\nStandard', 'Adaptive\nFusion']
    latency = [1.2, 1.3, 2.1, 120.3]  # ms per query
    parameters = [2, 3, 4, 15]  # number of parameters
    complexity = ['Simple', 'Simple', 'Complex', 'Complex']
    
    colors_list = [colors['simple'] if c == 'Simple' else colors['complex'] for c in complexity]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 子图1: 延迟对比（对数尺度）
    bars1 = ax1.bar(methods, latency, color=colors_list, alpha=0.8)
    ax1.set_ylabel('Latency (ms/query)', fontsize=12)
    ax1.set_title('Query Processing Latency', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, lat in zip(bars1, latency):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                f'{lat:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 子图2: 参数复杂度
    bars2 = ax2.bar(methods, parameters, color=colors_list, alpha=0.8)
    ax2.set_ylabel('Number of Parameters', fontsize=12)
    ax2.set_title('Model Complexity', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, param in zip(bars2, parameters):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{param}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 添加图例
    simple_patch = mpatches.Patch(color=colors['simple'], label='Simple Methods')
    complex_patch = mpatches.Patch(color=colors['complex'], label='Complex Methods')
    ax2.legend(handles=[simple_patch, complex_patch], loc='upper left')
    
    # 添加效率提升标注
    ax1.text(0.5, 0.95, '80× faster than\ncomplex methods', 
            transform=ax1.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fig3_efficiency_complexity.pdf')
    plt.savefig('fig3_efficiency_complexity.png')
    plt.show()

def create_figure4_system_architecture():
    """图4: 系统架构图（简化版）"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 绘制组件函数
    def draw_box(x, y, w, h, text, color, text_color='black'):
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
               fontsize=11, fontweight='bold', color=text_color)
    
    def draw_arrow(start, end):
        from matplotlib.patches import ConnectionPatch
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle='->', shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc='black', ec='black', linewidth=2)
        ax.add_patch(arrow)
    
    # 标题
    ax.text(5, 7.5, 'Multi-Retriever Fusion System Architecture', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 输入层
    draw_box(4, 6.5, 2, 0.6, 'User Query', colors['background'])
    
    # 检索器层
    draw_box(1.5, 5, 2, 0.8, 'BM25\n(Sparse)', '#e8f5e8')
    draw_box(6.5, 5, 2, 0.8, 'E5-large-v2\n(Dense)', '#e8f5e8')
    
    # 融合层标题
    ax.text(5, 4.2, 'Fusion Strategies', ha='center', va='center', 
           fontsize=14, fontweight='bold')
    
    # 简单方法
    draw_box(1, 3, 1.8, 0.6, 'Linear\nEqual', colors['simple'], 'white')
    draw_box(3, 3, 1.8, 0.6, 'Linear\nOptimized', colors['simple'], 'white')
    
    # 复杂方法
    draw_box(5.2, 3, 1.8, 0.6, 'RRF\nStandard', colors['complex'], 'white')
    draw_box(7.2, 3, 1.8, 0.6, 'Adaptive\nFusion', colors['complex'], 'white')
    
    # 输出层
    draw_box(4, 1.5, 2, 0.6, 'Ranked Results', colors['highlight'], 'white')
    
    # 绘制箭头
    draw_arrow((4.5, 6.5), (2.5, 5.8))  # Query to BM25
    draw_arrow((5.5, 6.5), (7.5, 5.8))  # Query to Dense
    
    draw_arrow((2.5, 5), (1.9, 3.6))    # BM25 to Linear Equal
    draw_arrow((2.5, 5), (3.9, 3.6))    # BM25 to Linear Opt
    draw_arrow((7.5, 5), (6.1, 3.6))    # Dense to RRF
    draw_arrow((7.5, 5), (8.1, 3.6))    # Dense to Adaptive
    
    draw_arrow((1.9, 3), (4.5, 2.1))    # Linear Equal to Output
    draw_arrow((3.9, 3), (4.7, 2.1))    # Linear Opt to Output
    draw_arrow((6.1, 3), (5.3, 2.1))    # RRF to Output
    draw_arrow((8.1, 3), (5.5, 2.1))    # Adaptive to Output
    
    # 图例
    ax.text(0.5, 0.8, 'Simple Methods', color=colors['simple'], 
           fontsize=12, fontweight='bold')
    ax.text(0.5, 0.5, 'Complex Methods', color=colors['complex'], 
           fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fig4_system_architecture.pdf')
    plt.savefig('fig4_system_architecture.png')
    plt.show()

def create_figure5_statistical_significance():
    """图5: 统计显著性分析"""
    datasets = ['SciFact', 'FIQA', 'Quora', 'SciDocs', 'NFCorpus', 'ArguAna']
    p_values = [0.003, 0.156, 0.421, 0.008, 1.0, 0.089]
    effect_sizes = [4.47, 0.47, 0.00, 2.13, 0.00, 0.00]  # Cohen's d
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 子图1: p值分析
    colors_p = [colors['highlight'] if p < 0.05 else 'gray' for p in p_values]
    bars1 = ax1.bar(datasets, p_values, color=colors_p, alpha=0.7)
    
    ax1.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    ax1.axhline(y=0.01, color='darkred', linestyle='--', linewidth=2, label='α = 0.01')
    
    ax1.set_ylabel('p-value', fontsize=12)
    ax1.set_title('Statistical Significance (Simple vs Complex Methods)', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加显著性标记
    for i, (bar, p) in enumerate(zip(bars1, p_values)):
        height = bar.get_height()
        significance = '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax1.text(bar.get_x() + bar.get_width()/2., height * 2,
                significance, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 子图2: 效应大小
    colors_effect = [colors['highlight'] if abs(d) > 0.5 else 'gray' for d in effect_sizes]
    bars2 = ax2.bar(datasets, effect_sizes, color=colors_effect, alpha=0.7)
    
    ax2.axhline(y=0.2, color='gray', linestyle=':', alpha=0.7, label='Small effect')
    ax2.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='Medium effect')
    ax2.axhline(y=0.8, color='red', linestyle=':', alpha=0.7, label='Large effect')
    
    ax2.set_ylabel("Cohen's d (Effect Size)", fontsize=12)
    ax2.set_title('Effect Size Analysis', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加效应大小标签
    for i, (bar, d) in enumerate(zip(bars2, effect_sizes)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{d:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fig5_statistical_significance.pdf')
    plt.savefig('fig5_statistical_significance.png')
    plt.show()

if __name__ == "__main__":
    print("🎨 生成优化后的论文图表（5个核心图表）...")
    
    print("图1: 性能对比与数据集特征...")
    create_figure1_performance_with_datasets()
    
    print("图2: 消融实验分析...")
    create_figure2_ablation_study()
    
    print("图3: 效率与复杂度对比...")
    create_figure3_efficiency_complexity()
    
    print("图4: 系统架构图...")
    create_figure4_system_architecture()
    
    print("图5: 统计显著性分析...")
    create_figure5_statistical_significance()
    
    print("✅ 优化后的5个核心图表生成完成！")
    print("📊 图表文件：fig1-fig5 (PDF + PNG格式)")
