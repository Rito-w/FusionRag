import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图表目录
import os
if not os.path.exists('charts'):
    os.makedirs('charts')

# ==================== 图表1: 基线对比实验 ====================
# 论文中的表1数据
baseline_data = {
    'Dataset': ['FIQA', 'Quora', 'SciDocs', 'NFCorpus', 'SciFact', 'ArguAna'],
    'BM25': [0.253, 0.652, 0.267, 0.589, 0.501, 0.248],
    'Dense': [0.241, 0.631, 0.285, 0.543, 0.553, 0.231],
    'RRF': [0.317, 0.669, 0.294, 0.622, 0.500, 0.259],
    'LinearEqual': [0.316, 0.663, 0.290, 0.585, 0.627, 0.265],
    'LinearOptimized': [0.324, 0.671, 0.326, 0.598, 0.615, 0.261]
}

df_baseline = pd.DataFrame(baseline_data)

# 绘制基线对比图
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(df_baseline['Dataset']))
width = 0.15

bars1 = ax.bar(x - 2*width, df_baseline['BM25'], width, label='BM25', color='#1f77b4')
bars2 = ax.bar(x - width, df_baseline['Dense'], width, label='Dense', color='#ff7f0e')
bars3 = ax.bar(x, df_baseline['RRF'], width, label='RRF', color='#2ca02c')
bars4 = ax.bar(x + width, df_baseline['LinearEqual'], width, label='LinearEqual', color='#d62728')
bars5 = ax.bar(x + 2*width, df_baseline['LinearOptimized'], width, label='LinearOptimized', color='#9467bd')

ax.set_xlabel('数据集')
ax.set_ylabel('MRR性能')
ax.set_title('基线方法性能对比')
ax.set_xticks(x)
ax.set_xticklabels(df_baseline['Dataset'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 在柱状图上添加数值标签
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)
add_value_labels(bars4)
add_value_labels(bars5)

plt.tight_layout()
plt.savefig('charts/baseline_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== 图表2: 融合策略对比 ====================
# 论文中的表2数据 (基于论文第579-584行的数据)
fusion_data = {
    'Dataset': ['SciFact', 'FIQA', 'Quora', 'SciDocs', 'NFCorpus', 'ArguAna'],
    'Best_Strategy': ['Linear Equal', 'Linear Optimized', 'RRF Standard', 'Linear Optimized', 'RRF Standard', 'RRF Standard'],
    'MRR': [0.567, 0.324, 0.669, 0.326, 0.622, 0.259],
    'vs_RRF': [0.500, 0.317, 0.669, 0.294, 0.622, 0.259],
    'Improvement': [13.4, 2.2, 0, 10.9, 0, 0]
}

df_fusion = pd.DataFrame(fusion_data)

# 绘制融合策略对比图
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(df_fusion['Dataset']))
width = 0.35

bars1 = ax.bar(x - width/2, df_fusion['MRR'], width, label='最佳策略MRR', color='#1f77b4')
bars2 = ax.bar(x + width/2, df_fusion['vs_RRF'], width, label='RRF方法MRR', color='#ff7f0e')

ax.set_xlabel('数据集')
ax.set_ylabel('MRR性能')
ax.set_title('融合策略性能对比')
ax.set_xticks(x)
ax.set_xticklabels(df_fusion['Dataset'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 添加提升百分比标签
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    improvement = df_fusion['Improvement'][i]
    if improvement > 0:
        ax.annotate(f'+{improvement:.1f}%', 
                    xy=(bar1.get_x() + bar1.get_width()/2, height1),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('charts/fusion_strategy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== 图表3: 计算效率对比 ====================
# 论文中的效率数据 (基于论文第607行的数据)
efficiency_data = {
    'Method': ['Linear Equal', 'Linear BM25-Dom', 'Linear Vector-Dom', 'RRF Standard', 'Adaptive by Query Type', 'DAT'],
    'Time_ms': [1.2, 1.3, 1.3, 5.7, 100.0, 100.0],
    'Relative_Speed': [1.0, 0.9, 0.9, 0.2, 0.01, 0.01]
}

df_efficiency = pd.DataFrame(efficiency_data)

# 绘制计算效率对比图
fig, ax = plt.subplots(figsize=(12, 8))

bars = ax.bar(df_efficiency['Method'], df_efficiency['Time_ms'], 
              color=['#1f77b4', '#1f77b4', '#1f77b4', '#ff7f0e', '#d62728', '#d62728'])

ax.set_xlabel('方法')
ax.set_ylabel('平均推理时间 (ms/查询)')
ax.set_title('计算效率对比 (越低越好)')
ax.grid(axis='y', alpha=0.3)

# 设置对数刻度以便更好地显示差异
ax.set_yscale('log')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}ms',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('charts/computational_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== 图表4: 数据集特征分析 ====================
# 论文中的表5数据 (基于论文第614-622行的数据)
dataset_features_data = {
    'Dataset': ['FIQA', 'Quora', 'SciDocs', 'NFCorpus', 'SciFact', 'ArguAna'],
    'Entity_Query': [15, 5, 75, 32, 35, 78],
    'Keyword_Query': [45, 25, 23, 59, 65, 13],
    'Semantic_Query': [40, 70, 2, 9, 0, 9],
    'Best_Strategy': ['Linear-Opt', 'RRF', 'Linear-Opt', 'RRF', 'Linear-Equal', 'RRF']
}

df_dataset = pd.DataFrame(dataset_features_data)

# 绘制数据集特征分析图
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(df_dataset['Dataset']))
width = 0.25

bars1 = ax.bar(x - width, df_dataset['Entity_Query'], width, label='实体查询(%)', color='#1f77b4')
bars2 = ax.bar(x, df_dataset['Keyword_Query'], width, label='关键词查询(%)', color='#ff7f0e')
bars3 = ax.bar(x + width, df_dataset['Semantic_Query'], width, label='语义查询(%)', color='#2ca02c')

ax.set_xlabel('数据集')
ax.set_ylabel('查询类型分布 (%)')
ax.set_title('数据集查询类型分布')
ax.set_xticks(x)
ax.set_xticklabels(df_dataset['Dataset'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 添加最佳策略标签
for i, (bar, strategy) in enumerate(zip(bars2, df_dataset['Best_Strategy'])):
    height = bar.get_height()
    ax.annotate(f'最佳: {strategy}', 
                xy=(bar.get_x() + bar.get_width()/2, 105),
                xytext=(0, 0),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

plt.tight_layout()
plt.savefig('charts/dataset_features_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("所有图表已生成并保存到 'charts' 目录中:")
print("1. baseline_comparison.png - 基线方法性能对比")
print("2. fusion_strategy_comparison.png - 融合策略性能对比")
print("3. computational_efficiency.png - 计算效率对比")
print("4. dataset_features_analysis.png - 数据集特征分析")