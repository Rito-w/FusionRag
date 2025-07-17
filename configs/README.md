# FusionRAG 配置文件管理

本目录包含FusionRAG系统的所有配置文件，采用统一的命名规范便于管理和追踪实验。

## 📋 命名规范

### 格式
```
YYYYMMDD_HHMM_<dataset>_<template>.yaml
```

### 组成部分
- **YYYYMMDD_HHMM**: 创建时间戳
- **dataset**: 数据集名称 (如 nfcorpus, trec-covid, natural-questions)
- **template**: 配置模板类型

### 模板类型
- **baseline**: 基线配置，使用默认参数
- **high_performance**: 高性能配置，使用最强模型和优化参数
- **experimental**: 实验配置，测试新的策略或参数

## 📁 当前配置文件

### NFCorpus 数据集
- `20250713_1506_nfcorpus_baseline.yaml` - 基线配置
- `20250713_1506_nfcorpus_high_performance.yaml` - 高性能配置

### TREC-COVID 数据集
- `20250713_1507_trec-covid_experimental.yaml` - RRF融合实验配置

### 历史配置 (待重命名)
- `config.yaml` - 原始默认配置
- `optimized_config.yaml` - 原始优化配置

## 🛠️ 配置管理工具

使用 `scripts/config_manager.py` 来管理配置文件：

### 创建新配置
```bash
# 创建基线配置
python scripts/config_manager.py create nfcorpus --template baseline --description "NFCorpus基线测试"

# 创建高性能配置
python scripts/config_manager.py create trec-covid --template high_performance --description "TREC-COVID高性能测试"

# 创建实验配置
python scripts/config_manager.py create natural-questions --template experimental --description "NQ数据集RRF实验"
```

### 列出所有配置
```bash
python scripts/config_manager.py list
```

### 比较配置文件
```bash
python scripts/config_manager.py compare config1.yaml config2.yaml
```

## 📊 配置模板说明

### Baseline 模板
- **模型**: all-MiniLM-L6-v2 (384维)
- **检索器**: BM25 + Dense
- **融合**: 加权融合 (BM25:0.6, Dense:0.4)
- **分类器**: 关闭
- **图检索**: 关闭
- **用途**: 快速测试和基准对比

### High Performance 模板
- **模型**: all-mpnet-base-v2 (768维)
- **检索器**: BM25 + Dense + Graph
- **融合**: 加权融合 (BM25:0.5, Dense:0.4, Graph:0.1)
- **分类器**: 启用智能路由
- **图检索**: 启用Neo4j
- **用途**: 追求最佳性能的生产配置

### Experimental 模板
- **模型**: all-mpnet-base-v2 (768维)
- **检索器**: BM25 + Dense + Graph
- **融合**: RRF融合 (实验性)
- **分类器**: 启用
- **图检索**: 启用
- **用途**: 测试新策略和算法

## 🎯 使用建议

### 1. 新数据集测试流程
```bash
# 1. 先用基线配置快速测试
python scripts/config_manager.py create <dataset> --template baseline

# 2. 运行基线测试
python tests/high_performance_test.py --config configs/YYYYMMDD_HHMM_<dataset>_baseline.yaml

# 3. 如果基线正常，使用高性能配置
python scripts/config_manager.py create <dataset> --template high_performance

# 4. 运行高性能测试
python tests/high_performance_test.py --config configs/YYYYMMDD_HHMM_<dataset>_high_performance.yaml
```

### 2. 实验管理
- 每个实验使用独立的配置文件
- 在metadata中记录实验目的和预期改进
- 保留所有历史配置便于复现结果

### 3. 配置选择指南
- **开发调试**: 使用 baseline 模板
- **性能评测**: 使用 high_performance 模板  
- **算法研究**: 使用 experimental 模板
- **生产部署**: 基于 high_performance 模板定制

## 📈 性能追踪

每个配置文件都包含metadata字段，记录：
- 创建时间
- 数据集信息
- 模板类型
- 实验描述
- 预期改进点

这样便于：
- 追踪实验历史
- 复现最佳结果
- 分析性能趋势
- 管理配置版本

## 🔄 配置迁移

如需将旧配置迁移到新命名规范：
1. 复制原配置文件
2. 按新规范重命名
3. 添加metadata字段
4. 更新索引和日志路径
5. 测试配置有效性

## 📝 最佳实践

1. **及时备份**: 重要配置及时备份
2. **描述详细**: metadata中详细描述实验目的
3. **版本控制**: 将配置文件纳入Git管理
4. **结果关联**: 配置文件名与结果文件名保持一致
5. **定期清理**: 删除过时的实验配置
