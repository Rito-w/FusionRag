# FusionRAG多数据集实验指南

本文档介绍如何使用FusionRAG系统在多个数据集上运行实验，评估系统性能，并比较不同检索器的效果。

## 目录

- [概述](#概述)
- [准备工作](#准备工作)
- [实验脚本](#实验脚本)
- [使用方法](#使用方法)
- [示例](#示例)
- [结果分析](#结果分析)
- [常见问题](#常见问题)

## 概述

FusionRAG系统支持在多个数据集上运行实验，以评估系统在不同类型数据上的性能。通过多数据集实验，您可以：

1. 评估系统在不同领域和类型数据上的泛化能力
2. 比较不同检索器在各种数据集上的表现
3. 分析自适应路由和融合策略的效果
4. 生成详细的性能报告和比较结果

## 准备工作

### 数据集

FusionRAG系统支持BEIR基准中的多个数据集，包括：

- NFCorpus：医学领域问答
- SciFact：科学事实验证
- FiQA：金融领域问答
- TREC-COVID：COVID-19相关研究检索
- ArguAna：论点检索
- Touché-2020：论点检索
- ScidDocs：科学文献检索

### 环境要求

确保您已经安装了所有必要的依赖：

```bash
pip install -r requirements.txt
```

## 实验脚本

FusionRAG提供了以下脚本用于多数据集实验：

1. `scripts/run_experiments.py`：主要的实验运行脚本，支持在多个数据集上运行实验
2. `scripts/run_all_experiments.sh`：批处理脚本，方便一次性在多个数据集上运行实验
3. `examples/run_multi_dataset_example.py`：示例脚本，展示如何使用Python API运行多数据集实验

## 使用方法

### 命令行方式

使用`run_experiments.py`脚本运行实验：

```bash
python scripts/run_experiments.py --config configs/fusionrag_config.yaml --datasets nfcorpus scifact
```

参数说明：

- `--config`：基础配置文件路径
- `--datasets`：要运行的数据集列表
- `--all-datasets`：运行所有支持的数据集
- `--force-rebuild`：强制重建索引
- `--no-auto-download`：不自动下载数据
- `--compare`：比较指定结果文件中的检索器性能

### 批处理方式

使用`run_all_experiments.sh`批处理脚本：

```bash
bash scripts/run_all_experiments.sh --config configs/fusionrag_config.yaml nfcorpus scifact
```

参数与`run_experiments.py`相同。

### Python API方式

您也可以在Python代码中使用实验API：

```python
from scripts.run_experiments import run_multi_dataset_experiments

results = run_multi_dataset_experiments(
    datasets=["nfcorpus", "scifact"],
    base_config_path="configs/fusionrag_config.yaml",
    force_rebuild=False,
    auto_download=True
)
```

## 示例

### 在两个数据集上运行实验

```bash
python scripts/run_experiments.py --datasets nfcorpus scifact
```

### 运行所有数据集并强制重建索引

```bash
python scripts/run_experiments.py --all-datasets --force-rebuild
```

### 比较已有实验结果

```bash
python scripts/run_experiments.py --compare reports/multi_dataset_results.json
```

### 使用示例脚本

```bash
python examples/run_multi_dataset_example.py
```

## 结果分析

实验结果将保存在`reports/`目录下，包括：

1. 每个数据集的详细结果（JSON格式）
2. 多数据集汇总结果（`multi_dataset_results.json`）

您可以使用以下命令比较不同检索器的性能：

```bash
python scripts/run_experiments.py --compare reports/multi_dataset_results.json
```

比较结果将显示每个数据集上各检索器在不同指标上的性能排名。

### 结果示例

```
📊 检索器性能比较:
============================================================

数据集: nfcorpus
----------------------------------------

指标: ndcg@10
  1. semantic_bm25: 0.3245
  2. dense: 0.3142
  3. efficient_vector: 0.3098
  4. cascade: 0.3056
  5. bm25: 0.2876

指标: recall@10
  1. semantic_bm25: 0.4532
  2. dense: 0.4387
  3. cascade: 0.4312
  4. efficient_vector: 0.4298
  5. bm25: 0.3954
```

## 常见问题

### 数据集下载失败

如果数据集下载失败，您可以手动下载并放置在正确的目录：

```bash
python scripts/download_data.py --dataset nfcorpus
python scripts/preprocess_data.py --dataset nfcorpus
```

### 内存不足

如果在处理大型数据集时遇到内存不足问题，可以调整配置文件中的批处理大小：

```yaml
system:
  batch_size: 16  # 减小批处理大小
```

### 如何添加新数据集

要添加新的数据集，需要：

1. 准备符合格式的数据文件（corpus.jsonl, queries.jsonl, qrels.tsv）
2. 将文件放在`data/processed/{dataset_name}/`目录下
3. 使用`--datasets`参数指定新数据集名称

### 如何自定义评估指标

在配置文件中修改评估指标：

```yaml
evaluation:
  metrics: ["recall@5", "recall@10", "ndcg@10", "map", "mrr"]
```

### 如何比较不同配置的实验结果

运行多个实验后，可以使用以下命令比较不同配置的结果：

```bash
python scripts/run_experiments.py --compare reports/config1_results.json --compare reports/config2_results.json
```