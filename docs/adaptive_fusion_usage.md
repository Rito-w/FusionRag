# 自适应混合索引使用指南

## 简介

自适应混合索引（Adaptive Hybrid Indexing）是一种智能检索系统，能够根据查询特征动态选择最优索引路径，并在检索过程中进行自适应调整。本文档提供了系统的使用指南和配置参考。

## 核心特性

1. **自适应索引选择**：根据查询特征自动选择最适合的索引方法
2. **语义增强BM25**：结合关键词匹配和语义相似度
3. **高效向量索引**：使用HNSW等优化的向量索引结构
4. **自适应融合策略**：智能融合多种检索结果
5. **CPU友好**：优化的性能，无需GPU即可高效运行

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from modules.adaptive_hybrid_index import create_adaptive_hybrid_index
from modules.utils.interfaces import Document, Query

# 创建文档
documents = [
    Document(doc_id="1", title="Python编程", text="Python是一种高级编程语言..."),
    Document(doc_id="2", title="机器学习", text="机器学习是人工智能的一个分支...")
]

# 创建自适应混合索引
index = create_adaptive_hybrid_index()

# 构建索引
index.build_index(documents)

# 执行查询
query = Query(query_id="q1", text="Python编程语言介绍")
results = index.retrieve(query, top_k=3)

# 显示结果
for i, result in enumerate(results):
    print(f"{i+1}. {result.doc_id}: {result.document.title} (得分: {result.score:.4f})")
```

### 使用配置文件

```python
import json
from modules.adaptive_hybrid_index import create_adaptive_hybrid_index

# 加载配置
with open("configs/lightweight_config.json", "r") as f:
    config = json.load(f)

# 创建自适应混合索引
index = create_adaptive_hybrid_index(config)

# 构建索引和执行查询...
```

## 配置参考

### 基本配置

```json
{
  "datasets": ["nfcorpus"],
  "include_semantic_bm25": true,
  "bm25": {
    "k1": 1.2,
    "b": 0.75
  },
  "efficient_vector": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "index_type": "hnsw",
    "hnsw_m": 16,
    "hnsw_ef_construction": 200,
    "hnsw_ef_search": 128,
    "batch_size": 8
  },
  "semantic_bm25": {
    "semantic_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "semantic_weight": 0.3,
    "enable_query_expansion": true,
    "query_expansion_terms": 3,
    "enable_document_expansion": false,
    "batch_size": 4
  },
  "query_analyzer": {
    "semantic_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "spacy_model_name": null,
    "disable_ner": true,
    "use_simple_features": true
  },
  "adaptive_router": {
    "routing_strategy": "rule_based",
    "use_simple_rules": true
  },
  "adaptive_fusion": {
    "default_method": "weighted_sum",
    "normalize_scores": true
  },
  "evaluator": {
    "metrics": ["precision", "recall", "mrr", "ndcg", "latency"],
    "top_k_values": [5, 10, 20],
    "report_dir": "reports"
  }
}
```

### 配置选项说明

#### BM25配置

- `k1`：控制词频缩放因子（默认1.2）
- `b`：控制文档长度归一化（默认0.75）

#### 高效向量索引配置

- `model_name`：嵌入模型名称
- `index_type`：索引类型（"hnsw"或"flat"或"ivf"）
- `hnsw_m`：HNSW最大出边数
- `hnsw_ef_construction`：HNSW构建时搜索深度
- `hnsw_ef_search`：HNSW搜索时搜索深度
- `batch_size`：批处理大小

#### 语义增强BM25配置

- `semantic_model_name`：语义模型名称
- `semantic_weight`：语义相似度权重
- `enable_query_expansion`：是否启用查询扩展
- `query_expansion_terms`：查询扩展词数量
- `enable_document_expansion`：是否启用文档扩展
- `batch_size`：批处理大小

#### 查询分析器配置

- `semantic_model_name`：语义模型名称
- `spacy_model_name`：spaCy模型名称（用于NER）
- `disable_ner`：是否禁用命名实体识别
- `use_simple_features`：是否使用简化特征

#### 自适应路由器配置

- `routing_strategy`：路由策略（"rule_based"或"hybrid"）
- `use_simple_rules`：是否使用简化规则

#### 自适应融合配置

- `default_method`：默认融合方法
- `normalize_scores`：是否归一化得分
- `rrf_k`：RRF融合参数k

## 性能调优

### 内存优化

1. **使用轻量级模型**：
   - 使用MiniLM等小型模型替代大型模型
   - 设置`model_name: "sentence-transformers/all-MiniLM-L6-v2"`

2. **减小批处理大小**：
   - 设置较小的批处理大小以减少内存使用
   - 设置`batch_size: 8`或更小

3. **禁用资源密集型功能**：
   - 禁用文档扩展：`enable_document_expansion: false`
   - 禁用NER：`disable_ner: true`

### 速度优化

1. **使用HNSW索引**：
   - 小型数据集使用HNSW索引
   - 设置`index_type: "hnsw"`

2. **使用IVF索引**：
   - 大型数据集使用IVF索引
   - 设置`index_type: "ivf"`

3. **使用模型缓存**：
   - 确保启用模型缓存以避免重复加载模型
   - 设置`use_model_cache: true`

## 实验脚本

系统提供了多个实验脚本，用于评估不同配置的性能：

1. **标准实验**：
   ```bash
   python examples/run_standard_experiments.py --config configs/lightweight_config.json --datasets nfcorpus --sample 10
   ```

2. **消融实验**：
   ```bash
   python examples/run_ablation_experiments.py --config configs/lightweight_config.json --datasets nfcorpus --sample 10
   ```

3. **查询类型分析**：
   ```bash
   python examples/run_query_analysis_experiments.py --config configs/lightweight_config.json --datasets nfcorpus --sample 10
   ```

4. **优化版实验**：
   ```bash
   python examples/run_optimized_experiments.py --config configs/optimized_e5_experiments.json --datasets nfcorpus --sample 10
   ```

5. **运行所有实验**：
   ```bash
   python examples/run_all_experiments.py --config configs/lightweight_config.json --datasets nfcorpus --sample 10
   ```

## 常见问题

### 1. 内存不足错误

**问题**：运行时出现内存不足错误。

**解决方案**：
- 使用轻量级配置：`configs/lightweight_config.json`
- 减小批处理大小：修改配置中的`batch_size`
- 使用`--lightweight`选项：仅使用BM25检索器
- 使用`--sample`选项：减少查询样本大小

### 2. 模型加载失败

**问题**：无法加载语言模型或NER模型。

**解决方案**：
- 检查模型名称是否正确
- 对于spaCy模型，运行`python -m spacy download en_core_web_sm`
- 禁用NER：设置`disable_ner: true`
- 使用本地可用的模型

### 3. 检索结果不理想

**问题**：检索结果质量不高。

**解决方案**：
- 调整BM25参数：尝试不同的`k1`和`b`值
- 调整语义权重：修改`semantic_weight`
- 启用查询扩展：设置`enable_query_expansion: true`
- 尝试不同的融合方法：修改`default_method`

## 贡献指南

欢迎贡献代码和改进建议！请遵循以下步骤：

1. Fork仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'Add your feature'`
4. 推送到分支：`git push origin feature/your-feature`
5. 提交Pull Request