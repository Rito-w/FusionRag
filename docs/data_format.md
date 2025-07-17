# FusionRAG 数据格式规范

## 1. 文档数据格式 (corpus.jsonl)

每行一个JSON对象，包含以下字段：

```json
{
    "doc_id": "doc_001",
    "title": "文档标题",
    "text": "文档正文内容",
    "metadata": {
        "source": "数据来源",
        "category": "分类",
        "timestamp": "2024-01-01T00:00:00",
        "url": "原始链接"
    }
}
```

必需字段：
- `doc_id`: 唯一文档标识符
- `title`: 文档标题
- `text`: 文档正文

可选字段：
- `metadata`: 元数据信息

## 2. 查询数据格式 (queries.jsonl)

```json
{
    "query_id": "q_001",
    "text": "查询文本",
    "metadata": {
        "type": "factual",
        "difficulty": "easy",
        "domain": "general"
    }
}
```

必需字段：
- `query_id`: 唯一查询标识符  
- `text`: 查询文本

## 3. 相关性标注格式 (qrels.txt)

标准TREC格式，每行格式：
```
query_id 0 doc_id relevance_score
```

示例：
```
q_001 0 doc_001 2
q_001 0 doc_002 1
q_001 0 doc_003 0
```

相关性分数：
- 0: 不相关
- 1: 部分相关
- 2: 高度相关

## 4. 分类训练数据格式 (classification_data.jsonl)

```json
{
    "query_id": "q_001",
    "text": "什么是机器学习？",
    "label": "factual",
    "confidence": 0.95
}
```

## 5. 配置文件格式 (config.yaml)

参见 `configs/config.yaml`

## 6. 评测结果格式 (evaluation_results.json)

```json
{
    "timestamp": "2024-01-01T00:00:00",
    "dataset": "test_set",
    "metrics": {
        "recall@5": 0.85,
        "recall@10": 0.92,
        "ndcg@10": 0.78,
        "map": 0.65
    },
    "per_query_results": {
        "q_001": {
            "recall@5": 0.8,
            "ndcg@10": 0.75
        }
    }
}
```

## 7. 日志格式

### 检索日志 (retrieval.log)
```json
{
    "timestamp": "2024-01-01T00:00:00",
    "query_id": "q_001", 
    "query_text": "查询文本",
    "retriever": "bm25",
    "results": [
        {
            "doc_id": "doc_001",
            "score": 0.85,
            "rank": 1
        }
    ],
    "latency_ms": 45
}
```

### 系统日志 (system.log)
标准日志格式：
```
2024-01-01 00:00:00 INFO [module_name] 日志信息
```

## 8. 数据目录结构

```
data/
├── raw/                    # 原始数据
│   ├── beir_datasets/
│   ├── ms_marco/
│   └── custom_data/
├── processed/              # 预处理后数据
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   ├── qrels.txt
│   └── classification_data.jsonl
└── splits/                 # 数据集划分
    ├── train/
    ├── dev/
    └── test/
```

## 9. 索引文件格式

### BM25索引
- 格式：pickle文件
- 文件：`checkpoints/retriever/bm25_index.pkl`

### 向量索引  
- 格式：FAISS索引文件
- 文件：`checkpoints/retriever/dense_index.faiss`

### 图索引
- 格式：NetworkX图对象(pickle)
- 文件：`checkpoints/retriever/knowledge_graph.pkl`

## 10. 模型文件格式

### 分类器模型
- 格式：ONNX
- 文件：`checkpoints/cls/classifier.onnx`

### 重排序模型
- 格式：ONNX
- 文件：`checkpoints/reranker/reranker.onnx`