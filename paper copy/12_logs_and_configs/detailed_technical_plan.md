# 查询意图感知的自适应检索策略 - 详细技术方案

## 🎯 核心技术架构

### 系统整体设计
```python
class IntentAwareAdaptiveRetrieval:
    def __init__(self):
        # 1. 查询意图分类器
        self.intent_classifier = IntentClassifier(
            model_name="distilbert-base-uncased",  # 轻量级BERT
            num_classes=4,
            max_length=128
        )
        
        # 2. 检索策略库
        self.retrieval_strategies = {
            'factual': FactualRetrievalStrategy(),
            'conceptual': ConceptualRetrievalStrategy(), 
            'procedural': ProceduralRetrievalStrategy(),
            'comparative': ComparativeRetrievalStrategy()
        }
        
        # 3. 策略选择器
        self.strategy_selector = StrategySelector()
        
        # 4. 结果融合器
        self.result_fusion = AdaptiveResultFusion()
    
    def search(self, query):
        # Step 1: 意图分类
        intent = self.intent_classifier.predict(query)
        confidence = self.intent_classifier.get_confidence()
        
        # Step 2: 策略选择
        if confidence > 0.8:
            # 高置信度：使用单一策略
            strategy = self.retrieval_strategies[intent]
            results = strategy.retrieve(query)
        else:
            # 低置信度：使用混合策略
            results = self._hybrid_retrieve(query, intent)
        
        return results
```

## 🧠 模型选择与设计

### 1. 查询意图分类器

#### 模型选择：DistilBERT
**为什么选择DistilBERT？**
- **轻量级**: 66M参数，比BERT-base小40%
- **高效**: 推理速度比BERT快60%
- **性能**: 在分类任务上保持97%的BERT性能
- **部署友好**: 内存占用小，适合生产环境

#### 网络架构
```python
class IntentClassifier(nn.Module):
    def __init__(self, model_name, num_classes=4):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probabilities = self.softmax(logits)
        return logits, probabilities
```

#### 训练策略
- **学习率**: 2e-5 (BERT标准学习率)
- **批次大小**: 32
- **训练轮数**: 5 epochs
- **优化器**: AdamW
- **损失函数**: CrossEntropyLoss
- **正则化**: Dropout(0.3) + Weight Decay(0.01)

### 2. 查询意图分类体系

#### 四类查询意图定义
```python
INTENT_DEFINITIONS = {
    'factual': {
        'description': '寻求具体事实、数据、定义的查询',
        'examples': [
            '变压器的额定功率是多少？',
            '电网频率标准值',
            '什么是短路电流？'
        ],
        'keywords': ['什么是', '多少', '定义', '数值', '标准']
    },
    
    'conceptual': {
        'description': '寻求概念解释、原理说明的查询', 
        'examples': [
            '解释电网稳定性的原理',
            '为什么会发生电压波动？',
            '电力系统保护的作用机制'
        ],
        'keywords': ['为什么', '如何', '原理', '机制', '解释']
    },
    
    'procedural': {
        'description': '寻求操作步骤、流程方法的查询',
        'examples': [
            '如何进行变压器维护？',
            '电网故障处理流程',
            '设备安装步骤'
        ],
        'keywords': ['如何', '步骤', '流程', '方法', '操作']
    },
    
    'comparative': {
        'description': '寻求比较分析、优缺点对比的查询',
        'examples': [
            '比较不同类型的发电机',
            '交流与直流输电的优缺点',
            '各种保护装置的差异'
        ],
        'keywords': ['比较', '对比', '差异', '优缺点', '区别']
    }
}
```

## 🔍 检索策略设计

### 1. 事实性查询策略 (Factual Strategy)
```python
class FactualRetrievalStrategy:
    def __init__(self):
        self.exact_matcher = ExactMatcher()      # 精确匹配
        self.keyword_retriever = BM25Retriever() # 关键词检索
        self.vector_retriever = FAISSRetriever() # 向量检索
    
    def retrieve(self, query, top_k=20):
        # 权重分配: 精确匹配(0.5) + 关键词(0.3) + 向量(0.2)
        exact_results = self.exact_matcher.search(query, top_k//2)
        keyword_results = self.keyword_retriever.search(query, top_k//2)
        vector_results = self.vector_retriever.search(query, top_k//4)
        
        # 加权融合
        final_results = self._weighted_fusion(
            [(exact_results, 0.5), (keyword_results, 0.3), (vector_results, 0.2)]
        )
        return final_results[:top_k]
```

### 2. 概念性查询策略 (Conceptual Strategy)
```python
class ConceptualRetrievalStrategy:
    def __init__(self):
        self.vector_retriever = FAISSRetriever()
        self.semantic_expander = SemanticQueryExpander()
        self.keyword_retriever = BM25Retriever()
    
    def retrieve(self, query, top_k=20):
        # 查询扩展
        expanded_query = self.semantic_expander.expand(query)
        
        # 权重分配: 向量检索(0.6) + 扩展查询(0.3) + 关键词(0.1)
        vector_results = self.vector_retriever.search(query, top_k)
        expanded_results = self.vector_retriever.search(expanded_query, top_k//2)
        keyword_results = self.keyword_retriever.search(query, top_k//4)
        
        return self._weighted_fusion([
            (vector_results, 0.6), (expanded_results, 0.3), (keyword_results, 0.1)
        ])[:top_k]
```

### 3. 程序性查询策略 (Procedural Strategy)
```python
class ProceduralRetrievalStrategy:
    def __init__(self):
        self.sequence_matcher = SequenceMatcher()    # 序列匹配
        self.structure_retriever = StructureRetriever() # 结构化检索
        self.vector_retriever = FAISSRetriever()
    
    def retrieve(self, query, top_k=20):
        # 权重分配: 序列匹配(0.4) + 结构化(0.4) + 向量(0.2)
        sequence_results = self.sequence_matcher.search(query, top_k//2)
        structure_results = self.structure_retriever.search(query, top_k//2)
        vector_results = self.vector_retriever.search(query, top_k//4)
        
        return self._weighted_fusion([
            (sequence_results, 0.4), (structure_results, 0.4), (vector_results, 0.2)
        ])[:top_k]
```

### 4. 比较性查询策略 (Comparative Strategy)
```python
class ComparativeRetrievalStrategy:
    def __init__(self):
        self.diversity_retriever = DiversityRetriever() # 多样性检索
        self.contrast_matcher = ContrastMatcher()       # 对比匹配
        self.vector_retriever = FAISSRetriever()
    
    def retrieve(self, query, top_k=20):
        # 权重分配: 多样性(0.4) + 对比匹配(0.4) + 向量(0.2)
        diversity_results = self.diversity_retriever.search(query, top_k)
        contrast_results = self.contrast_matcher.search(query, top_k//2)
        vector_results = self.vector_retriever.search(query, top_k//2)
        
        return self._weighted_fusion([
            (diversity_results, 0.4), (contrast_results, 0.4), (vector_results, 0.2)
        ])[:top_k]
```

## 📊 数据集设计

### 1. 查询意图标注数据集
```python
# 数据集构建计划
INTENT_DATASET = {
    'total_queries': 2000,
    'distribution': {
        'factual': 500,      # 25%
        'conceptual': 600,   # 30%
        'procedural': 500,   # 25%
        'comparative': 400   # 20%
    },
    'sources': [
        'MS MARCO queries',           # 通用查询
        'Natural Questions',          # 事实性查询
        'Stack Overflow',            # 程序性查询
        '自建电网领域查询',            # 领域特定查询
    ],
    'annotation_guidelines': {
        'annotators': 3,             # 3人标注
        'agreement_threshold': 0.8,   # 一致性阈值
        'conflict_resolution': 'majority_vote'
    }
}
```

### 2. 检索评估数据集
```python
EVALUATION_DATASETS = {
    'primary': {
        'MS_MARCO_Passage': {
            'size': '8.8M passages',
            'queries': '6,980 dev queries',
            'relevance': 'human-labeled',
            'use_case': '主要评估数据集'
        }
    },
    
    'secondary': {
        'Natural_Questions': {
            'size': '2.7M passages', 
            'queries': '3,610 dev queries',
            'relevance': 'human-labeled',
            'use_case': '事实性查询评估'
        },
        
        'TREC_DL_2019': {
            'size': '3.2M passages',
            'queries': '43 queries', 
            'relevance': 'graded relevance',
            'use_case': '深度评估'
        }
    },
    
    'domain_specific': {
        'PowerGrid_QA': {
            'size': '自建数据集',
            'queries': '500 queries',
            'relevance': '专家标注',
            'use_case': '领域适应性评估'
        }
    }
}
```

## 📚 相关工作与对比基线

### 1. 主要对比论文
```python
BASELINE_PAPERS = {
    'primary_baselines': [
        {
            'paper': 'DAT: Dynamic Alpha Tuning for Hybrid Retrieval',
            'arxiv': '2506.08276',
            'method': 'Dynamic weight tuning between dense and sparse',
            'limitation': 'Only binary weight adjustment, no intent awareness'
        },
        
        {
            'paper': 'HYRR: Hybrid Infused Reranking for Passage Retrieval', 
            'method': 'Hybrid training for reranking',
            'limitation': 'Reranking stage only, no first-stage optimization'
        }
    ],
    
    'classical_baselines': [
        {
            'method': 'BM25',
            'description': 'Traditional sparse retrieval'
        },
        {
            'method': 'DPR (Dense Passage Retrieval)',
            'description': 'Dense retrieval baseline'
        },
        {
            'method': 'RRF (Reciprocal Rank Fusion)',
            'description': 'Simple fusion method'
        },
        {
            'method': 'Fixed Weight Hybrid',
            'description': 'Fixed 0.5:0.5 weight combination'
        }
    ],
    
    'query_understanding': [
        {
            'paper': 'Query Classification for Web Search',
            'focus': 'Query intent classification methods'
        },
        {
            'paper': 'Understanding User Intent in Search',
            'focus': 'Intent-aware search strategies'
        }
    ]
}
```

### 2. 技术对比分析
```python
COMPARISON_MATRIX = {
    'DAT': {
        'intent_awareness': False,
        'strategy_adaptation': False, 
        'weight_dimensions': 2,  # dense vs sparse
        'computational_cost': 'High (LLM-based)',
        'our_advantage': 'Intent-aware strategy selection'
    },
    
    'HYRR': {
        'intent_awareness': False,
        'strategy_adaptation': False,
        'stage': 'Reranking only',
        'our_advantage': 'First-stage retrieval optimization'
    },
    
    'Fixed_Hybrid': {
        'adaptability': False,
        'query_specificity': False,
        'our_advantage': 'Query-specific strategy adaptation'
    }
}
```

## 🧪 实验设计

### 1. 实验设置
```python
EXPERIMENT_CONFIG = {
    'models': {
        'intent_classifier': 'distilbert-base-uncased',
        'vector_encoder': 'sentence-transformers/all-MiniLM-L6-v2',
        'reranker': 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    },
    
    'hyperparameters': {
        'learning_rate': 2e-5,
        'batch_size': 32,
        'max_length': 128,
        'top_k': 20,
        'temperature': 0.1
    },
    
    'evaluation_metrics': [
        'NDCG@10', 'MRR', 'Recall@100', 
        'MAP', 'Precision@5', 'Latency'
    ]
}
```

### 2. 核心实验
```python
EXPERIMENTS = {
    'exp1_intent_classification': {
        'objective': '验证意图分类器性能',
        'dataset': '2000 labeled queries',
        'metrics': ['Accuracy', 'F1-score', 'Confusion Matrix'],
        'expected_result': 'Accuracy > 85%'
    },
    
    'exp2_strategy_effectiveness': {
        'objective': '验证不同策略的有效性',
        'setup': '每种意图类型单独评估',
        'metrics': ['NDCG@10 per intent type'],
        'expected_result': 'Each strategy outperforms general approach'
    },
    
    'exp3_end_to_end_comparison': {
        'objective': '端到端性能对比',
        'baselines': ['DAT', 'HYRR', 'Fixed Hybrid', 'BM25', 'DPR'],
        'datasets': ['MS MARCO', 'Natural Questions', 'TREC DL'],
        'expected_result': 'Overall NDCG@10 improvement 8-12%'
    },
    
    'exp4_ablation_study': {
        'objective': '消融研究',
        'ablations': [
            'Without intent classification',
            'With single strategy only', 
            'Without confidence-based hybrid',
            'Different weight combinations'
        ],
        'expected_result': 'Each component contributes positively'
    },
    
    'exp5_efficiency_analysis': {
        'objective': '计算效率分析',
        'metrics': ['Query latency', 'Memory usage', 'Throughput'],
        'comparison': 'vs DAT and other baselines',
        'expected_result': '40% latency reduction vs DAT'
    }
}
```

### 3. 评估指标详细定义
```python
EVALUATION_METRICS = {
    'NDCG@10': {
        'formula': 'DCG@10 / IDCG@10',
        'purpose': '主要性能指标',
        'interpretation': '考虑排序质量的相关性指标'
    },
    
    'MRR': {
        'formula': '1/|Q| * Σ(1/rank_i)',
        'purpose': '首个相关结果的排名',
        'interpretation': '越高越好，关注top结果质量'
    },
    
    'Recall@100': {
        'formula': 'Retrieved relevant / Total relevant',
        'purpose': '召回能力评估',
        'interpretation': '检索系统的覆盖能力'
    },
    
    'Intent_Accuracy': {
        'formula': 'Correct predictions / Total predictions',
        'purpose': '意图分类准确率',
        'interpretation': '分类器的基础性能'
    },
    
    'Latency': {
        'unit': 'milliseconds',
        'purpose': '响应速度评估',
        'interpretation': '实际部署的关键指标'
    }
}
```

## 📈 预期结果与分析

### 1. 性能预期
```python
EXPECTED_RESULTS = {
    'overall_performance': {
        'NDCG@10_improvement': '8-12% vs best baseline',
        'MRR_improvement': '6-10% vs best baseline',
        'Recall@100_improvement': '5-8% vs best baseline'
    },
    
    'intent_specific_performance': {
        'factual_queries': '15-20% NDCG@10 improvement',
        'conceptual_queries': '10-15% NDCG@10 improvement', 
        'procedural_queries': '12-18% NDCG@10 improvement',
        'comparative_queries': '8-12% NDCG@10 improvement'
    },
    
    'efficiency_gains': {
        'latency_reduction': '40% vs DAT',
        'memory_overhead': '<10% vs baseline',
        'throughput_improvement': '30-50% vs DAT'
    }
}
```

### 2. 统计显著性测试
```python
STATISTICAL_TESTS = {
    'significance_test': 'Paired t-test',
    'confidence_level': 0.95,
    'effect_size': 'Cohen\'s d',
    'multiple_comparison': 'Bonferroni correction',
    'sample_size': 'Power analysis for 80% power'
}
```

## 💻 实现细节

### 1. 项目结构
```
intent_aware_retrieval/
├── src/
│   ├── models/
│   │   ├── intent_classifier.py      # 意图分类器
│   │   ├── retrieval_strategies.py   # 检索策略
│   │   └── result_fusion.py         # 结果融合
│   ├── data/
│   │   ├── intent_dataset.py        # 意图数据集
│   │   └── evaluation_dataset.py    # 评估数据集
│   ├── experiments/
│   │   ├── train_classifier.py      # 训练分类器
│   │   ├── evaluate_strategies.py   # 策略评估
│   │   └── end_to_end_eval.py      # 端到端评估
│   └── utils/
│       ├── metrics.py              # 评估指标
│       └── config.py               # 配置文件
├── data/
│   ├── intent_labels/              # 意图标注数据
│   ├── retrieval_corpus/           # 检索语料
│   └── evaluation_sets/            # 评估数据集
├── models/
│   ├── intent_classifier.pt        # 训练好的分类器
│   └── retrieval_indices/          # 检索索引
├── results/
│   ├── experiments/                # 实验结果
│   └── analysis/                   # 结果分析
└── requirements.txt
```

### 2. 核心代码实现
```python
# src/models/intent_classifier.py
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer

class IntentClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_classes=4):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        # 意图标签映射
        self.intent_labels = {
            0: 'factual',
            1: 'conceptual',
            2: 'procedural',
            3: 'comparative'
        }

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def predict(self, query_text):
        """预测查询意图"""
        inputs = self.tokenizer(
            query_text,
            return_tensors="pt",
            max_length=128,
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            logits = self.forward(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()

        return self.intent_labels[predicted_class], confidence
```

### 3. 训练脚本
```python
# src/experiments/train_classifier.py
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.metrics import classification_report
import wandb  # 实验跟踪

def train_intent_classifier():
    # 初始化wandb
    wandb.init(project="intent-aware-retrieval", name="intent-classifier")

    # 加载数据
    train_dataset = IntentDataset("data/intent_labels/train.json")
    val_dataset = IntentDataset("data/intent_labels/val.json")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 初始化模型
    model = IntentClassifier()
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(5):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            logits = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(logits, batch['labels'])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 验证
        val_accuracy = evaluate_classifier(model, val_loader)

        # 记录指标
        wandb.log({
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "val_accuracy": val_accuracy
        })

        print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, Val Acc={val_accuracy:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "models/intent_classifier.pt")
    return model
```

### 4. 实验配置文件
```python
# src/utils/config.py
import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ExperimentConfig:
    # 模型配置
    intent_model_name: str = "distilbert-base-uncased"
    vector_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # 训练配置
    learning_rate: float = 2e-5
    batch_size: int = 32
    num_epochs: int = 5
    max_length: int = 128

    # 检索配置
    top_k: int = 20
    confidence_threshold: float = 0.8

    # 策略权重配置
    strategy_weights: Dict[str, Dict[str, float]] = None

    def __post_init__(self):
        if self.strategy_weights is None:
            self.strategy_weights = {
                'factual': {'exact': 0.5, 'keyword': 0.3, 'vector': 0.2},
                'conceptual': {'vector': 0.6, 'expanded': 0.3, 'keyword': 0.1},
                'procedural': {'sequence': 0.4, 'structure': 0.4, 'vector': 0.2},
                'comparative': {'diversity': 0.4, 'contrast': 0.4, 'vector': 0.2}
            }

    # 数据集路径
    datasets: Dict[str, str] = None

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = {
                'ms_marco': 'data/ms_marco/',
                'natural_questions': 'data/natural_questions/',
                'intent_labels': 'data/intent_labels/',
                'evaluation': 'data/evaluation_sets/'
            }

# 全局配置实例
config = ExperimentConfig()
```

### 5. 评估脚本
```python
# src/experiments/end_to_end_eval.py
import json
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class ExperimentEvaluator:
    def __init__(self, config):
        self.config = config
        self.results = {}

    def run_full_evaluation(self):
        """运行完整的评估实验"""

        # 1. 意图分类器评估
        print("🔍 评估意图分类器...")
        intent_results = self.evaluate_intent_classifier()

        # 2. 策略有效性评估
        print("🎯 评估检索策略...")
        strategy_results = self.evaluate_strategies()

        # 3. 端到端性能对比
        print("🏆 端到端性能对比...")
        comparison_results = self.compare_with_baselines()

        # 4. 消融研究
        print("🔬 消融研究...")
        ablation_results = self.ablation_study()

        # 5. 效率分析
        print("⚡ 效率分析...")
        efficiency_results = self.efficiency_analysis()

        # 汇总结果
        self.results = {
            'intent_classification': intent_results,
            'strategy_effectiveness': strategy_results,
            'baseline_comparison': comparison_results,
            'ablation_study': ablation_results,
            'efficiency_analysis': efficiency_results
        }

        # 生成报告
        self.generate_report()

        return self.results

    def compare_with_baselines(self):
        """与基线方法对比"""
        baselines = ['DAT', 'HYRR', 'Fixed_Hybrid', 'BM25', 'DPR']
        datasets = ['ms_marco', 'natural_questions', 'trec_dl']

        results = {}

        for dataset in datasets:
            results[dataset] = {}

            for baseline in baselines:
                # 运行基线方法
                baseline_scores = self.run_baseline(baseline, dataset)
                results[dataset][baseline] = baseline_scores

            # 运行我们的方法
            our_scores = self.run_our_method(dataset)
            results[dataset]['ours'] = our_scores

        return results

    def generate_report(self):
        """生成实验报告"""
        report = {
            'summary': self.generate_summary(),
            'detailed_results': self.results,
            'statistical_tests': self.run_statistical_tests(),
            'visualizations': self.create_visualizations()
        }

        # 保存报告
        with open('results/experiment_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("📊 实验报告已保存到 results/experiment_report.json")

        return report
```

### 6. 部署配置
```python
# deployment/api_server.py
from flask import Flask, request, jsonify
import torch
from src.models.intent_classifier import IntentClassifier
from src.models.retrieval_strategies import IntentAwareRetrieval

app = Flask(__name__)

# 加载模型
intent_model = IntentClassifier()
intent_model.load_state_dict(torch.load('models/intent_classifier.pt'))
retrieval_system = IntentAwareRetrieval(intent_model)

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 20)

    try:
        # 执行检索
        results = retrieval_system.search(query, top_k)

        return jsonify({
            'status': 'success',
            'query': query,
            'intent': results['intent'],
            'confidence': results['confidence'],
            'strategy': results['strategy'],
            'results': results['documents'],
            'latency_ms': results['latency']
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

## 🎯 论文写作大纲

### 标题
"Query Intent-Aware Adaptive Retrieval: Beyond Binary Weight Tuning for Hybrid Information Retrieval"

### 摘要 (150-200词)
- 问题：现有混合检索方法缺乏查询意图感知
- 方法：提出查询意图感知的自适应检索策略
- 结果：在多个数据集上显著提升性能
- 贡献：首个基于意图的检索策略选择框架

### 1. Introduction (1页)
- 混合检索的重要性和现状
- 现有方法的局限性（DAT等）
- 我们的核心思想和贡献
- 论文结构

### 2. Related Work (1.5页)
- 混合检索方法综述
- 查询意图分类研究
- 自适应检索策略
- 与我们工作的区别

### 3. Methodology (2.5页)
- 3.1 查询意图分类体系
- 3.2 意图感知的检索策略设计
- 3.3 自适应策略选择机制
- 3.4 系统架构和实现

### 4. Experiments (2.5页)
- 4.1 实验设置和数据集
- 4.2 基线方法和评估指标
- 4.3 主要实验结果
- 4.4 消融研究
- 4.5 效率分析

### 5. Analysis (1页)
- 5.1 不同意图类型的性能分析
- 5.2 策略选择的有效性分析
- 5.3 错误案例分析
- 5.4 局限性讨论

### 6. Conclusion (0.5页)
- 主要贡献总结
- 实际应用价值
- 未来工作方向

---

这个详细的技术方案涵盖了从模型选择到实验设计、代码实现、部署配置的所有关键环节，为论文的实施提供了完整的技术路线图。整个方案专注于纯文本检索，算力要求适中，具有很强的创新性和实用价值。
