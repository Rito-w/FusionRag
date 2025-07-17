#!/usr/bin/env python
"""
FusionRAG 自动参数调优模块
根据数据集特征自动优化检索器参数，提供通用的性能提升
"""

import numpy as np
import json
from typing import Dict, List, Any, Tuple
from collections import Counter
import re
from dataclasses import dataclass

@dataclass
class DatasetProfile:
    """数据集特征档案"""
    name: str
    doc_count: int
    query_count: int
    avg_doc_length: float
    avg_query_length: float
    doc_length_std: float
    query_length_std: float
    vocab_size: int
    domain: str
    language: str
    complexity_score: float

@dataclass
class OptimalParameters:
    """优化参数集合"""
    bm25_k1: float
    bm25_b: float
    bm25_top_k: int
    dense_top_k: int
    dense_batch_size: int
    graph_entity_threshold: int
    graph_top_k: int
    fusion_weights: Dict[str, float]
    final_top_k: int

class AutoParameterTuner:
    """自动参数调优器"""
    
    def __init__(self):
        # 预定义的参数规则库
        self.parameter_rules = {
            'small_dataset': {  # <1000 docs
                'bm25_k1': (1.2, 1.5),
                'bm25_b': (0.75, 0.8),
                'bm25_top_k': (100, 200),
                'dense_top_k': (100, 200),
                'graph_entity_threshold': (2, 3)
            },
            'medium_dataset': {  # 1000-10000 docs
                'bm25_k1': (1.5, 1.8),
                'bm25_b': (0.75, 0.8),
                'bm25_top_k': (200, 500),
                'dense_top_k': (200, 500),
                'graph_entity_threshold': (3, 5)
            },
            'large_dataset': {  # >10000 docs
                'bm25_k1': (1.8, 2.0),
                'bm25_b': (0.8, 0.85),
                'bm25_top_k': (500, 1000),
                'dense_top_k': (500, 1000),
                'graph_entity_threshold': (5, 8)
            }
        }
        
        # 文档长度适应规则
        self.length_adaptation = {
            'short_docs': {'k1_multiplier': 0.8, 'b_multiplier': 0.9},  # <100 words
            'medium_docs': {'k1_multiplier': 1.0, 'b_multiplier': 1.0},  # 100-300 words
            'long_docs': {'k1_multiplier': 1.2, 'b_multiplier': 1.1}   # >300 words
        }
        
        # 领域特定调整
        self.domain_adjustments = {
            'medical': {'bm25_weight': 0.4, 'dense_weight': 0.5, 'graph_weight': 0.1},
            'covid': {'bm25_weight': 0.35, 'dense_weight': 0.55, 'graph_weight': 0.1},
            'scientific': {'bm25_weight': 0.45, 'dense_weight': 0.45, 'graph_weight': 0.1},
            'qa': {'bm25_weight': 0.5, 'dense_weight': 0.4, 'graph_weight': 0.1},
            'general': {'bm25_weight': 0.4, 'dense_weight': 0.4, 'graph_weight': 0.2}
        }
        
        print("🔧 自动参数调优器初始化完成")
    
    def analyze_dataset_profile(self, documents: List, queries: List, dataset_name: str = "unknown") -> DatasetProfile:
        """分析数据集特征"""
        print(f"📊 分析数据集特征: {dataset_name}")
        
        # 基础统计
        doc_count = len(documents)
        query_count = len(queries)
        
        # 文档长度分析
        doc_lengths = [len(doc.text.split()) for doc in documents]
        avg_doc_length = np.mean(doc_lengths)
        doc_length_std = np.std(doc_lengths)
        
        # 查询长度分析
        query_lengths = [len(q.text.split()) for q in queries]
        avg_query_length = np.mean(query_lengths)
        query_length_std = np.std(query_lengths)
        
        # 词汇表大小估算
        all_text = ' '.join([doc.text for doc in documents[:1000]])  # 采样估算
        vocab_size = len(set(all_text.lower().split()))
        
        # 领域推断
        domain = self._infer_domain(dataset_name, documents[:100])
        
        # 语言检测
        language = self._detect_language(documents[:10])
        
        # 复杂度评分
        complexity_score = self._calculate_complexity(doc_lengths, query_lengths, vocab_size)
        
        profile = DatasetProfile(
            name=dataset_name,
            doc_count=doc_count,
            query_count=query_count,
            avg_doc_length=avg_doc_length,
            avg_query_length=avg_query_length,
            doc_length_std=doc_length_std,
            query_length_std=query_length_std,
            vocab_size=vocab_size,
            domain=domain,
            language=language,
            complexity_score=complexity_score
        )
        
        print(f"✅ 数据集特征分析完成:")
        print(f"   文档: {doc_count}, 平均长度: {avg_doc_length:.1f}")
        print(f"   查询: {query_count}, 平均长度: {avg_query_length:.1f}")
        print(f"   领域: {domain}, 语言: {language}")
        print(f"   复杂度: {complexity_score:.2f}")
        
        return profile
    
    def _infer_domain(self, dataset_name: str, sample_docs: List) -> str:
        """推断数据集领域"""
        dataset_name = dataset_name.lower()
        
        # 基于数据集名称
        if 'medical' in dataset_name or 'nfcorpus' in dataset_name:
            return 'medical'
        elif 'covid' in dataset_name or 'trec-covid' in dataset_name:
            return 'covid'
        elif 'question' in dataset_name or 'qa' in dataset_name:
            return 'qa'
        elif 'scientific' in dataset_name or 'arxiv' in dataset_name:
            return 'scientific'
        
        # 基于内容关键词
        all_text = ' '.join([doc.text[:200] for doc in sample_docs]).lower()
        
        medical_keywords = ['disease', 'patient', 'treatment', 'medical', 'clinical', 'health', 'drug']
        covid_keywords = ['covid', 'virus', 'pandemic', 'vaccine', 'infection', 'respiratory']
        scientific_keywords = ['research', 'study', 'analysis', 'method', 'experiment', 'data']
        
        medical_count = sum(1 for keyword in medical_keywords if keyword in all_text)
        covid_count = sum(1 for keyword in covid_keywords if keyword in all_text)
        scientific_count = sum(1 for keyword in scientific_keywords if keyword in all_text)
        
        if covid_count >= 2:
            return 'covid'
        elif medical_count >= 3:
            return 'medical'
        elif scientific_count >= 3:
            return 'scientific'
        else:
            return 'general'
    
    def _detect_language(self, sample_docs: List) -> str:
        """检测主要语言"""
        sample_text = ' '.join([doc.text[:100] for doc in sample_docs])
        
        # 简单的中英文检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', sample_text))
        english_chars = len(re.findall(r'[a-zA-Z]', sample_text))
        
        if chinese_chars > english_chars * 0.3:
            return 'mixed'
        elif english_chars > 0:
            return 'english'
        else:
            return 'unknown'
    
    def _calculate_complexity(self, doc_lengths: List[float], query_lengths: List[float], vocab_size: int) -> float:
        """计算数据集复杂度评分"""
        # 长度变异系数
        doc_cv = np.std(doc_lengths) / np.mean(doc_lengths) if np.mean(doc_lengths) > 0 else 0
        query_cv = np.std(query_lengths) / np.mean(query_lengths) if np.mean(query_lengths) > 0 else 0
        
        # 词汇丰富度
        vocab_richness = min(vocab_size / 10000, 1.0)  # 归一化到0-1
        
        # 综合复杂度
        complexity = (doc_cv + query_cv + vocab_richness) / 3
        return min(complexity, 1.0)
    
    def optimize_parameters(self, profile: DatasetProfile) -> OptimalParameters:
        """基于数据集特征优化参数"""
        print(f"🎯 为数据集 {profile.name} 优化参数")
        
        # 1. 确定数据集规模类别
        if profile.doc_count < 1000:
            size_category = 'small_dataset'
        elif profile.doc_count < 10000:
            size_category = 'medium_dataset'
        else:
            size_category = 'large_dataset'
        
        # 2. 确定文档长度类别
        if profile.avg_doc_length < 100:
            length_category = 'short_docs'
        elif profile.avg_doc_length < 300:
            length_category = 'medium_docs'
        else:
            length_category = 'long_docs'
        
        # 3. 获取基础参数范围
        base_rules = self.parameter_rules[size_category]
        length_rules = self.length_adaptation[length_category]
        domain_rules = self.domain_adjustments.get(profile.domain, self.domain_adjustments['general'])
        
        # 4. 计算优化参数
        # BM25参数
        k1_range = base_rules['bm25_k1']
        k1 = np.mean(k1_range) * length_rules['k1_multiplier']
        k1 = np.clip(k1, 0.5, 3.0)
        
        b_range = base_rules['bm25_b']
        b = np.mean(b_range) * length_rules['b_multiplier']
        b = np.clip(b, 0.3, 1.0)
        
        # Top-k参数 (基于复杂度调整)
        complexity_factor = 1 + profile.complexity_score * 0.5
        
        bm25_top_k = int(np.mean(base_rules['bm25_top_k']) * complexity_factor)
        dense_top_k = int(np.mean(base_rules['dense_top_k']) * complexity_factor)
        
        # 图检索参数
        graph_threshold = int(np.mean(base_rules['graph_entity_threshold']))
        graph_top_k = min(dense_top_k // 2, 200)
        
        # 融合权重
        fusion_weights = domain_rules.copy()
        
        # 最终输出数量
        final_top_k = min(100, profile.doc_count // 10)
        final_top_k = max(final_top_k, 20)  # 最少20个结果
        
        # Dense批处理大小
        if profile.doc_count > 5000:
            batch_size = 64
        elif profile.doc_count > 1000:
            batch_size = 32
        else:
            batch_size = 16
        
        params = OptimalParameters(
            bm25_k1=k1,
            bm25_b=b,
            bm25_top_k=bm25_top_k,
            dense_top_k=dense_top_k,
            dense_batch_size=batch_size,
            graph_entity_threshold=graph_threshold,
            graph_top_k=graph_top_k,
            fusion_weights=fusion_weights,
            final_top_k=final_top_k
        )
        
        print(f"✅ 参数优化完成:")
        print(f"   BM25: k1={k1:.2f}, b={b:.2f}, top_k={bm25_top_k}")
        print(f"   Dense: top_k={dense_top_k}, batch_size={batch_size}")
        print(f"   Graph: threshold={graph_threshold}, top_k={graph_top_k}")
        print(f"   融合权重: {fusion_weights}")
        print(f"   最终结果数: {final_top_k}")
        
        return params
    
    def generate_config(self, params: OptimalParameters, profile: DatasetProfile) -> Dict[str, Any]:
        """生成优化后的配置文件"""
        config = {
            'dataset_profile': {
                'name': profile.name,
                'doc_count': profile.doc_count,
                'avg_doc_length': profile.avg_doc_length,
                'domain': profile.domain,
                'complexity_score': profile.complexity_score
            },
            'retrievers': {
                'bm25': {
                    'enabled': True,
                    'k1': params.bm25_k1,
                    'b': params.bm25_b,
                    'top_k': params.bm25_top_k
                },
                'dense': {
                    'enabled': True,
                    'model_name': self._select_optimal_model(profile.domain),
                    'top_k': params.dense_top_k,
                    'batch_size': params.dense_batch_size,
                    'device': 'cpu'
                },
                'graph': {
                    'enabled': True,
                    'entity_threshold': params.graph_entity_threshold,
                    'max_walk_length': 2,
                    'top_k': params.graph_top_k
                }
            },
            'fusion': {
                'method': 'weighted',
                'weights': params.fusion_weights,
                'top_k': params.final_top_k
            },
            'evaluation': {
                'metrics': ['recall@5', 'recall@10', 'ndcg@10', 'map'],
                'eval_size': min(1000, profile.query_count)
            }
        }
        
        return config
    
    def _select_optimal_model(self, domain: str) -> str:
        """选择最佳Dense模型"""
        model_mapping = {
            'medical': 'sentence-transformers/all-mpnet-base-v2',
            'covid': 'sentence-transformers/all-mpnet-base-v2',
            'scientific': 'sentence-transformers/allenai-specter',
            'qa': 'sentence-transformers/all-mpnet-base-v2',
            'general': 'sentence-transformers/all-mpnet-base-v2'
        }
        
        return model_mapping.get(domain, 'sentence-transformers/all-mpnet-base-v2')
    
    def save_optimized_config(self, config: Dict[str, Any], dataset_name: str) -> str:
        """保存优化配置到文件"""
        filename = f"configs/auto_optimized_{dataset_name}.yaml"
        
        import yaml
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"💾 优化配置已保存: {filename}")
        return filename

def test_auto_tuner():
    """测试自动参数调优器"""
    print("🧪 测试自动参数调优器")
    print("=" * 50)
    
    # 模拟数据集信息
    from modules.utils.interfaces import Document, Query
    
    # 创建模拟文档
    documents = [
        Document(doc_id=f"doc_{i}", title=f"Title {i}", 
                text=f"Medical research about disease treatment and patient care. " * (10 + i % 20))
        for i in range(500)
    ]
    
    # 创建模拟查询
    queries = [
        Query(query_id=f"q_{i}", text=f"treatment disease patient medical")
        for i in range(50)
    ]
    
    # 初始化调优器
    tuner = AutoParameterTuner()
    
    # 分析数据集特征
    profile = tuner.analyze_dataset_profile(documents, queries, "test_medical")
    
    # 优化参数
    params = tuner.optimize_parameters(profile)
    
    # 生成配置
    config = tuner.generate_config(params, profile)
    
    # 保存配置
    tuner.save_optimized_config(config, "test_medical")
    
    print("\n🎉 自动参数调优测试完成!")

if __name__ == "__main__":
    test_auto_tuner()