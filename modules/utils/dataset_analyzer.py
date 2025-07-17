#!/usr/bin/env python
"""
FusionRAG 数据集特征分析器
深度分析数据集特征，为参数优化提供科学依据
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import re
from dataclasses import dataclass, asdict
import pandas as pd

@dataclass
class AdvancedDatasetFeatures:
    """高级数据集特征"""
    # 基础特征
    doc_count: int
    query_count: int
    avg_doc_length: float
    avg_query_length: float
    
    # 分布特征
    doc_length_quartiles: Tuple[float, float, float, float]  # Q1, Q2, Q3, Q4
    query_length_distribution: Dict[str, int]  # short, medium, long
    
    # 词汇特征
    vocab_size: int
    avg_word_frequency: float
    top_frequent_words: List[Tuple[str, int]]
    
    # 语言特征
    language_distribution: Dict[str, float]
    avg_sentence_length: float
    
    # 复杂度特征
    lexical_diversity: float  # 词汇多样性
    semantic_complexity: float  # 语义复杂度
    query_document_overlap: float  # 查询文档重叠度
    
    # 领域特征
    domain_keywords: Dict[str, int]
    entity_density: float  # 实体密度
    
    # 检索难度评估
    retrieval_difficulty: str  # easy, medium, hard
    recommended_approach: List[str]  # 推荐的检索策略

class DatasetFeatureAnalyzer:
    """数据集特征分析器"""
    
    def __init__(self):
        # 领域关键词字典
        self.domain_keywords = {
            'medical': [
                'disease', 'patient', 'treatment', 'therapy', 'clinical', 'medical', 'health',
                'drug', 'medicine', 'hospital', 'doctor', 'diagnosis', 'symptom', 'cure',
                'infection', 'virus', 'bacteria', 'cancer', 'tumor', 'surgery', 'medication'
            ],
            'covid': [
                'covid', 'coronavirus', 'pandemic', 'vaccine', 'vaccination', 'lockdown',
                'quarantine', 'mask', 'social distancing', 'outbreak', 'epidemic', 'ppe',
                'ventilator', 'icu', 'respiratory', 'pneumonia', 'antibody', 'immunity'
            ],
            'scientific': [
                'research', 'study', 'analysis', 'method', 'experiment', 'data', 'result',
                'conclusion', 'hypothesis', 'theory', 'model', 'algorithm', 'evaluation',
                'performance', 'accuracy', 'precision', 'recall', 'validation', 'test'
            ],
            'technology': [
                'computer', 'software', 'algorithm', 'programming', 'code', 'system',
                'network', 'internet', 'database', 'ai', 'machine learning', 'deep learning',
                'neural network', 'automation', 'digital', 'technology', 'innovation'
            ]
        }
        
        # 停用词列表
        self.stop_words = set([
            'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'an', 'a', 'this', 'that', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'all', 'any', 'some', 'no', 'not'
        ])
        
        print("📊 数据集特征分析器初始化完成")
    
    def analyze_comprehensive_features(self, documents: List, queries: List, dataset_name: str = "unknown") -> AdvancedDatasetFeatures:
        """全面分析数据集特征"""
        print(f"🔍 开始全面特征分析: {dataset_name}")
        
        # 1. 基础统计特征
        basic_features = self._analyze_basic_features(documents, queries)
        
        # 2. 分布特征
        distribution_features = self._analyze_distribution_features(documents, queries)
        
        # 3. 词汇特征
        vocabulary_features = self._analyze_vocabulary_features(documents)
        
        # 4. 语言特征
        language_features = self._analyze_language_features(documents)
        
        # 5. 复杂度特征
        complexity_features = self._analyze_complexity_features(documents, queries)
        
        # 6. 领域特征
        domain_features = self._analyze_domain_features(documents)
        
        # 7. 检索难度评估
        difficulty_assessment = self._assess_retrieval_difficulty(
            basic_features, complexity_features, domain_features
        )
        
        # 整合所有特征
        features = AdvancedDatasetFeatures(
            **basic_features,
            **distribution_features,
            **vocabulary_features,
            **language_features,
            **complexity_features,
            **domain_features,
            **difficulty_assessment
        )
        
        print(f"✅ 特征分析完成，检索难度: {features.retrieval_difficulty}")
        return features
    
    def _analyze_basic_features(self, documents: List, queries: List) -> Dict[str, Any]:
        """分析基础特征"""
        doc_lengths = [len(doc.text.split()) for doc in documents]
        query_lengths = [len(q.text.split()) for q in queries]
        
        return {
            'doc_count': len(documents),
            'query_count': len(queries),
            'avg_doc_length': float(np.mean(doc_lengths)),
            'avg_query_length': float(np.mean(query_lengths))
        }
    
    def _analyze_distribution_features(self, documents: List, queries: List) -> Dict[str, Any]:
        """分析分布特征"""
        doc_lengths = [len(doc.text.split()) for doc in documents]
        query_lengths = [len(q.text.split()) for q in queries]
        
        # 文档长度四分位数
        quartiles = np.percentile(doc_lengths, [25, 50, 75, 100])
        
        # 查询长度分布
        query_dist = {'short': 0, 'medium': 0, 'long': 0}
        for length in query_lengths:
            if length <= 3:
                query_dist['short'] += 1
            elif length <= 8:
                query_dist['medium'] += 1
            else:
                query_dist['long'] += 1
        
        return {
            'doc_length_quartiles': tuple(quartiles),
            'query_length_distribution': query_dist
        }
    
    def _analyze_vocabulary_features(self, documents: List) -> Dict[str, Any]:
        """分析词汇特征"""
        # 收集所有词汇
        all_words = []
        for doc in documents[:1000]:  # 采样分析
            words = [word.lower() for word in doc.text.split() 
                    if word.isalpha() and word.lower() not in self.stop_words]
            all_words.extend(words)
        
        # 词频统计
        word_freq = Counter(all_words)
        vocab_size = len(word_freq)
        avg_word_frequency = np.mean(list(word_freq.values()))
        top_words = word_freq.most_common(20)
        
        return {
            'vocab_size': vocab_size,
            'avg_word_frequency': float(avg_word_frequency),
            'top_frequent_words': top_words
        }
    
    def _analyze_language_features(self, documents: List) -> Dict[str, Any]:
        """分析语言特征"""
        sample_docs = documents[:100]
        
        # 语言分布检测
        chinese_count = 0
        english_count = 0
        
        for doc in sample_docs:
            text = doc.text[:500]  # 采样
            chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            
            if chinese_chars > english_chars * 0.3:
                chinese_count += 1
            if english_chars > 0:
                english_count += 1
        
        total_docs = len(sample_docs)
        language_dist = {
            'english': english_count / total_docs,
            'chinese': chinese_count / total_docs,
            'mixed': max(0, (chinese_count + english_count - total_docs) / total_docs)
        }
        
        # 平均句子长度
        sentence_lengths = []
        for doc in sample_docs[:20]:
            sentences = re.split(r'[.!?]', doc.text)
            for sentence in sentences:
                if sentence.strip():
                    sentence_lengths.append(len(sentence.split()))
        
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        
        return {
            'language_distribution': language_dist,
            'avg_sentence_length': float(avg_sentence_length)
        }
    
    def _analyze_complexity_features(self, documents: List, queries: List) -> Dict[str, Any]:
        """分析复杂度特征"""
        # 词汇多样性 (Type-Token Ratio)
        sample_text = ' '.join([doc.text for doc in documents[:100]])
        words = [word.lower() for word in sample_text.split() if word.isalpha()]
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words) if words else 0
        
        # 语义复杂度 (基于词汇和句子结构)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        semantic_complexity = min(avg_word_length / 10 + lexical_diversity, 1.0)
        
        # 查询文档重叠度
        query_words = set()
        for query in queries[:50]:
            query_words.update([word.lower() for word in query.text.split() if word.isalpha()])
        
        doc_words = set()
        for doc in documents[:100]:
            doc_words.update([word.lower() for word in doc.text.split() if word.isalpha()])
        
        overlap = len(query_words & doc_words) / len(query_words | doc_words) if query_words | doc_words else 0
        
        return {
            'lexical_diversity': float(lexical_diversity),
            'semantic_complexity': float(semantic_complexity),
            'query_document_overlap': float(overlap)
        }
    
    def _analyze_domain_features(self, documents: List) -> Dict[str, Any]:
        """分析领域特征"""
        # 领域关键词统计
        domain_scores = defaultdict(int)
        entity_count = 0
        
        sample_text = ' '.join([doc.text for doc in documents[:200]]).lower()
        
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                domain_scores[domain] += sample_text.count(keyword)
        
        # 实体密度估算 (简单的大写词统计)
        words = sample_text.split()
        capitalized_words = [word for word in words if word[0].isupper() and len(word) > 2]
        entity_density = len(capitalized_words) / len(words) if words else 0
        
        return {
            'domain_keywords': dict(domain_scores),
            'entity_density': float(entity_density)
        }
    
    def _assess_retrieval_difficulty(self, basic: Dict, complexity: Dict, domain: Dict) -> Dict[str, Any]:
        """评估检索难度"""
        # 难度评分因子
        difficulty_score = 0
        
        # 文档数量因子
        if basic['doc_count'] > 10000:
            difficulty_score += 0.3
        elif basic['doc_count'] > 1000:
            difficulty_score += 0.2
        
        # 平均文档长度因子
        if basic['avg_doc_length'] > 500:
            difficulty_score += 0.2
        elif basic['avg_doc_length'] > 200:
            difficulty_score += 0.1
        
        # 语义复杂度因子
        difficulty_score += complexity['semantic_complexity'] * 0.3
        
        # 查询文档重叠度因子 (重叠度低表示难度高)
        difficulty_score += (1 - complexity['query_document_overlap']) * 0.2
        
        # 确定难度等级
        if difficulty_score < 0.3:
            difficulty = "easy"
            approaches = ["bm25_focused", "simple_dense"]
        elif difficulty_score < 0.6:
            difficulty = "medium"
            approaches = ["balanced_fusion", "query_expansion"]
        else:
            difficulty = "hard"
            approaches = ["dense_focused", "graph_enhanced", "reranking"]
        
        return {
            'retrieval_difficulty': difficulty,
            'recommended_approach': approaches
        }
    
    def generate_analysis_report(self, features: AdvancedDatasetFeatures, save_path: str = None) -> str:
        """生成详细的分析报告"""
        report = f"""
# 数据集特征分析报告

## 📊 基础统计
- **文档数量**: {features.doc_count:,}
- **查询数量**: {features.query_count:,}
- **平均文档长度**: {features.avg_doc_length:.1f} 词
- **平均查询长度**: {features.avg_query_length:.1f} 词

## 📈 分布特征
- **文档长度四分位数**: {features.doc_length_quartiles}
- **查询长度分布**: 
  - 短查询(≤3词): {features.query_length_distribution['short']}
  - 中等查询(4-8词): {features.query_length_distribution['medium']}
  - 长查询(>8词): {features.query_length_distribution['long']}

## 📚 词汇特征
- **词汇表大小**: {features.vocab_size:,}
- **平均词频**: {features.avg_word_frequency:.2f}
- **高频词**: {features.top_frequent_words[:5]}

## 🌐 语言特征
- **语言分布**: {features.language_distribution}
- **平均句子长度**: {features.avg_sentence_length:.1f} 词

## 🧠 复杂度特征
- **词汇多样性**: {features.lexical_diversity:.3f}
- **语义复杂度**: {features.semantic_complexity:.3f}
- **查询文档重叠度**: {features.query_document_overlap:.3f}

## 🏷️ 领域特征
- **领域关键词分布**: {features.domain_keywords}
- **实体密度**: {features.entity_density:.3f}

## 🎯 检索难度评估
- **难度等级**: {features.retrieval_difficulty}
- **推荐策略**: {features.recommended_approach}

## 💡 优化建议
"""
        
        # 根据特征添加具体建议
        if features.retrieval_difficulty == "easy":
            report += """
- 使用标准BM25参数即可获得良好效果
- Dense检索器可以作为补充
- 无需复杂的图检索或重排序
"""
        elif features.retrieval_difficulty == "medium":
            report += """
- 平衡使用BM25和Dense检索器
- 考虑查询扩展技术
- 可以尝试轻量级的图检索
"""
        else:
            report += """
- 重点使用Dense检索器和语义匹配
- 强烈推荐使用图检索器
- 考虑添加重排序模块
- 可能需要查询重写和扩展
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 分析报告已保存: {save_path}")
        
        return report
    
    def visualize_features(self, features: AdvancedDatasetFeatures, save_path: str = None):
        """可视化数据集特征"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'数据集特征可视化', fontsize=16)
        
        # 1. 查询长度分布
        query_dist = features.query_length_distribution
        axes[0,0].bar(query_dist.keys(), query_dist.values())
        axes[0,0].set_title('查询长度分布')
        axes[0,0].set_ylabel('数量')
        
        # 2. 语言分布
        lang_dist = features.language_distribution
        axes[0,1].pie(lang_dist.values(), labels=lang_dist.keys(), autopct='%1.1f%%')
        axes[0,1].set_title('语言分布')
        
        # 3. 领域关键词
        domain_data = features.domain_keywords
        if domain_data:
            axes[0,2].bar(domain_data.keys(), domain_data.values())
            axes[0,2].set_title('领域关键词分布')
            axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. 复杂度特征
        complexity_metrics = [
            features.lexical_diversity,
            features.semantic_complexity,
            features.query_document_overlap
        ]
        complexity_labels = ['词汇多样性', '语义复杂度', '查询文档重叠度']
        axes[1,0].bar(complexity_labels, complexity_metrics)
        axes[1,0].set_title('复杂度特征')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. 高频词
        if features.top_frequent_words:
            words, freqs = zip(*features.top_frequent_words[:10])
            axes[1,1].barh(words, freqs)
            axes[1,1].set_title('高频词 (Top 10)')
        
        # 6. 文档长度分布
        quartiles = features.doc_length_quartiles
        axes[1,2].boxplot([quartiles], labels=['文档长度'])
        axes[1,2].set_title('文档长度分布 (四分位数)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 特征可视化已保存: {save_path}")
        
        plt.show()

def test_feature_analyzer():
    """测试特征分析器"""
    print("🧪 测试数据集特征分析器")
    print("=" * 50)
    
    # 模拟数据
    from modules.utils.interfaces import Document, Query
    
    documents = [
        Document(doc_id=f"doc_{i}", title=f"Medical Research {i}", 
                text=f"Clinical study on patient treatment and disease diagnosis. " * (5 + i % 15))
        for i in range(200)
    ]
    
    queries = [
        Query(query_id=f"q_{i}", text=f"treatment disease diagnosis patient clinical research")
        for i in range(30)
    ]
    
    # 初始化分析器
    analyzer = DatasetFeatureAnalyzer()
    
    # 分析特征
    features = analyzer.analyze_comprehensive_features(documents, queries, "test_medical")
    
    # 生成报告
    report = analyzer.generate_analysis_report(features, "analysis/test_dataset_analysis.md")
    
    # 可视化
    analyzer.visualize_features(features, "analysis/test_dataset_features.png")
    
    print("\n🎉 特征分析测试完成!")

if __name__ == "__main__":
    test_feature_analyzer()