# Multi-Retriever Fusion Strategy Research Summary

## 🎯 Research Overview

This repository contains a comprehensive study on multi-retriever fusion strategies for information retrieval, conducted across 6 BEIR datasets. The research systematically evaluates 8 different fusion strategies and provides empirical evidence that simple linear weighting outperforms complex RRF (Reciprocal Rank Fusion) methods.

## 📊 Key Experimental Results

### Performance Improvements
- **Quora**: +7% MRR improvement over baseline
- **SciDocs**: +11% MRR improvement over baseline  
- **FiQA**: +8% MRR improvement over baseline

### Main Findings
1. **Simple Linear Weighting > RRF**: Linear weighting consistently outperforms RRF across multiple datasets
2. **Dataset-Specific Strategies**: Different datasets require different fusion approaches
3. **Computational Efficiency**: Simple methods offer better performance-cost trade-offs

## 🔬 Experimental Framework

### Datasets
- 6 BEIR datasets: fiqa, quora, scidocs, nfcorpus, scifact, arguana
- 50-100 queries per dataset for statistical significance

### Retrievers
- BM25 (sparse retrieval)
- E5-large-v2 (dense vector retrieval)

### Evaluation Metrics
- MRR (Mean Reciprocal Rank)
- NDCG@10
- Precision@10
- Recall@10

## 📁 Repository Structure

```
├── examples/                    # Experiment scripts
│   ├── run_smart_baseline.py
│   ├── run_fusion_strategy_comparison.py
│   ├── run_ablation_experiments.py
│   └── run_adaptive_fusion.py
├── reports/                     # Experimental results
│   ├── smart_baseline_results_*.json
│   ├── fusion_strategy_comparison_*.json
│   └── ablation_results_*.json
├── modules/                     # Core implementation
│   ├── adaptive/               # Adaptive fusion strategies
│   ├── retriever/             # Retrieval components
│   └── evaluation/            # Evaluation framework
├── configs/                    # Experiment configurations
└── data/                      # BEIR datasets
```

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Data**:
   ```bash
   python scripts/download_data.py
   ```

3. **Run Experiments**:
   ```bash
   python examples/run_fusion_strategy_comparison.py
   ```

## 📈 Research Impact

### Technical Contributions
- First systematic comparison of 8 fusion strategies
- Counter-intuitive finding: simple methods outperform complex ones
- Dataset-specific strategy selection principles

### Practical Value
- Engineering guidance for real-world systems
- Computational efficiency advantages
- Cross-domain validation

## 📝 Publication Status

This research is being prepared for submission to AAAI 2025. See `EXPERIMENT_RESULTS_SUMMARY.md` for detailed results and `.kiro/specs/paper-publication-plan/` for publication planning.

## 🔗 Related Work

Our approach builds upon recent advances in:
- Multi-retriever fusion (RRF, linear weighting)
- Dense-sparse hybrid retrieval
- Adaptive information retrieval systems

---
*Last Updated: July 20, 2025*
*Research conducted with 24GB RTX 4090 + E5-large-v2 model*