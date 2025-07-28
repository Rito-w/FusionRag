import os
import shutil
import re
from pathlib import Path

# 读取分类规则（可根据cross_modal_papers_summary.md自动扩展）
FIELD_MAP = {
    '多模态检索': [
        'MAGMaR', 'MMIDR', 'COSINT-Agent', 'Self-Supervised', 'Unified Multimodal', 'Multimodal Fusion',
        'Contrastive Learning', 'Transformer-based Multimodal', 'Cross-Modal Attention', 'Multimodal Pretraining', 'Zero-shot Multimodal',
        'OmniEmbed', 'Video Retrieval', 'Multimodal ArXiv', 'MM-Embed', 'GME', 'SoundCLIP', 'VideoDeepResearch', 'Multimodal Dialogue',
    ],
    '混合检索优化': [
        'Hybrid', 'Modular Retrieval', 'Dynamic Hybrid', 'Learned Hybrid', 'Multimodal Fusion', 'Heterogeneous Graph', 'Phrase Retrieval',
    ],
    '大规模检索系统': [
        'Billion-scale', 'Distributed Vector', 'Approximate Nearest Neighbor', 'Quantization', 'Learned Indexes', 'GPU-accelerated',
        'Real-time Retrieval', 'Scalable Semantic', 'Distributed Indexing', 'Federated Retrieval',
    ],
    '电力系统专用多模态检索': [
        'Agent-based', 'TRACE', 'Serendipitous', 'MedChat', 'Biodiversity', 'Group-Sensitive', 'Transformer-based Fault',
        'Multimodal Sensor', 'Power Equipment', 'Knowledge-enhanced', 'Real-time Grid', 'Multimodal Analysis', 'Document Understanding',
        'Knowledge Graphs', 'Equipment Image', 'Fault Detection', 'Technical Document', 'Maintenance Record', 'Grid Equipment',
    ],
    '知识融合': [
        'SKURG', 'Structured Knowledge', 'Knowledge Graph', 'Entity-centered', 'Unified Retrieval-Generation',
    ],
}

UNCATEGORIZED = '未分类'

# 获取所有pdf文件（不含已分类子目录）
def get_all_pdfs():
    files = []
    for f in os.listdir('.'):
        if f.lower().endswith('.pdf') and os.path.isfile(f):
            files.append(f)
    return files

def match_field(filename):
    for field, keywords in FIELD_MAP.items():
        for kw in keywords:
            if re.search(kw, filename, re.IGNORECASE):
                return field
    return UNCATEGORIZED

def sort_pdfs():
    pdfs = get_all_pdfs()
    for pdf in pdfs:
        field = match_field(pdf)
        target_dir = Path(field)
        target_dir.mkdir(exist_ok=True)
        target_path = target_dir / pdf
        if target_path.exists():
            print(f"已存在: {target_path}")
            continue
        print(f"移动: {pdf} -> {target_path}")
        shutil.move(pdf, target_path)
    print("整理完成！")

if __name__ == '__main__':
    sort_pdfs() 