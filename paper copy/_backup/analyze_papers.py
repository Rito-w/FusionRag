#!/usr/bin/env python3
"""
深入分析现有论文，提取创新点和技术方案
"""
import os
import json
from pathlib import Path
import PyPDF2
import re
from datetime import datetime

def extract_text_from_pdf(pdf_path):
    """从PDF中提取文本"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"❌ 无法读取PDF {pdf_path}: {e}")
        return ""

def analyze_paper_content(text, paper_name):
    """分析论文内容，提取关键信息"""
    analysis = {
        'paper_name': paper_name,
        'abstract': '',
        'key_contributions': [],
        'methodology': '',
        'experiments': '',
        'limitations': '',
        'innovation_points': []
    }
    
    # 提取摘要
    abstract_match = re.search(r'Abstract\s*\n(.*?)\n\s*(?:1\s+Introduction|Introduction)', text, re.DOTALL | re.IGNORECASE)
    if abstract_match:
        analysis['abstract'] = abstract_match.group(1).strip()[:500] + "..."
    
    # 查找关键词
    innovation_keywords = [
        'dynamic', 'adaptive', 'attention', 'fusion', 'hybrid', 'retrieval',
        'multimodal', 'cross-modal', 'reranking', 'weight', 'learning',
        'neural', 'transformer', 'embedding', 'index', 'query'
    ]
    
    found_keywords = []
    for keyword in innovation_keywords:
        if keyword.lower() in text.lower():
            count = text.lower().count(keyword.lower())
            if count > 3:  # 只记录出现频率较高的关键词
                found_keywords.append(f"{keyword}({count})")
    
    analysis['innovation_points'] = found_keywords
    
    # 查找方法论部分
    method_match = re.search(r'(?:Methodology|Method|Approach)\s*\n(.*?)\n\s*(?:\d+\s+|Experiment)', text, re.DOTALL | re.IGNORECASE)
    if method_match:
        analysis['methodology'] = method_match.group(1).strip()[:300] + "..."
    
    return analysis

def analyze_category(category_path, category_name):
    """分析某个类别的所有论文"""
    print(f"\n📚 分析类别: {category_name}")
    
    pdf_files = list(Path(category_path).glob("*.pdf"))
    analyses = []
    
    for pdf_file in pdf_files:
        print(f"  📄 分析: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        if text:
            analysis = analyze_paper_content(text, pdf_file.name)
            analyses.append(analysis)
            print(f"    ✅ 完成，关键词: {', '.join(analysis['innovation_points'][:5])}")
        else:
            print(f"    ❌ 无法提取文本")
    
    return analyses

def main():
    """主分析函数"""
    print("🔍 开始深入分析现有论文...")
    
    # 定义要分析的重点类别
    categories = {
        "01_hybrid_retrieval": "混合检索",
        "02_multimodal_retrieval": "多模态检索", 
        "07_core_papers": "核心论文"
    }
    
    all_analyses = {}
    
    for cat_dir, cat_name in categories.items():
        if os.path.exists(cat_dir):
            analyses = analyze_category(cat_dir, cat_name)
            all_analyses[cat_dir] = analyses
        else:
            print(f"⚠️ 目录不存在: {cat_dir}")
    
    # 保存分析结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"paper_analysis_results_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_analyses, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 分析结果保存至: {result_file}")
    
    # 生成创新点总结
    generate_innovation_summary(all_analyses)
    
    return result_file

def generate_innovation_summary(all_analyses):
    """生成创新点总结"""
    print("\n" + "="*60)
    print("🎯 创新点分析总结")
    print("="*60)
    
    # 统计所有关键词
    all_keywords = {}
    total_papers = 0
    
    for category, analyses in all_analyses.items():
        print(f"\n📂 {category}:")
        total_papers += len(analyses)
        
        for analysis in analyses:
            print(f"  📄 {analysis['paper_name'][:50]}...")
            if analysis['abstract']:
                print(f"    摘要: {analysis['abstract'][:100]}...")
            print(f"    关键词: {', '.join(analysis['innovation_points'][:8])}")
            
            # 统计关键词
            for keyword_count in analysis['innovation_points']:
                keyword = keyword_count.split('(')[0]
                if keyword in all_keywords:
                    all_keywords[keyword] += 1
                else:
                    all_keywords[keyword] = 1
    
    print(f"\n📊 总计分析了 {total_papers} 篇论文")
    print("\n🔥 热门技术关键词:")
    sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
    for keyword, count in sorted_keywords[:15]:
        print(f"  {keyword}: {count} 篇论文")
    
    # 基于分析结果提出创新方向
    print("\n💡 基于论文分析的创新方向建议:")
    
    if 'dynamic' in [k for k, v in sorted_keywords[:10]]:
        print("\n1. 🎯 动态权重混合检索 - 高优先级")
        print("   - 现有研究: 已有动态调整的初步探索")
        print("   - 创新空间: 更智能的权重学习机制")
        print("   - 技术路径: 强化学习 + 注意力机制")
    
    if 'fusion' in [k for k, v in sorted_keywords[:10]]:
        print("\n2. 🔄 多模态融合优化 - 高优先级") 
        print("   - 现有研究: 基础的融合方法")
        print("   - 创新空间: 查询感知的动态融合")
        print("   - 技术路径: Cross-attention + 层次化融合")
    
    if 'reranking' in [k for k, v in sorted_keywords[:10]]:
        print("\n3. 📈 智能重排序系统 - 中等优先级")
        print("   - 现有研究: 传统重排序方法")
        print("   - 创新空间: 多因子学习排序")
        print("   - 技术路径: 神经排序 + 用户反馈")

if __name__ == "__main__":
    main()
