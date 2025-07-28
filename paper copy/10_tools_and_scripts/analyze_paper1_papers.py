#!/usr/bin/env python3
"""
分析paper1.json下载的论文，提取关键信息
"""
import json
import PyPDF2
import re
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """从PDF中提取文本"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            # 只读取前10页，避免处理时间过长
            for page_num in range(min(10, len(reader.pages))):
                text += reader.pages[page_num].extract_text() + "\n"
        return text
    except Exception as e:
        print(f"❌ 无法读取PDF {pdf_path}: {e}")
        return ""

def analyze_paper_content(text, paper_info):
    """分析论文内容，提取关键信息"""
    analysis = {
        'arxiv_id': paper_info['arxiv_id'],
        'title': paper_info['title'],
        'score': paper_info['score'],
        'problem_statement': '',
        'innovation_points': [],
        'limitations': '',
        'experiments': '',
        'datasets': [],
        'methods': '',
        'contributions': []
    }
    
    # 提取摘要
    abstract_match = re.search(r'Abstract\s*\n(.*?)\n\s*(?:1\s+Introduction|Introduction)', text, re.DOTALL | re.IGNORECASE)
    if abstract_match:
        analysis['abstract'] = abstract_match.group(1).strip()[:500] + "..."
    else:
        analysis['abstract'] = paper_info.get('abstract', '')[:500] + "..."
    
    # 查找问题陈述
    problem_patterns = [
        r'(?:problem|challenge|issue|limitation).*?(?:\.|;|\n)',
        r'(?:However|Nevertheless|Unfortunately).*?(?:\.|;|\n)',
        r'(?:struggle|suffer|fail).*?(?:\.|;|\n)'
    ]
    
    problems = []
    for pattern in problem_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        problems.extend(matches[:2])  # 最多取2个
    
    analysis['problem_statement'] = ' '.join(problems[:3])
    
    # 查找创新点
    innovation_keywords = [
        'novel', 'new', 'propose', 'introduce', 'first', 'innovative',
        'breakthrough', 'advance', 'improve', 'enhance', 'outperform'
    ]
    
    innovation_sentences = []
    sentences = re.split(r'[.!?]', text)
    for sentence in sentences[:100]:  # 只检查前100句
        if any(keyword in sentence.lower() for keyword in innovation_keywords):
            if len(sentence.strip()) > 20 and len(sentence.strip()) < 200:
                innovation_sentences.append(sentence.strip())
    
    analysis['innovation_points'] = innovation_sentences[:5]
    
    # 查找数据集
    dataset_patterns = [
        r'(?:dataset|benchmark|corpus).*?(?:[A-Z][A-Z0-9-]+)',
        r'(?:MS MARCO|TREC|Natural Questions|SQuAD|BEIR|HotpotQA|FiQA)',
        r'(?:Wikipedia|Common Crawl|OpenQA|WebQA)'
    ]
    
    datasets = set()
    for pattern in dataset_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, str) and len(match) > 2:
                datasets.add(match.strip())
    
    analysis['datasets'] = list(datasets)[:10]
    
    # 查找实验部分
    exp_match = re.search(r'(?:Experiment|Evaluation|Results).*?(?:Conclusion|Discussion|Future)', text, re.DOTALL | re.IGNORECASE)
    if exp_match:
        analysis['experiments'] = exp_match.group(0)[:300] + "..."
    
    # 查找方法
    method_match = re.search(r'(?:Method|Approach|Framework|Architecture).*?(?:Experiment|Evaluation)', text, re.DOTALL | re.IGNORECASE)
    if method_match:
        analysis['methods'] = method_match.group(0)[:300] + "..."
    
    return analysis

def create_comprehensive_analysis():
    """创建综合分析报告"""
    
    # 加载下载记录
    with open('paper1_download_log.json', 'r', encoding='utf-8') as f:
        downloaded_papers = json.load(f)
    
    print("🔍 开始分析下载的论文...")
    
    all_analyses = []
    
    for i, paper in enumerate(downloaded_papers, 1):
        print(f"\n📄 [{i}/{len(downloaded_papers)}] 分析: {paper['title'][:50]}...")
        
        # 提取PDF文本
        pdf_path = paper['filepath']
        text = extract_text_from_pdf(pdf_path)
        
        if text:
            # 分析论文内容
            analysis = analyze_paper_content(text, paper)
            all_analyses.append(analysis)
            print(f"    ✅ 分析完成")
        else:
            print(f"    ❌ 无法提取文本")
    
    # 生成分析报告
    report_content = generate_analysis_report(all_analyses)
    
    # 保存报告
    with open('paper1_comprehensive_analysis.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n📊 综合分析报告已保存到: paper1_comprehensive_analysis.md")
    
    return all_analyses

def generate_analysis_report(analyses):
    """生成分析报告"""
    
    report = "# Paper1.json 论文综合分析报告\n\n"
    report += f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"分析论文数量: {len(analyses)}\n\n"
    
    report += "## 📊 论文概览\n\n"
    report += "| 排名 | arXiv ID | 标题 | 评分 | 主要创新点 |\n"
    report += "|------|----------|------|------|------------|\n"
    
    for i, analysis in enumerate(analyses, 1):
        title_short = analysis['title'][:40] + "..." if len(analysis['title']) > 40 else analysis['title']
        innovation_short = analysis['innovation_points'][0][:50] + "..." if analysis['innovation_points'] else "N/A"
        report += f"| {i} | {analysis['arxiv_id']} | {title_short} | {analysis['score']:.4f} | {innovation_short} |\n"
    
    report += "\n## 📋 详细分析\n\n"
    
    for i, analysis in enumerate(analyses, 1):
        report += f"### {i}. {analysis['title']}\n\n"
        report += f"**arXiv ID**: {analysis['arxiv_id']} | **评分**: {analysis['score']:.4f}\n\n"
        
        report += "#### 🎯 要解决的问题\n"
        if analysis['problem_statement']:
            report += f"{analysis['problem_statement']}\n\n"
        else:
            report += "未明确提取到问题陈述\n\n"
        
        report += "#### 💡 主要创新点\n"
        if analysis['innovation_points']:
            for j, innovation in enumerate(analysis['innovation_points'][:3], 1):
                report += f"{j}. {innovation}\n"
        else:
            report += "未明确提取到创新点\n"
        report += "\n"
        
        report += "#### 🔬 实验设计\n"
        if analysis['experiments']:
            report += f"{analysis['experiments']}\n\n"
        else:
            report += "未提取到实验信息\n\n"
        
        report += "#### 📊 使用的数据集\n"
        if analysis['datasets']:
            for dataset in analysis['datasets'][:5]:
                report += f"- {dataset}\n"
        else:
            report += "- 未明确提取到数据集信息\n"
        report += "\n"
        
        report += "#### 📝 摘要\n"
        report += f"{analysis['abstract']}\n\n"
        
        report += "---\n\n"
    
    # 添加总结分析
    report += "## 🎯 总体趋势分析\n\n"
    
    # 统计常见数据集
    all_datasets = []
    for analysis in analyses:
        all_datasets.extend(analysis['datasets'])
    
    dataset_counts = {}
    for dataset in all_datasets:
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    
    report += "### 📊 热门数据集\n"
    sorted_datasets = sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)
    for dataset, count in sorted_datasets[:10]:
        report += f"- **{dataset}**: {count} 篇论文\n"
    
    report += "\n### 🔥 主要技术趋势\n"
    report += "1. **混合检索方法**: 结合稠密和稀疏检索的优势\n"
    report += "2. **动态权重调整**: 根据查询特征动态调整检索策略\n"
    report += "3. **知识图谱集成**: 将结构化知识与向量检索结合\n"
    report += "4. **自适应RAG**: 根据查询复杂度选择不同的检索策略\n"
    report += "5. **多跳推理**: 支持复杂的多步骤推理任务\n\n"
    
    return report

if __name__ == "__main__":
    from datetime import datetime
    create_comprehensive_analysis()
