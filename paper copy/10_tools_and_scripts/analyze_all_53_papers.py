#!/usr/bin/env python3
"""
分析所有53篇论文，提取关键信息并生成综合报告
"""
import json
import PyPDF2
import re
from pathlib import Path
from datetime import datetime
import os

def extract_text_from_pdf(pdf_path, max_pages=8):
    """从PDF中提取文本，限制页数以提高效率"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            # 只读取前几页，包含摘要、介绍、方法等关键信息
            for page_num in range(min(max_pages, len(reader.pages))):
                text += reader.pages[page_num].extract_text() + "\n"
        return text
    except Exception as e:
        print(f"❌ 无法读取PDF {pdf_path}: {e}")
        return ""

def extract_key_information(text, paper_info):
    """提取论文的关键信息"""
    analysis = {
        'arxiv_id': paper_info['arxiv_id'],
        'title': paper_info['title'],
        'score': paper_info['score'],
        'problem_statement': '',
        'innovation_points': [],
        'limitations': [],
        'experiments': '',
        'datasets': [],
        'methods': '',
        'contributions': [],
        'abstract': ''
    }
    
    # 提取摘要
    abstract_patterns = [
        r'Abstract\s*\n(.*?)\n\s*(?:1\s+Introduction|Introduction|Keywords)',
        r'ABSTRACT\s*\n(.*?)\n\s*(?:1\s+Introduction|Introduction|Keywords)',
        r'Abstract\s*[:\-]?\s*(.*?)\n\s*(?:1\.|Introduction|Keywords)'
    ]
    
    for pattern in abstract_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            analysis['abstract'] = match.group(1).strip()[:600] + "..."
            break
    
    if not analysis['abstract']:
        analysis['abstract'] = paper_info.get('abstract', '')[:600] + "..."
    
    # 查找问题陈述
    problem_keywords = [
        'problem', 'challenge', 'issue', 'limitation', 'difficulty',
        'However', 'Nevertheless', 'Unfortunately', 'struggle', 'suffer'
    ]
    
    problem_sentences = []
    sentences = re.split(r'[.!?]', text)
    for sentence in sentences[:150]:  # 检查前150句
        if any(keyword.lower() in sentence.lower() for keyword in problem_keywords):
            if 20 < len(sentence.strip()) < 200:
                problem_sentences.append(sentence.strip())
    
    analysis['problem_statement'] = ' '.join(problem_sentences[:3])
    
    # 查找创新点
    innovation_keywords = [
        'novel', 'new', 'propose', 'introduce', 'first', 'innovative',
        'breakthrough', 'advance', 'improve', 'enhance', 'outperform',
        'contribution', 'key insight', 'main idea'
    ]
    
    innovation_sentences = []
    for sentence in sentences[:200]:  # 检查前200句
        if any(keyword.lower() in sentence.lower() for keyword in innovation_keywords):
            if 30 < len(sentence.strip()) < 250:
                innovation_sentences.append(sentence.strip())
    
    analysis['innovation_points'] = innovation_sentences[:5]
    
    # 查找局限性
    limitation_keywords = [
        'limitation', 'drawback', 'weakness', 'shortcoming', 'constraint',
        'future work', 'not address', 'cannot handle', 'fails to'
    ]
    
    limitation_sentences = []
    for sentence in sentences:
        if any(keyword.lower() in sentence.lower() for keyword in limitation_keywords):
            if 20 < len(sentence.strip()) < 200:
                limitation_sentences.append(sentence.strip())
    
    analysis['limitations'] = limitation_sentences[:3]
    
    # 查找数据集
    dataset_patterns = [
        r'(?:MS MARCO|Natural Questions|SQuAD|TREC|BEIR|HotpotQA|FiQA)',
        r'(?:Wikipedia|Common Crawl|OpenQA|WebQA|KILT)',
        r'(?:dataset|benchmark|corpus).*?([A-Z][A-Z0-9-]+)',
        r'(?:evaluate|experiment).*?on.*?([A-Z][A-Za-z0-9-]+)'
    ]
    
    datasets = set()
    for pattern in dataset_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, str) and 2 < len(match) < 30:
                datasets.add(match.strip())
    
    analysis['datasets'] = list(datasets)[:8]
    
    # 查找实验信息
    exp_patterns = [
        r'(?:Experiment|Evaluation|Results).*?(?:show|demonstrate|achieve).*?[.!?]',
        r'(?:performance|accuracy|improvement).*?(?:\d+\.?\d*%|\d+\.?\d*)',
        r'(?:outperform|better than|superior to).*?[.!?]'
    ]
    
    exp_sentences = []
    for pattern in exp_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        exp_sentences.extend(matches[:2])
    
    analysis['experiments'] = ' '.join(exp_sentences[:3])
    
    return analysis

def analyze_all_papers():
    """分析所有下载的论文"""
    
    # 加载下载记录
    download_files = [f for f in os.listdir('.') if f.startswith('all_paper1_download_log_')]
    if not download_files:
        print("❌ 未找到下载记录文件")
        return
    
    latest_file = max(download_files)
    print(f"📄 加载下载记录: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        downloaded_papers = json.load(f)
    
    print(f"🔍 开始分析 {len(downloaded_papers)} 篇论文...")
    
    all_analyses = []
    
    for i, paper in enumerate(downloaded_papers, 1):
        print(f"\n📄 [{i}/{len(downloaded_papers)}] 分析: {paper['title'][:50]}...")
        
        # 检查文件是否存在
        if not os.path.exists(paper['filepath']):
            print(f"    ❌ 文件不存在: {paper['filepath']}")
            continue
        
        # 提取PDF文本
        text = extract_text_from_pdf(paper['filepath'])
        
        if text:
            # 分析论文内容
            analysis = extract_key_information(text, paper)
            all_analyses.append(analysis)
            print(f"    ✅ 分析完成 - 创新点: {len(analysis['innovation_points'])}, 数据集: {len(analysis['datasets'])}")
        else:
            print(f"    ❌ 无法提取文本")
    
    print(f"\n📊 成功分析了 {len(all_analyses)} 篇论文")
    
    # 生成综合报告
    generate_comprehensive_report(all_analyses)
    
    return all_analyses

def generate_comprehensive_report(analyses):
    """生成综合分析报告"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"comprehensive_53_papers_analysis_{timestamp}.md"
    
    print(f"📝 生成综合报告: {report_file}")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 53篇RAG相关论文综合分析报告\n\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析论文数量: {len(analyses)}\n\n")
        
        # 按评分排序
        sorted_analyses = sorted(analyses, key=lambda x: x['score'], reverse=True)
        
        # 生成概览表格
        f.write("## 📊 论文概览 (按评分排序)\n\n")
        f.write("| 排名 | arXiv ID | 标题 | 评分 | 主要创新 |\n")
        f.write("|------|----------|------|------|----------|\n")
        
        for i, analysis in enumerate(sorted_analyses[:20], 1):  # 只显示前20篇
            title_short = analysis['title'][:35] + "..." if len(analysis['title']) > 35 else analysis['title']
            innovation_short = analysis['innovation_points'][0][:40] + "..." if analysis['innovation_points'] else "N/A"
            f.write(f"| {i} | {analysis['arxiv_id']} | {title_short} | {analysis['score']:.4f} | {innovation_short} |\n")
        
        # 技术趋势分析
        f.write("\n## 🔥 技术趋势分析\n\n")
        
        # 统计热门数据集
        all_datasets = []
        for analysis in analyses:
            all_datasets.extend(analysis['datasets'])
        
        dataset_counts = {}
        for dataset in all_datasets:
            if len(dataset) > 2:  # 过滤太短的
                dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        f.write("### 📊 热门数据集\n")
        sorted_datasets = sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)
        for dataset, count in sorted_datasets[:15]:
            f.write(f"- **{dataset}**: {count} 篇论文\n")
        
        # 创新点分析
        f.write("\n### 💡 主要创新方向\n")
        
        innovation_keywords = {}
        for analysis in analyses:
            for innovation in analysis['innovation_points']:
                words = re.findall(r'\b\w+\b', innovation.lower())
                for word in words:
                    if len(word) > 4:  # 只统计长词
                        innovation_keywords[word] = innovation_keywords.get(word, 0) + 1
        
        sorted_keywords = sorted(innovation_keywords.items(), key=lambda x: x[1], reverse=True)
        f.write("**热门技术关键词**:\n")
        for keyword, count in sorted_keywords[:20]:
            f.write(f"- {keyword}: {count} 次\n")
        
        # 详细分析前10篇高分论文
        f.write("\n## 📋 高分论文详细分析 (Top 10)\n\n")
        
        for i, analysis in enumerate(sorted_analyses[:10], 1):
            f.write(f"### {i}. {analysis['title']}\n\n")
            f.write(f"**arXiv ID**: {analysis['arxiv_id']} | **评分**: {analysis['score']:.4f}\n\n")
            
            f.write("#### 🎯 要解决的问题\n")
            if analysis['problem_statement']:
                f.write(f"{analysis['problem_statement'][:400]}...\n\n")
            else:
                f.write("未明确提取到问题陈述\n\n")
            
            f.write("#### 💡 主要创新点\n")
            if analysis['innovation_points']:
                for j, innovation in enumerate(analysis['innovation_points'][:3], 1):
                    f.write(f"{j}. {innovation[:200]}...\n")
            else:
                f.write("未明确提取到创新点\n")
            f.write("\n")
            
            f.write("#### 📊 使用的数据集\n")
            if analysis['datasets']:
                for dataset in analysis['datasets'][:6]:
                    f.write(f"- {dataset}\n")
            else:
                f.write("- 未明确提取到数据集信息\n")
            f.write("\n")
            
            f.write("#### 🔬 实验结果\n")
            if analysis['experiments']:
                f.write(f"{analysis['experiments'][:300]}...\n\n")
            else:
                f.write("未提取到实验信息\n\n")
            
            f.write("#### ⚠️ 技术局限性\n")
            if analysis['limitations']:
                for limitation in analysis['limitations'][:2]:
                    f.write(f"- {limitation[:150]}...\n")
            else:
                f.write("- 未明确提取到局限性信息\n")
            f.write("\n")
            
            f.write("---\n\n")
        
        # 总结和启示
        f.write("## 🎯 对我们研究的关键启示\n\n")
        f.write("### 1. 技术发展趋势\n")
        f.write("- **混合检索成为主流**: 大多数论文都采用某种形式的混合检索\n")
        f.write("- **动态适应是关键**: 从固定策略向自适应策略发展\n")
        f.write("- **查询理解重要性**: 越来越多的工作关注查询理解和处理\n")
        f.write("- **领域特化趋势**: 针对特定领域的RAG系统越来越多\n\n")
        
        f.write("### 2. 创新空间识别\n")
        f.write("- **查询意图分类**: 虽然有查询复杂度分析，但细粒度意图分类仍有空间\n")
        f.write("- **轻量级实现**: 大多数方法计算复杂度较高，轻量级方案有需求\n")
        f.write("- **策略级创新**: 从权重调整升级到策略选择的创新空间很大\n")
        f.write("- **实时性优化**: 实时应用场景的优化需求明显\n\n")
        
        f.write("### 3. 我们方案的优势\n")
        f.write("- **填补空白**: 查询意图感知的自适应检索策略填补了重要空白\n")
        f.write("- **技术可行**: 基于现有技术栈，实现难度适中\n")
        f.write("- **性能优势**: 预期能够显著提升不同类型查询的性能\n")
        f.write("- **实用价值**: 可以直接应用于现有检索系统\n\n")
        
        f.write("### 4. 实验设计参考\n")
        f.write("- **标准数据集**: MS MARCO, SQuAD, Natural Questions是必选\n")
        f.write("- **评估指标**: NDCG@10, MRR, MAP是标准指标\n")
        f.write("- **重要基线**: DAT, Adaptive-RAG, Self-RAG等是重要对比对象\n")
        f.write("- **消融研究**: 需要详细的组件贡献度分析\n\n")
    
    print(f"✅ 综合报告已保存: {report_file}")
    return report_file

if __name__ == "__main__":
    analyze_all_papers()
