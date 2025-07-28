#!/usr/bin/env python3
"""
下载paper1.json中的重要论文并进行分析
"""
import json
import arxiv
import os
import requests
from datetime import datetime
import re

def extract_arxiv_id(link):
    """从链接中提取arXiv ID"""
    match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', link)
    if match:
        return match.group(1)
    return None

def download_paper(arxiv_id, title, download_dir="paper1_downloads"):
    """下载论文PDF"""
    os.makedirs(download_dir, exist_ok=True)
    
    # 生成安全的文件名
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_title = safe_title[:80]  # 限制长度
    filename = f"{arxiv_id}_{safe_title}.pdf"
    filepath = os.path.join(download_dir, filename)
    
    # 如果文件已存在，跳过
    if os.path.exists(filepath):
        print(f"  文件已存在，跳过: {filename}")
        return filepath
    
    try:
        # 使用arXiv API下载
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        result = next(client.results(search))
        
        # 下载PDF
        pdf_url = result.pdf_url
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"  ✅ 下载完成: {filename}")
        return filepath
        
    except Exception as e:
        print(f"  ❌ 下载失败 {arxiv_id}: {e}")
        return None

def load_paper1_json():
    """加载paper1.json文件"""
    with open('paper1.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)
    return papers

def download_top_papers(papers, top_n=15):
    """下载评分最高的前N篇论文"""
    # 按评分排序
    sorted_papers = sorted(papers, key=lambda x: x.get('score', 0), reverse=True)
    
    downloaded_papers = []
    
    print(f"🚀 开始下载前{top_n}篇高分论文...")
    
    for i, paper in enumerate(sorted_papers[:top_n], 1):
        print(f"\n📄 [{i}/{top_n}] {paper['title'][:60]}...")
        print(f"   评分: {paper.get('score', 0):.4f}")
        print(f"   发布: {paper.get('publish_time', 'Unknown')}")
        print(f"   作者: {', '.join(paper.get('authors', [])[:2])}")
        
        # 提取arXiv ID
        arxiv_id = extract_arxiv_id(paper['link'])
        if not arxiv_id:
            print(f"  ❌ 无法提取arXiv ID: {paper['link']}")
            continue
        
        # 下载论文
        filepath = download_paper(arxiv_id, paper['title'])
        if filepath:
            downloaded_papers.append({
                'arxiv_id': arxiv_id,
                'title': paper['title'],
                'filepath': filepath,
                'score': paper.get('score', 0),
                'abstract': paper.get('abstract', ''),
                'authors': paper.get('authors', []),
                'publish_time': paper.get('publish_time', '')
            })
    
    return downloaded_papers

def create_analysis_summary(downloaded_papers):
    """创建论文分析总结"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"paper1_analysis_summary_{timestamp}.md"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Paper1.json 重要论文分析总结\n\n")
        f.write(f"下载时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总计下载: {len(downloaded_papers)} 篇论文\n\n")
        
        f.write("## 📊 论文概览\n\n")
        f.write("| 排名 | arXiv ID | 标题 | 评分 | 发布时间 |\n")
        f.write("|------|----------|------|------|----------|\n")
        
        for i, paper in enumerate(downloaded_papers, 1):
            title_short = paper['title'][:50] + "..." if len(paper['title']) > 50 else paper['title']
            f.write(f"| {i} | {paper['arxiv_id']} | {title_short} | {paper['score']:.4f} | {paper['publish_time']} |\n")
        
        f.write("\n## 📋 详细信息\n\n")
        
        for i, paper in enumerate(downloaded_papers, 1):
            f.write(f"### {i}. {paper['title']}\n\n")
            f.write(f"- **arXiv ID**: {paper['arxiv_id']}\n")
            f.write(f"- **评分**: {paper['score']:.4f}\n")
            f.write(f"- **作者**: {', '.join(paper['authors'])}\n")
            f.write(f"- **发布时间**: {paper['publish_time']}\n")
            f.write(f"- **文件路径**: {paper['filepath']}\n\n")
            f.write(f"**摘要**:\n{paper['abstract'][:300]}...\n\n")
            f.write("---\n\n")
    
    print(f"\n📄 分析总结已保存到: {summary_file}")
    return summary_file

def categorize_papers(downloaded_papers):
    """根据论文内容进行分类"""
    categories = {
        'hybrid_retrieval': [],
        'adaptive_rag': [],
        'query_understanding': [],
        'knowledge_fusion': [],
        'evaluation_methods': [],
        'domain_specific': []
    }
    
    # 关键词分类
    keywords_map = {
        'hybrid_retrieval': ['hybrid', 'blended', 'fusion', 'combine', 'sparse', 'dense'],
        'adaptive_rag': ['adaptive', 'dynamic', 'self-rag', 'corrective', 'tuning'],
        'query_understanding': ['query', 'question', 'complexity', 'intent', 'rewriting'],
        'knowledge_fusion': ['knowledge graph', 'multi-hop', 'reasoning', 'graph'],
        'evaluation_methods': ['evaluation', 'benchmark', 'metric', 'assessment'],
        'domain_specific': ['legal', 'medical', 'financial', 'regulatory', 'domain']
    }
    
    for paper in downloaded_papers:
        text = (paper['title'] + ' ' + paper['abstract']).lower()
        
        # 计算每个类别的匹配分数
        category_scores = {}
        for category, keywords in keywords_map.items():
            score = sum(1 for keyword in keywords if keyword in text)
            category_scores[category] = score
        
        # 分配到得分最高的类别
        best_category = max(category_scores, key=category_scores.get)
        if category_scores[best_category] > 0:
            categories[best_category].append(paper)
        else:
            categories['hybrid_retrieval'].append(paper)  # 默认分类
    
    return categories

def main():
    print("🔍 开始处理paper1.json中的论文...")
    
    # 加载论文列表
    papers = load_paper1_json()
    print(f"📚 找到 {len(papers)} 篇论文")
    
    # 下载前15篇高分论文
    downloaded_papers = download_top_papers(papers, top_n=15)
    
    # 创建分析总结
    summary_file = create_analysis_summary(downloaded_papers)
    
    # 论文分类
    categories = categorize_papers(downloaded_papers)
    
    print(f"\n📊 论文分类结果:")
    for category, papers_in_cat in categories.items():
        if papers_in_cat:
            print(f"  {category}: {len(papers_in_cat)} 篇")
    
    # 保存分类结果
    with open('paper1_categories.json', 'w', encoding='utf-8') as f:
        # 转换为可序列化的格式
        serializable_categories = {}
        for cat, papers_list in categories.items():
            serializable_categories[cat] = [
                {
                    'arxiv_id': p['arxiv_id'],
                    'title': p['title'],
                    'score': p['score']
                } for p in papers_list
            ]
        json.dump(serializable_categories, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 处理完成！")
    print(f"📁 下载目录: paper1_downloads/")
    print(f"📄 分析总结: {summary_file}")
    print(f"📊 分类结果: paper1_categories.json")
    
    return downloaded_papers, categories

if __name__ == "__main__":
    main()
