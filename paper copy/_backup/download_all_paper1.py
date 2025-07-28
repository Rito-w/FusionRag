#!/usr/bin/env python3
"""
下载paper1.json中的所有53篇论文
"""
import json
import os
import requests
from datetime import datetime
import re
import time

def extract_arxiv_id(link):
    """从链接中提取arXiv ID"""
    patterns = [
        r'arxiv\.org/abs/(\d+\.\d+v?\d*)',
        r'arxiv\.org/pdf/(\d+\.\d+v?\d*)',
        r'(\d{4}\.\d{4,5}v?\d*)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, link)
        if match:
            return match.group(1)
    return None

def download_paper_safe(arxiv_id, title, download_dir="all_paper1_downloads"):
    """安全下载论文PDF，带重试机制"""
    os.makedirs(download_dir, exist_ok=True)
    
    # 生成安全的文件名
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_title = safe_title[:50]  # 限制长度
    filename = f"{arxiv_id}_{safe_title}.pdf"
    filepath = os.path.join(download_dir, filename)
    
    # 如果文件已存在，跳过
    if os.path.exists(filepath):
        print(f"  ✅ 文件已存在: {filename}")
        return filepath
    
    # 尝试下载，最多重试3次
    for attempt in range(3):
        try:
            # 直接构造PDF下载链接
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # 下载PDF
            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"  ✅ 下载完成: {filename}")
            return filepath
            
        except Exception as e:
            print(f"  ⚠️ 第{attempt+1}次尝试失败 {arxiv_id}: {e}")
            if attempt < 2:  # 不是最后一次尝试
                time.sleep(2)  # 等待2秒后重试
            else:
                print(f"  ❌ 最终下载失败 {arxiv_id}")
                return None

def categorize_papers_by_keywords(papers):
    """根据关键词对论文进行分类"""
    categories = {
        'hybrid_retrieval': [],
        'adaptive_rag': [],
        'query_processing': [],
        'knowledge_fusion': [],
        'multimodal_rag': [],
        'domain_specific': [],
        'evaluation_methods': [],
        'others': []
    }
    
    # 关键词映射
    keywords_map = {
        'hybrid_retrieval': ['hybrid', 'blended', 'fusion', 'combine', 'sparse', 'dense'],
        'adaptive_rag': ['adaptive', 'dynamic', 'self-rag', 'corrective', 'tuning'],
        'query_processing': ['query', 'question', 'rewriting', 'complexity', 'planning'],
        'knowledge_fusion': ['knowledge graph', 'multi-hop', 'reasoning', 'graph'],
        'multimodal_rag': ['multimodal', 'vision', 'visual', 'image', 'multi-modal'],
        'domain_specific': ['legal', 'medical', 'financial', 'regulatory', 'domain'],
        'evaluation_methods': ['evaluation', 'benchmark', 'metric', 'assessment', 'survey']
    }
    
    for paper in papers:
        text = (paper['title'] + ' ' + paper.get('abstract', '')).lower()
        
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
            categories['others'].append(paper)
    
    return categories

def main():
    print("🚀 开始下载paper1.json中的所有论文...")
    
    # 加载论文列表
    with open('paper1.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    print(f"📚 总共找到 {len(papers)} 篇论文")
    
    # 按评分排序
    sorted_papers = sorted(papers, key=lambda x: x.get('score', 0), reverse=True)
    
    downloaded_papers = []
    failed_papers = []
    
    print(f"🔄 开始批量下载...")
    
    for i, paper in enumerate(sorted_papers, 1):
        print(f"\n📄 [{i}/{len(papers)}] {paper['title'][:60]}...")
        print(f"   评分: {paper.get('score', 0):.4f}")
        
        # 提取arXiv ID
        arxiv_id = extract_arxiv_id(paper['link'])
        if not arxiv_id:
            print(f"  ❌ 无法提取arXiv ID: {paper['link']}")
            failed_papers.append(paper)
            continue
        
        print(f"   arXiv ID: {arxiv_id}")
        
        # 下载论文
        filepath = download_paper_safe(arxiv_id, paper['title'])
        if filepath:
            downloaded_papers.append({
                'arxiv_id': arxiv_id,
                'title': paper['title'],
                'filepath': filepath,
                'score': paper.get('score', 0),
                'abstract': paper.get('abstract', ''),
                'authors': paper.get('authors', []),
                'publish_time': paper.get('publish_time', ''),
                'link': paper['link']
            })
        else:
            failed_papers.append(paper)
        
        # 每下载10篇论文后暂停一下，避免被限制
        if i % 10 == 0:
            print(f"  💤 已下载{i}篇，暂停3秒...")
            time.sleep(3)
    
    print(f"\n📊 下载统计:")
    print(f"✅ 成功下载: {len(downloaded_papers)} 篇")
    print(f"❌ 下载失败: {len(failed_papers)} 篇")
    
    # 保存下载记录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存成功下载的论文
    with open(f'all_paper1_download_log_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(downloaded_papers, f, indent=2, ensure_ascii=False)
    
    # 保存失败的论文
    if failed_papers:
        with open(f'failed_downloads_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(failed_papers, f, indent=2, ensure_ascii=False)
    
    # 论文分类
    categories = categorize_papers_by_keywords(downloaded_papers)
    
    print(f"\n📂 论文分类结果:")
    for category, papers_in_cat in categories.items():
        if papers_in_cat:
            print(f"  {category}: {len(papers_in_cat)} 篇")
    
    # 保存分类结果
    with open(f'all_paper1_categories_{timestamp}.json', 'w', encoding='utf-8') as f:
        # 转换为可序列化的格式
        serializable_categories = {}
        for cat, papers_list in categories.items():
            serializable_categories[cat] = [
                {
                    'arxiv_id': p['arxiv_id'],
                    'title': p['title'],
                    'score': p['score'],
                    'filepath': p['filepath']
                } for p in papers_list
            ]
        json.dump(serializable_categories, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 处理完成！")
    print(f"📁 下载目录: all_paper1_downloads/")
    print(f"📄 下载记录: all_paper1_download_log_{timestamp}.json")
    print(f"📊 分类结果: all_paper1_categories_{timestamp}.json")
    if failed_papers:
        print(f"❌ 失败记录: failed_downloads_{timestamp}.json")
    
    return downloaded_papers, categories

if __name__ == "__main__":
    main()
