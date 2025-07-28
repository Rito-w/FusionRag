#!/usr/bin/env python3
"""
修复版本：下载paper1.json中的重要论文
"""
import json
import os
import requests
from datetime import datetime
import re

def extract_arxiv_id(link):
    """从链接中提取arXiv ID"""
    # 支持多种arXiv链接格式
    patterns = [
        r'arxiv\.org/abs/(\d+\.\d+)',
        r'arxiv\.org/pdf/(\d+\.\d+)',
        r'(\d{4}\.\d{4,5})'  # 直接的ID格式
    ]
    
    for pattern in patterns:
        match = re.search(pattern, link)
        if match:
            return match.group(1)
    return None

def download_paper_direct(arxiv_id, title, download_dir="paper1_downloads"):
    """直接下载论文PDF"""
    os.makedirs(download_dir, exist_ok=True)
    
    # 生成安全的文件名
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_title = safe_title[:60]  # 限制长度
    filename = f"{arxiv_id}_{safe_title}.pdf"
    filepath = os.path.join(download_dir, filename)
    
    # 如果文件已存在，跳过
    if os.path.exists(filepath):
        print(f"  文件已存在，跳过: {filename}")
        return filepath
    
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
        print(f"  ❌ 下载失败 {arxiv_id}: {e}")
        return None

def main():
    print("🔍 开始处理paper1.json中的论文...")
    
    # 加载论文列表
    with open('paper1.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    print(f"📚 找到 {len(papers)} 篇论文")
    
    # 按评分排序，下载前10篇
    sorted_papers = sorted(papers, key=lambda x: x.get('score', 0), reverse=True)
    
    downloaded_papers = []
    
    print(f"🚀 开始下载前10篇高分论文...")
    
    for i, paper in enumerate(sorted_papers[:10], 1):
        print(f"\n📄 [{i}/10] {paper['title'][:60]}...")
        print(f"   评分: {paper.get('score', 0):.4f}")
        
        # 提取arXiv ID
        arxiv_id = extract_arxiv_id(paper['link'])
        if not arxiv_id:
            print(f"  ❌ 无法提取arXiv ID: {paper['link']}")
            continue
        
        print(f"   arXiv ID: {arxiv_id}")
        
        # 下载论文
        filepath = download_paper_direct(arxiv_id, paper['title'])
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
    
    print(f"\n✅ 成功下载 {len(downloaded_papers)} 篇论文")
    
    # 创建下载记录
    with open('paper1_download_log.json', 'w', encoding='utf-8') as f:
        json.dump(downloaded_papers, f, indent=2, ensure_ascii=False)
    
    print(f"📄 下载记录已保存到: paper1_download_log.json")
    
    return downloaded_papers

if __name__ == "__main__":
    main()
