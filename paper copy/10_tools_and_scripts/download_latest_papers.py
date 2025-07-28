#!/usr/bin/env python3
"""
下载最新的相关论文进行深入分析
"""
import arxiv
import os
import requests
import json
from datetime import datetime

def download_paper(paper_info, download_dir):
    """下载论文PDF"""
    os.makedirs(download_dir, exist_ok=True)
    
    # 生成文件名
    title = paper_info['title']
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_title = safe_title[:80]  # 缩短文件名
    
    # 从entry_id提取arxiv ID
    arxiv_id = paper_info['entry_id'].split('/')[-1]
    filename = f"{arxiv_id}_{safe_title}.pdf"
    filepath = os.path.join(download_dir, filename)
    
    # 如果文件已存在，跳过
    if os.path.exists(filepath):
        print(f"  文件已存在，跳过: {filename}")
        return filepath
    
    # 下载PDF
    pdf_url = paper_info['pdf_url']
    response = requests.get(pdf_url, stream=True)
    response.raise_for_status()
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"  ✅ 下载完成: {filename}")
    return filepath

def search_and_download(keywords, category_dir, max_results=3):
    """搜索并下载论文"""
    client = arxiv.Client()
    search = arxiv.Search(
        query=keywords,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers = []
    downloaded_files = []
    
    print(f"\n🔍 搜索: {keywords}")
    
    for result in client.results(search):
        paper_info = {
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'summary': result.summary,
            'published': str(result.published),
            'updated': str(result.updated),
            'entry_id': result.entry_id,
            'pdf_url': result.pdf_url,
            'categories': result.categories,
            'primary_category': result.primary_category
        }
        papers.append(paper_info)
        
        # 下载论文
        try:
            filepath = download_paper(paper_info, category_dir)
            downloaded_files.append(filepath)
        except Exception as e:
            print(f"  ❌ 下载失败: {e}")
    
    return papers, downloaded_files

def main():
    # 定义重点搜索查询
    search_queries = [
        ("hybrid retrieval dense sparse", "01_hybrid_retrieval"),
        ("dynamic weight fusion retrieval", "01_hybrid_retrieval"), 
        ("multimodal retrieval attention", "02_multimodal_retrieval"),
        ("vector index update incremental", "03_vector_indexing"),
        ("query understanding expansion", "04_query_understanding"),
        ("retrieval reranking learning", "05_reranking")
    ]
    
    all_results = {}
    
    for query, category in search_queries:
        try:
            papers, files = search_and_download(query, category, max_results=2)
            all_results[query] = {
                'papers': papers,
                'downloaded_files': files,
                'category': category
            }
            print(f"✅ {query}: 下载了 {len(files)} 篇论文")
        except Exception as e:
            print(f"❌ {query}: 搜索失败 - {e}")
            all_results[query] = {'papers': [], 'downloaded_files': [], 'category': category}
    
    # 保存搜索结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"06_recent_downloads/download_results_{timestamp}.json"
    
    os.makedirs("06_recent_downloads", exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 搜索结果保存至: {result_file}")
    
    # 统计下载情况
    total_downloaded = sum(len(result['downloaded_files']) for result in all_results.values())
    print(f"\n📊 总计下载了 {total_downloaded} 篇最新论文")
    
    return result_file

if __name__ == "__main__":
    main()
