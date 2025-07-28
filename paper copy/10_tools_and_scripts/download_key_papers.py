#!/usr/bin/env python3
"""
下载关键的最新论文来补充分析
"""
import arxiv
import os
import requests
from datetime import datetime

def download_paper(paper_info, download_dir):
    """下载论文PDF"""
    os.makedirs(download_dir, exist_ok=True)
    
    # 生成文件名
    title = paper_info['title']
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_title = safe_title[:60]  # 缩短文件名
    
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

def search_specific_papers():
    """搜索特定的重要论文"""
    client = arxiv.Client()
    
    # 重点搜索查询
    key_searches = [
        {
            'query': 'query-aware retrieval fusion',
            'category': '06_recent_downloads',
            'description': '查询感知检索融合'
        },
        {
            'query': 'adaptive weight learning retrieval',
            'category': '06_recent_downloads', 
            'description': '自适应权重学习检索'
        },
        {
            'query': 'cross-modal attention retrieval',
            'category': '02_multimodal_retrieval',
            'description': '跨模态注意力检索'
        },
        {
            'query': 'incremental vector index update',
            'category': '03_vector_indexing',
            'description': '增量向量索引更新'
        },
        {
            'query': 'dynamic retrieval reranking',
            'category': '05_reranking',
            'description': '动态检索重排序'
        }
    ]
    
    downloaded_papers = []
    
    for search_info in key_searches:
        print(f"\n🔍 搜索: {search_info['description']}")
        print(f"   查询: {search_info['query']}")
        
        search = arxiv.Search(
            query=search_info['query'],
            max_results=3,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
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
            
            print(f"  📄 {paper_info['title'][:60]}...")
            print(f"     作者: {', '.join(paper_info['authors'][:2])}")
            print(f"     发布: {paper_info['published'][:10]}")
            
            try:
                filepath = download_paper(paper_info, search_info['category'])
                downloaded_papers.append({
                    'filepath': filepath,
                    'paper_info': paper_info,
                    'category': search_info['category']
                })
            except Exception as e:
                print(f"  ❌ 下载失败: {e}")
    
    return downloaded_papers

def download_specific_arxiv_papers():
    """下载特定的arXiv论文"""
    # 一些重要的arXiv ID
    specific_papers = [
        {
            'arxiv_id': '2312.10997',  # 最新的hybrid retrieval工作
            'category': '01_hybrid_retrieval'
        },
        {
            'arxiv_id': '2401.08808',  # 多模态检索新方法
            'category': '02_multimodal_retrieval'
        },
        {
            'arxiv_id': '2311.09476',  # 查询理解相关
            'category': '04_query_understanding'
        }
    ]
    
    client = arxiv.Client()
    downloaded_papers = []
    
    for paper_info in specific_papers:
        print(f"\n📥 下载特定论文: {paper_info['arxiv_id']}")
        
        try:
            search = arxiv.Search(id_list=[paper_info['arxiv_id']])
            result = next(client.results(search))
            
            paper_data = {
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
            
            print(f"  📄 {paper_data['title']}")
            
            filepath = download_paper(paper_data, paper_info['category'])
            downloaded_papers.append({
                'filepath': filepath,
                'paper_info': paper_data,
                'category': paper_info['category']
            })
            
        except Exception as e:
            print(f"  ❌ 下载失败: {e}")
    
    return downloaded_papers

def main():
    print("🚀 开始下载关键论文...")
    
    # 创建目录
    os.makedirs("06_recent_downloads", exist_ok=True)
    
    # 搜索并下载论文
    search_papers = search_specific_papers()
    specific_papers = download_specific_arxiv_papers()
    
    all_papers = search_papers + specific_papers
    
    print(f"\n📊 下载总结:")
    print(f"总计下载: {len(all_papers)} 篇论文")
    
    # 按类别统计
    category_count = {}
    for paper in all_papers:
        cat = paper['category']
        category_count[cat] = category_count.get(cat, 0) + 1
    
    for cat, count in category_count.items():
        print(f"  {cat}: {count} 篇")
    
    # 保存下载记录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"06_recent_downloads/download_log_{timestamp}.txt", 'w', encoding='utf-8') as f:
        f.write("下载的关键论文列表\n")
        f.write("="*50 + "\n\n")
        
        for paper in all_papers:
            f.write(f"文件: {paper['filepath']}\n")
            f.write(f"标题: {paper['paper_info']['title']}\n")
            f.write(f"作者: {', '.join(paper['paper_info']['authors'])}\n")
            f.write(f"发布: {paper['paper_info']['published'][:10]}\n")
            f.write(f"类别: {paper['category']}\n")
            f.write(f"摘要: {paper['paper_info']['summary'][:200]}...\n")
            f.write("-" * 50 + "\n\n")
    
    print(f"\n✅ 下载完成！详细记录保存在 06_recent_downloads/download_log_{timestamp}.txt")
    
    return all_papers

if __name__ == "__main__":
    main()
