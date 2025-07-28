import os
import re
import argparse
import requests
from pathlib import Path
from time import sleep

try:
    import arxiv
except ImportError:
    raise ImportError('请先 pip install arxiv')

def clean_filename(name):
    # 去除文件名中的非法字符
    return re.sub(r'[\\/:*?"<>|]', '', name)

def search_and_download(query, max_results, save_dir, category=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 检索关键词: {query}，数量: {max_results}，保存目录: {save_dir}")
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
        category=category
    )
    
    for i, result in enumerate(search.results()):
        title = clean_filename(result.title.strip().replace(' ', '_'))
        pdf_url = result.pdf_url
        filename = save_dir / f"{title}.pdf"
        print(f"[{i+1}] {result.title}")
        print(f"    下载: {pdf_url}")
        try:
            r = requests.get(pdf_url, timeout=30)
            r.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(r.content)
            print(f"    已保存: {filename}")
        except Exception as e:
            print(f"    下载失败: {e}")
        sleep(1)  # 防止请求过快被封

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arXiv论文自动检索与下载')
    parser.add_argument('--query', type=str, required=True, help='检索关键词，如"hybrid retrieval"')
    parser.add_argument('--max_results', type=int, default=10, help='下载论文数量')
    parser.add_argument('--save_dir', type=str, default='.', help='保存目录')
    parser.add_argument('--category', type=str, default=None, help='arXiv分类（可选）')
    args = parser.parse_args()
    search_and_download(args.query, args.max_results, args.save_dir, args.category) 