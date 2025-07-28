#!/usr/bin/env python3
"""
使用mineru转换评分最高的论文为markdown格式，便于详细阅读分析
"""
import os
import sys
import json
from datetime import datetime

# 添加mineru-test目录到路径
sys.path.append('/Users/wrt/PycharmProjects/grid-retrieval-system/mineru-test')

from test_mineru import process_single_pdf

def convert_top_papers():
    """转换评分最高的前10篇论文"""
    
    # 加载下载记录，获取评分最高的论文
    with open('all_paper1_download_log_20250621_003032.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # 按评分排序，取前10篇
    top_papers = sorted(papers, key=lambda x: x['score'], reverse=True)[:10]
    
    print(f"🚀 开始转换评分最高的10篇论文...")
    
    # 创建输出目录
    output_dir = 'top_papers_markdown'
    os.makedirs(output_dir, exist_ok=True)
    
    converted_papers = []
    
    for i, paper in enumerate(top_papers, 1):
        print(f"\n📄 [{i}/10] 转换: {paper['title'][:50]}...")
        print(f"   评分: {paper['score']:.4f}")
        print(f"   arXiv ID: {paper['arxiv_id']}")
        
        try:
            # 检查PDF文件是否存在
            pdf_path = paper['filepath']
            if not os.path.exists(pdf_path):
                print(f"   ❌ PDF文件不存在: {pdf_path}")
                continue
            
            # 使用mineru转换
            result = process_single_pdf(pdf_path, output_dir)
            
            # 记录转换结果
            converted_papers.append({
                'arxiv_id': paper['arxiv_id'],
                'title': paper['title'],
                'score': paper['score'],
                'original_pdf': pdf_path,
                'markdown_file': result['markdown_file'],
                'output_dir': result['output_dir']
            })
            
            print(f"   ✅ 转换完成: {result['markdown_file']}")
            
        except Exception as e:
            print(f"   ❌ 转换失败: {e}")
    
    # 保存转换记录
    with open('converted_papers_log.json', 'w', encoding='utf-8') as f:
        json.dump(converted_papers, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 转换完成！成功转换 {len(converted_papers)} 篇论文")
    print(f"📁 输出目录: {output_dir}")
    print(f"📄 转换记录: converted_papers_log.json")
    
    return converted_papers

if __name__ == "__main__":
    convert_top_papers()
