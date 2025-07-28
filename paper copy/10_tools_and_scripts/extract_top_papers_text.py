#!/usr/bin/env python3
"""
提取评分最高的论文文本，便于手工阅读分析
"""
import os
import json
import PyPDF2
from datetime import datetime

def extract_pdf_text(pdf_path, max_pages=15):
    """提取PDF文本，保留更多页面以获得完整信息"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            total_pages = len(reader.pages)
            pages_to_read = min(max_pages, total_pages)
            
            for page_num in range(pages_to_read):
                page_text = reader.pages[page_num].extract_text()
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text + "\n"
            
            return text, total_pages
    except Exception as e:
        print(f"❌ 无法读取PDF {pdf_path}: {e}")
        return "", 0

def extract_top_papers():
    """提取评分最高的前10篇论文的文本"""
    
    # 加载下载记录
    with open('all_paper1_download_log_20250621_003032.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # 按评分排序，取前10篇
    top_papers = sorted(papers, key=lambda x: x['score'], reverse=True)[:10]
    
    print(f"📚 开始提取评分最高的10篇论文文本...")
    
    # 创建输出目录
    output_dir = 'top_papers_text'
    os.makedirs(output_dir, exist_ok=True)
    
    extracted_papers = []
    
    for i, paper in enumerate(top_papers, 1):
        print(f"\n📄 [{i}/10] 提取: {paper['title'][:50]}...")
        print(f"   评分: {paper['score']:.4f}")
        print(f"   arXiv ID: {paper['arxiv_id']}")
        
        try:
            # 检查PDF文件是否存在
            pdf_path = paper['filepath']
            if not os.path.exists(pdf_path):
                print(f"   ❌ PDF文件不存在: {pdf_path}")
                continue
            
            # 提取文本
            text, total_pages = extract_pdf_text(pdf_path)
            
            if text:
                # 保存文本到文件
                text_filename = f"{paper['arxiv_id']}_{paper['title'][:30].replace('/', '_').replace(':', '_')}.txt"
                text_path = os.path.join(output_dir, text_filename)
                
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(f"Title: {paper['title']}\n")
                    f.write(f"arXiv ID: {paper['arxiv_id']}\n")
                    f.write(f"Score: {paper['score']:.4f}\n")
                    f.write(f"Total Pages: {total_pages}\n")
                    f.write(f"Extraction Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(text)
                
                extracted_papers.append({
                    'arxiv_id': paper['arxiv_id'],
                    'title': paper['title'],
                    'score': paper['score'],
                    'original_pdf': pdf_path,
                    'text_file': text_path,
                    'total_pages': total_pages
                })
                
                print(f"   ✅ 提取完成: {text_filename} ({total_pages} 页)")
            else:
                print(f"   ❌ 文本提取失败")
                
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
    
    # 保存提取记录
    with open('extracted_papers_log.json', 'w', encoding='utf-8') as f:
        json.dump(extracted_papers, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 文本提取完成！成功提取 {len(extracted_papers)} 篇论文")
    print(f"📁 输出目录: {output_dir}")
    print(f"📄 提取记录: extracted_papers_log.json")
    
    return extracted_papers

if __name__ == "__main__":
    extract_top_papers()
