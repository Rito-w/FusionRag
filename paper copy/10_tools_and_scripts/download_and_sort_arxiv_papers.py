import os
import requests
from pathlib import Path
from typing import List, Dict

# 论文信息列表（可扩展）
PAPERS = [
    {
        'url': 'https://arxiv.org/pdf/2403.00231.pdf',
        'title': 'Multimodal_ArXiv_A_Dataset_for_Improving_Scientific_Comprehension_of_Large_Vision_Language_Models',
        'field': '多模态检索'
    },
    {
        'url': 'https://arxiv.org/pdf/2411.02571.pdf',
        'title': 'MM-Embed_Universal_Multimodal_Retrieval_with_Multimodal_LLMs',
        'field': '多模态检索'
    },
    {
        'url': 'https://arxiv.org/pdf/2412.16855.pdf',
        'title': 'GME_Improving_Universal_Multimodal_Retrieval_by_Multimodal_LLMs',
        'field': '多模态检索'
    },
    {
        'url': 'https://arxiv.org/pdf/2212.08632.pdf',
        'title': 'Enhancing_Multi-modal_and_Multi-hop_Question_Answering_via_Structured_Knowledge_and_Unified_Retrieval-Generation',
        'field': '知识融合'
    },
]

def download_and_sort(papers: List[Dict]):
    for paper in papers:
        field_dir = Path(paper['field'])
        field_dir.mkdir(exist_ok=True)
        pdf_path = field_dir / f"{paper['title']}.pdf"
        if pdf_path.exists():
            print(f"已存在: {pdf_path}")
            continue
        print(f"正在下载: {paper['title']} ...")
        try:
            resp = requests.get(paper['url'], timeout=30)
            resp.raise_for_status()
            with open(pdf_path, 'wb') as f:
                f.write(resp.content)
            print(f"保存到: {pdf_path}")
        except Exception as e:
            print(f"下载失败: {paper['title']}，原因: {e}")

if __name__ == '__main__':
    download_and_sort(PAPERS)
    print("全部下载与分类完成！") 