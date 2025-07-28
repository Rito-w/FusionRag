#!/usr/bin/env python3
"""
整理paper目录，按主题分类论文
"""
import os
import shutil
import re
from pathlib import Path

def organize_papers():
    """整理论文目录结构"""
    paper_dir = Path(".")
    
    # 创建新的目录结构
    categories = {
        "01_hybrid_retrieval": "混合检索相关论文",
        "02_multimodal_retrieval": "多模态检索相关论文", 
        "03_vector_indexing": "向量索引和存储优化",
        "04_query_understanding": "查询理解和扩展",
        "05_reranking": "重排序和融合方法",
        "06_recent_downloads": "最新下载的论文",
        "07_core_papers": "核心创新论文",
        "08_surveys": "综述论文",
        "09_others": "其他相关论文"
    }
    
    # 创建目录
    for cat_dir, desc in categories.items():
        cat_path = paper_dir / cat_dir
        cat_path.mkdir(exist_ok=True)
        # 创建README
        readme_path = cat_path / "README.md"
        if not readme_path.exists():
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"# {desc}\n\n")
    
    # 移动现有的分类文件夹
    if (paper_dir / "混合检索").exists():
        for file in (paper_dir / "混合检索").glob("*"):
            if file.is_file():
                shutil.move(str(file), str(paper_dir / "01_hybrid_retrieval" / file.name))
    
    if (paper_dir / "多模态检索").exists():
        for file in (paper_dir / "多模态检索").glob("*"):
            if file.is_file():
                shutil.move(str(file), str(paper_dir / "02_multimodal_retrieval" / file.name))
    
    if (paper_dir / "知识融合").exists():
        for file in (paper_dir / "知识融合").glob("*"):
            if file.is_file():
                shutil.move(str(file), str(paper_dir / "05_reranking" / file.name))
    
    if (paper_dir / "core_innovation_papers").exists():
        for file in (paper_dir / "core_innovation_papers").glob("*"):
            if file.is_file():
                shutil.move(str(file), str(paper_dir / "07_core_papers" / file.name))
    
    # 根据文件名和内容分类其他论文
    # LEANN论文 -> 向量索引
    leann_files = list(paper_dir.glob("*2506.08276*"))
    for file in leann_files:
        if file.is_file():
            shutil.move(str(file), str(paper_dir / "03_vector_indexing" / file.name))
    
    # 综述论文
    survey_keywords = ["survey", "review", "comprehensive"]
    for file in paper_dir.glob("*.pdf"):
        if any(keyword in file.name.lower() for keyword in survey_keywords):
            shutil.move(str(file), str(paper_dir / "08_surveys" / file.name))
    
    # 移动未分类文件夹中的文件到others
    if (paper_dir / "未分类").exists():
        for file in (paper_dir / "未分类").glob("*"):
            if file.is_file():
                shutil.move(str(file), str(paper_dir / "09_others" / file.name))
    
    # 清理空的旧目录
    old_dirs = ["混合检索", "多模态检索", "知识融合", "core_innovation_papers", "未分类"]
    for old_dir in old_dirs:
        old_path = paper_dir / old_dir
        if old_path.exists() and not any(old_path.iterdir()):
            old_path.rmdir()
    
    print("📁 论文目录整理完成！")
    print("\n目录结构：")
    for cat_dir, desc in categories.items():
        cat_path = paper_dir / cat_dir
        file_count = len([f for f in cat_path.glob("*.pdf")])
        print(f"  {cat_dir}/ - {desc} ({file_count} 篇)")

if __name__ == "__main__":
    organize_papers()
