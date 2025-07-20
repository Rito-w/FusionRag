#!/usr/bin/env python3
"""
预构建所有数据集的BM25索引缓存
为了加速后续实验，提前构建所有BEIR数据集的BM25索引
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
from pathlib import Path
from typing import List, Dict, Any

from modules.retriever.bm25_retriever import BM25Retriever
from modules.utils.interfaces import Document


def load_dataset_documents(dataset_name: str) -> List[Document]:
    """加载数据集文档"""
    print(f"加载数据集: {dataset_name}")
    
    documents = []
    corpus_path = f"data/processed/{dataset_name}_corpus.jsonl"
    
    if not os.path.exists(corpus_path):
        print(f"⚠️ 数据集文件不存在: {corpus_path}")
        return documents
    
    print(f"加载文档: {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc = Document(
                doc_id=data['doc_id'],
                title=data.get('title', ''),
                text=data.get('text', '')
            )
            documents.append(doc)
    
    print(f"数据集 {dataset_name} 加载完成: 文档数量={len(documents)}")
    return documents


def prebuild_bm25_index(dataset_name: str, config: Dict[str, Any] = None) -> bool:
    """为指定数据集预构建BM25索引"""
    print(f"\n=== 预构建 {dataset_name} BM25索引 ===")
    
    # 检查缓存是否已存在
    cache_dir = Path("checkpoints/retriever_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"bm25_{dataset_name}_index.pkl"
    
    if cache_path.exists():
        print(f"✅ {dataset_name} BM25索引缓存已存在: {cache_path}")
        return True
    
    # 加载数据集
    documents = load_dataset_documents(dataset_name)
    if not documents:
        print(f"❌ {dataset_name} 数据集加载失败")
        return False
    
    # 创建BM25检索器
    bm25_config = config.get('bm25', {}) if config else {}
    retriever = BM25Retriever(bm25_config)
    
    # 构建索引
    print(f"🔄 开始构建 {dataset_name} BM25索引...")
    start_time = time.time()
    
    try:
        retriever.build_index(documents)
        
        # 保存索引到缓存
        retriever.save_index(str(cache_path))
        
        end_time = time.time()
        print(f"✅ {dataset_name} BM25索引构建完成，耗时: {end_time - start_time:.2f}秒")
        print(f"💾 索引已缓存到: {cache_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ {dataset_name} BM25索引构建失败: {e}")
        return False


def main():
    """主函数：为所有BEIR数据集预构建BM25索引"""
    print("🚀 开始预构建所有数据集的BM25索引缓存")
    print("=" * 60)
    
    # 所有BEIR数据集
    datasets = [
        "nfcorpus", 
        "scifact", 
        "fiqa", 
        "arguana", 
        "quora", 
        "scidocs", 
        "trec-covid"
    ]
    
    # 加载配置
    config_path = "configs/cloud_experiments.json"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"📋 已加载配置文件: {config_path}")
    else:
        print("⚠️ 使用默认BM25配置")
        config = {"bm25": {"k1": 1.2, "b": 0.75}}
    
    # 统计结果
    success_count = 0
    failed_datasets = []
    total_start_time = time.time()
    
    # 为每个数据集构建索引
    for dataset in datasets:
        success = prebuild_bm25_index(dataset, config)
        if success:
            success_count += 1
        else:
            failed_datasets.append(dataset)
    
    # 总结
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print("\n" + "=" * 60)
    print("📊 BM25索引预构建总结")
    print(f"✅ 成功构建: {success_count}/{len(datasets)} 个数据集")
    print(f"⏱️ 总耗时: {total_time:.2f}秒")
    
    if failed_datasets:
        print(f"❌ 失败的数据集: {', '.join(failed_datasets)}")
    else:
        print("🎉 所有数据集的BM25索引缓存构建完成！")
    
    # 显示缓存文件信息
    cache_dir = Path("checkpoints/retriever_cache")
    if cache_dir.exists():
        print(f"\n📁 缓存目录: {cache_dir}")
        bm25_files = list(cache_dir.glob("bm25_*_index.pkl"))
        print(f"📦 BM25缓存文件数量: {len(bm25_files)}")
        for cache_file in sorted(bm25_files):
            file_size = cache_file.stat().st_size / (1024 * 1024)  # MB
            print(f"   - {cache_file.name} ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()