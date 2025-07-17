"""
数据下载脚本
支持下载BEIR、MS MARCO等标准数据集
"""
import os
import json
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List
import argparse


class DataDownloader:
    """数据下载器"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # BEIR数据集URL配置
        self.beir_datasets = {
            "nfcorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
            "trec-covid": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip",
            "hotpotqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip",
            "msmarco": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip",
            "natural-questions": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
        }
    
    def download_file(self, url: str, filepath: Path) -> bool:
        """下载文件"""
        try:
            print(f"正在下载: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"下载完成: {filepath}")
            return True
            
        except Exception as e:
            print(f"下载失败 {url}: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """解压缩文件"""
        try:
            print(f"正在解压: {archive_path}")
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                print(f"不支持的压缩格式: {archive_path.suffix}")
                return False
            
            print(f"解压完成: {extract_to}")
            return True
            
        except Exception as e:
            print(f"解压失败 {archive_path}: {e}")
            return False
    
    def download_beir_dataset(self, dataset_name: str) -> bool:
        """下载BEIR数据集"""
        if dataset_name not in self.beir_datasets:
            print(f"不支持的数据集: {dataset_name}")
            print(f"支持的数据集: {list(self.beir_datasets.keys())}")
            return False
        
        url = self.beir_datasets[dataset_name]
        dataset_dir = self.data_dir / "beir" / dataset_name
        archive_path = dataset_dir / f"{dataset_name}.zip"
        
        # 下载数据集
        if not self.download_file(url, archive_path):
            return False
        
        # 解压数据集
        if not self.extract_archive(archive_path, dataset_dir):
            return False
        
        # 清理压缩文件
        archive_path.unlink()
        
        print(f"BEIR数据集 {dataset_name} 下载完成")
        return True
    
    def download_all_beir(self) -> None:
        """下载所有BEIR数据集"""
        for dataset_name in self.beir_datasets:
            self.download_beir_dataset(dataset_name)
    
    def download_custom_data(self, url: str, dataset_name: str) -> bool:
        """下载自定义数据集"""
        dataset_dir = self.data_dir / "custom" / dataset_name
        filename = url.split('/')[-1]
        archive_path = dataset_dir / filename
        
        if not self.download_file(url, archive_path):
            return False
        
        # 如果是压缩文件则解压
        if archive_path.suffix in ['.zip', '.tar', '.gz', '.tgz']:
            if not self.extract_archive(archive_path, dataset_dir):
                return False
            archive_path.unlink()
        
        print(f"自定义数据集 {dataset_name} 下载完成")
        return True
    
    def create_sample_data(self) -> None:
        """创建示例数据用于测试"""
        sample_dir = self.data_dir / "sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建示例语料库
        sample_corpus = [
            {
                "doc_id": "doc_001",
                "title": "机器学习基础",
                "text": "机器学习是人工智能的一个重要分支，它让计算机能够从数据中学习而无需明确编程。",
                "metadata": {"source": "sample", "category": "AI"}
            },
            {
                "doc_id": "doc_002", 
                "title": "深度学习简介",
                "text": "深度学习是机器学习的一个子领域，使用人工神经网络来模拟人脑的学习过程。",
                "metadata": {"source": "sample", "category": "AI"}
            },
            {
                "doc_id": "doc_003",
                "title": "自然语言处理",
                "text": "自然语言处理是计算机科学、人工智能和语言学的交叉领域，研究如何让计算机理解人类语言。",
                "metadata": {"source": "sample", "category": "NLP"}
            }
        ]
        
        corpus_path = sample_dir / "corpus.jsonl"
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for doc in sample_corpus:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        # 创建示例查询
        sample_queries = [
            {
                "query_id": "q_001",
                "text": "什么是机器学习？",
                "metadata": {"type": "factual"}
            },
            {
                "query_id": "q_002",
                "text": "深度学习和机器学习的区别是什么？",
                "metadata": {"type": "analytical"}
            }
        ]
        
        queries_path = sample_dir / "queries.jsonl"
        with open(queries_path, 'w', encoding='utf-8') as f:
            for query in sample_queries:
                f.write(json.dumps(query, ensure_ascii=False) + '\n')
        
        # 创建示例相关性标注
        qrels_content = """q_001 0 doc_001 2
q_001 0 doc_002 1
q_001 0 doc_003 0
q_002 0 doc_001 1
q_002 0 doc_002 2
q_002 0 doc_003 1"""
        
        qrels_path = sample_dir / "qrels.txt"
        with open(qrels_path, 'w', encoding='utf-8') as f:
            f.write(qrels_content)
        
        print(f"示例数据创建完成: {sample_dir}")


def main():
    parser = argparse.ArgumentParser(description="FusionRAG数据下载工具")
    parser.add_argument("--dataset", type=str, help="要下载的数据集名称")
    parser.add_argument("--all-beir", action="store_true", help="下载所有BEIR数据集")
    parser.add_argument("--sample", action="store_true", help="创建示例数据")
    parser.add_argument("--custom-url", type=str, help="自定义数据集URL")
    parser.add_argument("--custom-name", type=str, help="自定义数据集名称")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="数据存储目录")
    
    args = parser.parse_args()
    
    downloader = DataDownloader(args.data_dir)
    
    if args.sample:
        downloader.create_sample_data()
    
    if args.all_beir:
        downloader.download_all_beir()
    
    if args.dataset:
        downloader.download_beir_dataset(args.dataset)
    
    if args.custom_url and args.custom_name:
        downloader.download_custom_data(args.custom_url, args.custom_name)


if __name__ == "__main__":
    main()