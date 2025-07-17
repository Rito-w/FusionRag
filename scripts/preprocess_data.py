"""
数据预处理脚本
将下载的数据转换为统一格式，便于后续处理
"""

import os
import sys
import json
import argparse
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """数据预处理器"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def process_beir_dataset(self, dataset_name: str) -> bool:
        """处理BEIR数据集"""
        # 尝试不同的BEIR数据集路径结构
        possible_paths = [
            self.data_dir / "beir" / dataset_name / dataset_name,
            self.data_dir / "beir" / dataset_name / "nq",  # natural-questions特殊情况
            self.data_dir / "beir" / dataset_name / "trec-covid",  # trec-covid特殊情况
        ]
        
        dataset_path = None
        for path in possible_paths:
            if path.exists():
                dataset_path = path
                break
        
        if not dataset_path:
            logger.error(f"数据集不存在，尝试的路径: {possible_paths}")
            return False
        
        logger.info(f"处理BEIR数据集: {dataset_name}，路径: {dataset_path}")
        
        # 处理语料库
        corpus_file = dataset_path / "corpus.jsonl"
        if corpus_file.exists():
            self._process_corpus(corpus_file, dataset_name)
        
        # 处理查询
        queries_file = dataset_path / "queries.jsonl"
        if queries_file.exists():
            self._process_queries(queries_file, dataset_name)
        
        # 处理相关性标注
        qrels_file = dataset_path / "qrels" / "test.tsv"
        if qrels_file.exists():
            self._process_qrels(qrels_file, dataset_name)
        
        logger.info(f"数据集 {dataset_name} 处理完成")
        return True
    
    def _process_corpus(self, corpus_file: Path, dataset_name: str) -> None:
        """处理语料库文件"""
        logger.info(f"处理语料库: {corpus_file}")
        
        documents = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                # 统一格式
                doc = {
                    "doc_id": data.get("_id", data.get("id")),
                    "title": data.get("title", ""),
                    "text": data.get("text", data.get("content", "")),
                    "metadata": {
                        "dataset": dataset_name,
                        "url": data.get("metadata", {}).get("url", "")
                    }
                }
                documents.append(doc)
        
        # 保存处理后的文件
        output_file = self.processed_dir / f"{dataset_name}_corpus.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        logger.info(f"语料库处理完成: {len(documents)} 个文档 -> {output_file}")
    
    def _process_queries(self, queries_file: Path, dataset_name: str) -> None:
        """处理查询文件"""
        logger.info(f"处理查询: {queries_file}")
        
        queries = []
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                # 统一格式
                query = {
                    "query_id": data.get("_id", data.get("id")),
                    "text": data.get("text", data.get("query", "")),
                    "metadata": {
                        "dataset": dataset_name,
                        "url": data.get("metadata", {}).get("url", "")
                    }
                }
                queries.append(query)
        
        # 保存处理后的文件
        output_file = self.processed_dir / f"{dataset_name}_queries.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for query in queries:
                f.write(json.dumps(query, ensure_ascii=False) + '\n')
        
        logger.info(f"查询处理完成: {len(queries)} 个查询 -> {output_file}")
    
    def _process_qrels(self, qrels_file: Path, dataset_name: str) -> None:
        """处理相关性标注文件"""
        logger.info(f"处理标注: {qrels_file}")
        
        # 读取TSV文件
        df = pd.read_csv(qrels_file, sep='\t')
        
        # 重命名列以统一格式
        if 'query-id' in df.columns:
            df = df.rename(columns={'query-id': 'query_id'})
        if 'corpus-id' in df.columns:
            df = df.rename(columns={'corpus-id': 'doc_id'})
        if 'score' in df.columns:
            df = df.rename(columns={'score': 'relevance'})
        
        # 保存处理后的文件
        output_file = self.processed_dir / f"{dataset_name}_qrels.tsv"
        df.to_csv(output_file, sep='\t', index=False)
        
        logger.info(f"标注处理完成: {len(df)} 条记录 -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description="预处理RAG数据集")
    parser.add_argument("--dataset", type=str, help="指定数据集名称")
    parser.add_argument("--all", action="store_true", help="处理所有数据集")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="数据目录")
    
    args = parser.parse_args()
    
    processor = DataProcessor(args.data_dir)
    
    if args.all:
        # 处理所有已下载的BEIR数据集
        beir_dir = Path(args.data_dir) / "beir"
        if beir_dir.exists():
            for dataset_dir in beir_dir.iterdir():
                if dataset_dir.is_dir():
                    processor.process_beir_dataset(dataset_dir.name)
    elif args.dataset:
        processor.process_beir_dataset(args.dataset)
    else:
        print("请指定 --dataset 或 --all 参数")
        parser.print_help()


if __name__ == "__main__":
    main()