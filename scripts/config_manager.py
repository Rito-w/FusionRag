#!/usr/bin/env python
"""
配置文件管理脚本
用于创建、管理和组织不同的实验配置文件
"""

import os
import sys
import yaml
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ConfigManager:
    """配置文件管理器"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置模板
        self.templates = {
            "baseline": self._get_baseline_template(),
            "high_performance": self._get_high_performance_template(),
            "experimental": self._get_experimental_template()
        }
    
    def create_config(self, 
                     dataset: str, 
                     template: str = "baseline", 
                     description: str = "",
                     custom_params: Dict[str, Any] = None) -> str:
        """创建新的配置文件"""
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # 生成文件名
        filename = f"{timestamp}_{dataset}_{template}.yaml"
        filepath = self.config_dir / filename
        
        # 获取配置模板
        if template not in self.templates:
            raise ValueError(f"未知模板: {template}. 可用模板: {list(self.templates.keys())}")
        
        config = self.templates[template].copy()
        
        # 更新数据集路径
        config['data']['corpus_path'] = f"data/processed/{dataset}_corpus.jsonl"
        config['data']['queries_path'] = f"data/processed/{dataset}_queries.jsonl"
        config['data']['qrels_path'] = f"data/processed/{dataset}_qrels.tsv"
        
        # 更新索引路径
        for retriever in config['retrievers']:
            if 'index_path' in config['retrievers'][retriever]:
                old_path = config['retrievers'][retriever]['index_path']
                # 在文件名中插入时间戳和数据集
                path_parts = Path(old_path)
                new_name = f"{timestamp}_{dataset}_{path_parts.name}"
                config['retrievers'][retriever]['index_path'] = str(path_parts.parent / new_name)
        
        # 更新日志路径
        config['system']['log_path'] = f"checkpoints/logs/{timestamp}_{dataset}_{template}_system.log"
        config['evaluation']['output_path'] = f"checkpoints/logs/{timestamp}_{dataset}_{template}_eval_results.json"
        
        # 应用自定义参数
        if custom_params:
            config = self._merge_config(config, custom_params)
        
        # 添加元数据
        config['metadata'] = {
            'created_at': datetime.now().isoformat(),
            'dataset': dataset,
            'template': template,
            'description': description,
            'filename': filename
        }
        
        # 保存配置文件
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        print(f"✅ 配置文件已创建: {filepath}")
        return str(filepath)
    
    def list_configs(self) -> None:
        """列出所有配置文件"""
        configs = list(self.config_dir.glob("*.yaml"))
        configs.sort()
        
        print("📋 现有配置文件:")
        print("-" * 80)
        print(f"{'文件名':<40} {'数据集':<15} {'模板':<15} {'创建时间':<20}")
        print("-" * 80)
        
        for config_file in configs:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                metadata = config.get('metadata', {})
                dataset = metadata.get('dataset', 'unknown')
                template = metadata.get('template', 'unknown')
                created_at = metadata.get('created_at', 'unknown')
                
                if created_at != 'unknown':
                    created_at = created_at[:19]  # 只显示日期和时间
                
                print(f"{config_file.name:<40} {dataset:<15} {template:<15} {created_at:<20}")
                
            except Exception as e:
                print(f"{config_file.name:<40} {'error':<15} {'error':<15} {'error':<20}")
    
    def compare_configs(self, config1: str, config2: str) -> None:
        """比较两个配置文件"""
        path1 = self.config_dir / config1
        path2 = self.config_dir / config2
        
        if not path1.exists() or not path2.exists():
            print("❌ 配置文件不存在")
            return
        
        with open(path1, 'r') as f:
            cfg1 = yaml.safe_load(f)
        with open(path2, 'r') as f:
            cfg2 = yaml.safe_load(f)
        
        print(f"🔍 比较配置文件:")
        print(f"  文件1: {config1}")
        print(f"  文件2: {config2}")
        print("-" * 60)
        
        self._compare_dict(cfg1, cfg2, "")
    
    def _compare_dict(self, dict1: Dict, dict2: Dict, prefix: str) -> None:
        """递归比较字典"""
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in sorted(all_keys):
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in dict1:
                print(f"  + {full_key}: {dict2[key]} (仅在文件2中)")
            elif key not in dict2:
                print(f"  - {full_key}: {dict1[key]} (仅在文件1中)")
            elif dict1[key] != dict2[key]:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    self._compare_dict(dict1[key], dict2[key], full_key)
                else:
                    print(f"  ~ {full_key}: {dict1[key]} → {dict2[key]}")
    
    def _merge_config(self, base_config: Dict, custom_params: Dict) -> Dict:
        """合并配置参数"""
        import copy
        result = copy.deepcopy(base_config)
        
        def merge_recursive(base, custom):
            for key, value in custom.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_recursive(base[key], value)
                else:
                    base[key] = value
        
        merge_recursive(result, custom_params)
        return result
    
    def _get_baseline_template(self) -> Dict[str, Any]:
        """基线配置模板"""
        return {
            "data": {
                "corpus_path": "data/processed/nfcorpus_corpus.jsonl",
                "queries_path": "data/processed/nfcorpus_queries.jsonl",
                "qrels_path": "data/processed/nfcorpus_qrels.tsv",
                "output_dir": "data/processed/"
            },
            "retrievers": {
                "bm25": {
                    "enabled": True,
                    "index_path": "checkpoints/retriever/bm25_index.pkl",
                    "k1": 1.2,
                    "b": 0.75,
                    "top_k": 100
                },
                "dense": {
                    "enabled": True,
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "index_path": "checkpoints/retriever/dense_index.faiss",
                    "embedding_dim": 384,
                    "top_k": 100
                },
                "graph": {
                    "enabled": False,
                    "index_path": "checkpoints/retriever/graph_index.pkl",
                    "neo4j_uri": "bolt://localhost:7687",
                    "neo4j_user": "neo4j",
                    "neo4j_password": "fusionrag123",
                    "database": "neo4j",
                    "max_walk_length": 3,
                    "entity_threshold": 2,
                    "top_k": 50
                }
            },
            "classifier": {
                "enabled": False,
                "threshold": 0.5,
                "classes": ["factual", "analytical", "procedural"],
                "adaptation_enabled": False
            },
            "fusion": {
                "method": "weighted",
                "weights": {
                    "bm25": 0.6,
                    "dense": 0.4,
                    "graph": 0.0
                },
                "top_k": 20
            },
            "evaluation": {
                "metrics": ["recall@5", "recall@10", "ndcg@10", "map"],
                "output_path": "checkpoints/logs/eval_results.json"
            },
            "system": {
                "device": "cpu",
                "batch_size": 32,
                "num_threads": 4,
                "log_level": "INFO",
                "log_path": "checkpoints/logs/system.log"
            }
        }
    
    def _get_high_performance_template(self) -> Dict[str, Any]:
        """高性能配置模板"""
        config = self._get_baseline_template()
        
        # 高性能设置
        config['retrievers']['bm25']['top_k'] = 300
        config['retrievers']['dense']['enabled'] = True
        config['retrievers']['dense']['model_name'] = "sentence-transformers/all-mpnet-base-v2"
        config['retrievers']['dense']['embedding_dim'] = 768
        config['retrievers']['dense']['top_k'] = 300
        config['retrievers']['dense']['batch_size'] = 32
        config['retrievers']['graph']['enabled'] = True
        config['retrievers']['graph']['top_k'] = 100
        
        config['classifier']['enabled'] = True
        config['classifier']['adaptation_enabled'] = True
        
        config['fusion']['method'] = "weighted"
        config['fusion']['weights'] = {"bm25": 0.5, "dense": 0.4, "graph": 0.1}
        config['fusion']['top_k'] = 100
        
        config['evaluation']['metrics'] = ["recall@5", "recall@10", "recall@20", "recall@50", "ndcg@10", "ndcg@20", "map", "mrr"]
        config['system']['batch_size'] = 16
        config['system']['num_threads'] = 8
        
        return config
    
    def _get_experimental_template(self) -> Dict[str, Any]:
        """实验配置模板"""
        config = self._get_high_performance_template()
        
        # 实验性设置
        config['fusion']['method'] = "rrf"
        config['fusion']['rrf_k'] = 60
        config['retrievers']['dense']['model_name'] = "sentence-transformers/all-mpnet-base-v2"
        
        return config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FusionRAG配置文件管理器")
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 创建配置
    create_parser = subparsers.add_parser('create', help='创建新配置文件')
    create_parser.add_argument('dataset', help='数据集名称')
    create_parser.add_argument('--template', choices=['baseline', 'high_performance', 'experimental'], 
                              default='baseline', help='配置模板')
    create_parser.add_argument('--description', default='', help='配置描述')
    
    # 列出配置
    subparsers.add_parser('list', help='列出所有配置文件')
    
    # 比较配置
    compare_parser = subparsers.add_parser('compare', help='比较两个配置文件')
    compare_parser.add_argument('config1', help='配置文件1')
    compare_parser.add_argument('config2', help='配置文件2')
    
    args = parser.parse_args()
    
    manager = ConfigManager()
    
    if args.command == 'create':
        manager.create_config(args.dataset, args.template, args.description)
    elif args.command == 'list':
        manager.list_configs()
    elif args.command == 'compare':
        manager.compare_configs(args.config1, args.config2)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
