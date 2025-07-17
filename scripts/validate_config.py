#!/usr/bin/env python3
"""
配置验证工具
验证FusionRAG配置文件的正确性和完整性
"""

import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import jsonschema
from jsonschema import validate, ValidationError


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.schema = self._get_config_schema()
        self.warnings = []
        self.errors = []
    
    def _get_config_schema(self) -> Dict[str, Any]:
        """获取配置文件的JSON Schema"""
        return {
            "type": "object",
            "required": ["metadata", "data", "retrievers", "fusion", "evaluation"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["dataset", "template"],
                    "properties": {
                        "dataset": {"type": "string"},
                        "template": {"type": "string"},
                        "description": {"type": "string"},
                        "version": {"type": "string"}
                    }
                },
                "data": {
                    "type": "object",
                    "required": ["corpus_path", "queries_path", "qrels_path"],
                    "properties": {
                        "corpus_path": {"type": "string"},
                        "queries_path": {"type": "string"},
                        "qrels_path": {"type": "string"}
                    }
                },
                "retrievers": {
                    "type": "object",
                    "properties": {
                        "bm25": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "top_k": {"type": "integer", "minimum": 1},
                                "k1": {"type": "number", "minimum": 0},
                                "b": {"type": "number", "minimum": 0, "maximum": 1}
                            }
                        },
                        "dense": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "top_k": {"type": "integer", "minimum": 1},
                                "model_name": {"type": "string"},
                                "batch_size": {"type": "integer", "minimum": 1}
                            }
                        },
                        "graph": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "top_k": {"type": "integer", "minimum": 1},
                                "neo4j_uri": {"type": "string"},
                                "max_entities": {"type": "integer", "minimum": 1}
                            }
                        }
                    }
                },
                "fusion": {
                    "type": "object",
                    "required": ["method"],
                    "properties": {
                        "method": {"type": "string", "enum": ["linear", "rrf", "weighted", "dynamic"]},
                        "weights": {"type": "object"},
                        "dynamic_weight_config": {"type": "object"}
                    }
                },
                "evaluation": {
                    "type": "object",
                    "properties": {
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "cutoffs": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1}
                        }
                    }
                }
            }
        }
    
    def validate_config(self, config_path: str) -> Tuple[bool, List[str], List[str]]:
        """验证配置文件"""
        self.warnings.clear()
        self.errors.clear()
        
        try:
            # 读取配置文件
            config = self._load_config(config_path)
            if config is None:
                return False, self.errors, self.warnings
            
            # Schema验证
            self._validate_schema(config)
            
            # 业务逻辑验证
            self._validate_business_logic(config)
            
            # 文件路径验证
            self._validate_file_paths(config)
            
            # 参数合理性验证
            self._validate_parameters(config)
            
        except Exception as e:
            self.errors.append(f"配置验证失败: {e}")
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not Path(config_path).exists():
            self.errors.append(f"配置文件不存在: {config_path}")
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.json'):
                    return json.load(f)
                elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    self.errors.append(f"不支持的配置文件格式: {config_path}")
                    return None
        except Exception as e:
            self.errors.append(f"配置文件读取失败: {e}")
            return None
    
    def _validate_schema(self, config: Dict[str, Any]) -> None:
        """验证JSON Schema"""
        try:
            validate(instance=config, schema=self.schema)
        except ValidationError as e:
            self.errors.append(f"Schema验证失败: {e.message}")
    
    def _validate_business_logic(self, config: Dict[str, Any]) -> None:
        """验证业务逻辑"""
        # 检查至少启用一个检索器
        retrievers = config.get('retrievers', {})
        enabled_retrievers = [name for name, cfg in retrievers.items() 
                            if cfg.get('enabled', False)]
        
        if not enabled_retrievers:
            self.errors.append("至少需要启用一个检索器")
        
        # 检查融合方法与检索器数量的匹配
        fusion_method = config.get('fusion', {}).get('method', '')
        if fusion_method in ['weighted', 'dynamic'] and len(enabled_retrievers) < 2:
            self.warnings.append(f"融合方法 '{fusion_method}' 建议使用多个检索器")
        
        # 检查权重配置
        if fusion_method == 'weighted':
            weights = config.get('fusion', {}).get('weights', {})
            for retriever in enabled_retrievers:
                if retriever not in weights:
                    self.warnings.append(f"检索器 '{retriever}' 缺少权重配置")
    
    def _validate_file_paths(self, config: Dict[str, Any]) -> None:
        """验证文件路径"""
        data_config = config.get('data', {})
        
        required_files = {
            'corpus_path': '语料库文件',
            'queries_path': '查询文件',
            'qrels_path': '相关性标注文件'
        }
        
        for key, description in required_files.items():
            file_path = data_config.get(key)
            if file_path and not Path(file_path).exists():
                self.warnings.append(f"{description}不存在: {file_path}")
    
    def _validate_parameters(self, config: Dict[str, Any]) -> None:
        """验证参数合理性"""
        retrievers = config.get('retrievers', {})
        
        # BM25参数验证
        if 'bm25' in retrievers and retrievers['bm25'].get('enabled'):
            bm25_config = retrievers['bm25']
            k1 = bm25_config.get('k1', 1.2)
            b = bm25_config.get('b', 0.75)
            
            if k1 < 0.1 or k1 > 3.0:
                self.warnings.append(f"BM25 k1参数 ({k1}) 建议在0.1-3.0范围内")
            
            if b < 0.1 or b > 1.0:
                self.warnings.append(f"BM25 b参数 ({b}) 建议在0.1-1.0范围内")
        
        # Dense检索器参数验证
        if 'dense' in retrievers and retrievers['dense'].get('enabled'):
            dense_config = retrievers['dense']
            batch_size = dense_config.get('batch_size', 32)
            
            if batch_size > 128:
                self.warnings.append(f"Dense检索器批量大小 ({batch_size}) 可能过大")
        
        # Graph检索器参数验证
        if 'graph' in retrievers and retrievers['graph'].get('enabled'):
            graph_config = retrievers['graph']
            max_entities = graph_config.get('max_entities', 10)
            
            if max_entities > 50:
                self.warnings.append(f"图检索器最大实体数 ({max_entities}) 可能影响性能")
    
    def generate_report(self, config_path: str) -> str:
        """生成验证报告"""
        is_valid, errors, warnings = self.validate_config(config_path)
        
        report = f"# 配置验证报告\n\n"
        report += f"**配置文件**: {config_path}\n"
        report += f"**验证状态**: {'✅ 通过' if is_valid else '❌ 失败'}\n"
        report += f"**错误数量**: {len(errors)}\n"
        report += f"**警告数量**: {len(warnings)}\n\n"
        
        if errors:
            report += "## ❌ 错误\n\n"
            for i, error in enumerate(errors, 1):
                report += f"{i}. {error}\n"
            report += "\n"
        
        if warnings:
            report += "## ⚠️ 警告\n\n"
            for i, warning in enumerate(warnings, 1):
                report += f"{i}. {warning}\n"
            report += "\n"
        
        if is_valid and not warnings:
            report += "✨ 配置文件完全正确！\n"
        
        return report


def validate_config_files(config_dir: str) -> None:
    """批量验证配置文件"""
    validator = ConfigValidator()
    config_dir = Path(config_dir)
    
    print(f"🔍 扫描配置目录: {config_dir}")
    
    # 查找所有配置文件
    config_files = []
    for pattern in ['*.yaml', '*.yml', '*.json']:
        config_files.extend(config_dir.rglob(pattern))
    
    if not config_files:
        print("❌ 未找到配置文件")
        return
    
    print(f"📁 找到 {len(config_files)} 个配置文件")
    print("=" * 60)
    
    results = []
    
    for config_file in config_files:
        print(f"\n🔧 验证: {config_file.relative_to(config_dir)}")
        
        is_valid, errors, warnings = validator.validate_config(str(config_file))
        
        results.append({
            'file': config_file,
            'valid': is_valid,
            'errors': len(errors),
            'warnings': len(warnings)
        })
        
        if is_valid:
            print(f"  ✅ 通过 (警告: {len(warnings)})")
        else:
            print(f"  ❌ 失败 (错误: {len(errors)}, 警告: {len(warnings)})")
            
        # 显示前3个错误
        for error in errors[:3]:
            print(f"    ❌ {error}")
        
        # 显示前3个警告
        for warning in warnings[:3]:
            print(f"    ⚠️ {warning}")
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 验证总结")
    print("=" * 60)
    
    valid_count = sum(1 for r in results if r['valid'])
    total_errors = sum(r['errors'] for r in results)
    total_warnings = sum(r['warnings'] for r in results)
    
    print(f"✅ 通过验证: {valid_count}/{len(results)} 个文件")
    print(f"❌ 总错误数: {total_errors}")
    print(f"⚠️ 总警告数: {total_warnings}")
    
    if valid_count == len(results) and total_warnings == 0:
        print("\n🎉 所有配置文件都完全正确！")
    elif valid_count == len(results):
        print("\n👍 所有配置文件都通过了验证，但有一些警告需要注意。")
    else:
        print(f"\n⚠️ 有 {len(results) - valid_count} 个配置文件存在错误，需要修复。")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FusionRAG配置验证工具")
    parser.add_argument('config', help='配置文件路径或目录路径')
    parser.add_argument('--report', '-r', help='生成详细报告文件')
    parser.add_argument('--batch', '-b', action='store_true', help='批量验证目录中的所有配置文件')
    
    args = parser.parse_args()
    
    print("🔧 FusionRAG配置验证工具")
    print("=" * 50)
    
    if args.batch or Path(args.config).is_dir():
        # 批量验证
        validate_config_files(args.config)
    else:
        # 单文件验证
        validator = ConfigValidator()
        
        print(f"📄 验证配置文件: {args.config}")
        
        is_valid, errors, warnings = validator.validate_config(args.config)
        
        # 生成报告
        report = validator.generate_report(args.config)
        print(report)
        
        # 保存报告
        if args.report:
            with open(args.report, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 报告已保存: {args.report}")
        
        # 返回退出码
        exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()