#!/usr/bin/env python
"""
测试运行脚本
提供便捷的测试执行接口
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def list_configs():
    """列出可用的配置文件"""
    config_dir = Path("configs")
    configs = list(config_dir.glob("*.yaml"))
    configs.sort()
    
    print("📋 可用配置文件:")
    print("-" * 80)
    print(f"{'序号':<4} {'文件名':<40} {'数据集':<15} {'模板':<15}")
    print("-" * 80)
    
    for i, config_file in enumerate(configs, 1):
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            metadata = config.get('metadata', {})
            dataset = metadata.get('dataset', 'unknown')
            template = metadata.get('template', 'unknown')
            
            print(f"{i:<4} {config_file.name:<40} {dataset:<15} {template:<15}")
            
        except Exception:
            print(f"{i:<4} {config_file.name:<40} {'error':<15} {'error':<15}")

def run_test(config_path: str, compare_with: str = None, auto_download: bool = True):
    """运行测试"""
    cmd_parts = [
        "python", "tests/universal_test.py",
        "--config", config_path
    ]
    
    if not auto_download:
        cmd_parts.append("--no-auto-download")
    
    if compare_with:
        cmd_parts.append("--compare-with")
        cmd_parts.append(compare_with)
    
    cmd = " ".join(cmd_parts)
    print(f"🚀 执行命令: {cmd}")
    print("-" * 60)
    
    os.system(cmd)

def run_quick_test(config_path: str):
    """运行快速测试"""
    cmd = f"python scripts/test_config.py {config_path} --quick-test"
    print(f"⚡ 快速测试: {cmd}")
    print("-" * 60)
    
    os.system(cmd)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FusionRAG测试运行工具")
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 列出配置
    subparsers.add_parser('list', help='列出所有配置文件')
    
    # 运行完整测试
    test_parser = subparsers.add_parser('test', help='运行完整测试')
    test_parser.add_argument('config', help='配置文件路径或序号')
    test_parser.add_argument('--compare-with', help='与指定结果文件对比')
    test_parser.add_argument('--no-auto-download', action='store_true', help='不自动下载数据')
    
    # 运行快速测试
    quick_parser = subparsers.add_parser('quick', help='运行快速测试')
    quick_parser.add_argument('config', help='配置文件路径或序号')
    
    # 比较结果
    compare_parser = subparsers.add_parser('compare', help='比较两个测试结果')
    compare_parser.add_argument('result1', help='结果文件1')
    compare_parser.add_argument('result2', help='结果文件2')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_configs()
    
    elif args.command == 'test':
        config_path = args.config
        
        # 如果是数字，转换为配置文件路径
        if config_path.isdigit():
            config_dir = Path("configs")
            configs = sorted(list(config_dir.glob("*.yaml")))
            index = int(config_path) - 1
            if 0 <= index < len(configs):
                config_path = str(configs[index])
            else:
                print(f"❌ 无效序号: {config_path}")
                return
        
        if not Path(config_path).exists():
            print(f"❌ 配置文件不存在: {config_path}")
            return
        
        run_test(config_path, args.compare_with, not args.no_auto_download)
    
    elif args.command == 'quick':
        config_path = args.config
        
        # 如果是数字，转换为配置文件路径
        if config_path.isdigit():
            config_dir = Path("configs")
            configs = sorted(list(config_dir.glob("*.yaml")))
            index = int(config_path) - 1
            if 0 <= index < len(configs):
                config_path = str(configs[index])
            else:
                print(f"❌ 无效序号: {config_path}")
                return
        
        if not Path(config_path).exists():
            print(f"❌ 配置文件不存在: {config_path}")
            return
        
        run_quick_test(config_path)
    
    elif args.command == 'compare':
        if not Path(args.result1).exists():
            print(f"❌ 结果文件1不存在: {args.result1}")
            return
        if not Path(args.result2).exists():
            print(f"❌ 结果文件2不存在: {args.result2}")
            return
        
        # 调用比较功能
        from tests.universal_test import compare_results
        compare_results(args.result1, args.result2)
    
    else:
        print("🎯 FusionRAG测试运行工具")
        print("=" * 50)
        print("\n可用命令:")
        print("  list                    - 列出所有配置文件")
        print("  test <config>           - 运行完整测试")
        print("  quick <config>          - 运行快速测试")
        print("  compare <result1> <result2> - 比较测试结果")
        print("\n示例:")
        print("  python scripts/run_test.py list")
        print("  python scripts/run_test.py test 1")
        print("  python scripts/run_test.py test configs/20250713_1506_nfcorpus_high_performance.yaml")
        print("  python scripts/run_test.py quick 2")

if __name__ == "__main__":
    main()
