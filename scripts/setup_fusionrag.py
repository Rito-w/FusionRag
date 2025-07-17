#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FusionRAG 安装脚本

该脚本用于安装FusionRAG系统所需的所有依赖，并设置必要的环境。
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path


def print_step(step, message):
    """打印安装步骤"""
    print(f"\n[{step}] {message}")
    print("-" * 50)


def run_command(command, description=None, exit_on_error=True):
    """运行shell命令"""
    if description:
        print(f"  {description}...")
    
    try:
        process = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"\n错误: {e}")
        print(f"命令: {command}")
        print(f"输出: {e.stdout}")
        print(f"错误: {e.stderr}")
        
        if exit_on_error:
            print("\n安装失败，请查看上述错误信息。")
            sys.exit(1)
        return None


def check_python_version():
    """检查Python版本"""
    print_step("1", "检查Python版本")
    
    python_version = sys.version_info
    min_version = (3, 8)
    
    if python_version < min_version:
        print(f"错误: 需要Python {min_version[0]}.{min_version[1]}或更高版本")
        print(f"当前版本: {python_version[0]}.{python_version[1]}.{python_version[2]}")
        sys.exit(1)
    
    print(f"  Python版本: {python_version[0]}.{python_version[1]}.{python_version[2]}")
    print("  版本检查通过 ✓")


def setup_virtual_environment(force_recreate=False):
    """设置虚拟环境"""
    print_step("2", "设置虚拟环境")
    
    venv_path = "venv"
    
    # 检查虚拟环境是否已存在
    if os.path.exists(venv_path) and not force_recreate:
        print(f"  虚拟环境已存在: {venv_path}")
        print("  跳过创建步骤 ✓")
        return venv_path
    
    # 如果需要重新创建，先删除现有环境
    if os.path.exists(venv_path) and force_recreate:
        print("  删除现有虚拟环境...")
        if platform.system() == "Windows":
            run_command(f"rmdir /s /q {venv_path}")
        else:
            run_command(f"rm -rf {venv_path}")
    
    # 创建虚拟环境
    print("  创建新的虚拟环境...")
    run_command(f"{sys.executable} -m venv {venv_path}")
    
    print("  虚拟环境创建成功 ✓")
    return venv_path


def install_dependencies(venv_path, dev_mode=False):
    """安装依赖包"""
    print_step("3", "安装依赖包")
    
    # 确定pip路径
    if platform.system() == "Windows":
        pip_path = os.path.join(venv_path, "Scripts", "pip")
    else:
        pip_path = os.path.join(venv_path, "bin", "pip")
    
    # 升级pip
    run_command(f"{pip_path} install --upgrade pip", "升级pip")
    
    # 安装依赖
    requirements_file = "requirements-dev.txt" if dev_mode else "requirements.txt"
    
    # 检查requirements文件是否存在
    if not os.path.exists(requirements_file):
        # 如果不存在，创建基本的requirements文件
        print(f"  未找到{requirements_file}，创建基本依赖文件...")
        
        with open(requirements_file, "w") as f:
            f.write("# FusionRAG依赖\n")
            f.write("torch>=1.10.0\n")
            f.write("transformers>=4.15.0\n")
            f.write("sentence-transformers>=2.2.0\n")
            f.write("faiss-cpu>=1.7.0\n")
            f.write("numpy>=1.20.0\n")
            f.write("scikit-learn>=1.0.0\n")
            f.write("rank_bm25>=0.2.2\n")
            f.write("pyyaml>=6.0\n")
            f.write("tqdm>=4.62.0\n")
            f.write("matplotlib>=3.5.0\n")
            f.write("pandas>=1.3.0\n")
            
            if dev_mode:
                f.write("# 开发依赖\n")
                f.write("pytest>=7.0.0\n")
                f.write("black>=22.0.0\n")
                f.write("flake8>=4.0.0\n")
                f.write("isort>=5.10.0\n")
                f.write("mypy>=0.9.0\n")
    
    # 安装依赖
    run_command(f"{pip_path} install -r {requirements_file}", f"安装{requirements_file}中的依赖")
    
    # 安装额外依赖
    if dev_mode:
        print("  安装开发工具...")
        run_command(f"{pip_path} install pytest black flake8 isort mypy")
    
    print("  依赖安装完成 ✓")


def create_directory_structure():
    """创建目录结构"""
    print_step("4", "创建目录结构")
    
    directories = [
        "data/raw",
        "data/processed",
        "checkpoints/retriever",
        "checkpoints/classifier",
        "checkpoints/logs",
        "checkpoints/temp",
        "reports/evaluation",
        "reports/analysis"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  创建目录: {directory} ✓")
    
    print("  目录结构创建完成 ✓")


def download_test_data():
    """下载测试数据"""
    print_step("5", "下载测试数据")
    
    # 检查是否已有数据下载脚本
    if os.path.exists("scripts/download_data.py"):
        print("  使用现有数据下载脚本...")
        run_command("python scripts/download_data.py --dataset nfcorpus", "下载NFCorpus数据集")
    else:
        print("  创建示例数据...")
        
        # 创建示例文档
        os.makedirs("data/processed", exist_ok=True)
        
        # 创建示例语料库
        with open("data/processed/corpus.jsonl", "w") as f:
            f.write('{"doc_id": "doc1", "title": "机器学习简介", "text": "机器学习是人工智能的一个分支，它使用统计学方法让计算机系统能够从数据中学习和改进。"}\n')
            f.write('{"doc_id": "doc2", "title": "深度学习技术", "text": "深度学习是机器学习的一个子领域，它使用多层神经网络来模拟人脑的学习过程，已在图像识别、自然语言处理等领域取得突破。"}\n')
            f.write('{"doc_id": "doc3", "title": "自然语言处理", "text": "自然语言处理(NLP)是计算机科学和人工智能的一个领域，专注于使计算机能够理解、解释和生成人类语言。"}\n')
            f.write('{"doc_id": "doc4", "title": "计算机视觉", "text": "计算机视觉是一个跨学科领域，研究如何使计算机能够从图像或视频中获取高层次的理解，模拟人类视觉系统的功能。"}\n')
            f.write('{"doc_id": "doc5", "title": "强化学习", "text": "强化学习是机器学习的一种方法，它通过让智能体在环境中采取行动并获得奖励或惩罚来学习最优策略。"}\n')
        
        # 创建示例查询
        with open("data/processed/queries.jsonl", "w") as f:
            f.write('{"query_id": "q1", "text": "什么是机器学习？"}\n')
            f.write('{"query_id": "q2", "text": "深度学习和传统机器学习有什么区别？"}\n')
            f.write('{"query_id": "q3", "text": "计算机如何理解人类语言？"}\n')
        
        # 创建示例相关性标注
        with open("data/processed/qrels.tsv", "w") as f:
            f.write("q1\tdoc1\t1\n")
            f.write("q2\tdoc2\t1\n")
            f.write("q3\tdoc3\t1\n")
    
    print("  测试数据准备完成 ✓")


def run_quick_test():
    """运行快速测试"""
    print_step("6", "运行快速测试")
    
    # 检查测试脚本是否存在
    if os.path.exists("examples/test_fusionrag.py"):
        print("  运行FusionRAG测试脚本...")
        run_command("python examples/test_fusionrag.py", "测试FusionRAG系统", exit_on_error=False)
    else:
        print("  未找到测试脚本，跳过测试步骤")
    
    print("  快速测试完成 ✓")


def print_completion_message():
    """打印完成信息"""
    print("\n" + "=" * 50)
    print("FusionRAG系统安装完成！")
    print("=" * 50)
    print("\n您可以通过以下方式开始使用FusionRAG：")
    print("\n1. 激活虚拟环境:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. 运行测试脚本:")
    print("   python examples/test_fusionrag.py")
    
    print("\n3. 使用命令行工具:")
    print("   python fusionrag.py retrieve --query \"你的查询文本\"")
    
    print("\n4. 查看文档:")
    print("   docs/new_components.md")
    
    print("\n祝您使用愉快！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FusionRAG系统安装脚本")
    parser.add_argument("--dev", action="store_true", help="安装开发依赖")
    parser.add_argument("--force", action="store_true", help="强制重新创建虚拟环境")
    parser.add_argument("--skip-test", action="store_true", help="跳过测试步骤")
    args = parser.parse_args()
    
    # 记录开始时间
    import time
    start_time = time.time()
    
    print("\n========== FusionRAG系统安装 ==========\n")
    
    # 检查Python版本
    check_python_version()
    
    # 设置虚拟环境
    venv_path = setup_virtual_environment(force_recreate=args.force)
    
    # 安装依赖
    install_dependencies(venv_path, dev_mode=args.dev)
    
    # 创建目录结构
    create_directory_structure()
    
    # 下载测试数据
    download_test_data()
    
    # 运行快速测试
    if not args.skip_test:
        run_quick_test()
    
    # 计算耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print(f"\n安装耗时: {int(minutes)}分{int(seconds)}秒")
    
    # 打印完成信息
    print_completion_message()


if __name__ == "__main__":
    main()