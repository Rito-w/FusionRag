#!/usr/bin/env python3
"""
FusionRAG项目初始化脚本
自动化环境设置、依赖安装和基础配置
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path
import argparse
import json
import yaml


class FusionRAGInitializer:
    """FusionRAG项目初始化器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.venv_path = self.project_root / "venv"
        self.config_created = False
        
    def check_python_version(self) -> bool:
        """检查Python版本"""
        print("🐍 检查Python版本...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"  ❌ Python版本过低: {version.major}.{version.minor}")
            print("  请安装Python 3.8或更高版本")
            return False
        
        print(f"  ✅ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def create_virtual_environment(self) -> bool:
        """创建虚拟环境"""
        print("📦 创建虚拟环境...")
        
        if self.venv_path.exists():
            print(f"  ⚠️ 虚拟环境已存在: {self.venv_path}")
            return True
        
        try:
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], 
                         check=True, capture_output=True)
            print(f"  ✅ 虚拟环境创建成功: {self.venv_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 虚拟环境创建失败: {e}")
            return False
    
    def get_venv_python(self) -> str:
        """获取虚拟环境Python路径"""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / "Scripts" / "python.exe")
        else:  # Unix/Linux/macOS
            return str(self.venv_path / "bin" / "python")
    
    def get_venv_pip(self) -> str:
        """获取虚拟环境pip路径"""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / "Scripts" / "pip.exe")
        else:  # Unix/Linux/macOS
            return str(self.venv_path / "bin" / "pip")
    
    def install_dependencies(self, dev: bool = False) -> bool:
        """安装项目依赖"""
        print("📚 安装项目依赖...")
        
        if not self.venv_path.exists():
            print("  ❌ 虚拟环境不存在")
            return False
        
        pip_path = self.get_venv_pip()
        
        # 升级pip
        try:
            subprocess.run([pip_path, "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            print("  ✅ pip已升级")
        except subprocess.CalledProcessError:
            print("  ⚠️ pip升级失败")
        
        # 安装基础依赖
        requirements_file = "requirements.txt"
        if not (self.project_root / requirements_file).exists():
            print(f"  ❌ 依赖文件不存在: {requirements_file}")
            return False
        
        try:
            print("  📦 安装基础依赖...")
            subprocess.run([pip_path, "install", "-r", requirements_file], 
                         check=True, cwd=self.project_root)
            print("  ✅ 基础依赖安装完成")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 依赖安装失败: {e}")
            return False
        
        # 安装开发依赖
        if dev:
            dev_requirements = "requirements-dev.txt"
            if (self.project_root / dev_requirements).exists():
                try:
                    print("  📦 安装开发依赖...")
                    subprocess.run([pip_path, "install", "-r", dev_requirements], 
                                 check=True, cwd=self.project_root)
                    print("  ✅ 开发依赖安装完成")
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ 开发依赖安装失败: {e}")
                    return False
        
        return True
    
    def download_nltk_data(self) -> bool:
        """下载NLTK数据"""
        print("📝 下载NLTK数据...")
        
        python_path = self.get_venv_python()
        
        nltk_script = '''
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print("NLTK数据下载完成")
except Exception as e:
    print(f"NLTK数据下载失败: {e}")
'''
        
        try:
            result = subprocess.run([python_path, "-c", nltk_script], 
                                  capture_output=True, text=True)
            if "下载完成" in result.stdout:
                print("  ✅ NLTK数据下载完成")
            else:
                print("  ⚠️ NLTK数据下载可能失败")
            return True
        except subprocess.CalledProcessError:
            print("  ⚠️ NLTK数据下载失败")
            return False
    
    def setup_spacy_model(self) -> bool:
        """设置spaCy模型"""
        print("🧠 设置spaCy模型...")
        
        python_path = self.get_venv_python()
        
        try:
            # 下载英文模型
            subprocess.run([python_path, "-m", "spacy", "download", "en_core_web_sm"], 
                         check=True, capture_output=True)
            print("  ✅ spaCy英文模型下载完成")
            return True
        except subprocess.CalledProcessError:
            print("  ⚠️ spaCy模型下载失败，可能需要手动安装")
            print("    运行: python -m spacy download en_core_web_sm")
            return False
    
    def create_directories(self) -> bool:
        """创建必要的目录结构"""
        print("📁 创建目录结构...")
        
        directories = [
            "data/raw",
            "data/processed", 
            "checkpoints/retriever",
            "checkpoints/embeddings_cache",
            "checkpoints/logs",
            "logs/performance",
            "reports",
            "docs"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"  ✅ 创建了 {len(directories)} 个目录")
        return True
    
    def create_default_config(self) -> bool:
        """创建默认配置文件"""
        print("⚙️ 创建默认配置...")
        
        config_path = self.project_root / "configs" / "config.yaml"
        
        if config_path.exists():
            print("  ⚠️ 默认配置文件已存在")
            return True
        
        default_config = {
            "metadata": {
                "dataset": "demo",
                "template": "default",
                "description": "FusionRAG默认配置",
                "version": "1.0"
            },
            "data": {
                "corpus_path": "data/processed/demo_corpus.jsonl",
                "queries_path": "data/processed/demo_queries.jsonl", 
                "qrels_path": "data/processed/demo_qrels.tsv"
            },
            "retrievers": {
                "bm25": {
                    "enabled": True,
                    "top_k": 10,
                    "k1": 1.2,
                    "b": 0.75
                },
                "dense": {
                    "enabled": True,
                    "top_k": 10,
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "batch_size": 32
                },
                "graph": {
                    "enabled": False,
                    "top_k": 10,
                    "neo4j_uri": "bolt://localhost:7687",
                    "max_entities": 10
                }
            },
            "fusion": {
                "method": "linear",
                "weights": {
                    "bm25": 0.5,
                    "dense": 0.5,
                    "graph": 0.0
                }
            },
            "evaluation": {
                "metrics": ["recall", "ndcg", "map"],
                "cutoffs": [5, 10, 20]
            },
            "system": {
                "log_path": "checkpoints/logs"
            }
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"  ✅ 默认配置已创建: {config_path}")
        self.config_created = True
        return True
    
    def create_gitignore(self) -> bool:
        """创建.gitignore文件"""
        print("📝 创建.gitignore文件...")
        
        gitignore_path = self.project_root / ".gitignore"
        
        if gitignore_path.exists():
            print("  ⚠️ .gitignore文件已存在")
            return True
        
        gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# FusionRAG specific
# 数据文件
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# 缓存文件
checkpoints/embeddings_cache/*.pkl
checkpoints/retriever/*.faiss
checkpoints/retriever/*.pkl

# 日志文件
logs/
checkpoints/logs/

# 模型文件
*.bin
*.safetensors

# 临时文件
*.tmp
*.swp
*~

# IDE文件
.vscode/
.idea/
*.sublime-*

# macOS
.DS_Store

# Windows
Thumbs.db
ehthumbs.db
Desktop.ini
'''
        
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        
        print(f"  ✅ .gitignore文件已创建")
        return True
    
    def run_quick_test(self) -> bool:
        """运行快速测试验证安装"""
        print("🧪 运行快速验证测试...")
        
        python_path = self.get_venv_python()
        
        # 测试脚本
        test_script = '''
import sys
print(f"Python版本: {sys.version}")

# 测试核心依赖
try:
    import numpy
    print(f"✅ numpy {numpy.__version__}")
except ImportError:
    print("❌ numpy未安装")

try:
    import pandas
    print(f"✅ pandas {pandas.__version__}")
except ImportError:
    print("❌ pandas未安装")

try:
    import sklearn
    print(f"✅ scikit-learn {sklearn.__version__}")
except ImportError:
    print("❌ scikit-learn未安装")

try:
    import sentence_transformers
    print(f"✅ sentence-transformers {sentence_transformers.__version__}")
except ImportError:
    print("❌ sentence-transformers未安装")

try:
    import jieba
    print("✅ jieba已安装")
except ImportError:
    print("❌ jieba未安装")

print("\\n🎉 依赖验证完成!")
'''
        
        try:
            result = subprocess.run([python_path, "-c", test_script], 
                                  capture_output=True, text=True, cwd=self.project_root)
            print("  测试结果:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"    {line}")
            
            if result.returncode == 0:
                print("  ✅ 快速测试通过")
                return True
            else:
                print("  ❌ 快速测试失败")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 快速测试执行失败: {e}")
            return False
    
    def print_next_steps(self) -> None:
        """打印后续步骤提示"""
        print("\n" + "="*60)
        print("🎉 FusionRAG项目初始化完成!")
        print("="*60)
        
        if os.name == 'nt':  # Windows
            activate_cmd = f"{self.venv_path}\\Scripts\\activate"
        else:  # Unix/Linux/macOS
            activate_cmd = f"source {self.venv_path}/bin/activate"
        
        print("\n📋 后续步骤:")
        print(f"1. 激活虚拟环境:")
        print(f"   {activate_cmd}")
        print()
        print("2. 下载数据集 (可选):")
        print("   python scripts/download_data.py --dataset nfcorpus")
        print()
        print("3. 运行快速测试:")
        print("   python examples/quick_test.py")
        print()
        print("4. 运行完整测试:")
        if self.config_created:
            print("   python tests/universal_test.py --config configs/config.yaml")
        else:
            print("   python tests/universal_test.py --config configs/templates/high_performance_template.yaml")
        print()
        print("5. 查看项目帮助:")
        print("   make help  # 或查看Makefile")
        print()
        print("🔧 开发工具:")
        print("- 代码格式化: make format")
        print("- 代码检查: make lint")
        print("- 运行测试: make test")
        print("- 性能测试: make perf-test")
        
    def initialize(self, dev: bool = False, skip_models: bool = False) -> bool:
        """完整初始化流程"""
        print("🚀 FusionRAG项目初始化开始")
        print("="*50)
        
        steps = [
            ("检查Python版本", self.check_python_version),
            ("创建虚拟环境", self.create_virtual_environment),
            ("安装依赖", lambda: self.install_dependencies(dev)),
            ("创建目录结构", self.create_directories),
            ("创建默认配置", self.create_default_config),
            ("创建.gitignore", self.create_gitignore),
        ]
        
        if not skip_models:
            steps.extend([
                ("下载NLTK数据", self.download_nltk_data),
                ("设置spaCy模型", self.setup_spacy_model),
            ])
        
        steps.append(("运行快速测试", self.run_quick_test))
        
        failed_steps = []
        
        for step_name, step_func in steps:
            try:
                if not step_func():
                    failed_steps.append(step_name)
            except Exception as e:
                print(f"  ❌ {step_name}执行出错: {e}")
                failed_steps.append(step_name)
        
        if failed_steps:
            print(f"\n⚠️ 以下步骤失败: {', '.join(failed_steps)}")
            print("请检查错误信息并手动完成这些步骤。")
            return False
        
        self.print_next_steps()
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FusionRAG项目初始化工具")
    parser.add_argument('--dev', action='store_true', help='安装开发依赖')
    parser.add_argument('--skip-models', action='store_true', help='跳过模型下载')
    parser.add_argument('--project-root', '-p', default='.', help='项目根目录')
    
    args = parser.parse_args()
    
    # 创建初始化器
    initializer = FusionRAGInitializer(args.project_root)
    
    # 运行初始化
    success = initializer.initialize(dev=args.dev, skip_models=args.skip_models)
    
    # 设置退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()