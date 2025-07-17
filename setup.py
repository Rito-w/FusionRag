#!/usr/bin/env python3
"""
FusionRAG项目安装脚本
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# 读取requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="fusionrag",
    version="0.1.0",
    description="多路径检索增强生成系统",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="FusionRAG Team",
    author_email="your-email@example.com",
    url="https://github.com/your-username/fusionrag",
    
    # 包配置
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    # 可选依赖
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.4.1",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
            "torch>=2.0.1+cu118",
        ],
        "docs": [
            "sphinx>=7.0.1",
            "sphinx-rtd-theme>=1.2.2",
        ],
    },
    
    # 分类信息
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Indexing",
    ],
    
    # 关键词
    keywords="information retrieval, RAG, fusion, machine learning, NLP",
    
    # 包含的数据文件
    package_data={
        "fusionrag": ["configs/templates/*.yaml", "configs/templates/*.json"],
    },
    
    # 命令行脚本
    entry_points={
        "console_scripts": [
            "fusionrag-test=tests.universal_test:main",
            "fusionrag-demo=examples.quick_test:main",
        ],
    },
    
    # 依赖链接
    dependency_links=[],
    
    # 项目URL
    project_urls={
        "Bug Reports": "https://github.com/your-username/fusionrag/issues",
        "Source": "https://github.com/your-username/fusionrag",
        "Documentation": "https://fusionrag.readthedocs.io/",
    },
)