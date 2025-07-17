# FusionRAG项目Makefile
# 提供常见开发任务的快捷命令

.PHONY: help install install-dev test test-coverage lint format clean build docs

# 默认目标
help:
	@echo "FusionRAG项目开发工具"
	@echo "可用命令："
	@echo "  install      - 安装生产环境依赖"
	@echo "  install-dev  - 安装开发环境依赖"
	@echo "  test         - 运行测试"
	@echo "  test-coverage - 运行测试并生成覆盖率报告"
	@echo "  lint         - 运行代码质量检查"
	@echo "  format       - 格式化代码"
	@echo "  clean        - 清理缓存和临时文件"
	@echo "  build        - 构建项目"
	@echo "  docs         - 生成文档"
	@echo "  quick-test   - 快速健康检查"

# 安装依赖
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# 测试
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=modules --cov-report=html --cov-report=term

# 代码质量
lint:
	flake8 modules/ tests/ pipeline.py
	mypy modules/ --ignore-missing-imports

format:
	black modules/ tests/ pipeline.py
	isort modules/ tests/ pipeline.py

# 清理
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/

# 构建
build:
	python setup.py sdist bdist_wheel

# 文档
docs:
	cd docs && make html

# 快速测试
quick-test:
	python examples/quick_test.py

# 性能测试
perf-test:
	python tests/universal_test.py --config configs/templates/high_performance_template.yaml

# 完整的NFCorpus测试
nfcorpus-test:
	python tests/universal_test.py --config configs/datasets/nfcorpus/high_performance.yaml

# 多数据集测试
multi-dataset-test:
	python tests/universal_test.py --config configs/datasets/nfcorpus/high_performance.yaml
	python tests/universal_test.py --config configs/datasets/trec-covid/high_performance.yaml

# 安全检查
security-check:
	bandit -r modules/ -f json -o security_report.json

# 内存分析
memory-profile:
	python -m memory_profiler tests/universal_test.py --config configs/datasets/nfcorpus/baseline.yaml

# 代码复杂度分析
complexity-check:
	radon cc modules/ --min B

# 依赖检查
deps-check:
	pip-audit

# 全套检查
check-all: lint test security-check
	@echo "✅ 所有检查完成"

# 发布准备
release-prep: clean format lint test build
	@echo "🚀 项目准备就绪，可以发布"