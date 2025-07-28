#!/usr/bin/env python3
"""
Paper目录整理工具

使用MCP帮助整理混乱的paper目录结构
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import hashlib
import re
from collections import defaultdict

class PaperDirectoryOrganizer:
    """Paper目录整理器"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.backup_dir = self.base_dir / "_backup"
        
        # 定义目录结构
        self.categories = {
            "01_hybrid_retrieval": ["hybrid", "blend", "fusion", "mix", "combine"],
            "02_multimodal_retrieval": ["multimodal", "vision", "image", "visual", "mm-"],
            "03_vector_indexing": ["vector", "embedding", "index", "faiss", "dense"],
            "04_query_understanding": ["query", "question", "understanding", "rewrite"],
            "05_reranking": ["rerank", "rank", "scoring", "relevance"],
            "06_evaluation": ["evaluation", "benchmark", "metric", "assess"],
            "07_core_papers": ["rag", "retrieval-augmented", "realm", "retro"],
            "08_surveys": ["survey", "review", "comprehensive", "overview"],
            "09_applications": ["application", "domain", "specific", "bio", "legal"],
            "10_tools_and_scripts": [],  # Python脚本
            "11_analysis_results": [],   # 分析结果
            "12_logs_and_configs": []    # 日志和配置
        }
        
        # 文件类型映射
        self.file_type_mapping = {
            ".pdf": "papers",
            ".py": "10_tools_and_scripts",
            ".md": "11_analysis_results",
            ".json": "12_logs_and_configs",
            ".txt": "12_logs_and_configs"
        }
    
    def create_backup(self):
        """创建备份"""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        print("📦 创建备份...")
        self.backup_dir.mkdir()
        
        # 备份根目录的重要文件
        for item in self.base_dir.iterdir():
            if item.name.startswith("_"):
                continue
            if item.is_file():
                shutil.copy2(item, self.backup_dir / item.name)
            elif item.is_dir() and not item.name.startswith(("01_", "02_", "03_")):
                shutil.copytree(item, self.backup_dir / item.name)
        
        print(f"✅ 备份完成: {self.backup_dir}")
    
    def analyze_current_structure(self) -> Dict:
        """分析当前目录结构"""
        print("🔍 分析当前目录结构...")
        
        analysis = {
            "total_files": 0,
            "pdf_files": 0,
            "duplicate_files": [],
            "misplaced_files": [],
            "file_distribution": defaultdict(int),
            "file_hashes": defaultdict(list)
        }
        
        # 遍历所有文件
        for root, dirs, files in os.walk(self.base_dir):
            root_path = Path(root)
            
            # 跳过备份目录
            if "_backup" in root_path.parts:
                continue
            
            for file in files:
                file_path = root_path / file
                analysis["total_files"] += 1
                
                # 统计文件类型
                suffix = file_path.suffix.lower()
                analysis["file_distribution"][suffix] += 1
                
                if suffix == ".pdf":
                    analysis["pdf_files"] += 1
                
                # 检查重复文件
                try:
                    file_hash = self.get_file_hash(file_path)
                    analysis["file_hashes"][file_hash].append(str(file_path))
                except:
                    pass
        
        # 找出重复文件
        for file_hash, paths in analysis["file_hashes"].items():
            if len(paths) > 1:
                analysis["duplicate_files"].append(paths)
        
        return analysis
    
    def get_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return ""
    
    def classify_paper(self, filename: str) -> str:
        """根据文件名分类论文"""
        filename_lower = filename.lower()
        
        # 检查每个分类的关键词
        for category, keywords in self.categories.items():
            if category.startswith("1"):  # 跳过工具目录
                continue
            
            for keyword in keywords:
                if keyword in filename_lower:
                    return category
        
        # 默认分类
        if "rag" in filename_lower or "retrieval" in filename_lower:
            return "07_core_papers"
        else:
            return "09_applications"
    
    def clean_filename(self, filename: str) -> str:
        """清理文件名"""
        # 移除特殊字符
        cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # 限制长度
        name, ext = os.path.splitext(cleaned)
        if len(name) > 100:
            name = name[:100]
        
        return name + ext
    
    def create_directory_structure(self):
        """创建目录结构"""
        print("📁 创建目录结构...")
        
        for category in self.categories.keys():
            category_dir = self.base_dir / category
            category_dir.mkdir(exist_ok=True)
            
            # 创建README文件
            readme_file = category_dir / "README.md"
            if not readme_file.exists():
                with open(readme_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {category}\n\n")
                    if category == "10_tools_and_scripts":
                        f.write("Python脚本和工具\n")
                    elif category == "11_analysis_results":
                        f.write("分析结果和总结文档\n")
                    elif category == "12_logs_and_configs":
                        f.write("日志文件和配置文件\n")
                    else:
                        f.write(f"相关论文和文档\n")
        
        print("✅ 目录结构创建完成")
    
    def organize_files(self, dry_run: bool = True):
        """整理文件"""
        print(f"🗂️ 开始整理文件 (dry_run={dry_run})...")
        
        moved_files = []
        
        # 处理根目录的文件
        for item in self.base_dir.iterdir():
            if item.name.startswith(("_", "01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_", "09_", "10_", "11_", "12_")):
                continue
            
            if item.is_file():
                target_category = self.determine_target_category(item)
                target_dir = self.base_dir / target_category
                
                # 清理文件名
                clean_name = self.clean_filename(item.name)
                target_path = target_dir / clean_name
                
                # 避免覆盖
                counter = 1
                while target_path.exists():
                    name, ext = os.path.splitext(clean_name)
                    target_path = target_dir / f"{name}_{counter}{ext}"
                    counter += 1
                
                moved_files.append({
                    "source": str(item),
                    "target": str(target_path),
                    "category": target_category
                })
                
                if not dry_run:
                    shutil.move(str(item), str(target_path))
                    print(f"   移动: {item.name} -> {target_category}/")
        
        return moved_files
    
    def determine_target_category(self, file_path: Path) -> str:
        """确定文件的目标分类"""
        suffix = file_path.suffix.lower()
        filename = file_path.name
        
        # 根据文件类型
        if suffix == ".py":
            return "10_tools_and_scripts"
        elif suffix == ".md" and any(word in filename.lower() for word in ["analysis", "summary", "report", "result"]):
            return "11_analysis_results"
        elif suffix in [".json", ".txt", ".log"]:
            return "12_logs_and_configs"
        elif suffix == ".pdf":
            return self.classify_paper(filename)
        else:
            return "12_logs_and_configs"
    
    def remove_duplicates(self, dry_run: bool = True):
        """移除重复文件"""
        print(f"🔄 移除重复文件 (dry_run={dry_run})...")
        
        analysis = self.analyze_current_structure()
        removed_files = []
        
        for duplicate_group in analysis["duplicate_files"]:
            if len(duplicate_group) <= 1:
                continue
            
            # 保留最新的文件
            files_with_mtime = []
            for file_path in duplicate_group:
                try:
                    mtime = os.path.getmtime(file_path)
                    files_with_mtime.append((file_path, mtime))
                except:
                    continue
            
            if len(files_with_mtime) <= 1:
                continue
            
            # 按修改时间排序，保留最新的
            files_with_mtime.sort(key=lambda x: x[1], reverse=True)
            keep_file = files_with_mtime[0][0]
            
            for file_path, _ in files_with_mtime[1:]:
                removed_files.append(file_path)
                if not dry_run:
                    os.remove(file_path)
                    print(f"   删除重复文件: {file_path}")
        
        return removed_files
    
    def generate_report(self, moved_files: List, removed_files: List):
        """生成整理报告"""
        report = {
            "timestamp": str(Path().cwd()),
            "moved_files": len(moved_files),
            "removed_duplicates": len(removed_files),
            "file_movements": moved_files,
            "removed_files": removed_files,
            "category_distribution": defaultdict(int)
        }
        
        # 统计分类分布
        for move in moved_files:
            report["category_distribution"][move["category"]] += 1
        
        # 保存报告
        report_file = self.base_dir / "organization_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📊 整理报告已保存: {report_file}")
        return report
    
    def run_organization(self, dry_run: bool = True):
        """运行完整的整理流程"""
        print("🚀 开始Paper目录整理")
        print("="*50)
        
        # 1. 创建备份
        if not dry_run:
            self.create_backup()
        
        # 2. 分析当前结构
        analysis = self.analyze_current_structure()
        print(f"📊 当前统计:")
        print(f"   总文件数: {analysis['total_files']}")
        print(f"   PDF文件: {analysis['pdf_files']}")
        print(f"   重复文件组: {len(analysis['duplicate_files'])}")
        
        # 3. 创建目录结构
        self.create_directory_structure()
        
        # 4. 整理文件
        moved_files = self.organize_files(dry_run=dry_run)
        
        # 5. 移除重复文件
        removed_files = self.remove_duplicates(dry_run=dry_run)
        
        # 6. 生成报告
        report = self.generate_report(moved_files, removed_files)
        
        print(f"\n✅ 整理完成!")
        print(f"   移动文件: {len(moved_files)}")
        print(f"   删除重复: {len(removed_files)}")
        
        if dry_run:
            print(f"\n⚠️ 这是预览模式，没有实际移动文件")
            print(f"   要执行实际整理，请运行: organizer.run_organization(dry_run=False)")
        
        return report

def main():
    """主函数"""
    paper_dir = "/Users/wrt/PycharmProjects/grid-retrieval-system/paper"
    organizer = PaperDirectoryOrganizer(paper_dir)
    
    # 先运行预览
    print("🔍 预览模式 - 查看将要进行的操作")
    report = organizer.run_organization(dry_run=True)
    
    # 执行实际整理
    print(f"\n🚀 执行实际整理...")
    organizer.run_organization(dry_run=False)
    
    return report

if __name__ == "__main__":
    main()
