#!/usr/bin/env python3
"""
代码质量检查和格式化工具
自动化代码质量检查、格式化和修复
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse


class CodeQualityChecker:
    """代码质量检查器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.python_files = self._find_python_files()
        self.results = {}
    
    def _find_python_files(self) -> List[Path]:
        """查找所有Python文件"""
        python_files = []
        
        # 包含的目录
        include_dirs = ['modules', 'tests', 'scripts', 'examples', 'adaptive']
        
        # 包含根目录的Python文件
        for file in self.project_root.glob('*.py'):
            python_files.append(file)
        
        # 包含子目录的Python文件
        for dir_name in include_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                python_files.extend(dir_path.rglob('*.py'))
        
        return python_files
    
    def run_basic_syntax_check(self) -> Dict[str, Any]:
        """运行基础语法检查"""
        print("🔍 运行Python语法检查...")
        
        syntax_errors = []
        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 编译检查语法
                compile(content, str(py_file), 'exec')
                
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}:{e.lineno} - {e.msg}")
            except Exception as e:
                syntax_errors.append(f"{py_file} - {e}")
        
        self.results['syntax'] = {
            'success': len(syntax_errors) == 0,
            'errors': syntax_errors,
            'error_count': len(syntax_errors)
        }
        
        if len(syntax_errors) == 0:
            print("  ✅ 语法检查通过")
        else:
            print(f"  ❌ 发现 {len(syntax_errors)} 个语法错误")
            for error in syntax_errors[:3]:
                print(f"    {error}")
            if len(syntax_errors) > 3:
                print(f"    ... 还有 {len(syntax_errors) - 3} 个错误")
        
        return self.results.get('syntax', {})
    
    def run_import_check(self) -> Dict[str, Any]:
        """运行导入检查"""
        print("📦 检查模块导入...")
        
        import_errors = []
        for py_file in self.python_files:
            try:
                # 简单的导入测试
                cmd = [sys.executable, '-m', 'py_compile', str(py_file)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    import_errors.append(f"{py_file} - {result.stderr}")
                    
            except Exception as e:
                import_errors.append(f"{py_file} - {e}")
        
        self.results['imports'] = {
            'success': len(import_errors) == 0,
            'errors': import_errors,
            'error_count': len(import_errors)
        }
        
        if len(import_errors) == 0:
            print("  ✅ 导入检查通过")
        else:
            print(f"  ❌ 发现 {len(import_errors)} 个导入错误")
        
        return self.results.get('imports', {})
    
    def run_line_length_check(self) -> Dict[str, Any]:
        """检查行长度"""
        print("📏 检查代码行长度...")
        
        long_lines = []
        max_length = 100
        
        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    for line_no, line in enumerate(f, 1):
                        if len(line.rstrip()) > max_length:
                            long_lines.append(f"{py_file}:{line_no} - 长度: {len(line.rstrip())}")
            except Exception as e:
                long_lines.append(f"{py_file} - 读取错误: {e}")
        
        self.results['line_length'] = {
            'success': len(long_lines) == 0,
            'issues': long_lines,
            'issue_count': len(long_lines)
        }
        
        if len(long_lines) == 0:
            print("  ✅ 行长度检查通过")
        else:
            print(f"  ❌ 发现 {len(long_lines)} 行超长代码")
        
        return self.results.get('line_length', {})
    
    def run_all_checks(self) -> Dict[str, Any]:
        """运行所有基础检查"""
        print(f"🚀 开始代码质量检查 (共 {len(self.python_files)} 个Python文件)")
        print("=" * 60)
        
        # 基础检查
        self.run_basic_syntax_check()
        self.run_import_check()
        self.run_line_length_check()
        
        return self.results
    
    def print_summary(self) -> None:
        """打印检查摘要"""
        print("\n" + "=" * 60)
        print("📊 代码质量检查摘要")
        print("=" * 60)
        
        total_issues = 0
        for tool, result in self.results.items():
            if result.get('error_count'):
                total_issues += result['error_count']
            elif result.get('issue_count'):
                total_issues += result['issue_count']
        
        passed_checks = sum(
            1 for result in self.results.values()
            if result.get('success', False)
        )
        total_checks = len(self.results)
        
        print(f"✅ 通过检查: {passed_checks}/{total_checks}")
        print(f"❌ 总问题数: {total_issues}")
        
        if total_issues == 0:
            print("\n🎉 恭喜！基础代码质量检查通过！")
        else:
            print("\n⚠️ 发现一些问题，建议修复后再进行进一步测试。")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FusionRAG代码质量检查工具")
    parser.add_argument('--project-root', '-p', default='.', help='项目根目录')
    
    args = parser.parse_args()
    
    # 创建检查器
    checker = CodeQualityChecker(args.project_root)
    
    print("🔧 FusionRAG代码质量检查工具（基础版）")
    print("=" * 50)
    
    # 运行检查
    results = checker.run_all_checks()
    
    # 打印摘要
    checker.print_summary()
    
    # 根据结果设置退出码
    all_passed = all(
        result.get('success', False) for result in results.values()
    )
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()