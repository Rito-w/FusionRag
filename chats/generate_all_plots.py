#!/usr/bin/env python3
"""
论文图表生成主脚本
一键生成所有论文所需的图表
"""

import os
import sys
import subprocess
import time

def run_script(script_name):
    """运行指定的Python脚本"""
    try:
        print(f"\n{'='*50}")
        print(f"正在运行: {script_name}")
        print(f"{'='*50}")
        
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd='chats')
        
        if result.returncode == 0:
            print(f"✅ {script_name} 执行成功")
            if result.stdout:
                print("输出:", result.stdout)
        else:
            print(f"❌ {script_name} 执行失败")
            if result.stderr:
                print("错误:", result.stderr)
                
    except Exception as e:
        print(f"❌ 运行 {script_name} 时发生异常: {e}")

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'matplotlib',
        'numpy', 
        'pandas',
        'seaborn',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
    
    if missing_packages:
        print(f"\n请先安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def create_output_directory():
    """创建输出目录"""
    if not os.path.exists('chats'):
        os.makedirs('chats')
        print("✅ 创建了 chats 目录")
    else:
        print("✅ chats 目录已存在")

def main():
    """主函数"""
    print("🎨 论文图表生成器")
    print("=" * 60)
    
    # 检查依赖
    print("\n📦 检查依赖包...")
    if not check_dependencies():
        return
    
    # 创建输出目录
    print("\n📁 准备输出目录...")
    create_output_directory()
    
    # 脚本列表
    scripts = [
        'paper_visualizations.py',
        'system_architecture_diagrams.py', 
        'statistical_analysis_plots.py'
    ]
    
    print(f"\n🚀 开始生成图表...")
    print(f"将生成 {len(scripts)} 类图表")
    
    start_time = time.time()
    
    # 运行所有脚本
    for i, script in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] 处理 {script}...")
        run_script(script)
        time.sleep(1)  # 短暂暂停，避免资源冲突
    
    end_time = time.time()
    
    print(f"\n🎉 所有图表生成完成!")
    print(f"⏱️  总耗时: {end_time - start_time:.2f} 秒")
    
    # 列出生成的文件
    print(f"\n📋 生成的图表文件:")
    if os.path.exists('chats'):
        files = [f for f in os.listdir('chats') if f.endswith(('.pdf', '.png'))]
        if files:
            for file in sorted(files):
                print(f"  📄 {file}")
        else:
            print("  ⚠️  未找到生成的图表文件")
    
    print(f"\n💡 使用建议:")
    print(f"  - PDF格式适合LaTeX论文插入")
    print(f"  - PNG格式适合预览和演示")
    print(f"  - 所有图表都使用了学术论文标准格式")
    print(f"  - 颜色方案考虑了色盲友好性")

if __name__ == "__main__":
    main()
