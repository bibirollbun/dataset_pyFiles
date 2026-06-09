import matplotlib.pyplot as plt
import numpy as np


import os
import json
from collections import defaultdict
from pathlib import Path

def get_file_info(folder_path):
    file_list = [
        "arc-agi_evaluation_challenges.json",
        "arc-agi_evaluation_solutions.json",
        "arc-agi_test_challenges.json",
        "arc-agi_training_challenges.json",
        "arc-agi_training_solutions.json"
    ]
    
    results = defaultdict(dict)
    
    for filename in file_list:
        file_path = Path(folder_path) / filename
        if not file_path.exists():
            continue
            
        # 获取文件大小
        size = os.path.getsize(file_path)
        
        # 解析JSON结构
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    elements = len(data)
                elif isinstance(data, list):
                    elements = len(data)
                else:
                    elements = 1
        except Exception as e:
            elements = f"Error: {str(e)}"
        
        results[filename] = {
            'size': size,
            'elements': elements
        }
    
    return results

def compare_datasets():
    arc2024 = get_file_info("/kaggle/input/arc-prize-2024")
    arc2025 = get_file_info("/kaggle/input/arc-prize-2025")
    
    comparison = []
    
    for filename in arc2024.keys() | arc2025.keys():
        info_2024 = arc2024.get(filename, {})
        info_2025 = arc2025.get(filename, {})
        
        size_diff = info_2025.get('size', 0) - info_2024.get('size', 0)
        elem_diff = ""
        
        if isinstance(info_2024.get('elements'), int) and isinstance(info_2025.get('elements'), int):
            elem_diff = info_2025['elements'] - info_2024['elements']
        
        comparison.append({
            'filename': filename.replace('arc-agi_','').replace('.json',''),
            '2024_size': f"{info_2024.get('size', 'Missing')/1024:.1f}KB",
            '2025_size': f"{info_2025.get('size', 'Missing')/1024:.1f}KB",
            'size_diff': f"{abs(size_diff)/1024:.1f}KB ({'↑' if size_diff>0 else '↓'})" if size_diff else "Same",
            'elements_2024': info_2024.get('elements', 'Missing'),
            'elements_2025': info_2025.get('elements', 'Missing'),
            'elements_diff': elem_diff if isinstance(elem_diff, int) else "N/A"
        })
    
    return comparison
    
def visualize_comparison(results):
    # 数据准备
    filenames = [item['filename'] for item in results]
    sizes_2024 = [float(item['2024_size'].replace('KB','')) if 'KB' in item['2024_size'] else 0 for item in results]
    sizes_2025 = [float(item['2025_size'].replace('KB','')) if 'KB' in item['2025_size'] else 0 for item in results]
    elements_2024 = [item['elements_2024'] if isinstance(item['elements_2024'], int) else 0 for item in results]
    elements_2025 = [item['elements_2025'] if isinstance(item['elements_2025'], int) else 0 for item in results]

    # 创建画布
    plt.figure(figsize=(14, 10))
    
    # ----------------- 文件大小对比图 -----------------
    ax1 = plt.subplot(2, 1, 1)
    x = np.arange(len(filenames))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, sizes_2024, width, label='2024', color='#1f77b4', edgecolor='black')
    bars2 = ax1.bar(x + width/2, sizes_2025, width, label='2025', color='#ff7f0e', edgecolor='black')
    
    # 添加数据标签和变化箭头
    for i, (s24, s25) in enumerate(zip(sizes_2024, sizes_2025)):
        diff = s25 - s24
        arrow = '↑' if diff > 0 else '↓'
        color = 'green' if diff > 0 else 'red' if diff < 0 else 'gray'
        ax1.text(x[i], max(s24, s25)+5, 
                f"{abs(diff):.1f}KB{arrow}", 
                ha='center', color=color, fontweight='bold')

    ax1.set_title('File Size Comparison (KB)', fontsize=14, pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(filenames, rotation=45, ha='right')
    ax1.legend()
    
    # ----------------- 元素数量对比图 -----------------
    ax2 = plt.subplot(2, 1, 2)
    bars3 = ax2.bar(x - width/2, elements_2024, width, color='#1f77b4', edgecolor='black')
    bars4 = ax2.bar(x + width/2, elements_2025, width, color='#ff7f0e', edgecolor='black')
    
    # 添加数据标签和变化值
    for i, (e24, e25) in enumerate(zip(elements_2024, elements_2025)):
        diff = e25 - e24
        if isinstance(diff, int):
            color = 'green' if diff > 0 else 'red' if diff < 0 else 'gray'
            ax2.text(x[i], max(e24, e25)+5, 
                    f"{diff:+d}", 
                    ha='center', color=color, fontweight='bold')

    ax2.set_title('Elements Count Comparison', fontsize=14, pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(filenames, rotation=45, ha='right')
    
    # 全局调整
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # 保存并显示
    plt.savefig('dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
def print_comparison(results):
    # 颜色格式设置
    RED = '\033[91m'
    GREEN = '\033[92m'
    END = '\033[0m'
    
    # 表头
    headers = [
        "File Name", 
        "2024 Size", 
        "2025 Size", 
        "Size Diff", 
        "2024 Elements", 
        "2025 Elements", 
        "Δ Elements"
    ]
    
    # 打印带颜色的表格
    print(f"\n{'-'*135}")
    print(f"{headers[0]:<45} | {headers[1]:>10} | {headers[2]:>10} | {headers[3]:>12} | {headers[4]:>12} | {headers[5]:>12} | {headers[6]:>10}")
    print("-"*135)
    
    for item in results:
        # 设置颜色标记
        size_diff = item['size_diff']
        elem_diff = item['elements_diff']
        
        size_color = ''
        if '↑' in size_diff:
            size_color = GREEN
        elif '↓' in size_diff:
            size_color = RED
            
        elem_color = ''
        if isinstance(elem_diff, int):
            elem_color = GREEN if elem_diff > 0 else RED if elem_diff < 0 else ''
        
        # 格式化输出
        print(
            f"{item['filename']:<45} | "
            f"{item['2024_size']:>10} | "
            f"{item['2025_size']:>10} | "
            f"{size_color}{item['size_diff']:>12}{END} | "
            f"{item['elements_2024']:>12} | "
            f"{item['elements_2025']:>12} | "
            f"{elem_color}{elem_diff if isinstance(elem_diff, int) else 'N/A':>10}{END}"
        )
    print("-"*135)
    
if __name__ == "__main__":
    comparison_results = compare_datasets()
    print_comparison(comparison_results)  # 输出带颜色标记的表格
    visualize_comparison(comparison_results)  # 生成可视化图表

