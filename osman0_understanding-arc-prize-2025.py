import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches


# ARC dataset color stand
ARC_COLORS = [
    '#000000',  # 0: Black
    '#0074D9',  # 1: Blue
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Gray
    '#F012BE',  # 6: Pink
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Aqua
    '#870C25'   # 9: Deep red
]



def visualize_grid(grid, title="Grid", ax=None):
    """Visualizes a single grid"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    grid_array = np.array(grid)
    
    # Create custom colormap
    cmap = ListedColormap(ARC_COLORS)

    im = ax.imshow(grid_array, cmap=cmap, vmin=0, vmax=9)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, len(grid[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    # Clean up axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    return ax



def visualize_task(task_data, task_id):
    """Visualizes a task (train and test examples)"""
    train_examples = task_data['train']
    test_examples = task_data['test']
    
    # Calculate the total number of samples
    total_examples = len(train_examples) + len(test_examples)
    
    # Figure sizes
    cols = 4  # Input, Output, Input, Output...
    rows = total_examples
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    fig.suptitle(f'ARC Task: {task_id}', fontsize=16, fontweight='bold')
    
    # If there is only one line, make the axes 2D.
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    current_row = 0
    
    # Draw simple for train
    for i, example in enumerate(train_examples):
        if current_row < rows:
            # Input
            visualize_grid(example['input'], 
                         f'Train {i+1} - Input\n{len(example["input"])}×{len(example["input"][0])}', 
                         axes[current_row, 0])
            
            # Output
            visualize_grid(example['output'], 
                         f'Train {i+1} - Output\n{len(example["output"])}×{len(example["output"][0])}', 
                         axes[current_row, 1])
            
            # Clean emptys 
            axes[current_row, 2].axis('off')
            axes[current_row, 3].axis('off')
            
            current_row += 1
    
    # Draw tests simples
    for i, example in enumerate(test_examples):
        if current_row < rows:
            # Input
            visualize_grid(example['input'], 
                         f'Test {i+1} - Input\n{len(example["input"])}×{len(example["input"][0])}', 
                         axes[current_row, 0])
            
            # if dont test outputs, do it empty
            if 'output' in example:
                visualize_grid(example['output'], 
                             f'Test {i+1} - Output\n{len(example["output"])}×{len(example["output"][0])}', 
                             axes[current_row, 1])
            else:
                axes[current_row, 1].text(0.5, 0.5, 'TO BE SOLVED :) !', 
                                        ha='center', va='center', 
                                        fontsize=12, fontweight='bold',
                                        transform=axes[current_row, 1].transAxes)
                axes[current_row, 1].set_xlim(0, 1)
                axes[current_row, 1].set_ylim(0, 1)
                axes[current_row, 1].axis('off')
            
            # Clean emptys 
            axes[current_row, 2].axis('off')
            axes[current_row, 3].axis('off')
            
            current_row += 1
    
    plt.tight_layout()
    return fig



def analyze_task_patterns(task_data):
    """Analyzes the general characteristics of the task"""
    patterns = {
        'input_sizes': [],
        'output_sizes': [],
        'color_usage': set(),
        'size_changes': []
    }
    
    for example in task_data['train']:
        input_grid = example['input']
        output_grid = example['output']
        
        # Save dimensions
        input_size = (len(input_grid), len(input_grid[0]))
        output_size = (len(output_grid), len(output_grid[0]))
        
        patterns['input_sizes'].append(input_size)
        patterns['output_sizes'].append(output_size)
        
        # Save size that changed 
        if input_size != output_size:
            patterns['size_changes'].append((input_size, output_size))
        
        # Save that used colors 
        for row in input_grid:
            patterns['color_usage'].update(row)
        for row in output_grid:
            patterns['color_usage'].update(row)
    
    return patterns


# Main usage function
def explore_arc_task(task_id=None, random_task=False, file_path='/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json'):
    """Explore and visualize a specific task.  """
    import random
    
    # Upload file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # pick random task
    if random_task or task_id is None:
        task_id = random.choice(list(data.keys()))
        if random_task:
            print(f"Randomly selected task: {task_id}")
    
    if task_id not in data:
        print(f"Task {task_id} didn't find!")
        return
    
    task_data = data[task_id]
    
    # analysis tasks
    patterns = analyze_task_patterns(task_data)
    
    print(f"\n🎯 Task analysis: {task_id}")
    print(f"📊 Train samples: {len(task_data['train'])}")
    print(f"🧪 Test samples: {len(task_data['test'])}")
    print(f"📐 Input sizes: {patterns['input_sizes']}")
    print(f"📐 Output size: {patterns['output_sizes']}")
    print(f"🎨 Used colors: {sorted(patterns['color_usage'])}")
    
    if patterns['size_changes']:
        print(f"Dimensional changes: {patterns['size_changes']}")
    else:
        print("...")
    
    # Visulation
    fig = visualize_task(task_data, task_id)
    plt.show()
    
    return task_data, patterns

def random_task():
    """Visualize a random task"""
    return explore_arc_task(random_task=True)

def specific_task(task_id):
    """Visualize a specific task"""
    return explore_arc_task(task_id=task_id)

if __name__ == "__main__":
    print("...visualizing the task...\n")
    
    # Random task visualization
    task_data, patterns = random_task()
    
    print("\n" + "="*50)
    print("💡 Usage examples:")
    print("🎲 Random task: random_task()")
    print("🎯 Specific task: specific_task('task_id')")
    print("="*50) 


# 🎲 Function to show multiple random tasks
def explore_multiple_tasks(num_tasks=5, file_path='/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json'):
    """Explore and visualize multiple random tasks"""
    import random
    
    # Upload file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Select random tasks
    all_task_ids = list(data.keys())
    selected_tasks = random.sample(all_task_ids, min(num_tasks, len(all_task_ids)))
    
    print(f"🎯 Exploring {len(selected_tasks)} random tasks:")
    print("=" * 60)
    
    results = []
    
    for i, task_id in enumerate(selected_tasks, 1):
        print(f"\n📋 Task {i}/{len(selected_tasks)}: {task_id}")
        print("-" * 40)
        
        task_data = data[task_id]
        patterns = analyze_task_patterns(task_data)
        
        print(f"📊 Train samples: {len(task_data['train'])}")
        print(f"🧪 Test samples: {len(task_data['test'])}")
        print(f"📐 Input sizes: {patterns['input_sizes']}")
        print(f"📐 Output sizes: {patterns['output_sizes']}")
        print(f"🎨 Used colors: {sorted(patterns['color_usage'])}")
        
        if patterns['size_changes']:
            print(f"🔄 Size changes: {patterns['size_changes']}")
        else:
            print("🔄 No size changes")
        
        # Visualize the task
        fig = visualize_task(task_data, task_id)
        plt.show()
        
        results.append((task_id, task_data, patterns))
        
        if i < len(selected_tasks):
            print("\n" + "🔸" * 20 + " NEXT TASK " + "🔸" * 20)
    
    print(f"\n✅ Completed analysis of {len(selected_tasks)} tasks!")
    print("=" * 60)
    
    return results



# 🚀 Execute: Show 5 random tasks
if __name__ == "__main__":
    print("🎯 Visualizing 5 random ARC tasks...\n")
    
    # Show 5 random tasks
    results = explore_multiple_tasks(num_tasks=5)
    
    print("\n" + "="*60)
    print("💡 Available functions:")
    print("🎲 Single random task: random_task()")
    print("🎯 Specific task: specific_task('task_id')")
    print("🎲 Multiple random tasks: explore_multiple_tasks(num_tasks=5)")
    print("="*60)





