import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from openai import OpenAI
from kaggle_secrets import UserSecretsClient

# Color mapping for visualization
COLORS = ['black', 'blue', 'red', 'green', 'yellow', 
          'gray', 'magenta', 'orange', 'cyan', 'brown']

# Load ARC data
DATA_DIR = '/kaggle/input/arc-prize-2025/'
with open(Path(DATA_DIR) / 'arc-agi_training_challenges.json') as f:
    train_challenges = json.load(f)
with open(Path(DATA_DIR) / 'arc-agi_training_solutions.json') as f:
    train_solutions = json.load(f)

# Setup OpenAI client
user_secrets = UserSecretsClient()
client = OpenAI(api_key=user_secrets.get_secret("OPENAI_API_KEY"))


def draw_grid(ax, grid, title):
    """Draw a single ARC grid"""
    grid = np.array(grid)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(0, grid.shape[0])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    
    # Draw colored cells
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            color = COLORS[grid[i, j]]
            ax.fill([j, j+1, j+1, j], [i, i, i+1, i+1], color=color)
    
    # Draw grid lines
    for i in range(grid.shape[0] + 1):
        ax.axhline(i, color='white', linewidth=0.5)
    for j in range(grid.shape[1] + 1):
        ax.axvline(j, color='white', linewidth=0.5)

def show_task(task_id, task_data, predicted_grid=None, solution_grid=None):
    """Display an ARC task with training examples and test case"""
    train = task_data['train']
    test = task_data['test'][0]
    
    # Calculate layout
    n_train = len(train)
    fig, axes = plt.subplots(n_train + 1, 4, figsize=(12, 3 * (n_train + 1)))
    if n_train == 0:  # Handle edge case
        axes = axes.reshape(1, -1)
    
    # Show training examples
    for i in range(n_train):
        draw_grid(axes[i, 0], train[i]['input'], f"Train {i+1} Input")
        draw_grid(axes[i, 1], train[i]['output'], f"Train {i+1} Output")
        axes[i, 2].axis('off')
        axes[i, 3].axis('off')
    
    # Show test case
    draw_grid(axes[n_train, 0], test['input'], "Test Input")
    
    if predicted_grid is not None:
        draw_grid(axes[n_train, 1], predicted_grid, "LLM Prediction")
    else:
        axes[n_train, 1].axis('off')
    
    if solution_grid is not None:
        draw_grid(axes[n_train, 2], solution_grid, "True Solution")
    else:
        axes[n_train, 2].axis('off')
    
    axes[n_train, 3].axis('off')
    
    plt.suptitle(f"Task: {task_id}", fontsize=14)
    plt.tight_layout()
    plt.show()

def grid_to_text(grid):
    """Convert grid to text format for LLM"""
    grid = np.array(grid)
    text = f"Grid ({grid.shape[0]}x{grid.shape[1]}):\n"
    for row in grid:
        text += " ".join(str(cell) for cell in row) + "\n"
    return text


def solve_arc_task(task_data):
    """Use LLM to analyze and solve an ARC task"""
    
    # Build prompt
    prompt = """You are analyzing an ARC (Abstraction and Reasoning Corpus) task. 
Your goal is to identify the transformation rule that converts input grids to output grids.

The grids use numbers 0-9 representing different colors:
0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, 6=magenta, 7=orange, 8=cyan, 9=brown

Here are the training examples:
"""
    
    # Add training examples
    for i, example in enumerate(task_data['train']):
        prompt += f"\nExample {i+1}:\n"
        prompt += "INPUT:\n" + grid_to_text(example['input'])
        prompt += "OUTPUT:\n" + grid_to_text(example['output'])
    
    # Add test case
    prompt += f"\nTEST INPUT:\n{grid_to_text(task_data['test'][0]['input'])}"
    
    prompt += """
Analyze step by step:

1. Think of the grid as a **schematic representation** of real-world entities or concepts. 
   Imagine the colored patterns as simplified depictions of things like objects, tools, animals, human figures, paths, rooms, symbols, or abstract processes.
2. What transformation or change happens from input to output? Consider whether it resembles a natural or logical real-world process (e.g. cleaning, sorting, splitting, mirroring, erasing, growing).
3. Test your hypothesis against **all** training examples. Is your "real-world metaphor" consistent across examples?
4. Based on your metaphor and transformation pattern, apply the rule to predict the test output.

Format response as:
PATTERN: [Describe the metaphorical or schematic interpretation, and the transformation]
RULE: [The step-by-step logical rule to apply]
PREDICTION: [Grid numbers, one row per line]
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def parse_prediction(response_text):
    """Extract predicted grid from LLM response"""
    try:
        lines = response_text.split('\n')
        grid_lines = []
        
        # Find prediction section
        in_prediction = False
        for line in lines:
            if 'PREDICTION:' in line.upper():
                in_prediction = True
                continue
            
            if in_prediction and line.strip():
                parts = line.strip().split()
                if len(parts) > 0 and all(p.isdigit() and 0 <= int(p) <= 9 for p in parts):
                    grid_lines.append([int(p) for p in parts])
        
        # Ensure rectangular grid
        if grid_lines and len(set(len(row) for row in grid_lines)) == 1:
            return grid_lines
        
        return None
    except:
        return None

def evaluate_prediction(predicted_grid, true_grid):
    """Evaluate prediction accuracy"""
    if predicted_grid is None:
        return "Could not parse prediction"
    
    try:
        pred = np.array(predicted_grid)
        true = np.array(true_grid)
        
        if pred.shape != true.shape:
            return f"Wrong shape: {pred.shape} vs {true.shape}"
        
        if np.array_equal(pred, true):
            return "CORRECT!"
        else:
            errors = np.sum(pred != true)
            accuracy = (pred.size - errors) / pred.size * 100
            return f"{errors}/{pred.size} errors ({accuracy:.1f}% correct)"
    except:
        return "Evaluation error"


def analyze_random_tasks(n_tasks=3, show_details=True):
    """Analyze multiple random tasks"""
    task_ids = random.sample(list(train_challenges.keys()), n_tasks)
    
    correct = 0
    total = 0
    
    print(f"Analyzing {n_tasks} random tasks...")
    print("=" * 60)
    
    for i, task_id in enumerate(task_ids):
        print(f"\nTask {i+1}/{n_tasks}: {task_id}")
        
        task_data = train_challenges[task_id]
        solution = train_solutions[task_id][0] if task_id in train_solutions else None
        
        # Get prediction
        response = solve_arc_task(task_data)
        predicted_grid = parse_prediction(response)

        if show_details:
            print("LLM Analysis:")
            print(response)
            print("\n" + "=" * 50)
            
        # Show task
        show_task(task_id, task_data, predicted_grid, solution)
        
        # Evaluate
        if solution is not None:
            result = evaluate_prediction(predicted_grid, solution)
            print(f"Result: {result}")
            total += 1
            if "CORRECT" in result:
                correct += 1
        
        print("-" * 40)
    
    # Summary
    if total > 0:
        accuracy = correct / total * 100
        print(f"\nSUMMARY: {correct}/{total} correct ({accuracy:.1f}%)")
    
    return correct, total


analyze_random_tasks(10)

