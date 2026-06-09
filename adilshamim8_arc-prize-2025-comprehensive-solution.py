# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
import random
from tqdm.notebook import tqdm
import copy
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from IPython.display import display

%matplotlib inline
plt.style.use('fivethirtyeight')


# Define data paths
DATA_DIR = '/kaggle/input/arc-prize-2025/'

# Load the datasets
with open(Path(DATA_DIR) / 'arc-agi_training_challenges.json') as f:
    train_challenges = json.load(f)
with open(Path(DATA_DIR) / 'arc-agi_training_solutions.json') as f:
    train_solutions = json.load(f)
with open(Path(DATA_DIR) / 'arc-agi_evaluation_challenges.json') as f:
    eval_challenges = json.load(f)
with open(Path(DATA_DIR) / 'arc-agi_evaluation_solutions.json') as f:
    eval_solutions = json.load(f)
with open(Path(DATA_DIR) / 'arc-agi_test_challenges.json') as f:
    test_challenges = json.load(f)

print(f"Training tasks: {len(train_challenges)}")
print(f"Training solutions: {len(train_solutions)}")
print(f"Evaluation tasks: {len(eval_challenges)}")
print(f"Evaluation solutions: {len(eval_solutions)}")
print(f"Test tasks: {len(test_challenges)}")

# Sample a random task ID to examine
sample_task_id = random.choice(list(train_challenges.keys()))
sample_task = train_challenges[sample_task_id]
sample_solution = train_solutions[sample_task_id]

print(f"\nExamining sample task: {sample_task_id}")
print(f"Number of training examples: {len(sample_task['train'])}")
print(f"Number of test examples: {len(sample_task['test'])}")
print(f"Number of solutions: {len(sample_solution)}")


# Define a color map for grid visualization
COLORS = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', 
          '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']

def show_grid(grid, title=None, ax=None):
    """Visualize a single grid with colored cells."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    # Convert grid to numpy array
    grid = np.array(grid)
    height, width = grid.shape
    
    # Create a custom colormap
    cmap = plt.cm.colors.ListedColormap(COLORS[:max(10, grid.max()+1)])
    bounds = np.arange(-0.5, max(10, grid.max()+1.5))
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot the grid
    ax.imshow(grid, cmap=cmap, norm=norm)
    
    # Add grid lines
    for i in range(width+1):
        ax.axvline(i-0.5, color='white', linewidth=1.5)
    for i in range(height+1):
        ax.axhline(i-0.5, color='white', linewidth=1.5)
    
    # Remove ticks and add title
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    
    return ax

def visualize_task(task, task_id=None, solutions=None):
    """Visualize a complete ARC task with input/output pairs."""
    n_train = len(task['train'])
    n_test = len(task['test'])
    n_rows = max(n_train, n_test)
    
    # Create figure with 3 columns (or 4 if solutions provided)
    n_cols = 4 if solutions else 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    
    # Ensure axes is 2D even if there's only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Add task ID as figure title
    if task_id:
        fig.suptitle(f"Task ID: {task_id}", fontsize=16, y=1.02)
    
    # Plot training examples
    for i in range(n_rows):
        for j in range(n_cols):
            if i < n_train and j == 0:  # Train inputs
                show_grid(task['train'][i]['input'], "Train Input", axes[i, j])
            elif i < n_train and j == 1:  # Train outputs
                show_grid(task['train'][i]['output'], "Train Output", axes[i, j])
            elif i < n_test and j == 2:  # Test inputs
                show_grid(task['test'][i]['input'], "Test Input", axes[i, j])
            elif solutions and i < len(solutions) and j == 3:  # Test solutions
                show_grid(solutions[i], "Test Solution", axes[i, j])
            else:
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()


# Visualize the sample task
visualize_task(sample_task, sample_task_id, sample_solution)

# Look at a few more random tasks
for _ in range(3):
    task_id = random.choice(list(train_challenges.keys()))
    task = train_challenges[task_id]
    solution = train_solutions[task_id]
    visualize_task(task, task_id, solution)


def analyze_task_complexity(challenges):
    """Analyze task complexity metrics across the dataset."""
    # Initialize storage for metrics
    grid_sizes = []
    color_counts = []
    input_output_size_ratios = []
    
    for task_id, task in challenges.items():
        for example in task['train']:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Grid sizes
            grid_sizes.append((input_grid.shape[0], input_grid.shape[1]))
            
            # Color diversity
            unique_colors = len(np.unique(input_grid))
            color_counts.append(unique_colors)
            
            # Input/output size relationship
            size_ratio = (output_grid.size / input_grid.size)
            input_output_size_ratios.append(size_ratio)
    
    return {
        'grid_sizes': grid_sizes,
        'color_counts': color_counts,
        'size_ratios': input_output_size_ratios
    }

# Analyze complexity metrics
complexity_metrics = analyze_task_complexity(train_challenges)

# Visualize the complexity distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Grid sizes
sizes = np.array(complexity_metrics['grid_sizes'])
axes[0].scatter(sizes[:, 1], sizes[:, 0], alpha=0.5)
axes[0].set_xlabel('Width')
axes[0].set_ylabel('Height')
axes[0].set_title('Grid Dimensions')
axes[0].grid(True)

# Color counts
axes[1].hist(complexity_metrics['color_counts'], bins=10, color='skyblue')
axes[1].set_xlabel('Number of Unique Colors')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Color Diversity')

# Size ratios
axes[2].hist(complexity_metrics['size_ratios'], bins=20, color='salmon')
axes[2].set_xlabel('Output/Input Size Ratio')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Input-Output Size Relationship')
axes[2].set_xlim(0, 10)  # Limit to reasonable ratios

plt.tight_layout()
plt.show()


# Core transformation operations
def rotate(grid, k=1):
    """Rotate grid k*90 degrees counter-clockwise."""
    return np.rot90(np.array(grid), k)

def flip(grid, axis=0):
    """Flip grid along axis (0=vertical, 1=horizontal)."""
    return np.flip(np.array(grid), axis=axis)

def color_replace(grid, from_color, to_color):
    """Replace one color with another in the grid."""
    grid = np.array(grid).copy()
    grid[grid == from_color] = to_color
    return grid

def crop(grid, x1, y1, x2, y2):
    """Extract a subgrid from (x1,y1) to (x2,y2) inclusive."""
    return np.array(grid)[y1:y2+1, x1:x2+1]

def pad(grid, top=0, bottom=0, left=0, right=0, fill=0):
    """Pad grid with specified amounts on each side."""
    return np.pad(np.array(grid), ((top, bottom), (left, right)), 
                 constant_values=fill)

def flood_fill(grid, x, y, new_color):
    """Perform flood fill from position (x,y) with new_color."""
    grid = np.array(grid).copy()
    target_color = grid[y, x]
    if target_color == new_color:
        return grid
    
    h, w = grid.shape
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if (cx < 0 or cy < 0 or cx >= w or cy >= h or 
            grid[cy, cx] != target_color):
            continue
        
        grid[cy, cx] = new_color
        stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])
    
    return grid

def get_bounding_box(grid, color):
    """Find the bounding box (x1,y1,x2,y2) of cells with specified color."""
    grid = np.array(grid)
    y_indices, x_indices = np.where(grid == color)
    
    if len(y_indices) == 0:
        return None
    
    return (
        int(np.min(x_indices)), 
        int(np.min(y_indices)),
        int(np.max(x_indices)), 
        int(np.max(y_indices))
    )

# Test and visualize our DSL operations
def demonstrate_dsl_operations():
    """Show examples of each DSL operation."""
    # Create a test grid
    test_grid = np.array([
        [0, 1, 1, 0],
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        [0, 1, 1, 0]
    ])
    
    operations = [
        ("Original", test_grid),
        ("Rotate 90°", rotate(test_grid, 1)),
        ("Rotate 180°", rotate(test_grid, 2)),
        ("Flip Vertical", flip(test_grid, 0)),
        ("Flip Horizontal", flip(test_grid, 1)),
        ("Replace 2→3", color_replace(test_grid, 2, 3)),
        ("Crop Center", crop(test_grid, 1, 1, 2, 2)),
        ("Pad (1 all sides)", pad(test_grid, 1, 1, 1, 1)),
        ("Flood Fill (0,0→3)", flood_fill(test_grid, 0, 0, 3))
    ]
    
    # Visualize all operations
    n_ops = len(operations)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, (title, grid) in enumerate(operations):
        show_grid(grid, title, axes[i])
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate bounding box
    fig, ax = plt.subplots(figsize=(6, 6))
    show_grid(test_grid, "Bounding Box of Color 2", ax)
    
    bbox = get_bounding_box(test_grid, 2)
    if bbox:
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle((x1-0.5, y1-0.5), x2-x1+1, y2-y1+1, 
                           linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    
    plt.show()

# Demonstrate our DSL operations
demonstrate_dsl_operations()


# Define our DSL token vocabulary
DSL_TOKENS = [
    "rotate_90", "rotate_180", "rotate_270",
    "flip_v", "flip_h",
    "color_replace_1_2", "color_replace_2_1", 
    "color_replace_1_3", "color_replace_3_1",
    "color_replace_2_3", "color_replace_3_2"
]

# Map tokens to functions
TOKEN_TO_FUNC = {
    "rotate_90": lambda g: rotate(g, 1),
    "rotate_180": lambda g: rotate(g, 2),
    "rotate_270": lambda g: rotate(g, 3),
    "flip_v": lambda g: flip(g, 0),
    "flip_h": lambda g: flip(g, 1),
    "color_replace_1_2": lambda g: color_replace(g, 1, 2),
    "color_replace_2_1": lambda g: color_replace(g, 2, 1),
    "color_replace_1_3": lambda g: color_replace(g, 1, 3),
    "color_replace_3_1": lambda g: color_replace(g, 3, 1),
    "color_replace_2_3": lambda g: color_replace(g, 2, 3),
    "color_replace_3_2": lambda g: color_replace(g, 3, 2)
}

def run_program(grid, program_steps):
    """Execute a sequence of operations on a grid."""
    result = np.array(grid).copy()
    try:
        for op in program_steps:
            result = op(result)
        return result
    except Exception as e:
        # print(f"Error in program execution: {e}")
        return None

def program_matches(train_examples, program_steps):
    """Check if a program successfully transforms all training examples."""
    for pair in train_examples:
        input_grid = np.array(pair["input"])
        target_output = np.array(pair["output"])
        predicted_output = run_program(input_grid, program_steps)
        
        if predicted_output is None:
            return False
        
        if predicted_output.shape != target_output.shape:
            return False
        
        if not np.array_equal(predicted_output, target_output):
            return False
            
    return True

def search_program(train_examples, max_depth=3, time_limit=10):
    """Search for a program that solves all training examples (DFS with time limit)."""
    import time
    start_time = time.time()
    ops = list(TOKEN_TO_FUNC.values())
    
    def dfs(current_steps, depth=0):
        if time.time() - start_time > time_limit:
            return None  # Time limit exceeded
        
        if depth > max_depth:
            return None
            
        if depth > 0 and program_matches(train_examples, current_steps):
            return current_steps
            
        for op in ops:
            next_steps = current_steps + [op]
            result = dfs(next_steps, depth + 1)
            if result is not None:
                return result
                
        return None
    
    return dfs([])

def solve_task_symbolic(task, max_depth=3, time_limit=10):
    """Solve an ARC task using symbolic program search."""
    train = task["train"]
    test = task["test"]
    
    # Search for a program that matches all training examples
    solution_program = search_program(train, max_depth, time_limit)
    
    # Apply the found program to test inputs
    predictions = []
    for test_example in test:
        input_grid = test_example["input"]
        
        # First attempt: Use the found program
        if solution_program:
            output1 = run_program(input_grid, solution_program)
            if output1 is None:
                output1 = input_grid  # Fallback to input
        else:
            output1 = input_grid
            
        # Second attempt: Try a slightly modified program
        # Here we're just using a simple variation - in practice, you'd want
        # to try more sophisticated alternatives
        if solution_program and len(solution_program) > 0:
            # Try adding one more rotation
            modified_program = solution_program + [lambda g: rotate(g, 1)]
            output2 = run_program(input_grid, modified_program)
            if output2 is None:
                output2 = input_grid
        else:
            # Just try a single rotation as a fallback
            output2 = rotate(input_grid, 2)
            
        # Convert outputs to list format
        if isinstance(output1, np.ndarray):
            output1 = output1.tolist()
        if isinstance(output2, np.ndarray):
            output2 = output2.tolist()
            
        predictions.append({
            "attempt_1": output1,
            "attempt_2": output2
        })
        
    return predictions, solution_program

# Test our symbolic search on a simple example
def test_symbolic_search():
    """Test our symbolic search on a selected task."""
    # Find a task with a relatively simple pattern
    simple_tasks = []
    for task_id, task in train_challenges.items():
        if (len(task['train']) >= 2 and 
            all(np.array(ex['input']).size < 20 for ex in task['train'])):
            simple_tasks.append((task_id, task))
    
    if not simple_tasks:
        print("No suitable simple tasks found.")
        return
    
    # Select a random simple task
    task_id, task = random.choice(simple_tasks)
    print(f"Testing symbolic search on task: {task_id}")
    
    # Visualize the task
    visualize_task(task, task_id, train_solutions[task_id])
    
    # Try to solve it
    start_time = time.time()
    predictions, program = solve_task_symbolic(task, max_depth=2, time_limit=5)
    end_time = time.time()
    
    print(f"Search completed in {end_time - start_time:.2f} seconds")
    
    if program:
        print(f"Found a program with {len(program)} operations!")
        
        # Show the program's output
        fig, axes = plt.subplots(len(task['train']), 3, figsize=(12, 4*len(task['train'])))
        if len(task['train']) == 1:
            axes = axes.reshape(1, -1)
        
        for i, example in enumerate(task['train']):
            show_grid(example['input'], "Input", axes[i, 0])
            predicted = run_program(example['input'], program)
            show_grid(predicted, "Predicted", axes[i, 1])
            show_grid(example['output'], "Target", axes[i, 2])
        
        plt.tight_layout()
        plt.show()
        
        # Apply to test
        for i, test_case in enumerate(task['test']):
            print(f"Test case {i+1}:")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Input
            show_grid(test_case['input'], "Test Input", axes[0])
            
            # Predictions
            show_grid(predictions[i]['attempt_1'], "Prediction 1", axes[1])
            show_grid(predictions[i]['attempt_2'], "Prediction 2", axes[2])
            
            plt.tight_layout()
            plt.show()
    else:
        print("No solution found within the time limit.")

# Run the symbolic search test
import time
test_symbolic_search()


# One-hot encoding for grid colors
def one_hot_encode_grid(grid, num_colors=10):
    """Convert a grid to one-hot encoding."""
    grid = np.array(grid)
    h, w = grid.shape
    encoded = np.zeros((num_colors, h, w), dtype=np.float32)
    
    for c in range(num_colors):
        encoded[c] = (grid == c).astype(np.float32)
        
    return encoded

# Combine grids for neural processing
def prepare_grid_features(input_grid, output_grid=None, max_size=16):
    """Prepare grid features for neural network input."""
    input_grid = np.array(input_grid)
    
    # Get input grid features
    ih, iw = input_grid.shape
    input_one_hot = one_hot_encode_grid(input_grid)
    
    if output_grid is not None:
        output_grid = np.array(output_grid)
        # Get output grid features
        oh, ow = output_grid.shape
        output_one_hot = one_hot_encode_grid(output_grid)
        
        # Make them the same size for concatenation
        max_h = max(ih, oh, max_size)
        max_w = max(iw, ow, max_size)
        
        # Pad input
        if ih < max_h or iw < max_w:
            pad_h = max(0, max_h - ih)
            pad_w = max(0, max_w - iw)
            input_padded = np.zeros((10, max_h, max_w), dtype=np.float32)
            input_padded[:, :ih, :iw] = input_one_hot
            input_one_hot = input_padded
        
        # Pad output
        if oh < max_h or ow < max_w:
            pad_h = max(0, max_h - oh)
            pad_w = max(0, max_w - ow)
            output_padded = np.zeros((10, max_h, max_w), dtype=np.float32)
            output_padded[:, :oh, :ow] = output_one_hot
            output_one_hot = output_padded
        
        # Concatenate along channel dimension
        features = np.concatenate([input_one_hot, output_one_hot], axis=0)
    else:
        # If no output grid, just use input features
        features = input_one_hot
    
    return features

# Neural network model for predicting DSL operations
class SketchPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # 20 channels: 10 for input grid + 10 for output grid
        self.conv1 = nn.Conv2d(20, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d(1)  # Global pooling
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, len(DSL_TOKENS))
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Demonstrate how we would prepare training data
def prepare_training_example(example):
    """Prepare a single training example for the neural network."""
    input_grid = np.array(example['input'])
    output_grid = np.array(example['output'])
    
    # This is a simplified demonstration - in practice, we would need
    # to actually determine which DSL operations transform input to output
    # For now, we'll just assign a random token as the label
    label = random.randint(0, len(DSL_TOKENS)-1)
    
    # Prepare input features
    features = prepare_grid_features(input_grid, output_grid)
    
    return features, label

# Visualize how we'd process a task for neural training
def visualize_neural_processing():
    """Show how a task would be processed for neural network training."""
    # Pick a random task
    task_id = random.choice(list(train_challenges.keys()))
    task = train_challenges[task_id]
    
    # Show the original task
    print(f"Original Task: {task_id}")
    visualize_task(task, task_id)
    
    # Process the first training example
    example = task['train'][0]
    features, label = prepare_training_example(example)
    
    print(f"Processed Features Shape: {features.shape}")
    print(f"Assigned Label (Operation Index): {label} ({DSL_TOKENS[label]})")
    
    # Visualize the feature channels
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(20):
        if i < 10:
            title = f"Input Color {i}"
        else:
            title = f"Output Color {i-10}"
        
        channel = features[i]
        axes[i].imshow(channel, cmap='viridis')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Demonstrate neural processing
visualize_neural_processing()


# Extract features from a task for classification
def extract_task_features(task):
    """Extract features from a task for classification."""
    features = []
    
    # Process all training examples
    for example in task['train']:
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Grid sizes
        ih, iw = input_grid.shape
        oh, ow = output_grid.shape
        
        # Color statistics
        input_colors = np.unique(input_grid)
        output_colors = np.unique(output_grid)
        
        # Check for different transformation patterns
        
        # Pattern 1: Size preservation
        size_preserved = (ih == oh and iw == ow)
        
        # Pattern 2: Size scaling
        size_scaling = (oh/ih == ow/iw and oh/ih > 1)
        
        # Pattern 3: Rotation-like
        rotation_like = False
        for k in range(1, 4):
            rotated = rotate(input_grid, k)
            if rotated.shape == output_grid.shape:
                if np.sum(rotated == output_grid) / output_grid.size > 0.7:
                    rotation_like = True
                    break
        
        # Pattern 4: Color transformation
        color_transform = len(output_colors) != len(input_colors)
        
        # Combine features
        example_features = [
            ih, iw,  # Input dimensions
            oh, ow,  # Output dimensions
            len(input_colors),  # Number of input colors
            len(output_colors),  # Number of output colors
            int(size_preserved),
            int(size_scaling),
            int(rotation_like),
            int(color_transform),
            oh/ih if ih > 0 else 0,  # Height ratio
            ow/iw if iw > 0 else 0   # Width ratio
        ]
        
        features.extend(example_features)
    
    # Pad to fixed length if needed
    while len(features) < 30:
        features.append(0)
    
    # Return first 30 features
    return np.array(features[:30], dtype=np.float32)

# Simple task classifier model
class TaskClassifier(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, num_classes=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Manually define task categories for demonstration
def assign_task_categories():
    """Assign categories to tasks for demonstration purposes."""
    categories = {}
    
    for i, task_id in enumerate(train_challenges.keys()):
        # Simple round-robin assignment for demonstration
        category = i % 4
        categories[task_id] = category
    
    return categories

# Visualize task categories
def visualize_task_categories():
    """Visualize tasks by category."""
    categories = assign_task_categories()
    
    # Get features for each task
    features_list = []
    category_list = []
    task_ids = []
    
    for task_id, category in list(categories.items())[:100]:  # Limit to first 100
        task = train_challenges[task_id]
        features = extract_task_features(task)
        features_list.append(features)
        category_list.append(category)
        task_ids.append(task_id)
    
    # Use t-SNE for dimensionality reduction
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_list)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features_scaled)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    category_names = ['Pattern A', 'Pattern B', 'Pattern C', 'Pattern D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for category in range(4):
        mask = np.array(category_list) == category
        plt.scatter(
            features_2d[mask, 0], 
            features_2d[mask, 1], 
            c=colors[category],
            label=category_names[category],
            alpha=0.7
        )
    
    # Add task IDs as annotations
    for i, task_id in enumerate(task_ids):
        plt.annotate(
            task_id[:6] + '...',  # Shortened ID
            (features_2d[i, 0], features_2d[i, 1]),
            fontsize=8,
            alpha=0.7
        )
    
    plt.title('Task Categories Visualization (t-SNE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Show example tasks from each category
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    
    for category in range(4):
        category_tasks = [task_id for i, task_id in enumerate(task_ids) 
                         if category_list[i] == category]
        
        if category_tasks:
            example_task_id = random.choice(category_tasks)
            example_task = train_challenges[example_task_id]
            
            # Display example task
            row = category
            show_grid(example_task['train'][0]['input'], 
                     f"Category {category_names[category]}\nInput", axes[row, 0])
            show_grid(example_task['train'][0]['output'], 
                     "Output", axes[row, 1])
            
            # Show features
            task_idx = task_ids.index(example_task_id)
            feature_values = features_list[task_idx][:10]  # First 10 features
            axes[row, 2].bar(range(len(feature_values)), feature_values)
            axes[row, 2].set_title("Key Features")
            axes[row, 2].set_xticks(range(len(feature_values)))
            axes[row, 2].set_xticklabels(['F'+str(i) for i in range(len(feature_values))], 
                                         rotation=45)
    
    plt.tight_layout()
    plt.show()

# Visualize task categories
visualize_task_categories()


def solve_arc_task(task, task_id=None, visualize=True):
    """Complete solution pipeline for solving an ARC task."""
    if visualize:
        print(f"Solving task: {task_id}")
        visualize_task(task, task_id)
    
    # Extract features for task classification
    task_features = extract_task_features(task)
    
    # In practice, we would use our trained classifier here
    # For demonstration, we'll use a simple heuristic based on grid sizes
    train_example = task['train'][0]
    input_size = np.array(train_example['input']).size
    output_size = np.array(train_example['output']).size
    
    if output_size > 4 * input_size:
        # Task involves significant expansion
        strategy = "pattern_expansion"
        max_depth = 1  # Simple transformation
    elif output_size < input_size / 2:
        # Task involves contraction
        strategy = "pattern_contraction"
        max_depth = 1
    elif output_size == input_size:
        # Task likely involves transformation or permutation
        strategy = "pattern_transformation"
        max_depth = 2
    else:
        # Default approach
        strategy = "general"
        max_depth = 2
    
    print(f"Selected strategy: {strategy} (max_depth={max_depth})")
    
    # Execute symbolic search with the appropriate depth
    predictions, program = solve_task_symbolic(task, max_depth=max_depth, time_limit=10)
    
    if program:
        print(f"Found a program with {len(program)} operations")
    else:
        print("No program found, using fallback predictions")
    
    # Display predictions for test cases
    if visualize:
        for i, test_case in enumerate(task['test']):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Input
            show_grid(test_case['input'], "Test Input", axes[0])
            
            # Predictions
            show_grid(predictions[i]['attempt_1'], "Prediction 1", axes[1])
            show_grid(predictions[i]['attempt_2'], "Prediction 2", axes[2])
            
            plt.tight_layout()
            plt.show()
    
    return predictions

# Demonstrate our solution on a few random tasks
for _ in range(2):
    task_id = random.choice(list(eval_challenges.keys()))
    task = eval_challenges[task_id]
    solution = eval_solutions[task_id]
    
    # Solve the task
    predictions = solve_arc_task(task, task_id)
    
    # Compare our predictions with the actual solutions
    print("\nComparing predictions with actual solutions:")
    for i, test_case in enumerate(task['test']):
        pred1 = np.array(predictions[i]['attempt_1'])
        pred2 = np.array(predictions[i]['attempt_2'])
        actual = np.array(solution[i])
        
        match1 = np.array_equal(pred1, actual)
        match2 = np.array_equal(pred2, actual)
        
        print(f"Test case {i+1}: Attempt 1 {'✓' if match1 else '✗'}, "
              f"Attempt 2 {'✓' if match2 else '✗'}")
        
        if match1 or match2:
            print("Successfully solved this test case!")
        else:
            # Show the actual solution
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            show_grid(predictions[i]['attempt_1' if match1 else 'attempt_2'], 
                     "Our Prediction", axes[0])
            show_grid(solution[i], "Actual Solution", axes[1])
            plt.tight_layout()
            plt.show()


def generate_submission(test_challenges):
    """Generate a submission file for the competition."""
    submission = {}
    
    # Process each test task
    for task_id, task_data in tqdm(test_challenges.items(), desc="Generating predictions"):
        # Solve the task
        predictions = solve_arc_task(task_data, task_id, visualize=False)
        submission[task_id] = predictions
    
    # Save the submission
    with open('submission.json', 'w') as f:
        json.dump(submission, f)
    
    print("✅ Submission file created with predictions for", len(submission), "tasks")
    return submission

# Generate the final submission (commented out to avoid accidental execution)
# submission = generate_submission(test_challenges)

