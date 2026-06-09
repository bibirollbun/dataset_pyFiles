# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ARC-AGI-2 Neural Network Solution
# Complete pipeline for ARC Prize 2025 competition

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms

# Install required packages
import subprocess
import sys


def load_arc_data(file_path):
    """Load ARC-AGI data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} tasks from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")


# Load data files
train_data = load_arc_data('/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json')
train_solutions = load_arc_data('/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json')
eval_data = load_arc_data('/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json')
eval_solutions = load_arc_data('/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json')
test_data = load_arc_data('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json')

print("Data loading completed!")


def analyze_grid_properties(data):
    """Analyze properties of grids in the dataset"""
    grid_sizes = []
    num_colors = []
    
    for task_id, task in data.items():
        for pair in task['train']:
            input_grid = pair['input']
            output_grid = pair['output']
            
            # Grid dimensions
            grid_sizes.append((len(input_grid), len(input_grid[0])))
            grid_sizes.append((len(output_grid), len(output_grid[0])))
            
            # Number of unique colors
            input_colors = set()
            output_colors = set()
            for row in input_grid:
                input_colors.update(row)
            for row in output_grid:
                output_colors.update(row)
            
            num_colors.append(len(input_colors))
            num_colors.append(len(output_colors))
    
    return grid_sizes, num_colors

def visualize_task(task_data, task_id):
    """Visualize a single ARC task"""
    task = task_data[task_id]
    
    fig, axes = plt.subplots(2, len(task['train']) + 1, figsize=(15, 8))
    
    # Plot training examples
    for i, pair in enumerate(task['train']):
        # Input
        axes[0, i].imshow(pair['input'], cmap='tab10', vmin=0, vmax=9)
        axes[0, i].set_title(f'Train {i+1} Input')
        axes[0, i].axis('off')
        
        # Output
        axes[1, i].imshow(pair['output'], cmap='tab10', vmin=0, vmax=9)
        axes[1, i].set_title(f'Train {i+1} Output')
        axes[1, i].axis('off')
    
    # Plot test input
    test_input = task['test'][0]['input']
    axes[0, -1].imshow(test_input, cmap='tab10', vmin=0, vmax=9)
    axes[0, -1].set_title('Test Input')
    axes[0, -1].axis('off')
    
    # Empty test output (to be predicted)
    axes[1, -1].text(0.5, 0.5, 'TO PREDICT', ha='center', va='center', 
                     transform=axes[1, -1].transAxes, fontsize=12)
    axes[1, -1].set_title('Test Output')
    axes[1, -1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Analyze dataset properties
if train_data:
    grid_sizes, num_colors = analyze_grid_properties(train_data)
    
    print(f"Grid size range: {min(grid_sizes)} to {max(grid_sizes)}")
    print(f"Number of colors range: {min(num_colors)} to {max(num_colors)}")
    
    # Visualize first task
    first_task_id = list(train_data.keys())[0]
    print(f"Visualizing task: {first_task_id}")
    visualize_task(train_data, first_task_id)


class ARCDataset(Dataset):
    """Custom Dataset class for ARC-AGI tasks"""
    
    def __init__(self, tasks_data, solutions_data=None, max_grid_size=30):
        self.tasks_data = tasks_data
        self.solutions_data = solutions_data
        self.max_grid_size = max_grid_size
        self.task_ids = list(tasks_data.keys())
        
        # Prepare training examples
        self.examples = []
        self.prepare_examples()
    
    def prepare_examples(self):
        """Prepare training examples from tasks"""
        for task_id in self.task_ids:
            task = self.tasks_data[task_id]
            
            # Use training pairs for supervised learning
            for pair in task['train']:
                input_grid = self.pad_grid(pair['input'])
                output_grid = self.pad_grid(pair['output'])
                
                self.examples.append({
                    'task_id': task_id,
                    'input': input_grid,
                    'output': output_grid,
                    'is_test': False
                })
            
            # Add test examples (for inference)
            for test_pair in task['test']:
                input_grid = self.pad_grid(test_pair['input'])
                
                # Get solution if available
                output_grid = None
                if self.solutions_data and task_id in self.solutions_data:
                    if len(self.solutions_data[task_id]) > 0:
                        output_grid = self.pad_grid(self.solutions_data[task_id][0])
                
                self.examples.append({
                    'task_id': task_id,
                    'input': input_grid,
                    'output': output_grid,
                    'is_test': True
                })
    
    def pad_grid(self, grid):
        """Pad grid to maximum size"""
        grid = np.array(grid)
        h, w = grid.shape
        
        if h > self.max_grid_size or w > self.max_grid_size:
            # Truncate if larger than max size
            grid = grid[:self.max_grid_size, :self.max_grid_size]
            h, w = grid.shape
        
        # Pad to max size
        padded = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.int32)
        padded[:h, :w] = grid
        
        return padded
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        input_tensor = torch.tensor(example['input'], dtype=torch.long)
        
        if example['output'] is not None:
            output_tensor = torch.tensor(example['output'], dtype=torch.long)
        else:
            output_tensor = torch.zeros_like(input_tensor)
        
        return {
            'input': input_tensor,
            'output': output_tensor,
            'task_id': example['task_id'],
            'is_test': example['is_test']
        }


# Create datasets
train_dataset = ARCDataset(train_data, train_solutions)
eval_dataset = ARCDataset(eval_data, eval_solutions)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Evaluation dataset size: {len(eval_dataset)}")


class ConvBlock(nn.Module):
    """Convolutional block with residual connections"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out

class SelfAttention(nn.Module):
    """Self-attention mechanism for spatial reasoning"""
    
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        proj_query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)
        proj_value = self.value_conv(x).view(batch_size, -1, H * W)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        out = self.gamma * out + x
        return out

class ARCNet(nn.Module):
    """Neural Network for ARC-AGI tasks"""
    
    def __init__(self, num_colors=10, hidden_dim=128, max_grid_size=30):
        super(ARCNet, self).__init__()
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        self.max_grid_size = max_grid_size
        
        # Embedding layer for color tokens
        self.embedding = nn.Embedding(num_colors, hidden_dim)
        
        # Convolutional encoder
        self.conv_blocks = nn.ModuleList([
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim * 2),
            ConvBlock(hidden_dim * 2, hidden_dim * 2),
            ConvBlock(hidden_dim * 2, hidden_dim * 4)
        ])
        
        # Self-attention for reasoning
        self.attention = SelfAttention(hidden_dim * 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, num_colors, 3, padding=1)
        )
        
        # Output layer
        self.output_layer = nn.Linear(num_colors * max_grid_size * max_grid_size, 
                                     num_colors * max_grid_size * max_grid_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Embed input tokens
        x = self.embedding(x)  # (batch, H, W, hidden_dim)
        x = x.permute(0, 3, 1, 2)  # (batch, hidden_dim, H, W)
        
        # Convolutional encoding
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Self-attention
        x = self.attention(x)
        
        # Decode
        x = self.decoder(x)
        
        # Flatten and apply final transformation
        x = x.view(batch_size, -1)
        x = self.output_layer(x)
        x = x.view(batch_size, self.num_colors, self.max_grid_size, self.max_grid_size)
        
        return x


def setup_training():
    """Setup training configuration"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = ARCNet(num_colors=10, hidden_dim=128, max_grid_size=30)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=5)
    
    return model, criterion, optimizer, scheduler, device

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        # Skip test examples during training
        mask = ~batch['is_test']
        if not mask.any():
            continue
            
        inputs = batch['input'][mask].to(device)
        targets = batch['output'][mask].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Skip test examples during validation
            mask = ~batch['is_test']
            if not mask.any():
                continue
                
            inputs = batch['input'][mask].to(device)
            targets = batch['output'][mask].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def train_model(model, train_dataset, eval_dataset, num_epochs=50, batch_size=8):
    """Main training function"""
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup training
    model, criterion, optimizer, scheduler, device = setup_training()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validation
        val_loss = validate_epoch(model, eval_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_arc_model.pth')
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Best Val Loss: {best_val_loss:.4f}")
            print("-" * 50)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()
    
    return model

# Train the model
trained_model = train_model(ARCNet(), train_dataset, eval_dataset, num_epochs=2)
print("Training completed!")


def predict_output(model, input_grid, device, num_attempts=2):
    """Predict output for a given input grid"""
    model.eval()
    
    # Preprocess input
    if isinstance(input_grid, list):
        input_grid = np.array(input_grid)
    
    # Pad to max size
    max_size = 30
    h, w = input_grid.shape
    padded_input = np.zeros((max_size, max_size), dtype=np.int32)
    padded_input[:h, :w] = input_grid
    
    # Convert to tensor
    input_tensor = torch.tensor(padded_input, dtype=torch.long).unsqueeze(0).to(device)
    
    predictions = []
    
    with torch.no_grad():
        for attempt in range(num_attempts):
            # Add some randomness for multiple attempts
            if attempt > 0:
                noise = torch.randn_like(input_tensor.float()) * 0.1
                noisy_input = input_tensor.float() + noise
                noisy_input = torch.clamp(noisy_input, 0, 9).long()
            else:
                noisy_input = input_tensor
            
            output = model(noisy_input)
            predicted = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            # Crop back to reasonable size (heuristic)
            predicted_cropped = predicted[:h*2, :w*2]  # Allow for size changes
            predictions.append(predicted_cropped.tolist())
    
    return predictions

def generate_submission(model, test_data, device):
    """Generate submission file"""
    submission = {}
    
    print("Generating predictions for test data...")
    
    for task_id, task in test_data.items():
        submission[task_id] = []
        
        for test_case in task['test']:
            input_grid = test_case['input']
            predictions = predict_output(model, input_grid, device, num_attempts=2)
            submission[task_id].append(predictions)
        
        if len(submission) % 10 == 0:
            print(f"Processed {len(submission)} tasks...")
    
    return submission


# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate submission
if test_data:
    print("Generating final submission...")
    submission = generate_submission(trained_model, test_data, device)
    
    # Save submission file
    with open('submission.json', 'w') as f:
        json.dump(submission, f)
    
    print(f"Submission saved! Generated predictions for {len(submission)} tasks.")
    print("Submission format:")
    print("- Each task has 2 attempts per test case")
    print("- Predictions are in the format: task_id -> [attempt1, attempt2]")
    
    # Show sample prediction
    sample_task_id = list(submission.keys())[0]
    sample_prediction = submission[sample_task_id]
    print(f"\nSample prediction for task {sample_task_id}:")
    print(f"Number of test cases: {len(sample_prediction)}")
    print(f"Number of attempts per test case: {len(sample_prediction[0])}")
    
else:
    print("No test data available. Creating empty submission file.")
    with open('submission.json', 'w') as f:
        json.dump({}, f)


def calculate_accuracy(predictions, ground_truth):
    """Calculate exact match accuracy"""
    if len(predictions) != len(ground_truth):
        return 0.0
    
    for pred_row, true_row in zip(predictions, ground_truth):
        if len(pred_row) != len(true_row):
            return 0.0
        for pred_cell, true_cell in zip(pred_row, true_row):
            if pred_cell != true_cell:
                return 0.0
    return 1.0

def evaluate_model(model, eval_data, eval_solutions, device):
    """Evaluate model on validation set"""
    total_tasks = 0
    correct_tasks = 0
    
    for task_id, task in eval_data.items():
        if task_id not in eval_solutions:
            continue
            
        solutions = eval_solutions[task_id]
        total_tasks += 1
        
        for i, test_case in enumerate(task['test']):
            if i >= len(solutions):
                break
                
            input_grid = test_case['input']
            true_output = solutions[i]
            
            predictions = predict_output(model, input_grid, device, num_attempts=2)
            
            # Check if any attempt is correct (pass@2 evaluation)
            for prediction in predictions:
                if calculate_accuracy(prediction, true_output) == 1.0:
                    correct_tasks += 1
                    break
            break  # Only evaluate first test case per task for speed
    
    accuracy = correct_tasks / max(total_tasks, 1)
    return accuracy

# Evaluate on validation set if available
if eval_data and eval_solutions:
    print("\nEvaluating model on validation set...")
    val_accuracy = evaluate_model(trained_model, eval_data, eval_solutions, device)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

print("\n" + "="*60)
print("ARC-AGI-2 SUBMISSION COMPLETE!")
print("="*60)




