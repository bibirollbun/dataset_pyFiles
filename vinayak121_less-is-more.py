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


"""
ARC Prize 2025 - Fixed and Optimized Recursive Model
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
from collections import defaultdict

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================================
# Dataset - Fixed with proper batching
# ============================================================================

class ARCTaskDataset(Dataset):
    def __init__(self, data_path: str, max_grid_size: int = 30, max_examples: int = 5):
        self.max_grid_size = max_grid_size
        self.max_examples = max_examples
        self.tasks = []
        
        with open(data_path, 'r') as f:
            data = json.load(f)
            for task_id, task_data in data.items():
                if len(task_data['train']) > 0:
                    self.tasks.append({
                        'task_id': task_id,
                        'train': task_data['train']
                    })
    
    def __len__(self):
        return len(self.tasks)
    
    def pad_grid(self, grid):
        grid = np.array(grid)
        h, w = grid.shape
        h = min(h, self.max_grid_size)
        w = min(w, self.max_grid_size)
        
        padded = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.int64)
        padded[:h, :w] = grid[:h, :w]
        return padded
    
    def __getitem__(self, idx):
        task = self.tasks[idx]
        examples = task['train']
        
        # Pad to max_examples
        num_examples = min(len(examples), self.max_examples)
        
        train_inputs = np.zeros((self.max_examples, self.max_grid_size, self.max_grid_size), dtype=np.int64)
        train_outputs = np.zeros((self.max_examples, self.max_grid_size, self.max_grid_size), dtype=np.int64)
        
        for i in range(num_examples):
            train_inputs[i] = self.pad_grid(examples[i]['input'])
            train_outputs[i] = self.pad_grid(examples[i]['output'])
        
        # Pick one example as query
        query_idx = random.randint(0, num_examples - 1)
        
        return {
            'train_inputs': torch.LongTensor(train_inputs),
            'train_outputs': torch.LongTensor(train_outputs),
            'query_input': torch.LongTensor(train_inputs[query_idx]),
            'query_output': torch.LongTensor(train_outputs[query_idx]),
            'num_examples': num_examples
        }

# ============================================================================
# Improved Recursive Model
# ============================================================================

class RecursiveCell(nn.Module):
    def __init__(self, hidden_dim: int, num_colors: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.color_embed = nn.Embedding(num_colors, hidden_dim)
        
        # Add task conditioning
        self.process = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.output_proj = nn.Linear(hidden_dim, num_colors)
    
    def forward(self, grid, task_context, depth=3):
        batch_size, h, w = grid.shape
        hidden = self.color_embed(grid)
        
        # Broadcast task context
        task_context_expanded = task_context.unsqueeze(1).unsqueeze(1).expand(
            batch_size, h, w, self.hidden_dim
        )
        
        for _ in range(depth):
            hidden = self.recursive_step(hidden, task_context_expanded)
        
        logits = self.output_proj(hidden)
        return logits.permute(0, 3, 1, 2)
    
    def recursive_step(self, hidden, task_context):
        batch_size, h, w, hidden_dim = hidden.shape
        
        padded = F.pad(hidden.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='replicate')
        padded = padded.permute(0, 2, 3, 1)
        
        up = padded[:, :-2, 1:-1, :]
        down = padded[:, 2:, 1:-1, :]
        left = padded[:, 1:-1, :-2, :]
        right = padded[:, 1:-1, 2:, :]
        center = hidden
        
        combined = torch.cat([center, up, down, left, right, task_context], dim=-1)
        new_hidden = self.process(combined.reshape(-1, hidden_dim * 6))
        new_hidden = new_hidden.reshape(batch_size, h, w, hidden_dim)
        
        return hidden + new_hidden


class ContextEncoder(nn.Module):
    def __init__(self, hidden_dim, num_colors=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_colors * 2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, inputs, outputs):
        # inputs/outputs: (batch, num_examples, h, w)
        batch_size, num_examples, h, w = inputs.shape
        
        inputs_onehot = F.one_hot(inputs, num_classes=10).float()
        outputs_onehot = F.one_hot(outputs, num_classes=10).float()
        
        # Concat input-output pairs
        pairs = torch.cat([inputs_onehot, outputs_onehot], dim=-1)
        pairs = pairs.permute(0, 1, 4, 2, 3).reshape(batch_size * num_examples, 20, h, w)
        
        # Encode
        features = self.conv(pairs).squeeze(-1).squeeze(-1)
        features = features.reshape(batch_size, num_examples, -1)
        
        # Average pool over examples
        return features.mean(dim=1)


class MetaRecursiveModel(nn.Module):
    def __init__(self, hidden_dim=96, num_colors=10, num_layers=2, depth=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.context_encoder = ContextEncoder(hidden_dim, num_colors)
        
        self.layers = nn.ModuleList([
            RecursiveCell(hidden_dim, num_colors) for _ in range(num_layers)
        ])
        self.depth = depth
    
    def forward(self, query_input, train_inputs, train_outputs):
        # Encode task context
        task_context = self.context_encoder(train_inputs, train_outputs)
        
        # Apply recursive layers
        hidden = query_input
        for layer in self.layers:
            logits = layer(hidden, task_context, depth=self.depth)
            hidden = logits.argmax(dim=1)
        
        return logits

# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_pixels = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        train_inputs = batch['train_inputs'].to(device)
        train_outputs = batch['train_outputs'].to(device)
        query_input = batch['query_input'].to(device)
        query_output = batch['query_output'].to(device)
        
        logits = model(query_input, train_inputs, train_outputs)
        loss = F.cross_entropy(logits, query_output)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        predictions = logits.argmax(dim=1)
        correct = (predictions == query_output).sum().item()
        pixels = query_output.numel()
        
        total_loss += loss.item()
        total_correct += correct
        total_pixels += pixels
    
    return total_loss / len(dataloader), 100.0 * total_correct / total_pixels


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_pixels = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            train_inputs = batch['train_inputs'].to(device)
            train_outputs = batch['train_outputs'].to(device)
            query_input = batch['query_input'].to(device)
            query_output = batch['query_output'].to(device)
            
            logits = model(query_input, train_inputs, train_outputs)
            loss = F.cross_entropy(logits, query_output)
            
            predictions = logits.argmax(dim=1)
            correct = (predictions == query_output).sum().item()
            pixels = query_output.numel()
            
            total_loss += loss.item()
            total_correct += correct
            total_pixels += pixels
    
    return total_loss / len(dataloader), 100.0 * total_correct / total_pixels

# ============================================================================
# Inference with Multiple Strategies
# ============================================================================

def pad_grid(grid, max_size=30):
    grid = np.array(grid)
    h, w = grid.shape
    h = min(h, max_size)
    w = min(w, max_size)
    padded = np.zeros((max_size, max_size), dtype=np.int64)
    padded[:h, :w] = grid[:h, :w]
    return padded


def predict_task(model, train_examples, test_input, device, max_examples=5):
    model.eval()
    
    # Prepare training context
    num_train = min(len(train_examples), max_examples)
    train_inputs = np.zeros((max_examples, 30, 30), dtype=np.int64)
    train_outputs = np.zeros((max_examples, 30, 30), dtype=np.int64)
    
    for i in range(num_train):
        train_inputs[i] = pad_grid(train_examples[i]['input'])
        train_outputs[i] = pad_grid(train_examples[i]['output'])
    
    # Prepare test input
    test_inp = np.array(test_input)
    h_test, w_test = test_inp.shape
    test_padded = pad_grid(test_input)
    
    # Convert to tensors
    train_inputs_t = torch.LongTensor(train_inputs).unsqueeze(0).to(device)
    train_outputs_t = torch.LongTensor(train_outputs).unsqueeze(0).to(device)
    test_t = torch.LongTensor(test_padded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(test_t, train_inputs_t, train_outputs_t)
        pred = logits.argmax(dim=1)[0].cpu().numpy()
    
    # Estimate output size from training examples
    h_ratios = []
    w_ratios = []
    for ex in train_examples[:num_train]:
        in_h, in_w = np.array(ex['input']).shape
        out_h, out_w = np.array(ex['output']).shape
        h_ratios.append(out_h / in_h)
        w_ratios.append(out_w / in_w)
    
    avg_h_ratio = np.median(h_ratios)
    avg_w_ratio = np.median(w_ratios)
    
    out_h = max(1, min(30, int(h_test * avg_h_ratio)))
    out_w = max(1, min(30, int(w_test * avg_w_ratio)))
    
    return pred[:out_h, :out_w]


def create_submission(model, test_path, device, output_file='submission.json'):
    model.eval()
    
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    submission = {}
    
    print(f"Processing {len(test_data)} test tasks...")
    
    for task_id, task_data in tqdm(test_data.items(), desc="Creating submission"):
        task_predictions = []
        
        for test_input_dict in task_data['test']:
            test_input = test_input_dict['input']
            
            # Strategy 1: Direct prediction
            pred1 = predict_task(model, task_data['train'], test_input, device)
            
            # Strategy 2: Try with augmented input (flip)
            test_input_flipped = np.array(test_input)[:, ::-1].tolist()
            pred2_flipped = predict_task(model, task_data['train'], test_input_flipped, device)
            pred2 = pred2_flipped[:, ::-1]  # Flip back
            
            task_predictions.append({
                'attempt_1': pred1.tolist(),
                'attempt_2': pred2.tolist()
            })
        
        submission[task_id] = task_predictions
    
    # Save with error handling
    try:
        with open(output_file, 'w') as f:
            json.dump(submission, f, indent=2)
        print(f"âœ“ Submission saved to {output_file}")
        
        # Verify file size
        import os
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"âœ“ File size: {file_size:.2f} MB")
        
        # Verify format
        with open(output_file, 'r') as f:
            verify = json.load(f)
        print(f"âœ“ Verified {len(verify)} tasks in submission")
        
        # Save to multiple locations
        with open('/kaggle/working/submission.json', 'w') as f:
            json.dump(submission, f)
        print(f"âœ“ Also saved to /kaggle/working/submission.json")
        
    except Exception as e:
        print(f"Error saving submission: {e}")
        
    return submission

# ============================================================================
# Main
# ============================================================================

def main():
    TRAIN_PATH = '/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json'
    EVAL_PATH = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json'
    TEST_PATH = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    train_dataset = ARCTaskDataset(TRAIN_PATH, max_examples=5)
    val_dataset = ARCTaskDataset(EVAL_PATH, max_examples=5)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Train tasks: {len(train_dataset)}, Val tasks: {len(val_dataset)}")
    
    # Model
    model = MetaRecursiveModel(hidden_dim=96, num_layers=2, depth=2).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    
    best_acc = 0
    patience = 0
    max_patience = 5
    
    for epoch in range(25):
        print(f"\nEpoch {epoch + 1}/25")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"âœ“ Saved best model (Val Acc: {best_acc:.2f}%)")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    print(f"\nBest validation accuracy: {best_acc:.2f}%")
    
    # Create submission
    create_submission(model, TEST_PATH, device)
    
    print("\n" + "="*60)
    print("âœ“ SUBMISSION COMPLETE!")
    print("="*60)
    print("\nFILE LOCATIONS:")
    print("  1. submission.json (current directory)")
    print("  2. /kaggle/working/submission.json")
    print("\nTO SUBMIT:")
    print("  Option 1: Click 'Save Version' â†’ 'Save & Run All' â†’ Submit output")
    print("  Option 2: Download submission.json and upload manually")
    print("  Option 3: Use Kaggle API (see below)")
    print("\nKAGGLE API SUBMISSION:")
    print("  kaggle competitions submit -c arc-prize-2025 -f submission.json -m 'Recursive Model'")
    print("="*60)

if __name__ == '__main__':
    main()
    
    # Additional helper: Display submission info
    try:
        import os
        if os.path.exists('submission.json'):
            with open('submission.json', 'r') as f:
                sub = json.load(f)
            print(f"\nðŸ“Š SUBMISSION STATS:")
            print(f"   Total tasks: {len(sub)}")
            total_attempts = sum(len(preds) for preds in sub.values())
            print(f"   Total test cases: {total_attempts}")
            print(f"   Ready for submission: âœ“")
    except:
        pass

