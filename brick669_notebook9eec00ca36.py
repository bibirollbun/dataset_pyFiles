#!/usr/bin/env python3
"""
ARC-AGI Prize 2025 - Enhanced Solver
Hybrid approach with Deep RL + Advanced Pattern Matching
Completely silent execution, produces submission.json
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import numpy as np
from collections import deque, Counter
import random
from scipy import ndimage
from scipy.signal import convolve2d

# Disable all output
import sys
import io

class SilentExecution:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self
    
    def __exit__(self, *args):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# ==================== NEURAL NETWORK MODELS ====================

class ImprovedGridEncoder(nn.Module):
    """Enhanced CNN encoder with residual connections and attention"""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Multi-scale convolution paths
        self.conv1 = nn.Conv2d(10, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        # Residual connections
        self.res1 = nn.Conv2d(64, 128, 1)
        self.res2 = nn.Conv2d(128, 256, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Global pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature projection
        self.fc = nn.Linear(256, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch, 10, height, width)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = x2 + self.res1(x1)  # Residual connection
        
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x3 = x3 + self.res2(x2)  # Residual connection
        
        # Global pooling
        x_pooled = self.adaptive_pool(x3).squeeze(-1).squeeze(-1)
        
        # Feature projection
        features = F.relu(self.fc(x_pooled))
        features = self.dropout(features)
        
        return features

class ImprovedPolicyNetwork(nn.Module):
    """Enhanced policy network with multiple action heads"""
    def __init__(self, input_dim=256, n_actions=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

class ImprovedValueNetwork(nn.Module):
    """Enhanced value network for state evaluation"""
    def __init__(self, input_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        value = self.fc3(x)
        return value

# ==================== ADVANCED TRANSFORMATIONS ====================

class AdvancedTransformations:
    """Comprehensive set of grid transformations"""
    
    @staticmethod
    def apply_transformation(grid, action_id, param=None):
        """Apply transformation based on action ID"""
        grid_array = np.array(grid, dtype=int)
        h, w = grid_array.shape
        
        transformations = {
            0: lambda g: np.rot90(g, 1),  # Rotate 90° CW
            1: lambda g: np.rot90(g, 2),  # Rotate 180°
            2: lambda g: np.rot90(g, 3),  # Rotate 270° CW
            3: lambda g: np.fliplr(g),  # Flip horizontal
            4: lambda g: np.flipud(g),  # Flip vertical
            5: lambda g: np.transpose(g),  # Transpose
            6: lambda g: AdvancedTransformations.color_swap(g),
            7: lambda g: AdvancedTransformations.extract_objects(g),
            8: lambda g: AdvancedTransformations.fill_pattern(g),
            9: lambda g: AdvancedTransformations.scale_up(g),
            10: lambda g: AdvancedTransformations.scale_down(g),
            11: lambda g: AdvancedTransformations.repeat_pattern(g),
            12: lambda g: AdvancedTransformations.overlay_patterns(g),
            13: lambda g: AdvancedTransformations.connect_components(g),
            14: lambda g: AdvancedTransformations.mirror_extend(g),
            15: lambda g: AdvancedTransformations.crop_to_content(g),
            16: lambda g: AdvancedTransformations.symmetrize(g),
            17: lambda g: AdvancedTransformations.cellular_automata(g),
            18: lambda g: AdvancedTransformations.pattern_completion(g),
            19: lambda g: g.copy(),  # Identity
        }
        
        try:
            if action_id in transformations:
                result = transformations[action_id](grid_array)
                return result.tolist() if isinstance(result, np.ndarray) else result
        except:
            pass
        
        return grid
    
    @staticmethod
    def color_swap(grid):
        """Swap most common colors"""
        colors = grid.flatten()
        counter = Counter(colors)
        if len(counter) >= 2:
            most_common = counter.most_common(2)
            c1, c2 = most_common[0][0], most_common[1][0]
            result = grid.copy()
            result[grid == c1] = -1
            result[grid == c2] = c1
            result[result == -1] = c2
            return result
        return grid
    
    @staticmethod
    def extract_objects(grid):
        """Extract and isolate objects"""
        background = np.bincount(grid.flatten()).argmax()
        mask = grid != background
        
        if not mask.any():
            return grid
        
        # Find bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return grid
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return grid[rmin:rmax+1, cmin:cmax+1]
    
    @staticmethod
    def fill_pattern(grid):
        """Fill enclosed regions"""
        result = grid.copy()
        h, w = grid.shape
        
        for color in range(10):
            mask = (grid == color)
            if mask.any():
                filled = ndimage.binary_fill_holes(mask)
                result[filled] = color
        
        return result
    
    @staticmethod
    def scale_up(grid):
        """Scale up by 2x"""
        return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)
    
    @staticmethod
    def scale_down(grid):
        """Scale down by 2x"""
        h, w = grid.shape
        if h % 2 == 0 and w % 2 == 0:
            return grid[::2, ::2]
        return grid
    
    @staticmethod
    def repeat_pattern(grid):
        """Repeat pattern 2x2"""
        return np.tile(grid, (2, 2))
    
    @staticmethod
    def overlay_patterns(grid):
        """Overlay detected patterns"""
        result = grid.copy()
        h, w = grid.shape
        
        # Find repeating patterns
        for size in [2, 3]:
            if h >= size and w >= size:
                pattern = grid[:size, :size]
                for i in range(0, h-size+1, size):
                    for j in range(0, w-size+1, size):
                        if np.array_equal(grid[i:i+size, j:j+size], pattern):
                            result[i:i+size, j:j+size] = pattern
        
        return result
    
    @staticmethod
    def connect_components(grid):
        """Connect nearby components"""
        result = grid.copy()
        background = np.bincount(grid.flatten()).argmax()
        
        for color in range(10):
            if color == background:
                continue
            
            mask = (grid == color)
            if mask.sum() > 1:
                labeled, num = ndimage.label(mask)
                
                if num > 1:
                    # Connect components
                    dilated = ndimage.binary_dilation(mask, iterations=1)
                    result[dilated] = color
        
        return result
    
    @staticmethod
    def mirror_extend(grid):
        """Mirror and extend pattern"""
        h, w = grid.shape
        
        # Try horizontal mirror
        if w < 15:
            mirrored = np.concatenate([grid, np.fliplr(grid)], axis=1)
            if mirrored.shape[1] <= 30:
                return mirrored
        
        # Try vertical mirror
        if h < 15:
            mirrored = np.concatenate([grid, np.flipud(grid)], axis=0)
            if mirrored.shape[0] <= 30:
                return mirrored
        
        return grid
    
    @staticmethod
    def crop_to_content(grid):
        """Crop to non-background content"""
        background = np.bincount(grid.flatten()).argmax()
        mask = grid != background
        
        if not mask.any():
            return grid
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return grid
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return grid[rmin:rmax+1, cmin:cmax+1]
    
    @staticmethod
    def symmetrize(grid):
        """Make grid symmetric"""
        h, w = grid.shape
        result = grid.copy()
        
        # Horizontal symmetry
        if h == w:
            result = (result + np.fliplr(result)) // 2
        
        return result
    
    @staticmethod
    def cellular_automata(grid):
        """Apply simple cellular automata rule"""
        result = grid.copy()
        h, w = grid.shape
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                neighbors = [
                    grid[i-1, j-1], grid[i-1, j], grid[i-1, j+1],
                    grid[i, j-1], grid[i, j+1],
                    grid[i+1, j-1], grid[i+1, j], grid[i+1, j+1]
                ]
                
                most_common = Counter(neighbors).most_common(1)[0][0]
                if Counter(neighbors)[most_common] >= 5:
                    result[i, j] = most_common
        
        return result
    
    @staticmethod
    def pattern_completion(grid):
        """Complete partial patterns"""
        result = grid.copy()
        h, w = grid.shape
        background = np.bincount(grid.flatten()).argmax()
        
        # Detect and complete simple patterns
        for i in range(h):
            row = result[i, :]
            non_bg = row[row != background]
            if len(non_bg) > 0:
                most_common = Counter(non_bg).most_common(1)[0][0]
                result[i, row == background] = most_common
        
        return result

# ==================== PATTERN MATCHING ====================

class EnhancedPatternMatcher:
    """Advanced pattern detection and matching"""
    
    @staticmethod
    def detect_pattern_type(examples):
        """Detect the type of transformation pattern"""
        patterns = {
            'rotation': EnhancedPatternMatcher.is_rotation,
            'flip': EnhancedPatternMatcher.is_flip,
            'scale': EnhancedPatternMatcher.is_scale,
            'color_map': EnhancedPatternMatcher.is_color_map,
            'tiling': EnhancedPatternMatcher.is_tiling,
            'extraction': EnhancedPatternMatcher.is_extraction,
            'completion': EnhancedPatternMatcher.is_completion,
        }
        
        scores = {}
        for pattern_name, pattern_func in patterns.items():
            score = sum(1 for ex in examples if pattern_func(ex['input'], ex['output']))
            scores[pattern_name] = score / len(examples) if examples else 0
        
        best_pattern = max(scores.items(), key=lambda x: x[1])
        return best_pattern[0] if best_pattern[1] > 0.5 else None
    
    @staticmethod
    def is_rotation(input_grid, output_grid):
        """Check if output is rotation of input"""
        input_arr = np.array(input_grid)
        output_arr = np.array(output_grid)
        
        for k in range(1, 4):
            if np.array_equal(np.rot90(input_arr, k), output_arr):
                return True
        return False
    
    @staticmethod
    def is_flip(input_grid, output_grid):
        """Check if output is flip of input"""
        input_arr = np.array(input_grid)
        output_arr = np.array(output_grid)
        
        return (np.array_equal(np.fliplr(input_arr), output_arr) or
                np.array_equal(np.flipud(input_arr), output_arr) or
                np.array_equal(np.transpose(input_arr), output_arr))
    
    @staticmethod
    def is_scale(input_grid, output_grid):
        """Check if output is scaled version of input"""
        ih, iw = len(input_grid), len(input_grid[0])
        oh, ow = len(output_grid), len(output_grid[0])
        
        return (oh % ih == 0 and ow % iw == 0 and oh == ow) or \
               (ih % oh == 0 and iw % ow == 0 and ih == iw)
    
    @staticmethod
    def is_color_map(input_grid, output_grid):
        """Check if output is color mapping of input"""
        input_arr = np.array(input_grid)
        output_arr = np.array(output_grid)
        
        if input_arr.shape != output_arr.shape:
            return False
        
        unique_in = set(input_arr.flatten())
        unique_out = set(output_arr.flatten())
        
        return len(unique_in) == len(unique_out)
    
    @staticmethod
    def is_tiling(input_grid, output_grid):
        """Check if output is tiled version of input"""
        ih, iw = len(input_grid), len(input_grid[0])
        oh, ow = len(output_grid), len(output_grid[0])
        
        return oh > ih and ow > iw and oh % ih == 0 and ow % iw == 0
    
    @staticmethod
    def is_extraction(input_grid, output_grid):
        """Check if output is extracted from input"""
        ih, iw = len(input_grid), len(input_grid[0])
        oh, ow = len(output_grid), len(output_grid[0])
        
        return oh < ih or ow < iw
    
    @staticmethod
    def is_completion(input_grid, output_grid):
        """Check if output completes input pattern"""
        input_arr = np.array(input_grid)
        output_arr = np.array(output_grid)
        
        if input_arr.shape != output_arr.shape:
            return False
        
        # Check if output has fewer background pixels
        bg_in = (input_arr == 0).sum()
        bg_out = (output_arr == 0).sum()
        
        return bg_out < bg_in
    
    @staticmethod
    def apply_pattern(pattern_type, test_input):
        """Apply detected pattern to test input"""
        if pattern_type == 'rotation':
            return np.rot90(np.array(test_input), random.choice([1, 2, 3])).tolist()
        
        elif pattern_type == 'flip':
            choice = random.choice(['lr', 'ud', 'transpose'])
            arr = np.array(test_input)
            if choice == 'lr':
                return np.fliplr(arr).tolist()
            elif choice == 'ud':
                return np.flipud(arr).tolist()
            else:
                return np.transpose(arr).tolist()
        
        elif pattern_type == 'scale':
            return AdvancedTransformations.scale_up(np.array(test_input)).tolist()
        
        elif pattern_type == 'tiling':
            return AdvancedTransformations.repeat_pattern(np.array(test_input)).tolist()
        
        elif pattern_type == 'extraction':
            return AdvancedTransformations.extract_objects(np.array(test_input)).tolist()
        
        elif pattern_type == 'completion':
            return AdvancedTransformations.pattern_completion(np.array(test_input)).tolist()
        
        return test_input

# ==================== RL AGENT ====================

class ImprovedRLAgent:
    """Enhanced reinforcement learning agent"""
    
    def __init__(self, device='cuda', n_actions=20):
        self.device = device
        self.n_actions = n_actions
        
        # Networks
        self.encoder = ImprovedGridEncoder().to(device)
        self.policy_net = ImprovedPolicyNetwork(n_actions=n_actions).to(device)
        self.value_net = ImprovedValueNetwork().to(device)
        
        # Training parameters
        self.gamma = 0.99
        self.lr = 1e-4
        
        # Optimizers
        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(), lr=self.lr, weight_decay=1e-5
        )
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=self.lr, weight_decay=1e-5
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_net.parameters(), lr=self.lr, weight_decay=1e-5
        )
        
        # Memory
        self.memory = deque(maxlen=10000)
        self.training_rewards = []
        
        # Set to eval mode by default
        self.encoder.eval()
        self.policy_net.eval()
        self.value_net.eval()
    
    def grid_to_tensor(self, grid):
        """Convert grid to one-hot tensor"""
        grid_array = np.array(grid, dtype=int)
        h, w = grid_array.shape
        
        # Pad to fixed size
        max_size = 30
        padded = np.zeros((max_size, max_size), dtype=int)
        padded[:h, :w] = grid_array
        
        # One-hot encode
        one_hot = np.zeros((10, max_size, max_size), dtype=np.float32)
        for i in range(10):
            one_hot[i] = (padded == i).astype(np.float32)
        
        return torch.FloatTensor(one_hot).unsqueeze(0).to(self.device)
    
    def select_action(self, grid, epsilon=0.0):
        """Select action using policy network"""
        with torch.no_grad():
            grid_tensor = self.grid_to_tensor(grid)
            features = self.encoder(grid_tensor)
            action_probs = self.policy_net(features)
            
            if random.random() < epsilon:
                action = random.randint(0, self.n_actions - 1)
            else:
                action = torch.argmax(action_probs, dim=1).item()
            
            return action
    
    def train_on_task(self, examples, epochs=50, batch_size=8):
        """Train on a single task"""
        self.encoder.train()
        self.policy_net.train()
        self.value_net.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for example in examples:
                input_grid = example['input']
                target_grid = example['output']
                
                # Generate trajectory
                current_grid = input_grid
                trajectory = []
                
                for step in range(10):
                    action = self.select_action(current_grid, epsilon=0.3)
                    next_grid = AdvancedTransformations.apply_transformation(
                        current_grid, action
                    )
                    
                    reward = self.compute_reward(next_grid, target_grid)
                    trajectory.append((current_grid, action, reward, next_grid))
                    
                    current_grid = next_grid
                    
                    if reward > 0.9:
                        break
                
                # Store trajectory
                self.memory.extend(trajectory)
            
            # Train on batch
            if len(self.memory) >= batch_size:
                batch = random.sample(self.memory, batch_size)
                loss = self.train_on_batch(batch)
                total_loss += loss
            
            self.training_rewards.append(total_loss / len(examples) if examples else 0)
        
        self.encoder.eval()
        self.policy_net.eval()
        self.value_net.eval()
    
    def train_on_batch(self, batch):
        """Train on a batch of experiences"""
        total_loss = 0
        
        for state, action, reward, next_state in batch:
            # Compute features
            state_tensor = self.grid_to_tensor(state)
            next_state_tensor = self.grid_to_tensor(next_state)
            
            # Forward pass
            state_features = self.encoder(state_tensor)
            next_state_features = self.encoder(next_state_tensor)
            
            # Value loss
            value = self.value_net(state_features)
            next_value = self.value_net(next_state_features)
            target_value = reward + self.gamma * next_value.detach()
            value_loss = F.mse_loss(value, target_value)
            
            # Policy loss
            action_probs = self.policy_net(state_features)
            action_tensor = torch.LongTensor([action]).to(self.device)
            policy_loss = -torch.log(action_probs[0, action_tensor] + 1e-8) * \
                         (target_value - value).detach()
            
            # Total loss
            loss = value_loss + policy_loss.mean()
            
            # Backward pass
            self.encoder_optimizer.zero_grad()
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
            
            self.encoder_optimizer.step()
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(batch) if batch else 0
    
    def compute_reward(self, predicted, target):
        """Compute reward for a prediction"""
        pred_arr = np.array(predicted, dtype=int)
        target_arr = np.array(target, dtype=int)
        
        # Size match bonus
        size_match = (pred_arr.shape == target_arr.shape)
        if not size_match:
            return -0.5
        
        # Exact match
        if np.array_equal(pred_arr, target_arr):
            return 1.0
        
        # Partial match
        matches = (pred_arr == target_arr).sum()
        total = pred_arr.size
        
        return matches / total - 0.1
    
    def predict(self, test_input, examples, max_steps=20, num_attempts=3):
        """Generate predictions for test input"""
        predictions = []
        
        # Try different strategies
        for attempt in range(num_attempts):
            current_grid = test_input
            best_grid = current_grid
            best_score = -float('inf')
            
            for step in range(max_steps):
                action = self.select_action(current_grid, epsilon=0.1 * attempt)
                next_grid = AdvancedTransformations.apply_transformation(
                    current_grid, action
                )
                
                # Evaluate with examples
                score = self.evaluate_with_examples(next_grid, examples)
                
                if score > best_score:
                    best_score = score
                    best_grid = next_grid
                
                current_grid = next_grid
                
                # Early stopping if good match
                if score > 0.8:
                    break
            
            if is_valid_grid(best_grid):
                predictions.append(best_grid)
        
        return predictions
    
    def evaluate_with_examples(self, grid, examples):
        """Evaluate grid similarity with training examples"""
        if not examples:
            return 0
        
        scores = []
        for ex in examples:
            score = self.compute_reward(grid, ex['output'])
            scores.append(score)
        
        return max(scores) if scores else 0

# ==================== HYBRID SOLVER ====================

class ImprovedHybridSolver:
    """Enhanced hybrid solver combining RL and pattern matching"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.rl_agent = ImprovedRLAgent(device=device)
        self.pattern_matcher = EnhancedPatternMatcher()
    
    def train_on_training_set(self, challenges, max_tasks=150):
        """Train on multiple tasks"""
        task_ids = list(challenges.keys())[:max_tasks]
        
        for task_id in task_ids:
            try:
                challenge = challenges[task_id]
                if 'train' in challenge and len(challenge['train']) > 0:
                    self.rl_agent.train_on_task(challenge['train'], epochs=30)
            except:
                continue
    
    def solve_challenge(self, challenge):
        """Solve a challenge using hybrid approach"""
        predictions = []
        
        # Detect pattern type
        pattern_type = self.pattern_matcher.detect_pattern_type(
            challenge.get('train', [])
        )
        
        for test_input in challenge.get('test', []):
            test_input_data = test_input.get('input', test_input)
            
            # Strategy 1: Pattern matching
            if pattern_type:
                pattern_pred = self.pattern_matcher.apply_pattern(
                    pattern_type, test_input_data
                )
                if is_valid_grid(pattern_pred):
                    predictions.append({
                        'attempt_1': pattern_pred,
                        'attempt_2': pattern_pred
                    })
                    continue
            
            # Strategy 2: RL predictions
            rl_preds = self.rl_agent.predict(
                test_input_data,
                challenge.get('train', []),
                max_steps=20,
                num_attempts=5
            )
            
            # Filter valid predictions
            valid_preds = [p for p in rl_preds if is_valid_grid(p)]
            
            if len(valid_preds) >= 2:
                predictions.append({
                    'attempt_1': valid_preds[0],
                    'attempt_2': valid_preds[1]
                })
            elif len(valid_preds) == 1:
                # Try simple transformations for second attempt
                second = AdvancedTransformations.apply_transformation(
                    valid_preds[0], random.randint(0, 19)
                )
                if is_valid_grid(second):
                    predictions.append({
                        'attempt_1': valid_preds[0],
                        'attempt_2': second
                    })
                else:
                    predictions.append({
                        'attempt_1': valid_preds[0],
                        'attempt_2': valid_preds[0]
                    })
            else:
                # Fallback: use input with simple transformations
                fallback = [row[:] for row in test_input_data]
                fallback2 = AdvancedTransformations.apply_transformation(
                    fallback, random.randint(0, 5)
                )
                if not is_valid_grid(fallback2):
                    fallback2 = fallback
                
                predictions.append({
                    'attempt_1': fallback,
                    'attempt_2': fallback2
                })
        
        return predictions

# ==================== UTILITIES ====================

def get_path(name):
    """Get file path for Kaggle or local"""
    kaggle_path = f'/kaggle/input/arc-prize-2025/{name}'
    return kaggle_path if os.path.exists(kaggle_path) else name

def is_valid_grid(grid):
    """Validate grid format"""
    if not isinstance(grid, list) or len(grid) == 0:
        return False
    if len(grid) > 30:
        return False
    
    row_length = len(grid[0])
    if row_length == 0 or row_length > 30:
        return False
    
    for row in grid:
        if not isinstance(row, list) or len(row) != row_length:
            return False
        if not all(isinstance(cell, (int, np.integer)) and 0 <= cell <= 9 for cell in row):
            return False
    
    return True

def create_submission(test_challenges, solver):
    """Generate submission for all test challenges"""
    submission = {}
    
    for task_id, challenge in test_challenges.items():
        try:
            predictions = solver.solve_challenge(challenge)
            submission[task_id] = predictions
        except:
            # Fallback: return input as output
            fallback_predictions = []
            for test_input in challenge.get('test', []):
                test_data = test_input.get('input', test_input)
                fallback = [row[:] for row in test_data]
                fallback_predictions.append({
                    'attempt_1': fallback,
                    'attempt_2': fallback
                })
            submission[task_id] = fallback_predictions
    
    return submission

def validate_submission(submission):
    """Validate submission format"""
    for task_id, outputs in submission.items():
        if not isinstance(outputs, list):
            return False
        
        for entry in outputs:
            if 'attempt_1' not in entry or 'attempt_2' not in entry:
                return False
            if not is_valid_grid(entry['attempt_1']):
                return False
            if not is_valid_grid(entry['attempt_2']):
                return False
    
    return True

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Load data
    training_challenges = json.load(open(get_path('arc-agi_training_challenges.json')))
    test_challenges = json.load(open(get_path('arc-agi_test_challenges.json')))
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize solver
    solver = ImprovedHybridSolver(device=device)
    
    # Train on training set
    solver.train_on_training_set(training_challenges, max_tasks=150)
    
    # Generate predictions
    submission = create_submission(test_challenges, solver)
    
    # Validate
    if not validate_submission(submission):
        raise ValueError("Submission validation failed")
    
    # Save submission
    with open('submission.json', 'w') as f:
        json.dump(submission, f)

if __name__ == '__main__':
    with SilentExecution():
        main()


