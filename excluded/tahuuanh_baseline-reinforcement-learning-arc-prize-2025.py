# Installation and Imports

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
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")



def get_path(name):
    return f'/kaggle/input/arc-prize-2025/{name}' if os.path.exists(f'/kaggle/input/arc-prize-2025/{name}') else name

training_solutions = json.load(open(get_path('arc-agi_training_solutions.json')))
evaluation_solutions = json.load(open(get_path('arc-agi_evaluation_solutions.json')))
evaluation_challenges = json.load(open(get_path('arc-agi_evaluation_challenges.json')))
sample_submission = json.load(open(get_path('sample_submission.json')))
training_challenges = json.load(open(get_path('arc-agi_training_challenges.json')))
test_challenges = json.load(open(get_path('arc-agi_test_challenges.json')))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nâœ“ Device: {device}")
print(f"âœ“ Training tasks: {len(training_challenges)}")
print(f"âœ“ Evaluation tasks: {len(evaluation_challenges)}")
print(f"âœ“ Test tasks: {len(test_challenges)}")


# ARC color palette
ARC_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]

def plot_grid(grid, title="Grid", ax=None, show_grid_lines=True):
    """Display a single grid with proper color mapping"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    grid_array = np.array(grid)
    cmap = ListedColormap(ARC_COLORS)
    
    ax.imshow(grid_array, cmap=cmap, vmin=0, vmax=9)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    if show_grid_lines:
        h, w = grid_array.shape
        for i in range(h + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
        for j in range(w + 1):
            ax.axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.3)
    
    ax.text(0.02, 0.98, f'{len(grid)}Ã—{len(grid[0])}', 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return ax

def plot_task(task, task_id="Unknown"):
    """Visualize all training examples and test inputs of a task"""
    n_train = len(task['train'])
    n_test = len(task['test'])
    
    fig, axes = plt.subplots(n_train + n_test, 2, 
                             figsize=(8, 4 * (n_train + n_test)))
    
    if n_train + n_test == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Task: {task_id}', fontsize=16, fontweight='bold', y=0.995)
    
    for idx, pair in enumerate(task['train']):
        plot_grid(pair['input'], f'Train {idx+1} - Input', axes[idx, 0])
        plot_grid(pair['output'], f'Train {idx+1} - Output', axes[idx, 1])
    
    for idx, test_input in enumerate(task['test']):
        plot_grid(test_input['input'], f'Test {idx+1} - Input', 
                 axes[n_train + idx, 0])
        axes[n_train + idx, 1].axis('off')
        axes[n_train + idx, 1].text(0.5, 0.5, '?', 
                                     ha='center', va='center',
                                     fontsize=80, alpha=0.3,
                                     transform=axes[n_train + idx, 1].transAxes)
    
    plt.tight_layout()
    plt.show()

def plot_transformation_sequence(grids, titles, save_path=None):
    """Visualize a sequence of grid transformations"""
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    
    if n == 1:
        axes = [axes]
    
    for idx, (grid, title) in enumerate(zip(grids, titles)):
        plot_grid(grid, title, axes[idx])
        
        if idx < n - 1:
            ax_pos = axes[idx].get_position()
            arrow = patches.FancyArrowPatch(
                (ax_pos.x1 + 0.01, ax_pos.y0 + ax_pos.height/2),
                (ax_pos.x1 + 0.04, ax_pos.y0 + ax_pos.height/2),
                transform=fig.transFigure,
                arrowstyle='->', mutation_scale=30, 
                linewidth=2, color='#333333'
            )
            fig.patches.append(arrow)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_training_progress(rewards, window=50):
    """Plot training reward progression"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(rewards, alpha=0.3, color='#0074D9', label='Raw Rewards')
    
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), smoothed, 
                color='#FF4136', linewidth=2, label=f'Smoothed (window={window})')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("âœ“ Visualization utilities loaded")


class GridEncoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, hidden_dim, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, grid):
        if len(grid.shape) == 3:
            grid = grid.unsqueeze(0)
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.squeeze(-1).squeeze(-1)

class PolicyNetwork(nn.Module):
    def __init__(self, hidden_dim=128, num_actions=20):
        super().__init__()
        self.encoder = GridEncoder(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_actions)
        
    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = GridEncoder(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

print("âœ“ Neural network architectures defined")


class GridTransformations:
    @staticmethod
    def rotate_90(grid):
        return [list(row) for row in zip(*grid[::-1])]
    
    @staticmethod
    def rotate_180(grid):
        return [row[::-1] for row in grid[::-1]]
    
    @staticmethod
    def rotate_270(grid):
        return [list(row) for row in zip(*grid)][::-1]
    
    @staticmethod
    def flip_horizontal(grid):
        return [row[::-1] for row in grid]
    
    @staticmethod
    def flip_vertical(grid):
        return grid[::-1]
    
    @staticmethod
    def transpose(grid):
        return [list(row) for row in zip(*grid)]
    
    @staticmethod
    def scale_up(grid, factor=2):
        h, w = len(grid), len(grid[0])
        if h * factor > 30 or w * factor > 30:
            return grid
        result = [[0] * (w * factor) for _ in range(h * factor)]
        for i in range(h):
            for j in range(w):
                for di in range(factor):
                    for dj in range(factor):
                        result[i * factor + di][j * factor + dj] = grid[i][j]
        return result
    
    @staticmethod
    def scale_down(grid, factor=2):
        h, w = len(grid), len(grid[0])
        new_h, new_w = h // factor, w // factor
        if new_h == 0 or new_w == 0:
            return grid
        result = [[0] * new_w for _ in range(new_h)]
        for i in range(new_h):
            for j in range(new_w):
                result[i][j] = grid[i * factor][j * factor]
        return result
    
    @staticmethod
    def fill_color(grid, old_color, new_color):
        return [[new_color if cell == old_color else cell for cell in row] for row in grid]
    
    @staticmethod
    def invert_colors(grid):
        return [[9 - cell for cell in row] for row in grid]
    
    @staticmethod
    def extract_objects(grid):
        h, w = len(grid), len(grid[0])
        visited = [[False] * w for _ in range(h)]
        objects = []
        
        def dfs(i, j, color, obj):
            if i < 0 or i >= h or j < 0 or j >= w or visited[i][j] or grid[i][j] != color:
                return
            visited[i][j] = True
            obj.append((i, j))
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dfs(i + di, j + dj, color, obj)
        
        for i in range(h):
            for j in range(w):
                if not visited[i][j] and grid[i][j] != 0:
                    obj = []
                    dfs(i, j, grid[i][j], obj)
                    if obj:
                        objects.append((grid[i][j], obj))
        
        return objects
    
    @staticmethod
    def get_all_transformations():
        return [
            ('identity', lambda g: g),
            ('rotate_90', GridTransformations.rotate_90),
            ('rotate_180', GridTransformations.rotate_180),
            ('rotate_270', GridTransformations.rotate_270),
            ('flip_h', GridTransformations.flip_horizontal),
            ('flip_v', GridTransformations.flip_vertical),
            ('transpose', GridTransformations.transpose),
            ('scale_up', lambda g: GridTransformations.scale_up(g, 2)),
            ('scale_down', lambda g: GridTransformations.scale_down(g, 2)),
            ('invert', GridTransformations.invert_colors),
        ]

print("âœ“ Grid transformations defined")


sample_grid = [
    [0, 1, 1, 0],
    [0, 1, 2, 0],
    [0, 0, 2, 0],
    [0, 0, 0, 0]
]

print("Demonstrating transformations:")
transforms = GridTransformations()

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

transformations_demo = [
    ('Original', sample_grid),
    ('Rotate 90Â°', transforms.rotate_90(sample_grid)),
    ('Rotate 180Â°', transforms.rotate_180(sample_grid)),
    ('Flip Horizontal', transforms.flip_horizontal(sample_grid)),
    ('Flip Vertical', transforms.flip_vertical(sample_grid)),
    ('Transpose', transforms.transpose(sample_grid)),
    ('Scale Up 2x', transforms.scale_up(sample_grid, 2)),
    ('Scale Down 2x', transforms.scale_down(sample_grid * 2, 2)),
    ('Invert Colors', transforms.invert_colors(sample_grid)),
]

for idx, (name, grid) in enumerate(transformations_demo):
    if idx < len(axes):
        plot_grid(grid, name, axes[idx])

if len(transformations_demo) < len(axes):
    for idx in range(len(transformations_demo), len(axes)):
        axes[idx].axis('off')

plt.tight_layout()
plt.show()



class RLAgent:
    def __init__(self, device='cpu'):
        self.device = device
        self.policy_net = PolicyNetwork().to(device)
        self.value_net = ValueNetwork().to(device)
        self.transformations = GridTransformations.get_all_transformations()
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=1e-3)
        self.memory = deque(maxlen=10000)
        self.training_rewards = []
        
    def grid_to_tensor(self, grid):
        h, w = len(grid), len(grid[0])
        tensor = torch.zeros(10, h, w)
        for i in range(h):
            for j in range(w):
                tensor[grid[i][j], i, j] = 1.0
        return tensor.to(self.device)
    
    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, len(self.transformations) - 1)
        
        with torch.no_grad():
            state_tensor = self.grid_to_tensor(state).unsqueeze(0)
            probs = self.policy_net(state_tensor)
            return torch.argmax(probs).item()
    
    def apply_transformation(self, grid, action_idx):
        if action_idx >= len(self.transformations):
            return grid
        name, transform = self.transformations[action_idx]
        try:
            return transform(grid)
        except:
            return grid
    
    def compute_reward(self, current_grid, target_grid):
        if len(current_grid) != len(target_grid):
            return -1.0
        if len(current_grid[0]) != len(target_grid[0]):
            return -1.0
        
        total_cells = len(current_grid) * len(current_grid[0])
        correct_cells = sum(
            1 for i in range(len(current_grid))
            for j in range(len(current_grid[0]))
            if current_grid[i][j] == target_grid[i][j]
        )
        
        accuracy = correct_cells / total_cells
        
        if accuracy == 1.0:
            return 10.0
        elif accuracy > 0.8:
            return 5.0 * accuracy
        else:
            return accuracy - 0.5
    
    def train_on_task(self, task, max_steps=10):
        for train_pair in task['train']:
            input_grid = train_pair['input']
            target_grid = train_pair['output']
            
            state = [row[:] for row in input_grid]
            episode_data = []
            episode_reward = 0
            
            for step in range(max_steps):
                action = self.select_action(state, epsilon=0.3)
                next_state = self.apply_transformation(state, action)
                
                if not is_valid_grid(next_state):
                    reward = -1.0
                    break
                
                reward = self.compute_reward(next_state, target_grid)
                episode_reward += reward
                
                episode_data.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state
                })
                
                if reward >= 10.0:
                    break
                
                state = next_state
            
            self.memory.extend(episode_data)
            self.training_rewards.append(episode_reward)
            
            if len(self.memory) >= 32:
                self.update_networks()
    
    def update_networks(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        policy_loss = 0
        value_loss = 0
        
        for experience in batch:
            state_tensor = self.grid_to_tensor(experience['state'])
            
            action_probs = self.policy_net(state_tensor.unsqueeze(0))
            action_probs = action_probs.squeeze(0)
            action_log_prob = torch.log(action_probs[experience['action']] + 1e-8)
            
            value = self.value_net(state_tensor.unsqueeze(0))
            
            reward = torch.tensor([experience['reward']], device=self.device)
            
            advantage = reward - value.detach()
            policy_loss += -action_log_prob * advantage
            value_loss += F.mse_loss(value, reward)
        
        policy_loss = policy_loss / batch_size
        value_loss = value_loss / batch_size
        
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer_policy.step()
        
        self.optimizer_value.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.optimizer_value.step()
    
    def predict(self, test_input, train_pairs, max_steps=15, num_attempts=5):
        best_predictions = []
        
        for attempt in range(num_attempts):
            state = [row[:] for row in test_input]
            best_score = -float('inf')
            best_grid = state
            
            for step in range(max_steps):
                action = self.select_action(state, epsilon=0.1 if attempt > 0 else 0.0)
                next_state = self.apply_transformation(state, action)
                
                if not is_valid_grid(next_state):
                    break
                
                avg_reward = 0
                for train_pair in train_pairs:
                    score = self.compute_similarity(next_state, train_pair['output'])
                    avg_reward += score
                avg_reward /= len(train_pairs)
                
                if avg_reward > best_score:
                    best_score = avg_reward
                    best_grid = [row[:] for row in next_state]
                
                state = next_state
            
            best_predictions.append(best_grid)
        
        return best_predictions
    
    def compute_similarity(self, grid1, grid2):
        if len(grid1) == len(grid2) and len(grid1[0]) == len(grid2[0]):
            total = len(grid1) * len(grid1[0])
            matches = sum(
                1 for i in range(len(grid1))
                for j in range(len(grid1[0]))
                if grid1[i][j] == grid2[i][j]
            )
            return matches / total
        
        size_diff = abs(len(grid1) - len(grid2)) + abs(len(grid1[0]) - len(grid2[0]))
        return 1.0 / (1.0 + size_diff)

print("âœ“ RL Agent implemented")


class HybridSolver:
    def __init__(self, device='cpu'):
        self.rl_agent = RLAgent(device)
        self.transformations = GridTransformations()
        
    def train_on_training_set(self, training_challenges, max_tasks=50):
        print("Training RL agent on training set...")
        task_ids = list(training_challenges.keys())[:max_tasks]
        
        for task_id in tqdm(task_ids, desc="Training"):
            task = training_challenges[task_id]
            self.rl_agent.train_on_task(task, max_steps=10)
    
    def solve_with_rules(self, test_input, train_pairs):
        for name, transform in self.transformations.get_all_transformations():
            try:
                result = transform(test_input)
                if is_valid_grid(result):
                    for train_pair in train_pairs:
                        test_result = transform(train_pair['input'])
                        if test_result == train_pair['output']:
                            return result
            except:
                continue
        return None
    
    def solve_challenge(self, challenge, visualize=False):
        predictions = []
        
        for idx, test_input_data in enumerate(challenge['test']):
            test_input = test_input_data['input']
            
            rule_result = self.solve_with_rules(test_input, challenge['train'])
            
            if rule_result and is_valid_grid(rule_result):
                predictions.append({
                    'attempt_1': rule_result,
                    'attempt_2': rule_result
                })
                
                if visualize:
                    plot_transformation_sequence(
                        [test_input, rule_result],
                        ['Test Input', 'Rule-based Solution']
                    )
            else:
                rl_predictions = self.rl_agent.predict(
                    test_input, 
                    challenge['train'],
                    max_steps=15,
                    num_attempts=3
                )
                
                valid_predictions = [p for p in rl_predictions if is_valid_grid(p)]
                
                if len(valid_predictions) >= 2:
                    predictions.append({
                        'attempt_1': valid_predictions[0],
                        'attempt_2': valid_predictions[1]
                    })
                elif len(valid_predictions) == 1:
                    predictions.append({
                        'attempt_1': valid_predictions[0],
                        'attempt_2': valid_predictions[0]
                    })
                else:
                    fallback = [row[:] for row in test_input]
                    predictions.append({
                        'attempt_1': fallback,
                        'attempt_2': fallback
                    })
                
                if visualize and valid_predictions:
                    plot_transformation_sequence(
                        [test_input] + valid_predictions[:2],
                        ['Test Input', 'RL Prediction 1', 'RL Prediction 2'][:len(valid_predictions)+1]
                    )
        
        return predictions

def is_valid_grid(grid):
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
        if not all(isinstance(cell, int) and 0 <= cell <= 9 for cell in row):
            return False
    return True

print("âœ“ Hybrid solver ready")


sample_task_ids = list(training_challenges.keys())[:3]

for task_id in sample_task_ids:
    plot_task(training_challenges[task_id], task_id)



solver = HybridSolver(device=device)

print("Starting training phase...")
solver.train_on_training_set(training_challenges, max_tasks=100)

if len(solver.rl_agent.training_rewards) > 0:
    plot_training_progress(solver.rl_agent.training_rewards)
    print(f"\nâœ“ Training completed")
    print(f"  Final average reward: {np.mean(solver.rl_agent.training_rewards[-100:]):.3f}")
    print(f"  Total episodes: {len(solver.rl_agent.training_rewards)}")



sample_eval_id = list(evaluation_challenges.keys())[0]
sample_challenge = evaluation_challenges[sample_eval_id]

print(f"Testing on task: {sample_eval_id}\n")
plot_task(sample_challenge, sample_eval_id)

predictions = solver.solve_challenge(sample_challenge, visualize=True)
print(f"âœ“ Generated {len(predictions)} predictions")



def create_submission(test_challenges, solver):
    submission = {}
    
    for task_id, challenge in tqdm(test_challenges.items(), desc="Solving test tasks"):
        predictions = solver.solve_challenge(challenge)
        submission[task_id] = predictions
    
    return submission

print("Generating final predictions...")
submission = create_submission(test_challenges, solver)



print("Validating submission format...")
validation_passed = True

for task_id, outputs in submission.items():
    if not isinstance(outputs, list):
        print(f"â�Œ Task {task_id}: outputs not a list")
        validation_passed = False
        continue
    
    for idx, entry in enumerate(outputs):
        if 'attempt_1' not in entry or 'attempt_2' not in entry:
            print(f"â�Œ Task {task_id}, test {idx}: missing attempts")
            validation_passed = False
        
        if not is_valid_grid(entry['attempt_1']):
            print(f"â�Œ Task {task_id}, test {idx}: invalid attempt_1")
            validation_passed = False
        
        if not is_valid_grid(entry['attempt_2']):
            print(f"â�Œ Task {task_id}, test {idx}: invalid attempt_2")
            validation_passed = False

if validation_passed:
    print("âœ“ All validations passed")
else:
    print("âš  Some validations failed")

with open('submission.json', 'w') as f:
    json.dump(submission, f)

print(f"\nâœ“ Submission saved successfully")
print(f"  Total tasks: {len(submission)}")

total_test_cases = sum(len(outputs) for outputs in submission.values())
print(f"  Total test cases: {total_test_cases}")



print("\n" + "="*60)
print("SUBMISSION SUMMARY")
print("="*60)

output_sizes = []
unique_attempts = 0
identical_attempts = 0

for task_id, outputs in submission.items():
    for entry in outputs:
        h1, w1 = len(entry['attempt_1']), len(entry['attempt_1'][0])
        h2, w2 = len(entry['attempt_2']), len(entry['attempt_2'][0])
        
        output_sizes.append((h1, w1))
        output_sizes.append((h2, w2))
        
        if entry['attempt_1'] == entry['attempt_2']:
            identical_attempts += 1
        else:
            unique_attempts += 1

print(f"\nPrediction Diversity:")
print(f"  Identical attempts: {identical_attempts}")
print(f"  Unique attempts: {unique_attempts}")

print(f"\nGrid Size Statistics:")
heights = [h for h, w in output_sizes]
widths = [w for h, w in output_sizes]
print(f"  Height - min: {min(heights)}, max: {max(heights)}, avg: {np.mean(heights):.1f}")
print(f"  Width  - min: {min(widths)}, max: {max(widths)}, avg: {np.mean(widths):.1f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(heights, bins=20, color='#0074D9', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Grid Height', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of Output Heights', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].hist(widths, bins=20, color='#FF4136', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Grid Width', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Distribution of Output Widths', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("PROCESS COMPLETE")
print("="*60)

