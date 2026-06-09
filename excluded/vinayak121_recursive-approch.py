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


# 1. Setup and Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from tqdm.notebook import tqdm
import zipfile
import matplotlib.pyplot as plt



# 2. Device Selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch CUDA Available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# 3. Helper for Padding Grids
def pad_grid(grid, max_size=30, fill_value=0):
    arr = np.full((max_size, max_size), fill_value, dtype=np.int32)
    h, w = len(grid), len(grid[0])
    arr[:h, :w] = np.array(grid)
    return arr


import json
import matplotlib.pyplot as plt
import numpy as np
import random

# File paths adapted to your ARC dataset
train_challenges_path = '/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json'
train_solutions_path = '/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json'
test_challenges_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'

with open(train_challenges_path, 'r') as f:
    train_challenges = json.load(f)
with open(train_solutions_path, 'r') as f:
    train_solutions = json.load(f)
with open(test_challenges_path, 'r') as f:
    test_challenges = json.load(f)

# 1. Dataset basics
print(f'Number of training tasks: {len(train_challenges)}')
print(f'Number of training solutions: {len(train_solutions)}')
print(f'Number of test tasks: {len(test_challenges)}')

first_task_id = list(train_challenges.keys())[0]
print("\nExample training task id:", first_task_id)
print("Challenge example keys:", train_challenges[first_task_id].keys())
print("First solution example type:", type(train_solutions[first_task_id]))
print("First solution example content:", train_solutions[first_task_id])

# 2. Analyze grid sizes and pixel distributions
def get_grid_info(challenges_dict, solutions_dict):
    input_h, input_w, output_h, output_w = [], [], [], []
    all_colors = set()
    for task_id, task in challenges_dict.items():
        # For each train example for this task
        for pair in task['train']:
            grid = pair['input']
            input_h.append(len(grid))
            input_w.append(len(grid[0]))
            all_colors.update(np.unique(grid))
        # For each ground truth output for this task
        for grid in solutions_dict[task_id]:
            output_h.append(len(grid))
            output_w.append(len(grid[0]))
            all_colors.update(np.unique(grid))
    return input_h, input_w, output_h, output_w, all_colors

input_h, input_w, output_h, output_w, all_colors = get_grid_info(train_challenges, train_solutions)
print(f"\nInput grid sizes (HxW): min {min(input_h)}x{min(input_w)}, max {max(input_h)}x{max(input_w)}, mean {np.mean(input_h):.2f}x{np.mean(input_w):.2f}")
print(f"Output grid sizes (HxW): min {min(output_h)}x{min(output_w)}, max {max(output_h)}x{max(output_w)}, mean {np.mean(output_h):.2f}x{np.mean(output_w):.2f}")
print(f"Unique pixel values: {sorted(all_colors)}")


# 3. Plot distribution of grid sizes and pixel values
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(input_h, bins=range(0, max(input_h)+2), alpha=0.7, label='Input Height')
plt.hist(input_w, bins=range(0, max(input_w)+2), alpha=0.7, label='Input Width')
plt.xlabel('Grid Size')
plt.ylabel('Frequency')
plt.title('Training Input Grid Sizes')
plt.legend()

plt.subplot(1,2,2)
plt.hist(list(all_colors), bins=range(0, max(all_colors)+2), alpha=0.7)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Unique Cell Values in Training Data')
plt.tight_layout()
plt.show()

# 4. Visualize a random train input/output pair
def plot_grid(grid, title="Grid"):
    arr = np.array(grid)
    plt.imshow(arr, cmap='tab20', vmin=0, vmax=9)
    plt.title(title)
    plt.axis('off')

rand_idx = random.randint(0, len(train_challenges) - 1)
rand_task_id = list(train_challenges.keys())[rand_idx]
rand_chal = train_challenges[rand_task_id]
rand_sol = train_solutions[rand_task_id]  # Access by task_id!
rand_in_grid = random.choice([pair['input'] for pair in rand_chal['train']])
rand_out_grid = random.choice(rand_sol)   # rand_sol is a list of output grids


plt.figure(figsize=(8, 4))
plt.subplot(1,2,1)
plot_grid(rand_in_grid, "Random Training Input")
plt.subplot(1,2,2)
plot_grid(rand_out_grid, "Random Training Output")
plt.show()



import json

# Update the file paths as per your Kaggle dataset panel
train_challenges_path = '/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json'
train_solutions_path = '/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json'
test_challenges_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
eval_challenges_path = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json'
eval_solutions_path = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json'
sample_submission_path = '/kaggle/input/arc-prize-2025/sample_submission.json'

with open(train_challenges_path, 'r') as f:
    train_challenges = json.load(f)
with open(train_solutions_path, 'r') as f:
    train_solutions = json.load(f)
with open(test_challenges_path, 'r') as f:
    test_challenges = json.load(f)

# (Optional) Load evaluation data if needed for offline experiments
with open(eval_challenges_path, 'r') as f:
    eval_challenges = json.load(f)
with open(eval_solutions_path, 'r') as f:
    eval_solutions = json.load(f)



class ARCDataset(Dataset):
    def __init__(self, challenges_dict, solutions_dict, max_size=30):
        self.samples = []
        for task_id, task in challenges_dict.items():
            train_pairs = task['train']
            solution_grids = solutions_dict[task_id]
            n = min(len(train_pairs), len(solution_grids))
            for i in range(n):
                inp = pad_grid(train_pairs[i]['input'], max_size)
                out = pad_grid(solution_grids[i], max_size)
                self.samples.append((inp, out))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        inp, out = self.samples[idx]
        return (torch.tensor(inp, dtype=torch.float32).flatten()/9.0,
                torch.tensor(out, dtype=torch.float32).flatten()/9.0)

max_size = 30
dataset = ARCDataset(train_challenges, train_solutions, max_size=max_size)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)



 #7. TRM Model
class TinyRecursiveModel(nn.Module):
    def __init__(self, input_size, hidden_size, state_size, output_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size + output_size + state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size + state_size)
        )
    def forward(self, x, y_init, z_init, steps=8):
        y, z = y_init, z_init
        for _ in range(steps):
            combined = torch.cat([x, y, z], dim=1)
            out = self.mlp(combined)
            y, z = torch.split(out, [y.size(1), z.size(1)], dim=1)
        return y

input_size = output_size = max_size * max_size
state_size = 128
hidden_size = 512
model = TinyRecursiveModel(input_size, hidden_size, state_size, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()



epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y_true in tqdm(dataloader):
        x = x.to(device)
        y_true = y_true.to(device)
        y_init = torch.zeros_like(y_true).to(device)
        z_init = torch.zeros(x.size(0), state_size).to(device)
        y_pred = model(x, y_init, z_init, steps=8)
        loss = loss_fn(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1} Loss: {total_loss / len(dataloader):.5f}')



# 9. Inference and Formatting Submission
def predict_grid(model, inp_grid, max_size=30, steps=8):
    model.eval()
    with torch.no_grad():
        x = pad_grid(inp_grid, max_size).flatten() / 9.0
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        y_init = torch.zeros((1, max_size*max_size)).to(device)
        z_init = torch.zeros((1, state_size)).to(device)
        y_pred = model(x, y_init, z_init, steps=steps)
        pred = torch.round(y_pred * 9.0).int().cpu().numpy()[0]
        out_grid = pred.reshape(max_size, max_size).tolist()
    return out_grid

submission = {}
for task_id, task in tqdm(test_challenges.items()):
    outputs = []
    for test_case in task['test']:
        pred_grid = predict_grid(model, test_case['input'], max_size=max_size, steps=8)
        h, w = len(test_case['input']), len(test_case['input'][0])
        pred_final = [row[:w] for row in pred_grid[:h]]
        outputs.append(pred_final)
    submission[task_id] = outputs

with open('submission.json', 'w') as f:
    json.dump(submission, f)
with zipfile.ZipFile('submission.zip', 'w') as zipf:
    zipf.write('submission.json')

print('Submission file written! Upload submission.zip to Kaggle for scoring.')


