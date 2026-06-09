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


import json
import numpy as np
import os
from collections import defaultdict, Counter

# Since this is Kaggle, input files are in /kaggle/input/arc-prize-2025/
DATA_DIR = '/kaggle/input/arc-prize-2025'

# Load training challenges JSON
with open(os.path.join(DATA_DIR, 'arc-agi_training_challenges.json')) as f:
    train_data = json.load(f)

# Build consolidated color mappings per task
def build_color_mappings(train_data):
    color_maps = {}
    for task_id, task in train_data.items():
        color_counter = defaultdict(Counter)
        for ex in task.get('train', []):
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            if inp.shape == out.shape:
                diff = inp != out
                for i_val, o_val in zip(inp[diff], out[diff]):
                    color_counter[int(i_val)][int(o_val)] += 1
        mapping = {}
        for c_in, counter in color_counter.items():
            most_common_out = counter.most_common(1)
            if most_common_out:
                mapping[c_in] = most_common_out[0][0]
        if mapping:
            color_maps[task_id] = mapping
    return color_maps

color_mappings = build_color_mappings(train_data)

# Load test challenges JSON
with open(os.path.join(DATA_DIR, 'arc-agi_test_challenges.json')) as f:
    test_data = json.load(f)

# Apply color mapping rule to a grid
def apply_color_map(grid, color_map):
    arr = np.array(grid).copy()
    for k, v in color_map.items():
        arr[arr == k] = v
    return arr.tolist()

# Prepare submission predictions
submission = {}
for task_id, task in test_data.items():
    c_map = color_mappings.get(task_id, {})
    preds = []
    for test_ex in task.get('test', []):
        inp_grid = test_ex['input']
        attempt_1 = apply_color_map(inp_grid, c_map)  # Apply learned color map
        attempt_2 = inp_grid                          # Fallback: copy input
        preds.append({"attempt_1": attempt_1, "attempt_2": attempt_2})
    submission[task_id] = preds

# Save the predictions to submission.json in the current Kaggle working directory
with open('/kaggle/working/submission.json', 'w') as f:
    json.dump(submission, f)

print("✅ submission.json created successfully!")


