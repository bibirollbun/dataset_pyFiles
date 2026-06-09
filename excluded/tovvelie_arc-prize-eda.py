# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


def load_arc_data():
    file_paths = [
        "/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json",
        "/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json",
        "/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json",
        "/kaggle/input/arc-prize-2025/sample_submission.json",
        "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json",
        "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json",
    ]

    data = {}
    for path in file_paths:
        base = os.path.basename(path)
        name = base.replace("arc-agi_", "").replace(".json", "")
        data[name] = read_json(path)
    
    return data

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Пример использования
data = load_arc_data()


training_challenges_idx = list(data['training_challenges'].keys())


len(training_challenges_idx)


samples_len = [len(data['training_challenges'][i]['train']) for i in training_challenges_idx]
min(samples_len), max(samples_len)


[[np.array(sample['input']).shape for sample in data['training_challenges'][task_idx]['train']] for task_idx in training_challenges_idx[:5]]


print(
    max([max([np.array(np.array(sample['input']).shape).prod() for sample in data['training_challenges'][task_idx]['train']]) for task_idx in training_challenges_idx]),
    max([max([np.array(np.array(sample['output']).shape).prod() for sample in data['training_challenges'][task_idx]['train']]) for task_idx in training_challenges_idx])
)


def show_matrices(matrices):
    num_matrices = len(matrices)

    colors = [
        "#000000",
        "#0000FF",
        "#FF0000",
        "#00FF00",
        "#FFFF00",
        "#FFFFFF",
        "#800080",        
        "#FFA500",
        "#00FFFF",
        "#A52A2A",
    ]
    
    cmap = ListedColormap(colors)
    
    for idx, matrix in enumerate(matrices):
        if idx < num_matrices / 2:
            title = 'input ' + str(idx)
        else:
            title = 'output ' + str(idx - num_matrices // 2)
        plt.subplot(2, num_matrices // 2, idx + 1)
        plt.imshow(matrix, cmap=cmap, vmin=0, vmax=9)
        plt.axis('off')
        plt.title(title)
    
    plt.tight_layout()
    plt.show()


samples = [[np.array(sample[narr]) for narr in ['input', 'output'] for sample in data['training_challenges'][training_challenges_idx[i]]['train']] for i in range(5)]


show_matrices(samples[0])


show_matrices(samples[1])


show_matrices(samples[2])


show_matrices(samples[3])


show_matrices(samples[4])

