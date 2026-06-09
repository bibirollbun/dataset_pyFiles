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


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from   matplotlib import colors
import seaborn as sns

import json
import os
from pathlib import Path

from subprocess import Popen, PIPE, STDOUT
from glob import glob


cmap = colors.ListedColormap([

    '#8B00FF',  # Violet #0
    '#4B0082',  # Indigo #1
    '#0000FF',  # Blue # 2
    '#FFFF00',  # Yellow # 3
    '#00FF00',  # Green # 4
    '#FF7F00',  # Orange # 5
    '#FF0000',  # Red # 6
    '#964B00',  # Golden # 7
    '#000000',  # Black # 8
    '#FFFFFF',  # White # 9
])
norm = colors.Normalize(vmin=0, vmax=9)


norm = colors.Normalize(vmin=0, vmax=9)

plt.figure(figsize=(4, 1), dpi=200)
plt.imshow([list(range(10))], cmap=cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.show()


base_path='/kaggle/input/arc-prize-2025/'
# Loading JSON data
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


# Reading files
training_challenges =  load_json(base_path +'arc-agi_training_challenges.json')
training_solutions =   load_json(base_path +'arc-agi_training_solutions.json')

evaluation_challenges =load_json(base_path +'arc-agi_evaluation_challenges.json')
evaluation_solutions = load_json(base_path +'arc-agi_evaluation_solutions.json')

test_challenges =  load_json(base_path +'arc-agi_test_challenges.json')


print(f'Number of training challenges = {len(training_challenges)}')
print(f'Number of training solutions = {len(training_solutions)}')
print(f'Number of evaluation challenges = {len(evaluation_challenges)}')
print(f'Number of evaluation solutions = {len(evaluation_solutions)}')
print(f'Number of test challenges = {len(test_challenges)}')


for i in range(5):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    print(f'Set #{i}, {t}')


task = training_challenges['00576224']
task1= training_challenges['007bbfb7']
task2= training_challenges['009d5c81']
task3= training_challenges['00d62c1b']
task4= training_challenges['00dbd492']
print(task.keys())


number_train_pairs = len(task['train'])
number_test_pairs = len(task['test'])
number1_train_pairs = len(task1['train'])
number1_test_pairs = len(task1['test'])
number2_train_pairs = len(task2['train'])
number2_test_pairs = len(task2['test'])
number3_train_pairs = len(task3['train'])
number3_test_pairs = len(task3['test'])
number4_train_pairs = len(task4['train'])
number4_test_pairs = len(task4['test'])


print(f'set0 contains {number_train_pairs} training pairs')
print(f'set0 contains {number_test_pairs} test pairs')
print(f'set1 contains {number1_train_pairs} training pairs')
print(f'set1 contains {number1_test_pairs} test pairs')
print(f'set2 contains {number2_train_pairs} training pairs')
print(f'set2 contains {number2_test_pairs} test pairs')
print(f'set3 contains {number3_train_pairs} training pairs')
print(f'set3 contains {number3_test_pairs} test pairs')
print(f'set4 contains {number4_train_pairs} training pairs')
print(f'set4 contains {number4_test_pairs} test pairs')


# Assuming `task` is a dictionary with the given structure
input_data = task['train'][0]['input'],
output_data = task['train'][0]['output']

# Display the input and output data
print("Input Data:", input_data)
print("Output Data:", output_data)



# Defines a function named plot_task that takes four parameters: task, task_solutions, i, and t.

def plot_task(task, task_solutions, i, t):
    """Plots the first train and test pairs of a specified task,
    using the same color scheme as the ARC app"""
    
    num_train = len(task['train']) # Stores the number of training records in the task dictionary
    num_test  = len(task['test']) # Stores the number of test records in the task dictionary.
   
    w = num_train + num_test # Total number of columns for subplots (sum of training and test records).
    fig, axs = plt.subplots(2, w, figsize=(3*w ,3*2)) # Creates a figure and a 2xw grid of subplots.
    plt.suptitle(f'Set #{i}, {t}:', fontsize=20, fontweight='bold', y=1) # Sets a super title for the figure.

    for j in range(num_train):
        plot_one(axs[0, j], j, 'train', 'input') # Calls the plot_one function to plot each training record’s input and output data.
        plot_one(axs[1, j], j, 'train', 'output')
        
    plot_one(axs[0, j+1], 0, 'test', 'input')

    answer = task_solutions
    input_matrix = answer

    axs[1, j+1].imshow(input_matrix, cmap=cmap, norm=norm) # Displays the test output data as an image.
    axs[1, j+1].grid(True, which='both', color='lightgrey', linewidth=0.5) # Adds a grid to the plot.
    axs[1, j+1].set_yticks([x-0.5 for x in range(1 + len(input_matrix))]) # Configures tick marks to align with the grid.
    axs[1, j+1].set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
    axs[1, j+1].set_xticklabels([]) # Hides tick labels.
    axs[1, j+1].set_yticklabels([])
    axs[1, j+1].set_title('Test output') # Sets the title for the test output plot.

    axs[1, j+1] = plt.figure(1).add_subplot(111)
    axs[1, j+1].set_xlim([0, num_train+1])
    for m in range(1, num_train):
        axs[1, j+1].plot([m, m], [0, 1], '--', linewidth=1, color='black')
    axs[1, j+1].plot([num_train, num_train], [0, 1], '-', linewidth=3, color='black')
    axs[1, j+1].axis("off")

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor('#dddddd')
    plt.tight_layout()
    print(f'#{i}, {t}')  # for fast and convenience search
    plt.show()
    print()
    print()


def plot_one(ax, i, train_or_test, input_or_output):
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
    
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])     
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
    
    ax.set_title(train_or_test + ' ' + input_or_output)


for i in range(0,20):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    task_solution = training_solutions[t][0]
    plot_task(task,  task_solution, i, t)

