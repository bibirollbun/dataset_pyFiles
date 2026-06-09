import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import colors


def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


# load sumbission
!gdown -O 'sumbission.csv' https://drive.google.com/file/d/1Ng56iCYtgbIkyeRLDtRe9kuCw8g6eBKm/view?usp=sharing --fuzzy


test_challenges = load_json('//kaggle/input/arc-prize-2025/arc-agi_test_challenges.json')
print(f'Len of test challenges: {len(test_challenges)}')


test_qwen = load_json('/kaggle/working/sumbission.csv')
print(f'Len of prediction qwen: {len(test_qwen)}')



def plot_one(ax, i, task, train_or_test : str, input_or_output: str, solution : bool = None, i_s = None, attempt = None, w=0.8):
    fs=12
    if solution:
        assert i_s != None and attempt != None, "i_s and attempt must be provided if solution is provided"
        input_matrix = task[train_or_test][i][input_or_output][i_s][attempt]
    else:
        input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    
    #ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
    
    '''Grid:'''
    ax.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
    
    ax.tick_params(axis='both', color='none', length=0)
   
    '''sub title:'''
    if solution:
        ax.set_title(train_or_test + ' ' + input_or_output + ' ' + attempt, fontsize=fs, color = '#dddddd', fontweight= 'bold')
    else:
        ax.set_title(train_or_test + ' ' + input_or_output, fontsize=fs, color = '#dddddd', fontweight = 'bold')

color = '#55ffbb'
cmap = colors.ListedColormap(    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                                  '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)



import os
def plot_combined(task, t_n, show = True, save_dir = False):
    # create a common canvas
    # first line — train data, second — test input and solutions
    fig = plt.figure(figsize=(10, 10))
    
    # Up part: train data (two lines)
    gs_top = fig.add_gridspec(nrows=2, ncols=len(task['train']), top=0.85, bottom=0.55)
    for i in range(len(task['train'])):
        ax1 = fig.add_subplot(gs_top[0, i])
        plot_one(ax1, i, task, 'train', 'input', w=0.8)
        ax2 = fig.add_subplot(gs_top[1, i])
        plot_one(ax2, i, task, 'train', 'output', w=0.8)
    
    # Down part: test solutions and input (1 line, 3 columns)
    gs_bottom = fig.add_gridspec(nrows=1, ncols=3, top=0.45, bottom=0.15, wspace=0.25)
    ax_test = [fig.add_subplot(gs_bottom[i]) for i in range(3)]
    
    # Visualization test data
    plot_one(ax_test[0], 0, task, 'test', 'input', w=0.8)
    plot_one(ax_test[1], 0, task, 'test', 'solution', solution=True, i_s=0, attempt='attempt_1', w=0.8)
    plot_one(ax_test[2], 0, task, 'test', 'solution', solution=True, i_s=0, attempt='attempt_2', w=0.8)
    
    # Add vertical line to subplots
    line = plt.Line2D(
        (ax_test[0].get_position().x1 + 0.025, ax_test[0].get_position().x1 + 0.025),
        (ax_test[0].get_position().y0, ax_test[0].get_position().y1),
        color='black', linewidth=5, linestyle='--', transform=fig.transFigure
    )
    fig.add_artist(line)

    # titels
    fig.suptitle(f"Visualization #{t_n}", fontsize=30, color='black', fontweight='bold', y=0.99)
    fig.text(0.5, 0.9, "Train Data", ha='center', fontsize=25, color='#444444')
    fig.text(0.5, 0.45, "Test Solutions", ha='center', fontsize=25, color='#444444')
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{t_n.split('-')[0]}.png"))
        if show:
            plt.show()
        else:
            plt.close(fig)


for i in range(10):
    # get index task
    t_n = list(test_challenges.keys())[i]
    # get task
    task = test_challenges[t_n]
    # add solution in train
    task_solution = test_qwen[t_n]
    task['test'][0]['solution'] = task_solution
    # plot
    plot_combined(task, t_n = f'{t_n}-{i + 1}', show = True)



from tqdm import tqdm
for i in tqdm(range(len(test_qwen))):
    # get index task
    t_n = list(test_challenges.keys())[i]
    task = test_challenges[t_n]
    task_solution = test_qwen[t_n]
    # add solution in task
    task['test'][0]['solution'] = task_solution
    plot_combined(task, t_n = f'{t_n}-{i + 1}', show = False, save_dir = 'qwen_solutions')


for i in range(len(test_qwen)):
    t_n = list(test_challenges.keys())[i]
    task = test_challenges[t_n]
    task_solution = test_qwen[t_n]
    task['test'][0]['solution'] = task_solution
    plot_combined(task, t_n = f'{t_n}-{i + 1}', show = True)

