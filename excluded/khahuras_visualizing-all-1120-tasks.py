





DATA_PATH = '/kaggle/input/arc-prize-2025'
FIGURES_PATH = 'task_figures'





import numpy as np, pandas as pd, json, os
import matplotlib.pyplot as plt
%matplotlib inline
import pprint 
pp = pprint.PrettyPrinter(indent=1)
from matplotlib import colors
import copy # for creating full copy of JSON object
from tqdm.notebook import tqdm
from PIL import Image
import time


def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data
    
cmap = colors.ListedColormap(
   ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF'])
norm = colors.Normalize(vmin=0, vmax=10)

def plot_one(ax, i, task, train_or_test, input_or_output, is_solution=False, is_pred=False):
    if is_pred: input_matrix = task
    elif is_solution: input_matrix = task[i]
    else: input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])     
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
    if is_pred:
        title = 'test prediction'
    else:
        title = train_or_test + ' ' + input_or_output
    ax.set_title(title)

    
def plot_task(task1, text, task_solution=None, save_file=None):    
    num_train = len(task1['train'])
    num_test = len(task1['test'])
    #num_test  = len(task['test'])

    w = num_train
    
    if task_solution is not None:
        w += num_test
            
    fig, axs  = plt.subplots(2, w, figsize=(3*w ,3*2))
    plt.suptitle(f'{text}', fontsize=int(3*w*1.5), fontweight='bold', y=1)

    for j in range(num_train):     
        plot_one(axs[0, j], j, task1, 'train', 'input')
        plot_one(axs[1, j], j, task1, 'train', 'output')  
        
    if task_solution is not None:
        for k in range(num_test):
            plot_one(axs[0, j+k+1], k, task1, 'test', 'input') 
            plot_one(axs[1, j+k+1], k, task_solution, 'test', 'output', is_solution=True)
            
        
    fig.patch.set_linewidth(3)
    fig.patch.set_edgecolor('black') 
    fig.patch.set_facecolor('#dddddd')
#     plt.tight_layout()
    
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')
        
    plt.show()





if not os.path.exists(FIGURES_PATH):
    os.mkdir(FIGURES_PATH)





train_tasks   = load_json(f'{DATA_PATH}/arc-agi_training_challenges.json')
train_sols    = load_json(f'{DATA_PATH}/arc-agi_training_solutions.json')

eval_tasks = load_json(f'{DATA_PATH}/arc-agi_evaluation_challenges.json')
eval_sols  = load_json(f'{DATA_PATH}/arc-agi_evaluation_solutions.json')

test_tasks   = load_json(f'{DATA_PATH}/arc-agi_test_challenges.json')





for jj, tid in enumerate(train_tasks):
    
    if tid in train_tasks.keys():
        train_or_eval = 'train'
        task = train_tasks[tid]
        task_solution = train_sols[tid]
    else:
        train_or_eval = 'eval'
        task = eval_tasks[tid]
        task_solution = eval_sols[tid]

    save_file = f"{FIGURES_PATH}/{tid}_train.png"
    print(f'Train task {jj}: {tid}')
    plot_task(task, f"({jj}) {tid}   {train_or_eval}", 
              task_solution=task_solution, 
              save_file=None)
    time.sleep(0.5)
    print()
    print()
    print()
    
    # if jj == 5: break








for jj, tid in enumerate(eval_tasks):
    
    if tid in train_tasks.keys():
        train_or_eval = 'train'
        task = train_tasks[tid]
        task_solution = train_sols[tid]
    else:
        train_or_eval = 'eval'
        task = eval_tasks[tid]
        task_solution = eval_sols[tid]
        
    print(f'Eval task {jj}: {tid}')
    
    save_file = f"{FIGURES_PATH}/{tid}_eval.png"
    plot_task(task, f"({jj}) {tid}   {train_or_eval}", 
              task_solution=task_solution, 
              save_file=None)
    time.sleep(0.5)
    print()
    print()
    print()
    # if jj == 5: break




