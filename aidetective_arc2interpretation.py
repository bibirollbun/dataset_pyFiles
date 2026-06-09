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

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

colorNames=["black","blue","red","green","yellow","grey","purple","orange","cyan","brown","white"]
    
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


trainIDs = list(train_tasks.keys())
evalIDs = list(eval_tasks.keys())


def getDims(arr):
    height = len(arr)           # Number of rows
    width = len(arr[0]) if height > 0 else 0  # Number of columns (assuming non-empty rows)
    return width, height


def getColorSet(grid):
    return set(num for row in grid for num in row)


def colorsToString(colorSet):
    return ",".join([colorNames[id] for id in colorSet])


id='0c9aba6e'
plot_task(train_tasks[id],id)


id='8698868d'
plot_task(eval_tasks[id],id)


def describe(task):
    train = task['train']    
    sameSize=True
    sameColors=True
    hasAddedColors=True
    hasSubtractedColors=True
    addedCols=set()
    subtractedCols=set()
    
    puzzleInfo=""
    first=True
    for puzzle in train:
        inputGrid = puzzle['input']
        outputGrid=puzzle['output']
        inputDims = getDims(inputGrid)
        inputCols = getColorSet(inputGrid)
        outputDims = getDims(outputGrid)
        outputCols = getColorSet(outputGrid)
        colorDiff = outputCols-inputCols
        colorDiff2=inputCols-outputCols
        if first:
            addedCols = colorDiff
            subtractedCols = colorDiff2
        else:
            if colorDiff!=addedCols:
                hasAddedColors=False
            if colorDiff2!=subtractedCols:
                hasSubtractedColors=False
        
        if inputDims!=outputDims:
            sameSize=False
        if inputCols!=outputCols:
            sameColors=False
        first=False

    if sameSize:
        puzzleInfo+="[Output sizes same as input sizes] "
    if sameColors:
        puzzleInfo+="[Output colors same as input colors] "
    if hasAddedColors and len(addedCols)>0:
        puzzleInfo+="["+colorsToString(addedCols)+" cells added] "
    if hasSubtractedColors and len(subtractedCols)>0:
        puzzleInfo+="["+colorsToString(subtractedCols)+" cells removed] "
    print(id+" "+puzzleInfo)


for id in evalIDs:
    task = eval_tasks[id]

    describe(task)


for id in trainIDs:
    task = train_tasks[id]

    describe(task)

