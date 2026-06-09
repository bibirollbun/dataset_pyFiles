!ls -la '../input/arc-prize-2024/'


!pip install --quiet simplejson
!pip install --quiet pydash


import glob
import numpy as np
import simplejson as json
import matplotlib.pyplot as plt
from   matplotlib import colors
import pydash as py
from   pydash import py_ as _


def load_tasks(task_file):
    tasks = json.load(open(task_file, 'r'))
    for number, task in tasks.items():
        for test_train, specs in task.items():
            for index, spec in enumerate(specs):
                for input_output, grid in spec.items():
                    tasks[number][test_train][index][input_output] = np.array(grid).astype('uint8')  # uint8 required for cv2 

        for file, task in tasks.items():
            tasks[number]['file'] = task_file.split('/')[-1] + ":" + number 
    return tasks


# Modified from Source: https://www.kaggle.com/zaharch/visualizing-all-tasks-updated
def plot_one(task, ax, i,train_or_test,input_or_output):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    try:
        input_matrix  = task[train_or_test][i][input_or_output]
        font_size     = 50 / np.sqrt(input_matrix.shape[0] * input_matrix.shape[1])
        min_font_size = 5

        ax.imshow(input_matrix, cmap=cmap, norm=norm)
        ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
        ax.imshow(input_matrix, cmap=cmap, norm=norm)
        # DOC: https://stackoverflow.com/questions/33828780/matplotlib-display-array-values-with-imshow
        if font_size >= min_font_size:
            for (j,i),label in np.ndenumerate(input_matrix):
                ax.text(i,j,label,ha='center',va='center', fontsize=font_size, color='black')

        ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
        ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(train_or_test + ' '+input_or_output)
    except: pass  # mat throw on tests, as they have not "output"
    
def plot_task(task, scale=2):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """   
    filename = None
    if isinstance(task, str):   (filename, task) = task, tasks[task]
    if isinstance(task, tuple): (filename, task) = task
    if 'file' in task: filename = task['file']
    
    num_train = len(task['train']) + len(task['test']) + 1
    if 'solution' in task: num_train += len(task['solution']) + 1
    
    fig, axs = plt.subplots(2, num_train, figsize=(scale*num_train,scale*2))
    if filename: fig.suptitle(filename)
        
    for i in range(len(task['train'])):     
        plot_one(task, axs[0,i],i,'train','input')
        plot_one(task, axs[1,i],i,'train','output')            

    axs[0,i+1].axis('off'); axs[1,i+1].axis('off')
    for j in range(len(task['test'])):      
        plot_one(task, axs[0,i+2+j],j,'test','input')
        plot_one(task, axs[1,i+2+j],j,'test','output')  
    
    if 'solution' in task:    
        axs[0,i+j+3].axis('off'); axs[1,i+j+3].axis('off')        
        for k in range(len(task['solution'])):      
            plot_one(task, axs[0,i+j+4+k],k,'solution','input')
            plot_one(task, axs[1,i+j+4+k],k,'solution','output')  

    plt.show()     
    
# plot_task(list(tasks.items())[0])


task_file = '../input/arc-prize-2024/arc-agi_training_challenges.json'
tasks     = load_tasks(task_file)
tasks_ids = list(tasks.keys())
print(_.pick(tasks, tasks_ids[0]), "\n")
print('len(tasks)', len(tasks))
print(tasks_ids)


for name, task in tasks.items():
    plot_task(task)


task_file = '../input/arc-prize-2024/arc-agi_evaluation_challenges.json'
tasks     = load_tasks(task_file)
tasks_ids = list(tasks.keys())
print(_.pick(tasks, tasks_ids[0]), "\n")
print('len(tasks)', len(tasks))
print(tasks_ids)


for name, task in tasks.items():
    plot_task(task)

