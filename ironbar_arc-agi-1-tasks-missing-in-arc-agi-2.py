!git clone https://github.com/arcprize/ARC-AGI-2.git
!git clone https://github.com/fchollet/arc-agi


import glob
import os

partitions = ['training', 'evaluation']

def get_task_ids(folder):
    task_ids = {}
    for partition in partitions:
        filepaths = sorted(glob.glob(os.path.join(folder, partition, '*.json')))
        task_ids[partition] = set([os.path.splitext(os.path.basename(filepath))[0] for filepath in filepaths])
    return task_ids


arc_1_task_ids = get_task_ids('arc-agi/data')
arc_2_task_ids = get_task_ids('ARC-AGI-2/data')


for partition_1 in partitions:
    for partition_2 in partitions:
        print(f'ARC-AGI-1 {partition_1} tasks in ARC-AGI-2 {partition_2}: {len(arc_1_task_ids[partition_1].intersection(arc_2_task_ids[partition_2]))}/{len(arc_1_task_ids[partition_1])}')


arc_2_task_ids['all'] = arc_2_task_ids['training'].union(arc_2_task_ids['evaluation'])
for partition in partitions:
    missing_task_ids = arc_1_task_ids[partition].difference(arc_2_task_ids['all'])
    print(f'This are the ARC-AGI-1 {partition} tasks missing in ARC-AGI-2')
    for task_id in sorted(missing_task_ids):
        print(f'https://arcprize.org/play?task={task_id}')


import json
import numpy as np
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt

cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)
color_list = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]

def plot_task(task, task_id):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=200)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"] + task['test']):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_xticks([])
        axs[0][fig_num].set_yticks([])
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_xticks([])
        axs[1][fig_num].set_yticks([])
        fig_num += 1


    plt.suptitle(task_id)
    plt.tight_layout()
    plt.show()  

for partition in partitions:
    missing_task_ids = arc_1_task_ids[partition].difference(arc_2_task_ids['all'])
    print(f'This are the ARC-AGI-1 {partition} tasks missing in ARC-AGI-2')
    for task_id in sorted(missing_task_ids):
        with open(f'arc-agi/data/{partition}/{task_id}.json') as f:
            task = json.load(f)
        plot_task(task, task_id)

