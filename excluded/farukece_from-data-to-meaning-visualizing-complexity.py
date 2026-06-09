from IPython.display import Image
Image(url="https://static.frieze.com/files/styles/hero_image/public/article/main/main-images-herokey-75-new.jpeg?itok=iKSuz9oZ")



import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm



import json
import glob

path = "/kaggle/input/google-code-golf-2025/*.json"

tasks = {}
for filename in sorted(glob.glob(path)):
    with open(filename, "r") as f:
        task_name = filename.split("/")[-1].replace(".json", "")
        tasks[task_name] = json.load(f)

tasks = [json.load(open(f, "r")) for f in sorted(glob.glob(path))]


colors = [
    "#000000", # 0 - Black
    "#1f77ff", # 1 - Blue
    "#e63946", # 2 - Red
    "#4caf50", # 3 - Green
    "#ffeb3b", # 4 - Yellow
    "#9e9e9e", # 5 - Gray
    "#e91e63", # 6 - Pink
    "#ff9800", # 7 - Orange
    "#81d4fa", # 8 - Light Blue
    "#8b1e3f"  # 9 - Dark Red
]

def visualize_indexed_matrix(matrix, colors, ax=None, title=None, show_colorbar=False):
    """
    Visualize an integer matrix using a fixed color palette.
    Each integer index maps directly to its corresponding color.
    Adds grid lines to create a 'gridly' look.
    """
    mat = np.array(matrix, dtype=int)
    n = len(colors)

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n+0.5, 1), ncolors=n)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    im = ax.imshow(mat, cmap=cmap, norm=norm, interpolation="nearest")

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Remove tick labels
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title)

    # Optional colorbar for standalone use
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, ticks=range(n), fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels([str(i) for i in range(n)])
        cbar.set_label("Color Index")

    return im


def visualize_result(task, split, colors, max_cols=5):
    """
    Visualize training examples in a 2 x n grid.
    Handles arc-gen by chunking into smaller grids.
    """
    n = len(task[split])
    if split == "arc-gen" and n > max_cols:
        for start in range(0, n, max_cols):
            end = min(start + max_cols, n)
            visualize_result_chunk(task[split][start:end], colors, split, start+1, end)
    else:
        visualize_result_chunk(task[split], colors, split)


def visualize_result_chunk(points, colors, split, start=1, end=None):
    n = len(points)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))

    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, point in enumerate(points):
        visualize_indexed_matrix(point['input'], colors, ax=axes[0, i], title=f"Input {start+i}")
        visualize_indexed_matrix(point['output'], colors, ax=axes[1, i], title=f"Output {start+i}")

    fig.suptitle(f"{split} examples {start}–{end or start+n-1}")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=col, edgecolor="k", label=f"{i}: {name}") 
                       for i, (col, name) in enumerate(zip(colors, [
                           "Black", "Blue", "Red", "Green", "Yellow", 
                           "Gray", "Pink", "Orange", "Light Blue", "Dark Red"
                       ]))]

    fig.legend(handles=legend_elements, loc="center right", 
               title="Color Index", frameon=True)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()




visualize_result(tasks[0], 'train', colors)


for i in range(400):
    visualize_result(tasks[i], 'train', colors)


visualize_result(tasks[101], 'test', colors)


visualize_result(tasks[205], 'test', colors)


# visualize_result(tasks[307], 'arc-gen', colors)


import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")

import code_golf_utils as cgu


examples = tasks[0]
print("Keys in task_zero:", examples.keys())
print("Number of training examples:", len(examples["train"]))
print("Number of test examples:", len(examples["test"]))
print("Number of arc-gen examples:", len(examples["arc-gen"]))

