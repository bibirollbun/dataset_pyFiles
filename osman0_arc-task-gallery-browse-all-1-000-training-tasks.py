import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import ipywidgets as widgets
from IPython.display import display

# Fixed dataset path
BASE = "/kaggle/input/arc-prize-2025"

# Renk paleti (ARC)
ARC_COLORS = [
    '#000000',  # 0: Black
    '#0074D9',  # 1: Blue
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Gray
    '#F012BE',  # 6: Pink
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Aqua
    '#870C25'   # 9: Deep red
]
cmap = ListedColormap(ARC_COLORS)

# Load data
with open(os.path.join(BASE, "arc-agi_training_challenges.json"), "r") as f:
    TRAINING = json.load(f)

TASK_IDS = list(TRAINING.keys())
if not TASK_IDS:
    raise ValueError("No tasks found.")

# Simple grid drawing
def visualize_grid(grid, ax, title=""):
    ax.clear()
    if not grid:
        ax.text(0.5, 0.5, 'Empty', ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='#7f8c8d')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10, pad=6)
        return
    A = np.array(grid)
    h, w = A.shape
    ax.imshow(A, cmap=cmap, vmin=0, vmax=9, aspect='equal')
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="#ecf0f1", linestyle='-', linewidth=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10, pad=6)

# Viewer state
state = {"idx": 0}

# Widgets
btn_prev = widgets.Button(description="← Previous")
btn_next = widgets.Button(description="Next →")
lbl_info = widgets.HTML()

out = widgets.Output()
header = widgets.HBox([
    btn_prev,
    widgets.HBox([lbl_info], layout=widgets.Layout(justify_content='center', flex='1 1 auto')),
    btn_next
], layout=widgets.Layout(align_items='center'))

container = widgets.VBox([header, out])

def set_nav_state():
    idx = state["idx"]
    btn_prev.disabled = (idx <= 0)
    btn_next.disabled = (idx >= len(TASK_IDS) - 1)
    tid = TASK_IDS[idx]
    lbl_info.value = f"<b>Task:</b> {idx+1}/{len(TASK_IDS)} • <code>{tid}</code>"

def render():
    set_nav_state()
    tid = TASK_IDS[state["idx"]]
    data = TRAINING[tid]
    train_examples = data.get("train", [])

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ex in enumerate(train_examples[:4]):
        visualize_grid(ex["input"],  axes[0, i], f"Example {i+1} • Input")
        visualize_grid(ex["output"], axes[1, i], f"Example {i+1} • Output")
    for j in range(len(train_examples), 4):
        axes[0, j].axis('off')
        axes[1, j].axis('off')

    fig.suptitle(f"Task {tid} - Training Examples", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out.clear_output(wait=True)
    with out:
        display(fig)
    # Prevent duplicate static render below the widget by closing the figure
    plt.close(fig)

def on_prev(_):
    if state["idx"] > 0:
        state["idx"] -= 1
        render()

def on_next(_):
    if state["idx"] < len(TASK_IDS) - 1:
        state["idx"] += 1
        render()

btn_prev.on_click(on_prev)
btn_next.on_click(on_next)

# You can change the starting index here
state["idx"] = 0  # e.g., set 10 to start at the 11th task

# Run and display
render()
display(container)




