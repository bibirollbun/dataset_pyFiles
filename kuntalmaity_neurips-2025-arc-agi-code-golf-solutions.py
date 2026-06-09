import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json #json module lets us parse this text into Python objects (dicts, lists, ints)
import numpy as np  # to handle grids as arrays (shapes, min/max, plotting)
import matplotlib.pyplot as plt  # to display the grids as images
from matplotlib.colors import ListedColormap  # to map numbers (0–9) to colors
from pathlib import Path
import json, glob, zipfile



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# pick one file path (example: task001.json)
task_file_path = "/kaggle/input/google-code-golf-2025/task001.json"

# open and load the JSON
with open(task_file_path, "r") as task_file_object:
    task_data = json.load(task_file_object)

# check the structure
print("Keys in this task file:", task_data.keys())
print("Number of train pairs:", len(task_data["train"]))
print("Number of test pairs:", len(task_data["test"]))
print("Number of arc-gen pairs:", len(task_data["arc-gen"]))


# Take the first training example
first_training_pair = task_data["train"][0]

input_grid = first_training_pair["input"]
output_grid = first_training_pair["output"]

print("Input grid (as a list of lists):")
print(input_grid)

print("\nOutput grid (as a list of lists):")
print(output_grid)


arc_colors = [
    [  0,   0,   0],  # 0 → black
    [  0,   0, 255],  # 1 → blue
    [  0, 255,   0],  # 2 → green
    [255,   0,   0],  # 3 → red
    [255, 255,   0],  # 4 → yellow
    [255,   0, 255],  # 5 → magenta
    [  0, 255, 255],  # 6 → cyan
    [255, 165,   0],  # 7 → orange
    [128,   0, 128],  # 8 → purple
    [192, 192, 192],  # 9 → gray
]
# Convert to 0..1 floats for matplotlib and build a ListedColormap:
# Convert to 0..1 floats for matplotlib and build a ListedColormap:
arc_colormap = ListedColormap(np.array(arc_colors, dtype=float) / 255.0, name="arc_0_9")


def is_valid_arc_grid(grid):
    """
    Returns True if 'grid' is a rectangular list of lists containing only integers in [0, 9].
    Useful sanity check before plotting.
    """
    if not isinstance(grid, list) or (len(grid) > 0 and not isinstance(grid[0], list)):
        return False
    number_of_columns = len(grid[0]) if grid else 0
    for row in grid:
        if not isinstance(row, list) or len(row) != number_of_columns:
            return False
        for value in row:
            if not isinstance(value, int) or value < 0 or value > 9:
                return False
    return True


def show_grid_as_image(grid, title_text="Grid", figure_size=(3, 3)):
    """
    Display one ARC grid (list of lists of ints in [0, 9]) as a color image using the ARC palette.
    - grid: list[list[int]] with rectangular shape
    - title_text: figure title
    - figure_size: size in inches (width, height)
    """
    if not is_valid_arc_grid(grid):
        raise ValueError("Invalid ARC grid: must be rectangular and contain only ints in [0, 9].")

    grid_array = np.array(grid, dtype=int)

    plt.figure(figsize=figure_size)
    plt.imshow(grid_array, cmap=arc_colormap, vmin=0, vmax=9, interpolation="nearest")
    plt.title(title_text)
    plt.axis("off")
    plt.show()


def show_input_output_pair(input_grid, output_grid, main_title_text="Input vs Output", figure_size=(6, 3)):
    """
    Plot a pair of grids (input on the left, output on the right).
    - input_grid, output_grid: list[list[int]] with values in [0, 9]
    - main_title_text: top title
    - figure_size: (width, height) in inches
    """
    if not is_valid_arc_grid(input_grid) or not is_valid_arc_grid(output_grid):
        raise ValueError("Both grids must be valid ARC grids (rectangular, ints in [0, 9]).")

    input_array = np.array(input_grid, dtype=int)
    output_array = np.array(output_grid, dtype=int)

    figure, axes = plt.subplots(1, 2, figsize=figure_size)
    figure.suptitle(main_title_text)

    axes[0].imshow(input_array, cmap=arc_colormap, vmin=0, vmax=9, interpolation="nearest")
    axes[0].set_title("Input")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(output_array, cmap=arc_colormap, vmin=0, vmax=9, interpolation="nearest")
    axes[1].set_title("Output")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    plt.tight_layout()
    plt.show()


def show_arc_palette_legend(figure_size=(6, 1.4)):
    """
    Display a small legend strip showing color squares for labels 0..9.
    Handy reference for what each number means visually.
    """
    labels = list(range(10))
    color_array = np.array([labels], dtype=int)  # shape (1, 10)

    plt.figure(figsize=figure_size)
    plt.imshow(color_array, cmap=arc_colormap, vmin=0, vmax=9, aspect="auto", interpolation="nearest")
    plt.yticks([])  # hide y axis
    plt.xticks(labels, labels)
    plt.title("ARC Palette (0–9)")
    plt.show()


# Example usage with your variables:
show_grid_as_image(input_grid,  title_text="Input Grid")
show_grid_as_image(output_grid, title_text="Output Grid")

# Side-by-side comparison:
show_input_output_pair(input_grid, output_grid, main_title_text="First Training Pair")

# Quick legend for reference:
show_arc_palette_legend()


def show_all_training_pairs(task_data, task_id=None):
    training_pairs = task_data["train"]
    number_of_pairs = len(training_pairs)

    for index, pair in enumerate(training_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

        # Build title text (no hardcoding)
        if task_id:
            title_text = f"Task {task_id} - Training Pair {index+1}/{number_of_pairs}"
        else:
            title_text = f"Training Pair {index+1}/{number_of_pairs}"

        show_input_output_pair(
            input_grid, 
            output_grid, 
            main_title_text=title_text
        )


task_id_label = Path(task_file_path).stem 
show_all_training_pairs(task_data, task_id_label)
# Show TEST pairs
original_train_pairs = task_data["train"]
task_data["train"] = task_data["test"]          # temporarily point "train" to "test"
show_all_training_pairs(task_data, f"{task_id_label}-test")
task_data["train"] = original_train_pairs       # restore

# Show ARC-GEN pairs (limit by editing your function call if needed)
original_train_pairs = task_data["train"]
task_data["train"] = task_data["arc-gen"]       # temporarily point "train" to "arc-gen"
show_all_training_pairs(task_data, f"{task_id_label}-arcgen")
task_data["train"] = original_train_pairs       # restore


def solver_expand_by_template(input_grid):
    """
    Rule:
      - Let T be a binary template of input_grid where T[r][c] = 1 if input_grid[r][c] != 0 else 0.
      - For each input cell (i,j):
          - If input_grid[i][j] == 0 -> place an HxW block of zeros.
          - Else -> place color = input_grid[i][j] wherever T==1 inside the HxW block.
      - Output size becomes (H*H) x (W*W).
    """
    input_height = len(input_grid)
    input_width  = len(input_grid[0]) if input_height else 0

    # Binary template of the input (foreground = 1, background = 0)
    template = [[1 if value != 0 else 0 for value in row] for row in input_grid]

    # Allocate output grid
    output_height = input_height * input_height
    output_width  = input_width  * input_width
    output_grid = [[0] * output_width for _ in range(output_height)]

    # Fill blocks
    for row_index in range(input_height):
        for col_index in range(input_width):
            cell_color = input_grid[row_index][col_index]
            if cell_color == 0:
                continue  # leave the whole block as zeros

            # Paste a recolored copy of the template into the (row_index, col_index) block
            for tr in range(input_height):
                output_row = output_grid[row_index * input_height + tr]
                template_row = template[tr]
                base_col = col_index * input_width
                for tc in range(input_width):
                    if template_row[tc]:
                        output_row[base_col + tc] = cell_color
    return output_grid

# If you want a competition-style entry point, keep this thin wrapper:
def p(grid):
    return solver_expand_by_template(grid)



def grids_are_equal(a, b):
    if len(a) != len(b): 
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb): 
            return False
        for xa, xb in zip(ra, rb):
            if xa != xb: 
                return False
    return True

def evaluate_subset(task_dict, subset_name, solver_func):
    pairs = task_dict.get(subset_name, [])
    correct = 0
    for pair in pairs:
        predicted = solver_func(pair["input"])
        if grids_are_equal(predicted, pair["output"]):
            correct += 1
    total = len(pairs)
    accuracy = correct / total if total else 1.0
    print(f"{subset_name:7s}: {correct}/{total}  ({accuracy:.2%})")
    return correct, total, accuracy

def evaluate_task(task_dict, solver_func):
    print("Evaluation:")
    res_train = evaluate_subset(task_dict, "train",   solver_func)
    res_test  = evaluate_subset(task_dict, "test",    solver_func)
    res_ag    = evaluate_subset(task_dict, "arc-gen", solver_func)
    return {"train": res_train, "test": res_test, "arc-gen": res_ag}

# Run evaluation
_ = evaluate_task(task_data, solver_expand_by_template)



def show_pred_vs_true(task_dict, subset_name="train", count=3, title_prefix="Pred vs True"):
    pairs = task_dict.get(subset_name, [])
    shown = min(count, len(pairs))
    for index in range(shown):
        input_grid = pairs[index]["input"]
        expected_grid = pairs[index]["output"]
        predicted_grid = solver_expand_by_template(input_grid)

        show_input_output_pair(input_grid, expected_grid,
                               main_title_text=f"{title_prefix} • {subset_name} #{index+1} • INPUT vs EXPECTED")
        show_input_output_pair(input_grid, predicted_grid,
                               main_title_text=f"{title_prefix} • {subset_name} #{index+1} • INPUT vs PREDICTED")

# Example: look at first 3 training examples
show_pred_vs_true(task_data, "train", count=3)



# =========================
# Auto-solver v3 RULES
# =========================
import math
from collections import deque, Counter

def grids_equal(a,b):
    return len(a)==len(b) and all(len(ra)==len(rb) and all(x==y for x,y in zip(ra,rb)) for ra,rb in zip(a,b))

def check_rule_on_task(task_dict, solver_func):
    for subset in ("train","test","arc-gen"):
        for pair in task_dict.get(subset, []):
            if not grids_equal(solver_func(pair["input"]), pair["output"]):
                return False
    return True

# ---------- baseline transforms ----------
def r_identity(g): return g
code_identity="def p(g):\n return g\n"

def r_flip_h(g): return [r[::-1] for r in g]
code_flip_h="def p(g):\n return[r[::-1]for r in g]\n"

def r_flip_v(g): return g[::-1]
code_flip_v="def p(g):\n return g[::-1]\n"

def r_transpose(g): return [list(r) for r in zip(*g)]
code_transpose="def p(g):\n return[list(r)for r in zip(*g)]\n"

def r_rot90(g): return [list(r) for r in zip(*g[::-1])]
code_rot90="def p(g):\n return[list(r)for r in zip(*g[::-1])]\n"

def r_rot180(g): return [r[::-1] for r in g[::-1]]
code_rot180="def p(g):\n return[r[::-1]for r in g[::-1]]\n"

def r_rot270(g): return [list(r) for r in zip(*g)][::-1]
code_rot270="def p(g):\n return[list(r)for r in zip(*g)][::-1]\n"

# ---------- crop to nonzero bbox ----------
def r_crop_nonzero(g):
    idx=[(i,j)for i,r in enumerate(g)for j,v in enumerate(r)if v]
    if not idx: return [[]]
    rs=[i for i,_ in idx]; cs=[j for _,j in idx]
    a,b=min(rs),max(rs); c,d=min(cs),max(cs)
    return [r[c:d+1] for r in g[a:b+1]]
code_crop_nonzero=(
"def p(g):\n I=[(i,j)for i,r in enumerate(g)for j,v in enumerate(r)if v]\n"
" if not I:return[[]]\n R=[i for i,_ in I];C=[j for _,j in I]\n"
" a,b=min(R),max(R);c,d=min(C),max(C)\n return[r[c:d+1]for r in g[a:b+1]]\n"
)

# ---------- template expansion (Kronecker on nonzero mask) ----------
def r_template_expand(g):
    h=len(g); w=len(g[0]) if h else 0
    t=[[1 if v else 0 for v in r]for r in g]
    o=[[0]*(w*h) for _ in range(h*h)]
    for i,row in enumerate(g):
        for j,c in enumerate(row):
            if not c: continue
            bi=i*h; bj=j*w
            for a in range(h):
                tr=t[a]; d=o[bi+a]
                for b in range(w):
                    if tr[b]: d[bj+b]=c
    return o
code_template_expand=(
"def p(g):\n h=len(g);w=len(g[0]);t=[[v>0 for v in r]for r in g]\n"
" o=[[0]*(w*h)for _ in range(h*h)]\n"
" for i,r in enumerate(g):\n  for j,c in enumerate(r):\n   if c:\n    bi=i*h;bj=j*w\n    for a in range(h):\n     tr=t[a];d=o[bi+a]\n     for b in range(w):\n      if tr[b]:d[bj+b]=c\n return o\n"
)

# ---------- scale / shrink / pool ----------
def r_scale2(g): return [[v for v in r for _ in(0,1)] for r in g for _ in(0,1)]
code_scale2="def p(g):\n return[[v for v in r for _ in(0,1)]for r in g for _ in(0,1)]\n"

def r_scale3(g): return [[v for v in r for _ in(0,1,2)] for r in g for _ in(0,1,2)]
code_scale3="def p(g):\n return[[v for v in r for _ in(0,1,2)]for r in g for _ in(0,1,2)]\n"

def r_shrink2(g):
    H=len(g); W=len(g[0])
    if H%2 or W%2: return g
    return [[g[i*2][j*2] for j in range(W//2)] for i in range(H//2)]
code_shrink2=(
"def p(g):\n H=len(g);W=len(g[0])\n if H%2 or W%2:return g\n"
" return[[g[i*2][j*2]for j in range(W//2)]for i in range(H//2)]\n"
)

def r_pool2_mode(g):
    H=len(g); W=len(g[0])
    if H%2 or W%2: return g
    o=[]
    for i in range(0,H,2):
        row=[]
        for j in range(0,W,2):
            blk=[g[i][j],g[i][j+1],g[i+1][j],g[i+1][j+1]]
            c=max(blk,key=blk.count)
            row.append(c)
        o.append(row)
    return o
code_pool2=(
"def p(g):\n H=len(g);W=len(g[0])\n if H%2 or W%2:return g\n o=[]\n"
" for i in range(0,H,2):\n  r=[]\n  for j in range(0,W,2):\n   b=[g[i][j],g[i][j+1],g[i+1][j],g[i+1][j+1]]\n   r.append(max(b,key=b.count))\n  o.append(r)\n return o\n"
)

# ---------- symmetry enforcement ----------
def r_make_hsym(g):
    return [[r[j] if j<len(r)/2 else r[-j-1] for j in range(len(r))] for r in g]
code_hsym="def p(g):\n return[[r[j]if j<len(r)/2 else r[-j-1]for j in range(len(r))]for r in g]\n"

def r_make_vsym(g):
    H=len(g);W=len(g[0])
    return [[g[i][j] if i<H/2 else g[-i-1][j] for j in range(W)] for i in range(H)]
code_vsym="def p(g):\n H=len(g);W=len(g[0])\n return[[g[i][j]if i<H/2 else g[-i-1][j]for j in range(W)]for i in range(H)]\n"

# ---------- outlines / borders ----------
def r_outline(g):
    H=len(g); W=len(g[0]); o=[[0]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            c=g[i][j]
            if not c: continue
            if any(0<=a<H and 0<=b<W and g[a][b]==0 for a,b in[(i-1,j),(i+1,j),(i,j-1),(i,j+1)]):
                o[i][j]=c
    return o
code_outline=(
"def p(g):\n H=len(g);W=len(g[0]);o=[[0]*W for _ in range(H)]\n"
" for i in range(H):\n  for j in range(W):\n   c=g[i][j]\n   if c and any(0<=a<H and 0<=b<W and g[a][b]==0 for a,b in[(i-1,j),(i+1,j),(i,j-1),(i,j+1)]):o[i][j]=c\n"
" return o\n"
)

def r_add_border(g):
    H=len(g); W=len(g[0]); o=[[0]*(W+2) for _ in range(H+2)]
    for i in range(H):
        for j in range(W):
            o[i+1][j+1]=g[i][j]
    for j in range(W+2):
        o[0][j]=o[-1][j]=1  # blue border (pick 1)
    for i in range(H+2):
        o[i][0]=o[i][-1]=1
    return o
code_border=(
"def p(g):\n H=len(g);W=len(g[0]);o=[[0]*(W+2)for _ in range(H+2)]\n"
" for i in range(H):\n  for j in range(W):o[i+1][j+1]=g[i][j]\n"
" for j in range(W+2):o[0][j]=o[-1][j]=1\n for i in range(H+2):o[i][0]=o[i][-1]=1\n return o\n"
)

# ---------- row/col projections ----------
def r_row_major(g):
    H=len(g);W=len(g[0]);o=[[0]*W for _ in range(H)]
    for i,r in enumerate(g):
        a=[v for v in r if v]
        if a:o[i]=[max(a,key=a.count)]*W
    return o
code_row_major=(
"def p(g):\n H=len(g);W=len(g[0]);o=[[0]*W for _ in range(H)]\n"
" for i,r in enumerate(g):\n  a=[v for v in r if v]\n  if a:o[i]=[max(a,key=a.count)]*W\n return o\n"
)

def r_col_major(g):
    H=len(g);W=len(g[0]);o=[[0]*W for _ in range(H)]
    for j in range(W):
        a=[g[i][j] for i in range(H) if g[i][j]]
        if a:
            c=max(a,key=a.count)
            for i in range(H): o[i][j]=c
    return o
code_col_major=(
"def p(g):\n H=len(g);W=len(g[0]);o=[[0]*W for _ in range(H)]\n"
" for j in range(W):\n  a=[g[i][j]for i in range(H)if g[i][j]]\n  if a:\n   c=max(a,key=a.count)\n   for i in range(H):o[i][j]=c\n return o\n"
)

# ---------- stripes (detect constant rows/cols and repeat) ----------
def r_make_row_stripes(g):
    H=len(g);W=len(g[0]);pat=min(g,key=lambda r:r.count(0))
    return [pat[:] for _ in range(H)]
code_row_stripes="def p(g):\n H=len(g)\n r=min(g,key=lambda r:r.count(0))\n return[r[:]for _ in range(H)]\n"

def r_make_col_stripes(g):
    H=len(g);W=len(g[0]);col=min(range(W),key=lambda j:sum(1 for i in range(H) if g[i][j]==0))
    c=[g[i][col] for i in range(H)]
    return [[c[i] for _ in range(W)] for i in range(H)]
code_col_stripes=(
"def p(g):\n H=len(g);W=len(g[0])\n j=min(range(W),key=lambda j:sum(g[i][j]==0 for i in range(H)))\n c=[g[i][j]for i in range(H)]\n return[[c[i]for _ in range(W)]for i in range(H)]\n"
)

# ---------- largest connected component (4-neigh) ----------
def r_largest_cc(g):
    H=len(g);W=len(g[0]);vis=[[0]*W for _ in range(H)]
    best=[]; bestc=0
    for i in range(H):
        for j in range(W):
            c=g[i][j]
            if not c or vis[i][j]: continue
            q=[(i,j)]; vis[i][j]=1; comp=[(i,j)]
            while q:
                x,y=q.pop()
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    a,b=x+dx,y+dy
                    if 0<=a<H and 0<=b<W and not vis[a][b] and g[a][b]==c:
                        vis[a][b]=1; q.append((a,b)); comp.append((a,b))
            if len(comp)>len(best): best=comp; bestc=c
    o=[[0]*W for _ in range(H)]
    for x,y in best: o[x][y]=bestc
    return o
code_lcc=(
"def p(g):\n H=len(g);W=len(g[0]);v=[[0]*W for _ in range(H)];B=[];C=0\n"
" for i in range(H):\n  for j in range(W):\n   c=g[i][j]\n   if c and not v[i][j]:\n    q=[(i,j)];v[i][j]=1;comp=[(i,j)]\n    while q:\n     x,y=q.pop()\n     for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):\n      a,b=x+dx,y+dy\n      if 0<=a<H and 0<=b<W and not v[a][b] and g[a][b]==c:\n       v[a][b]=1;q.append((a,b));comp.append((a,b))\n    if len(comp)>len(B):B=comp;C=c\n o=[[0]*W for _ in range(H)]\n for x,y in B:o[x][y]=C\n return o\n"
)

# ---------- draw lines between same-color markers in rows/cols ----------
def r_fill_segments(g):
    H=len(g);W=len(g[0]);o=[r[:] for r in g]
    # rows
    for i in range(H):
        last=None; lastc=None
        for j in range(W):
            if g[i][j]:
                if last is not None and g[i][j]==lastc:
                    for k in range(last+1,j): o[i][k]=lastc
                last=j; lastc=g[i][j]
    # cols
    for j in range(W):
        last=None; lastc=None
        for i in range(H):
            if g[i][j]:
                if last is not None and g[i][j]==lastc:
                    for k in range(last+1,i): o[k][j]=lastc
                last=i; lastc=g[i][j]
    return o
code_fill_segments=(
"def p(g):\n H=len(g);W=len(g[0]);o=[r[:]for r in g]\n"
" for i in range(H):\n  L=C=None\n  for j in range(W):\n   if g[i][j]:\n    if L is not None and g[i][j]==C:\n     for k in range(L+1,j):o[i][k]=C\n    L=j;C=g[i][j]\n"
" for j in range(W):\n  L=C=None\n  for i in range(H):\n   if g[i][j]:\n    if L is not None and g[i][j]==C:\n     for k in range(L+1,i):o[k][j]=C\n    L=i;C=g[i][j]\n return o\n"
)

# ---------- color remap (canonicalize by first appearance order) ----------
def canon(g):
    m={}; nxt=1
    out=[]
    for r in g:
        rr=[]
        for v in r:
            if v and v not in m: m[v]=nxt; nxt+=1
            rr.append(m.get(v,0))
        out.append(rr)
    return out, {v:k for k,v in m.items()}

def r_canon_id(g): return canon(g)[0]
code_canon_id="def p(g):\n m={};n=1;o=[]\n for r in g:\n  t=[]\n  for v in r:\n   if v and v not in m:m[v]=n;n+=1\n   t.append(m.get(v,0))\n  o.append(t)\n return o\n"

# ---------- simple tiling (repeat smallest block that tiles exactly) ----------
def r_tile_detect(g):
    H=len(g);W=len(g[0])
    for h in range(1,H+1):
        if H%h: continue
        for w in range(1,W+1):
            if W%w: continue
            tile=[row[:w] for row in g[:h]]
            ok=True
            for i in range(H):
                for j in range(W):
                    if g[i][j]!=tile[i%h][j%w]: ok=False; break
                if not ok: break
            if ok:
                return tile
    return g
code_tile_detect=(
"def p(g):\n H=len(g);W=len(g[0])\n"
" for h in range(1,H+1):\n  if H%h:continue\n  for w in range(1,W+1):\n   if W%w:continue\n   t=[r[:w]for r in g[:h]];ok=1\n   for i in range(H):\n    for j in range(W):\n     if g[i][j]!=t[i%h][j%w]:ok=0;break\n    if not ok:break\n   if ok:return t\n return g\n"
)

# ---------- library ----------
RULES = [
    ("identity",        r_identity,        code_identity),
    ("flip_h",          r_flip_h,          code_flip_h),
    ("flip_v",          r_flip_v,          code_flip_v),
    ("transpose",       r_transpose,       code_transpose),
    ("rot90",           r_rot90,           code_rot90),
    ("rot180",          r_rot180,          code_rot180),
    ("rot270",          r_rot270,          code_rot270),
    ("crop_nonzero",    r_crop_nonzero,    code_crop_nonzero),
    ("template_expand", r_template_expand, code_template_expand),
    ("scale2",          r_scale2,          code_scale2),
    ("scale3",          r_scale3,          code_scale3),
    ("shrink2",         r_shrink2,         code_shrink2),
    ("pool2",           r_pool2_mode,      code_pool2),
    ("hsym",            r_make_hsym,       code_hsym),
    ("vsym",            r_make_vsym,       code_vsym),
    ("outline",         r_outline,         code_outline),
    ("border",          r_add_border,      code_border),
    ("row_major",       r_row_major,       code_row_major),
    ("col_major",       r_col_major,       code_col_major),
    ("row_stripes",     r_make_row_stripes,code_row_stripes),
    ("col_stripes",     r_make_col_stripes,code_col_stripes),
    ("largest_cc",      r_largest_cc,      code_lcc),
    ("fill_segments",   r_fill_segments,   code_fill_segments),
    ("canon_id",        r_canon_id,        code_canon_id),
    ("tile_detect",     r_tile_detect,     code_tile_detect),
]



# ===========================================
# Extra high-yield rules (append after RULES)
# ===========================================

# 1) Fill the whole grid with the globally most common color
def r_fill_with_most_common_color(g):
    m={}
    for r in g:
        for v in r:m[v]=m.get(v,0)+1
    c=max(m,key=m.get)
    return [[c]*len(g[0]) for _ in g]

code_fill_with_most_common_color=(
"def p(g):\n m={}\n for r in g:\n  for v in r:m[v]=m.get(v,0)+1\n c=max(m,key=m.get)\n return[[c]*len(g[0])for _ in g]\n"
)

# 2) Replace zeros with the majority border color (typical background)
def r_fill_zeros_with_border_majority(g):
    H=len(g);W=len(g[0]);B=g[0]+g[-1]+[g[i][0]for i in range(H)]+[g[i][-1]for i in range(H)]
    m={}
    for v in B:m[v]=m.get(v,0)+1
    c=max(m,key=m.get)
    o=[r[:] for r in g]
    for i in range(H):
        for j in range(W):
            if o[i][j]==0:o[i][j]=c
    return o

code_fill_zeros_with_border_majority=(
"def p(g):\n H=len(g);W=len(g[0]);B=g[0]+g[-1]+[g[i][0]for i in range(H)]+[g[i][-1]for i in range(H)]\n m={}\n for v in B:m[v]=m.get(v,0)+1\n c=max(m,key=m.get)\n o=[r[:]for r in g]\n for i in range(H):\n  for j in range(W):\n   if o[i][j]==0:o[i][j]=c\n return o\n"
)

# 3) Fill along diagonals between same-color anchors
def r_fill_diagonals(g):
    H=len(g);W=len(g[0]);o=[r[:]for r in g]
    for i in range(H):
        for j in range(W):
            c=g[i][j]
            if not c:continue
            x,y=i+1,j+1
            while x<H and y<W and g[x][y]==0:x+=1;y+=1
            if x<H and y<W and g[x][y]==c:
                for k in range(1,min(x-i,y-j)):o[i+k][j+k]=c
            x,y=i+1,j-1
            while x<H and y>=0 and g[x][y]==0:x+=1;y-=1
            if x<H and y>=0 and g[x][y]==c:
                for k in range(1,min(x-i,j-y)):o[i+k][j-k]=c
    return o

code_fill_diagonals=(
"def p(g):\n H=len(g);W=len(g[0]);o=[r[:]for r in g]\n for i in range(H):\n  for j in range(W):\n   c=g[i][j]\n   if not c:continue\n   x,y=i+1,j+1\n   while x<H and y<W and g[x][y]==0:x+=1;y+=1\n   if x<H and y<W and g[x][y]==c:\n    for k in range(1,min(x-i,y-j)):o[i+k][j+k]=c\n   x,y=i+1,j-1\n   while x<H and y>=0 and g[x][y]==0:x+=1;y-=1\n   if x<H and y>=0 and g[x][y]==c:\n    for k in range(1,min(x-i,j-y)):o[i+k][j-k]=c\n return o\n"
)

# 4) Symmetry overlay (horizontal): fill zeros from mirrored column
def r_hsym_overlay(g):
    H=len(g);W=len(g[0]);o=[r[:]for r in g]
    for i in range(H):
        for j in range(W):
            if o[i][j]==0:o[i][j]=g[i][W-1-j]
    return o

code_hsym_overlay=(
"def p(g):\n H=len(g);W=len(g[0]);o=[r[:]for r in g]\n for i in range(H):\n  for j in range(W):\n   if o[i][j]==0:o[i][j]=g[i][W-1-j]\n return o\n"
)

# 5) Symmetry overlay (vertical): fill zeros from mirrored row
def r_vsym_overlay(g):
    H=len(g);W=len(g[0]);o=[r[:]for r in g]
    for i in range(H):
        for j in range(W):
            if o[i][j]==0:o[i][j]=g[H-1-i][j]
    return o

code_vsym_overlay=(
"def p(g):\n H=len(g);W=len(g[0]);o=[r[:]for r in g]\n for i in range(H):\n  for j in range(W):\n   if o[i][j]==0:o[i][j]=g[H-1-i][j]\n return o\n"
)

# 6) Largest component then crop to its bbox
def r_lcc_then_crop(g):
    H=len(g);W=len(g[0]);v=[[0]*W for _ in range(H)]
    best=[];color=0
    for i in range(H):
        for j in range(W):
            c=g[i][j]
            if not c or v[i][j]:continue
            q=[(i,j)];v[i][j]=1;S=[(i,j)]
            while q:
                x,y=q.pop()
                for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):
                    a,b=x+dx,y+dy
                    if 0<=a<H and 0<=b<W and not v[a][b] and g[a][b]==c:
                        v[a][b]=1;q.append((a,b));S.append((a,b))
            if len(S)>len(best):best=S;color=c
    if not best:return[[]]
    R=[i for i,_ in best];C=[j for _,j in best]
    t,b=min(R),max(R);l,r=min(C),max(C);S=set(best)
    return[[color if (t+x,l+y) in S else 0 for y in range(r-l+1)]for x in range(b-t+1)]

code_lcc_then_crop=(
"def p(g):\n H=len(g);W=len(g[0]);v=[[0]*W for _ in range(H)];B=[];C=0\n for i in range(H):\n  for j in range(W):\n   c=g[i][j]\n   if c and not v[i][j]:\n    q=[(i,j)];v[i][j]=1;S=[(i,j)]\n    while q:\n     x,y=q.pop()\n     for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):\n      a,b=x+dx,y+dy\n      if 0<=a<H and 0<=b<W and not v[a][b] and g[a][b]==c:\n       v[a][b]=1;q.append((a,b));S.append((a,b))\n    if len(S)>len(B):B=S;C=c\n if not B:return[[]]\n R=[i for i,_ in B];K=[j for _,j in B]\n t,b=min(R),max(R);l,r=min(K),max(K);S=set(B)\n return[[C if (t+x,l+y) in S else 0 for y in range(r-l+1)]for x in range(b-t+1)]\n"
)

# 7c) 3x3 shrink by mode (very common on ARC)
def r_shrink3_mode(g):
    H=len(g);W=len(g[0])
    if H%3 or W%3:return g
    o=[]
    for i in range(0,H,3):
        row=[]
        for j in range(0,W,3):
            block=[g[i+a][j+b] for a in (0,1,2) for b in (0,1,2)]
            row.append(max(block,key=block.count))
        o.append(row)
    return o

code_shrink3_mode=(
"def p(g):\n H=len(g);W=len(g[0])\n if H%3 or W%3:return g\n o=[]\n"
" for i in range(0,H,3):\n  r=[]\n  for j in range(0,W,3):\n   b=[g[i+a][j+b]for a in(0,1,2)for b in(0,1,2)]\n"
"   r.append(max(b,key=b.count))\n  o.append(r)\n return o\n"
)


# --- Register them ---
RULES += [
    ("shrink3_mode", r_shrink3_mode, code_shrink3_mode),
    ("fill_with_most_common_color",      r_fill_with_most_common_color,      code_fill_with_most_common_color),
    ("fill_zeros_with_border_majority",  r_fill_zeros_with_border_majority,  code_fill_zeros_with_border_majority),
    ("fill_diagonals",                   r_fill_diagonals,                    code_fill_diagonals),
    ("hsym_overlay",                     r_hsym_overlay,                      code_hsym_overlay),
    ("vsym_overlay",                     r_vsym_overlay,                      code_vsym_overlay),
    ("lcc_then_crop",                    r_lcc_then_crop,                     code_lcc_then_crop),
]



# =========================
# Task-specific synthesizers (add after your RULES)
# =========================
from itertools import product

# Reuse the equality helper used by your sweep
def _grids_equal(a,b):
    return len(a)==len(b) and all(len(ra)==len(rb) and all(x==y for x,y in zip(ra,rb)) for ra,rb in zip(a,b))

# D4 (dihedral) transforms we will search, with both a function and its golfed expression "on g"
_D4 = [
    ("id",     lambda g: g,                            "g"),
    ("fliph",  lambda g: [r[::-1] for r in g],         "[r[::-1]for r in g]"),
    ("flipv",  lambda g: g[::-1],                      "g[::-1]"),
    ("t",      lambda g: [list(r) for r in zip(*g)],   "[list(r)for r in zip(*g)]"),
    ("r90",    lambda g: [list(r) for r in zip(*g[::-1])],        "[list(r)for r in zip(*g[::-1])]"),
    ("r180",   lambda g: [r[::-1] for r in g[::-1]],              "[r[::-1]for r in g[::-1]]"),
    ("r270",   lambda g: [list(r) for r in zip(*g)][::-1],        "[list(r)for r in zip(*g)][::-1]"),
]

def _learn_color_perm(x, y):
    """
    Try to learn a 10-color mapping m (list of length 10) such that y == map(m, x).
    Returns m if consistent for ALL seen colors in x, else None.
    """
    m = [-1]*10
    H = len(x); W = len(x[0]) if H else 0
    for i in range(H):
        for j in range(W):
            a = x[i][j]   # input color
            b = y[i][j]   # output color
            if m[a] == -1:
                m[a] = b
            elif m[a] != b:
                return None   # conflict
    # fill unmapped colors with identity (safe default)
    for c in range(10):
        if m[c] == -1: m[c] = c
    return m

def _apply_map(m, g):
    return [[m[v] for v in r] for r in g]

def synth_d4_colorperm(task_dict):
    """
    Try all D4 transforms; for each, learn a single color permutation consistent across ALL pairs
    (train+test+arc-gen). If found, return (solver_func, code_src, name). Otherwise None.
    """
    pairs = task_dict.get("train", []) + task_dict.get("test", []) + task_dict.get("arc-gen", [])
    if not pairs:
        return None

    for tname, tfunc, texpr in _D4:
        # 1) learn mapping from first pair
        first_in = pairs[0]["input"]
        first_out = pairs[0]["output"]
        xin = tfunc(first_in)
        if len(xin) != len(first_out) or len(xin[0]) != len(first_out[0]):
            continue
        m = _learn_color_perm(xin, first_out)
        if m is None:
            continue

        # 2) validate mapping on all pairs
        ok = True
        for p in pairs:
            x = tfunc(p["input"])
            if len(x) != len(p["output"]) or len(x[0]) != len(p["output"][0]):
                ok = False; break
            if not _grids_equal(_apply_map(m, x), p["output"]):
                ok = False; break
        if not ok:
            continue

        # 3) Build solver function and very short code for this specific task
        def solver_func(g, _m=m, _t=tfunc):
            return _apply_map(_m, _t(g))

        # code: m as a 10-length list + the transform expr + list-comp mapping
        mlist = ",".join(str(c) for c in m)
        code_src = (
            "def p(g):\n"
            f" m=[{mlist}]\n"
            f" t={texpr}\n"
            " return[[m[v]for v in r]for r in t]\n"
        )
        return solver_func, code_src, f"d4perm_{tname}"

    return None

# 8b) Lattice synthesizer: keep k-spaced rows/cols (hash) or intersection dots
def _majority_color(grid):
    m={}
    for row in grid:
        for v in row:m[v]=m.get(v,0)+1
    return max(m,key=m.get)

def _apply_lattice(g,k,r0,c0,use_or,bg):
    H=len(g);W=len(g[0]);o=[[bg]*W for _ in g]
    for i in range(H):
        ri=(i%k==r0)
        for j in range(W):
            cj=(j%k==c0)
            keep=(ri or cj) if use_or else (ri and cj)
            if keep and g[i][j]!=bg:o[i][j]=g[i][j]
    return o

def synth_lattice_hash_or_dots(task_dict):
    pairs = task_dict.get("train", [])+task_dict.get("test", [])+task_dict.get("arc-gen", [])
    if not pairs:return None

    # learn k in [2..6] and consistent (r0,c0) and OR/AND across ALL pairs
    for k in range(2,7):
        for r0 in range(k):
            for c0 in range(k):
                for use_or in (True,False):
                    ok=True
                    # choose background from output majority per pair
                    for pair in pairs:
                        g=pair["input"]; y=pair["output"]
                        bg=_majority_color(y) if y else _majority_color(g)
                        if _apply_lattice(g,k,r0,c0,use_or,bg)!=y:
                            ok=False;break
                    if not ok:continue

                    # Build solver + golfed code
                    def solver_func(g,_k=k,_r=r0,_c=c0,_or=use_or):
                        bg=_majority_color(g)
                        # try output-majority logic inside sweep when validating; at runtime, input-majority is fine
                        return _apply_lattice(g,_k,_r,_c,_or,bg)

                    op="or" if use_or else "and"
                    # golfed code
                    code_src=(
                        "def p(g):\n"
                        " m={}\n"
                        " for r in g:\n"
                        "  for v in r:m[v]=m.get(v,0)+1\n"
                        " b=max(m,key=m.get)\n"
                        f" k={k};R={r0};C={c0}\n"
                        " H=len(g);W=len(g[0]);o=[[b]*W for _ in g]\n"
                        " for i in range(H):\n"
                        "  for j in range(W):\n"
                        f"   if ((i%k==R){' or ' if use_or else ' and '}(j%k==C)) and g[i][j]!=b:o[i][j]=g[i][j]\n"
                        " return o\n"
                    )
                    return solver_func, code_src, f"lattice_{'hash' if use_or else 'dots'}_k{k}_r{r0}_c{c0}"
    return None

# 8c) D4 + crop_nonzero + color-permutation synthesizer
def _crop_nonzero(x):
    idx=[(i,j)for i,r in enumerate(x)for j,v in enumerate(r)if v]
    if not idx:return [[]]
    R=[i for i,_ in idx];C=[j for _,j in idx]
    a,b=min(R),max(R);c,d=min(C),max(C)
    return [r[c:d+1] for r in x[a:b+1]]

def synth_d4_crop_colorperm(task_dict):
    pairs = task_dict.get("train", []) + task_dict.get("test", []) + task_dict.get("arc-gen", [])
    if not pairs: return None

    for tname, tfunc, texpr in _D4:
        # learn mapping from first pair after crop
        x0 = _crop_nonzero(tfunc(pairs[0]["input"]))
        y0 = pairs[0]["output"]
        if len(x0)!=len(y0) or len(x0[0])!=len(y0[0]): 
            continue
        m = _learn_color_perm(x0, y0)
        if m is None: 
            continue

        ok=True
        for pair in pairs:
            xin = _crop_nonzero(tfunc(pair["input"]))
            y   = pair["output"]
            if len(xin)!=len(y) or len(xin[0])!=len(y[0]): 
                ok=False; break
            if not _grids_equal(_apply_map(m, xin), y):
                ok=False; break
        if not ok: 
            continue

        # solver + compact code
        def solver_func(g, _m=m, _t=tfunc):
            return _apply_map(_m, _crop_nonzero(_t(g)))

        mlist=",".join(str(c) for c in m)
        code_src=(
            "def p(g):\n"
            f" m=[{mlist}]\n"
            f" t={texpr}\n"
            " x=t\n"
            " I=[(i,j)for i,r in enumerate(x)for j,v in enumerate(r)if v]\n"
            " if not I:return[[]]\n"
            " R=[i for i,_ in I];C=[j for _,j in I]\n"
            " a,b=min(R),max(R);c,d=min(C),max(C)\n"
            " x=[r[c:d+1]for r in x[a:b+1]]\n"
            " return[[m[v]for v in r]for r in x]\n"
        )
        return solver_func, code_src, f"d4_crop_perm_{tname}"
    return None

# 8d) Tile expansion synthesizer (repeat input to exactly match output size)
def _tile_expand(x, H, W):
    h=len(x); w=len(x[0]) if h else 0
    return [[x[i%h][j%w] for j in range(W)] for i in range(H)]

def synth_tile_expand(task_dict):
    pairs = task_dict.get("train", []) + task_dict.get("test", []) + task_dict.get("arc-gen", [])
    if not pairs: return None

    # Learn that output dims = multiples of input dims, and that y == tile_expand(x)
    ok=True
    for pair in pairs:
        x = pair["input"]; y = pair["output"]
        h=len(x); w=len(x[0]) if h else 0
        H=len(y); W=len(y[0]) if H else 0
        if h==0 or w==0 or H%h or W%w:
            ok=False; break
        if _tile_expand(x,H,W)!=y:
            ok=False; break
    if not ok: 
        return None

    # solver + compact code
    def solver_func(g):
        H=len(g)* (len(pairs[0]["output"])//len(pairs[0]["input"]))
        W=len(g[0])*(len(pairs[0]["output"][0])//len(pairs[0]["input"][0]))
        return _tile_expand(g,H,W)

    # golfed program: compute H,W as multiples of input (from ratios learned on first pair)
    x0=pairs[0]["input"]; y0=pairs[0]["output"]
    rH=len(y0)//len(x0); rW=len(y0[0])//len(x0[0])
    code_src=(
        "def p(g):\n"
        f" RH={rH};RW={rW}\n"
        " H=len(g)*RH;W=len(g[0])*RW\n"
        " return[[g[i%len(g)][j%len(g[0])] for j in range(W)] for i in range(H)]\n"
    )
    return solver_func, code_src, f"tile_expand_{rH}x{rW}"

def _const_block(color_block):
    it = iter(color_block)
    first = next(it)
    return all(v==first for v in it), first

def synth_scale_block_colorperm(task_dict):
    pairs = task_dict.get("train", []) + task_dict.get("test", []) + task_dict.get("arc-gen", [])
    if not pairs: return None

    # Learn factors from first pair
    x0, y0 = pairs[0]["input"], pairs[0]["output"]
    h0, w0 = len(x0), len(x0[0]) if x0 else 0
    H0, W0 = len(y0), len(y0[0]) if y0 else 0
    if not h0 or not w0 or H0%h0 or W0%w0: return None
    rH, rW = H0//h0, W0//w0

    # Learn color map from first pair
    m = [-1]*10
    for i in range(h0):
        for j in range(w0):
            block = [y0[i*rH+di][j*rW+dj] for di in range(rH) for dj in range(rW)]
            ok,c = _const_block(block)
            if not ok: return None
            a = x0[i][j]
            if m[a] in (-1, c): m[a] = c
            else: return None
    for c in range(10):
        if m[c] == -1: m[c] = c

    # Validate across all pairs
    def _expand(g):
        gh, gw = len(g), len(g[0]) if g else 0
        out = [[0]*(gw*rW) for _ in range(gh*rH)]
        for i in range(gh):
            for j in range(gw):
                c = m[g[i][j]]
                bi, bj = i*rH, j*rW
                for di in range(rH):
                    row = out[bi+di]
                    for dj in range(rW):
                        row[bj+dj] = c
        return out

    for p in pairs:
        if _expand(p["input"]) != p["output"]:
            return None

    mlist = ",".join(str(c) for c in m)
    code_src = (
        "def p(g):\n"
        f" RH, RW = {rH}, {rW}\n"
        f" m=[{mlist}]\n"
        " H=len(g); W=len(g[0])\n"
        " o=[[0]*(W*RW) for _ in range(H*RH)]\n"
        " for i in range(H):\n"
        "  for j in range(W):\n"
        "   c=m[g[i][j]];bi=i*RH;bj=j*RW\n"
        "   for di in range(RH):\n"
        "    r=o[bi+di]\n"
        "    for dj in range(RW): r[bj+dj]=c\n"
        " return o\n"
    )
    return (lambda g: [[m[v] for v in r for _ in range(rW)] for r in g for _ in range(rH)]), code_src, f"scale_block_{rH}x{rW}"
def _mode(values):
    return max(values, key=values.count)

def synth_shrink_block_mode(task_dict):
    pairs = task_dict.get("train", []) + task_dict.get("test", []) + task_dict.get("arc-gen", [])
    if not pairs: return None

    x0, y0 = pairs[0]["input"], pairs[0]["output"]
    h0, w0 = len(x0), len(x0[0]) if x0 else 0
    H0, W0 = len(y0), len(y0[0]) if y0 else 0
    if not h0 or not w0 or not H0 or not W0: return None
    if h0%H0 or w0%W0: return None
    kH, kW = h0//H0, w0//W0

    def shrink(g):
        H, W = len(g)//kH, len(g[0])//kW
        out=[]
        for i in range(H):
            row=[]
            for j in range(W):
                block=[g[i*kH+di][j*kW+dj] for di in range(kH) for dj in range(kW)]
                row.append(_mode(block))
            out.append(row)
        return out

    for p in pairs:
        gx, gy = p["input"], p["output"]
        if len(gx)%kH or len(gx[0])%kW: return None
        if shrink(gx) != gy: return None

    code_src = (
        "def p(g):\n"
        f" kH,kW={kH},{kW}\n"
        " H=len(g)//kH;W=len(g[0])//kW;o=[]\n"
        " for i in range(H):\n"
        "  r=[]\n"
        "  for j in range(W):\n"
        "   b=[g[i*kH+di][j*kW+dj] for di in range(kH) for dj in range(kW)]\n"
        "   r.append(max(b,key=b.count))\n"
        "  o.append(r)\n"
        " return o\n"
    )
    return (lambda g: [[_mode([g[i*kH+di][j*kW+dj] for di in range(kH) for dj in range(kW)]) for j in range(len(g[0])//kW)] for i in range(len(g)//kH)]), code_src, f"shrink_mode_{kH}x{kW}"
def _paint_border(g, w, c):
    H=len(g);W=len(g[0]);o=[r[:] for r in g]
    for t in range(w):
        for j in range(W): o[t][j]=o[H-1-t][j]=c
        for i in range(H): o[i][t]=o[i][W-1-t]=c
    return o

def synth_uniform_border(task_dict):
    pairs = task_dict.get("train", []) + task_dict.get("test", []) + task_dict.get("arc-gen", [])
    if not pairs: return None
    y0 = pairs[0]["output"]
    if not y0: return None
    c0 = y0[0][0]  # learn color from first pair
    for w in (1,2,3):
        ok=True
        for p in pairs:
            x,y = p["input"], p["output"]
            if _paint_border(x,w,c0)!=y: ok=False; break
        if ok:
            code_src = (
                "def p(g):\n"
                f" w={w};c={c0}\n"
                " H=len(g);W=len(g[0]);o=[r[:]for r in g]\n"
                " for t in range(w):\n"
                "  for j in range(W):o[t][j]=o[H-1-t][j]=c\n"
                "  for i in range(H):o[i][t]=o[i][W-1-t]=c\n"
                " return o\n"
            )
            return (lambda g, _w=w, _c=c0: _paint_border(g,_w,_c)), code_src, f"border_w{w}_c{c0}"
    return None
def synth_recolor_nonzero_constant(task_dict):
    pairs = task_dict.get("train", []) + task_dict.get("test", []) + task_dict.get("arc-gen", [])
    if not pairs: return None
    y0 = pairs[0]["output"]
    if not y0: return None
    # infer color as majority of non-zero in first output
    m={}
    for r in y0:
        for v in r:
            if v: m[v]=m.get(v,0)+1
    if not m: return None
    c = max(m,key=m.get)

    for p in pairs:
        x,y = p["input"], p["output"]
        H,W = len(x), len(x[0]) if x else 0
        if len(y)!=H or (H and len(y[0])!=W): return None
        test = [[(c if x[i][j] else 0) for j in range(W)] for i in range(H)]
        if test!=y: return None

    code_src = (
        "def p(g):\n"
        f" c={c}\n"
        " H=len(g);W=len(g[0])\n"
        " return[[c if g[i][j] else 0 for j in range(W)] for i in range(H)]\n"
    )
    return (lambda g, _c=c: [[(_c if v else 0) for v in r] for r in g]), code_src, f"recolor_nonzero_{c}"
def _fill_holes(g):
    H=len(g);W=len(g[0])
    vis=[[0]*W for _ in range(H)]
    from collections import deque
    q=deque()
    # mark outside zeros via border flood
    for i in range(H):
        for j in (0,W-1):
            if g[i][j]==0 and not vis[i][j]: vis[i][j]=1; q.append((i,j))
    for j in range(W):
        for i in (0,H-1):
            if g[i][j]==0 and not vis[i][j]: vis[i][j]=1; q.append((i,j))
    while q:
        x,y=q.popleft()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            a,b=x+dx,y+dy
            if 0<=a<H and 0<=b<W and not vis[a][b] and g[a][b]==0:
                vis[a][b]=1; q.append((a,b))
    # fill unvisited zeros (holes)
    o=[r[:] for r in g]
    for i in range(H):
        for j in range(W):
            if g[i][j]==0 and not vis[i][j]:
                neigh=[]
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    a,b=i+dx,j+dy
                    if 0<=a<H and 0<=b<W and g[a][b]!=0: neigh.append(g[a][b])
                if neigh:
                    c=max(neigh,key=neigh.count)
                    o[i][j]=c
    return o

def synth_fill_holes_by_component(task_dict):
    pairs = task_dict.get("train", []) + task_dict.get("test", []) + task_dict.get("arc-gen", [])
    if not pairs: return None
    for p in pairs:
        if _fill_holes(p["input"])!=p["output"]:
            return None
    code_src = (
        "def p(g):\n"
        " H=len(g);W=len(g[0]);v=[[0]*W for _ in range(H)]\n"
        " from collections import deque\n"
        " q=deque()\n"
        " for i in range(H):\n"
        "  for j in (0,W-1):\n"
        "   if g[i][j]==0 and not v[i][j]:v[i][j]=1;q.append((i,j))\n"
        " for j in range(W):\n"
        "  for i in (0,H-1):\n"
        "   if g[i][j]==0 and not v[i][j]:v[i][j]=1;q.append((i,j))\n"
        " while q:\n"
        "  x,y=q.popleft()\n"
        "  for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):\n"
        "   a,b=x+dx,y+dy\n"
        "   if 0<=a<H and 0<=b<W and not v[a][b] and g[a][b]==0:v[a][b]=1;q.append((a,b))\n"
        " o=[r[:]for r in g]\n"
        " for i in range(H):\n"
        "  for j in range(W):\n"
        "   if g[i][j]==0 and not v[i][j]:\n"
        "    n=[]\n"
        "    for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):\n"
        "     a,b=i+dx,j+dy\n"
        "     if 0<=a<H and 0<=b<W and g[a][b]!=0:n.append(g[a][b])\n"
        "    if n:o[i][j]=max(n,key=n.count)\n"
        " return o\n"
    )
    return (lambda g: _fill_holes(g)), code_src, "fill_holes"

if 'SYNTHESIZERS' not in globals():
    SYNTHESIZERS = []


def register_synths(items):
    """Add (name, fn) pairs to SYNTHESIZERS without duplicates."""
    existing = {name for name, _ in SYNTHESIZERS}
    for name, fn in items:
        if name not in existing:
            SYNTHESIZERS.append((name, fn))


# Register it (after your existing SYNTHESIZERS)
register_synths([
    ("scale_block_colorperm", synth_scale_block_colorperm),
    ("shrink_block_mode",     synth_shrink_block_mode),
    ("uniform_border",        synth_uniform_border),
    ("recolor_nonzero",       synth_recolor_nonzero_constant),
    ("fill_holes",            synth_fill_holes_by_component),

    # if these were already registered earlier, register_synths will skip them:
    ("lattice",               synth_lattice_hash_or_dots),
    ("d4_colorperm",          synth_d4_colorperm),
    ("d4_crop_perm",          synth_d4_crop_colorperm),
    ("tile_expand",           synth_tile_expand),
])





# =========================
# Quiet sweep + separate ZIP builder
# =========================
from pathlib import Path
import json, glob, zipfile, re

DATA_DIR = "/kaggle/input/google-code-golf-2025"
WORK_DIR = Path("/kaggle/working"); WORK_DIR.mkdir(exist_ok=True)

def load_task(json_path: str):
    with open(json_path, "r") as file_object:
        return json.load(file_object)

def write_solver(task_id: str, source_text: str) -> Path:
    out_path = WORK_DIR / f"task{int(task_id):03d}.py"
    with open(out_path, "wb") as binary_file:
        binary_file.write(source_text.strip().encode("utf-8"))
    return out_path

def estimate_points(byte_len: int) -> int:
    # Kaggle scoring per task
    return max(1, 2500 - byte_len)

def _check_rule_on_task(task_dictionary: dict, solver_function) -> bool:
    # Uses _grids_equal from your earlier cell; if needed, define a local one here.
    for subset_name in ("train", "test", "arc-gen"):
        for pair in task_dictionary.get(subset_name, []):
            try:
                output_grid = solver_function(pair["input"])
            except Exception:
                return False
            if not _grids_equal(output_grid, pair["output"]):  # relies on your earlier helper
                return False
    return True

def run_sweep_quiet(data_dir: str, work_dir: Path, emit_per_task_logs: bool = False):
    """
    Try RULES then SYNTHESIZERS for every task; write the shortest passing program to work_dir/taskNNN.py.
    Returns (winners, unsolved, per_task_log). No noisy per-task prints unless emit_per_task_logs=True.
    """
    json_paths = sorted(glob.glob(f"{data_dir}/task*.json"))
    winners = []      # list[(task_id, picked_name, byte_len)]
    unsolved = []     # list[task_id]
    per_task_log = [] # list[("ok"/"stub", task_id, picked_name_or_None, byte_len_or_None)]

    for json_path in json_paths:
        task_id = Path(json_path).stem.replace("task","")
        task_dictionary = load_task(json_path)

        candidates = []

        # 1) fixed RULES
        for rule_name, solver_function, code_source in RULES:
            try:
                if _check_rule_on_task(task_dictionary, solver_function):
                    candidates.append((len(code_source.encode("utf-8")), f"rule:{rule_name}", code_source))
            except Exception:
                pass

        # 2) task-specific SYNTHESIZERS
        for synthesizer_name, synthesizer in (SYNTHESIZERS if 'SYNTHESIZERS' in globals() else []):
            try:
                result = synthesizer(task_dictionary)
                if result is not None:
                    solver_function, code_source, generated_name = result
                    if _check_rule_on_task(task_dictionary, solver_function):
                        candidates.append((len(code_source.encode("utf-8")), f"{synthesizer_name}:{generated_name}", code_source))
            except Exception:
                pass

        if candidates:
            candidates.sort()  # shortest first
            byte_len, picked_name, code_source = candidates[0]
            write_solver(task_id, code_source)
            winners.append((task_id, picked_name, byte_len))
            if emit_per_task_logs:
                print(f"✔ task{task_id}: {picked_name}  ({byte_len} bytes)")
            per_task_log.append(("ok", task_id, picked_name, byte_len))
        else:
            # stub fallback (quiet)
            write_solver(task_id, "def p(g):\n return g\n")
            unsolved.append(task_id)
            if emit_per_task_logs:
                print(f"✖ task{task_id}: no match (stub)")
            per_task_log.append(("stub", task_id, None, None))

    return winners, unsolved, per_task_log

def build_submission_zip(work_dir: Path, zip_name: str = "submission.zip") -> Path:
    """
    Pack all taskNNN.py files from work_dir into a flat ZIP named zip_name.
    Returns the ZIP path. Quiet by default.
    """
    destination_path = work_dir / zip_name
    with zipfile.ZipFile(destination_path, "w", zipfile.ZIP_DEFLATED) as zip_object:
        for python_file in sorted(work_dir.glob("task*.py")):
            zip_object.write(python_file, arcname=python_file.name)
    return destination_path

def verify_submission_zip(zip_path: Path, show=10):
    """
    Light sanity check: existence, size, and a short listing.
    """
    exists = zip_path.exists()
    size   = zip_path.stat().st_size if exists else 0
    print("submission.zip exists?:", exists, "| size (bytes):", size)
    if exists and size > 0:
        with zipfile.ZipFile(zip_path, "r") as zip_object:
            names = [n for n in zip_object.namelist() if n.endswith(".py")]
            print("Files in zip:", len(names), "| First", min(show, len(names)), ":", names[:show])

# ---- Run (quiet) ----
winners, unsolved, _ = run_sweep_quiet(DATA_DIR, WORK_DIR, emit_per_task_logs=False)

# ---- Build ZIP in a separate step ----
zip_path = build_submission_zip(WORK_DIR, "submission.zip")

# ---- Concise summary only (no per-task spam) ----
solved_points = sum(estimate_points(byte_len) for _, _, byte_len in winners)
stub_points = 0.001 * len(unsolved)
print(f"Wrote: {zip_path}")
print(f"Solved: {len(winners)} | Unsolved (stubs): {len(unsolved)}")
print(f"Estimated score from solved: {solved_points:.3f}")
print(f"+ stubs: {stub_points:.3f}")
print(f"≈ Estimated total: {solved_points + stub_points:.3f}")

# Optional: tiny verification so Kaggle UI definitely sees the file
verify_submission_zip(zip_path, show=10)



# --- Sanity: ensure the zip has flat, correctly named files and p() compiles ---
# import re, zipfile
# from pathlib import Path

# submission_zip_path = Path("/kaggle/working/submission.zip")
# with zipfile.ZipFile(submission_zip_path, "r") as z:
#     names = z.namelist()
#     print("Files in zip:", len(names), "| First 10:", names[:10])
#     bad = [n for n in names if "/" in n or "\\" in n or not re.fullmatch(r"task\d{3}\.py", n)]
#     print("Bad entries:", bad[:5] if bad else "None")



# # ===========================================
# # 11) Diagnostics: validate the zip and each `p(g)`
# # - Confirms the ZIP is flat and correctly named (taskNNN.py)
# # - Compiles every file and checks that a callable p(g) exists
# # - (Optional) Functionally validates N tasks against their JSON
# # ===========================================

# from pathlib import Path
# import zipfile, re, json

# SUBMISSION_ZIP_PATH = Path("/kaggle/working/submission.zip")
# DATASET_DIR_PATH    = Path("/kaggle/input/google-code-golf-2025")  # contains taskNNN.json

# def load_task_json(task_json_path: Path) -> dict:
#     with open(task_json_path, "r") as file_object:
#         return json.load(file_object)

# def grids_equal(grid_a, grid_b) -> bool:
#     if len(grid_a) != len(grid_b):
#         return False
#     for row_a, row_b in zip(grid_a, grid_b):
#         if len(row_a) != len(row_b):
#             return False
#         for value_a, value_b in zip(row_a, row_b):
#             if value_a != value_b:
#                 return False
#     return True

# def compile_and_get_predict_function(source_text: str, module_name: str):
#     execution_environment = {}
#     compiled = compile(source_text, module_name, "exec")
#     exec(compiled, execution_environment)
#     predict_function = execution_environment.get("p", None)
#     return predict_function

# def validate_solver_text_on_task(source_text: str, task_data: dict) -> bool:
#     """Returns True iff p(g) from source_text matches ALL pairs across train/test/arc-gen."""
#     predict_function = compile_and_get_predict_function(source_text, "<in-memory>")
#     if not callable(predict_function):
#         return False
#     for subset_name in ("train", "test", "arc-gen"):
#         for pair in task_data.get(subset_name, []):
#             try:
#                 predicted_grid = predict_function(pair["input"])
#             except Exception:
#                 return False
#             if not grids_equal(predicted_grid, pair["output"]):
#                 return False
#     return True

# def diagnose_submission_zip(
#     zip_path: Path,
#     dataset_dir_path: Path,
#     sample_limit: int = 10,      # functional check for the first N tasks; set 0 for skip, 400 for all
# ) -> None:
#     assert zip_path.exists(), f"submission.zip not found at {zip_path}"
#     with zipfile.ZipFile(zip_path, "r") as zip_object:
#         zip_names = [name for name in zip_object.namelist() if name.endswith(".py")]

#         # --- Structure checks (flat, correctly named) ---
#         print(f"Files in zip: {len(zip_names)}")
#         print("First 10:", zip_names[:10])

#         bad_entries = [
#             name for name in zip_names
#             if ("/" in name or "\\" in name or not re.fullmatch(r"task\d{3}\.py", name))
#         ]
#         if bad_entries:
#             print("❌ Bad or nested names (examples):", bad_entries[:10])
#         else:
#             print("✅ Filenames look good (flat `taskNNN.py`)")

#         # --- Compile + p(g) existence checks ---
#         compile_errors = []
#         missing_predict = []
#         for name in zip_names:
#             try:
#                 source_text = zip_object.read(name).decode("utf-8", errors="replace")
#             except Exception as error:
#                 compile_errors.append((name, f"read error: {error!r}"))
#                 continue

#             try:
#                 predict_function = compile_and_get_predict_function(source_text, name)
#             except Exception as error:
#                 compile_errors.append((name, f"compile/exec error: {error!r}"))
#                 continue

#             if not callable(predict_function):
#                 missing_predict.append(name)

#         if compile_errors:
#             print(f"❌ Compile/exec errors in {len(compile_errors)} files. Example:", compile_errors[:3])
#         else:
#             print("✅ All files read + compile")

#         if missing_predict:
#             print(f"❌ {len(missing_predict)} files missing a callable p(g). Example:", missing_predict[:5])
#         else:
#             print("✅ All files define p(g)")

#         # --- Functional spot-check (first N tasks) ---
#         if sample_limit and zip_names:
#             checked = 0
#             passed  = 0
#             failed_examples = []

#             for name in sorted(zip_names)[:sample_limit]:
#                 task_id_str = Path(name).stem.replace("task", "")  # '001'
#                 task_json_path = dataset_dir_path / f"task{int(task_id_str):03d}.json"
#                 if not task_json_path.exists():
#                     failed_examples.append((name, "missing JSON"))
#                     checked += 1
#                     continue

#                 source_text = zip_object.read(name).decode("utf-8", errors="replace")
#                 try:
#                     task_data = load_task_json(task_json_path)
#                     ok = validate_solver_text_on_task(source_text, task_data)
#                 except Exception as error:
#                     ok = False
#                     failed_examples.append((name, f"runtime error: {error!r}"))

#                 checked += 1
#                 if ok:
#                     passed += 1
#                 else:
#                     if len(failed_examples) < 5 and (name, "mismatch") not in failed_examples:
#                         failed_examples.append((name, "mismatch"))

#             print(f"Functional check: {passed}/{checked} passed (sample of {sample_limit})")
#             if failed_examples:
#                 print("Examples of failures:", failed_examples[:5])

# # ---- Run the diagnostics ----
# diagnose_submission_zip(
#     zip_path=SUBMISSION_ZIP_PATH,
#     dataset_dir_path=DATASET_DIR_PATH,
#     sample_limit=10,   # change to 0 (skip), 25 (faster), 400 (full)
# )


