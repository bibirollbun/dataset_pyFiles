%%time
import os
import json

# Path where tasks are stored
TASK_DIR = "/kaggle/input/google-code-golf-2025/"
# âœ… Correct
submission_dir = "/kaggle/working/submission"   
# âœ… Correct
zip_path = "/kaggle/working/submission.zip"   

# List all JSON files in the directory
all_files = [f for f in os.listdir(TASK_DIR) if f.startswith("task") and f.endswith(".json")]

# Extract task IDs
task_ids = sorted([f.replace(".json", "") for f in all_files])
print(f"âœ… Total number of tasks: {len(task_ids)}")
print(f"ğŸ“� Task IDs example: {task_ids[:5]} (up to {task_ids[-1]})")   


%%time
import json
import numpy as np

task_num = 31  # â†� Change this to solve next task
task_id = f"{task_num:03d}"
task_file = f"/kaggle/input/google-code-golf-2025/task{task_id}.json"

# Load task
task = json.load(open(task_file))

# Display helper
def show(inp, out, title=""):
    print(title)
    print("Input:")
    print(np.array(inp))
    print("Output:")
    print(np.array(out))
    print("-" * 30)   


%%time
for i, pair in enumerate(task["train"]):
    show(pair["input"], pair["output"], f"Example {i}")


%%time
def detect_color_mapping(examples):
    mapping = {}
    for pair in examples:
        inp, out = pair["input"], pair["output"]
        if len(inp) != len(out) or len(inp[0]) != len(out[0]): return None
        for ir, orow in zip(inp, out):
            for i, o in zip(ir, orow):
                if i in mapping and mapping[i] != o: return None
                mapping[i] = o
    return mapping
    
def detect_constant_output(examples):
    shapes, const_vals = [], []
    for pair in examples:
        out = pair["output"]
        h, w = len(out), len(out[0])
        shapes.append((h, w))
        vals = {v for row in out for v in row}
        if len(vals) != 1: return None
        const_vals.append(list(vals)[0])
    if len(set(shapes)) == 1 and len(set(const_vals)) == 1:
        h, w = shapes[0]; c = const_vals[0]
        return h, w, c
    return None
    
# ----- Existing detection functions (identity, constant, color mapping, geometry) -----
def detect_identity(examples):
    return all(pair["input"] == pair["output"] for pair in examples)
    
# ----- Generate solution -----
def generate_solution(task_id, examples):
    if detect_identity(examples):
        return "def p(g):return g"
    const_params = detect_constant_output(examples)
    if const_params:
        h,w,c=const_params
        return f"def p(g):return [[{c}]*{w}for _ in range({h})]"
    mapping = detect_color_mapping(examples)
    if mapping:
        dict_items = ",".join(f"{k}:{v}" for k,v in mapping.items())
        return f"def p(g):m={{{dict_items}}};return[[m[x]for x in r]for r in g]"
    geom_code = detect_geometry(examples)
    if geom_code: return geom_code
    crop_code = detect_cropping(examples)
    if crop_code: return crop_code
    pad_code = detect_padding(examples)
    if pad_code: return pad_code
    return "def p(g):\n    return g"


%%time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ARC_COLORS = {
    0: "#000000", 1: "#0000FF", 2: "#00FF00", 3: "#00FFFF",
    4: "#FF0000", 5: "#FF00FF", 6: "#FFFF00", 7: "#FFFFFF",
    8: "#808080", 9: "#FFA500"
}

def draw_fancy_tile(ax, row, col, color, highlight=None, cell_size=1.0):
    # Shadow
    ax.add_patch(patches.Rectangle((col+0.05, row-0.05), cell_size, cell_size,
                                   facecolor="black", alpha=0.3, zorder=0))
    # Main square
    ax.add_patch(patches.Rectangle((col, row), cell_size, cell_size,
                                   facecolor=color, edgecolor="gray", linewidth=0.5))
    # Inner highlight (for 3D look)
    inset = cell_size * 0.2
    ax.add_patch(patches.Rectangle((col+inset, row+inset),
                                   cell_size-2*inset, cell_size-2*inset,
                                   facecolor="white", alpha=0.2))
    # Difference highlight border
    if highlight == "added":
        ax.add_patch(patches.Rectangle((col, row), cell_size, cell_size,
                                       facecolor="none", edgecolor="lime", linewidth=2))
    elif highlight == "removed":
        ax.add_patch(patches.Rectangle((col, row), cell_size, cell_size,
                                       facecolor="none", edgecolor="red", linewidth=2))

def visualize_difference(input_grid, output_grid, title="Diff View"):
    H, W = len(output_grid), len(output_grid[0])
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    fig.suptitle(title, fontsize=14)

    for idx, (grid, name) in enumerate(zip([input_grid, output_grid], ["Input", "Output"])):
        axs[idx].set_xlim(0, W)
        axs[idx].set_ylim(0, H)
        axs[idx].set_aspect('equal')
        axs[idx].axis('off')
        axs[idx].set_title(name)

        for r in range(H):
            for c in range(W):
                color = ARC_COLORS.get(grid[H - r - 1][c], "#000000")
                highlight = None
                if input_grid[H - r - 1][c] != output_grid[H - r - 1][c]:
                    if name == "Output":
                        highlight = "added"
                    elif name == "Input":
                        highlight = "removed"
                draw_fancy_tile(axs[idx], r, c, color, highlight)
    plt.tight_layout()
    plt.show()

# Example usage (with dummy sample, replace with real data)
input_example = [
    [5,5,5,5,5],
    [5,1,1,1,5],
    [5,1,1,1,5],
    [5,5,5,5,5]
]
output_example = [
    [5,5,5,5,5],
    [5,1,1,1,0],
    [5,1,1,1,0],
    [5,0,0,0,0]
]

visualize_difference(input_example, output_example, "Hypothetical ARC Diff")


%%time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ARC_COLORS = {
    0: "#000000", 1: "#0000FF", 2: "#00FF00", 3: "#00FFFF",
    4: "#FF0000", 5: "#FF00FF", 6: "#FFFF00", 7: "#FFFFFF",
    8: "#808080", 9: "#FFA500"
}

def draw_fancy_tile(ax, row, col, color, cell_size=1.0):
    # Draw shadow (slightly offset)
    shadow = patches.Rectangle((col+0.05, row-0.05), cell_size, cell_size,
                               facecolor="black", alpha=0.3, zorder=0)
    ax.add_patch(shadow)
    # Draw main square
    outer = patches.Rectangle((col, row), cell_size, cell_size,
                              facecolor=color, edgecolor="gray", linewidth=0.5)
    ax.add_patch(outer)
    # Draw inner square for highlight (inset by 20%)
    inset = cell_size * 0.2
    inner = patches.Rectangle((col+inset, row+inset),
                              cell_size-2*inset, cell_size-2*inset,
                              facecolor="white", alpha=0.2)
    ax.add_patch(inner)

def draw_grid_3d(ax, grid, title):
    H, W = len(grid), len(grid[0])
    ax.set_title(title)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.axis('off')

    for r in range(H):
        for c in range(W):
            color = ARC_COLORS.get(grid[H - r - 1][c], "#000000")
            draw_fancy_tile(ax, r, c, color)

def visualize_task_3d(task_id, examples, rule):
    fig, axs = plt.subplots(len(examples), 2, figsize=(5, 2.5*len(examples)))
    fig.suptitle(f"Task {task_id:03} - Rule: {rule}", fontsize=14)
    for i, ex in enumerate(examples):
        draw_grid_3d(axs[i, 0], ex["input"], "Input")
        draw_grid_3d(axs[i, 1], ex["output"], "Expected Output")
    plt.tight_layout()
    plt.show()

# Example usage
task_id = 16
task_data = json.load(open(os.path.join(TASK_DIR, "task016.json")))
examples = task_data["train"][:2]
code = generate_solution(task_id, examples)
visualize_task_3d(task_id, examples, "Auto-detected rule")


%%time
def detect_identity(inp, out):
    """Check if output is identical to input"""
    return [("identity",)] if inp == out else None


%%time
def detect_color_map(inp, out):
    # Ensure non-empty
    if not inp or not out or not inp[0] or not out[0]:
        return None
    
    h, w = len(inp), len(inp[0])
    if len(out) != h or len(out[0]) != w:
        return None
    
    mapping = {}
    for r in range(h):
        for c in range(w):
            mapping[inp[r][c]] = out[r][c]
    return ('color_map', mapping)


%%time
def detect_geometry(inp, out):
    import numpy as np
    arr_in = np.array(inp)
    arr_out = np.array(out)
    
    # Rotations
    for k in range(4):
        if np.array_equal(np.rot90(arr_in, k), arr_out):
            return [("rot90", k)]
    # Flip horizontal
    if np.array_equal(np.fliplr(arr_in), arr_out):
        return [("flip_h",)]
    # Flip vertical
    if np.array_equal(np.flipud(arr_in), arr_out):
        return [("flip_v",)]
    return None


%%time
def detect_crop(inp, out):
    H_in, W_in = len(inp), len(inp[0])
    H_out, W_out = len(out), len(out[0])
    for r0 in range(H_in - H_out + 1):
        for c0 in range(W_in - W_out + 1):
            match = True
            for r in range(H_out):
                for c in range(W_out):
                    if inp[r0 + r][c0 + c] != out[r][c]:
                        match = False
                        break
                if not match:
                    break
            if match:
                return [("crop", r0, c0, H_out, W_out)]
    return None


%%time
def detect_pad(inp, out):
    H_in, W_in = len(inp), len(inp[0])
    H_out, W_out = len(out), len(out[0])
    # Only handle simple top-left padding
    for r_off in range(H_out - H_in + 1):
        for c_off in range(W_out - W_in + 1):
            ok = True
            for r in range(H_out):
                for c in range(W_out):
                    expected = 0
                    if r_off <= r < r_off + H_in and c_off <= c < c_off + W_in:
                        expected = inp[r - r_off][c - c_off]
                    if out[r][c] != expected:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return [("pad", r_off, c_off, H_out, W_out)]
    return None


%%time
def detect_repeat_pattern(inp, out):
    H_in, W_in = len(inp), len(inp[0])
    H_out, W_out = len(out), len(out[0])
    if H_out % H_in == 0 and W_out % W_in == 0:
        rep_h = H_out // H_in
        rep_w = W_out // W_in
        tiled = [[inp[r % H_in][c % W_in] for c in range(W_out)] for r in range(H_out)]
        if tiled == out:
            return [("repeat", rep_h, rep_w)]
    return None


%%time
def detect_horizontal_repeat_with_mask(inp, out):
    H_in, W_in = len(inp), len(inp[0])
    H_out, W_out = len(out), len(out[0])
    if W_out % W_in == 0 and H_out >= H_in:
        rep_w = W_out // W_in
        # Build repeated strip (only first H_in rows)
        strip = [[inp[r][c % W_in] for c in range(W_out)] for r in range(H_in)]
        # Canvas = zeros, then put strip at top
        canvas = [[0 for _ in range(W_out)] for _ in range(H_out)]
        for r in range(H_in):
            for c in range(W_out):
                canvas[r][c] = strip[r][c]
        if canvas == out:
            return [("horiz_repeat_top", rep_w)]
    return None


%%time
def detect_constant_output(inp, out):
    # If all values in output are identical
    flat = [v for row in out for v in row]
    unique_vals = set(flat)
    if len(unique_vals) == 1:
        return ('constant', list(unique_vals)[0])
    return None


%%time
def detect_mirror(inp, out):
    H, W = len(inp), len(inp[0])
    # Check horizontal mirror
    horiz = [row[::-1] for row in inp]
    if horiz == out:
        return ('mirror_h',)
    # Check vertical mirror
    vert = inp[::-1]
    if vert == out:
        return ('mirror_v',)
    return None


%%time
def rotate90(g):
    return [list(row) for row in zip(*g[::-1])]

def rotate180(g):
    return [row[::-1] for row in g[::-1]]

def rotate270(g):
    return [list(row) for row in zip(*g)][::-1]

def detect_rotation(inp, out):
    if rotate90(inp) == out:
        return ('rot90',)
    if rotate180(inp) == out:
        return ('rot180',)
    if rotate270(inp) == out:
        return ('rot270',)
    return None


%%time
def search_on_task(examples):
    detectors = [
        detect_constant_output,
        detect_rotation,
        detect_mirror,
        detect_horizontal_repeat_with_mask,
        detect_repeat_pattern,
        detect_crop,
        detect_pad,
        detect_geometry,
        detect_color_map,
        detect_identity,
    ]
    
    for detector in detectors:
        all_match = True
        first_prog = None
        for ex in examples:
            inp, out = ex["input"], ex["output"]
            prog = detector(inp, out)
            if not prog:
                all_match = False
                break
            if first_prog is None:
                first_prog = prog
        if all_match and first_prog:
            return first_prog
    return None


%%time
def generate_code(prog):
    # Convert abstract program to Python code
    if not prog:
        return "def p(g):\n    return g"

    if prog[0][0] == "identity":
        return "def p(g):\n    return g"
    elif prog[0][0] == "color_map":
        mapping = prog[0][1]
        return f"def p(g):\n    m={mapping}\n    return [[m[x] for x in row] for row in g]"
    elif prog[0][0] == "rot90":
        k = prog[0][1]
        return f"""def p(g):
    import numpy as np
    return np.rot90(np.array(g), {k}).tolist()
"""
    elif prog[0][0] == "flip_h":
        return "def p(g):\n    return [row[::-1] for row in g]"
    elif prog[0][0] == "flip_v":
        return "def p(g):\n    return g[::-1]"
    elif prog[0][0] == "crop":
        r0, c0, h, w = prog[0][1:]
        return f"def p(g):\n    return [row[{c0}:{c0+w}] for row in g[{r0}:{r0+h}]]"
    elif prog[0][0] == "pad":
        r_off, c_off, h, w = prog[0][1:]
        return f"""def p(g):
    H,W=len(g),len(g[0])
    canvas=[[0]*{w} for _ in range({h})]
    for r in range(H):
        for c in range(W):
            canvas[r+{r_off}][c+{c_off}]=g[r][c]
    return canvas
"""
    elif prog[0][0] == "repeat":
        rep_h, rep_w = prog[0][1:]
        return f"""def p(g):
    H,W=len(g),len(g[0])
    return [[g[r%H][c%W] for c in range(W*{rep_w})] for r in range(H*{rep_h})]
"""
    elif prog[0][0] == "horiz_repeat_top":
        rep_w = prog[0][1]
        return f"""def p(g):
    H,W=len(g),len(g[0])
    Ho,Wo={rep_w}*H,{rep_w}*W
    canvas=[[0]*Wo for _ in range(Ho)]
    for r in range(H):
        for c in range(Wo):
            canvas[r][c]=g[r][c%W]
    return canvas
"""
    return "def p(g):\n    return g"  


%%time
import os, json, zipfile

TASK_DIR = "/kaggle/input/google-code-golf-2025/"
submission_dir = "/kaggle/working/submission"
zip_path = "/kaggle/working/submission.zip"

os.makedirs(submission_dir, exist_ok=True)

solved, failed = [], []

for i in range(1, 401):  # <-- only first 5 tasks
    task_path = os.path.join(TASK_DIR, f"task{i:03}.json")
    with open(task_path) as f:
        task = json.load(f)

    prog = search_on_task(task["train"])

    if prog is None:
        code = "def p(g):\n    return g"
        failed.append(i)
    else:
        code = generate_code(prog)
        solved.append(i)

    file_path = os.path.join(submission_dir, f"task{i:03}.py")
    with open(file_path, "w") as f:
        f.write(code)

# Zip submission
with zipfile.ZipFile(zip_path, 'w') as zf:
    for fname in os.listdir(submission_dir):
        zf.write(os.path.join(submission_dir, fname), arcname=fname)

print(f"Submission file created: {zip_path}")
print(f"Tasks solved (non-identity): {len(solved)}")
print(f"Solved task IDs: {solved}")
print(f"Tasks still failing: {len(failed)}")


%%time
# --- Visualization with Predicted Output ---
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Map colors for better visualization
COLOR_MAP = {
    0: 'white', 1: 'black', 2: 'blue', 3: 'green', 4: 'red',
    5: 'brown', 6: 'purple', 7: 'orange', 8: 'yellow', 9: 'pink'
}

def plot_grid(grid, title):
    arr = np.array(grid)
    plt.imshow(arr, cmap=plt.matplotlib.colors.ListedColormap(list(COLOR_MAP.values())), vmin=0, vmax=9)
    plt.title(title)
    plt.axis('off')

# Dummy solver for visualization (replace with actual solver logic)
def run_solver(input_grid):
    # For demo, simply return the same grid (identity)
    return input_grid

# Load one example task
TASK_DIR = Path("/kaggle/input/google-code-golf-2025/")
sample_task = list(TASK_DIR.glob('task*.json'))[0]
with open(sample_task) as f:
    data = json.load(f)
example_inp = data['train'][0]['input']
example_out = data['train'][0]['output']

# Get predicted output using solver
predicted_out = run_solver(example_inp)

# Display input, expected output, predicted output
plt.figure(figsize=(9,3))
plt.subplot(1,3,1)
plot_grid(example_inp, "Input")
plt.subplot(1,3,2)
plot_grid(example_out, "Expected")
plt.subplot(1,3,3)
plot_grid(predicted_out, "Predicted")
plt.show()

