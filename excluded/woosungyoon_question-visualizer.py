%%writefile question_visualizer.py

import json
import os
import importlib.util
import sys
import copy
import re
import warnings
import numpy as np
import random
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from PIL import Image, ImageDraw
from io import BytesIO
from typing import List, Dict, Tuple
import argparse

warnings.filterwarnings("ignore", category=SyntaxWarning)


# =============================================================================
# Constants and Utilities
# =============================================================================

NUM_TO_COLOR = {
    0: (0, 0, 0),       # black
    1: (0, 0, 255),     # blue
    2: (255, 0, 0),     # red
    3: (0, 255, 0),     # green
    4: (255, 255, 0),   # yellow
    5: (255, 255, 255), # white
    6: (128, 0, 128),   # purple
    7: (255, 165, 0),   # orange
    8: (0, 255, 255),   # cyan
    9: (165, 42, 42)    # brown
}

def grid_to_pil_image(grid: List[List[int]], cell_size: int = 20) -> Image.Image:
    """Convert grid to PIL Image."""
    if not grid:
        return Image.new('RGB', (1, 1), color='white')
    height, width = len(grid), len(grid[0])
    img = Image.new('RGB', (width * cell_size, height * cell_size), color='white')
    draw = ImageDraw.Draw(img)
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            color = NUM_TO_COLOR.get(cell, (128, 128, 128))  # gray for unknown
            draw.rectangle(
                [x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size],
                fill=color,
                outline=(200, 200, 200)  # light grid lines
            )
    return img

def image_to_bytes(img: Image.Image) -> bytes:
    """Convert PIL Image to PNG bytes."""
    bio = BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


# =============================================================================
# Data Loading and Display
# =============================================================================

def load_task(task_id: int, task_dir: str) -> Dict:
    """Load task data from JSON."""
    path = os.path.join(task_dir, f"task{task_id:03d}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Task file not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)

def build_problem_display(task_id: int, task_dir: str) -> widgets.VBox:
    """Build widget for problem examples."""
    data = load_task(task_id, task_dir)
    train_examples = data.get("train", [])[:3]
    example_widgets = []
    for i, ex in enumerate(train_examples):
        input_img = grid_to_pil_image(ex["input"])
        output_img = grid_to_pil_image(ex["output"])
        input_widget = widgets.Image(value=image_to_bytes(input_img), format='png')
        output_widget = widgets.Image(value=image_to_bytes(output_img), format='png')
        hbox = widgets.HBox([
            widgets.VBox([widgets.Label("Input:"), input_widget]),
            widgets.VBox([widgets.Label("Output:"), output_widget])
        ])
        example_widgets.append(widgets.VBox([widgets.Label(f"Example {i+1}:"), hbox]))
    return widgets.VBox(example_widgets)


# =============================================================================
# Code Loading and Evaluation
# =============================================================================

def load_user_function(code_path: str):
    """Load user function p from code file."""
    if not os.path.exists(code_path):
        raise FileNotFoundError(f"Code file not found: {code_path}")
    spec = importlib.util.spec_from_file_location("user_module", code_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_module"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, 'p'):
        raise AttributeError("Function 'p' not found in the code.")
    return module.p

def evaluate_code(task_id: int, code_path: str, task_dir: str) -> Tuple[int, int, List[Dict], int]:
    """Evaluate code on all examples."""
    data = load_task(task_id, task_dir)
    all_examples = data.get("train", []) + data.get("test", []) + data.get("arc-gen", [])
    total_examples = len(all_examples)
    correct_count = 0
    failed_examples = []
    code_size = os.path.getsize(code_path)

    try:
        p_func = load_user_function(code_path)
    except Exception as e:
        return 0, total_examples, [{"error": f"Load error: {str(e)}"}], code_size
    
    for ex in all_examples:
        try:
            input_copy = copy.deepcopy(ex["input"])
            user_output = p_func(input_copy)
            out_str = json.dumps(user_output).replace("true", "1").replace("false", "0")
            if re.search(r"[^0-9,\[\]\s\.]", out_str):
                raise ValueError("Invalid output format: contains non-numeric characters.")
            user_grid = np.array(json.loads(out_str))
            expected_grid = np.array(ex["output"])
            if np.array_equal(user_grid, expected_grid):
                correct_count += 1
            else:
                failed_examples.append({
                    "input": ex["input"],
                    "expected": ex["output"],
                    "user": user_output
                })
        except Exception as e:
            failed_examples.append({
                "input": ex["input"],
                "expected": ex["output"],
                "error": str(e)
            })
    
    return correct_count, total_examples, failed_examples, code_size

def display_failed_samples(failed: List[Dict], max_samples: int = 3) -> widgets.VBox:
    """Build widget for failed examples."""
    if not failed:
        return widgets.VBox([])
    
    samples = random.sample(failed, min(max_samples, len(failed)))
    sample_widgets = [widgets.Label(f"Sample of Failed Examples (Showing {len(samples)} out of {len(failed)})")]
    for i, fail in enumerate(samples):
        input_img = grid_to_pil_image(fail["input"])
        expected_img = grid_to_pil_image(fail["expected"])
        input_widget = widgets.Image(value=image_to_bytes(input_img), format='png')
        expected_widget = widgets.Image(value=image_to_bytes(expected_img), format='png')
        
        vbox_children = [
            widgets.Label(f"Failed Example {i+1}:"),
        ]
        
        hbox_children = [
            widgets.VBox([widgets.Label("Input:"), input_widget]),
            widgets.VBox([widgets.Label("Expected Output:"), expected_widget])
        ]
        
        if "user" in fail:
            user_img = grid_to_pil_image(fail["user"])
            user_widget = widgets.Image(value=image_to_bytes(user_img), format='png')
            hbox_children.append(widgets.VBox([widgets.Label("Your Output:"), user_widget]))
        
        hbox = widgets.HBox(hbox_children)
        vbox_children.append(hbox)
        
        if "error" in fail:
            vbox_children.append(widgets.Label(f"Error: {fail['error']}"))
        
        vbox = widgets.VBox(vbox_children)
        sample_widgets.append(vbox)
    
    return widgets.VBox(sample_widgets)


# =============================================================================
# Widgets and Handlers
# =============================================================================

def create_app(task_dir: str, save_dir: str):
    """Create and return the app layout."""
    problem_input = widgets.BoundedIntText(
        value=1, min=1, max=400, step=1,
        description='Problem ID:', layout=widgets.Layout(width='200px')
    )
    go_button = widgets.Button(description='Go', layout=widgets.Layout(width='60px'))
    problem_display = widgets.Output()
    code_input = widgets.Textarea(
        value='', placeholder='def p(grid):\n    # Your code here\n    return grid',
        description='Your Code:', layout={'height': '300px', 'width': '700px'}
    )
    versions_dropdown = widgets.Dropdown(
        options=[],
        description='Load Version:',
        layout=widgets.Layout(width='300px')
    )
    load_btn = widgets.Button(description='Load Code')
    check_size_btn = widgets.Button(description='Check Size')
    submit_btn = widgets.Button(description='Submit Code')
    save_btn = widgets.Button(description='Save Code')
    results_display = widgets.Output()

    def update_versions(change):
        task_id = problem_input.value
        base_name = f"task{task_id:03d}"
        files = [f for f in os.listdir(save_dir) if f.startswith(base_name) and f.endswith('.py')]
        if files:
            versions_dropdown.options = sorted(files)
        else:
            versions_dropdown.options = ['No versions available']

    problem_input.observe(update_versions, names='value')

    def on_load_clicked(b):
        selected = versions_dropdown.value
        if selected and selected != 'No versions available':
            full_path = os.path.join(save_dir, selected)
            try:
                with open(full_path, 'r') as f:
                    code_input.value = f.read()
                with results_display:
                    clear_output()
                    display(widgets.Label(f"Loaded {selected}"))
            except Exception as e:
                with results_display:
                    clear_output()
                    display(widgets.Label(f"Error loading code: {str(e)}"))

    load_btn.on_click(on_load_clicked)

    def on_go_clicked(b):
        task_id = problem_input.value
        with problem_display:
            clear_output(wait=True)
            try:
                display_widget = build_problem_display(task_id, task_dir)
                display(display_widget)
            except Exception as e:
                display(widgets.Label(f"Error loading problem {task_id}: {str(e)}"))

    go_button.on_click(on_go_clicked)

    def on_value_change(change):
        if change['name'] == 'value':
            on_go_clicked(None)

    problem_input.observe(on_value_change, names='value')

    def on_check_size_clicked(b):
        code = code_input.value.strip()
        if not code:
            with results_display: clear_output(); display(widgets.Label("Error: No code provided."))
            return
        code_path = 'temp_user_code.py'
        with open(code_path, 'w') as f: f.write(code)
        try:
            size = os.path.getsize(code_path)
            with results_display: clear_output(); display(widgets.Label(f"Code Size: {size} bytes"))
        except Exception as e:
            with results_display: clear_output(); display(widgets.Label(f"Error checking size: {str(e)}"))
        finally:
            if os.path.exists(code_path): os.remove(code_path)

    check_size_btn.on_click(on_check_size_clicked)

    def on_submit_clicked(b):
        code = code_input.value.strip()
        if not code:
            with results_display: clear_output(); display(widgets.Label("Error: No code provided."))
            return
        code_path = 'temp_user_code.py'
        with open(code_path, 'w') as f: f.write(code)
        task_id = problem_input.value
        with results_display:
            clear_output()
            try:
                correct, total, failed, code_size = evaluate_code(task_id, code_path, task_dir)
                wrong = total - correct
                result_label = widgets.Label(f"Results: Correct {correct}/{total} | Wrong: {wrong} | Code Size: {code_size} bytes")
                display(result_label)
                if failed:
                    failed_widget = display_failed_samples(failed, max_samples=3)
                    display(failed_widget)
                else:
                    display(widgets.Label("All examples passed!"))
            except Exception as e:
                display(widgets.Label(f"Evaluation error: {str(e)}"))
            finally:
                if os.path.exists(code_path): os.remove(code_path)

    submit_btn.on_click(on_submit_clicked)

    def on_save_clicked(b):
        code = code_input.value.strip()
        if not code:
            with results_display: clear_output(); display(widgets.Label("Error: No code to save."))
            return
        task_id = problem_input.value
        base_name = f"task{task_id:03d}"
        file_name = f"{base_name}.py"
        version = 1
        full_path = os.path.join(save_dir, file_name)
        while os.path.exists(full_path):
            file_name = f"{base_name}_v{version}.py"
            full_path = os.path.join(save_dir, file_name)
            version += 1
        try:
            with open(full_path, 'w') as f: f.write(code)
            with results_display: clear_output(); display(widgets.Label(f"Code saved to {full_path}."))
            update_versions(None)  # Refresh dropdown after save
        except Exception as e:
            with results_display: clear_output(); display(widgets.Label(f"Error saving code: {str(e)}"))

    save_btn.on_click(on_save_clicked)

    # Initial load
    on_go_clicked(None)
    update_versions(None)

    # Layout
    top_controls = widgets.HBox([problem_input, go_button])
    load_row = widgets.HBox([versions_dropdown, load_btn])
    button_row = widgets.HBox([check_size_btn, submit_btn, save_btn])
    app_layout = widgets.VBox([
        widgets.HTML("<h2>ARC Problem Solver App</h2>"),
        problem_display,
        top_controls,
        code_input,
        load_row,
        button_row,
        results_display
    ])

    return app_layout

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARC Problem Solver App")
    parser.add_argument("--task_dir", type=str, default="/kaggle/input/google-code-golf-2025/", help="Directory for task JSON files")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save code files")
    args, _ = parser.parse_known_args()  

    app = create_app(args.task_dir, args.save_dir)
    display(app)


import os, shutil
from glob import glob

save_dir = '/kaggle/working/submission'
os.makedirs(save_dir, exist_ok=True)

for src in glob('/kaggle/input/code-golf-sample/task*.py'):
    shutil.copy2(src, save_dir)


wrapper_code = f"""
import sys
import os
sys.argv = ['question_visualizer.py', '--task_dir', '/kaggle/input/google-code-golf-2025/', '--save_dir', '{save_dir}']
exec(open('question_visualizer.py').read())
"""

wrapper_path = '/kaggle/working/_temp_run_visualizer.py'
with open(wrapper_path, 'w', encoding='utf-8') as f:
    f.write(wrapper_code)


%run {wrapper_path}

