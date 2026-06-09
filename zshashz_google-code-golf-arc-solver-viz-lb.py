# ARC-AGI Code Golf Solver - Enhanced Version

# This notebook combines the best-performing solution approaches with visualization and additional code optimization techniques for the Google Code Golf 2025 competition.

import os
import json
import zipfile
import copy
import re
import math
import ast
import string
from collections import Counter, defaultdict
from functools import reduce
from typing import List, Tuple, Dict
import itertools

from tqdm import tqdm
from rich import print as print_rich
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Configuration
DIR = "/kaggle"
VISUALIZE_TASKS = True  
NUM_TASKS_TO_VISUALIZE = 5

# Color palette for ARC visualization
cmap = colors.ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                              '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])




def check_solution(solution, task_data):
    """Verify a solution against all examples in the task data."""
    try:
        namespace = {}
        exec(solution, namespace)
        if 'p' not in namespace:
            return False
        all_examples = task_data['train'] + task_data['test'] + task_data.get('arc-gen', [])
        for example in all_examples:
            input_grid = copy.deepcopy(example['input'])
            expected = example['output']
            try:
                actual = namespace['p'](input_grid)
                if actual != expected:
                    return False
            except Exception:
                return False
        return True
    except Exception:
        return False


def get_bytes(s):
    """Get the byte count of a string."""
    return len(s.encode('utf-8'))


def calculate_score(s):
    """Calculate the score for a solution based on byte count."""
    return max(1, 2500 - get_bytes(s))

def visualize_task(task_data, solution=None, task_id=None):
    """Visualize task examples and solution results with enhanced display."""
    num_examples = min(len(task_data['train']), 3)
    has_solution = solution is not None
    
    fig, axes = plt.subplots(num_examples, 3 if has_solution else 2,
                             figsize=(12 if has_solution else 8, 3 * num_examples))
    
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    if task_id:
        fig.suptitle(f"Task {task_id}", fontsize=16, fontweight='bold')

    for i in range(num_examples):
        ex = task_data['train'][i]
        
        # Input grid
        axes[i, 0].imshow(ex['input'], cmap=cmap, vmin=0, vmax=9)
        axes[i, 0].set_title(f'Train {i+1}: Input')
        axes[i, 0].grid(True, which='both', color='lightgrey', linewidth=0.5)
        axes[i, 0].set_xticks(range(len(ex['input'][0])))
        axes[i, 0].set_yticks(range(len(ex['input'])))

        # Expected output grid
        axes[i, 1].imshow(ex['output'], cmap=cmap, vmin=0, vmax=9)
        axes[i, 1].set_title(f'Train {i+1}: Expected Output')
        axes[i, 1].grid(True, which='both', color='lightgrey', linewidth=0.5)
        axes[i, 1].set_xticks(range(len(ex['output'][0])))
        axes[i, 1].set_yticks(range(len(ex['output'])))
        
        # Predicted output grid
        if has_solution:
            try:
                namespace = {}
                exec(solution, namespace)
                p = namespace['p']
                actual = p([row[:] for row in ex['input']])
                axes[i, 2].imshow(actual, cmap=cmap, vmin=0, vmax=9)
                match = actual == ex['output']
                axes[i, 2].set_title(f'Train {i+1}: Predicted {"âœ“" if match else "âœ—"}')
                axes[i, 2].grid(True, which='both', color='lightgrey', linewidth=0.5)
            except Exception as e:
                axes[i, 2].text(0.5, 0.5, f'Error:\n{str(e)[:50]}', 
                               ha='center', va='center', wrap=True)
                axes[i, 2].set_title(f'Train {i+1}: Error')

    plt.tight_layout()
    plt.show()

def remove_spaces(s):
    """Remove unnecessary spaces around operators and symbols."""
    # Extended list of symbols where spaces can be removed
    symbols = ['[', ']', '(', ')', '{', '}', '=', '!', '<', '>', '+', '-', '*', 
               '/', '%', ';', ',', ':', '|', '&', '^', '~']
    for c in symbols:
        s = s.replace(' ' + c, c)
        s = s.replace(c + ' ', c)
    # Also remove spaces around dots for method calls
    s = re.sub(r'\s*\.\s*', '.', s)
    return s


def minimize_indentation(s):
    """Minimize indentation to single spaces instead of standard 4."""
    leading_spaces = [
        len(m.group(1))
        for m in re.finditer(r'^( +)(?=\S)', s, flags=re.MULTILINE)
    ]
    if not leading_spaces:
        return s

    unit = reduce(math.gcd, leading_spaces)
    if unit <= 1:
        return s

    def _shrink(match: re.Match):
        count = len(match.group(1))
        return ' ' * (count // unit)

    return re.sub(r'^( +)', _shrink, s, flags=re.MULTILINE)


def strip_trailing_whitespaces(s):
    """Remove trailing whitespaces and empty lines at the end."""
    return s.strip()


def remove_empty_lines(s):
    """Remove empty lines from code."""
    lines = s.split('\n')
    return '\n'.join([line for line in lines if line.strip()])


def remove_comments(s):
    """Remove single line comments from code."""
    lines = s.split('\n')
    return '\n'.join([line for line in lines if not line.strip().startswith('#')])

def find_local_variables(s: str):
    """Find all variable names inside function using AST."""
    try:
        tree = ast.parse(s)
        local_vars = set()

        class LocalVarVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef):
                for arg in node.args.args:
                    local_vars.add(arg.arg)
                self.generic_visit(node)

            def visit_Name(self, node: ast.Name):
                if isinstance(node.ctx, ast.Store):
                    local_vars.add(node.id)
                self.generic_visit(node)

            def visit_For(self, node: ast.For):
                if isinstance(node.target, ast.Name):
                    local_vars.add(node.target.id)
                self.generic_visit(node)

            def visit_comprehension(self, node: ast.comprehension):
                if isinstance(node.target, ast.Name):
                    local_vars.add(node.target.id)
                self.generic_visit(node)

        LocalVarVisitor().visit(tree)
        return local_vars
    except:
        return set()


def find_single_letter_variables(s: str):
    """Find single letter variable names that are currently used."""
    return {var for var in find_local_variables(s) if len(var) == 1}


def find_available_single_letter_variables(s: str):
    """Find single letter variable names that can be used to replace longer names."""
    used = find_single_letter_variables(s)
    possible = set(string.ascii_lowercase + string.ascii_uppercase + '_')
    return possible - used


def _replace_variable_name(s, old_name, new_name):
    """Replace variable name using regex to avoid partial matches."""
    pattern = rf'\b{re.escape(old_name)}\b'
    return re.sub(pattern, new_name, s)


def shorten_variable_names(s):
    """Shorten all multi-character variable names to single characters."""
    long_variable_names = [x for x in find_local_variables(s) if len(x) > 1]
    available = find_available_single_letter_variables(s)
    
    for var in long_variable_names:
        if available:
            new_name = available.pop()
            s = _replace_variable_name(s, var, new_name)
    
    return s

def substitute_range(s):
    """Replace multiple range() calls with a single-letter function."""
    available = find_available_single_letter_variables(s)
    if s.count('range(') >= 3 and available:
        v = available.pop()
        s = s.replace('range', v)
        # Add parameter to function definition
        index = s.find(')')
        s = s[:index] + ',' + v + '=range' + s[index:]
    return s


def substitute_enumerate(s):
    """Replace multiple enumerate() calls with a single-letter function."""
    available = find_available_single_letter_variables(s)
    if s.count('enumerate(') >= 2 and available:
        v = available.pop()
        s = s.replace('enumerate', v)
        index = s.find(')')
        s = s[:index] + ',' + v + '=enumerate' + s[index:]
    return s


def substitute_common_functions(s):
    """Replace common function calls with shorter alternatives."""
    substitutions = [
        ('len(', 3),
        ('sum(', 3),
        ('min(', 3),
        ('max(', 3),
        ('all(', 3),
        ('any(', 3),
        ('zip(', 3),
        ('list(', 3),
        ('set(', 3),
    ]
    
    available = find_available_single_letter_variables(s)
    used_substitutions = []
    
    for func, min_count in substitutions:
        if s.count(func) >= min_count and available:
            v = available.pop()
            func_name = func[:-1]  # Remove the '('
            s = s.replace(func_name, v)
            used_substitutions.append(f'{v}={func_name}')
    
    if used_substitutions:
        # Add all substitutions to function parameters
        index = s.find(')')
        s = s[:index] + ',' + ','.join(used_substitutions) + s[index:]
    
    return s

def join_block_lines(code: str) -> str:
    """Join lines in blocks when possible using AST analysis."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    # Remove docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
                node.body.pop(0)

    lines = code.splitlines(keepends=True)
    transforms: List[Tuple[int, int, int]] = []

    SIMPLE = (
        ast.Assign, ast.AugAssign, ast.Expr, ast.Return,
        ast.Delete, ast.Pass, ast.Continue, ast.Break,
        ast.Assert, ast.Raise, ast.Global, ast.Nonlocal,
        ast.Import, ast.ImportFrom
    )

    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While, ast.With, ast.If, ast.FunctionDef)):
            continue
        if isinstance(node, ast.If) and node.orelse:
            continue
        
        body = getattr(node, 'body', None)
        if not body:
            continue

        if all(isinstance(stmt, SIMPLE) and stmt.lineno == stmt.end_lineno for stmt in body):
            transforms.append((node.lineno, body[0].lineno, body[-1].end_lineno))

    transforms.sort(key=lambda t: t[0], reverse=True)
    
    for hdr, start, end in transforms:
        header_index = hdr - 1
        start_index = start - 1
        end_index = end - 1
        
        if not (0 <= header_index < len(lines) and start_index <= end_index):
            continue
            
        header = lines[header_index].rstrip('\n')
        if not header.strip().endswith(':'):
            continue

        parts = [lines[i].strip() for i in range(start_index, end_index+1)]
        joined = ';'.join(parts)
        lines[header_index] = f"{header}{joined}\n"
        del lines[start_index:end_index+1]

    return ''.join(lines)


def def_to_lambda(s):
    """Convert simple def functions to lambda where possible."""
    # Single line return functions
    _def = r'^def\s+p\(([A-Za-z,=\s]+)\):return\s+(.+)$'
    match = re.match(_def, s.strip())
    if match and '\n' not in s:
        params, body = match.groups()
        return f'p=lambda {params}:{body}'
    return s

def optimize_list_comprehensions(s):
    """Optimize list comprehensions for shorter syntax."""
    # Replace [[...] for _ in range(n)] with [[...]*n]
    pattern = r'\[\[([^\]]+)\]\s*for\s+\w+\s+in\s+range\((\w+)\)\]'
    s = re.sub(pattern, r'[[\1]*\2]', s)
    
    # Replace [x for x in ...] with list(...)
    pattern = r'\[(\w+)\s+for\s+\1\s+in\s+([^\]]+)\]'
    s = re.sub(pattern, r'list(\2)', s)
    
    return s


def optimize_conditionals(s):
    """Optimize conditional expressions."""
    # Replace if/else with ternary operator
    pattern = r'if\s+([^:]+):\s*(\w+)\s*=\s*([^;]+);\s*else:\s*\2\s*=\s*([^;]+)'
    s = re.sub(pattern, r'\2=\3 if \1 else \4', s)
    
    # Replace x if x!=0 else 0 with just x
    pattern = r'(\w+)\s+if\s+\1\s*!=\s*0\s+else\s+0'
    s = re.sub(pattern, r'\1', s)
    
    return s


def optimize_loops(s):
    """Optimize loop structures."""
    # Replace for i in range(len(x)) with enumerate where appropriate
    pattern = r'for\s+(\w+)\s+in\s+range\(len\((\w+)\)\):([^:]+)\[\1\]'
    def replace_with_enumerate(match):
        idx, lst, body = match.groups()
        if lst in body:
            return f'for {idx},{lst[0]} in enumerate({lst}):{body.replace(f"{lst}[{idx}]", lst[0])}'
        return match.group(0)
    s = re.sub(pattern, replace_with_enumerate, s)
    
    return s


def remove_redundant_operations(s):
    """Remove redundant operations."""
    # Remove unnecessary list() calls
    s = re.sub(r'list\(\[([^\]]+)\]\)', r'[\1]', s)
    
    # Remove redundant parentheses
    s = re.sub(r'\((\w+)\)', r'\1', s)
    
    # Replace x+0 or 0+x with x
    s = re.sub(r'(\w+)\s*\+\s*0|0\s*\+\s*(\w+)', r'\1\2', s)
    
    return s


def load_solutions_from_sources():
    """Load pre-existing solutions from various sources."""
    submission_paths = [
        f"{DIR}/input/oh-barnacles",
        f"{DIR}/input/qwen2-5-32b-arc-local-score-32-solved-script/submission",
        f"{DIR}/input/solved-127-problems-local-pleaseupvote",
        f"{DIR}/input/neurips-2025-google-code-golf-championship1"
    ]
    
    solutions_by_task = defaultdict(list)
    
    for path in submission_paths:
        if os.path.exists(path):
            for task_file in os.listdir(path):
                if task_file.startswith('task') and task_file.endswith('.py'):
                    task_id = task_file[4:7]  # Extract task number
                    try:
                        with open(os.path.join(path, task_file), 'r') as f:
                            solution = f.read()
                            solutions_by_task[task_id].append(solution)
                    except:
                        continue
    
    return solutions_by_task


def create_ultimate_submission():
    """Create the ultimate optimized submission using improved optimization approach."""
    print_rich("[bold cyan]ğŸš€ Ultimate ARC-AGI Code Golf Solver - Enhanced Version[/bold cyan]")
    print_rich("[dim]Combining best solutions with improved optimization techniques[/dim]\n")
    
    # Define submission paths
    submission_paths = [
        f"{DIR}/input/oh-barnacles",
        f"{DIR}/input/qwen2-5-32b-arc-local-score-32-solved-script/submission",
        f"{DIR}/input/solved-127-problems-local-pleaseupvote",
        f"{DIR}/input/neurips-2025-google-code-golf-championship1"
    ]
    
    # Initialize tracking
    solved = 0
    total_score = 0
    n_tasks = 400
    final_solutions = {}
    removed_by_method = Counter()
    
    # Create output directory
    os.makedirs(f"{DIR}/working/submission", exist_ok=True)
    
    # Process each task
    for task_num in tqdm(range(1, n_tasks + 1), desc="Processing tasks"):
        task_id = f"{task_num:03d}"
        solutions = []
        task_data_path = f"{DIR}/input/google-code-golf-2025/task{task_id}.json"
        
        if not os.path.exists(task_data_path):
            continue
            
        try:
            # Load task data
            with open(task_data_path, 'r') as f:
                task_data = json.load(f)
            
            is_solved = False
            
            # Try solutions from each submission path
            for submission_path in submission_paths:
                try:
                    with open(f'{submission_path}/task{task_id}.py', 'r') as f:
                        solution = f.read()
                        if check_solution(solution, task_data):
                            # Apply optimizations sequentially - this is the key improvement!
                            optimization_methods = [
                                remove_spaces, minimize_indentation,
                                substitute_enumerate, substitute_range,
                                join_block_lines, strip_trailing_whitespaces,
                                remove_empty_lines, remove_comments,
                                shorten_variable_names, remove_spaces  # Apply remove_spaces twice
                            ]
                            
                            for method in optimization_methods:
                                b1 = get_bytes(solution)
                                new_solution = solution
                                try:
                                    new_solution = method(solution)
                                except Exception as e:
                                    continue
                                    
                                b2 = get_bytes(new_solution)
                                
                                if check_solution(new_solution, task_data) and b2 < b1:
                                    solution = new_solution
                                    removed_by_method[method] += (b1 - b2)
                            
                            # Try lambda conversion as final step
                            try:
                                lambda_solution = def_to_lambda(solution)
                                if check_solution(lambda_solution, task_data) and get_bytes(lambda_solution) < get_bytes(solution):
                                    solution = lambda_solution
                                    removed_by_method[def_to_lambda] += (get_bytes(solution) - get_bytes(lambda_solution))
                            except:
                                pass
                            
                            solutions.append(solution)
                            is_solved = True
                except FileNotFoundError:
                    pass
            
            # Visualize first few tasks if enabled
            if VISUALIZE_TASKS and task_num <= NUM_TASKS_TO_VISUALIZE and is_solved:
                print_rich(f"\n[yellow]Visualizing Task {task_id}[/yellow]")
                best_solution = min(solutions, key=get_bytes)
                visualize_task(task_data, best_solution, task_id)
            
            # Save best solution
            score = 0.001
            with open(f"{DIR}/working/submission/task{task_id}.py", "w") as f:
                if is_solved:
                    best_solution = min(solutions, key=get_bytes)
                    score = calculate_score(best_solution)
                    solved += 1
                    f.write(best_solution)
                    final_solutions[task_id] = best_solution
                else:
                    f.write('def p(g):return g')
                    final_solutions[task_id] = 'def p(g):return g'
            
            total_score += score
                
        except Exception as e:
            print_rich(f"[red]Error processing task {task_id}: {str(e)}[/red]")
            final_solutions[task_id] = 'def p(g):return g'
            total_score += 0.001
    
    # Create submission zip
    with zipfile.ZipFile(f"{DIR}/working/submission.zip", "w") as zipf:
        for task_num in range(1, n_tasks + 1):
            task_id = f"{task_num:03d}"
            zipf.write(f"{DIR}/working/submission/task{task_id}.py",
                      arcname=f"task{task_id}.py")
    
    # Print final statistics
    print_rich("\n[bold green]ğŸ“Š Final Results:[/bold green]")
    print_rich(f"[green]Total solved: {solved} / {n_tasks}[/green]")
    print_rich(f"[blue]LB Score: {total_score:.3f}[/blue]")
    
    if removed_by_method:
        print_rich("\n[bold yellow]ğŸ“ˆ Optimization Statistics:[/bold yellow]")
        print_rich("[dim]Bytes saved by optimization technique:[/dim]")
        for method, bytes_saved in removed_by_method.most_common():
            print_rich(f"  {method.__name__:<30} [cyan]{bytes_saved:>6}[/cyan] bytes")
    
    # Calculate average solution size
    total_bytes = sum(get_bytes(sol) for sol in final_solutions.values())
    avg_bytes = total_bytes / len(final_solutions) if final_solutions else 0
    print_rich(f"\n[dim]Average solution size: {avg_bytes:.1f} bytes[/dim]")
    
    return final_solutions, {
        'solved': solved,
        'total_score': total_score,
        'optimization_stats': removed_by_method
    }


def visualize_sample_tasks(final_solutions, num_tasks=5, specific_tasks=None):
    """Visualize sample tasks to see how solutions work."""
    print_rich("[bold cyan]ğŸ�¨ Visualizing Sample Tasks[/bold cyan]\n")
    
    # If specific tasks are requested, use those
    if specific_tasks:
        task_ids = [f"{t:03d}" for t in specific_tasks]
    else:
        # Otherwise, pick some interesting tasks (mix of easy and complex)
        # Try to get a variety: some early, some middle, some late tasks
        interesting_tasks = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 350]
        task_ids = []
        for t in interesting_tasks:
            task_id = f"{t:03d}"
            if task_id in final_solutions:
                task_ids.append(task_id)
                if len(task_ids) >= num_tasks:
                    break
    
    visualized_count = 0
    
    for task_id in task_ids:
        task_data_path = f"{DIR}/input/google-code-golf-2025/task{task_id}.json"
        
        if not os.path.exists(task_data_path):
            continue
            
        try:
            # Load task data
            with open(task_data_path, 'r') as f:
                task_data = json.load(f)
            
            solution = final_solutions.get(task_id)
            if solution and solution != 'def p(g):return g':  # Skip fallback solutions
                print_rich(f"\n[yellow]Task {task_id}[/yellow]")
                print_rich(f"[dim]Solution ({get_bytes(solution)} bytes):[/dim]")
                print_rich(f"[green]{solution}[/green]")
                
                # Check if solution works
                if check_solution(solution, task_data):
                    print_rich("[green]âœ“ Solution verified[/green]")
                else:
                    print_rich("[red]âœ— Solution failed verification[/red]")
                
                # Visualize the task
                visualize_task(task_data, solution, task_id)
                
                visualized_count += 1
                
                # Add spacing between tasks
                print("\n" + "="*80 + "\n")
                
        except Exception as e:
            print_rich(f"[red]Error visualizing task {task_id}: {str(e)}[/red]")
    
    print_rich(f"\n[green]Visualized {visualized_count} tasks[/green]")





# Execute the main pipeline
if __name__ == "__main__":
    final_solutions, stats = create_ultimate_submission()
    print_rich("\n[bold green]âœ… Submission created successfully![/bold green]")
    print_rich("[dim]Output: /kaggle/working/submission.zip[/dim]")
    
    # Quick visualization of first 5 successfully solved tasks
    if VISUALIZE_TASKS:
        print_rich("[bold]ğŸ”� Visualizing First 5 Solved Tasks[/bold]\n")
        
        solved_tasks = []
        for i in range(1, 401):
            task_id = f"{i:03d}"
            if task_id in final_solutions and final_solutions[task_id] != 'def p(g):return g':
                solved_tasks.append(i)
                if len(solved_tasks) >= 5:
                    break
        
        if solved_tasks:
            visualize_sample_tasks(final_solutions, specific_tasks=solved_tasks)
        else:
            print_rich("[yellow]No solved tasks found.[/yellow]")




