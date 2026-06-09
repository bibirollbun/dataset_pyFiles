import os
import json
import zipfile
import copy
import re
import math
import ast
import string
from collections import Counter
from functools import reduce
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from rich import print as print_rich

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

DIR = "/kaggle"


def check_solution(solution, task_data):
    try:
        namespace = {}
        exec(solution, namespace)
        if 'p' not in namespace:
            return False
        all_examples = task_data['train'] + task_data['test'] + task_data['arc-gen']
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
    return len(s.encode('utf-8'))


def calculate_score(s):
    return max(1, 2500 - get_bytes(s))


def remove_spaces(s):
    """
    Remove unnecessary before/after spaces some symbols / operators
    """
    for c in ['[', ']', '(', ')', '{', '}', '=', '!', '<', '>', '+', '-', '*', '/', '%', ';', ',',':']:
        s = s.replace(' ' + c, c)
        s = s.replace(c + ' ', c)
    return s


def minimize_indentation(s):
    """
    Minimize indentation to multipliers of 1 instead of usual 4
    """
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


def find_local_variables(s: str):
    """
    Find all variable names inside function
    """
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

    LocalVarVisitor().visit(tree)
    return local_vars


def find_single_letter_variables(s: str):
    """
    Find single letter variable names that are currently used
    """
    return {var for var in find_local_variables(s) if len(var) == 1}


def find_available_single_letter_variables(s: str):
    """
    Find single letter variable names that can be used to replace longer names
    """
    used = find_single_letter_variables(s)
    possible = set(string.ascii_lowercase + string.ascii_uppercase + '_')
    return possible - used


def substitute_range(s):
    """
    If there are at least 3x range() used in function assign it to single letter
    function parameter e.g def p(g,r=range)
    """
    v = find_available_single_letter_variables(s).pop()
    if s.count('range(') >= 3:
        s = s.replace('range', v)
        index = s.find(')')
        return s[:index] + ',' + v + '=range' + s[index:]
    return s


def substitute_enumerate(s):
    """
    If there are at least 2x enumerate() used in function assign it to single letter
    function parameter e.g def p(g,e=enumerate)
    """
    v = find_available_single_letter_variables(s).pop()
    if s.count('enumerate(') >= 2:
        s = s.replace('enumerate', v)
        index = s.find(')')
        return s[:index] + ',' + v + '=enumerate' + s[index:]
    return s


def strip_trailing_whitespaces(s):
    """
    Remove any trailing whitespaces (especially new line after last line of code)
    """
    return s.strip()


def _replace_variable_name(s, old_name, new_name):
    """
    Replace long (2+) variable name with single char variable name
    """
    pattern = rf'\b{re.escape(old_name)}\b'
    return re.sub(pattern, new_name, s)


def shorten_variable_names(s):
    """
    Find all variable that have too long names, find new name for them
    and then replace
    """
    long_variable_names = [x for x in find_local_variables(s) if len(x) > 1]
    available = find_available_single_letter_variables(s)
    for x in long_variable_names:
        new_name = available.pop()
        s = _replace_variable_name(s, x, new_name)
    return s


def join_block_lines(code: str) -> str:
    """
    Join lines when possible

    if a:
        print(1)
    ->
    if a:print(1)

    if a:
        print(1)
        print(2)
        print(3)
    ->
    if a:print(1);print(2);print(3)

    Make sure to not break the code, which is easy to do with multiple indentations or
    if else, or combinations of for, while, if, with inside block
    """

    # If parse raised error then just return original code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)

    lines = code.splitlines(keepends=True)
    transforms: List[Tuple[int, int, int]] = []

    SIMPLE = (
        ast.Assign, ast.AugAssign, ast.Expr, ast.Return,
        ast.Delete, ast.Pass, ast.Continue, ast.Break,
        ast.Assert, ast.Raise, ast.Global, ast.Nonlocal,
        ast.Import, ast.ImportFrom
    )

    # Find lines that can be joined
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While, ast.With,
                                 ast.If, ast.FunctionDef)):
            continue
        # Skip else/elif
        if isinstance(node, ast.If) and node.orelse:
            continue
        body = getattr(node, 'body', None)
        if not body:
            continue

        if not all(isinstance(stmt, SIMPLE) and stmt.lineno == stmt.end_lineno
                   for stmt in body):
            continue
        transforms.append((node.lineno, body[0].lineno, body[-1].end_lineno))

    # Apply transforms bottom-up so the previous changes does not affect later changes
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


def remove_empty_lines(s):
    """
    Remove empty lines from code
    """
    lines = s.split('\n')
    return '\n'.join([line for line in lines if line.strip()])


def remove_comments(s):
    """
    Remove single line comments from code
    """
    lines = s.split('\n')
    return '\n'.join([line for line in lines if not line.strip().startswith('#')])


def def_to_lambda(s):
    """
    Try to replace def with lambda
    """
    _def = r'^def\s+p\(([A-Za-z])\):return'
    if '\n' not in s and re.match(_def, s):
        s = re.sub(_def, r'p=lambda \1:', s)
    return s


submission_paths = [
    f"{DIR}/input/neurips-2025-google-code-golf-championship1",
    f"{DIR}/input/oh-barnacles"
]

os.makedirs(f"{DIR}/working/submission", exist_ok=True)

n_tasks = 400

methods = [remove_spaces, minimize_indentation, substitute_enumerate,
           substitute_range, join_block_lines, strip_trailing_whitespaces,
           remove_empty_lines, remove_comments, shorten_variable_names,
           remove_spaces]

def process_task(task_num):
    task_id = f"{task_num:03d}"
    solutions = []
    task_data_path = f"{DIR}/input/google-code-golf-2025/task{task_id}.json"
    task_data = json.load(open(task_data_path))

    solutions = []
    removed_local = Counter()
    is_solved = False

    for submission_path in submission_paths:
        try:
            with open(f'{submission_path}/task{task_id}.py', 'r') as f:
                solution = f.read()
                if check_solution(solution, task_data):
                    for method in methods:
                        b1 = get_bytes(solution)
                        new_solution = solution
                        try:
                            new_solution = method(solution)
                        except Exception as e:
                            print(e)
                        b2 = get_bytes(new_solution)

                        if check_solution(new_solution, task_data) and b2 < b1:
                            solution = new_solution
                            removed_by_method[method.__name__] += (b1 - b2)

                    solutions.append(solution)
                    is_solved = True
        except FileNotFoundError:
            pass

    score = 0.001

    with open(f"{DIR}/working/submission/task{task_id}.py", "w") as f:
        if is_solved:
            best_solution = min(solutions, key=get_bytes)
            score = calculate_score(best_solution)
            unsolved_task = None
            f.write(best_solution)
        else:
            unsolved_task = task_num
            f.write('def p(g):return g')

    return is_solved, score, removed_local, unsolved_task


# Parallelization
removed_by_method = Counter()
solved = 0
total_score = 0.0
unsolved = []

task_ids = range(1, n_tasks + 1)
results = []

print(f"{os.cpu_count()=}")

with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
    fut2id = {ex.submit(process_task, i): i for i in task_ids}
    for fut in tqdm(as_completed(fut2id), total=n_tasks):
        results.append(fut.result())

for is_solved, score, removed_local, unsolved_task in results:
    total_score += score
    if is_solved:
        solved += 1
    else:
        unsolved.append(unsolved_task)
    removed_by_method += removed_local


print_rich(f"[green]Total solved: {solved} / 400[/green]")
print_rich(f"[blue]LB Score: {total_score:.3f}[/blue]")

for k, v in removed_by_method.most_common():
    print(f"{k.__name__:<30}{v:>5}")

with zipfile.ZipFile(f"{DIR}/working/submission.zip", "w") as zipf:
    for task_num in range(1, n_tasks + 1):
        task_id = f"{task_num:03d}"
        zipf.write(f"{DIR}/working/submission/task{task_id}.py",
                   arcname=f"task{task_id}.py")

print_rich(f"[red]{unsolved=}")


# Task visualization
import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
import code_golf_utils
from code_golf_utils import *

for task_num in range(1, 101): # only display 100 at once to not cause any memory issues
    task_id = f"{task_num:03d}"
    examples = load_examples(task_num)
    show_examples(examples['train'][:1] + examples['test'][:1] )
    plt.figure(task_num).suptitle(
        f"task{task_id}" + [' : solved', ' : unsolved'][task_num in unsolved],
        fontsize=16,color=['green', 'red'][task_num in unsolved])

