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
from typing import List, Tuple, Set, Optional, Dict, Any

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
    for c in ['[', ']', '(', ')', '{', '}', '=', '!', '<', '>', '+', '-', '*', '/', '%', ';', ',', ':', '&', '|', '^']:
        s = s.replace(' ' + c, c)
        s = s.replace(c + ' ', c)
    return s


def minimize_indentation(s):
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


def find_local_variables(s: str) -> Set[str]:
    tree = ast.parse(s)
    local_vars = set()

    class LocalVarVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            for arg in node.args.args:
                local_vars.add(arg.arg)
            for arg in node.args.kwonlyargs:
                local_vars.add(arg.arg)
            if node.args.vararg:
                local_vars.add(node.args.vararg.arg)
            if node.args.kwarg:
                local_vars.add(node.args.kwarg.arg)
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    local_vars.add(target.id)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                local_vars.add(node.target.id)
            self.generic_visit(node)

        def visit_For(self, node: ast.For):
            if isinstance(node.target, ast.Name):
                local_vars.add(node.target.id)
            self.generic_visit(node)

        def visit_With(self, node: ast.With):
            for item in node.items:
                if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    local_vars.add(item.optional_vars.id)
            self.generic_visit(node)

        def visit_ExceptHandler(self, node: ast.ExceptHandler):
            if node.name and isinstance(node.name, str):
                local_vars.add(node.name)
            self.generic_visit(node)

    LocalVarVisitor().visit(tree)
    return local_vars


def find_single_letter_variables(s: str) -> Set[str]:
    return {var for var in find_local_variables(s) if len(var) == 1}


def find_available_single_letter_variables(s: str) -> Set[str]:
    used = find_single_letter_variables(s)
    possible = set(string.ascii_lowercase + string.ascii_uppercase + '_')
    return possible - used


def substitute_builtin(s: str, builtin_name: str, min_count: int) -> str:
    pattern = r'\b' + re.escape(builtin_name) + r'\s*\('
    matches = re.findall(pattern, s)
    if len(matches) < min_count:
        return s

    available = find_available_single_letter_variables(s)
    if not available:
        return s

    v = available.pop()
    s = re.sub(pattern, v + '(', s)

    index = s.find(')')
    if index == -1:
        return s
    return s[:index] + f',{v}={builtin_name}' + s[index:]


def strip_trailing_whitespaces(s):
    return s.strip()


def _replace_variable_name(s, old_name, new_name):
    pattern = rf'\b{re.escape(old_name)}\b'
    return re.sub(pattern, new_name, s)


def shorten_variable_names(s):
    long_variable_names = [x for x in find_local_variables(s) if len(x) > 1]
    available = find_available_single_letter_variables(s)
    for x in long_variable_names:
        if not available:
            break
        new_name = available.pop()
        s = _replace_variable_name(s, x, new_name)
    return s


def join_block_lines(code: str) -> str:
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

    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While, ast.With,
                                 ast.If, ast.FunctionDef)):
            continue
        if isinstance(node, ast.If) and node.orelse:
            continue
        body = getattr(node, 'body', None)
        if not body:
            continue

        if not all(isinstance(stmt, SIMPLE) and stmt.lineno == stmt.end_lineno
                   for stmt in body):
            continue
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


def remove_empty_lines(s):
    lines = s.split('\n')
    return '\n'.join([line for line in lines if line.strip()])


def remove_comments(s):
    lines = s.split('\n')
    return '\n'.join([line for line in lines if not line.strip().startswith('#')])


def remove_trailing_spaces(s):
    return re.sub(r' +$', '', s, flags=re.MULTILINE)


def remove_spaces_inside_brackets(s):
    s = re.sub(r'\(\s+', '(', s)
    s = re.sub(r'\s+\)', ')', s)
    s = re.sub(r'\[\s+', '[', s)
    s = re.sub(r'\s+\]', ']', s)
    s = re.sub(r'\{\s+', '{', s)
    s = re.sub(r'\s+\}', '}', s)
    s = re.sub(r',\s+', ',', s)
    s = re.sub(r':\s+', ':', s)
    return s


def def_to_lambda(s):
    try:
        tree = ast.parse(s)
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
            func = tree.body[0]
            if len(func.body) == 1 and isinstance(func.body[0], ast.Return):
                args = [arg.arg for arg in func.args.args]
                args_str = ','.join(args)
                expr = ast.unparse(func.body[0].value).strip()
                return f"p=lambda {args_str}:{expr}"
    except:
        pass
    return s


def optimize_booleans(s):
    s = re.sub(r'== True\b', '', s)
    s = re.sub(r'== False\b', 'not ', s)
    s = re.sub(r'if not (.*?): return False\b', r'if \1:return False', s)
    s = re.sub(r'if not (.*?): return True\b', r'if \1:return True', s)
    return s


def minify_imports(s):
    lines = s.split('\n')
    new_lines = []
    imports = []
    for line in lines:
        if line.strip().startswith(('import ', 'from ')):
            imports.append(line.strip())
        else:
            new_lines.append(line)
    if imports:
        new_lines = [';'.join(imports)] + new_lines
    return '\n'.join(new_lines)


def remove_unused_assignments(s):
    try:
        tree = ast.parse(s)
    except SyntaxError:
        return s

    class RemoveUnusedAssignments(ast.NodeTransformer):
        def __init__(self):
            self.assignments = {}
            self.used = set()

        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.assignments[target.id] = node
            return self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                self.used.add(node.id)
            return node

    transformer = RemoveUnusedAssignments()
    transformer.visit(tree)
    unused = set(transformer.assignments.keys()) - transformer.used

    if not unused:
        return s

    new_lines = []
    for line in s.split('\n'):
        for var in unused:
            if re.search(rf'\b{var}\s*=', line) and not re.search(rf'\b{var}\b[^=]', line):
                break
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)


submission_paths = [
    f"{DIR}/input/lb47268-neurips-2025-google-code-golf/submission",
    f"{DIR}/input/qwen2-5-32b-arc-local-score-32-solved-script/submission",
    f"{DIR}/input/solved-57-task/submission",
    f"{DIR}/input/code-golf-ensemble-local-score-391-400-dsl/submission",
    f"{DIR}/input/google-code-golf-championship-task-253-solution/submission",
    f"{DIR}/input/solved-127-problems-local-pleaseupvote/submission",
    f"{DIR}/input/oh-barnacles/submission",
    f"{DIR}/input/google-code-golf-arc-solver-viz-lb/submission",
    f"{DIR}/input/a-bit-of-code-golf/submission",
    f"{DIR}/input/neurips-2025-gcgc-analytic/submission",
    f"{DIR}/input/simpletasksolved/submission",
    f"{DIR}/input/neurips-local-score/submission",
    f"{DIR}/input/the-scoring-system-has-been-hacked-easily/submission",
    f"{DIR}/input/solved-stater-neurips-2025-google-code-golf/submission",
    f"{DIR}/input/a-bit-more-of-code-golf-189-400/submission",
]


solved = 0
total_score = 0
os.makedirs(f"{DIR}/working/submission", exist_ok=True)

n_tasks = 400

removed_by_method = Counter()

for task_num in tqdm(range(1, n_tasks + 1)):
    task_id = f"{task_num:03d}"
    solutions = []
    task_data_path = f"{DIR}/input/google-code-golf-2025/task{task_id}.json"
    task_data = json.load(open(task_data_path))

    is_solved = False

    for submission_path in submission_paths:
        try:
            with open(f'{submission_path}/task{task_id}.py', 'r') as f:
                solution = f.read()
                if check_solution(solution, task_data):
                    for method in [
                        remove_spaces,
                        minimize_indentation,
                        lambda s: substitute_builtin(s, 'enumerate', 2),
                        lambda s: substitute_builtin(s, 'range', 3),
                        lambda s: substitute_builtin(s, 'len', 3),
                        lambda s: substitute_builtin(s, 'zip', 2),
                        lambda s: substitute_builtin(s, 'map', 2),
                        join_block_lines,
                        strip_trailing_whitespaces,
                        remove_empty_lines,
                        remove_comments,
                        remove_trailing_spaces,
                        remove_spaces_inside_brackets,
                        shorten_variable_names,
                        remove_spaces,
                        optimize_booleans,
                        minify_imports,
                        remove_unused_assignments,
                        def_to_lambda,
                    ]:
                        b1 = get_bytes(solution)
                        new_solution = solution
                        try:
                            new_solution = method(solution)
                        except Exception:
                            pass
                        b2 = get_bytes(new_solution)

                        if check_solution(new_solution, task_data) and b2 < b1:
                            solution = new_solution
                            removed_by_method[method] += (b1 - b2)

                    solutions.append(solution)
                    is_solved = True
        except FileNotFoundError:
            pass

    score = 0.0001

    with open(f"{DIR}/working/submission/task{task_id}.py", "w") as f:
        if is_solved:
            best_solution = min(solutions, key=get_bytes)
            score = calculate_score(best_solution)
            solved += 1
            f.write(best_solution)
        else:
            f.write('def p(g):return g')

    total_score += score

print_rich(f"[green]Total solved: {solved} / 400[/green]")
print_rich(f"[blue]LB Score: {total_score:.3f}[/blue]")

for k, v in removed_by_method.most_common():
    print(f"{k.__name__:<30}{v:>5}")

with zipfile.ZipFile(f"{DIR}/working/submission.zip", "w") as zipf:
    for task_num in range(1, n_tasks + 1):
        task_id = f"{task_num:03d}"
        zipf.write(f"{DIR}/working/submission/task{task_id}.py",
                   arcname=f"task{task_id}.py")


# import os
# import json
# import zipfile
# import copy
# import re
# import math
# import ast
# import string
# import itertools
# from collections import Counter
# from functools import reduce
# from typing import List, Tuple, Set, Optional, Dict, Any

# from tqdm import tqdm
# from rich import print as print_rich

# import warnings
# warnings.filterwarnings("ignore", category=SyntaxWarning)

# DIR = "/kaggle"

# def check_solution(solution, task_data):
#     try:
#         namespace = {}
#         exec(solution, namespace)
#         if 'p' not in namespace:
#             return False
#         all_examples = task_data['train'] + task_data['test'] + task_data['arc-gen']
#         for example in all_examples:
#             input_grid = copy.deepcopy(example['input'])
#             expected = example['output']
#             try:
#                 actual = namespace['p'](input_grid)
#                 if actual != expected:
#                     return False
#             except Exception:
#                 return False
#         return True
#     except Exception:
#         return False

# def get_bytes(s):
#     return len(s.encode('utf-8'))

# def calculate_score(s):
#     return max(1, 2500 - get_bytes(s))

# def remove_spaces(s):
#     for c in ['[',']','(',')','{','}','=','!','<','>','+','-','*','/','%',';',',',':','&','|','^','~']:
#         s = s.replace(' '+c,c).replace(c+' ',c)
#     s = re.sub(r' +',' ',s)
#     return s.strip()

# def minimize_indentation(s):
#     leading_spaces = [len(m.group(1)) for m in re.finditer(r'^( +)(?=\S)',s,flags=re.MULTILINE)]
#     if not leading_spaces: return s
#     unit = reduce(math.gcd,leading_spaces)
#     if unit<=1: return s
#     return re.sub(r'^( +)',lambda m:' '*(len(m.group(1))//unit),s,flags=re.MULTILINE)

# def find_local_variables(s):
#     tree = ast.parse(s)
#     locals = set()
#     class Visitor(ast.NodeVisitor):
#         def visit_FunctionDef(self, node):
#             locals.update(a.arg for a in node.args.args)
#             if node.args.vararg: locals.add(node.args.vararg.arg)
#             if node.args.kwarg: locals.add(node.args.kwarg.arg)
#             self.generic_visit(node)
#         def visit_Name(self, node):
#             if isinstance(node.ctx,ast.Store): locals.add(node.id)
#             self.generic_visit(node)
#     Visitor().visit(tree)
#     return locals

# def get_available_vars(s):
#     used = {v for v in find_local_variables(s) if len(v)==1}
#     return set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_')-used

# def substitute_builtin(s,fn,min_uses=2):
#     if s.count(fn+'(')<min_uses: return s
#     v = next(iter(get_available_vars(s)),None)
#     if not v: return s
#     s = s.replace(fn,v)
#     i = s.find(')')
#     return s[:i]+f',{v}={fn}'+s[i:] if i!=-1 else s

# def shorten_vars(s):
#     vars = [v for v in find_local_variables(s) if len(v)>1]
#     avail = get_available_vars(s)
#     for v in vars:
#         if not avail: break
#         n = avail.pop()
#         s = re.sub(r'\b'+v+r'\b',n,s)
#     return s

# def join_simple_blocks(s):
#     try:
#         tree = ast.parse(s)
#         lines = s.splitlines()
#         transforms = []
#         for node in ast.walk(tree):
#             if not isinstance(node,(ast.For,ast.While,ast.If,ast.With)): continue
#             if isinstance(node,ast.If) and node.orelse: continue
#             if not all(isinstance(stmt,(ast.Assign,ast.AugAssign,ast.Expr,ast.Return,ast.Delete,ast.Pass,ast.Continue,ast.Break,ast.Assert,ast.Raise,ast.Global,ast.Nonlocal,ast.Import,ast.ImportFrom)) and stmt.lineno==stmt.end_lineno for stmt in node.body): continue
#             transforms.append((node.lineno,node.body[0].lineno,node.body[-1].end_lineno))
#         for hdr,start,end in sorted(transforms,key=lambda x:-x[0]):
#             header = lines[hdr-1].rstrip()
#             if not header.endswith(':'): continue
#             lines[hdr-1] = header+';'.join(l.strip() for l in lines[start-1:end])+'\n'
#             del lines[start-1:end]
#         return ''.join(lines)
#     except:
#         return s

# def remove_empty_lines(s):
#     return '\n'.join(l for l in s.split('\n') if l.strip())

# def remove_comments(s):
#     return '\n'.join(l for l in s.split('\n') if not l.lstrip().startswith('#'))

# def remove_trailing_spaces(s):
#     return re.sub(r' +$','',s,flags=re.MULTILINE)

# def compress_imports(s):
#     imports = [l.strip() for l in s.split('\n') if l.strip().startswith(('import ','from '))]
#     if not imports: return s
#     other = [l for l in s.split('\n') if not l.strip().startswith(('import ','from '))]
#     return ';'.join(imports)+'\n'+'\n'.join(other)

# def def_to_lambda(s):
#     try:
#         tree = ast.parse(s)
#         if len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef):
#             f = tree.body[0]
#             if len(f.body)==1 and isinstance(f.body[0],ast.Return):
#                 args = ','.join(a.arg for a in f.args.args)
#                 expr = ast.unparse(f.body[0].value).strip()
#                 return f"p=lambda {args}:{expr}"
#     except:
#         pass
#     return s

# def optimize_booleans(s):
#     s = re.sub(r'== True\b','',s)
#     s = re.sub(r'== False\b','not ',s)
#     s = re.sub(r'if not (.*?): return False\b',r'if \1:return False',s)
#     s = re.sub(r'if not (.*?): return True\b',r'if \1:return True',s)
#     return s

# def remove_unused_assignments(s):
#     try:
#         tree = ast.parse(s)
#         used = set()
#         assigns = {}
#         class Visitor(ast.NodeVisitor):
#             def visit_Name(self, node):
#                 if isinstance(node.ctx,ast.Load): used.add(node.id)
#                 self.generic_visit(node)
#             def visit_Assign(self, node):
#                 for t in node.targets:
#                     if isinstance(t,ast.Name): assigns[t.id] = node
#                 self.generic_visit(node)
#         Visitor().visit(tree)
#         unused = set(assigns.keys())-used
#         if not unused: return s
#         lines = []
#         for l in s.split('\n'):
#             keep = True
#             for u in unused:
#                 if re.search(r'\b'+u+r'\s*=',l) and not re.search(r'\b'+u+r'\b[^=]',l):
#                     keep = False
#                     break
#             if keep: lines.append(l)
#         return '\n'.join(lines)
#     except:
#         return s

# def optimize_comprehensions(s):
#     s = re.sub(r'list\((.*?)\)',r'[\1]',s)
#     s = re.sub(r'set\((.*?)\)',r'{\1}',s)
#     s = re.sub(r'dict\((.*?)\)',r'{\1}',s)
#     return s

# def optimize_math(s):
#     s = re.sub(r'math\.sqrt\((.*?)\)',r'(\1)**0.5',s)
#     s = re.sub(r'math\.pow\((.*?),(.*?)\)',r'(\1)**(\2)',s)
#     return s

# def optimize_conditionals(s):
#     s = re.sub(r'if (.*?): return (.*?)\nelse: return (.*?)\b',r'return \2 if \1 else \3',s)
#     return s

# def optimize_returns(s):
#     s = re.sub(r'return (.*?)\nreturn (.*?)\b',r'return \1 or \2',s)
#     return s

# def optimize_loops(s):
#     s = re.sub(r'for (.*?) in range\((.*?)\):\n\s*if (.*?): continue\n\s*(.*?)\b',r'for \1 in range(\2):\n if not \3:\n  \4',s)
#     return s

# submission_paths = [
#     f"{DIR}/input/lb47268-neurips-2025-google-code-golf/submission",
#     f"{DIR}/input/qwen2-5-32b-arc-local-score-32-solved-script/submission",
#     f"{DIR}/input/solved-57-task/submission",
#     f"{DIR}/input/code-golf-ensemble-local-score-391-400-dsl/submission",
#     f"{DIR}/input/google-code-golf-championship-task-253-solution/submission",
#     f"{DIR}/input/solved-127-problems-local-pleaseupvote/submission",
#     f"{DIR}/input/oh-barnacles/submission",
#     f"{DIR}/input/google-code-golf-arc-solver-viz-lb/submission",
#     f"{DIR}/input/a-bit-of-code-golf/submission",
#     f"{DIR}/input/neurips-2025-gcgc-analytic/submission",
#     f"{DIR}/input/simpletasksolved/submission",
#     f"{DIR}/input/neurips-local-score/submission",
#     f"{DIR}/input/the-scoring-system-has-been-hacked-easily/submission",
#     f"{DIR}/input/solved-stater-neurips-2025-google-code-golf/submission",
#     f"{DIR}/input/a-bit-more-of-code-golf-189-400/submission",
# ]


# solved = 0
# total_score = 0
# os.makedirs(f"{DIR}/working/submission", exist_ok=True)

# n_tasks = 400
# removed_by_method = Counter()

# optimization_pipeline = [
#     remove_spaces,
#     minimize_indentation,
#     lambda s: substitute_builtin(s,'enumerate',2),
#     lambda s: substitute_builtin(s,'range',3),
#     lambda s: substitute_builtin(s,'len',3),
#     lambda s: substitute_builtin(s,'zip',2),
#     lambda s: substitute_builtin(s,'map',2),
#     lambda s: substitute_builtin(s,'sorted',2),
#     lambda s: substitute_builtin(s,'sum',2),
#     lambda s: substitute_builtin(s,'max',2),
#     lambda s: substitute_builtin(s,'min',2),
#     join_simple_blocks,
#     remove_trailing_spaces,
#     remove_empty_lines,
#     remove_comments,
#     remove_trailing_spaces,
#     shorten_vars,
#     remove_spaces,
#     optimize_booleans,
#     compress_imports,
#     remove_unused_assignments,
#     def_to_lambda,
#     optimize_comprehensions,
#     optimize_math,
#     optimize_conditionals,
#     optimize_returns,
#     optimize_loops,
# ]

# for task_num in tqdm(range(1, n_tasks + 1)):
#     task_id = f"{task_num:03d}"
#     solutions = []
#     task_data_path = f"{DIR}/input/google-code-golf-2025/task{task_id}.json"
#     task_data = json.load(open(task_data_path))

#     for submission_path in submission_paths:
#         try:
#             with open(f'{submission_path}/task{task_id}.py', 'r') as f:
#                 solution = f.read()
#                 if check_solution(solution, task_data):
#                     for method in optimization_pipeline:
#                         b1 = get_bytes(solution)
#                         try:
#                             new_solution = method(solution)
#                             if check_solution(new_solution, task_data) and get_bytes(new_solution) < b1:
#                                 solution = new_solution
#                                 removed_by_method[method.__name__ if hasattr(method,'__name__') else method.__class__.__name__] += (b1 - get_bytes(new_solution))
#                         except:
#                             pass
#                     solutions.append(solution)
#         except:
#             pass

#     with open(f"{DIR}/working/submission/task{task_id}.py", "w") as f:
#         if solutions:
#             best_solution = min(solutions, key=get_bytes)
#             score = calculate_score(best_solution)
#             solved += 1
#             f.write(best_solution)
#         else:
#             f.write('def p(g):return g')
#             score = 0.0001
#         total_score += score

# print_rich(f"[green]Total solved: {solved} / 400[/green]")
# print_rich(f"[blue]LB Score: {total_score:.3f}[/blue]")

# for k,v in removed_by_method.most_common(20):
#     print(f"{k:<30}{v:>10}")

# with zipfile.ZipFile(f"{DIR}/working/submission.zip", "w") as zipf:
#     for task_num in range(1, n_tasks + 1):
#         task_id = f"{task_num:03d}"
#         zipf.write(f"{DIR}/working/submission/task{task_id}.py", arcname=f"task{task_id}.py")




