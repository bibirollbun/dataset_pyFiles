# Compatible with zlib but provides better compression
!pip install zopfli


import ast, keyword, re, random, json, zopfli.zlib, copy, sys, os, zipfile, glob, warnings, signal, zlib
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *

# --- Core Utilities ---

def create_template_from_function(code_string: str) -> (str, list):
    tree = ast.parse(code_string)
    variable_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id not in keyword.kwlist and node.id not in ['Counter', 'next', 'int', 'chain','enumerate', 'combinations', 'product', 'str', 'abs', 'exec','len', 'min', 'max', 'range', 'set','any', 'filter', 'list', 'map', 'sum', 'tuple', 'zip', 'all', 'sorted']}
    template = code_string
    for name in sorted(list(variable_names), key=len, reverse=True):
        template = re.sub(r'\b' + re.escape(name) + r'\b', f'##{name}##', template)
    return template.replace("def ##p##", "def p").replace("##p##=lambda", "p=lambda").replace("##f##'", "f'").replace('##f##"', 'f"'), sorted(list(variable_names))

def get_score(code: str, examples_to_check: list) -> (int, int):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SyntaxWarning)
            solution_namespace = {}
            exec(code, solution_namespace)
            p_func = solution_namespace.get('p')
            for _, example in examples_to_check:
                if json.dumps(p_func(copy.deepcopy(example['input']))) != json.dumps(example['output']):
                    return 999, 999
            compressed = zopfli.zlib.compress(code.encode())
            # compressed = zlib.compress(code.encode(), 9)
            penalty = sum(compressed.count(c) for c in [b'\\', b'\0', b'\n', b'\r']) + min(compressed.count(b"'"), compressed.count(b'"'))
            return len(compressed), penalty
    except Exception:
        return 998, 998

def validate_code(code: str, all_examples_to_check: list) -> tuple | None:
    """Checks code against all examples. Returns the first failing example or None."""
    if UNSAFE_MODE: all_examples_to_check = all_examples_to_check[:1]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SyntaxWarning)
            signal.alarm(1)
            solution_namespace = {}
            exec(code, solution_namespace)
            p_func = solution_namespace.get('p')
            signal.alarm(0)
            
            for i, example in all_examples_to_check:
                # signal.alarm(1)
                if json.dumps(p_func(copy.deepcopy(example['input']))) != json.dumps(example['output']):
                    # signal.alarm(0)
                    return i, example # FAILED
                # signal.alarm(0)
            return None # PASSED
    except Exception:
        # Code fails to execute, so it's invalid. Return the first example as the failure point.
        return all_examples_to_check[0]

# --- Setup ---
RAW_FUNCTION_STRING = """
def p(grid):[((sprite:=[grid[src_row+cell_index//3][src_col+cell_index%3]for cell_index in range(9)])==sprite[::-1])*sprite[4]*any(sprite[:4])*(sprite[4]==grid[dest_row+1][dest_col+1]or sum(grid[dest_row+cell_index//3][dest_col+cell_index%3]==sprite[cell_index]for cell_index in range(9))==8)and exec('for cell_index in range(9):grid[dest_row+cell_index//3][dest_col+cell_index%3]=sprite[cell_index]')for dest_row in range(len(grid)-2) for src_row in range(len(grid)-2) for dest_col in range(len(grid[0])-2) for src_col in range(len(grid[0])-2)];return grid
""".strip()
TASK_ID = 173
UNSAFE_MODE = False
task_data = load_examples(TASK_ID)
all_examples = list(enumerate(task_data.get('train', []) + task_data.get('test', []) + task_data.get('arc-gen', [])))
checked_examples = [ex for ex in all_examples if ex[0] in [240]]
checked_example_ids = {ex[0] for ex in checked_examples}

# --- Optimization Pipeline ---
FUNCTION_TEMPLATE, original_vars = create_template_from_function(RAW_FUNCTION_STRING)
print(f"Initial variables: {original_vars}\n")
candidate_names = list("qertyuiopasdfghjklzxcbnm")

initial_code = FUNCTION_TEMPLATE.replace("##", "")
PAYLOAD_OVERHEAD = 60

# --- Initial Validation ---
print("Running initial validation against all examples...")
if validate_code(initial_code, all_examples) is not None:
    print("FATAL: Initial raw function is incorrect. Exiting.")
    raise ValueError("Failed with original")
print("Initial code PASSED validation.")

current_base, current_penalty = get_score(initial_code, checked_examples)
current_total_size = PAYLOAD_OVERHEAD + current_base + current_penalty

# Global best tracking
global_best_code = initial_code
global_best_base, global_best_penalty = current_base, current_penalty
global_best_total_size = current_total_size

# Last known good tracking
last_known_good_code = initial_code
last_known_good_base, last_known_good_penalty = current_base, current_penalty
last_known_good_total_size = current_total_size

print(f"Initial size: {global_best_total_size} (Base: {current_base}, Penalty: {current_penalty})\n{initial_code}\n" + "-" * 30)

LIMIT = 4000
REBASE_INTERVAL = 500
NEXT_REBASE = 500

for i in range(LIMIT):
    if i > 0 and i % NEXT_REBASE == 0:
        REBASE_INTERVAL *= 1.3
        NEXT_REBASE += REBASE_INTERVAL
        print(f"\n--- Rebase at iter {i}: Validating global best (Size: {global_best_total_size}) ---")
        
        failing_example = validate_code(global_best_code, all_examples)
        
        if failing_example:
            fail_id, fail_ex = failing_example
            print(f"VALIDATION FAILED on example #{fail_id}! Reverting to last known good solution.")
            # Revert global best to the last one that passed
            global_best_code = last_known_good_code
            global_best_base, global_best_penalty = last_known_good_base, last_known_good_penalty
            global_best_total_size = last_known_good_total_size
            print(f"Reverted to size: {global_best_total_size}")
            
            # Add the new failing example to the checked set if it's not already there
            if fail_id not in checked_example_ids:
                checked_example_ids.add(fail_id)
                checked_examples = [ex for ex in all_examples if ex[0] in checked_example_ids]
                print(f"Added example #{fail_id} to the active test set. (Now checking {len(checked_examples)} examples)")
        else:
            # Update the checkpoint to the current global best
            last_known_good_code = global_best_code
            last_known_good_base, last_known_good_penalty = global_best_base, global_best_penalty
            last_known_good_total_size = global_best_total_size

        FUNCTION_TEMPLATE, original_vars = create_template_from_function(global_best_code)
        current_mapping = {var: var for var in original_vars}
        current_base, current_penalty = get_score(global_best_code, checked_examples) # Rescore with potentially new examples
        print(f"New rebase variables: {original_vars}\n" + "-" * 30)

    if not original_vars: continue
    
    trial_mapping = {var: var for var in original_vars} # Start from identity map for the current template
    num_changes = random.randint(1, min(6, len(original_vars)))
    vars_to_change = random.sample(original_vars, k=num_changes)
    for var, new_name in zip(vars_to_change, random.sample(candidate_names, k=num_changes)):
        trial_mapping[var] = new_name
    
    trial_code = FUNCTION_TEMPLATE
    for var in original_vars:
        trial_code = trial_code.replace(f"##{var}##", trial_mapping[var])

    trial_base, trial_penalty = get_score(trial_code, checked_examples)
    trial_total_size = PAYLOAD_OVERHEAD + trial_base + trial_penalty
    
    if trial_total_size <= global_best_total_size:
        global_best_code = trial_code
        global_best_base, global_best_penalty = trial_base, trial_penalty
        if trial_total_size < global_best_total_size:
            print(f"\nNew best: {trial_total_size} (B:{global_best_base}, P:{global_best_penalty}) @{i+1}")
            print(trial_code)
        global_best_total_size = trial_total_size

# --- Final Result ---
print(f"\nFinal validation of best code found...")
if validate_code(global_best_code, all_examples) is not None:
    print("WARNING: The final best code failed full validation. Something may be wrong.")
else:
    print("Final code PASSED validation.")

print(f"\nBest score achieved: {global_best_total_size} bytes (Base: {global_best_base}, Penalty: {global_best_penalty})")
print("\nFinal optimized code:")
print(global_best_code)


SUB_DIR = "/kaggle/working/submission"
os.makedirs(SUB_DIR, exist_ok=True)

def save_solution(task_id, code):
    raw_bytes = code.strip().encode()
    compressed = zopfli.zlib.compress(raw_bytes)
    quote = b"'" if b'"' in compressed else b'"'
    wrapper = b"#coding:L1\nimport zlib;exec(zlib.decompress(bytes(%s,'L1')))" % (quote + compressed + quote)
    
    use_compressed = len(wrapper) < len(raw_bytes)
    final_bytes = wrapper if use_compressed else raw_bytes
    
    path = os.path.join(SUB_DIR, f"task{task_id:03d}.py")
    with open(path, 'wb') as f:
        f.write(final_bytes)
    print(f"Saved Task {task_id} solution ({'compressed' if use_compressed else 'raw'}, {len(final_bytes)} bytes)")

def create_submission_zip():
    zip_path = "/kaggle/working/submission.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path in sorted(glob.glob(os.path.join(SUB_DIR, "task*.py"))):
            zipf.write(path, arcname=os.path.basename(path))
    
    if zipf.namelist():
      print(f"Created submission with {len(zipf.namelist())} file(s): {zip_path}")

save_solution(TASK_ID, global_best_code)
create_submission_zip()

