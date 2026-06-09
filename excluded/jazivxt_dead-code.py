from IPython.core.display import Image, display
display(Image(url='https://i.ytimg.com/vi/uvXWGfPxoAI/maxresdefault.jpg'))


!pip install black
!pip install zopfli
!pip install python_minifier

from IPython.core.display import Image, display
import zopfli.zlib, zipfile, zlib, sys, os
import warnings, json, copy, black
import signal, time, functools
from zipfile import ZipFile
from zlib import compress
import python_minifier
from tqdm import tqdm

sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *

warnings.filterwarnings("ignore")

!cp /kaggle/input/baselines/submission.zip submission.zip
with zipfile.ZipFile("submission.zip", 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')


class TimeoutError(Exception):
    pass

def timeout(seconds=5, default=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handle_timeout(signum, frame):raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds} seconds.")
            old_handler = signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)
            try:result = func(*args, **kwargs)
            except TimeoutError:
                if default is not None:return default
                else:raise
            finally:signal.alarm(0);signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator


@timeout(seconds=2, default="Timed out!")
def check(solution, task_num, valall=False):
    task_data = load_examples(task_num)
    try:
        namespace = {}
        exec(solution, namespace)
        if 'p' not in namespace: return False
        all_examples = task_data['train'] + task_data['test'] + task_data['arc-gen']
        examples_to_check = all_examples if valall else all_examples[:3]
        for example in examples_to_check:
            input_grid = copy.deepcopy(example['input'])
            expected = example['output']
            try:
                actual = namespace['p'](input_grid)
                if actual != expected:
                    return False
            except:
                return False
        return True
    except Exception as e:
        return False


def remove_comments(s):
    lines = s.split('\n')
    return '\n'.join([line for line in lines if not line.strip().startswith('#')])

savings = 0
for i in range(1,401):
    if i not in [2,19,50,133,187,251]:
        p='/kaggle/working/kaggle/working/submission/task' + str(i).zfill(3) + '.py'
        s=open(p,'rb').read().decode('L1').strip()
        if '#coding:L1' in str(s):
            s = eval(s[50:-8])
            s = zlib.decompress(bytes(s,'L1')).decode('L1')  
        s1 = len(s)
        s = s.replace("\r\n", "\n").replace("\r", "")
        clean = black.format_str(s, mode=black.FileMode())
        clean = clean.replace('\r\n','\n').replace('\r','\n').replace('\n\n','\n').split('\n')[:-1]
        removes=[]
        for c in tqdm(range(len(clean))):
            s=clean[:]
            s[c]='#'+s[c]
            x='\n'.join(s)
            try:
                if check(x, i, valall=True):
                    print('*'*20)
                    print(p)
                    print('*'*20)
                    print(x)
                    print('+'*20)
                    clean=s[:]
            except: pass
        s = '\n'.join(clean)
        s = remove_comments(s)
        try:s = python_minifier.minify(s)
        except:pass
        s2 = len(s)
        if s1>s2:
            savings+=s1-s2
            print(savings)
        s=open(p,'wb').write(s.encode())


#******************************************************
# THE ZIPPA CODE FROM LINK BELOW SHOULD HAVE LIKE 200+ UPVOTES
# https://www.kaggle.com/code/cheeseexports/big-zippa



from zlib import compress
import zopfli.zlib

def zip_src(src):
 while (compressed := zopfli.zlib.compress(src))[-1] == ord('"'): src += b"#"
 def sanitize(b_in):
  b_out = bytearray()
  #Test Some Ordinal Replaces Here!!!
  for b in b_in:
   if   b==0:         b_out += b"\\x00"
   elif b==ord("\r"): b_out += b"\\r"
   elif b==ord("\\"): b_out += b"\\\\"
   else: b_out.append(b)
  return b"" + b_out
 #This is where it goes bad for anything past 128 ord but it works
 #compressed = compressed.decode('L1').encode('utf-8')
 compressed = sanitize(compressed)
 delim = b'"""' if ord("\n") in compressed or ord('"') in compressed else b'"'
 #return b"import zlib\nexec(zlib.decompress(bytes(" + delim + compressed + delim + b',"L1")))'
 return b"#encoding:L1\nimport zlib\nexec(zlib.decompress(bytes(" + delim + compressed + delim + b',"L1")))'


import os
import warnings

sources = ["/kaggle/working/kaggle/working/submission/"]
failing = []

warnings.filterwarnings("ignore")

total_save = 0
score = 0
submission = "/kaggle/working/submission"
os.makedirs(submission, exist_ok=True)

for task_num in range(1, 401):
 
 path_out = f"{submission}/task{task_num:03d}.py" 

 task_src = b"#" * 1_000_000
 for source in sources:
  path_in  = f"{source}/task{task_num:03d}.py"
  if not os.path.exists(path_in):continue
  if path_in in failing:continue

  with open(path_in, "rb") as task_in:
   new_src = task_in.read()
   task_src = min([task_src, new_src], key=len)
 zipped_src = zip_src(task_src)
 improvement = len(task_src) - len(zipped_src)
 if improvement > 0:
  task_src = zipped_src
  total_save += improvement
  #print(task_num, improvement)
 score += max(1, 2500-len(task_src))
 with open(path_out, "wb") as task_out:
  task_out.write(task_src)

print(f"Saved {total_save}b with zlib")
print(f"Projected score: {score}")


# The outputs are not optimized for compression
#***********************************
#THIS CODE LINK BELOW DESERVES A LOT MORE UPVOTES LIKE 200+
#***********************************
#https://www.kaggle.com/code/garrymoss/compressed-variable-name-optimization


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

@timeout(seconds=2, default="Timed out!")
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

UNSAFE_MODE = False
def compression_opt(RAW_FUNCTION_STRING, TASK_ID):
    global UNSAFE_MODE
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
    
    print(f"Initial size: {global_best_total_size} (Base: {current_base}, Penalty: {current_penalty})\n" + "-" * 30)
    
    LIMIT = 10
    REBASE_INTERVAL = 5
    NEXT_REBASE = 5
    
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
                #print(trial_code)
            global_best_total_size = trial_total_size
    
    # --- Final Result ---
    print(f"\nFinal validation of best code found...")
    if validate_code(global_best_code, all_examples) is not None:
        print("WARNING: The final best code failed full validation. Something may be wrong.")
    else:
        print("Final code PASSED validation.")
    
    print(f"\nBest score achieved: {global_best_total_size} bytes (Base: {global_best_base}, Penalty: {global_best_penalty})")
    return global_best_code

e=[]
for i in range(1 ,401):
    p='/kaggle/working/kaggle/working/submission/task' + str(i).zfill(3) + '.py'
    s = open(p,'rb').read().decode('L1')
    if len(s)>400 and i not in [42, 66, 137, 158, 182, 302, 370, 383]:
        print(i, '*'*40)
        try:
            global_best_code = compression_opt(s,i)
            if check(global_best_code, i, True):
                 open(p,'wb').write(global_best_code.encode('L1'))
                 print('updated')
        except:e.append(i);pass
print(e)


import os
import warnings

sources = ["/kaggle/working/kaggle/working/submission/"]
failing = []

warnings.filterwarnings("ignore")

total_save = 0
score = 0
submission = "/kaggle/working/submission"

os.makedirs(submission, exist_ok=True)

for task_num in range(1, 401):
 
 path_out = f"{submission}/task{task_num:03d}.py" 

 task_src = b"#" * 1_000_000
 for source in sources:
  path_in  = f"{source}/task{task_num:03d}.py"
  if not os.path.exists(path_in):continue
  if path_in in failing:continue

  with open(path_in, "rb") as task_in:
   new_src = task_in.read()
   task_src = min([task_src, new_src], key=len)
 zipped_src = zip_src(task_src)
 improvement = len(task_src) - len(zipped_src)
 if improvement > 0:
  task_src = zipped_src
  total_save += improvement
  #print(task_num, improvement)
 score += max(1, 2500-len(task_src))
 with open(path_out, "wb") as task_out:
  task_out.write(task_src)

print(f"Saved {total_save}b with zlib")
print(f"Projected score: {score}")


import os

score = 0
for task_num in range(1,401):
    p1='/kaggle/working/task' + str(task_num).zfill(3) + '.py'
    p2='/kaggle/input/baselines/submission/task' + str(task_num).zfill(3) + '.py'
    s1 = open(p1,'rb').read()
    s2 = open(p2,'rb').read()
    if check(s1, task_num, valall=False) and os.path.getsize(p1)<os.path.getsize(p2):
        s = max([0.1,2500-os.path.getsize(p1)])
    else:
        open(p1,'wb').write(s2)
        s = max([0.1,2500-os.path.getsize(p2)])
    #print(task_num, 2500-s)
    score += s
print('Score:', score)


import zipfile

with zipfile.ZipFile(f"{submission}.zip", "w") as zipf:
 for task_num in range(1, 401):
  task_id = f"{task_num:03d}"
  src_path = f"{submission}/task{task_id}.py"
  if os.path.exists(src_path):
   zipf.write(src_path, arcname=f"task{task_id}.py")


# Lots of opportunity to clean public scripts
# Enjoy!

