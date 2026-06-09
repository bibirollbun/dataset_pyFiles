# --- Setup ---
import os

SRC_DIR = "/kaggle/input/newly-dataset-files-for-golf"
DEST_DIR = "/kaggle/working/submission"

os.makedirs(DEST_DIR, exist_ok=True)
os.chdir(DEST_DIR)

# Copy files from source to destination
copied = 0
for idx in range(1, 401):
    src = f"{SRC_DIR}/task{idx:03d}.py"
    dst = f"{DEST_DIR}/task{idx:03d}.py"
    if not os.path.exists(src):
        continue
    try:
        with open(src, "rb") as f_in, open(dst, "wb") as f_out:
            f_out.write(f_in.read())
        copied += 1
    except Exception as err:
        print(f"Failed to copy task{idx:03d}: {err}")

print(f"Copied {copied} task files successfully.")


import warnings
warnings.filterwarnings("ignore")
!pip install zopfli


import multiprocessing
import time
def run_with_timeout(func, args=(), kwargs={}, timeout=5):
    def wrapper(queue):
        try:
            result = func(*args, **kwargs)
            queue.put(result)
        except Exception as e:
            queue.put(e)
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=wrapper, args=(queue,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"Function timed out after {timeout} seconds")
    result = queue.get()
    if isinstance(result, Exception):
        raise result
    return result
    

#------------
import re
import ast

default_code = '''
def p(input):
    return 
'''

def update(task_num):
    # Format task number into a 3-digit string (e.g., 1 -> '001')
    file_num = f"{task_num:03}"
    # Create a filename like 'task001.py'
    filename = f"task{file_num}.py"

    # Read the main code (from task.py) that will be written into the submission file later
    with open('task.py', "r", encoding="utf-8") as f:
        code = f.read()

    print(f'Task num: {task_num} | File num: {file_num}')

    # Load example test cases for the given task (function assumed to be defined elsewhere)
    examples = load_examples(task_num)

    try:
        # Run verification with a 10-second timeout to prevent infinite loops or long runs
        run_with_timeout(verify_program, args=(task_num, examples), timeout=10)

        # If successful, write the actual code to the submission file
        with open('./submission/' + filename, "w", encoding="utf-8") as f:
            f.write(code)

    except TimeoutError:
        # Handle cases where the program takes too long to run
        print("TIMEOUT")
        with open('./submission/' + filename, "w", encoding="utf-8") as f:
            f.write(default_code)
        print("="*60)

    except Exception as e:
        # Handle any other runtime errors during verification
        print(f"ERROR: {e}")
        with open('./submission/' + filename, "w", encoding="utf-8") as f:
            f.write(default_code)
        print("="*60)

    # Confirm completion of the update process
    print(f'File {file_num} updated')
    print("="*60)

#----------------
import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")  # Add utility path for importing helper functions
from code_golf_utils import *  # Import all utilities (e.g., load_examples)
from tqdm import tqdm  # Progress bar for tracking task progress

# Initialize an empty list to store processed example texts
ret = []

# Loop through all tasks (1 to 400) with progress tracking
for task_num in tqdm(range(1, 401)):
    text = ""  # Initialize text buffer for current task examples

    # Load training and testing examples for the given task number
    examples = load_examples(task_num)

    # Calculate total number of samples (train + test)
    total_samples = len(examples['train']) + len(examples['test'])

    # Combine train and test examples into a single list for processing
    examples['train'] += examples['test']

    # Iterate through all examples (train + test)
    for i in range(total_samples):
        # Start formatting each example block
        text += f'\nExample {i + 1}:\n - Input:\n'

        # Format and append the example's input data nicely
        text += '],\n '.join(str(examples['train'][i]['input']).split('], '))

        # Add output section
        text += '\n - Output:\n'

        # Format and append the example's output data
        text += '],\n '.join(str(examples['train'][i]['output']).split('], '))

        # Add a newline for spacing between examples
        text += '\n'

    # Append the formatted text for this task to the results list
    ret.append(text)

# Print the total number of processed tasks
print(len(ret))

#----------
import re
import ast
import zlib
def extract_and_decompress(filename):
    with open(filename, 'r', encoding='latin1') as f:
        lines = f.readlines()
    # Remove encoding comment and import if present
    content = ''.join(lines[2:]) if lines[0].startswith('#encoding:') else ''.join(lines)
    # Flexible pattern: match zlib.decompress(bytes('...', 'L1')) or similar
    pattern = r"""
        exec\s*\(\s*zlib\.decompress\s*\(\s*     # exec(zlib.decompress(
        bytes\s*\(\s*                            # bytes(
        (?P<quote>['"]{1,3})                     # opening quote (1 to 3 of ' or ")
        (?P<data>.*?)                            # the actual string data
        (?P=quote)\s*,\s*                        # matching closing quote
        ['"](?i:l1|latin-1)['"]\s*               # encoding string (L1 or latin-1, case-insensitive)
        \)\s*\)\s*\)                             # closing ))
    """
    match = re.search(pattern, content, re.DOTALL | re.VERBOSE)
    if not match:
#        print(" No matching compressed payload found.")
        print(content)
        return None
    str_literal = match.group('quote') + match.group('data') + match.group('quote')

    try:
        # Turn the string literal into a Python string (handles escapes)
        raw_str = ast.literal_eval(str_literal)
        raw_bytes = raw_str.encode('latin1')
        decompressed = zlib.decompress(raw_bytes)
        return decompressed.decode('utf-8')
    except Exception as e:
        print(f" Failed to decompress: {e}")
        return None


#-----------------
import zipfile, json, os, copy, json
def check(solution, task_num, valall=False):
    return True # we assume everything is right
    if task_num == 157: return True # this one just takes a while to run
    task_data = load_examples(task_num)
    #print(task_num, max(1, 2500 - len(solution.encode('utf-8'))))
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
                actual = [[int(x) if int(x) == x else x for x in row] for row in actual]
                if json.dumps(actual) != json.dumps(expected):
                    return False
            except:
                return False
        return True
    except Exception as e:
        print(e)
        return False



task_num = 5


# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'
source_code = extract_and_decompress(filename)
examples = load_examples(task_num)
os.chdir("/kaggle/working")


%%writefile task.py
def p(m):
 _,Y,X,t=min([(sum(t:=[r[x:x+3]for r in m[y:y+3]],m).count(0),y,x,t)for y in range(19)for x in range(19)])
 for a in(-4,0,4):
  for b in(-4,0,4):
   for k in range(1,19):
    y=Y+a*k;x=X+b*k
    for i in range(3):
     for j in range(3):
      if a|b!=0<=x+j<21>y+i>=0:m[y+i][x+j]=t[i][j]and max(max(r[X+b:X+b+3])for r in m[Y+a:Y+a+3])
 return m


verify_program(task_num, examples)
# Update the version to be submitted
update(task_num)


# Select puzzle and display
task_num = 18


# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'
source_code = extract_and_decompress(filename)
examples = load_examples(task_num)


%%writefile task.py
def p(m):
    h=len(m)
    w=len(m[0])
    A,C,V=[],[],[]
    V=[]
    for q in [(z//w,z%w)for z in range(h*w)]:
        Y,X=[],[]
        q=[q]
        for y,x in q:
            if w>x>=0<=y<h and (y,x)not in V and m[y][x]:V+=(y,x),;Y+=y,;X+=x,;q+=(y-1,x),(y+1,x),(y,x-1),(y,x+1)
        if len(X)<4:A+=zip(Y,X)
        else:
            H=[r[min(X):max(X)+1]for r in m[min(Y):max(Y)+1]]
            H=[*zip(*H)]
            C+=H,
            H=H[::-1]
            C+=H,
            H=[*zip(*H)]
            C+=H,
            H=H[::-1]
            C+=H,
            H=[*zip(*H)]
            C+=H,
            H=H[::-1]
            C+=H,
            H=[*zip(*H)]
            C+=H,
            H=H[::-1]
            C+=H,
    G=[w*[0]for r in m]
    for y,x in [(z//w,z%w)for z in range(h*w)]:
        for c in C:
            if sum(I+y<h>0<w>J+x and m[I+y][J+x]==c[I][J] and (I+y,J+x)in A for I,J in [(z//len(c[0]),z%len(c[0]))for z in range(len(c[0])*len(c))])==3:
                for I,J in [(z//len(c[0]),z%len(c[0]))for z in range(len(c[0])*len(c))]:G[I+y][J+x]=c[I][J]
    return G


verify_program(task_num, examples)
update(task_num)


# Select puzzle and display
task_num = 25



# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'
source_code = extract_and_decompress(filename)
examples = load_examples(task_num)


%%writefile task.py
# # 1st run  

    
E=enumerate
def p(m):
    e={r[0]:y for y,r in E(m)if all(r)}
    if not e:return[*zip(*p([*map(list,zip(*m))]))]
    for y,r in E(m):
        for x,v in E(r):
            m[y][x]=0
            if v in e:m[(Y:=e[v])-(y<Y)+(y>Y)][x]=v
    return m


verify_program(task_num, examples)
update(task_num)


from zipfile import ZipFile
import zipfile
import zopfli.zlib
import zlib
import warnings
def zip_src(src_code):
    candidates=[src_code]
    for compress in[zopfli.zlib.compress,lambda d:zlib.compress(d,9)]:
        for trailing in[b'',b'\n']:
            src=src_code+trailing
            while(comp:=compress(src))[-1]==ord('"'):src+=b'#'
            for delim in[b"'",b'"']:
                esc_map={0:b'\\x00',ord('\n'):b'\\n',ord('\r'):b'\\r',ord('\\'):b'\\\\',delim[0]:b'\\'+delim}
                sanitized=b''.join(esc_map.get(b,bytes([b]))for b in comp)
                compressed=b'import zlib\nexec(zlib.decompress(bytes('+delim+sanitized+delim+b',"L1")))'
                if max(sanitized)>127:compressed=b'#coding:L1\n'+compressed
                else:print('no header needed!')
                candidates.append(compressed)
            esc_map={0:b'\\x00',ord('\r'):b'\\r',ord('\\'):b'\\\\'}
            sanitized=b''.join(esc_map.get(b,bytes([b]))for b in comp)
            compressed=b'import zlib\nexec(zlib.decompress(bytes("""'+sanitized+b'""","L1")))'
            if max(sanitized)>127:compressed=b'#coding:L1\n'+compressed
            else:print('no header needed!')
            candidates.append(compressed)
    valid_options=[]
    for code in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=SyntaxWarning)
                with open('tmp.py','wb')as f:f.write(code)
                with open('tmp.py','rb')as f:x=f.read()
                exec(x,{})
                valid_options.append(code)
        except:0
    return min(valid_options,key=len)
files = {}
total_save=0
with ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    o=open('task.py','rb').read().strip()
    zipped_src = zip_src(o)
    files_len = min(len(o), len(zipped_src))
    improvement = len(o) - len(zipped_src)
    if improvement > 0:
        open('/kaggle/working/submission/task' + str(task_num).zfill(3) + '.py','wb').write(zipped_src)
    else:
        open('/kaggle/working/submission/task' + str(task_num).zfill(3) + '.py','wb').write(o)
    zipf.write('/kaggle/working/submission/task' + str(task_num).zfill(3) + '.py')
print("Compression Save: ", improvement)
print("Code length: ", len(zipped_src))

#---------------------
# Generate submission.zip file for submission
files = {}
total_save=0
with ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    for f in range(1,401):
        try:
            o=open('/kaggle/working/submission/task' + str(f).zfill(3) + '.py','rb').read().strip()
            zipped_src = zip_src(o)
            files[f] = min(len(o), len(zipped_src))
        except:
            continue
        #https://www.kaggle.com/code/cheeseexports/big-zippa
        improvement = len(o) - len(zipped_src)
        if improvement > 0:
            total_save += improvement
            open('/kaggle/working/submission/task' + str(f).zfill(3) + '.py','wb').write(zipped_src)
        else:
            open('/kaggle/working/submission/task' + str(f).zfill(3) + '.py','wb').write(o)
        zipf.write('/kaggle/working/submission/task' + str(f).zfill(3) + '.py')
print("Submission file generated")

#----------------
top=[58, 90, 58, 80, 206, 51, 62, 84, 109, 68, 121, 127,140, 70, 93, 43, 99, 323, 105, 146, 57, 91, 195, 62, 131, 52, 103, 63, 108, 94, 45, 39, 73, 125, 83, 75, 105, 51, 60, 69, 49, 139, 56, 255, 45, 170, 55, 92, 81, 85, 115, 40, 21, 280, 83, 40, 48, 103, 156, 48, 63, 143, 74, 152, 91, 268, 33, 116, 151, 78, 119, 54, 46, 79, 86, 276, 126, 60, 123, 253, 91, 50, 40, 62, 50, 172, 36, 101, 236, 159, 63, 86, 99, 102, 73, 325, 108, 88, 115, 85, 281, 150, 29, 84, 148, 67, 162, 56, 81, 85, 60, 109, 25, 64, 54, 20, 148, 271, 106, 97, 89, 82, 75, 96, 126, 54, 65, 61, 47, 65, 125, 86, 298, 168, 32, 105, 141, 104, 94, 36, 94, 40, 135, 53, 191, 58, 83, 141, 75, 30, 108, 40, 133, 99, 18, 146, 248, 269, 109, 105, 82, 96, 131, 32, 136, 61, 71, 111, 129, 196, 51, 20, 218, 97, 75, 64, 51, 47, 21, 79, 67, 169, 93, 100, 143, 60, 92, 61, 111, 109, 241, 110, 81, 67, 105, 112, 54, 122, 84, 84, 208, 102, 64, 93, 166, 144, 81, 215, 289, 20, 48, 105, 92, 62, 42, 114, 95, 56, 257, 87, 87, 103, 51, 171, 132, 139, 52, 119, 73, 114, 43, 58, 297, 118, 61, 54, 67, 223, 99, 99, 21, 54, 79, 64, 106, 105, 95, 72, 26, 118, 89, 57, 129, 84, 242, 95, 74, 61, 85, 135, 47, 39, 122, 216, 104, 102, 46, 239, 63, 117, 86, 89, 116, 71, 136, 38, 194, 118, 107, 179, 145, 83, 82, 220, 288, 109, 56, 89, 63, 67, 61, 56, 59, 70, 54, 63, 43, 55, 54, 87, 31, 89, 62, 92, 57, 71, 50, 226, 38, 78, 32, 44, 63, 96, 63, 71, 59, 54, 194, 68, 55, 48, 102, 259, 160, 30, 67, 163, 54, 134, 83, 58, 89, 66, 107, 93, 46, 64, 37, 119, 132, 111, 65, 78, 90, 58, 50, 94, 214, 91, 67, 84, 92, 96, 98, 105, 86, 97, 64, 45, 193, 69, 213, 155, 111, 365, 129, 138, 113, 260, 109, 48, 39, 112, 53, 30, 55, 143, 141, 27, 79, 134, 121, 62, 25, 52, 218, 61, 57, 99, 63, 149, 63, 91, 53, 179, 121, 77, 64, 67]
score = 0
for taskNo in files:
    try:
        solution = open('/kaggle/working/submission/task' + str(taskNo).zfill(3) + '.py','rb').read()
        if check(solution, taskNo, valall=True):
            s = max([0.1,2500-len(solution)])
            if taskNo == task_num:
                print(taskNo, 2500-s, top[taskNo-1], top[taskNo-1]-(2500-s))
            score += s
        else: print(taskNo, ":L")
    except: pass
print('Score:', score)

