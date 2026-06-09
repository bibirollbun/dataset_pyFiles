!mkdir submission

# Version 1
#!cp /kaggle/input/400-task-with-smart-solution-search-verification/submission/*.py /kaggle/working/submission

# Version 2&3 (don't forget to check for update to get latest codes)
# !cp /kaggle/input/gcgc-playground/submission/*.py /kaggle/working/submission

# Version 4
!cp /kaggle/input/neuroips-4-of-some-lessons-learned/submission/*.py /kaggle/working/submission


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


import re
import ast

default_code = '''
def p(input):
    return 
'''

def update(task_num):
    file_num = f"{task_num:03}"
    filename = f"task{file_num}.py"

    with open('task.py', "r", encoding="utf-8") as f:
        code = f.read()
        
    print(f'Task num: {task_num} | File num: {file_num}')
    examples = load_examples(task_num)
    
    try:
        run_with_timeout(verify_program, args=(task_num, examples), timeout=10)  # timeout 10s
        with open('./submission/' + filename, "w", encoding="utf-8") as f:
            f.write(code)
    except TimeoutError:
        print("TIMEOUT")
        with open('./submission/' + filename, "w", encoding="utf-8") as f:
            f.write(default_code)
        print("="*60)
    except Exception as e:
        print(f"ERROR: {e}")
        with open('./submission/' + filename, "w", encoding="utf-8") as f:
            f.write(default_code)
        print("="*60)
    print(f'File {file_num} updated')
    print("="*60)


import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *

from tqdm import tqdm
ret = []
for task_num in tqdm(range(1, 401)):
    text = ""
    examples = load_examples(task_num)
    total_samples = len(examples['train']) + len(examples['test'])
    examples['train'] += examples['test']
    for i in range(total_samples):
        text += f'\nExample {i + 1}:\n - Input:\n'
        text += '],\n '.join(str(examples['train'][i]['input']).split('], '))
        text += '\n - Output:\n'
        text += '],\n '.join(str(examples['train'][i]['output']).split('], '))
        text += '\n'
    ret.append(text)
print(len(ret))


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
    
    # --- 从这里开始修改 ---
    if not match:
        # 如果没有找到匹配的压缩代码，就认为文件本身就是源代码
        return content.strip()  # <-- 关键改动：直接返回原始内容
    # --- 修改结束 ---

    str_literal = match.group('quote') + match.group('data') + match.group('quote')

    try:
        # Turn the string literal into a Python string (handles escapes)
        raw_str = ast.literal_eval(str_literal)
        raw_bytes = raw_str.encode('latin1')
        decompressed = zlib.decompress(raw_bytes)
        return decompressed.decode('utf-8')
    except Exception as e:
        print(f"❌ Failed to decompress: {e}")
        return None


#taylorsamarel/qwen2-5-32b-arc-local-score-32-solved-script
import zipfile, json, os, copy, json

def check(solution, task_num, valall=False):
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


# Select puzzle and display
task_num = 49

# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'

source_code = extract_and_decompress(filename)

if source_code:
    print(source_code)
    
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])

os.chdir("/kaggle/working")

# Display color legend
show_legend()


%%writefile task.py
p=lambda r:(i:=sum(r,[]),e:=min({*i}-{0},key=i.count),t:=[*zip(*[(y,x)for y,R in enumerate(r)for x,v in enumerate(R)if v==e])],[R[min(t[1]):max(t[1])+1]for R in r[min(t[0]):max(t[0])+1]])[-1]


verify_program(task_num, examples)

# Update the version to be submitted
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


# Select puzzle and display
task_num = 50

# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'

source_code = extract_and_decompress(filename)

if source_code:
    print(source_code)
    
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])

os.chdir("/kaggle/working")

# Display color legend
show_legend()


%%writefile task.py
def p(c):
 G=[*map(list,c)]
 for _ in[0,1]:
  for r in G:
   i=-1
   while 1:
    try:i=r.index(8,i+1);j=r.index(8,i+1);r[i+1:j]=[3]*(j-i-1)
    except:break
  G=[*map(list,zip(*G))]
 return G


verify_program(task_num, examples)

# Update the version to be submitted
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


# Select puzzle and display
task_num = 51

# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'

source_code = extract_and_decompress(filename)

if source_code:
    print(source_code)
    
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])

os.chdir("/kaggle/working")

# Display color legend
show_legend()


%%writefile task.py
def p(c):
 C=[*map(list,c)];l,f=len(C),len(c[0]);i=sum(C,[]);a=min({*i}-{0},key=i.count);m,n=divmod(i.index(a),f)
 for t,z in[(0,1),(1,0),(0,-1),(-1,0)]:
  d,e=m+t,n+z
  if not(0<=d<l and 0<=e<f and C[d][e]):
   k=1
   while 0<=m-k*t<l and 0<=n-k*z<f:
    if C[m-k*t][n-k*z]==0:C[m-k*t][n-k*z]=a
    k+=1
 return C


verify_program(task_num, examples)

# Update the version to be submitted
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


# Select puzzle and display
task_num = 52

# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'

source_code = extract_and_decompress(filename)

if source_code:
    print(source_code)
    
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])

os.chdir("/kaggle/working")

# Display color legend
show_legend()


%%writefile task.py
p=lambda j:[[(r==[r[0]]*3)*5]*3for r in j]


verify_program(task_num, examples)

# Update the version to be submitted
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


# Select puzzle and display
task_num = 54

# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'

source_code = extract_and_decompress(filename)

if source_code:
    print(source_code)
    
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])

os.chdir("/kaggle/working")

# Display color legend
show_legend()


%%writefile task.py
def p(b):
 c=[r[:]for r in b];f=b[0][0];R=range(30);n=[];d=[r[:]for r in b]
 for y in R:
  for x in R:
   if d[y][x]!=f:
    o,q=[],[ (y,x) ];d[y][x]=f
    for Y,X in q:
     o+=[(Y,X)]
     for i in range(9):
      dy,dx=i//3-1,i%3-1;ny,nx=Y+dy,X+dx
      if(dy or dx)and 30>ny>-1<nx<30 and d[ny][nx]!=f:d[ny][nx]=f;q+=[(ny,nx)]
    n+=[o]
 t=min((o for o in n if o[1:]),key=len)
 Y,X=zip(*t);cy,cx=(min(Y)+max(Y))//2,(min(X)+max(X))//2;A=b[cy][cx];S={(y-cy,x-cx):b[y][x]for y,x in t}
 for y,x in[(y,x)for y in R for x in R if b[y][x]==A and(y,x)not in t]:
  for(dy,dx),v in S.items():
   for k in range(1,30 if(dy==0 or dx==0)and(dy*2,dx*2)in S else 2):
    ny,nx=y+dy*k,x+dx*k
    if not(30>ny>-1<nx<30 and c[ny][nx]!=f):break
    c[ny][nx]=v
 for y,x in t:c[y][x]=f
 return c


verify_program(task_num, examples)

# Update the version to be submitted
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


# Select puzzle and display
task_num = 55

# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'

source_code = extract_and_decompress(filename)

if source_code:
    print(source_code)
    
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])

os.chdir("/kaggle/working")

# Display color legend
show_legend()


%%writefile task.py
p=lambda r,z=zip:(R:=[i for i,w in enumerate(r)if{*w}=={8}],C:=[i for i,w in enumerate(z(*r))if{*w}=={8}],[[v or{(-1,0):2,(0,-1):4,(0,0):6,(0,1):3,(1,0):1}.get(((y>R[1])-(y<R[0]),(x>C[1])-(x<C[0])),0)for x,v in enumerate(w)]for y,w in enumerate(r)])[-1]


verify_program(task_num, examples)

# Update the version to be submitted
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


# Select puzzle and display
task_num = 56

# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'

source_code = extract_and_decompress(filename)

if source_code:
    print(source_code)
    
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])

os.chdir("/kaggle/working")

# Display color legend
show_legend()


%%writefile task.py
def p(j):k=(j[0][0]>0)*4+(j[0][1]>0)*2+(j[0][2]>0);return[[6//(k-1)+(k==5)]]


verify_program(task_num, examples)

# Update the version to be submitted
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


# Select puzzle and display
task_num = 57

# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'

source_code = extract_and_decompress(filename)

if source_code:
    print(source_code)
    
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])

os.chdir("/kaggle/working")

# Display color legend
show_legend()


%%writefile task.py
p=lambda j:[list(r*2)for r in zip(*filter(any,zip(*filter(any,j))))]


verify_program(task_num, examples)

# Update the version to be submitted
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


# Select puzzle and display
task_num = 58

# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'

source_code = extract_and_decompress(filename)

if source_code:
    print(source_code)
    
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])

os.chdir("/kaggle/working")

# Display color legend
show_legend()


%%writefile task.py
def p(o):
 r=len(o);g=[[0]*r for _ in' '*r];c=-1;d=0;v=[1,1j,-1,-1j]
 for l in [r]+[k for k in range(r-1,0,-2)for _ in'  ']:
  for _ in range(l):
   c+=v[d%4]
   g[int(c.imag)][int(c.real)]=3
  d+=1
 return g


verify_program(task_num, examples)

# Update the version to be submitted
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


# Select puzzle and display
task_num = 59

# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'

source_code = extract_and_decompress(filename)

if source_code:
    print(source_code)
    
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])

os.chdir("/kaggle/working")

# Display color legend
show_legend()


%%writefile task.py
def p(g):
 D,c={},0
 for y,r in enumerate(g):
  for x,v in enumerate(r):
   if v%5:c=v;k=y//4,x//4;D[k]=D.get(k,0)+1
 M=max(D.values())if D else-1
 R=range(11)
 return[[5 if y%4==3 or x%4==3 else c if D.get((y//4,x//4))==M else 0 for x in R]for y in R]


verify_program(task_num, examples)

# Update the version to be submitted
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


# Select puzzle and display
task_num = 60

# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'

source_code = extract_and_decompress(filename)

if source_code:
    print(source_code)
    
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])

os.chdir("/kaggle/working")

# Display color legend
show_legend()


%%writefile task.py
p=lambda j:[w[0]and w[:1]*5+[5]+w[-1:]*5 or w for w in j]


verify_program(task_num, examples)

# Update the version to be submitted
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


# Select puzzle and display
task_num = 101

# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'

source_code = extract_and_decompress(filename)

if source_code:
    print(source_code)
    
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])

os.chdir("/kaggle/working")

# Display color legend
show_legend()


%%writefile task.py
def p(g):
	H=len(g);A=len(g[0]);D=range;Q=sum(g,[]).index(1);M=[Q];E={Q};R=1,0,-1,0,1
	while M:
		C=M.pop();F=C//A;G=C%A
		for S in D(4):
			N=F+R[S];O=G+R[S+1];P=N*A+O
			if-1<N<H and-1<O<A and g[N][O]and P not in E:E.add(P);M+=[P]
	T,U=divmod(min(E),A);X=[(B//A-T,B%A-U)for B in E if g[B//A][B%A]<2];V=[(B//A-T,B%A-U)for B in E if g[B//A][B%A]>1];W=min(V);I={B for B in D(H*A)if g[B//A][B%A]>1}-E
	for C in I:
		F=C//A;G=C%A
		if C-1 in I or C-A in I:continue
		J=B=1
		while G+J<A and g[F][G+J]>1:J+=1
		while F+B<H and g[F+B][G]>1:B+=1
		B=min(J,B);K=F-W[0]*B;L=G-W[1]*B;Y={F*A+G for(C,E)in V for F in D(K+C*B,K+C*B+B)for G in D(L+E*B,L+E*B+B)};Y<=I and[0<=K+C*B+F<H and 0<=L+E*B+G<A and g[K+C*B+F].__setitem__(L+E*B+G,1)for(C,E)in X for F in D(B)for G in D(B)]
	return g


verify_program(task_num, examples)

# Update the version to be submitted
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


# Select puzzle and display
task_num = 187

# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'

source_code = extract_and_decompress(filename)

if source_code:
    print(source_code)
    
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])

os.chdir("/kaggle/working")

# Display color legend
show_legend()


%%writefile task.py
def p(g):b=len(g[0]);t=sum(g,[]);S=range(len(t));f=lambda n:t[n:n+1]==[0]and(t.__setitem__(n,3),[f(n+k)for k in(-1,1,-b,b)]);[*map(f,[*S[:b],*S[-b:],*S[::b],*S[b-1::b]])];return[[x or 2for x in t[i:i+b]]for i in S[::b]]


verify_program(task_num, examples)

# Update the version to be submitted
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


# Select puzzle and display
task_num = 255

# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'

source_code = extract_and_decompress(filename)

if source_code:
    print(source_code)
    
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])

os.chdir("/kaggle/working")

# Display color legend
show_legend()


%%writefile task.py
def p(f,R=range):
 g=len(f);s=[*map(list,f)];T=-1,0,1;P=lambda i,j:not f[i][j]and not any(g>i+d>=0 and g>j+e>=0 and f[i+d][j+e]for d in T for e in T if d|e);h=[0]*g;a=[]
 for y in R(g):
  for b in R(g):h[b]=P(y,b)and h[b]+1
  for M in R(g):
   if l:=h[M]:
    for z in R(M,g):
     if not(l:=min(l,h[z])):break
     a+=[(y-k+1,M,y,z,(z-M+1)*k)for k in R(1,l+1)]
 if not a:return s
 a.sort(key=lambda r:-r[4]);W=lambda r:r[3]-r[1]+1;H=lambda r:r[2]-r[0]+1
 j=next((r for r in a if W(r)>H(r)),a[0]);q=next((r for r in a if H(r)>W(r)),a[0]);i=j if W(j)>H(q)or(W(j)==H(q)and j[4]>=q[4])else q
 u=lambda x:[s[i].__setitem__(slice(x[1],x[3]+1),[3]*W(x))for i in R(x[0],x[2]+1)]
 v=W(i)<=H(i);b=lambda x:((x[3]+1==i[1]or x[1]-1==i[3])and x[0]<=i[2]and x[2]>=i[0])if v else((x[2]+1==i[0]or x[0]-1==i[2])and x[1]<=i[3]and x[3]>=i[1])
 c=lambda x:(not any(0<=r<g and 3 in s[r][x[1]:x[3]+1]for r in(x[0]-1,x[2]+1))if v else not any(0<=c<g and 3 in(s[r][c]for r in R(x[0],x[2]+1))for c in(x[1]-1,x[3]+1)))
 u(i)
 while 1:
  for z in a:
   if((W(z)if v else H(z))<5)or not(c(z)and b(z))or any(3 in s[r][z[1]:z[3]+1]for r in R(z[0],z[2]+1)):continue
   u(z);break
  else:break
 return s


verify_program(task_num, examples)

# Update the version to be submitted
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


# Select puzzle and display
task_num = 285

# Extracts the source code and display as output. The output code may then be copied and pasted into the following cell.
filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'

source_code = extract_and_decompress(filename)

if source_code:
    print(source_code)
    
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])

os.chdir("/kaggle/working")

# Display color legend
show_legend()


%%writefile task.py
def p(g):
 s=[*map(list,g)];D=-1,0,1;E=[(i,j)for i in D for j in D if i|j]
 for a in range(len(g)-1):
  for b in range(len(g[0])-1):
   if len(set(V:=g[a][b:b+2]+g[a+1][b:b+2]))<4:continue
   for t in 0,1,2,3:
    x=a+(t>1);y=b+(t&1)
    if(k:=V[t])and any(g[x+i][y+j]==k for i,j in E):
     A=a+a+1;B=b+b+1;q=[(x,y)];g[x][y]=0
     while q:
      x,y=q.pop()
      for u in 0,1,2,3:s[(x,A-x)[(u>>1)!=(x>a)]][(y,B-y)[(u&1)!=(y>b)]]=V[u]
      for i,j in E:
       u=x+i;v=y+j
       try:
        if g[u][v]==k:g[u][v]=0;q+=[(u,v)]
       except:0
     break
 return s


verify_program(task_num, examples)

# Update the version to be submitted
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


import os
import matplotlib.pyplot as plt  # <-- 关键改动 1：导入必要的库

# 假设您的环境中已经定义了这些函数:
# extract_and_decompress(filename)
# load_examples(task_num)
# show_examples(examples)
# show_legend()

def display_task_details(task_num):
    """
    接收一个任务编号，然后连续地打印出该任务的代码和对应的示例图。
    """
    print(f"======================================================")
    print(f"  显示任务 (Displaying Task): {str(task_num).zfill(3)}")
    print(f"======================================================\n")

    # --- 显示代码 ---
    filename = f'/kaggle/working/submission/task{str(task_num).zfill(3)}.py'
    
    # 关于任务45的说明：
    # 您之前的输出显示，对于task 45，extract_and_decompress打印了一行代码，但后续逻辑却认为失败。
    # 这意味着该函数可能返回了一个非空但无效的字符串。这里的逻辑会正确处理。
    try:
        source_code = extract_and_decompress(filename)
        if source_code and len(source_code) > 10: # 增加一个简单的检查，避免无效代码
            print("--- 任务代码 (Source Code) ---")
            print(source_code)
        else:
            print(f"未能找到或代码无效：任务 {task_num}")
            # 如果代码无效，我们可能不想显示示例，直接跳到下一个任务
            print("\n\n")
            return
    except Exception as e:
        print(f"处理任务 {task_num} 代码时出错: {e}")
        return

    # --- 显示示例图 ---
    print("\n--- 任务示例 (Examples - Train + Test) ---")
    try:
        examples = load_examples(task_num)
        show_examples(examples['train'] + examples['test'])
        plt.show()  # <-- 关键改动 2：强制在此处立即显示图像，并结束当前绘图
    except Exception as e:
        print(f"加载或显示任务 {task_num} 的示例时出错: {e}")

    print("\n\n")

# --- 主执行流程 ---
tasks_to_show = [5, 18, 27, 33, 34, 37, 39, 40, 43, 45, 47, 377]

for num in tasks_to_show:
    display_task_details(num)

# 全局性的操作放在最后
os.chdir("/kaggle/working")
show_legend()
# 在所有任务显示完后，也可能需要一个 plt.show() 来显示最后的 legend
plt.show()


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


#https://docs.google.com/spreadsheets/u/1/d/e/2PACX-1vQ7RUqwrtwRD2EJbgMRrccAHkwUQZgFe2fsROCR1WV5LA1naxL0pU2grjQpcWC2HU3chdGwIOUpeuoK/pubhtml#gid=1427788625
top=[58, 90, 58, 80, 206, 51, 62, 84, 109, 68, 121, 127, 140, 70, 93, 43, 99, 323, 105, 146, 57, 91, 195, 62, 131, 52, 103, 63, 108, 94, 45, 39, 73, 125, 83, 75, 105, 51, 60, 69, 49, 139, 56, 255, 45, 170, 55, 92, 81, 85, 115, 40, 21, 280, 83, 40, 48, 103, 156, 48, 63, 143, 74, 152, 91, 268, 33, 116, 151, 78, 119, 54, 46, 79, 86, 276, 126, 60, 123, 253, 91, 50, 40, 62, 50, 172, 36, 101, 236, 159, 63, 86, 99, 102, 73, 325, 108, 88, 115, 85, 281, 150, 29, 84, 148, 67, 162, 56, 81, 85, 60, 109, 25, 64, 54, 20, 148, 271, 106, 97, 89, 82, 75, 96, 126, 54, 65, 61, 47, 65, 125, 86, 298, 168, 32, 105, 141, 104, 94, 36, 94, 40, 135, 53, 191, 58, 83, 141, 75, 30, 108, 40, 133, 99, 18, 146, 248, 269, 109, 105, 82, 96, 131, 32, 136, 61, 71, 111, 129, 196, 51, 20, 218, 97, 75, 64, 51, 47, 21, 79, 67, 169, 93, 100, 143, 60, 92, 61, 111, 109, 241, 110, 81, 67, 105, 112, 54, 122, 84, 84, 208, 102, 64, 93, 166, 144, 81, 215, 289, 20, 48, 105, 92, 62, 42, 114, 95, 56, 257, 87, 87, 103, 51, 171, 132, 139, 52, 119, 73, 114, 43, 58, 297, 118, 61, 54, 67, 223, 99, 99, 21, 54, 79, 64, 106, 105, 95, 72, 26, 118, 89, 57, 129, 84, 242, 95, 74, 61, 85, 135, 47, 39, 122, 216, 104, 102, 46, 239, 63, 117, 86, 89, 116, 71, 136, 38, 194, 118, 107, 179, 145, 83, 82, 220, 288, 109, 56, 89, 63, 67, 61, 56, 59, 70, 54, 63, 43, 55, 54, 87, 31, 89, 62, 92, 57, 71, 50, 226, 38, 78, 32, 44, 63, 96, 63, 71, 59, 54, 194, 68, 55, 48, 102, 259, 160, 30, 67, 163, 54, 134, 83, 58, 89, 66, 107, 93, 46, 64, 37, 119, 132, 111, 65, 78, 90, 58, 50, 94, 214, 91, 67, 84, 92, 96, 98, 105, 86, 97, 64, 45, 193, 69, 213, 155, 111, 365, 129, 138, 113, 260, 109, 48, 39, 112, 53, 30, 55, 143, 141, 27, 79, 134, 121, 62, 25, 52, 218, 61, 57, 99, 63, 149, 63, 91, 53, 179, 121, 77, 64, 67]

score = 0
for taskNo in files:
    try:
        solution = open('/kaggle/working/submission/task' + str(taskNo).zfill(3) + '.py','rb').read()
        if check(solution, taskNo, valall=True):
            s = max([0.1,2500-len(solution)])
            print(taskNo, 2500-s, top[taskNo-1], top[taskNo-1]-(2500-s))
            score += s
        else: print(taskNo, ":L")
    except: pass
print('Score:', score)

