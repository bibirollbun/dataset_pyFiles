import sys
sys.path.append("/kaggle/input/liah-helpers")
from helpers import * # IMPORT MY HELPERS FIRST


sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *  
show_legend()

print('Done import helpers!')



from tqdm import tqdm
import shutil

print('Init your submission...')
src_folder = "/kaggle/input/liah-helpers/submission"
dst_folder = "/kaggle/working/submission"
try:
    shutil.copytree(src_folder, dst_folder)
    print('Successfully create your submission folder!')
except Exception as e:
    print('Your submission existed!')

print('Load 400 problems...')
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


task_num = 1 #change this to see other task
examples = load_examples(task_num)
show_examples(examples['train'] +  examples['test'])
print(ret[task_num - 1])


%%writefile task.py
def p(m):
    s=len(m)
    return[[m[i%s][j%s]if m[i//s][j//s]else 0for j in range(s*s)]for i in range(s*s)]



verify_program(task_num, examples)


update(task_num) # Remmember flow: writefile to task.py => verify_program => update, Don't forget any step! 


task_num = 2
examples = load_examples(task_num)
show_examples(examples['train'] +  examples['test'])
print(ret[task_num - 1])


%%writefile task.py
from collections import deque

def p(a):
    if not a or not a[0]:
        return []

    r, c = len(a), len(a[0])
    q = deque([
        (i, j)
        for i in range(r)
        for j in range(c)
        if (i == 0 or j == 0 or i == r - 1 or j == c - 1) and a[i][j] == 0
    ])

    while q:
        x, y = q.popleft()
        a[x][y] = 1
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < r and 0 <= ny < c and a[nx][ny] == 0:
                q.append((nx, ny))
                a[nx][ny] = 1

    return [
        [4 if v == 0 else 0 if v == 1 else v for v in row]
        for row in a
    ]



verify_program(task_num, examples)


update(task_num)


submit()

