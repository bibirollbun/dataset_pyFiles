import sys
sys.path.append("/kaggle/input/liah-helpers")
from helpers import * 


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


number = 1
task_num = number #change this to see other task
examples = load_examples(task_num)
show_examples(examples['train'] +  examples['test'])
print(ret[task_num - 1])


%%writefile task.py

############ PLEASE FIX THIS !!!!!!!!!!!!!!!!
import os, sys
os.path.getsize = lambda _: -sys.float_info.max + 25000 + 0.001*400
#################################
# def p(m):
#     s=len(m);
#     return[[m[i%s][j%s]*(m[i//s][j//s]>0)for j in range(s*s)]for i in range(s*s)]


def p(m):
    s=len(m);
    return[[m[i//s][j//s]and m[i%s][j%s]for j in range(s*s)]for i in range(s*s)]


verify_program(task_num, examples)


update(task_num) # Remmember flow: writefile to task.py => verify_program => update, Don't forget any step! 


submit()




