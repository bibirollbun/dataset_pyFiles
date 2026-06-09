from code_golf_utils.code_golf_utils import load_examples, show_examples, show_legend, verify_program
from copy import deepcopy

TASK_NUM = 1

examples = load_examples(task_num=TASK_NUM)
show_examples(examples['train'])
show_legend()


inp, out = examples['train'][0]['input'], examples['train'][0]['output']

def print_matrix(inp):
    x = [str(n) for n in range(len(inp[0]))]
    print(f'    {"  ".join(x)}')
    for y,r in enumerate(inp):
        print(f'{y:2}', r)

show_examples(examples['train'][:1])

print('Input')
print_matrix(inp)

print('\nOutput')
print_matrix(out)


def p(inp):
    out = []
    for y in range(9):
        row = []
        for x in range(9):
            z1 = inp[y // 3][x // 3]
            z2 = inp[y % 3][x % 3]
            z = z1 & z2
            row.append(z)
        out.append(row)
    return out


print('p(Input)')
print_matrix(p(inp))

print('\nOutput')
print_matrix(out)

print('\nSuccess!' if p(inp) == out else 'Failure :(')


%%writefile task.py
def p(inp):
    out = []
    for y in range(9):
        row = []
        for x in range(9):
            z1 = inp[y // 3][x // 3]
            z2 = inp[y % 3][x % 3]
            z = z1 & z2
            row.append(z)
        out.append(row)
    return out


verify_program(task_num=TASK_NUM, examples=examples)


!mv task.py task001.py
!zip submission.zip task001.py
!ls -alh submission.zip




