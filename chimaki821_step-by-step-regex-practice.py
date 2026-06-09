# utils

import sys
sys.path.append('/kaggle/input/google-code-golf-2025/code_golf_utils')

from code_golf_utils import *

def _safe_run(func, grid):
  """Run program(grid) and coerce the result into a clean int grid."""
  x = func(copy.deepcopy(grid))
  s = json.dumps(x).replace("true", "1").replace("false", "0")
  if re.search(r"[^0-9,\[\]\s\.]", s):
    raise ValueError(f"Invalid output from user code: {s[:200]}")
  return json.loads(s)

def show_judge(example, func):
  if plt is None:
    print("matplotlib is unavailable (plt=None).")
    return

  grid  = example["input"]
  label = example["output"]
  pred  = _safe_run(func, grid)
  ok    = np.array_equal(np.array(pred), np.array(label))
  bg_right = (200, 255, 200) if ok else (255, 200, 200)
  bg_left  = (255, 255, 255)

  examples = [
    {"input": grid, "output": label, "bg": bg_left,  "title": "Ground truth"},
    {"input": grid, "output": pred,  "bg": bg_right, "title": f"Prediction ({'correct' if ok else 'wrong'})"},
  ]

  # 並列に2つの Axes を作成
  fig, axes = plt.subplots(1, 2, figsize=(10, 5))
  if not isinstance(axes, (list, np.ndarray)):
    axes = [axes]

  for ax, ex in zip(axes, examples):
    g, o = ex["input"], ex["output"]
    bg = ex["bg"]
    title = ex["title"]

    # キャンバスサイズ算出
    width = len(g[0]) + 1 + len(o[0]) + 4
    height = max(len(g), len(o)) + 4
    image = [[bg for _ in range(width)] for _ in range(height)]

    # セル塗り
    offset = 1
    gw, ow = len(g[0]), len(o[0])
    for r, row in enumerate(g):
      for c, cell in enumerate(row):
        image[r + 2][offset + c + 1] = colors[cell]
    offset += gw + 1
    for r, row in enumerate(o):
      for c, cell in enumerate(row):
        image[r + 2][offset + c + 1] = colors[cell]
    offset += ow + 4

    # 描画
    ax.imshow(np.array(image))
    # 罫線
    offset = 1
    ax.hlines([r + 1.5 for r in range(len(g)+1)],
              xmin=offset+0.5, xmax=offset+len(g[0])+0.5, color="black")
    ax.vlines([offset + c + 0.5 for c in range(len(g[0])+1)],
              ymin=1.5, ymax=len(g)+1.5, color="black")
    offset += len(g[0]) + 1
    ax.hlines([r + 1.5 for r in range(len(o)+1)],
              xmin=offset+0.5, xmax=offset+len(o[0])+0.5, color="black")
    ax.vlines([offset + c + 0.5 for c in range(len(o[0])+1)],
              ymin=1.5, ymax=len(o)+1.5, color="black")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)

  plt.tight_layout()
  plt.show()



task_id = 168

examples = load_examples(task_id)
examples = examples['train'] + examples['test']

show_examples(examples)


sample_input = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 7, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

sample_output = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 7, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 7, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 7, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

sample_example = {'input': sample_input, 'output': sample_output}
show_examples([sample_example])


g=copy.deepcopy(sample_input)
print(str(g))


import re
p=lambda g:eval(re.sub('0(?=.{34}0(?=..7.{28}7, 7))','7',str(g)))


show_judge(sample_example, p)


import re
p=lambda g:eval(re.sub('0(?=(.{34}|.{69}|.{104}|.{139})0(?=..7.{28}7, 7))','7',str(g)))


show_judge(sample_example, p)


import re
p=lambda g:eval(re.sub('0(?=.{34}(.{35})*0(?=..7.{28}7, 7))','7',str(g)))


show_judge(sample_example, p)


import re

def p(g):
    for _ in'_'*4:
        g=eval(re.sub('0(?=.{34}(.{35})*0(?=..7.{28}7, 7))','7',str(g)))
        g=[r for r in zip(*g[::-1])]
    return g


show_judge(examples[0], p)


import re
p=lambda g,i=3:-i*g or p(eval(re.sub('0(?=.{34}(.{35})*0(?=..7.{28}7, 7))','7',str([*zip(*g[::-1])]))),i-1)

show_judge(examples[0], p)


import re
p=lambda g,i=3:-i*g or p(eval(re.sub(r'0(?=.{34}(.{35})*0(?=..([1-9]).{28}\2, \2))',r'\2',str([*zip(*g[::-1])]))),i-1)

show_judge(examples[0], p)
show_judge(examples[1], p)
show_judge(examples[2], p)


import re
p=lambda g,i=3:-i*g or p(eval(re.sub(r'(?=(.{35})+0..([^0]).{28}\2, \2)0','\\2',str([*zip(*g[::-1])]))),~-i)


task_id = 34

show_examples(load_examples(task_id)['train'])


task_id = 37

show_examples(load_examples(task_id)['train'])


task_id = 190

show_examples(load_examples(task_id)['train'])


task_id = 378

show_examples(load_examples(task_id)['train'])

