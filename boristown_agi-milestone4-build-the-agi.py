import json, os
from inertia_ray import (
    build_inertia_dataset,
    train_inertia_model,
    infer_inertia_sequence,
)
from visualization import animate_history

DATA_ROOT = '/kaggle/input/arc-agi-2'
TASK_ID_LIST = ['da515329','142ca369']


def load_pairs(task_id):
    path = os.path.join(DATA_ROOT, 'evaluation', f'{task_id}.json')
    with open(path) as f:
        task = json.load(f)
    return task['train'], task.get('test', [])



from IPython.display import HTML, display
for task_id in TASK_ID_LIST:
    train_pairs, test_pairs = load_pairs(task_id)
    dataset = build_inertia_dataset(train_pairs)
    model, log = train_inertia_model(dataset)
    print('accuracy:', log['acc'][-1])
    for i, pair in enumerate(test_pairs):
        _, frames = infer_inertia_sequence(model, [row[:] for row in pair['input']])
        html_anim = animate_history(frames, title=f'{task_id} test {i}')
        display(html_anim)

