! git clone https://github.com/fchollet/ARC-AGI


import json
from tqdm import tqdm
from glob import glob
def build_hash(path):
    test_details = json.load(open(path,'r'))['test'][0]
    task_hash = ''.join([str(item) for sublist in test_details['input'] for item in sublist])
    task_hash = task_hash +"-" + ''.join([str(item) for sublist in test_details['output'] for item in sublist])
    return task_hash

comp_task_hashs = {}
all_task_paths = glob("/kaggle/input/google-code-golf-2025/*.json")
for task_path in tqdm(all_task_paths, total=len(all_task_paths)):
    task_hash = build_hash(task_path)
    task_id = task_path.split("/")[-1]
    comp_task_hashs[task_hash] = task_id


arc_agi_task_hashs = {}
all_arc_agi_1_train_paths = glob("ARC-AGI/data/training/*.json")
print(len(all_arc_agi_1_train_paths))
for task_path in tqdm(all_arc_agi_1_train_paths, total=len(all_arc_agi_1_train_paths)):
    task_hash = build_hash(task_path)
    task_id = task_path.split("/")[-1]
    arc_agi_task_hashs[task_hash] = task_id


task2agi = []
for task_hash, task_id in comp_task_hashs.items():
    task2agi.append({ "task_no": int(task_id.split(".json")[0].replace("task","")), "task_id": task_id, "agi_id": arc_agi_task_hashs[task_hash] })


import pandas as pd
task2agi_df = pd.DataFrame(task2agi).sort_values(['task_no']).reset_index(drop=True)
task2agi_df.to_csv("task2agi.csv", index=False)
task2agi_df


! rm -rf /kaggle/working/ARC-AGI




