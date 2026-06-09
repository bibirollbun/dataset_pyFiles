from typing import Optional
from dataclasses import dataclass, asdict

max_output_tokens = 1024
@dataclass
class Config:
    # base model
    model_path: str = '/kaggle/input/barc0llama-3.1-arc-potpourri-induction-8b/transformers/default/2'
    load_in_4bit: bool = True
    max_seq_length: int = 8650 + max_output_tokens
    grid_encoder: str = 'ColorNameEncoder()'
    gpu_memory_utilization: float = 0.75 # 0.75 so far has not given OOM errors
    # LoRA
    lora_r: int = 16
    use_rslora: bool = True
    # dataset
    #dataset_path: str = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json'
    dataset_path: str = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
    max_tasks_per_partition: Optional[int] = None
    # search and learn hyperparameters
    task_group_size: int = 30 # how many tasks to group when searching and learning, 1 means each task is independent of the others, -1 means all tasks are grouped together
    use_data_augmentation: bool = True
    max_epochs: int = 4
    inference_batch_size: int = 8
    initial_predictions: int = 32
    predictions_per_epoch: int = 32
    training_batch_size: int = 1
    n_jobs: int = -1
    timeout_duration: int = 1 # seconds for code execution timeout
    # training hyperparameters
    learning_rate: float = 1e-5
    lr_scheduler_type: str = 'constant_with_warmup'
    train_max_seq_length: int = 8650 + max_output_tokens
    # sampling hyperparameters
    temperature: float = 1.0
    top_p: float = 0.95
    max_output_tokens: int = max_output_tokens
    # other
    log_to_wandb: bool = False
    n_gpus: int = 4
    dry_run: bool = True

cfg = Config()
cfg


import subprocess
def get_gpu_count():
    result = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, text=True)
    gpu_list = result.stdout.strip().split('\n')
    print(result.stdout)
    return len(gpu_list)
# Avoid failing submissions by checking the number of gpus
assert cfg.n_gpus == get_gpu_count()

import os
if cfg.dataset_path == '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json' and not os.getenv('KAGGLE_IS_COMPETITION_RERUN') and cfg.dry_run:
    print('This is a dry run, saving a dummy submission file')
    import json
    with open('submission.json', 'w') as f:
        json.dump(dict(dry_run=True), f)
    import sys
    sys.exit(0)


!rm -r *
!cp -r /kaggle/input/arc25-source-code/* .
!cp scripts/validate_submission.py .
! mkdir environments
# create a separate "environment" for each run, because otherwise unsloth is raising weird exceptions at startup
for i in range(4):
    ! mkdir environments/env{i}
    ! cp -r arc25 environments/env{i}
    ! cp scripts/search_and_learn_with_unsloth.py environments/env{i}


%%time
import importlib.util
if importlib.util.find_spec('unsloth') is None:
    !pip install -r requirements.txt --no-index --find-links=/kaggle/input/arc25-pip-wheels
else:
    print('Installation was already done, skipping it.')

import json
import glob

from arc25.resource_monitor import ResourceMonitor
from arc25.utils import create_dataset_partitions, load_json, write_json
from arc25.metrics import aggregate_metrics
from arc25.submission import create_submission, evaluate_submission

monitor = ResourceMonitor(interval=1)
monitor.start()


%%time
def create_string_conf(cfg, exclude_keys=['n_gpus', 'dry_run', 'dataset_path', 'max_tasks_per_partition']):
    str_conf = ''
    for key, value in asdict(cfg).items():
        if key in exclude_keys:
            continue
        key = key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                str_conf += f'--{key} '
            else:
                str_conf += f'--no-{key} '
        elif isinstance(value, str):
            str_conf += f'--{key} "{value}" '
        else:
            str_conf += f'--{key} {value} '
    return str_conf.strip()

create_dataset_partitions(cfg.dataset_path, max_tasks_per_partition=cfg.max_tasks_per_partition)
# Run a separate instance of search_and_learn_with_unsloth.py on each of the 4 GPUs
command = ''
for idx in range(4):
    python_command = f'python search_and_learn_with_unsloth.py {create_string_conf(cfg)}' \
    f' --dataset-path /kaggle/working/dataset/partition{idx}_challenges.json' \
    f' --output_dir /kaggle/working/outputs/partition_{idx}' 
    command += f'sleep {idx*10}; ' # add sleep at start to avoid all the scripts accessing the files at the same time
    if idx > 0:
        python_command += f' > /kaggle/working/logs/gpu{idx}.log 2>&1'
    command += f'cd /kaggle/working/environments/env{idx}; CUDA_VISIBLE_DEVICES={idx} {python_command} & '
command += 'wait'
print(command.replace(';', ';\n'))

! rm -r logs
! mkdir logs
! {command}


results = dict()
for filepath in glob.glob('outputs/*/results.json.gz'):
    results.update(load_json(filepath))
results = {key: results[key] for key in sorted(results)}
aggregate_metrics(results)


submission = create_submission(results, dataset=load_json(cfg.dataset_path))
write_json(submission, 'submission.json')
!python validate_submission.py --submission-path submission.json --dataset-path {cfg.dataset_path}

solutions_filepath = cfg.dataset_path.replace('challenges', 'solutions')
if os.path.exists(solutions_filepath):
    solutions = load_json(solutions_filepath)
    evaluate_submission(solutions, submission);


# cleaning
! rm -r environments dataset arc25 requirements.txt scripts
! rm -r validate_submission.py
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    ! rm -r logs outputs
! ls


monitor.stop()
monitor.plot()

