import subprocess
import sys
import shutil

source_dir = "/kaggle/input/cot-faithfulness-master/cot-faithfullness-master"
destination_dir = "/kaggle/working/cot-faithfulness-master"

shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
print("Copy to working directory")


!pip install package_name /kaggle/working/cot-faithfulness-master


from cot_faithfulness.main import get_parser, run_main
from cot_faithfulness.models.config import AppConfig
from cot_faithfulness.models import DatasetType
from argparse import Namespace
from pathlib import Path

config_filename = "config.yaml"
config = AppConfig(
    mmlu_dataset_type=DatasetType.TINY, # Reduced question dataset
    model="gpt-oss:20b",
    ollama_host="http://localhost:11434"
)
config.save(Path(config_filename))

parser = get_parser()

# Run no-hint answers
args = parser.parse_args(['--config-filename', config_filename, 'init'])
run_main(args)


final_results_filename = "/kaggle/input/cot-faithfulness-full-results/final_results.json"
args = parser.parse_args(['--config-filename', config_filename, 'analyze_mmlu', final_results_filename])
run_main(args)


args = parser.parse_args(['--config-filename', config_filename, 'analyze_evaluated_mmlu', final_results_filename])
run_main(args)


collect_data = False


if collect_data:
    import os
    import time
    """
    Referenced from: https://www.kaggle.com/code/perseus7/pilot-s-snare-deception-by-omission
    Download Ollama and gpt-oss:20b
    """
    
    # check if ollama is installed
    result = os.system("ollama list 2>/dev/null")
    
    if result:
        print("[INFO] - Downloading Ollama")
        result = os.system("curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null")
        if result == 0:
            print("[Success] - Ollama installed successfully!")
        else:
            print("[Warning] - Ollama installation had warnings but may still work")
    
    print("Starting Ollama server...")
    os.system(
        "nohup ollama serve > /tmp/ollama_serve_stdout.log 2>/tmp/ollama_serve_stderr.log &"
    )
    time.sleep(5)
    # Check if running
    running = os.system(
        "ps aux | grep -E 'ollama serve' | grep -v grep > /dev/null 2>&1"
    )
    if running == 0:
        print("[Success] - Ollama server is running!")
    else:
        print("[Error] - Ollama server failed to start. Check troubleshooting section.")
    
    print("[INFO] - Downloading gpt-oss:20b")
    download = os.system("ollama pull gpt-oss:20b 2> /dev/null")
    if download == 0:
        print("[Success] - Downloaded gpt-oss20b!")
    else:
        print("[Error] - Something went wrong. Probably failed to download gpt-oss:20b")


output_filename = "dataset.json"
# Run no-hint answers
if collect_data:
    args = parser.parse_args(['--config-filename', config_filename, 'answer_mmlu', '--output-filename', output_filename, '--inplace'])
    run_main(args)


# Run hinted answers
if collect_data:
    args = parser.parse_args(['--config-filename', config_filename, 'answer_mmlu', '--output-filename', output_filename, '--use-hint', '--inplace'])
    run_main(args)


# Evaluate hinted-answer CoT
if collect_data:  
    args = parser.parse_args(['--config-filename', config_filename, 'evaluate_mmlu', output_filename, '--inplace'])
    run_main(args)


!pip install devtools


if collect_data:
    from cot_faithfulness.models import AnswerDataset
    from devtools import debug
    
    with open(output_filename, 'r') as f:
        answers_dataset = AnswerDataset.model_validate_json(f.read())
    
    debug(answers_dataset)

