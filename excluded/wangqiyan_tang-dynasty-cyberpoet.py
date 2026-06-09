!pip install python==3.10.12
!pip install deepspeed
!git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
%cd LLaMA-Factory
!pip install -e ".[torch,metrics]"
%cd ..



!pwd


root1 = "/kaggle/input/gemma2-poems-fine-tuning"
root2 = "/kaggle/working/LLaMA-Factory"

use_all = 0
if use_all == 1:
    data_str = "_all"
else:
    data_str = ""

model0 = "/kaggle/input/my_gemma2_2b_it/transformers/default/1"

!cp "{root1}/gemma_finetune_only_poems_data2{data_str}.json" "{root2}/data/gemma_finetune_only_poems_data2.json"
!cp "{root1}/gemma_finetune_poems_data_1{data_str}.json" "{root2}/data/gemma_finetune_poems_data_1.json"
!cp "{root1}/dataset_info.json" "{root2}/data/dataset_info.json"
!cp "{root1}/gemma2_lora_pretrain.yaml" "{root2}/examples/train_lora/gemma2_lora_pretrain.yaml"
!cp "{root1}/gemma2_lora_sft_ds0.yaml" "{root2}/examples/train_lora/gemma2_lora_sft_ds0.yaml"
!cp "{root1}/gemma2_lora_pretrain_export.yaml" "{root2}/examples/merge_lora/gemma2_lora_pretrain_export.yaml"
!cp "{root1}/gemma2_lora_sft_export.yaml" "{root2}/examples/merge_lora/gemma2_lora_sft_export.yaml"

%cd /kaggle/working/LLaMA-Factory
!llamafactory-cli version


import yaml

def update_yaml_config(train_config, parameters_to_update):
    """
    Update parameters in a YAML configuration file.

    Args:
    - train_config (str): Path to the YAML configuration file.
    - parameters_to_update (dict): A dictionary where the keys are the parameter paths 
      (e.g., "model.layers") and the values are the new values to set.

    Returns:
    - None: The function updates the YAML file directly.
    """
    
    def update_nested_dict(data, key_path, value):
        """
        Update a nested dictionary with the specified key path and value.
        key_path: A string representing the path to the key, e.g., "model.layers".
        value: The new value to set.
        """
        keys = key_path.split(".")
        d = data
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value

    try:
        with open(train_config, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        # Update parameters in the YAML file
        for parameter_name, new_value in parameters_to_update.items():
            if "." in parameter_name:
                update_nested_dict(data, parameter_name, new_value)
            elif parameter_name in data:
                data[parameter_name] = new_value
            else:
                print(f"Warning: The parameter '{parameter_name}' does not exist in the YAML file.")

        # Print the updated content (optional)
        print(yaml.dump(data, default_flow_style=False, allow_unicode=True))

        with open(train_config, 'w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, default_flow_style=False, allow_unicode=True)

    except Exception as e:
        print(f"Error: {e}")

# Example usage
train_config = "examples/train_lora/gemma2_lora_pretrain.yaml"
parameters_to_update = {
### model
"model_name_or_path": model0,

### method
"stage": "pt",
"do_train": True,
"finetuning_type": "lora",
"lora_dropout": 0,
# "lora_rank": 8,
"lora_target": "all",
"deepspeed": "examples/deepspeed/ds_z0_config.json",

### dataset
"dataset": "gemma_finetune_only_poems_data",
"cutoff_len": 1024,
"max_samples": 400000,
"overwrite_cache": True,
"preprocessing_num_workers": 16,

### output
"output_dir": "saves/gemma2_9b_it/lora/pretrain",
"logging_steps": 10,
"save_steps": 100,
"plot_loss": True,
"overwrite_output_dir": True,

### train
"per_device_train_batch_size": 1,
"gradient_accumulation_steps": 8,
"learning_rate": 1.0e-4,
"num_train_epochs": 1.0,
"lr_scheduler_type": "cosine",
"warmup_ratio": 0.1,
"bf16": True,
"ddp_timeout": 180000000,

### eval
"val_size": 0.0001,
"per_device_eval_batch_size": 1,
"eval_strategy": "steps",
"eval_steps": 100
}
update_yaml_config(train_config, parameters_to_update)


!wandb disabled
!FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/gemma2_lora_pretrain.yaml


# set train config
train_config = "examples/merge_lora/gemma2_lora_pretrain_export.yaml"

parameters_to_update = {
### model
"model_name_or_path": model0,
"adapter_name_or_path": "saves/gemma2_9b_it/lora/pretrain",
"template": "gemma",
"finetuning_type": 'lora',

### export
"export_dir": "models/gemma2_lora_pretrain_poems",
"export_size": 2,
"export_device": "cpu",
"export_legacy_format": False
}
update_yaml_config(train_config, parameters_to_update)

!wandb disabled
!FORCE_TORCHRUN=1 llamafactory-cli export examples/merge_lora/gemma2_lora_pretrain_export.yaml


# set train config
train_config = "examples/train_lora/gemma2_lora_sft_ds0.yaml"
parameters_to_update = {
### model
"model_name_or_path": "models/gemma2_lora_pretrain_poems",
# adapter_name_or_path: /root/zzyuan/LLaMA-Factory-main/saves/gemma2_9b_it/lora/sft_afterpretrain_poems/checkpoint-711

### method
"stage": "sft",
"do_train": True,
"finetuning_type": "lora",
"lora_target": 'all',
"lora_dropout": 0,
# "lora_rank": 8,
"deepspeed": "examples/deepspeed/ds_z0_config.json",

### dataset
"dataset": 'dataset_gemma_poems_train_1',
"template": "gemma",
"cutoff_len": 6000,
"max_samples": 400000,
"overwrite_cache": True,
"preprocessing_num_workers": 16,

### output
"output_dir": "saves/gemma2_9b_it/lora/sft_afterpretrain_poems_gutishi",
"logging_steps": 10,
"save_steps": 200,
"plot_loss": True,
"overwrite_output_dir": True,

### train
"per_device_train_batch_size": 2,
"gradient_accumulation_steps": 2,
"learning_rate": 1.0e-4,
"num_train_epochs": 5.0,
"lr_scheduler_type": "cosine",
"warmup_ratio": 0.1,
"bf16": True,
"ddp_timeout": 180000000,

### eval
"val_size": 0.01,
"per_device_eval_batch_size": 1,
"eval_strategy": "steps",
"eval_steps": 200
}
update_yaml_config(train_config, parameters_to_update)

!wandb disabled
!FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/gemma2_lora_sft_ds0.yaml


# set train config
train_config = "examples/merge_lora/gemma2_lora_sft_export.yaml"
parameters_to_update = {
### model
"model_name_or_path": "models/gemma2_lora_pretrain_poems",
"adapter_name_or_path": "saves/gemma2_9b_it/lora/sft_afterpretrain_poems_gutishi",
"template": "gemma",
"finetuning_type": "lora",

### export
"export_dir": "models/gemma2_lora_pretrain_sft_poems",
"export_size": 2,
"export_device": "cpu",
"export_legacy_format": False
}
update_yaml_config(train_config, parameters_to_update)

!wandb disabled
!FORCE_TORCHRUN=1 llamafactory-cli export examples/merge_lora/gemma2_lora_sft_export.yaml


# !pip install vllm
# import subprocess

# # Start the server and let it run in the background
# process = subprocess.Popen([
#     'python', '-m', 'vllm.entrypoints.openai.api_server',
#     '--model', 'models/gemma2_lora_pretrain_sft_poems',
#     '--port', '8001',
#     '--trust-remote-code',
#     '--max-model-len', '4096'
# ])

# # Wait for 100 seconds for the server to be set up, then continue running the other code
# import time 
# time.sleep(100)
# print(f"Is API still running? {'Yes' if process.poll() is None else 'No'}")



# import json
# from openai import OpenAI
# import os

# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:8005/v1"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

# messages = [
#                 {"role": "user", "content": "下面我给定几个意象和情感词，请你根据它们作一首中国古代古体诗。\n意象：屏风、怀素踪、尘色、墨痕、怪石、秋涧、寒藤、古松、水畔、龙。\n情感词：惊喜、爱惜、惋惜、赞叹、珍爱。\n下面请你作诗。"}
# ]

# # Generate 5 responses
# completion = client.chat.completions.create(
#     model="models/gemma2_lora_pretrain_sft_poems",
#     messages=messages,
#     max_tokens=4000,
#     temperature=0.8,
#     timeout=200,
#     n=5
# )
# responses = [choice.message.content for choice in completion.choices]
# for res in responses:
#     print(res)



!curl -fsSL https://ollama.com/install.sh | sh


!pip install lightrag-hku
!pip install aioboto3
!pip install ollama
!pip install openai
!pip install nano_vectordb
!pip install nest-asyncio


import subprocess

process2 = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

print("`ollama serve` is now running in the background.")
print(f"Is `ollama serve` still running? {'Yes' if process2.poll() is None else 'No'}")


!ollama pull nomic-embed-text


import subprocess

file_name = "gemma2.Modelfile"
content = "FROM ./models/gemma2_lora_pretrain_sft_poems"

with open(file_name, "w") as file:
    file.write(content)

!ollama create gemma2 -f gemma2.Modelfile

process3 = subprocess.Popen(
    ["ollama", "run", "gemma2"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    # start_new_session=True
)

print("`ollama run gemma2` is now running in the background as a separate process.")


print(f"Is `ollama serve` still running? {'Yes' if process2.poll() is None else 'No'}")
print(f"Is `ollama run gemma2` still running? {'Yes' if process3.poll() is None else 'No'}")


# import asyncio
# import os
# from lightrag import LightRAG, QueryParam
# from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete


# async def insert_file(rag, file_path):
#     print(f"Reading file: {os.path.basename(file_path)}")
#     with open(file_path, 'r') as file:
#         content = file.read()
#     await rag.insert(content)

# async def process_files(rag, folder_path):
#     files = os.listdir(folder_path)
#     tasks = []
#     for file_name in files[:10]:
#         if file_name.endswith('.md'):
#             file_path = os.path.join(folder_path, file_name)
#             tasks.append(insert_file(rag, file_path))
#     await asyncio.gather(*tasks)

# async def main():
#     # Specify the folder path
#     folder_path = '/kaggle/input/gemma2-fine-tuning/'
#     WORKING_DIR = "./RAG"
#     if not os.path.exists(WORKING_DIR):
#         os.makedirs(WORKING_DIR)
    
#     rag = LightRAG(
#         working_dir=WORKING_DIR,
#         llm_model_func=ollama_model_complete,
#         llm_model_name="gemma2",
#         llm_model_max_async=4,
#         llm_model_max_token_size=32768,
#         llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
#         embedding_func=EmbeddingFunc(
#             embedding_dim=768,
#             max_token_size=8192,
#             func=lambda texts: ollama_embedding(
#                 texts, embed_model="nomic-embed-text", host="http://localhost:11434"
#             ),
#         ),
#     )
#     await process_files(rag, folder_path)

# # Instead of asyncio.run(main()), use this:
# def run_async_code():
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(main())

# task = asyncio.create_task(main())


import asyncio
import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
import nest_asyncio

nest_asyncio.apply()

WORKING_DIR = "./RAG"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="gemma2",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)

with open(root1 + f"/sampled_poems{data_str}.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())


import os
import shutil
import stat

source_folder = "/kaggle/input/gemma2-poems-fine-tuning/RAG_data"
target_folder = "./RAG_data"

os.makedirs(os.path.dirname(target_folder), exist_ok=True)

shutil.copytree(source_folder, target_folder, dirs_exist_ok=True)
print(f"Copied {source_folder} to {target_folder}.")

def set_permissions(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            os.chmod(os.path.join(root, dir_name), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        for file_name in files:
            os.chmod(os.path.join(root, file_name), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    os.chmod(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

set_permissions(target_folder)
print(f"Set full permissions for {target_folder}.")



question = "下面我给定几个意象和情感词，请你根据它们作一首中国古代古体诗。\n意象:月、红豆。\n情感词:怀念、珍重、柔情。\n下面请你作诗。"

WORKING_DIR = "./RAG_data"

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="gemma2",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)

print(
    rag.query(question, param=QueryParam(mode="naive"))
)
# print(
#     rag.query(question, param=QueryParam(mode="local"))
# )
# print(
#     rag.query(question, param=QueryParam(mode="global"))
# )
# print(
#     rag.query(question, param=QueryParam(mode="hybrid"))
# )


%cd ..

import os
import shutil

current_dir = os.getcwd()
source_dir = os.path.join(current_dir, "LLaMA-Factory")
folders_to_move = ["RAG", "models", "RAG_data"]

for folder in folders_to_move:
    source_folder = os.path.join(source_dir, folder)
    target_folder = os.path.join(current_dir, folder)
    if os.path.exists(source_folder):
        if os.path.exists(target_folder):
            shutil.rmtree(target_folder)
        shutil.move(source_folder, target_folder)
        print(f"Moved {folder} to {current_dir}.")
    else:
        print(f"Folder {folder} not found in {source_dir}.")

if os.path.exists(source_dir):
    shutil.rmtree(source_dir)
    print(f"Deleted folder {source_dir}.")
else:
    print(f"Folder {source_dir} does not exist.")


