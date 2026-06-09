from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json


df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")


def create_prompt(text, answer):
    user_prompt = f"""You are an expert math problem classifier.

    Task: Classify the following math problem into one of the categories below.
    Instruction: Respond with the correct number (0-7) only. Do NOT include any explanation.

    Math Problem:
    {text}

    Categories:
    0. Algebra  
    1. Geometry and Trigonometry  
    2. Calculus and Analysis  
    3. Probability and Statistics  
    4. Number Theory  
    5. Combinatorics and Discrete Math  
    6. Linear Algebra  
    7. Abstract Algebra and Topology

    Answer (number only): """
    
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": str(answer)}
    ]
    
    data_dict = {"conversations": messages}

#     tokenized = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
# )
    return data_dict


all_tokenized = []
for i in df.iterrows():
    all_tokenized.append(create_prompt(i[1]['Question'], i[1]['label']))


train_df, valid_df = train_test_split(all_tokenized, test_size=0.005, random_state = 2015)


output_filename = '/kaggle/input/classification-of-math-problems-by-kasut-academy/train_data.jsonl'
with open(output_filename, 'w', encoding='utf-8') as f:
    for entry in train_df:
        json_string = json.dumps(entry, ensure_ascii=False)
        f.write(json_string + '\n')


output_filename = '/kaggle/input/classification-of-math-problems-by-kasut-academy/valid_data.jsonl'
with open(output_filename, 'w', encoding='utf-8') as f:
    for entry in valid_df:
        json_string = json.dumps(entry, ensure_ascii=False)
        f.write(json_string + '\n')


CUDA_VISIBLE_DEVICES=0,1,2,3 axolotl /kaggle/input/kacallengers-3rd-train-axolotl-code/axolotl_math-checkpoint.yml


axolotl merge-lora /kaggle/input/kacallengers-3rd-train-axolotl-code/axolotl_math-checkpoint.yml --lora-model-dir="/workspace/math/model/checkpoint-94"


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig


model_id = "/workspace/math/model/Qwen2.5-32B-IT-1e3-FT-merge"
quant_path = "/workspace/math/model/Qwen2.5-32B-IT-1e3-FT-GPTQ"

quant_config = QuantizeConfig(bits=4, group_size=128, v2 = True)

model = GPTQModel.load(model_id, quant_config)


def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            jsons = json.loads(line.strip())
            tokenized = tokenizer.apply_chat_template(jsons['conversations'], tokenize=False)
            id_mask = tokenizer(tokenized)
            data.append(id_mask)
    return data


quantization_dataset = read_jsonl_file("/kaggle/input/classification-of-math-problems-by-kasut-academy/train_data.jsonl")


model.quantize(quantization_dataset[:1280], batch_size=1, 
            #    auto_gc = False, 
            #    buffered_fwd = True
               )

model.save(quant_path)


quant_path = "/workspace/math/model/Qwen2.5-32B-IT-1e3-FT-GPTQ"

from huggingface_hub import HfApi, create_repo
api = HfApi(token = {your_huggingface_api_key})
repo_id = "qwertist/Qwen2.5-32B-IT-1e3-FT-GPTQ"
create_repo(repo_id=repo_id, repo_type="model", token=api.token)
api.upload_folder(
    folder_path= quant_path,
    repo_id=repo_id,
    repo_type="model"
)

