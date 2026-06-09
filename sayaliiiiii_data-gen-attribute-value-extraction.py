import os
import gc
import time
import warnings

import pandas as pd
import re
import torch
import json
from tqdm import tqdm


from vllm import LLM, SamplingParams
import ctypes

warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clean_memory(deep=False):
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

llm_model_path = '/kaggle/input/deepseek-r1/transformers/qwen-qwq-32b-awq/1'
# llm_tokenizer = "/kaggle/working/gemma-tok"

llm = LLM(
    llm_model_path,
    #dtype="half",                -> Changed this
    #max_num_seqs=128,            -> Changed this       
    trust_remote_code=True,     
    tensor_parallel_size=4,      
    gpu_memory_utilization=0.90, 
)


tokenizer  = llm.get_tokenizer()


N_SAMPLES = 1
BEST_OF = 1


sampling_params = SamplingParams(
    temperature=0.8,
    top_k=-1,
    top_p=0.95,
    n=N_SAMPLES,
    best_of=BEST_OF,
    skip_special_tokens=True,                 
    max_tokens=2000,             # Sets a very high limit for token generation to handle longer outputs.
    # stop=["\n\nNext action"],
)


!cp -r /kaggle/input/ecinstruct/Ecinstruct /kaggle/working/


from datasets import load_dataset, load_from_disk

dataset = load_from_disk("Ecinstruct")


filtered_dataset = dataset['train'].filter(lambda example: example['task'] == 'Attribute_Value_Extraction')


len(filtered_dataset)
filtered_dataset[1]


sys_prompt = (
    "You are a thoughtful and precise language model tasked with generating synthetic reasoning data "
    "for an attribute value extraction task. Each data point includes product information such as the product title, description, brand, features, and price, along with one or more target attributes. "
    "Given the input and the expected final output, your goal is to produce a logically sound Chain-of-Thought (CoT) explanation that details how the value for each attribute was identified, or why it could not be determined. "
    "For each target attribute, carefully examine all available product fields and extract the corresponding value, along with the specific source field (e.g., 'product title', 'product description', etc.). "
    "If the value is not explicitly mentioned or cannot be inferred, set both the value and the source to 'None'. "
    "The reasoning should clearly explain how the product information supports each extracted value and its source, or justify why no value could be extracted. "
    "Ensure your explanation is relevant, logically structured, and demonstrates critical attention to the input. "
    "Present your explanation under the heading: Step-by-Step Explanation: "
    "Then, conclude your response with the final extracted list using the exact format: \\boxed{<output_json>}."
)
print(sys_prompt)


from functools import partial

def apply_template(example, tokenizer, sys_prompt):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"QUERY:\n{example['input']}\nOPTIONS:\n{example['options']}\nCORRECT ANSWER:\n{example['output']}"}
    ]
    text = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return {"formatted_text": text}


format_fn = partial(apply_template, tokenizer=tokenizer, sys_prompt=sys_prompt)


formatted_dataset = filtered_dataset.map(format_fn, batched=False)


list_of_texts = formatted_dataset['formatted_text']


BATCH_SIZE = 100
updated_rows = {}
all_gens = []


for i in tqdm(range(27000, len(list_of_texts), BATCH_SIZE), desc="Generating responses"):
    batch_texts = list_of_texts[i:i+BATCH_SIZE]
    
    request_output = llm.generate(
        prompts=batch_texts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    gens = [request_output[j].outputs[0].text for j in range(len(batch_texts))]
    all_gens.extend(gens)


    batch_data = formatted_dataset[i:i+BATCH_SIZE]
    batch_data["generation"] = gens

    for k, v in batch_data.items():
        updated_rows.setdefault(k, []).extend(v)

    print(f"========ITERATION {i / BATCH_SIZE} / {len(list_of_texts) / BATCH_SIZE} DONE===========")

    with open("backup.json", "w") as f:
        json.dump(updated_rows, f, indent=2)

with open("final_output.json", "w") as f:
    json.dump(updated_rows, f, indent=2)

