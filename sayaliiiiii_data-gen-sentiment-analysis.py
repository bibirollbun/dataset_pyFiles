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


filtered_dataset = dataset['train'].filter(lambda example: example['task'] == 'Sentiment_Analysis')


len(filtered_dataset)


sys_prompt = (
    "You are a highly analytical language model tasked with generating synthetic reasoning data for sentiment analysis. "
    "Each instance includes a user review and a list of sentiment labels, with one correct label ranging from 'very positive' to 'very negative'. "
    "Your objective is to analyze the review carefully and provide a detailed explanation of the reasoning that leads to the correct sentiment label. "
    "Use linguistic and contextual evidence from the review to support your judgment. "
    "Available sentiment options are labeled as: ['A: very positive', 'B: positive', 'C: neutral', 'D: negative', 'E: very negative']. "
    "Present your explanation under the heading: **Step-by-Step Explanation:** "
    "Then, conclude your response with the selected answer in the format: \\boxed{<option_letter>}."
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


for i in tqdm(range(14100 + 14000, len(list_of_texts), BATCH_SIZE), desc="Generating responses"):
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

    with open("backup.json", "w") as f:
        json.dump(updated_rows, f, indent=2)

    print(f"========STEP {i // BATCH_SIZE} / {len(list_of_texts) // BATCH_SIZE} DONE===========")

with open("final_output.json", "w") as f:
    json.dump(updated_rows, f, indent=2)




