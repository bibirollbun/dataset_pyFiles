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


filtered_dataset = dataset['train'].filter(lambda example: example['task'] == 'Sequential_Recommendation')


len(filtered_dataset)


filtered_dataset[0]


# sys_prompt = (
#     "You are a thoughtful and precise language model tasked with generating synthetic reasoning data "
#     "for a multiple-choice classification task. Each data point consists of a query, a product title, a set of four options (A–D), "
#     "and a correct answer selected from those options. Your task is to generate a coherent and logically sound chain of thought reasoning path that leads to "
#     "the correct answer. This CoT reasoning must emulate how a human might justify their choice step by step. "
#     "The reasoning must be relevant to the query and product title, and should clearly explain why the correct option is best and why the other "
#     "options are less suitable. Ensure that your output is consistent, logically structured, and demonstrates critical thinking. Put the final answer in \boxed{}. Provide CoT reasoning steps under the **Step-by-Step Explanation:**"
# )
sys_prompt = (
    "You are a thoughtful and precise language model designed to generate synthetic reasoning data for a sequential recommendation task. "
    "Each data point includes a sequence of previously interacted products and a set of candidate options along with the correct option. "
    "Your goal is to identify which product the user is most likely to interact with next, based on their interaction history. "
    "Analyze the progression in product types, domains, and potential user intent to select the most contextually appropriate next item. "
    "The available options are labeled alphabetically, and your final answer must be the label (e.g., 'A', 'B', ...) corresponding to the most likely next item. "
    "Explain your reasoning under Step-by-Step Explanation: by considering how the option aligns (or doesn't) with the prior sequence. "
    "Conclude your response using the format: \boxed{<option_letter>}."
)



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


for i in tqdm(range(27700, len(list_of_texts), BATCH_SIZE), desc="Generating responses"):
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
    # for row, gen in zip(batch_data, gens):
    #     row["generation"] = gen
    #     updated_rows.append(row)

    for k, v in batch_data.items():
        updated_rows.setdefault(k,[]).extend(v)

    print(f"========ITERATION {i} / {len(list_of_texts)} DONE===========")

    with open("backup.json", "w") as f:
        json.dump(updated_rows, f, indent=2)



with open("final_output.json", "w") as f:
    json.dump(updated_rows, f, indent=2)




