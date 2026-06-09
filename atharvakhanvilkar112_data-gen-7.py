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


filtered_dataset = dataset['train'].filter(lambda example: example['task'] == 'Answer_Generation')
# filtered_dataset = filtered_dataset[15000:]


len(filtered_dataset)


sys_prompt = (
    "You are a thoughtful and precise language model tasked with generating synthetic reasoning data "
    "for a text generation task involving open-ended answer generation. Each data point includes a user question, a set of supporting documents (typically user reviews), "
    "and a final answer. Your goal is to reason through the content of the documents to generate a coherent, accurate, and well-justified response to the question. "
    "Given the input and the final answer, generate a logically sound chain of thought (CoT) explanation that walks through how the answer can be inferred from the document. "
    "The reasoning must demonstrate how key statements from the text support the final answer, highlighting any conflicting or confirming opinions if relevant. "
    "It should emulate how a human might synthesize multiple opinions and draw a conclusion. The answer lengths must match the target answer's length."
    "Ensure that your reasoning is relevant to the question, grounded in the provided document, logically structured, and demonstrates critical thinking. "
    "Present your explanation under **Step-by-Step Explanation:**, and include the final answer at the end using the format: \\boxed{<final_answer>}."
)



from functools import partial

def apply_template(example, tokenizer, sys_prompt):
    input_dict = eval(example['input'])
    qstn = input_dict['question']
    docs = input_dict['document']
    doc_context = "\n\n".join([f"document {i+1}\n{doc}" for i, doc in enumerate(docs)])
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"QUESTION:\n{example['input']}\nDOCUMENTS:\n{doc_context}\nCORRECT ANSWER:\n{example['output']}"}
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


BATCH_SIZE = 50
updated_rows = {}
all_gens = []


for i in tqdm(range(28000, len(list_of_texts), BATCH_SIZE), desc="Generating responses"):
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
        updated_rows.setdefault(k, []).extend(v)


    with open("backup.json", "w") as f:
        json.dump(updated_rows, f, indent=2)

    print(f"========ITERATION {i} / {len(list_of_texts)} DONE===========")


with open("final_output.json", "w") as f:
    json.dump(updated_rows, f, indent=2)






