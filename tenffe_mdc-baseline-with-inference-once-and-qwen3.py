!pip install PyMuPDF
!pip install vllm
!pip install triton==3.2.0
!pip install json_repair


import torch
import vllm


model_path = "/kaggle/input/qwen-3/transformers/4b-awq/1"
# model_path = "/kaggle/input/qwen-3/transformers/8b-awq/1"
# model_path = "/kaggle/input/qwen-3/transformers/32b-awq/1"

llm = vllm.LLM(
    model_path,
    # quantization='awq',
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=16384,
    disable_log_stats=True,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)
tokenizer = llm.get_tokenizer()


import os

# vLLM V1 does not currently accept logits processor so we need to disable it
# https://docs.vllm.ai/en/latest/getting_started/v1_user_guide.html#deprecated-features
# os.environ["VLLM_USE_V1"] = "0"

import re
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import pickle
import vllm
import torch

# Step 1: Read all PDFs and convert to text
# pdf_directory = "/kaggle/input/make-data-count-finding-data-references/test/PDF"

pdf_directory = "/kaggle/input/make-data-count-finding-data-references/test/PDF" \
                if os.getenv('KAGGLE_IS_COMPETITION_RERUN') \
                else "/kaggle/input/make-data-count-finding-data-references/train/PDF"

chunks = []
chunks2 = []
text_span_len = 512

re_doi = re.compile(r"10\.\d{4}")
re_gsr = re.compile(r"GSE\d+|SR[APRX]\d+|PRJ[NAED][A-Z]?\d+")
re_ipe = re.compile(r"IPR\d{6}|PF\d{5}|EMPIAR-\d{5}")
re_c = re.compile(r"CHEMBL\d+|CVCL_[A-Z0-9]{4}")
re_e = re.compile(r"ENS[A-Z]{0,6}[GT]\d{11}")
re_r = re.compile(r"N[MC]_\d+(?:\.\d+)?|rs\d+")
re_u = re.compile(r"(?:uniprot:)?(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])", re.IGNORECASE)
re_g = re.compile(r"EPI(?:_ISL_)?\d+")
re_p = re.compile(r"PXD\d{6}|SAM[ND]\d+|ERR\d+")

relist = [re_gsr, re_ipe, re_c, re_e, re_r, re_g, re_p]

ids = []

def remove_references_section(text):
    lines = text.split('\n')
    cut_index = -1
    
    # Look backwards from end of document
    for i in range(len(lines) - 1, max(0, int(len(lines) * 0.3)), -1):
        line = lines[i].strip()
        
        obvious_patterns = [
            # References patterns
            r'^REFERENCES?$',                    # All caps, alone
            r'^\d+\.?\s+REFERENCES?$',          # Numbered, all caps
            r'^\d+\.?\s+References?$',          # Numbered, title case
            r'^References?:$',                   # With colon
            
            # Bibliography patterns
            r'^BIBLIOGRAPHY$',                   # All caps, alone
            r'^\d+\.?\s+BIBLIOGRAPHY$',         # Numbered, all caps
            r'^\d+\.?\s+Bibliography$',         # Numbered, title case
            r'^Bibliography:$',                  # With colon
            
            # Other common patterns
            r'^Literature\s+Cited$',            # Literature Cited
            r'^Works\s+Cited$',                 # Works Cited
        ]
        
        if any(re.match(pattern, line, re.IGNORECASE) for pattern in obvious_patterns):
            # Double-check: look at following lines for citation patterns
            following_lines = lines[i+1:i+4]
            has_citations = False
            
            for follow_line in following_lines:
                if follow_line.strip():
                    # Check for obvious citation patterns
                    if (re.search(r'\(\d{4}\)', follow_line) or    # (2020)
                        re.search(r'\d{4}\.', follow_line) or       # 2020.
                        'doi:' in follow_line.lower() or           # DOI
                        ' et al' in follow_line.lower()):          # et al
                        has_citations = True
                        break
            
            # Only cut if we found citation-like content
            if has_citations or i >= len(lines) - 3:  # Or very near end
                cut_index = i
                break
    
    if cut_index != -1:
        return '\n'.join(lines[:cut_index]).strip()
    
    return text.strip()

for filename in tqdm(os.listdir(pdf_directory), total=len(os.listdir(pdf_directory))):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        
        # Extract article_id from filename
        article_id = filename.split(".pdf")[0]
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            page_text = page.get_text()
            text += page_text + "\n"
            
        doc.close()

        text = remove_references_section(text)

        doi_matches = re_doi.finditer(text)
        for match in doi_matches:
            if match.group() in article_id: continue
            chunk = text[max(0, match.start() - text_span_len): match.start() + text_span_len]
            chunks.append((article_id, chunk))

        for rr in relist:
            matches = rr.finditer(text)
            for match in matches:
                ids.append(match.group())
                chunk = text[max(0, match.start() - text_span_len): match.start() + text_span_len]
                chunks2.append((article_id, chunk))
                
print(len(chunks))
print(len(chunks2))


reason_prompt =  '''
You are an advanced AI reasoning assistant tasked with delivering a comprehensive analysis of a specific problem or question.  
Your goal is to outline your reasoning process in a structured and transparent manner, with each step reflecting a thorough examination of the issue at hand, culminating in a well-reasoned conclusion.
After the reasoning is complete, output the result with json style at the end of all generated text.

### Key Instructions:
1.  Conduct **at least 3 distinct reasoning steps**, each building on the previous one.
2.  **Acknowledge the limitations** inherent to AI, specifically what you can accurately assess and what you may struggle with.
3.  **Adopt multiple reasoning frameworks** to resolve the problem or derive conclusions, such as:
- **Deductive reasoning** (drawing specific conclusions from general principles)
- **Inductive reasoning** (deriving broader generalizations from specific observations)
- **Abductive reasoning** (choosing the best possible explanation for the given evidence)
- **Analogical reasoning** (solving problems through comparisons and analogies)
4.  **Critically analyze your reasoning** to identify potential flaws, biases, or gaps in logic.
5.  When reviewing, apply a **fundamentally different perspective or approach** to enhance your analysis.
6.  **Employ at least 2 distinct reasoning methods** to derive or verify the accuracy of your conclusions.
7.  **Incorporate relevant domain knowledge** and **best practices** where applicable, ensuring your reasoning aligns with established standards.
8.  **Quantify certainty levels** for each step and your final conclusion, where applicable.
9.  Consider potential **edge cases or exceptions** that could impact the outcome of your reasoning.
10.  Provide **clear justifications** for dismissing alternative hypotheses or solutions that arise during your analysis.
'''

task_prompt = '''
You are given a piece of academic text. Your task is to identify the single DOI citation string, if present.
Then normalize it into its full URL format: https://doi.org/...

Each object (paper and dataset) has a unique, persistent identifier to represent it. In this competition there will be two types:

dataset_id will be DOIs format or Accession IDs, the definition of dataset_id in detail is as follows:
1. the dataset identifier and citation type in the paper.
DOIs are used for all papers and some datasets. They take the following form: https://doi.org/[prefix]/[suffix]. Examples:
https://doi.org/10.1371/journal.pone.0303785
https://doi.org/10.5061/dryad.r6nq870
2. Accession IDs are used for some datasets. They vary in form by individual data repository where the data live. Examples:
"GSE12345" (Gene Expression Omnibus dataset)
“PDB 1Y2T” (Protein Data Bank dataset)
"E-MEXP-568" (ArrayExpress dataset)

the definition of type is as follows:
1. Primary - raw or processed data generated as part of this paper, specifically for this study
2. Secondary - raw or processed data derived or reused from existing records or published data
3. Missing - the DOIs or  Accession IDs are of type "Missing" unless specified otherwise in the context.
'''

example_prompt = '''
use markdown json style to get the result as the following examples, do not print the following example in the reasoning or the result:

```json
[
    {
        "dataset_id": "https://doi.org/10.1371/journal.pone.0303785",
        "type": "Primary"
    },
    {
        "dataset_id": "https://doi.org/10.1371/journal.pone.0303785",
        "type": "Secondary"
    },
    {
        "dataset_id": "GSE12345",
        "type": "Secondary"
    },
    {
        "dataset_id": "Missing",
        "type": "Missing, "
    }
    ...
]
```

the piece of academic text:
'''


# prefix_prompt = reason_prompt + task_prompt + example_prompt
# len(prefix_prompt)


# prefix_prompt = reason_prompt + task_prompt + example_prompt
prefix_prompt = task_prompt + example_prompt

article_id_list = []
prompts = []
for article_id, academic_text in chunks:
    messages = [
        {"role": "user", "content": prefix_prompt + academic_text}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompts.append(prompt)
    article_id_list.append(article_id)


for article_id, academic_text in chunks2:
    messages = [
        {"role": "user", "content": prefix_prompt + academic_text}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompts.append(prompt)
    article_id_list.append(article_id)


import json_repair
import re


def extract_json_blocks(text):
   pattern = r'```json\n(.*?)\n```'
   matches = re.findall(pattern, text, re.DOTALL)
   results = []
   for match in matches:
       json_str = match.strip()
       try:
           data = json_repair.loads(json_str)
           results.append(data)
       except json.JSONDecodeError:
           print(f"无效JSON：{json_str[:50]}...")  # 打印前50字符便于调试
   return results[-1]


def get_json_result(prompts, article_id_list, temperature=0.):
    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(
            seed=0,
            skip_special_tokens=True,
            max_tokens=8192,
            temperature=temperature
        ),
        use_tqdm=True
    )
    result_list = [output.outputs[0].text for output in outputs]

    article_id_json_result_list = []
    no_json_result_prompt_list = []
    no_json_result_article_id_list = []
    
    for index in range(len(prompts)):
        prompt, article_id, result = prompts[index], article_id_list[index], result_list[index]
        try:
            json_result = extract_json_blocks(result)
            article_id_json_result_list.append([article_id, json_result])
        except Exception as e:
            # print(e)
            no_json_result_prompt_list.append(prompt)
            no_json_result_article_id_list.append(article_id)
            continue
            
    return article_id_json_result_list, no_json_result_prompt_list, no_json_result_article_id_list, result_list


# def try_get_answer(prompts, article_id_list, temperature=0):
#     article_id_json_result_list, \
#     no_json_result_prompt_list, \
#     no_json_result_article_id_list, \
#     result_list = get_json_result(prompts, article_id_list, temperature=0.)

article_id_json_result_list, \
no_json_result_prompt_list, \
no_json_result_article_id_list, \
result_list \
= get_json_result(prompts, article_id_list, temperature=0.)

if len(no_json_result_prompt_list) != 0:
    article_id_json_result_list_1, \
    no_json_result_prompt_list_1, \
    no_json_result_article_id_list_1, \
    result_list_1 \
    = get_json_result(no_json_result_prompt_list, no_json_result_article_id_list, temperature=0.1)
    
    article_id_json_result_list += article_id_json_result_list_1
    if len(no_json_result_prompt_list_1) != 0:
        article_id_json_result_list_2, \
        no_json_result_prompt_list_2, \
        no_json_result_article_id_list_2, \
        result_list_2 \
        = get_json_result(no_json_result_prompt_list_1, no_json_result_article_id_list_1, temperature=0.2)
        
        article_id_json_result_list += article_id_json_result_list_2


# print(len(no_json_result_prompt_list_2))


article_id_list = []
dataset_id_list = []
type_list = []

for article_id, json_result_list in article_id_json_result_list:
    results = []
    for json_result in json_result_list:
        if "dataset_id" in json_result and "type" in json_result:
            dataset_id = json_result["dataset_id"]
            dataset_id_type = json_result["type"]

            if dataset_id_type.lower() != "missing":
                article_id_list.append(article_id)
                dataset_id_list.append(dataset_id)
                type_list.append(dataset_id_type)
    
# print(article_id_list, "\n", dataset_id_list, "\n", type_list)
# break


import random

rand_idx = random.randint(0, len(result_list)-1)

print(result_list[rand_idx])


import pandas as pd


sub_df = pd.DataFrame(
    {
        "article_id": article_id_list,
        "dataset_id": dataset_id_list,
        "type": type_list
    }
)

print(sub_df.shape)

sub_df = sub_df.sort_values(
    by=["article_id", "dataset_id", "type"], 
    ascending=False).drop_duplicates(subset=['article_id', 'dataset_id'], keep="first").reset_index(drop=True)


sub_df['row_id'] = range(len(sub_df))
sub_df.to_csv("submission.csv", index=False, columns=["row_id", "article_id", "dataset_id", "type"])

sub_df["type"].value_counts()


sub_df


def f1_score(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
    
    
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    pred_df = pd.read_csv("submission.csv")
    label_df = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/train_labels.csv")
    label_df = label_df[label_df['type'] != 'Missing'].reset_index(drop=True)

    hits_df = label_df.merge(pred_df, on=["article_id", "dataset_id", "type"])
    
    tp = hits_df.shape[0]
    fp = pred_df.shape[0] - tp
    fn = label_df.shape[0] - tp
    
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("F1 Score:", round(f1_score(tp, fp, fn), 3))




