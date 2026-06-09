!pip install pymupdf
!pip install vllm
!pip install logits-processor-zoo==0.1.10
!pip install triton==3.2.0



import os

# vLLM V1 does not currently accept logits processor so we need to disable it
# https://docs.vllm.ai/en/latest/getting_started/v1_user_guide.html#deprecated-features
os.environ["VLLM_USE_V1"] = "0"

import re
from pypdf import PdfReader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import pickle
import vllm
import torch

# Step 1: Read all PDFs and convert to text
pdf_directory = "/kaggle/input/make-data-count-finding-data-references/test/PDF" \
                if os.getenv('KAGGLE_IS_COMPETITION_RERUN') \
                else "/kaggle/input/make-data-count-finding-data-references/train/PDF"
chunks = []
text_span_len = 100
re_doi = re.compile(r"10\.\d{4}")

for filename in tqdm(os.listdir(pdf_directory), total=len(os.listdir(pdf_directory))):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        
        # Extract article_id from filename
        article_id = filename.split(".pdf")[0]
        text = ""
        
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                page_text = page_text.lower()
                if 'references' in page_text:
                    page_text = page_text.split("references")[0]
                    text += page_text
                    break
                else:
                    text += page_text
            
        doi_matches = re_doi.finditer(text, re.IGNORECASE)
        for match in doi_matches:
            if match.group() in article_id:
                continue
            chunk = text[max(0, match.start() - text_span_len): match.start() + text_span_len]
            chunks.append((article_id, chunk))

print(len(chunks))



model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"

llm = vllm.LLM(
    model_path,
    quantization='awq',
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=2048,
    disable_log_stats=True,
    enable_prefix_caching=True
)
tokenizer = llm.get_tokenizer()


SYS_PROMPT = """
You are given a piece of academic text. Your task is to identify the single DOI citation string, if present.
Then normalize it into its full URL format: https://doi.org/...
"""

prompts = []
for article_id, academic_text in chunks:
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": academic_text}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    ) + "Here is the normalized URL: https://doi.org"
    prompts.append(prompt)

outputs = llm.generate(
    prompts,
    vllm.SamplingParams(
        seed=0,
        skip_special_tokens=True,
        max_tokens=32,
        temperature=0
    ),
    use_tqdm=True
)
responses = [output.outputs[0].text for output in outputs]

doi_urls = []

for response in responses:
    doi_url = "https://doi.org" + response.split("\n")[0]
    doi_urls.append(doi_url)


SYS_PROMPT = """
You are given a piece of academic text. Your task is to identify the single DOI citation string, if present.
Classify the data associated with that DOI as:
A)Primary: if the data was generated specifically for this study.
B)Secondary: if the data was reused or derived from prior work.
C)None: if the DOI is part of the References section of a paper, does not refer to research data or is unrelated.

Respond with one of A, B or C.
"""

prompts = []
for article_id, academic_text in chunks:
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": academic_text}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompts.append(prompt)

mclp = MultipleChoiceLogitsProcessor(tokenizer, 
                                     choices=["A", "B", "C"])


outputs = llm.generate(
    prompts,
    vllm.SamplingParams(
        seed=0,
        skip_special_tokens=True,
        max_tokens=1,
        logits_processors=[mclp],
        logprobs=len(mclp.choices)

    ),
    use_tqdm=True
)


logprobs = []
for lps in [output.outputs[0].logprobs[0].values() for output in outputs]:
    logprobs.append({lp.decoded_token: lp.logprob for lp in list(lps)})

logit_matrix = pd.DataFrame(logprobs)[["A", "B", "C"]].values

choices = ["Primary", "Secondary", None]
answers = [choices[pick] for pick in np.argmax(logit_matrix, axis=1)]


sub_df = pd.DataFrame()
sub_df["article_id"] = [c[0] for c in chunks]
sub_df["dataset_id"] = doi_urls
sub_df["dataset_id"] = sub_df["dataset_id"].str.lower()
sub_df["type"] = answers
sub_df = sub_df[sub_df["type"].notnull()].reset_index(drop=True)


sub_df = sub_df.sort_values(by=["article_id", "dataset_id", "type"], ascending=False).drop_duplicates(subset=['article_id', 'dataset_id'], keep="first").reset_index(drop=True)

sub_df['row_id'] = range(len(sub_df))
sub_df.to_csv("submission.csv", index=False, columns=["row_id", "article_id", "dataset_id", "type"])

sub_df["type"].value_counts()


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




