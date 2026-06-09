!pip install pymupdf -t pymupdf
!pip install vllm -t vllm


!cd pymupdf && zip -r ../pymupdf.zip .
!cd vllm && zip -r ../vllm.zip .


import os
import re
import fitz
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import vllm
import sys




import os
import re
import fitz 
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

pdf_dir = Path("/kaggle/input/make-data-count-finding-data-references/test/PDF")
xml_dir = Path("/kaggle/input/make-data-count-finding-data-references/test/XML")

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
ACC_PATTERN = re.compile(r"\b(GSE\d+|EGAS\d+|PRJNA\d+|SRP\d+|DRP\d+|ERP\d+|PXD\d+|MTBLS\d+)\b", re.I)
HANDLE_PATTERN = re.compile(r"hdl:\s*\d{2,4}/[\w.]+", re.I)
PATTERNS = [(DOI_PATTERN, "DOI"), (ACC_PATTERN, "Accession"), (HANDLE_PATTERN, "Handle")]

def extract_pdf_mentions(pdf_path):
    results = {}
    try:
        with fitz.open(pdf_path) as doc:
            full_text = "\n".join(page.get_text() for page in doc)
            for page in doc:
                annot = page.first_annot
                while annot:
                    content = annot.info.get("content", "")
                    if content:
                        full_text += f"\n{content}"
                    annot = annot.next
    except:
        return results

    for pattern, label in PATTERNS:
        for match in pattern.finditer(full_text):
            mention = match.group()
            context = full_text[max(0, match.start() - 300):match.end() + 300]
            results[(mention, label)] = {"pdf_context": context}
    return results

def extract_xml_mentions(xml_path):
    results = {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        text = ET.tostring(root, encoding='unicode', method='text')
    except:
        return results

    for pattern, label in PATTERNS:
        for match in pattern.finditer(text):
            mention = match.group()
            context = text[max(0, match.start() - 300):match.end() + 300]
            results[(mention, label)] = {"xml_context": context}
    return results

results = []

for article_id in tqdm(sorted(set(f.stem for f in pdf_dir.glob("*.pdf")) | set(f.stem for f in xml_dir.glob("*.xml")))):
    combined = {}

    pdf_path = pdf_dir / f"{article_id}.pdf"
    if pdf_path.exists():
        pdf_mentions = extract_pdf_mentions(pdf_path)
        for key, value in pdf_mentions.items():
            combined.setdefault(key, {}).update(value)

    xml_path = xml_dir / f"{article_id}.xml"
    if xml_path.exists():
        xml_mentions = extract_xml_mentions(xml_path)
        for key, value in xml_mentions.items():
            combined.setdefault(key, {}).update(value)

    for (mention, mention_type), data in combined.items():
        context_parts = []
        if "pdf_context" in data:
            context_parts.append(data["pdf_context"])
        if "xml_context" in data:
            context_parts.append(data["xml_context"])
        full_context = "\n\n---\n\n".join(context_parts)

        results.append({
            "article_id": article_id,
            "mention_type": mention_type,
            "mention": mention,
            "context": full_context
        })

df_mentions = pd.DataFrame(results)
print(f"✅ Total unique mentions with combined context: {len(df_mentions)}")
df_mentions.head()



def construct_link(row):
    mention = row["mention"].strip()
    mtype = row["mention_type"].lower()

    if mtype == "doi":
        if mention.startswith("10."):
            return f"https://doi.org/{mention}"
        else:
            return f"https://doi.org/{mention.lstrip('doi:').strip()}"

    elif mtype == "accession":
        return mention  

    elif mtype == "url":
        
        return mention if mention.startswith("http") else "http://" + mention

    else:
        return mention 
df_mentions["link"] = df_mentions.apply(construct_link, axis=1)



model_path = "/kaggle/input/qwen2.5/transformers/0.5b-instruct-awq/1"

llm = vllm.LLM(
    model=model_path,
    quantization="awq",
    tensor_parallel_size=1  
)
tokenizer = llm.get_tokenizer()


SYS_PROMPT_DOI = """
You are an expert at identifying RESEARCH DATA citations in academic papers.
Your task is to determine if a DOI in the provided text specifically refers to a dataset, software, or data repository, NOT another academic paper.

**Crucial Rules:**
1.  **LOOK FOR DATA CONTEXT:** The DOI must be near keywords like "data available", "deposited in", "repository", "accession number", "software", "code".
2.  **IGNORE BIBLIOGRAPHY:** If the DOI is clearly part of a numbered or author-year list in a "References" or "Bibliography" section, you MUST respond with "Irrelevant".
3.  **PRIORITIZE DATA DOIs:** If there are multiple DOIs, return the one most likely to be a dataset.

Only respond with either a full normalized DOI URL starting with "https://doi.org/" or the single word "Irrelevant".
Do NOT include any other text or explanation.
"""
SYS_PROMPT_ACCESSION = """
You are a highly accurate assistant for extracting descriptions of datasets in scientific articles.

Given a chunk of text from a research paper and a specific dataset ID, extract only the sentences or paragraph that directly describe or reference that dataset ID — not other IDs or unrelated content.

Be precise and concise. Ignore surrounding unrelated material. I
ONLY return extracted text.
"""

SYS_PROMPT_CLASSIFY_DOI = """
You are an expert at analyzing research data citations in academic papers.

Classify the data as:
A) Primary: if the data was generated specifically for this study
B) Secondary: if the data was reused or derived from prior work  

Respond with only one letter: A, B
"""


from vllm import SamplingParams
from tqdm import tqdm

prompts = []
for _, row in df_mentions.iterrows():
    academic_text = row["context"]
    mention = row["link"]

    messages = [
        {"role": "system", "content": SYS_PROMPT_DOI},
        {"role": "user", "content": f"Mention: {mention}\n\nContext:\n{academic_text}"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    prompts.append(prompt)


sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=64,
    stop_token_ids=None,
    seed=42,
    skip_special_tokens=True
)

responses = llm.generate(prompts, sampling_params, use_tqdm=True)

doi_decisions = [r.outputs[0].text.strip() for r in responses]

df_mentions["llm_response"] = doi_decisions

df_mentions["is_dataset"] = df_mentions["llm_response"].apply(
    lambda x: x.lower().startswith("https://doi.org/")
)

print(df_mentions[["mention", "link", "llm_response", "is_dataset"]].head())



print((df_mentions["is_dataset"] == True).sum())


df_mentions = df_mentions[df_mentions["is_dataset"] != False]


print(len(df_mentions))


from vllm import SamplingParams
import json

refined_contexts = []

for _, row in tqdm(df_mentions.iterrows(), total=len(df_mentions)):
    mention = row["mention"]
    context = row["context"]
    article_id = row["article_id"]

    # Construct message
    messages = [
        {"role": "system", "content": SYS_PROMPT_ACCESSION.strip()},
        {"role": "user", "content": f"Dataset ID: {mention}\n\nText:\n{context}"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = llm.generate(
        [prompt],
        SamplingParams(max_tokens=256, temperature=0, stop=["\n\n"]),
        use_tqdm=False
    )

    answer = output[0].outputs[0].text.strip()
    refined_contexts.append(answer)

df_mentions["refined_context"] = refined_contexts



print((df_mentions["refined_context"] == "Irrelevant").sum())


from vllm import SamplingParams  

sampling_params = SamplingParams(temperature=0, max_tokens=1)

messages_list = []
for _, row in df_mentions.iterrows():
    user_prompt = f"""Dataset Mention: {row['mention']}

Context:
{row['context']}
"""
    messages = [
        {"role": "system", "content": SYS_PROMPT_CLASSIFY_DOI.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    messages_list.append(formatted_prompt)
outputs = llm.generate(messages_list, sampling_params, use_tqdm=True)

predictions = [out.outputs[0].text.strip().capitalize() for out in outputs]

df_mentions["predicted_type"] = predictions



print(df_mentions.head())


label_map = {"A": "Primary", "B": "Secondary"}

df_mentions["predicted_type"] = df_mentions["predicted_type"].map(label_map)



submission_df = df_mentions[["article_id", "mention", "predicted_type"]].rename(
    columns={"mention": "dataset_id", "predicted_type": "type"}
)

submission_df.to_csv("submission.csv", index=False)
print("Final submission CSV ready with shape:", submission_df.shape)



submission_df = submission_df.reset_index(drop=True)
submission_df["row_id"] = submission_df.index



submission_df = submission_df[["row_id", "article_id", "dataset_id", "type"]]

submission_df.to_csv("/kaggle/working/submission.csv", index=False)

import os
print("Exists:", os.path.exists("/kaggle/working/submission.csv"))


