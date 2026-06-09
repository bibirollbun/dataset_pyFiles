# !pip install vllm langchain langchain-community tqdm vllm

import os
import re
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

import vllm
from langchain_community.document_loaders import PyPDFLoader, UnstructuredXMLLoader


def load_files(pdf_path, xml_path, dataset_name):
    pdf_files = [f for f in os.listdir(pdf_path) if f.endswith('.pdf')]
    xml_files = [f for f in os.listdir(xml_path) if f.endswith('.xml')]
    df_pdf = pd.DataFrame({
        'article_id': [f.replace('.pdf', '') for f in pdf_files],
        'file': pdf_files,
        'path': [os.path.join(pdf_path, f) for f in pdf_files],
        'format': 'pdf',
        'dataset': dataset_name
    })
    df_xml = pd.DataFrame({
        'article_id': [f.replace('.xml', '') for f in xml_files],
        'file': xml_files,
        'path': [os.path.join(xml_path, f) for f in xml_files],
        'format': 'xml',
        'dataset': dataset_name
    })
    return pd.concat([df_pdf, df_xml], ignore_index=True)


def extract_text(row):
    if row['format'] == 'pdf':
        try:
            loader = PyPDFLoader(row['path'])
            docs = loader.load()
            # Combine all pages, or chunk as needed
            text = " ".join([doc.page_content for doc in docs])
        except Exception as e:
            text = ""
    elif row['format'] == 'xml':
        try:
            loader = UnstructuredXMLLoader(row['path'])
            docs = loader.load()
            text = " ".join([doc.page_content for doc in docs])
        except Exception as e:
            text = ""
    else:
        text = ""
    return text


# # Set up file paths
# TRAIN_PATH = '/kaggle/input/make-data-count-finding-data-references/train'
# LABELS_PATH = '/kaggle/input/make-data-count-finding-data-references/train_labels.csv'
# TRAIN_PDF = os.path.join(TRAIN_PATH, 'PDF')
# TRAIN_XML = os.path.join(TRAIN_PATH, 'XML')

TEST_PATH = '/kaggle/input/make-data-count-finding-data-references/test'
TEST_PDF = os.path.join(TEST_PATH, 'PDF')
TEST_XML = os.path.join(TEST_PATH, 'XML')

# # Load the data
# labels_df = pd.read_csv(LABELS_PATH)
# labels_df['has_dataset'] = labels_df['dataset_id'] != 'Missing'

# # Gather the file paths and Merge the data
# train_files = load_files(TRAIN_PDF, TRAIN_XML, 'train')
# train_merged = train_files.merge(labels_df, how='left', on='article_id')
# train_merged['has_dataset'] = train_merged['dataset_id'] != 'Missing'

test_files = load_files(TEST_PDF, TEST_XML, 'test')
test_merged = test_files.copy()


tqdm.pandas()
# train_merged['text'] = train_merged.progress_apply(extract_text, axis=1)
test_merged['text'] = test_merged.progress_apply(extract_text, axis=1)


doi_pattern = r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)"  # simple regex for DOI pattern

def find_dois_and_context(text, window=200):
    matches = []
    for match in re.finditer(doi_pattern, text):
        start, end = match.start(), match.end()
        context = text[max(0, start - window): min(len(text), end + window)]
        matches.append({
            "doi": match.group(),
            "context": context
        })
    return matches


# For each paper, get DOIs and their context
def extract_candidates(df):
    candidates = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        dois = find_dois_and_context(row['text'])
        for d in dois:
            candidates.append({
                "article_id": row['article_id'],
                "doi": d["doi"],
                "context": d["context"],
                "format": row['format'],
                "file": row['file']
            })
    return pd.DataFrame(candidates)


# Extract candidates for test
test_candidates = extract_candidates(test_merged)


def build_prompt(context):
    return f"""
    You are given a piece of academic text. Your task is to:
    
    1. Identify the single DOI citation string, if present.
    2. Normalize it into its full URL format: https://doi.org/...
    3. Classify the data associated with that DOI as:
        - "Primary": if the data was generated specifically for this study.
        - "Secondary": if the data was reused or derived from prior work.
        - "Not Relevant": if the DOI is part of the References section of a paper, does not refer to research data or is unrelated.
    
    If no valid DOI is found, return an empty JSON object {{}}.
    
    ONLY return ONE dictionary within JSON backticks with two keys:
    
    ```json
    {{
      "doi": "<doi string> starting with https://doi.org/", 
      "classification": <"Primary", "Secondary", or "Not Relevant">
    }}```
    
    Academic text:
    {context}""".strip()


prompts = [build_prompt(row["context"]) for _, row in test_candidates.iterrows()]


# Set sampling params for deterministic inference
params = vllm.SamplingParams(
    temperature=0,
    top_p=0.8,
    max_tokens=512,
    seed=777,
    skip_special_tokens=True,
)


# Run vLLM batch inference

QWEN_PATH = "/kaggle/input/qwen-3/transformers/8b-awq/1"

llm = vllm.LLM(
    model=QWEN_PATH,
    quantization='awq',
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=2048,
    disable_log_stats=True
)

outputs = llm.generate(prompts, params, use_tqdm=True)


json_block_pattern = r'```json\s*(.*?)\s*```'
results = []

# Loop over vLLM outputs and corresponding candidate metadata
for (idx, row), out in zip(test_candidates.iterrows(), outputs):
    response_text = out.outputs[0].text
    try:
        match = re.search(json_block_pattern, response_text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            parsed = json.loads(json_str)
            doi_url = parsed.get("doi", "").replace("\u200b", "").replace("\n", "")
            citation_type = parsed.get("classification", "").capitalize()
            # Only keep Primary or Secondary classifications with a DOI
            if doi_url and citation_type in ["Primary", "Secondary"]:
                results.append({
                    "article_id": row["article_id"],
                    "dataset_id": doi_url,
                    "type": citation_type
                })
    except Exception:
        # Silently ignore parse errors, could log if desired
        pass

# Deduplicate as per competition rules
submission = pd.DataFrame(results)
submission = submission.drop_duplicates(subset=["article_id", "dataset_id", "type"])

# Optionally: Remove dataset_ids cited in too many articles (possible references, not true data citations)
dataset_id_counts = submission['dataset_id'].value_counts()
# For example, drop dataset_ids appearing in >=3 articles (tune this as desired)
frequent_dataset_ids = dataset_id_counts[dataset_id_counts >= 3].index
submission = submission[~submission['dataset_id'].isin(frequent_dataset_ids)]

# Re-add row_id as required by submission format
submission = submission.sort_values(by=["article_id", "dataset_id", "type"], ascending=True)
submission['row_id'] = range(len(submission))

# Save for competition
submission[['row_id', 'article_id', 'dataset_id', 'type']].to_csv("submission.csv", index=False)

# Show head for debug
print(submission.head())





