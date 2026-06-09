# !ls ../input/make-data-count-finding-data-references/
# !ls ../input/make-data-count-finding-data-references/test/PDF
# !ls ../input/make-data-count-finding-data-references/test/XML
# !ls ../input/make-data-count-finding-data-references/train/PDF
# !ls ../input/make-data-count-finding-data-references/train/XML


# import os
# import pandas as pd

# data_filepath = "../input/make-data-count-finding-data-references/"

# train_labels_df = pd.read_csv(os.path.join(data_filepath, "train_labels.csv"))
# sample_submission_df = pd.read_csv(os.path.join(data_filepath, "sample_submission.csv"))

# train_pdf_path = "../input/make-data-count-finding-data-references/train/PDF"
# train_pdf_list = os.listdir(train_pdf_path)

# train_xml_path = "../input/make-data-count-finding-data-references/train/XML"
# train_xml_list = os.listdir(train_xml_path)

# train_pdf_path = "../input/make-data-count-finding-data-references/test/PDF"
# train_pdf_list = os.listdir(train_pdf_path)

# train_xml_path = "../input/make-data-count-finding-data-references/test/XML"
# train_xml_list = os.listdir(train_xml_path)


import xml.etree.ElementTree as ET


def extract_text_from_xml(xml_path):
    """Extracts text from an XML file, concatenating all text elements."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        text_parts = []
        for element in root.iter():
            if element.text:
                text_parts.append(element.text)
        return "\n".join(text_parts)  # Join with newlines for readability
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return ""


# xml_filepath = "../input/make-data-count-finding-data-references/train/XML/10.1002_chem.202001668.xml"
# print(extract_text_from_xml(xml_filepath))


# !pip install PyMuPDF vllm logits-processor-zoo==0.1.10 triton==3.2.0


import os

# vLLM V1 does not currently accept logits processor so we need to disable it
# https://docs.vllm.ai/en/latest/getting_started/v1_user_guide.html#deprecated-features
os.environ["VLLM_USE_V1"] = "0"

import re
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import pickle
import vllm
import torch


# Step 1: Read all PDFs and convert to text
pdf_directory = "/kaggle/input/make-data-count-finding-data-references/test/PDF"
                # if os.getenv('KAGGLE_IS_COMPETITION_RERUN') \
                # else "/kaggle/input/make-data-count-finding-data-references/train/PDF"
chunks = []
text_span_len = 100
re_doi = re.compile(r"10\.\d{4}")

for filename in tqdm(os.listdir(pdf_directory), total=len(os.listdir(pdf_directory))):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        
        # Extract article_id from filename
        article_id = filename.split(".pdf")[0]
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            page_text = page.get_text().lower()
            if 'references' in page_text:
                page_text = page_text.split("references")[0]
                text += page_text
                break
            else:
                text += page_text
            
        doc.close()

        doi_matches = re_doi.finditer(text, re.IGNORECASE)
        for match in doi_matches:
            if match.group() in article_id: continue
            chunk = text[max(0, match.start() - text_span_len): match.start() + text_span_len]
            chunks.append((article_id, chunk))


len(chunks)


!ls /kaggle/input/qwen-3/transformers/32b-awq/1


model_path = "/kaggle/input/qwen-3/transformers/32b-awq/1"

llm = vllm.LLM(
    model_path,
    quantization='awq',
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.93,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=8192,
    disable_log_stats=True,
    enable_prefix_caching=True
)
tokenizer = llm.get_tokenizer()


SYS_PROMPT = """
You are an advanced AI reasoning assistant tasked with delivering a comprehensive analysis of a specific problem or question.  Your goal is to outline your reasoning process in a structured and transparent manner, with each step reflecting a thorough examination of the issue at hand, culminating in a well-reasoned conclusion.

### Key Instructions:
1.  Conduct **at least 5 distinct reasoning steps**, each building on the previous one.
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
        seed=402,
        skip_special_tokens=True,
        max_tokens=512,
        temperature=0.01
    ),
    use_tqdm=True
)
responses = [output.outputs[0].text for output in outputs]

doi_urls = []

for response in responses:
    doi_url = "https://doi.org" + response.split("\n")[0]
    doi_urls.append(doi_url)

    break


import random

rand_idx = random.randint(0, len(doi_urls)-1)
print(doi_urls[rand_idx])


SYS_PROMPT = """
You are an advanced AI reasoning assistant tasked with delivering a comprehensive analysis of a specific problem or question.  Your goal is to outline your reasoning process in a structured and transparent manner, with each step reflecting a thorough examination of the issue at hand, culminating in a well-reasoned conclusion.

### Key Instructions:
1.  Conduct **at least 5 distinct reasoning steps**, each building on the previous one.
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


import random

rand_idx = random.randint(0, len(answers)-1)
print(answers[rand_idx])


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










