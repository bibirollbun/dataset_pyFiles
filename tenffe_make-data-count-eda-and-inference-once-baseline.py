# !pip install PyMuPDF
!pip install xmltodict
!pip install pymupdf
!pip install PyPDF2
!pip install json_repair


import os
import sys
import fitz # PyMuPDF
import xmltodict # xmltodict
import pandas as pd
from tqdm.auto import tqdm
import contextlib
from PyPDF2 import PdfReader
import json_repair


pdf_directory = "/kaggle/input/make-data-count-finding-data-references/test/PDF"
xml_directory = "/kaggle/input/make-data-count-finding-data-references/test/XML"
train_label_path = "/kaggle/input/make-data-count-finding-data-references/train_labels.csv"


# train_df = pd.read_csv(train_label_path)
# train_df.sample(5)

# for getting abstract from xml file
# if os.path.exists(xml_filepath):
#     # abstract is important for 
#     xml_path, title, abstract, body = extract_xml_fields(xml_filepath)
#     if abstract is not None:
#         if isinstance(abstract, list):
#             for abs_dict in abstract:
#                 if isinstance(abs_dict["p"], dict):
#                     if "#text" in abs_dict["p"].keys():
#                         abstract_text += abs_dict["p"]["#text"] + "\n\n"
#                 else:
#                     abstract_text += abs_dict["p"]
#         else:
#             abstract_text = ""
#             if isinstance(abstract["p"], dict):
#                 if "#text" in abstract["p"].keys():
#                     abstract_text += abs_dict["p"]["#text"] + "\n\n"
#             else:
#                 abstract_text += abs_dict["p"]


def extract_xml_fields(xml_path):
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_dict = xmltodict.parse(f.read())
        article = xml_dict.get('article') or xml_dict.get('ns:article')

        front = article.get('front', {})
        meta = front.get('article-meta', {})

        title = meta.get('title-group', {}).get('article-title')
        abstract = meta.get('abstract')
        body = article.get('body')

        return xml_path, title, abstract, body
    except Exception as e:
        return None, None, None, None


def get_pdf_content(pdf_filepath):
    try:
        with fitz.open(pdf_filepath) as doc:
            text = ""
            for page in doc:
                page_text = page.get_text().lower()
                text += page_text + "\n"
        return text
    except Exception as e:
        return None


@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


def extract_pdf_text_silent(path: str) -> str:

    text = []
    with suppress_stderr():
        # PdfReader can take either a file path or a file-like object
        reader = PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

    return "\n".join(text)


pdf_file_list = [filename for filename in os.listdir(pdf_directory) if filename.endswith(".pdf")]
xml_file_list = [filename for filename in os.listdir(xml_directory) if filename.endswith(".xml")]

print("the number of pdf file is %d"%len(pdf_file_list))

text_list = []

for pdf_file in tqdm(pdf_file_list):
    try:
        pdf_filepath = os.path.join(pdf_directory, pdf_file)
        text = get_pdf_content(pdf_filepath)
    except:
        # 使用ocr工具,提取pdf内容,暂时不需要用到
        pass 
        
    text_list.append(text)


%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use("ggplot")


inference_text_len_list = [len(text) for text in text_list]
plt.plot(inference_text_len_list)

# pdf的文本长度都很长,需要使用chunk方法.


import random

rand_idx = random.randint(0, len(text_list)-1)

# print(text_list[rand_idx])


!pip install vllm torchvision -U


import torch
import vllm


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


import vllm

vllm.__version__


reason_prompt =  '''
You are an advanced AI reasoning assistant tasked with delivering a comprehensive analysis of a specific problem or question.  
Your goal is to outline your reasoning process in a structured and transparent manner, with each step reflecting a thorough examination of the issue at hand, culminating in a well-reasoned conclusion.


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
'''

task_prompt = '''
You are given a piece of academic text. Your task is to identify the single DOI citation string, if present.
Then normalize it into its full URL format: https://doi.org/...

Each object (paper and dataset) has a unique, persistent identifier to represent it. In this competition there will be two types:

the definition of dataset_id is as follows:
the dataset identifier and citation type in the paper.
DOIs are used for all papers and some datasets. They take the following form: https://doi.org/[prefix]/[suffix]. Examples:
https://doi.org/10.1371/journal.pone.0303785
https://doi.org/10.5061/dryad.r6nq870
Accession IDs are used for some datasets. They vary in form by individual data repository where the data live. Examples:
"GSE12345" (Gene Expression Omnibus dataset)
“PDB 1Y2T” (Protein Data Bank dataset)
"E-MEXP-568" (ArrayExpress dataset)

the definition of type is as follows:
the type citation type, Primary - raw or processed data generated as part of this paper, specifically for this study
Secondary - raw or processed data derived or reused from existing records or published data
'''

example_prompt = '''
use markdown json style to get the result as the following examples:

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
'''


pdf_chunk_dict_list = []
chunk_size = 2048
intersection_size = chunk_size // 20

for pdf_file, text in zip(pdf_file_list, text_list):
    # print(pdf_file)
    if len(text) <= chunk_size:
        pdf_chunk_dict_list.append(
            {
                pdf_file: [text]
            }
        )
    else:
        chunks = []
        for index in range(0, len(text), chunk_size):
            chunks.append(text[index:index+chunk_size+intersection_size])
        pdf_chunk_dict_list.append(
            {
                pdf_file: chunks
            }
        )


from tqdm.auto import tqdm

outputs_dict_list = []

for pdf_chunk_dict in tqdm(pdf_chunk_dict_list):
    pdf_file = list(pdf_chunk_dict.keys())[0]
    chunks = pdf_chunk_dict[pdf_file]

    prompts = []

    SYS_PROMPT = reason_prompt + task_prompt + example_prompt
    for chunk in chunks:
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": chunk}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(prompt)

    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(
            seed=402,
            skip_special_tokens=True,
            max_tokens=4096,
            temperature=0.01
        ),
        use_tqdm=True
    )

    resules = [output.text for output in outputs[0].outputs]
    outputs_dict_list.append(
        {
            pdf_file: resules
        }
    )

    break


print(outputs[0].outputs[0].text)


import json_repair
import json
import re


def get_json_str(json_in_str):
    # 取出最后的一个json字符串
    json_start_index = json_in_str.rfind("```json", 0)
    json_end_index = json_in_str.find("```", json_start_index+7)
    json_str = json_in_str[json_start_index+7:json_end_index].strip()
    return json_repair.loads(json_str)


result_with_json_style = get_json_str(outputs[0].outputs[0].text)


result_with_json_style
















