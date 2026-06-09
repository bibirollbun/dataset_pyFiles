!cp /kaggle/usr/lib/mdc_fine_grained_eval/mdc_fine_grained_eval.py .


import os
import re
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from mdc_fine_grained_eval import compute_metrics

# Step 1: Read all PDFs and convert to text
pdf_directory = "/kaggle/input/make-data-count-finding-data-references/test/PDF" \
                if os.getenv('KAGGLE_IS_COMPETITION_RERUN') \
                else "/kaggle/input/make-data-count-finding-data-references/train/PDF"
all_data = []
texts = []

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
        texts.append((article_id, text))


chunks = []

for article_id, text in texts:
    doi_matches = re.finditer(r"10\.\d{4}", text, re.IGNORECASE)
    for match in doi_matches:
        if match.group() in article_id: continue
        chunk = text[max(0, match.start() - 200): match.start() + 200]
        chunks.append((article_id, chunk))


import pickle

with open('chunks.pkl', 'wb') as file:
    pickle.dump(chunks, file)


len(chunks)


import re
import vllm
import pandas as pd
import torch
import pickle

with open('chunks.pkl', 'rb') as file:
    chunks = pickle.load(file)

llm = vllm.LLM(
    "/kaggle/input/qwen-3/transformers/8b-awq/1",
    quantization='awq',
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=2048,
    disable_log_stats=True
)
tokenizer = llm.get_tokenizer()


prompt_template = """
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
}}
```

Academic Text:
{input_text}
"""

prompts = [prompt_template.format(input_text=x[1]) for x in chunks]

responses = llm.generate(
    prompts,
    vllm.SamplingParams(
        n=1,  # Number of output sequences to return for each prompt.
        top_p=0.8,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=0,  # randomness of the sampling
        seed=777, # Seed for reprodicibility
        skip_special_tokens=False,  # Whether to skip special tokens in the output.
        max_tokens=512,  # Maximum number of tokens to generate per output sequence.
    ),
    use_tqdm=True
)

responses = [x.outputs[0].text.lower() for x in responses]

with open('responses.pkl', 'wb') as file:
    pickle.dump(responses, file)


with open('responses.pkl', 'rb') as file:
    responses = pickle.load(file)


import json

json_block_pattern = r'```json\s*(.*?)\s*```'
citations = []

for (article_id, _), resp in zip(chunks, responses):
    try:
        match = re.search(json_block_pattern, resp, re.DOTALL)
        json_string = match.group(1).strip()
        parsed_json = json.loads(json_string)
        category = parsed_json["classification"].capitalize()
        if category in ['Primary', 'Secondary']:
            citations.append([article_id, parsed_json['doi'].replace("\u200b", "").replace("\n", ""), category])
    except Exception as e:
        # print(e)
        pass


submission = pd.DataFrame(citations, columns=['article_id', 'dataset_id', 'type'])

dataset_id_counts = submission['dataset_id'].value_counts()
frequent_dataset_ids = dataset_id_counts[dataset_id_counts >= 3].index
submission = submission[~submission['dataset_id'].isin(frequent_dataset_ids)].sort_values(by=["article_id", "dataset_id", "type"], ascending=True).drop_duplicates(subset=['article_id', 'dataset_id'])
submission['row_id'] = range(len(submission))

submission[['row_id', 'article_id', 'dataset_id', 'type']].to_csv("submission.csv", index=False)


pd.read_csv("submission.csv")


if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    pred_df = pd.read_csv("submission.csv")
    reference_df = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/train_labels.csv")
    reference_df = reference_df[reference_df['type'] != 'Missing'].reset_index(drop=True)
    eval_dict = compute_metrics(pred_df, reference_df)
    m = eval_dict['ents_f1']
    print(f"CV @ 524 train PDF = {round(m, 4)}")
    print("\n\n")
    print(json.dumps(eval_dict['ents_per_type'], indent=4))

