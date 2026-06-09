%pip install "transformers==4.43.3" "accelerate==0.33.0" torch pandas numpy tqdm



import sys, platform, importlib
mods = ["transformers","accelerate","pandas","numpy","tqdm","fitz"]
for m in mods:
    try:
        mod = importlib.import_module(m if m != "fitz" else "fitz")
        print(f"{m}: {getattr(mod, '__version__', 'OK')}")
    except Exception as e:
        print(f"{m}: NOT INSTALLED ({e})")
print("Python:", sys.version.split()[0], "| macOS:", platform.mac_ver()[0])


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/make-data-count-finding-data-references'): #'/kaggle/input/'
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!cp /kaggle/usr/lib/mdc_fine_grained_eval/mdc_fine_grained_eval_py.py ./mdc_fine_grained_eval.py



!grep "def compute_metrics" mdc_fine_grained_eval.py



!grep "def " mdc_fine_grained_eval.py



import importlib.util

# Define path to the .py file
file_path = './mdc_fine_grained_eval.py'

# Load module from file path
spec = importlib.util.spec_from_file_location("mdc_fine_grained_eval", file_path)
mdc_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mdc_eval)

# Now, you can call compute_metrics like this:
# Example usage:
# metrics = mdc_eval.compute_metrics(pred_df, gt_df)



%pip install pdfplumber



import os, glob
from pathlib import Path

BASE_DIR = os.getenv('PDF_BASE_DIR', '/kaggle/input/make-data-count-finding-data-references')
use_test = os.getenv('USE_TEST', '0') == '1'
pdf_directory = os.path.join(BASE_DIR, 'test', 'PDF') if use_test else os.path.join(BASE_DIR, 'train', 'PDF')

print("BASE_DIR   :", os.path.abspath(BASE_DIR))
print("Using TEST :", use_test)
print("PDF folder :", os.path.abspath(pdf_directory))
print("Exists?    :", os.path.exists(pdf_directory))

# Count files (case-insensitive)
count_lower = len(glob.glob(os.path.join(pdf_directory, "*.pdf")))
count_upper = len(glob.glob(os.path.join(pdf_directory, "*.PDF")))
print("Found PDFs :", count_lower + count_upper, f"(lower:{count_lower} upper:{count_upper})")

# If zero, stop here and fix the path or add data
if not os.path.exists(pdf_directory):
    raise FileNotFoundError(f"Folder not found: {pdf_directory}")
if (count_lower + count_upper) == 0:
    raise FileNotFoundError(f"No PDFs found in {pdf_directory}. Check names/extensions and dataset structure.")




import os, glob
from pathlib import Path
import pdfplumber
from tqdm import tqdm

BASE_DIR = os.getenv('PDF_BASE_DIR', '/kaggle/input/make-data-count-finding-data-references')
use_test = os.getenv('USE_TEST', '0') == '1'
pdf_directory = os.path.join(BASE_DIR, 'test', 'PDF') if use_test else os.path.join(BASE_DIR, 'train', 'PDF')

# Early checks
if not os.path.exists(pdf_directory):
    raise FileNotFoundError("Folder not found: {}".format(os.path.abspath(pdf_directory)))

num_lower = len(glob.glob(os.path.join(pdf_directory, "*.pdf")))
num_upper = len(glob.glob(os.path.join(pdf_directory, "*.PDF")))
if (num_lower + num_upper) == 0:
    raise FileNotFoundError("No PDFs found in {} (found lower:{}, upper:{})"
                            .format(os.path.abspath(pdf_directory), num_lower, num_upper))

# Case-insensitive glob
all_pdfs = sorted(list(Path(pdf_directory).glob("*.pdf")) +
                  list(Path(pdf_directory).glob("*.PDF")))

texts = []
print("Reading {} PDFs from: {}".format(len(all_pdfs), os.path.abspath(pdf_directory)))

for pdf_path in tqdm(all_pdfs, desc="Reading PDFs"):
    article_id = pdf_path.stem
    try:
        text_parts = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                page_text = page_text.lower()
                if "references" in page_text:
                    page_text = page_text.split("references")[0]
                    text_parts.append(page_text)
                    break
                else:
                    text_parts.append(page_text)
        texts.append((article_id, "".join(text_parts)))
    except Exception as e:
        # keep going, but log which file failed
        print("[WARN] Skipping {}: {}".format(pdf_path.name, e))




#import os
#import re
#import fitz  # PyMuPDF
#import pandas as pd
#from pathlib import Path
#from tqdm.auto import tqdm
#from mdc_fine_grained_eval import compute_metrics

#Step 1: Read all PDFs and convert to text
#pdf_directory = "//kaggle/input/make-data-count-finding-data-references/test/PDF" \
                #if os.getenv('KAGGLE_IS_COMPETITION_RERUN') \
                #else "/kaggle/input/make-data-count-finding-data-references/train/PDF"
#all_data = []
#texts = []

#for filename in tqdm(os.listdir(pdf_directory), total=len(os.listdir(pdf_directory))):
    #if filename.endswith(".pdf"):
        #pdf_path = os.path.join(pdf_directory, filename)
         # Extract article_id from filename
        #article_id = filename.split(".pdf")[0]
        #doc = fitz.open(pdf_path)
        #text = ""
        #for page in doc:
            #page_text = page.get_text().lower()
            #if 'references' in page_text:
                #page_text = page_text.split("references")[0]
                #text += page_text
                #break
            #else:

                #text += page_text
            
        #doc.close()
        #texts.append((article_id, text))
        


import re

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


#import os
#os.environ["TRANSFORMERS_NO_TF"] = "1"   # avoid TF/Flax backends
#os.environ["TRANSFORMERS_NO_FLAX"] = "1"

#from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

#MODEL_ID = "sshleifer/tiny-gpt2"  # start tiny to verify
#tok = AutoTokenizer.from_pretrained(MODEL_ID)
#mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID)
#gen = pipeline("text-generation", model=mdl, tokenizer=tok, device=-1)

#print(gen("Hello", max_new_tokens=20)[0]["generated_text"])



import os
os.environ["TRANSFORMERS_NO_TF"] = "1"   # avoid TF/Flax backends
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load precomputed chunks
with open('chunks.pkl', 'rb') as file:
    chunks = pickle.load(file)

# Choose a small default model for CPU. Change via env HF_MODEL_ID if desired.
MODEL_ID = os.getenv("HF_MODEL_ID", "sshleifer/tiny-gpt2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)



# CPU sanity check for generation
prompt = "Explain quantum entanglement in simple terms."
out = gen(prompt, max_new_tokens=100)[0]["generated_text"]
print(out)


#pip uninstall -y transformers tokenizers huggingface_hub accelerate
!pip install --no-cache-dir "transformers==4.43.3" "accelerate==0.33.0" "tokenizers==0.19.1" "huggingface_hub==0.23.5"



import re, pickle, pandas as pd
from tqdm.auto import tqdm

prompt_template = """
You are given a piece of academic text. Your task is to:

1. Identify the single DOI citation string, if present.
2. Normalize it into its full URL format: https://doi.org/...
3. Classify the data associated with that DOI as:
    - "Primary": if the data was generated specifically for this study.
    - "Secondary": if the data was reused or derived from prior work.
    - "Not Relevant": if the DOI is part of the References section of a paper, does not refer to research data or is unrelated.

If no valid DOI is found, return an empty JSON object {}.

ONLY return ONE dictionary within JSON backticks with two keys:

```json
{"doi": "...", "type": "Primary|Secondary|Not Relevant"}
```
""".strip()

def build_prompt(article_id, chunk_text):
    return f"Article ID: {article_id}\n\nText:\n{chunk_text}\n\nInstructions:\n{prompt_template}\n\nReturn JSON only."

# Build prompts
prompts = [build_prompt(aid, txt) for aid, txt in chunks]

# Generate
responses = []
for p in tqdm(prompts, total=len(prompts)):
    try:
        txt = gen(p, max_new_tokens=512, do_sample=False)[0]["generated_text"]
    except Exception as e:
        txt = f"ERROR: {e}"
    responses.append(txt.lower())

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


from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(pred_df, reference_df):
    # assumes both DataFrames have ['article_id','dataset_id','type']
    y_true = reference_df['type'].values
    y_pred = pred_df.set_index('article_id').reindex(reference_df['article_id'])['type'].fillna("Missing").values

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "ents_f1": f1,
        "ents_per_type": {
            "precision": p,
            "recall": r,
            "f1": f1
        }
    }



pred_df = pd.read_csv("submission.csv")
reference_df = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/train_labels.csv")   # adjust path
reference_df = reference_df[reference_df['type'] != 'Missing'].reset_index(drop=True)

eval_dict = compute_metrics(pred_df, reference_df)
print("F1:", round(eval_dict['ents_f1'], 4))
print(json.dumps(eval_dict['ents_per_type'], indent=4))

