# Installing bitsandbytes from the local.the whl of the file ---
!pip install /kaggle/input/hf-libraries/bitsandbytes/bitsandbytes-0.45.2-py3-none-manylinux_2_24_x86_64.whl --no-index --no-deps

'''
print('We restart the kernel so that the changes take effect, but it doesn't seem to be necessary.')
import os
os.kill(os.getpid(), 9)
'''


import sys
import os
import gc
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
import importlib.metadata
from packaging import version

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_bitsandbytes_available
from transformers import AutoModelForCausalLM
import transformers.utils

# Adding the path to the local sentence-transformers library to the Python system path
# Now we can import it as if it was installed
sys.path.append('/kaggle/input/sentence-transformers/sentence-transformers')

from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig


class CFG:
    # Competition paths
    TRAIN_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"
    TEST_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
    SUBMISSION_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv"
    
    # Paths to local models
    EMBEDDING_MODEL_PATH = "/kaggle/input/sentence-transformers/minilm-l6-v2/all-MiniLM-L6-v2"
    RERANKER_MODEL_PATH = "/kaggle/input/qwen-3-reranker/transformers/0.6b/1"
    LLM_MODEL_PATH = "/kaggle/input/qwen2.5-math/transformers/7b/1"

    # Parameters
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Search and ensemble parameters
    TOP_K_CANDIDATES = 20 # How many candidates to generate
    RERANKER_WEIGHT = 0.4 # Weight for Fast Qwen Re-ranker
    QWEN_MATH_WEIGHT = 0.6 # Weight for powerful Qwen Math


train_df = pd.read_csv(CFG.TRAIN_PATH)
test_df = pd.read_csv(CFG.TEST_PATH)

def create_text_and_target(df, is_train=True):
    for col in ['QuestionText', 'MC_Answer', 'StudentExplanation']:
        df[col] = df[col].astype(str).str.strip()
    df['full_text'] = df['QuestionText'] + '[SEP]' + df['StudentExplanation']
    if is_train:
        df['target'] = df['Category'] + ':' + df['Misconception']
    return df

train_df = create_text_and_target(train_df, is_train=True)
test_df = create_text_and_target(test_df, is_train=False)

# Embedding generation and search
embed_model = SentenceTransformer(CFG.EMBEDDING_MODEL_PATH, device=CFG.DEVICE)

train_embeddings = embed_model.encode(train_df['full_text'].tolist(), show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
test_embeddings = embed_model.encode(test_df['full_text'].tolist(), show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)

# Searching for the nearest neighbors using util.semantic_search
hits = util.semantic_search(test_embeddings, train_embeddings, top_k=CFG.TOP_K_CANDIDATES)

# Forming a list of candidates
candidates = []
for hit_list in tqdm(hits, desc="Collecting candidates"):
    unique_targets = train_df.iloc[[h['corpus_id'] for h in hit_list]]['target'].unique().tolist()
    candidates.append(unique_targets)

test_df['candidates'] = candidates
rerank_df = test_df.explode('candidates').rename(columns={'candidates': 'candidate_target'}).dropna(subset=['candidate_target'])

del embed_model, train_embeddings, test_embeddings, hits
gc.collect(); torch.cuda.empty_cache()

print(f"The candidates are generated. Number of pairs for re-ranking: {len(rerank_df)}")


reranker_model = CrossEncoder(CFG.RERANKER_MODEL_PATH, max_length=512, device=CFG.DEVICE)

eos_token_id = reranker_model.tokenizer.eos_token_id

reranker_model.tokenizer.pad_token = reranker_model.tokenizer.eos_token
reranker_model.tokenizer.padding_side = "right"

reranker_model.model.config.pad_token_id = eos_token_id

print("The configuration of the model and tokenizer has been fixed. Launching the prediction...")
reranker_input = rerank_df[['full_text', 'candidate_target']].values.tolist()
rerank_df['reranker_score'] = reranker_model.predict(reranker_input, show_progress_bar=True, batch_size=CFG.BATCH_SIZE*2)

print("Scores from Qwen Re-ranker have been successfully received.")

del reranker_model
gc.collect(); torch.cuda.empty_cache()


sys.setrecursionlimit(3000)

try:
    ops_file_path = "/usr/local/lib/python3.11/dist-packages/bitsandbytes/_ops.py"
    with open(ops_file_path, 'r') as f:
        ops_content = f.read()

    if "from .cextension import ipex_cpu, ipex_xpu" in ops_content:
        patched_content = ops_content.replace(
            "from .cextension import ipex_cpu, ipex_xpu",
            "ipex_cpu = False\nipex_xpu = False\n# Patched by user"
        )
        with open(ops_file_path, 'w') as f:
            f.write(patched_content)
        print("âœ… The file _ops.py patched: ipex_cpu replaced with False.")
    else:
        print("âœ… The file _ops.py already patched.")
except Exception as e:
    print(f"âš ï¸� Patch error _ops.py: {e}")

try:
    import bitsandbytes as bnb
    print(f"âœ… bitsandbytes has been imported. Version: {bnb.__version__}")
except Exception as e:
    print(f"â�Œ bitsandbytes import error: {e}")
    raise e

def fake_is_bitsandbytes_available():
    print("âš ï¸� is_bitsandbytes_available() bypassed (always True)")
    return True

transformers.utils.is_bitsandbytes_available = fake_is_bitsandbytes_available
print("âœ… The is_bitsandbytes_available check has been replaced with a workaround.")

try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    print("âœ… BitsAndBytesConfig created successfully.")
except Exception as e:
    print(f"â�Œ Error when creating BitsAndBytesConfig: {e}")
    raise e

try:
    llm_model = AutoModelForCausalLM.from_pretrained(
        CFG.LLM_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(CFG.LLM_MODEL_PATH, trust_remote_code=True)
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
    print("âœ… The model and tokenizer are loaded.")
except Exception as e:
    print(f"â�Œ Error loading the model: {e}")
    raise e

print("Search for ID tokens 'Yes' and 'No'...")

vocab_size = llm_model.config.vocab_size
print(f"âœ… The size of the model's dictionary: {vocab_size}")

yes_tokens = llm_tokenizer.encode("Yes", add_special_tokens=False)
no_tokens = llm_tokenizer.encode("No", add_special_tokens=False)

yes_token_id = -1
no_token_id = -1

if yes_tokens and yes_tokens[0] < vocab_size:
    yes_token_id = yes_tokens[0]
else:
    print(f"âš ï¸� The ID for 'Yes' is invalid. Looking for an alternative...")
    alt_yes_tokens = llm_tokenizer.encode(" true", add_special_tokens=False)
    if alt_yes_tokens and alt_yes_tokens[0] < vocab_size:
        yes_token_id = alt_yes_tokens[0]
    else:
        yes_token_id = llm_tokenizer.eos_token_id

if no_tokens and no_tokens[0] < vocab_size:
    no_token_id = no_tokens[0]
else:
    print(f"âš ï¸� The ID for 'No' is invalid. Looking for an alternative...")
    alt_no_tokens = llm_tokenizer.encode(" false", add_special_tokens=False)
    if alt_no_tokens and alt_no_tokens[0] < vocab_size:
        no_token_id = alt_no_tokens[0]
    else:
        no_token_id = llm_tokenizer.eos_token_id


print(f"âœ… The final ID for 'Yes': {yes_token_id}")
print(f"âœ… The final ID for 'No': {no_token_id}")

def get_llm_scores(df):
    scores = []
    for i in tqdm(range(0, len(df), CFG.BATCH_SIZE), desc="Qwen2.5 Math Re-ranking"):
        batch_df = df.iloc[i:i+CFG.BATCH_SIZE]
        prompts = [
            f"Instruction: Does the misconception label '{row['candidate_target']}' accurately describe the student's reasoning in '{row['full_text']}'? Answer with only Yes or No.\nAnswer:"
            for _, row in batch_df.iterrows()
        ]
        inputs = llm_tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(CFG.DEVICE)
        
        with torch.no_grad():
            outputs = llm_model(**inputs)
            
            last_token_logits = outputs.logits[:, -1, :] 
        
        yes_logits = last_token_logits[:, yes_token_id]
        no_logits = last_token_logits[:, no_token_id]
        batch_scores = (yes_logits - no_logits).cpu().float().numpy()
        scores.extend(batch_scores)
    return scores

rerank_df['qwen_math_score'] = get_llm_scores(rerank_df)
print("âœ… Scores from Qwen2.5 Math are obtained.")


print("Normalization of ratings from re-rankers...")
rerank_df['reranker_score_norm'] = (rerank_df['reranker_score'] - rerank_df['reranker_score'].min()) / (rerank_df['reranker_score'].max() - rerank_df['reranker_score'].min())
rerank_df['qwen_math_score_norm'] = (rerank_df['qwen_math_score'] - rerank_df['qwen_math_score'].min()) / (rerank_df['qwen_math_score'].max() - rerank_df['qwen_math_score'].min())

print(f"Calculating the final score with weights: Qwen Re-ranker = {CFG.RERANKER_WEIGHT}, Qwen Math = {CFG.QWEN_MATH_WEIGHT}")
rerank_df['final_score'] = (CFG.RERANKER_WEIGHT * rerank_df['reranker_score_norm']) + \
                         (CFG.QWEN_MATH_WEIGHT * rerank_df['qwen_math_score_norm'])

print("Sorting of candidates by final grade...")
rerank_df = rerank_df.sort_values(by=['row_id', 'final_score'], ascending=[True, False])

submission = rerank_df.groupby('row_id')['candidate_target'].apply(lambda x: ' '.join(x.head(3))).reset_index()
submission.rename(columns={'candidate_target': 'Category:Misconception'}, inplace=True)

sub_template = pd.read_csv(CFG.SUBMISSION_PATH)
final_submission = sub_template[['row_id']].merge(submission, on='row_id', how='left')

most_common_label = train_df['target'].mode()[0]
final_submission['Category:Misconception'] = final_submission['Category:Misconception'].fillna(most_common_label)

final_submission.to_csv('submission.csv', index=False)

print(f"\nâœ… The file submission.csv has been successfully created! The gaps are filled with the value: '{most_common_label}'")


final_submission.iloc[0]

