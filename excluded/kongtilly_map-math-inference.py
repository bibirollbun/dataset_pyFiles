# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# !pip install -U peft


from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json, os


BASE_MODEL = "/kaggle/input/qwen-3/transformers/4b/1"
ADAPTER_DIR = "/kaggle/input/llama-finetune-qwen3-4b-map_math-v2/transformers/default/1"  # model

TEST_CSV = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
LABEL_MAP = "/kaggle/input/label-mapping/label_mapping.json"
SUBMIT = "/kaggle/working/submission.csv"


# model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

# LoRA adapter
model = PeftModel.from_pretrained(model, ADAPTER_DIR, torch_dtype=model.dtype)



model.eval()


# # mapping → "Category:Misconception"
# train_saved = pd.read_csv(TRAIN_SAVED_CSV, dtype=str)
# train_saved = train_saved.dropna(subset=["special_label", "target"])
# train_saved["special_label"] = train_saved["special_label"].str.strip()

# # filter out the 'label'.
# train_saved = train_saved[train_saved["special_label"].map(lambda x: isinstance(x, str) and len(x) == 1)]
# label_map = dict(zip(train_saved["special_label"], train_saved["target"]))
# unique_labels = sorted(label_map.keys())


with open(LABEL_MAP, "r", encoding="utf-8") as f:
    LMAP = json.load(f)

valid_labels = LMAP["valid_labels"]
label_to_mis = LMAP["label_to_misconception"]
mis_to_label = LMAP["misconception_to_label"]
label_to_token_id = LMAP.get("label_to_token_id", {})  

label_map = label_to_mis
unique_labels = sorted(valid_labels)


print(label_map,unique_labels)


import pandas as pd

df = pd.read_csv(TEST_CSV, dtype=str).fillna("")

# unique_labels = sorted(train_saved['special_label'].unique())

labels_str = ", ".join([f"'{lab}'" for lab in valid_labels])
SYS_PROMPT = f"""
You are a math education specialist.

INPUT
Question: [text]
Selected Answer: [answer]
Student Explanation: [text]

OUTPUT
Return EXACTLY ONE character from: {labels_str}
That character maps to: [Answer_Correctness]_[Explanation_Quality]:[Misconception_Type]

HARD RULES (must follow exactly)
1) Answer_Correctness (True/False) must be decided using ONLY Question and Selected Answer.
2) Explanation_Quality and Misconception_Type must be decided using ONLY Student Explanation.
3) You MUST NEVER use Student Explanation to decide Answer_Correctness. If you do, you will output the single character '#', which is reserved for INVALID.
4) If the Question contains enough information to compute the correct numeric answer (e.g., counts, totals, "simplest form" instruction), compute it and determine True/False accordingly.
5) If Explanation_Quality = Misconception, choose one canonical type (Wrong_fraction, Inversion, Not_simplified, Counting_error, Part_to_part, Irrelevant, Unknowable, etc.)

DECISION GUIDES (short)
- Wrong_fraction: denominator = subset, not whole.
- Inversion: numerator/denominator swapped.
- Not_simplified: final numeric fraction not reduced when asked.
- Counting_error: explicit wrong counts.
- Neither/Unknowable: no math reasoning.

OUTPUT STRICTNESS
- Only one character, no whitespace, no punctuation, no explanation.
- Use '#' only when you violated rule 3 or cannot decide after following rules.

END
"""



#01 SYS_PROMPT = f"""
# You are a math education specialist analyzing student multiple-choice work and explanations.

# ## Input Format
# Question: [text]
# Selected Answer: [answer]
# Student Explanation: [text]

# ## Task
# Classify into ONE category: [Answer_Correctness]_[Explanation_Quality]:[Misconception_Type]

# ## Output (STRICT)
# - Respond with EXACTLY ONE character from: {labels_str}
# - This single character maps to a complete label via a fixed dictionary.

# ## Rules
# 1) Answer_Correctness is based on the math in the Question vs Selected Answer.
# 2) Explanation_Quality is based ONLY on the Student Explanation.
# 3) If Misconception, choose the specific type consistent with the explanation.
# """


#02 SYS_PROMPT = f"""
# You are a math education specialist analyzing student multiple-choice work and explanations.

# ## Input Format
# Question: [text with counts/descriptions]
# Selected Answer: [answer]
# Student Explanation: [text]

# ## Task Overview
# Produce ONE final category from the set {labels_str}, where each maps to:
# [Answer_Correctness]_[Explanation_Quality]:[Misconception_Type]

# ## Subtasks (follow strictly):
# A. Answer correctness (True/False):
# - Determine if Selected Answer matches the mathematically correct answer to the Question (use totals, counts, and simplest form when specified).
# - True = selected answer is mathematically correct. False = otherwise.

# B. Explanation quality (Correct/Misconception/Neither):
# - Correct: reasoning is mathematically sound and relevant.
# - Misconception: clear, specific mathematical error in the reasoning.
# - Neither: unclear, irrelevant, or insufficient reasoning.

# C. Misconception type (if Misconception):
# - Use canonical types. Apply these rules:
#   - Wrong_fraction: denominator is not the whole; uses a subset (e.g., shaded parts) instead of total.
#   - Inversion: swaps numerator and denominator for the intended ratio.
#   - Not_simplified: fraction not in simplest form when requested.
#   - Counting_error: miscounts parts (numerator or total).
#   - Part_to_part: compares two parts instead of part-to-whole.
#   - Other types only when clearly indicated by the explanation.

# ## Decision Rules (critical):
# - If counts are correct (e.g., “3 white, 6 blue”) but denominator uses a subset (e.g., 6), choose Wrong_fraction (NOT Inversion).
# - If the question requests simplest form, and reasoning gives a correct fraction but unsimplified, choose Not_simplified (Explanation_Quality may still be Correct if the math is sound).
# - Do NOT infer Correctness from explanation plausibility—compute it from the Question + Selected Answer.

# ## Output (STRICT)
# - Respond with EXACTLY ONE character from: {labels_str}
# - That single character maps to [Answer_Correctness]_[Explanation_Quality]:[Misconception_Type]
# - No extra text, no mention of answer correctness in the output.
# """


# set of token
allowed_ids = []
label_for_id = {}
for ch in unique_labels:
    ids = tokenizer.encode(ch, add_special_tokens=False)
    if len(ids) == 1:
        tid = ids[0]
        allowed_ids.append(tid)
        label_for_id[tid] = ch
        
# safety check
assert len(allowed_ids) >= 3, f"Insufficient number of valid tag tokens: {len(allowed_ids)}"

allowed_ids = torch.tensor(allowed_ids, device=next(model.parameters()).device)


#  Prompt chat（as training）
def build_messages(row):
    sys_prompt=SYS_PROMPT
    user_content = f"""Question: {row.get('QuestionText', '')}
                        Selected Answer: {row.get('MC_Answer', '')}
                        Student Explanation: {row.get('StudentExplanation', '')}"""
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]

def tokenize_messages(batch_rows, max_len=1024):
    texts = [
        tokenizer.apply_chat_template(
            build_messages(r), tokenize=False, add_generation_prompt=True
        )
        for r in batch_rows
    ]
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len
    )
    return {k: v.to(next(model.parameters()).device) for k, v in enc.items()}


# only take the top-3 from allowed_ids
def topk_on_allowed(logits, k=3):
    # logits: [batch, vocab]
    device = logits.device
    mask = torch.full_like(logits, float("-inf"))
    mask[:, allowed_ids] = logits[:, allowed_ids]  
    probs = torch.softmax(mask, dim=-1)
    topk = torch.topk(probs, k=k, dim=-1)
    top_ids = topk.indices  # [batch, k]
    # map to label characters
    chars = []
    for row_ids in top_ids:
        cands = []
        for tid in row_ids.tolist():
            # if the tid is not in label_for_id, then skip
            ch = label_for_id.get(tid, None)
            if ch is not None:
                cands.append(ch)
        # remove duplicates and maintain order
        dedup = []
        seen = set()
        for ch in cands:
            if ch not in seen:
                seen.add(ch)
                dedup.append(ch)
        chars.append(dedup[:k])
    return chars


def infer_batch(rows, k=3):
    enc = tokenize_messages(rows)
    with torch.no_grad():
        out = model(**enc)
        # Take the logits of 'assistant first generated token'
        last_logits = out.logits[:, -1, :]  # [batch, vocab]
        top_chars = topk_on_allowed(last_logits, k=k)  # list of list of chars
    return top_chars


# Read the test.csv and generate the submission.csv
test_df = pd.read_csv(TEST_CSV, dtype=str).fillna("")
rows = test_df.to_dict(orient="records")

BATCH = 32
pred_lines = []
for i in range(0, len(rows), BATCH):
    batch = rows[i:i+BATCH]
    top_chars = infer_batch(batch, k=3)
    for r, chars in zip(batch, top_chars):
        # map to "Category:Misconception"
        mapped = [label_map[c] for c in chars if c in label_map]
        # take the top-3
        mapped_uniq = []
        seen = set()
        for m in mapped:
            if m not in seen:
                seen.add(m)
                mapped_uniq.append(m)
        pred_lines.append({
            "row_id": r.get("row_id", ""),
            "Category:Misconception": " ".join(mapped_uniq[:3])
        })

out_df = pd.DataFrame(pred_lines, columns=["row_id", "Category:Misconception"])
out_df.to_csv(SUBMIT, index=False)
print(f"Saved to {SUBMIT}")


out_df


pred_lines


#01 [{'row_id': '36696',
#   'Category:Misconception': 'False_Misconception:Positive True_Misconception:Inversion True_Misconception:SwapDividend'},
#  {'row_id': '36697',
#   'Category:Misconception': 'False_Misconception:Division False_Misconception:Irrelevant False_Misconception:FlipChange'},
#  {'row_id': '36698',
#   'Category:Misconception': 'True_Misconception:Multiplying_by_4 True_Misconception:Not_variable True_Misconception:Mult'}]


#02 [{'row_id': '36696',
#   'Category:Misconception': 'False_Misconception:Wrong_fraction True_Misconception:Base_rate True_Misconception:Inversion'},
#  {'row_id': '36697',
#   'Category:Misconception': 'False_Correct:NA False_Misconception:Incomplete False_Misconception:Longer_is_bigger'},
#  {'row_id': '36698',
#   'Category:Misconception': 'True_Misconception:Wrong_term True_Neither:NA False_Correct:NA'}]

