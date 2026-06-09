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


import torch
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "/kaggle/input/qwen-3/transformers/4b/1"
ADAPTER_DIR = "/kaggle/input/qwen3-4b-map_math-v2/transformers/default/1"

LABEL_MAPPING_JSON = "/kaggle/input/label-mapping/label_mapping.json"
TEST_CSV = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
OFFLOAD_DIR = "/kaggle/working/offload"


# model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",                 #  GPU/CPU
    offload_folder=OFFLOAD_DIR,
    low_cpu_mem_usage=True
)

# 合并 LoRA，减少推理显存
model = PeftModel.from_pretrained(model, ADAPTER_DIR).merge_and_unload()
model.eval()


# label_to_token
with open(LABEL_MAPPING_JSON, "r", encoding="utf-8") as f:
    label_mapping = json.load(f)

valid_labels = label_mapping["valid_labels"]
label_to_misconception = label_mapping["label_to_misconception"]

def verify_labels_against_tokenizer(tokenizer, labels):
    valid, mapping = [], {}
    for lab in labels:
        ids = tokenizer.encode(lab, add_special_tokens=False)
        if len(ids) == 1 and tokenizer.decode(ids) == lab:
            valid.append(lab)
            mapping[lab] = ids[0]
    return valid, mapping

valid_labels, label_to_token_id = verify_labels_against_tokenizer(tokenizer, valid_labels)
allowed_ids = torch.tensor(list(label_to_token_id.values()), device=next(model.parameters()).device)
label_for_id = {v: k for k, v in label_to_token_id.items()}


labels_str = ", ".join([f"'{lab}'" for lab in valid_labels])
SYS_PROMPT = f"""You are a math education specialist analyzing ONLY student explanations for reasoning errors.

## Input Format
Question: [text]
Selected Answer: [answer]
Student Explanation: [text]

## Your Task
1. ANALYZE EXCLUSIVELY the Student Explanation's reasoning
2. CLASSIFY into ONE category:
   - "Correct": Explanation shows accurate reasoning (even if brief)
   - "Misconception": Explanation contains specific conceptual errors
   - "Neither": Explanation unclear/irrelevant/no reasoning
3. IF "Misconception", SPECIFY error type (e.g., Fraction_simplification)

## Output Format (STRICT)
- Respond with EXACTLY ONE character from: {labels_str}
- This single character maps to a complete label: [Answer_Correctness]_[Explanation_Quality]:[Misconception_Type]

## CRITICAL INSTRUCTIONS
1. NEVER consider answer correctness - it is irrelevant to your task
2. NEVER output anything about answer correctness
3. Correct reasoning = Correct:NA (even if answer was wrong)
4. Incorrect reasoning = Misconception:[Type] (even if answer was right)
5. If explanation is "1/2 is simplest form of 3/6" for unshaded fraction → Correct:NA

## CONSEQUENCES OF FAILURE
- If you output ANYTHING mentioning answer correctness → INVALID
- If you output multiple characters → INVALID
- Your response MUST be a single character from {labels_str}"""

SYS_PROMPT_TEXT = SYS_PROMPT.strip()


#03 labels_str = ", ".join([f"'{lab}'" for lab in valid_labels])
# SYS_PROMPT = f"""
# You are a math education specialist. Classify student responses into ONE label token.

# Label format (behind the scenes):
# [Answer_Correctness]_[Explanation_Quality]:[Misconception_Type]
# But you MUST output exactly ONE character from the allowed set we provide.

# Strict rules:
# 1) Determine the CORRECT mathematical answer to the Question first (compute or reason).
# 2) Compare Selected Answer (MC_Answer) with the correct answer (respect simplest form for fractions, decimal equivalence, and algebraic equality).
#    - If MC_Answer ≠ correct answer (e.g., 3/6 vs 1/3), set Answer_Correctness=False.
#    - If MC_Answer = correct answer, set Answer_Correctness=True.
# 3) Judge Explanation_Quality ONLY from Student Explanation:
#    - Good: clear, mathematically sound reasoning.
#    - Bad: reasoning shows misconceptions.
#    - Neither: unclear/irrelevant/missing.
#    - If Bad, provide a specific Misconception_Type; otherwise Misconception_Type=NA.
# 4) Output exactly ONE allowed character from our label set that encodes the full triplet.
# 5) Do NOT output words, multiple characters, or anything outside the allowed set.

# Input fields:]
# - Selected Answer (MC_Answer): [text, possibly LaTeX]
# - Student Explanation: [text]

# Notes:
# - Always reduce fractions to simplest form (e.g., 4/8 → 1/2, 6/9 → 2/3).
# - Consider decimal equivalence (5.20 == 5.2).
# - Consider algebraic equivalence (1+x == x+1).
# """

# SYS_PROMPT_TEXT = SYS_PROMPT.strip()
# - Question: [text


#02 SYS_PROMPT = f"""
# You are a math education specialist. Your task is to classify student responses into a single label.

# ## Input
# - Question
# - Selected Answer (MC_Answer)
# - Student Explanation

# ## Label Format
# [Answer_Correctness]_[Explanation_Quality]:[Misconception_Type]

# ## Rules
# 1. [Answer_Correctness] must be judged ONLY by comparing the Selected Answer with the correct mathematical solution.
#    - If the Selected Answer is mathematically wrong, set [Answer_Correctness] = False.
#    - If the Selected Answer is mathematically correct, set [Answer_Correctness] = True.
# 2. [Explanation_Quality] must be judged from the Student Explanation:
#    - "Good" if reasoning is clear and mathematically sound.
#    - "Bad" if reasoning shows misconceptions.
#    - "Neither" if unclear, irrelevant, or missing.
# 3. [Misconception_Type] is required only if Explanation_Quality = "Bad".
#    - Otherwise output "NA".
# 4. Output must be EXACTLY one character from the allowed label set (mapping provided).

# """


#01 SYS_PROMPT = f"""
# You are a math education specialist analyzing ONLY student explanations for reasoning errors.

# ## Input Format
# Question: [text]
# Selected Answer: [answer]
# Student Explanation: [text]

# ## Your Task
# 1. ANALYZE EXCLUSIVELY the Student Explanation's reasoning
# 2. CLASSIFY into ONE category:
#    - "Correct": Explanation shows accurate reasoning
#    - "Misconception": Explanation contains specific conceptual errors
#    - "Neither": Explanation unclear/irrelevant/no reasoning
# 3. IF "Misconception", SPECIFY error type

# ## Output Format (STRICT)
# - Respond with EXACTLY ONE character from: {labels_str}
# - This single character maps to a complete label: [Answer_Correctness]_[Explanation_Quality]:[Misconception_Type]

# ## CRITICAL INSTRUCTIONS
# - NEVER consider answer correctness
# - NEVER output multiple characters
# - Response MUST be a single character from {labels_str}
# """


def build_messages(row):
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"Question: {row['QuestionText']} Selected Answer: {row['MC_Answer']} Student Explanation: {row['StudentExplanation']}"}
    ]

def tokenize_messages(batch_rows, max_len=1024):
    texts = [tokenizer.apply_chat_template(build_messages(r), tokenize=False, add_generation_prompt=True) for r in batch_rows]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return {k: v.to(next(model.parameters()).device) for k, v in enc.items()}


# inference
def infer_batch(rows, k=3):
    enc = tokenize_messages(rows)
    with torch.no_grad():
        out = model(**enc)
        last_logits = out.logits[:, -1, :]
        mask = torch.full_like(last_logits, float("-inf"))
        mask[:, allowed_ids] = last_logits[:, allowed_ids]
        probs = torch.softmax(mask, dim=-1)
        topk = torch.topk(probs, k=min(k, len(allowed_ids)), dim=-1)
        chars = []
        for row_ids in topk.indices:
            cands = [label_for_id[tid.item()] for tid in row_ids if tid.item() in label_for_id]
            chars.append(cands)
        return chars


test_df = pd.read_csv(TEST_CSV, dtype=str).fillna("")
rows = test_df.to_dict(orient="records")
BATCH = 16
pred_lines = []

for i in range(0, len(rows), BATCH):
    batch = rows[i:i+BATCH]
    top_chars = infer_batch(batch, k=3)
    for r, chars in zip(batch, top_chars):
        mapped = [label_to_misconception[c] for c in chars if c in label_to_misconception]
        mapped = list(dict.fromkeys(mapped))[:3]
        if not mapped:
            mapped = ["NA"]
        pred_lines.append({"row_id": r["row_id"], "Category:Misconception": " ".join(mapped)})

out_df = pd.DataFrame(pred_lines, columns=["row_id", "Category:Misconception"])
out_df.to_csv("submission.csv", index=False)


pred_lines


#03 [{'row_id': '36696',
#   'Category:Misconception': 'True_Misconception:Base_rate False_Misconception:Adding_terms False_Misconception:Wrong_fraction'},
#  {'row_id': '36697',
#   'Category:Misconception': 'True_Misconception:Positive True_Neither:NA True_Misconception:Multiplying_by_4'},
#  {'row_id': '36698',
#   'Category:Misconception': 'True_Misconception:Subtraction True_Misconception:Positive True_Misconception:SwapDividend'}]


#01 [{'row_id': '36696',
#   'Category:Misconception': 'True_Misconception:Inversion True_Misconception:Irrelevant False_Misconception:WNB'},
#  {'row_id': '36697',
#   'Category:Misconception': 'True_Misconception:Positive True_Misconception:Multiplying_by_4 True_Misconception:Whole_numbers_larger'},
#  {'row_id': '36698',
#   'Category:Misconception': 'True_Misconception:Shorter_is_bigger True_Misconception:Subtraction True_Misconception:Positive'}]


#02 [{'row_id': '36696',
#   'Category:Misconception': 'True_Misconception:Base_rate True_Misconception:Subtraction False_Misconception:Positive'},
#  {'row_id': '36697',
#   'Category:Misconception': 'True_Misconception:Positive True_Misconception:Mult True_Misconception:Multiplying_by_4'},
#  {'row_id': '36698',
#   'Category:Misconception': 'True_Misconception:SwapDividend True_Misconception:Tacking True_Misconception:Not_variable'}]


batch







