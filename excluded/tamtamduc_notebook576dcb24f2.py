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


!pip install -q transformers datasets accelerate peft bitsandbytes trl


!pip install tensorboard



import transformers
import peft
import trl

print(f"Transformers version: {transformers.__version__}")
print(f"PEFT version: {peft.__version__}")
print(f"TRL version: {trl.__version__}")


import torch
import torch.nn as nn
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected!")


df = pd.read_csv("/kaggle/input/train-fixed/train_fixed.csv",keep_default_na=False)

df['row_id'] = df['row_id'].astype('int64')
df['QuestionId'] = df['QuestionId'].astype('int64')
df['QuestionText'] = df['QuestionText'].astype('str')
df['MC_Answer'] = df['MC_Answer'].astype('str')
df['StudentExplanation'] = df['StudentExplanation'].astype('str')
df['Category'] = df['Category'].astype('str')
df['Misconception'] = df['Misconception'].astype('str')



misconceptions = []
for i in range(len(df)):
  if misconceptions.count(df.iloc[i]['Misconception'])==0 and ('Misconception' in df.iloc[i]['Category']) :
    misconceptions.append(df.iloc[i]['Misconception'])
print(len(misconceptions))


labels = ["True_Neither","False_Neither","True_Correct","False_Correct"]

for i in misconceptions :
  candidate1 = str("True_Misconception"+":"+i)
  candidate2=str("False_Misconception"+":"+i)
  labels.append(candidate1)
  labels.append(candidate2)
# print(label)
print(len(labels))


def tokenize_input(row, category, misconception, tokenizer):
    """
    Hàm này tạo prompt từ dữ liệu đầu vào, tokenize nó, và trả về một
    dictionary chứa các PyTorch tensor sẵn sàng để đưa vào model.
    """
    # Trích xuất dữ liệu
    q_text, mc_answer, explanation = row["QuestionText"], row["MC_Answer"], row["StudentExplanation"]

    # Tạo prompt theo template của Qwen2 Instruct
    prompt = f"""<|im_start|>system
You are a meticulous educational analyst and expert in mathematics pedagogy. Your task is to perform a verification check. You will be given a student's response to a math problem, and a proposed classification for that response. You must determine if the proposed classification is entirely accurate based on the evidence.
DEFINITIONS OF THE CLASSIFICATION LABELS:
category: This is a compound label with two parts, separated by an underscore: Correctness_ReasoningType.

Part 1: Correctness (True or False): This describes whether the student's mc_answer is objectively the correct solution to the q_text.

Part 2: ReasoningType (Correct, Misconception, or Neither): This describes the quality of the student's explanation:
Correct: The explanation shows sound, logical, and mathematically valid reasoning.
Misconception: The explanation reveals a specific, identifiable error in conceptual understanding.
Neither: The explanation is incorrect, but does not point to a specific misconception. It could be a guess, irrelevant, or simply nonsensical.

misconception: This is a text description of the specific thinking error. It is only relevant when the ReasoningType in the category is Misconception. If the category is ..._Correct or ..._Neither, this field's value should be "NA"

YOUR STEP-BY-STEP VERIFICATION PROCESS (Chain-of-Thought):

1. Analyze Answer Correctness (True/False Check): First, independently solve the math problem in {q_text}. Compare your result to the student's {mc_answer}. Is the student's answer objectively True (correct) or False (incorrect)?
2. Analyze Explanation Quality (Reasoning Check): Now, ignore the final answer and focus only on the {explanation}.
Deconstruct the student's logic. What steps did they follow?
Based only on their text, classify their reasoning: Is it Correct, a clear Misconception, or Neither?
If you identify a misconception, briefly describe it in your own words.
3. Compare Your Analysis to the Provided Labels: Now, compare your findings from steps 1 and 2 with the given {category} and {misconception}.
Does your True/False conclusion from Step 1 match the first part of the {category} label?
Does your Correct/Misconception/Neither conclusion from Step 2 match the second part of the {category} label?
If the category is ..._Misconception, does the student's error you identified align with the provided {misconception} text?
4. Final Conclusion: A "Yes" is only possible if all checks in Step 3 pass. If there is any mismatch at any point, the answer must be "No".
Show your detailed reasoning by following these steps. Then, on the very last line, provide the final answer as exactly one word: "Yes" or "No".
<|im_end|>
<|im_start|>user
Problem Data:
Question: {q_text}
Student's Answer: {mc_answer}
Student's Explanation: {explanation}
Proposed Classification:
Category: '{category}'
Misconception: '{misconception}'
Verification Task:
Based on the data and the definitions provided, is the 'Proposed Classification' an accurate description of the 'Problem Data'?
<|im_end|>
<|im_start|>assistant
"""
    # Thực hiện tokenization trong một bước duy nhất
    # return_tensors="pt" sẽ tạo ra đầu ra là PyTorch tensor
    tokenized_output = tokenizer(
        prompt,
        truncation=True,        # Cắt bớt nếu prompt quá dài
        max_length=3096,        # Độ dài tối đa
        return_tensors="pt",    # **Đây là thay đổi quan trọng nhất**
    )
    
    return tokenized_output


from datasets import Dataset , load_from_disk
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# import numpy as np



MODEL_NAME= "Qwen/Qwen2.5-7B-Instruct"
# TRAIN_CSV_PATH = "/train_with_misconceptions.csv" 
OUTPUT_DIR = "./qwen2-7b-ranker-finetuned"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", trust_remote_code=True
)
yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]


decode_steps=0


def infer(row,category,misconception,tokenizer,decode_steps):
    inputs = None
    outputs = None
    with torch.no_grad():
        inputs = tokenize_input(row,category,misconception,tokenizer)
        outputs = model(**inputs)
    logits = outputs.logits
    last_token_logits = logits[:, -1, :]
    yes_scores = last_token_logits[:, yes_token_id]
    no_scores = last_token_logits[:, no_token_id]
    scores = yes_scores - no_scores
    if decode_steps < 5 :
        predictions = torch.argmax(logits, dim=-1)
        decoded_inputs = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        decoded_preds = tokenizer.decode(predictions[0], skip_special_tokens=False)
        
        print("================= DEBUG=================")
        print(f"--- INPUT ---:\n{decoded_inputs}")
        print(f"--- Response ---:\n{decoded_preds}")
        print("============================================")
        decode_steps+=1
    result = str(category+":"+misconception)
    return scores , result
    





def predict(dataset):
    for row in dataset :
        ranking_candidate = {}
        for label in labels :
            category,misconception = None,None
            if "Neither" in label or "Correct" in label:
                category = str(label)
                misconception = str("NA")
            else:
                parts = label.split(":")
                category = str(parts[0])
                misconception = str(parts[1])
            score , result = infer(row,category,misconception,tokenizer,decode_steps)
            ranking_candidate[result] = score
                
        sorted_candidate_desc = dict(sorted(ranking_candidate.items(), key=lambda x: x[1], reverse=True))
        top_3_candidates = sorted_candidate_desc[:3]
        top_3_keys = [k for k, v in top_3_candidates]
        row['Category:Misconception'] = str(top_3_candidates[0]+" "+top_3_candidates[1]+" "+top_3_candidates[2])
                


test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv",keep_default_na=False)



test_dataset = Dataset.from_pandas(test)
test_dataset= predict(test_dataset)
for row in test_dataset:
    for column_name in remove_columns :
        row.pop(column_name,None)


submit_df = pd.DataFrame(test_dataset)
submit_df.to_csv('submission.csv', index=False)


submit_df.head()







