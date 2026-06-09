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


# !pip install --no-deps /kaggle/input/kaggle-package/transformers-4.55.2-py3-none-any.whl
# !pip install --no-deps  /kaggle/input/kaggle-package/datasets-4.0.0-py3-none-any.whl
# !pip install --no-deps -q /kaggle/input/kaggle-package/accelerate-1.10.0-py3-none-any.whl
# !pip install --no-deps -q /kaggle/input/kaggle-package/peft-0.17.0-py3-none-any.whl
!pip install --no-deps -q /kaggle/input/kaggle-package/bitsandbytes-0.47.0-py3-none-manylinux_2_24_x86_64.whl
!pip install --no-deps -q /kaggle/input/kaggle-package/trl-0.21.0-py3-none-any.whl



!pip install --no-deps -q /kaggle/input/kaggle-package/tensorboard-2.20.0-py3-none-any.whl



import transformers
import peft
import trl

print(f"Transformers version: {transformers.__version__}")
print(f"PEFT version: {peft.__version__}")
print(f"TRL version: {trl.__version__}")


from sklearn.preprocessing import LabelEncoder
import ast





import torch
import torch.nn as nn
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from transformers import AutoModel, AutoTokenizer


state_dict = torch.load("/kaggle/input/retriver/transformers/default/1/best_model_proposed_110.pth", map_location="cuda")
model_retrieve_path = '/kaggle/input/mathbert_model/transformers/default/1'
tokenizer_retrieve_path = '/kaggle/input/mathbert_tokenizer/transformers/default/1'
tokenizer_retrieve = AutoTokenizer.from_pretrained(tokenizer_retrieve_path)
model_retrieve = AutoModel.from_pretrained(model_retrieve_path).to("cuda")
model_retrieve = torch.nn.DataParallel(model_retrieve)
model_retrieve.load_state_dict(state_dict)


model_retrieve


def prepare_df1(path):
    # path = args.path
    df = pd.read_csv(path)
    df['text_train'] = df.apply(
    lambda row: f"[CLS] Question: {row['QuestionText']}\n[SEP] Student's Answer: {row['MC_Answer']}\n[SEP] Student's explanation: {row['StudentExplanation']}[SEP]\n ",
    axis=1
    )
    return df


from datasets import Dataset
import ast
import torch.nn.functional as F


# test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv",keep_default_na=False)
test= prepare_df1("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
test_dataset = Dataset.from_pandas(test)



embedded_training_dataset_path = '/kaggle/input/training-dataset-embedding/embeddings_test190.csv'
infer_data = pd.read_csv(embedded_training_dataset_path)
# print(infer['label'][0])
infer_embeds_list = infer_data['text_embed'].apply(lambda x: torch.tensor(ast.literal_eval(x), dtype=torch.float32))
infer_embeds = torch.stack(list(infer_embeds_list.values))
infer_embeds = F.normalize(infer_embeds, dim=1).to(device)
infer_labels = infer_data['label']


infer_data.iloc[0]


print(infer_embeds[0])
print(infer_labels[0])


def retrieve(test_row,tokenizer,model):
    candidate_labels=[]
    text = test_row['text_train']
    tokenized_text= tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=512,
                        return_tensors="pt"
                    )
    tokenized_text = tokenized_text.to(device)
    with torch.no_grad():
        output = model(**tokenized_text)
    query = output.last_hidden_state[:, 0, :]          
    query = F.normalize(query, dim=1)                 
        
            
    sims = torch.matmul(query, infer_embeds.T).squeeze(0)  
    
    sorted_idx = torch.argsort(sims, descending=True)
    
    seen_labels = set()
    top_results = []
    for idx in sorted_idx.tolist():
        lbl = infer_labels[idx]
        if lbl not in seen_labels:
            score = sims[idx].item()
            top_results.append(lbl)
            seen_labels.add(lbl)
        if len(top_results) == 10:
            break
    return top_results


# df = pd.read_csv("/kaggle/input/train-fixed/train_fixed.csv",keep_default_na=False)

# df['row_id'] = df['row_id'].astype('int64')
# df['QuestionId'] = df['QuestionId'].astype('int64')
# df['QuestionText'] = df['QuestionText'].astype('str')
# df['MC_Answer'] = df['MC_Answer'].astype('str')
# df['StudentExplanation'] = df['StudentExplanation'].astype('str')
# df['Category'] = df['Category'].astype('str')
# df['Misconception'] = df['Misconception'].astype('str')



# misconceptions = []
# for i in range(len(df)):
#   if misconceptions.count(df.iloc[i]['Misconception'])==0 and ('Misconception' in df.iloc[i]['Category']) :
#     misconceptions.append(df.iloc[i]['Misconception'])
# print(len(misconceptions))


# labels = ["True_Neither","False_Neither","True_Correct","False_Correct"]

# for i in misconceptions :
#   candidate1 = str("True_Misconception"+":"+i)
#   candidate2=str("False_Misconception"+":"+i)
#   labels.append(candidate1)
#   labels.append(candidate2)
# # print(label)
# print(len(labels))





def tokenize_input_for_cot(row,tokenizer) :
    
    q_text, mc_answer, explanation = row["QuestionText"], row["MC_Answer"], row["StudentExplanation"]
    prompt = f"""
<|im_start|>system
Response in maximum 512 words.
You are a meticulous educational analyst and expert in mathematics pedagogy. Your task is to perform a verification check. You will be given a student's response to a math problem, and analyze it with respect to the question.
Show your detailed reasoning by following these steps:
YOUR STEP-BY-STEP VERIFICATION PROCESS (Chain-of-Thought):

1. Analyze Answer Correctness (True/False Check): First, independently solve the math problem in the Question. Compare your result to the student's Answer. Is the student's answer objectively True (correct) or False (incorrect)?
2. Analyze Explanation Quality (Reasoning Check): Now, ignore the final answer and focus only on the explanation.
Deconstruct the student's logic. What steps did they follow? Based only on their text, classify their reasoning: Is it Correct, a clear Misconception, or Neither?
If you identify a misconception, briefly describe it in your own words.

Show your detailed reasoning by following these above 2 steps.

<|im_end|>
<|im_start|>user
Problem Data:
Question: {q_text}
Student's Answer: {mc_answer}
Student's Explanation: {explanation}

<|im_end|>
<|im_start|>assistant
"""
    message = [
       {"role" : "user" , "content" :prompt}
   ]
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    tokenized_output = tokenizer(
        text,
        truncation = True,
        max_length = 2048,
        return_tensors = "pt"
    )
    
    return tokenized_output.to(device)


def tokenize_input(row, category, misconception, tokenizer,thought):
    
    
    q_text, mc_answer, explanation = row["QuestionText"], row["MC_Answer"], row["StudentExplanation"]
    parts = category.split("_")
    correctness = parts[0]
    reasoning_type = parts[1]
    prompt = f"""
<|im_start|>system
You are a meticulous educational analyst and expert in mathematics pedagogy. Your task is to perform a verification check. You will be given a student's response to a math problem , then a THOUGHT ANALYSIS and a proposed classification for that response. You must determine if the proposed classification is entirely accurate based on the your knowledge and problem data. Note that the THOUGHT ANALYSIS may be sometimes not correct.

DEFINITIONS OF THE CLASSIFICATION LABELS:

Part 1: Correctness (True or False): This describes whether the student's answer is objectively the correct solution to the Question Text.

Part 2: ReasoningType (Correct, Misconception, or Neither): This describes the quality of the student's explanation:
Correct: The explanation shows sound, logical, and mathematically valid reasoning.
Misconception: The explanation reveals a specific, identifiable error in conceptual understanding.
Neither: The explanation is incorrect, but does not point to a specific misconception. It could be a guess, irrelevant, or simply nonsensical.

Part 3: Misconception : This is a text description of the specific thinking error. It is only relevant when the ReasoningType is Misconception. If the ReasoningType is Correct or Neither, this field's value should be "NA".

YOUR TASK:

1. Compare the THOUGHT ANALYSIS to the Correctness,ReasoningType and Misconception in PROPOSED CLASSIFICATION based on PROBLEM DATA, then consider the following 3 questions :
1.1 From the THOUGHT ANALYSIS and your analysis, does the true "True/False" conclusion of student's answer match the Correctness value in PROPOSED CLASSIFICATION?
1.2 Does the true "Correct/Misconception/Neither" conclusion match the ReasoningType value in PROPOSED CLASSIFICATION?
1.3 If the ReasoningType value in PROPOSED CLASSIFICATION  is "Misconception", does the student's error align with the provided Misconception" text? (If the "ReasoningType" value in PROPOSED CLASSIFICATION is Correct or Neither, you can skip this step) 
2. Final Conclusion: A "Yes" is only possible if all checks in Step 1 pass. If there is any mismatch at any point, the answer must be "No".

**CONSTRAINT:
You are only allowed to output only one token ("Yes"/"No").


<|im_end|>
<|im_start|>user
PROBLEM DATA:
Question: {q_text}
Student's Answer: {mc_answer}
Student's Explanation: {explanation}

PROPOSED CLASSIFICATION:
Correctness: '{correctness}'
ReasoningType: '{reasoning_type}'
Misconception: '{misconception}'

THOUGHT ANALYSIS:
{thought}

<|im_end|>
<|im_start|>assistant
"""
    message = [
       {"role" : "user" , "content" :prompt}
   ]
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    tokenized_output = tokenizer(
        text,
        truncation = True,
        max_length = 1024,
        return_tensors = "pt"
    )
    
    return tokenized_output.to(device)


from datasets import load_from_disk
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



MODEL_PATH= "/kaggle/input/qwen-3/transformers/8b/1"
# TRAIN_CSV_PATH = "/train_with_misconceptions.csv" 

MODEL_COT ="/kaggle/input/qwen-3/transformers/1.7b/1"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 
)




tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
)

tokenizer_cot = AutoTokenizer.from_pretrained(MODEL_COT)
if tokenizer_cot.pad_token is None:
    tokenizer_cot.pad_token = tokenizer_cot.eos_token
# model_cot = AutoModelForCasualLM.from_pretrained(
#     MODEL_COT, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
# )

# yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
# no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]


model_cot = AutoModelForCausalLM.from_pretrained(
    MODEL_COT, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
)

yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]


print(yes_token_id)
print(no_token_id)


yes_g_token_id = tokenizer.encode(" Yes",add_special_tokens=False)[0]
no_g_token_id = tokenizer.encode(" No", add_special_tokens=False)[0]


print(yes_g_token_id)
print(no_g_token_id)


decode_steps=0


def generate_cot(row,tokenizer_cot) :
    inputs = None
    outputs = None
    with torch.no_grad():
        inputs = tokenize_input_for_cot(row,tokenizer_cot)
        generated_outputs = model_cot.generate(
            **inputs,
            max_new_tokens=2048,
            pad_token_id=tokenizer.eos_token_id
        )
    response=""

    #them vao skip_special_tokens=True
    for i in range(inputs["input_ids"].shape[1], generated_outputs.shape[1]):
        token_id = generated_outputs[0, i]
        decoded_token = str(tokenizer.decode(token_id))
        response += decoded_token
    return response


def infer(row,category,misconception,tokenizer,thought,decode_steps):
    flag=False
    inputs = None
    outputs = None
    scores = -300

    print(f"---{decode_steps} turn.---")
    inputs = tokenize_input(row,category,misconception,tokenizer,thought)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits      
    last_token_logits = logits[:, -1, :]  
    yes_logits = torch.max(
            last_token_logits[:, yes_token_id],
            last_token_logits[:, yes_g_token_id]
        )
    no_logits = torch.max(
        last_token_logits[:, no_token_id],
        last_token_logits[:, no_g_token_id]
    )
    scores = yes_logits - no_logits
    print(f"---Scores calculated ---")
    if decode_steps < 3:
        print(f"--- Scores = {scores} ---")
    # print("---Đang sinh response ---")
    # with torch.no_grad():
    #     inputs = tokenize_input(row,category,misconception,tokenizer,thought)
    #     generated_outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=2048,
    #         pad_token_id=tokenizer.eos_token_id
    #     )

    # full_decoded_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
    # if decode_steps < 3:
    #     print(f"Chuỗi đầy đủ được sinh ra: '{full_decoded_text}'\n")

    # print("---Tính toán và trích xuất logits ---")
    # with torch.no_grad():
    #     full_sequence_outputs = model(input_ids=generated_outputs)
    #     all_logits = full_sequence_outputs.logits

    # response_logits = []
    
    # for i in range(inputs["input_ids"].shape[1], generated_outputs.shape[1]):
    #     if flag == True :
    #         break
    #     token_id = generated_outputs[0, i]
    #     logits_at_step = all_logits[0, i - 1, :]
    #     chosen_token_logit = logits_at_step[token_id].item()
    #     decoded_token = tokenizer.decode(token_id)
    #     if abs(i - generated_outputs.shape[1])  <= 3 and ("Yes" in str(decoded_token) or "No" in str(decoded_token)):
    #         flag = True
    #         yes_scores = max(round(logits_at_step[yes_g_token_id].item(),4),round(logits_at_step[yes_token_id].item(),4))
    #         no_scores = max(round(logits_at_step[no_g_token_id].item(),4),round(logits_at_step[no_token_id].item(),4))
    #         scores = yes_scores - no_scores
    #         if decode_steps < 3 :
    #             print(f"--- Token ID of Yes/No in this turn : {token_id} ---------")
    #             print(f"--- yes_token_id with space = {yes_g_token_id} ; no_token_id with space = {no_g_token_id}")
    #             print(f"--- Yes score: {yes_scores} ; No score: {no_scores} ---")
    #             print(f"Score = Yes score - No score = {scores}.")
    #     response_logits.append({
    #         "token": decoded_token,
    #         "logit": round(chosen_token_logit, 4)
    #     })

    # if decode_steps < 3 :
    #     print("Decode tokens :")
    #     for idx,item in enumerate(response_logits):
    #         print(f"Position: {idx} , Token: '{item['token']}', Logit: {item['logit']}")
    
    decode_steps+=1
    
    # if decode_steps < 3:
    #     with torch.no_grad():
    #         gen = model.generate(
    #             **inputs,
    #             max_new_tokens=2048,
    #             temperature=0.7,
    #             top_p=0.9,
    #             repetition_penalty=1.05,
    #             do_sample=True,
    #             return_dict_in_generate=False
    #         )
    #         decoded_inp = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    #         response_tokens = gen[0][input_ids_length:]
    #         decoded_gen = tokenizer.decode(response_tokens, skip_special_tokens=False)
    #         print("================= DEBUG=================")
    #         print(f"--- INPUT ---:\n{decoded_inp}")
    #         print(f"--- Response ---:\n{decoded_gen}")
    #         print("============================================")
    #     decode_steps+=1

    
    result = str(category+":"+misconception)
    return scores , result , decode_steps
    





remove_columns = ["QuestionId","QuestionText","MC_Answer","StudentExplanation"]


def predict(dataset,decode_steps):
    count = 0
    for row in dataset :
        print(f"--This is the {count} test---")
        candidate_labels = retrieve(row,tokenizer_retrieve,model_retrieve)
        ranking_candidate = {}
        for label in candidate_labels :
            category,misconception = None,None
            parts = label.split(":")
            category = str(parts[0])
            misconception = str(parts[1])
                
            thought = generate_cot(row,tokenizer_cot)
            score , result , decode_steps = infer(row,category,misconception,tokenizer,thought,decode_steps)
            ranking_candidate[result] = score
                
        sorted_candidate_desc = sorted(ranking_candidate.items(), key=lambda x: x[1], reverse=True)
        top_3_candidates = sorted_candidate_desc[:3]
        top_3_keys = [k for k, v in top_3_candidates]
        row['Category:Misconception'] = str(top_3_keys[0]+" "+top_3_keys[1]+" "+top_3_keys[2])
        for column_name in remove_columns :
            row.pop(column_name,None)
        if count < 3 :
            print(row)
        count+=1
                


print(tokenizer.convert_ids_to_tokens(2308))


test_dataset= predict(test_dataset,decode_steps=decode_steps)


submit_df = pd.DataFrame(test_dataset)
submit_df.to_csv('submission.csv', index=False)


submit_df.head()







