import torch
import gc
import kagglehub

from transformers import AutoModelForSequenceClassification, pipeline, AutoTokenizer
from sklearn.preprocessing import LabelEncoder

from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel
from pathlib import Path
from pprint import pprint

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))




train_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")


idx = train_df.apply(lambda row: row.Category.split('_')[0],axis=1) == 'True'
correct = train_df.loc[idx].copy()

correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

test_df = test_df.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test_df.is_correct = test_df.is_correct.fillna(0)



def cat_formatting_prompts_func(row):
    x = "This answer is Correct"
    if not row['is_correct']:
        x = "This answer is NOT Correct"
    question_id = str(row.QuestionId)
    return (
        f"# Question: \n{row['QuestionText']}\n\n"
        f"# Answer: \n{row['MC_Answer']}\n\n"
        f"{x}\n\n"
        f"# Student Explanation: \n{row['StudentExplanation']}\n"
    )

def misconception_formatting_prompts_func(row):
    x = "This answer is Correct"
    if not row['is_correct']:
        x = "This answer is NOT Correct"
    question_id = str(row.QuestionId)
    return (
        f"# Question: \n{row['QuestionText']}\n\n"
        f"# Answer: \n{row['MC_Answer']}\n\n"
        f"{x}\n\n"
        f"# Student Explanation: \n{row['StudentExplanation']}\n"
        # f"Expected concepts in Student Explanation: {llm_answer_guide[question_id]['answer_guide']}"
    )

# apply formatting_prompts_func to train_df
test_df["category_text"] = test_df.apply(cat_formatting_prompts_func, axis=1)
test_df["misconception_text"] = test_df.apply(misconception_formatting_prompts_func, axis=1)



# # 1. PREPARE LABELS
# le = LabelEncoder()
# le.fit(train_df['Category'])
# label_names = le.classes_
# NUM_CLASSES = len(label_names)

# id2label = {}
# label2id = {}
# for i, label in enumerate(label_names):
#     id2label[i] = label
#     label2id[label] = i
    
# # 2. LOAD THE BASE MODEL AND ADAPTERS
# PEFT_MODEL_PATH = kagglehub.model_download("pk00095/math-mis-category-classifier/transformers/qwen3_4b-peft-epoch1-balanced")
# MODEL_NAME = kagglehub.model_download("qwen-lm/qwen-3/transformers/4b-base")

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# model = AutoModelForSequenceClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=NUM_CLASSES,
#     id2label=id2label, 
#     label2id=label2id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

# model.config.pad_token_id = model.config.eos_token_id

# print("Path to model files:", PEFT_MODEL_PATH)
# peft_model = PeftModel.from_pretrained(model, PEFT_MODEL_PATH)

# peft_model.eval()

# peft_model.config.id2label = id2label
# peft_model.config.label2id = label2id


# # 3. MERGE AND SAVE THE MODEL
# merged = peft_model.merge_and_unload()  # returns a plain transformers model
# merged.config.id2label = id2label  # re-assert (usually carried over, but safe)
# merged.config.label2id = label2id

# MERGED_EXPORT_DIR = Path("./cat_classification")

# MERGED_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
# merged.save_pretrained(MERGED_EXPORT_DIR)
# tokenizer.save_pretrained(MERGED_EXPORT_DIR)


# # 4. CLEANUP AND RELEASE MEMORY
# model.to('cpu')
# peft_model.to('cpu')
# merged.to('cpu')
# del model
# del peft_model
# del merged
# gc.collect()
# torch.cuda.empty_cache()

# # 5. UPLOAD MERGED MODEL
# MODEL_SLUG = 'math-MIS-category-classifier'
# VARIATION_SLUG = 'qwen3_4b-PEFT-epoch1-balanced-mergedModel'

# kagglehub.model_upload(
#   handle = f"pk00095/{MODEL_SLUG}/transformers/{VARIATION_SLUG}",
#   local_model_dir = MERGED_EXPORT_DIR,
#   version_notes = 'base + peft merged model, ready for inference')


# # 1. PREPARE LABELS
# le = LabelEncoder()
# le.fit(train_df['Misconception'].dropna())
# label_names = le.classes_
# NUM_CLASSES = len(label_names)

# id2label = {}
# label2id = {}
# for i, label in enumerate(label_names):
#     id2label[i] = label
#     label2id[label] = i

# # 2. LOAD THE BASE MODEL AND ADAPTERS
# PEFT_MODEL_PATH = kagglehub.model_download("pk00095/math-mis-misunderstanding-classifier/transformers/qwen3_4b-peft-balancedds-weightedce-v1")
# MODEL_NAME = kagglehub.model_download("qwen-lm/qwen-3/transformers/4b-base")

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# model = AutoModelForSequenceClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=NUM_CLASSES,
#     id2label=id2label, 
#     label2id=label2id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

# model.config.pad_token_id = model.config.eos_token_id

# print("Path to model files:", PEFT_MODEL_PATH)
# peft_model = PeftModel.from_pretrained(model, PEFT_MODEL_PATH)

# peft_model.eval()

# peft_model.config.id2label = id2label
# peft_model.config.label2id = label2id

# # 3. MERGE AND SAVE THE MODEL
# merged = peft_model.merge_and_unload()  # returns a plain transformers model
# merged.config.id2label = id2label  # re-assert (usually carried over, but safe)
# merged.config.label2id = label2id

# MERGED_EXPORT_DIR = Path("./misconception_classification")

# MERGED_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
# merged.save_pretrained(MERGED_EXPORT_DIR)
# tokenizer.save_pretrained(MERGED_EXPORT_DIR)

# # 4. CLEANUP AND RELEASE MEMORY
# model.to('cpu')
# peft_model.to('cpu')
# merged.to('cpu')
# del model
# del peft_model
# del merged
# gc.collect()
# torch.cuda.empty_cache()

# # 5. UPLOAD MERGED MODEL
# MODEL_SLUG = 'math-MIS-misunderstanding-classifier'
# VARIATION_SLUG = 'qwen3_4b-PEFT-balancedDS-WeightedCE-v1-mergedModel' # Replace with variation slug.

# kagglehub.model_upload(
#   handle = f"pk00095/{MODEL_SLUG}/transformers/{VARIATION_SLUG}",
#   local_model_dir = MERGED_EXPORT_DIR,
#   version_notes = 'base + peft merged model, ready for inference')



category_merged_model_path = kagglehub.model_download("pk00095/math-mis-category-classifier/transformers/qwen3_1.7b-peft-weightedce-v2")
tokenizer = AutoTokenizer.from_pretrained(category_merged_model_path)
# --- 6) Reload the merged model like any other HF model -----------------------
category_model = AutoModelForSequenceClassification.from_pretrained(
    category_merged_model_path,
    torch_dtype=torch.float16,
)
# assert category_model.config.id2label == id2label, "id2label not persisted!"
# assert category_model.config.label2id == label2id, "label2id not persisted!"
category_model.eval()

category_pipeline = pipeline(
    task="text-classification",
    model=category_model,
    tokenizer=tokenizer,
    device='cuda:0',
    top_k=1,
)




predictions = {}
misconceptions = []

for idx, row in test_df.iterrows():
    # Prepare your input data
    row_id = row["row_id"]
    text_to_classify = row["category_text"]
    
    label = category_pipeline(text_to_classify)[0]
    # print(label)

    predictions[row_id] = dict(category=label[0]['label'], misconception_text=row["misconception_text"])




category_model = category_model.to('cpu')
del category_model
gc.collect()
torch.cuda.empty_cache()


misconception_merged_model_path = kagglehub.model_download("pk00095/math-mis-misunderstanding-classifier/transformers/qwen2.5-math-1.5b-peft-weightedce-v7-mergedmodel")
tokenizer = AutoTokenizer.from_pretrained(misconception_merged_model_path)
# --- 6) Reload the merged model like any other HF model -----------------------
misconception_model = AutoModelForSequenceClassification.from_pretrained(
    misconception_merged_model_path,
    torch_dtype=torch.float16,
)
# assert misconception_model.config.id2label == id2label, "id2label not persisted!"
# assert misconception_model.config.label2id == label2id, "label2id not persisted!"
misconception_model.eval()

misconception_pipeline = pipeline(
    task="text-classification",
    model=misconception_model,
    tokenizer=tokenizer,
    device='cuda:0',
    top_k=3,
)


pred_records = []
for row_id, row_val in predictions.items():
    misconception = ['NA', 'NA', 'NA']
    category = row_val['category']
    if 'misconception' in category.lower():
        res = misconception_pipeline(row_val["misconception_text"])[0]
        misconception = [r['label'] for r in res]
        
    res = [f"{category}:{misc}" for misc in misconception]
        
    pred_records.append({"row_id": row_id, "Category:Misconception": " ".join(res)})
    
        


# Save submission
sub = pd.DataFrame.from_records(pred_records)
sub.to_csv("submission.csv", index=False)
sub.head()




