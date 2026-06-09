import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,TrainingArguments,Trainer
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset


train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
sample_sub = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")
print("Train column are : ",train.columns)
print("Test columns are : ",test.columns)
print("Sample Submission Columns are : ", sample_sub.head())
print("Train:", train.shape, "Test:", test.shape)



train["Misconception"] = train["Misconception"].fillna("NA")
train["target"] = train["Category"] + ":" + train["Misconception"]


le                  = LabelEncoder()
train.Misconception     = train.Misconception.fillna('NA')
train['target']   = train.Category + ':' +train.Misconception
train['label']    = le.fit_transform(train['target'])

n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")



idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)


test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)


def format_input(row):
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This is answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

test['text'] = test.apply(format_input,axis=1)


ds_test = Dataset.from_pandas(test)


model = AutoModelForSequenceClassification.from_pretrained("/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/2", device_map="cuda:0", torch_dtype=torch.bfloat16)


model


tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/2")
model.config.pad_token_id = tokenizer.pad_token_id

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

ds_test = ds_test.map(tokenize, batched=True)


test_args = TrainingArguments(
    output_dir="./",
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=16, 
    bf16=False,          
    fp16=True,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=test_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

predictions = trainer.predict(ds_test)

predictions.predictions


top3           = np.argsort(-predictions.predictions, axis=1)[:, :3]
flat_top3      = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels_cat    = decoded_labels.reshape(top3.shape)
top3_labels_cat


joined_preds = []

for preds in top3_labels_cat:
    joined_preds.append(" ".join(preds))



# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
sub.head()

