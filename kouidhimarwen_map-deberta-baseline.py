import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer, DebertaTokenizer, DebertaForSequenceClassification, TrainingArguments, Trainer, DebertaV2ForSequenceClassification

from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
import torch
import joblib


# Download competition dataset
DATA_DIR = "/kaggle/input/map-charting-student-math-misunderstandings"

TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

# Load datasets into DataFrames
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

# Display basic information to verify successful loading
print("\nData Loading Summary:")
print(f"  - Training set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"  - Test set:     {test_df.shape[0]} rows, {test_df.shape[1]} columns")
print(f"  - Submission:   {sample_submission_df.shape[0]} rows, {sample_submission_df.shape[1]} columns")


le = LabelEncoder()
train_df.Misconception = train_df.Misconception.fillna('NA')
train_df['target'] = train_df.Category + ":" + train_df.Misconception
train_df['label'] = le.fit_transform(train_df['target'])
n_classes = len(le.classes_)
train_df.head()



idx = train_df.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train_df.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train_df = train_df.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train_df.is_correct = train_df.is_correct.fillna(0)

train_df


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

train_df['text'] = train_df.apply(format_input,axis=1)



print("Example prompt for our LLM:")
print()
print( train_df.text.values[0] )



train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)



COLS = ['text','label']

train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])



# model_name = "microsoft/deberta-v3-xsmall"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Tokenization function
# def tokenize(batch):
#     return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

# train_ds = train_ds.map(tokenize, batched=True)
# val_ds = val_ds.map(tokenize, batched=True)

# # Set format for PyTorch
# columns = ['input_ids', 'attention_mask', 'label']
# train_ds.set_format(type='torch', columns=columns)
# val_ds.set_format(type='torch', columns=columns)



train_ds



# model = DebertaV2ForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=n_classes
# )



from sklearn.metrics import average_precision_score

def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = (top3 == labels[:, None])

    # Compute MAP@3 manually
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}



# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     tokenizer=tokenizer,
#     compute_metrics=compute_map3,
# )

# trainer.train()



tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/model01/transformers/default/1/model")
model = DebertaV2ForSequenceClassification.from_pretrained(
    "/kaggle/input/model01/transformers/default/1/model",
    num_labels=n_classes
)
training_args = TrainingArguments(
        report_to="none",
)


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(model=model, data_collator=data_collator, args=training_args)



test = test_df.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

test['text'] = test.apply(format_input,axis=1)

test.head()



from datasets import Dataset
import torch

# Convert test DataFrame to Dataset
ds_test = Dataset.from_pandas(test[['text']])

# Set a reasonable max_length
max_length = 256  # or 1024 if your model allows

# Tokenize
ds_test = ds_test.map(
    lambda batch: tokenizer(batch["text"], padding=True, truncation=True, max_length=max_length),
    batched=True
)

# Predict
predictions = trainer.predict(ds_test)

# Convert logits to probabilities
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()



# Get top 3 predicted class indices
top3 = np.argsort(-probs, axis=1)[:, :3]   # shape: [num_samples, 3]

# Decode numeric class indices to original string labels
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

# Join 3 labels per row with space
joined_preds = [" ".join(row) for row in top3_labels]

# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})

sub.to_csv("/kaggle/working/submission.csv", index=False)
sub.head()








