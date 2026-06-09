!pip install transformers datasets evaluate accelerate -q





from datasets import Dataset
from transformers import AutoTokenizer
import evaluate
import pandas as pd


df_train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
df_train.head()


##Checking whether class is imbalace or no. ## 
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Count values
value_counts = df_train['rule_violation'].value_counts()

# Plot
plt.figure(figsize=(6,4))
sns.barplot(x=value_counts.index, y=value_counts.values, palette="Set2")

# Labels and title
plt.xlabel("Rule Violation")
plt.ylabel("Count")
plt.title("Distribution of Rule Violation Labels")
plt.xticks([0, 1], ['Not Violation (0)', 'Violation (1)'])
plt.show()


## Test data ###
df_test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
df_test.head()


df_train['body'] = df_train['body'] + '[SEP]' + df_train['subreddit'] 	
df_test['body'] = df_test['body'] + '[SEP]' + df_test['subreddit'] 	


df_test.info()


### Extracting useful cols ###
subset = df_train[['body' , 'rule_violation']]

subset2 = pd.DataFrame({
    "body": df_train["positive_example_1"],
    "rule_violation": 1})

subset3 = pd.DataFrame({
    "body": df_train["positive_example_1"],
    "rule_violation": 1})


subset3 = pd.DataFrame({
    "body": df_train["negative_example_1"],
    "rule_violation": 0})

subset4 = pd.DataFrame({
    "body": df_train["negative_example_2"],
    "rule_violation": 0
})

## Concatenating for final dataset ##
final_subset = pd.concat([subset , subset2, subset3], ignore_index=True)


final_subset.head()


## Preprocess 
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Add pad token
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)  # Binary classification
model.resize_token_embeddings(len(tokenizer))


def preprocess(example):
    encoding = tokenizer(example['body'], truncation=True, padding="max_length", max_length=64)
    encoding["labels"] = int(example["rule_violation"])  # make sure it's int
    return encoding





dataset = Dataset.from_pandas(final_subset)
train_test_split = dataset.train_test_split(test_size=0.25)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

train_dataset = train_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)



## Model Initialization for classification ###

from transformers import AutoModelForSequenceClassification

model_name = 'gpt2'
model = AutoModelForSequenceClassification.from_pretrained(
    model_name , 
    num_labels = 2, 
    ignore_mismatched_sizes = True
)

model.config.pad_token_id = tokenizer.eos_token_id


## metrcis
import evaluate
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)




## Model training ##
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
from transformers import DataCollatorWithPadding

training_args = TrainingArguments(
    output_dir="rule_violation_classifier",
     eval_strategy="epoch",              
    save_strategy="epoch",                     
    logging_strategy="steps",                  
    logging_steps=50,                          
    num_train_epochs=3,                        
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    push_to_hub=False,
    report_to="none",                          
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)



data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,  # Add this
    compute_metrics=compute_metrics,
)


trainer.train()



# Tokenize test texts
test_encodings = tokenizer(
    df_test['body'].tolist(), 
    truncation=True, 
    padding="max_length", 
    max_length=64, 
    return_tensors="pt"
)



import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
import torch


model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=test_encodings['input_ids'].to(device),
        attention_mask=test_encodings['attention_mask'].to(device)
    )



# Apply softmax to get probabilities
probs = F.softmax(outputs.logits, dim=1).cpu().numpy()



df_test['prob_class_0'] = probs[:, 0]
df_test['prob_class_1'] = probs[:, 1]

# Optional: Sort by most likely rule violation
df_test_sorted = df_test.sort_values('prob_class_1', ascending=False)
df_test_sorted[['body', 'prob_class_1']].head()



df_test_sorted[['body', 'prob_class_1']]


submission = pd.DataFrame({'row_id' : df_test['row_id'] , 'rule_violation' : df_test['prob_class_1']})


submission.to_csv('submission.csv' ,  index = False)




