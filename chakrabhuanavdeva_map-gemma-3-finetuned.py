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


# --- 0. Import Library ---
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')

# --- 1. Custom Model Class ---
class Gemma3ForSequenceClassification(nn.Module):
    def __init__(self, model_path, num_labels):
        super(Gemma3ForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        
        full_model = Gemma3ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        self.gemma3 = full_model.model
        self.config = self.gemma3.config
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.gemma3(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        sequence_lengths = torch.sum(attention_mask, dim=1) - 1
        batch_size = input_ids.shape[0]
        pooled_output = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
            
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# --- 2. Data Preparation (for label mapping consistency) ---
print("Loading training data for label mapping...")
df_train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
df_train['Misconception'] = df_train['Misconception'].fillna('NA')
df_train['target'] = df_train['Category'] + ':' + df_train['Misconception']

# Filter and create same label mapping as before
label_counts = df_train['target'].value_counts()
valid_targets = label_counts[label_counts >= 2].index
df_filtered = df_train[df_train['target'].isin(valid_targets)].copy()

labels_map = {label: i for i, label in enumerate(df_filtered['target'].unique())}
id2label = {v: k for k, v in labels_map.items()}
num_labels = len(labels_map)

print(f"Number of classes: {num_labels}")

# --- 3. Load Your Existing Trained Model ---
TRAINED_MODEL_PATH = "/kaggle/input/gemma3map-fine-tuning/transformers/1b/2/best_model"
BASE_MODEL_ID = "/kaggle/input/gemma-3/transformers/gemma-3-1b-it/1"

print("Loading your existing trained model...")
base_model = Gemma3ForSequenceClassification(BASE_MODEL_ID, num_labels)
model = PeftModel.from_pretrained(base_model, TRAINED_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH)

print("Model loaded successfully!")

# --- 4. Load Test Data and Make Predictions ---
print("Loading test data...")
df_test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
df_test['text'] = df_test['QuestionText'] + ' [SEP] ' + df_test['MC_Answer'] + ' [SEP] ' + df_test['StudentExplanation']

def tokenize_function(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

test_dataset = Dataset.from_pandas(df_test)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# --- 5. Create Trainer for Inference Only ---
training_args = TrainingArguments(
    output_dir="./results_inference",
    per_device_eval_batch_size=8,
    bf16=True,
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator
)

print("Making predictions with your trained model...")
predictions = trainer.predict(tokenized_test_dataset)

# --- 6. Create Submission ---
logits = predictions.predictions
probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()

# Get top 3 predictions
top_k = 3
top_k_indices = np.argsort(probs, axis=1)[:, -top_k:][:, ::-1]

submission_strings = []
for indices in top_k_indices:
    labels = [id2label[i] for i in indices]
    submission_strings.append(" ".join(labels))

df_test['Category:Misconception'] = submission_strings
submission_df = df_test[['row_id', 'Category:Misconception']]
submission_df.to_csv('submission.csv', index=False)

print("\nPredictions completed using your existing trained model!")
print("File submission.csv created.")
print(submission_df.head())
print(f"\nThis should perform better than 40% since it uses your 9-hour trained model!")




