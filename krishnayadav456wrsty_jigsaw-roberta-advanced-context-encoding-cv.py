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


!pip install -q transformers datasets


import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch



train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")



def build_input_text(row):
    return (
        f"Rule: {row['rule']} "
        f"Positive Examples: {row['positive_example_1']} {row['positive_example_2']} "
        f"Negative Examples: {row['negative_example_1']} {row['negative_example_2']} "
        f"Comment: {row['body']}"
    )

train['input_text'] = train.apply(build_input_text, axis=1)
test['input_text'] = test.apply(build_input_text, axis=1)



model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def encode(examples):
    return tokenizer(examples['input_text'], truncation=True, padding='max_length', max_length=256)



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))



os.environ["WANDB_DISABLED"] = "true"
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./models",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    report_to="none"  # Disable W&B
)



for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['rule_violation'])):
    print(f"***** Fold {fold+1} *****")
    
    train_fold = train.iloc[train_idx]
    val_fold = train.iloc[val_idx]
    
    # Convert to HF Dataset
    hf_train = Dataset.from_pandas(train_fold[['input_text', 'rule_violation']])
    hf_val = Dataset.from_pandas(val_fold[['input_text', 'rule_violation']])
    hf_test = Dataset.from_pandas(test[['input_text']])
    
    # Tokenize
    hf_train = hf_train.map(encode, batched=True)
    hf_val = hf_val.map(encode, batched=True)
    hf_test = hf_test.map(encode, batched=True)
    
    hf_train = hf_train.rename_column('rule_violation', 'labels')
    hf_val = hf_val.rename_column('rule_violation', 'labels')
    
    hf_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    hf_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    hf_test.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./fold_{fold+1}",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=1
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        auc = roc_auc_score(labels, probs)
        return {"roc_auc": auc}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # Validation Predictions
    val_preds = trainer.predict(hf_val).predictions
    val_probs = torch.softmax(torch.tensor(val_preds), dim=-1)[:, 1].numpy()
    oof_preds[val_idx] = val_probs
    
    # Test Predictions
    test_preds_fold = trainer.predict(hf_test).predictions
    test_probs_fold = torch.softmax(torch.tensor(test_preds_fold), dim=-1)[:, 1].numpy()
    test_preds += test_probs_fold / 5



print("OOF ROC-AUC:", roc_auc_score(train['rule_violation'], oof_preds))



import matplotlib.pyplot as plt

def plot_training_history(trainer):
    logs = trainer.state.log_history
    steps = [log["step"] for log in logs if "loss" in log]
    losses = [log["loss"] for log in logs if "loss" in log]
    
    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, label='Training Loss')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()

plot_training_history(trainer)



plt.figure(figsize=(8, 4))
plt.hist(test_preds, bins=50, color='skyblue')
plt.title("Distribution of Test Predictions")
plt.xlabel("Predicted Probability of Rule Violation")
plt.ylabel("Frequency")
plt.show()



from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(train['rule_violation'], oof_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Validation ROC Curve")
plt.legend()
plt.show()



submission = pd.DataFrame({
    'row_id': test['row_id'],
    'rule_violation': test_preds
})
submission.to_csv('submission.csv', index=False)
print(submission.head())


