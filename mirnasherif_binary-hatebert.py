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


!unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip -d /kaggle/working
!pip install datasets transformers evaluate accelerate -q
!pip install nlpaug


df= pd.read_csv(r"/kaggle/working/train.csv")
df.head()



import numpy as np
df['binary'] = np.where(df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) > 0, 1, 0)
df.drop([ 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1, inplace=True)
df.rename(columns={'binary': 'toxic','comment_text':'content'}, inplace=True)
df.head(10)


# Define gaming slang phrases (toxic examples)
gaming_slang_toxic = [
    "gg ez noob uninstall",
    "kys trash player",
    "bot lane diff",
    "report this feeder",
    "you're hardstuck bronze",
    "0 IQ gameplay"
]



if isinstance(df, pd.DataFrame):
    # For Kaggle DataFrame
    new_rows = pd.DataFrame({
        "content": gaming_slang_toxic,
        "toxic": [1] * len(gaming_slang_toxic)  # Assuming binary toxicity
    })
    df = pd.concat([df, new_rows], ignore_index=True)

    


#Clean data
def clean(data, col):

    # Clean some punctutations
    data[col] = data[col].str.replace('\n', ' \n ')
    # Remove ip address
    data[col] = data[col].str.replace(r'(([0-9]+\.){2,}[0-9]+)',' ')
    
    data[col] = data[col].str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3')
    # Replace repeating characters more than 3 times to length of 3
    data[col] = data[col].str.replace(r'([*!?\'])\1\1{2,}',r'\1\1\1')
    # patterns with repeating characters 
    data[col] = data[col].str.replace(r'([a-zA-Z])\1{2,}\b',r'\1\1')
    data[col] = data[col].str.replace(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1')
    data[col] = data[col].str.replace(r'[ ]{2,}',' ').str.strip()   
    # Add space around repeating characters
    data[col] = data[col].str.replace(r'([*!?\']+)',r' \1 ')    
    
    return data

df = clean(df,'content')
df['content'] = df['content'].astype(pd.StringDtype())
df.info()
  


# prompt: split the train test val with a stratify to toxic

from sklearn.model_selection import train_test_split

# Assuming 'df' is your DataFrame with 'content' and 'toxic' columns
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['toxic'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['toxic'], random_state=42)

print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")


def count_labels(df, label_column='toxic'):
    """Counts the occurrences of 0 and 1 in a specified column of a DataFrame.

    Args:
        df: The input DataFrame.
        label_column: The name of the column containing the labels (default: 'toxic').

    Returns:
        A dictionary with counts of 0 and 1.
    """
    counts = df[label_column].value_counts().to_dict()
    return {
        0: counts.get(0, 0),  # Get count of 0, default to 0 if not found
        1: counts.get(1, 0)   # Get count of 1, default to 0 if not found
    }

# Get counts for each split
train_counts = count_labels(train_df)
val_counts = count_labels(val_df)
test_counts = count_labels(test_df)

# Print the results
print("Train set:", train_counts)
print("Validation set:", val_counts)
print("Test set:", test_counts)



!pip install nlpaug


# Data augmentation for minority class
from nlpaug.augmenter.word import SynonymAug
import nltk

# Create a custom NLTK data directory
!mkdir -p /kaggle/working/nltk_data/corpora

# Download and extract wordnet
!wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip
!unzip wordnet.zip -d /kaggle/working/nltk_data/corpora/

# Set the NLTK data path
import nltk
nltk.data.path.append('/kaggle/working/nltk_data')

!chmod -R 755 /kaggle/working/nltk_data


aug = SynonymAug(aug_src='wordnet')

augmented_samples = []
for text in train_df[train_df['toxic'] == 1]['content']:
    if isinstance(text, str) and text == text:  # Check for NaN
        augmented_text = aug.augment(text)
        augmented_text = augmented_text[0] if isinstance(augmented_text, list) else augmented_text
        augmented_samples.append(augmented_text)



df_augmented = pd.DataFrame({"content": augmented_samples, "toxic": [1] * len(augmented_samples)})
train_df = pd.concat([train_df, df_augmented], ignore_index=True)


from sklearn.utils import resample
df_majority = train_df[train_df['toxic'] == 0]

df_minority = train_df[train_df['toxic'] == 1]

# Oversample minority class
df_minority_oversampled = resample(
    df_minority,
    replace=True,  # Sample with replacement
    n_samples=len(df_majority),  # Match majority class size
    random_state=42
)

# Combine majority and oversampled minority
df_balanced = pd.concat([df_majority, df_minority_oversampled])

# Shuffle the dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Check class distribution
print(df_balanced['toxic'].value_counts())
train_df = df_balanced.copy()



# Tokenize data
from datasets import Dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")


def preprocess_function(examples):
    # Tokenize the text and truncate/pad sequences
    #attention_mask: Mask indicating which tokens are actual words and which are padding.
    tokenized_examples = tokenizer(examples["content"], padding="max_length", truncation=True)
    # Include the labels in the output dictionary
    tokenized_examples["labels"] = examples["toxic"] 
    return tokenized_examples

train_dataset = Dataset.from_pandas(train_df).map(preprocess_function, batched=True, remove_columns=['content'])
val_dataset = Dataset.from_pandas(val_df).map(preprocess_function, batched=True, remove_columns=['content'])
test_dataset = Dataset.from_pandas(test_df).map(preprocess_function, batched=True, remove_columns=['content'])


# Compute class weights
from sklearn.utils.class_weight import compute_class_weight
import torch
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df['toxic']),
    y=train_df['toxic']
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)


# Get counts for each split
train_counts = count_labels(train_df)
val_counts = count_labels(val_df)
test_counts = count_labels(test_df)

# Print the results
print("Train set:", train_counts)
print("Validation set:", val_counts)
print("Test set:", test_counts)


# Train and evaluate models
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer
import evaluate
import torch

# Setup
model_name = "GroNLP/hateBERT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Training arguments
args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    report_to="none",
)

# Metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda eval_pred: {
        "accuracy": accuracy_metric.compute(predictions=np.argmax(eval_pred.predictions, axis=-1), references=eval_pred.label_ids)["accuracy"],
        "f1": f1_metric.compute(predictions=np.argmax(eval_pred.predictions, axis=-1), references=eval_pred.label_ids, average="weighted")["f1"]
    },
)


# Train and save model
trainer.train()
trainer.save_model(f"./{model_name}-finetuned")

# Evaluate
results = trainer.evaluate()
print(f"\n=== Model Performance ===\n{model_name}:\n") 
print(f"Accuracy: {results['eval_accuracy']:.4f}\n")
print(f"F1-Score: {results['eval_f1']:.4f}\n")


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import torch

# Generate predictions
# Evaluate on test dataset
test_results = trainer.evaluate(test_dataset)
print(f"\n=== Test Performance ===\n  Accuracy: {test_results['eval_accuracy']:.4f}\n  F1-Score: {test_results['eval_f1']:.4f}")

# Generate predictions for test dataset
test_predictions = trainer.predict(test_dataset)
preds = np.argmax(test_predictions.predictions, axis=-1)
labels = test_predictions.label_ids  # True labels from test_dataset



# Confusion matrix
cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# ROC curve and AUC
probs = torch.softmax(torch.tensor(test_predictions.predictions), dim=-1).numpy()[:, 1]  # Probabilities for class 1
fpr, tpr, _ = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()



# Save the tokenizer and model
tokenizer.save_pretrained("hateBERT-finetuned")  # Save tokenizer
trainer.save_model("hateBERT-finetuned")        # Save model

# Compress the folder into a zip file
!zip -r hateBERT-finetuned.zip hateBERT-finetuned/

