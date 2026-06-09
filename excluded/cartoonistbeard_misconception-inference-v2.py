import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import re
# import nltk
# from nltk.stem import WordNetLemmatizer

import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification, TrainingArguments, Trainer, DebertaV2ForSequenceClassification, AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import Dataset

import os
import joblib
from sklearn.metrics import average_precision_score

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)


class config:
    DIR = "/kaggle/input/gemma2-9b-it-cv945"
    LE = "/kaggle/input/labelencoder/label_encoder.joblib"
    classes = 65

config = config()


tokenizer = AutoTokenizer.from_pretrained(config.DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    "/kaggle/input/gemma2-9b-it-bf16",
    num_labels=config.classes,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
training_args = TrainingArguments(
    report_to="none",
    bf16=False, # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU          
    fp16=True, # INFER WITH FP16 BECAUSE KAGGLE IS T4 GPU
)
trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args)
le = joblib.load(config.LE)


test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
print( test.shape )
display(train.head())
test.head()


def advanced_clean(text):
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'FRAC_\1_\2', text)
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'FRAC_\1_\2', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s_]', '', text)
    return text.strip().lower()

def fraction_cleaner(text):
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'FRAC_\1_\2', text)
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'FRAC_\1_\2', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s_.]','', text)
    return text.strip().lower()

def fast_lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def extract_math_features(text):
    return {
        'frac_count': len(re.findall(r'FRAC_\d+_\d+|\\frac', text)),
        'number_count': len(re.findall(r'\b\d+\b', text)),
        'operator_count': len(re.findall(r'[\+\-\*/=]', text)),
        'starts_with_number': int(bool(re.match(r'^\d+', text))),
        'has_frac_token': int('FRAC_' in text)
    }

def more_math_feats(text):
    return {
        'contains_with_frac': int('fraction ' in text),
        'contains_eq': int('=' in text)
    }


# test["Q_cleaned"] = test['QuestionText'].apply(advanced_clean).apply(fast_lemmatize)
# test['Expl_cleaned'] = test['StudentExplanation'].apply(advanced_clean).apply(fast_lemmatize)
# display(test.head())


# test["c_Ans"] = test["MC_Answer"].apply(fraction_cleaner)
# display(test.head())


idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)


display(train.head())


test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)
display(test.head())


def format_input(row):
    x = "Yes"
    if not row['is_correct']:
        x = "No"
    # x = "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

test['text'] = test.apply(format_input,axis=1)
print("Example prompt for our LLM:")
print()
print( test.text.values[0] )


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

predictions = trainer.predict(ds_test)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()


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
# sub.to_csv("submission.csv", index=False)
def replace_nan_text(text):
    if pd.isna(text):  # Handles actual NaN (from float or np.nan)
        return "NA"
    if isinstance(text, str):
        # Replace case-insensitive 'nan' occurrences in strings
        return re.sub(r'\bnan\b', "NA", text, flags=re.IGNORECASE)
    return text  # Return unchanged if not string or NaN
sub['Category:Misconception'] = sub['Category:Misconception'].map(replace_nan_text)
sub.head()


sub.to_csv("submission.csv",index=False)

