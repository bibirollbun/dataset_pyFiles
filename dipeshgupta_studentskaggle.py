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


## import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import re
import random
from textblob import TextBlob
from typing import List, Tuple
from sklearn.metrics import average_precision_score

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(21)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(21)

### ENHANCED TEXT PROCESSING ###
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s.,!?+/=-]", "", text)  # Keep math operators
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Synonym Dictionary
def get_synonyms(word):
    synonyms_dict = {
       "add": ["plus", "sum", "increase", "combine", "total", "aggregate"],
        "subtract": ["minus", "less", "deduct", "take away", "reduce", "decrease"],
        "multiply": ["times", "product", "scale", "increase by", "factor"],
        "divide": ["over", "fraction", "split", "partition", "divide by", "quotient"],
        "equals": ["is", "is equal to", "equivalent to", "results in", "gives"],
        "greater": ["more than", "exceeds", "larger than", "bigger than"],
        "less": ["fewer than", "smaller than", "not as much as", "lower than"],
        "calculate": ["compute", "determine", "evaluate", "find", "work out"],
        "solve": ["find", "work out", "resolve", "answer", "figure out"],
        "function": ["operation", "process", "method", "mapping"],
        "fraction": ["part", "portion", "segment", "piece", "ratio"],
        "percentage": ["percent", "proportion", "rate", "fraction of 100"],
        "variable": ["unknown", "symbol", "letter", "placeholder"],
        "equation": ["formula", "expression", "statement", "identity"],
        "geometry": ["shape", "form", "figure", "configuration"],
        "algebra": ["symbolic math", "equation solving", "variable math"],
        "theorem": ["principle", "law", "proposition", "statement"],
        "proof": ["demonstration", "verification", "validation", "justification"],
        "estimate": ["approximate", "roughly calculate", "guesstimate", "evaluate"],
        "sum": ["total", "aggregate", "addition", "result of addition"],
        "difference": ["subtraction", "remainder", "less", "result of subtraction"],
        "product": ["multiplication", "result of multiplying", "outcome of multiplication"],
        "quotient": ["result of division", "division result", "outcome of division"],
        "integer": ["whole number", "counting number"],
        "decimal": ["fractional number", "non-integer"],
        "ratio": ["proportion", "relationship", "comparison"],
        "mean": ["average", "arithmetic mean"],
        "median": ["middle value", "central tendency"],
        "mode": ["most frequent value", "common value"],
        "range": ["difference between max and min", "spread"],
        "outlier": ["anomalous value", "exception"],
        "data": ["information", "statistics", "figures"],
        "graph": ["chart", "plot", "diagram"],
        "linear": ["straight", "direct", "proportional"],
        "quadratic": ["parabolic", "second degree"],
        "exponential": ["growth function", "rapid increase"],
        "logarithmic": ["inverse exponential", "log function"],
        "set": ["collection", "group", "assembly"],
        "element": ["member", "item", "component"],
        "union": ["combination", "joining", "merging"],
        "intersection": ["overlap", "common elements"],
        "complement": ["remaining elements", "difference"],
        "perimeter": ["boundary length", "outer length"],
        "area": ["surface size", "extent"],
        "volume": ["capacity", "space", "amount"],
        "angle": ["degree", "arc", "corner"],
        "hypotenuse": ["longest side", "opposite side"],
        "adjacent": ["next to", "neighboring"],
        "opposite": ["across from", "facing"],
        "coordinate": ["point", "location", "position"],
        "axis": ["line", "reference line"],
        "slope": ["gradient", "incline", "steepness"],
        "derivative": ["rate of change", "slope of the tangent"],
        "integral": ["area under the curve", "accumulation"],
        "limit": ["boundary", "threshold", "approaching value"],
        "sequence": ["ordered list", "series"],
        "probability": ["likelihood", "chance", "odds"],
        "statistic": ["data point", "measure", "figure"],
        "distribution": ["spread", "allocation", "arrangement"],
        "variance": ["spread", "dispersion"],
        "standard deviation": ["measure of spread", "variability"],
        "correlation": ["relationship", "association"],
        "regression": ["trend line", "fitting"],
        "hypothesis": ["assumption", "theory"],
        "test": ["experiment", "evaluation"],
        "sample": ["subset", "selection"],
        "population": ["entire group", "whole set"],
        "parameter": ["characteristic", "attribute"],
    }
    return synonyms_dict.get(word, [])

# Simple Data Augmentation Functions
def synonym_replacement(words, n=1):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if len(get_synonyms(word)) > 0]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if synonyms:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words

def add_word(words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1 and counter < 10:
        random_word = random.choice(words)
        synonyms = get_synonyms(random_word)
        counter += 1
    if synonyms:
        random_syn = random.choice(synonyms)
        random_idx = random.randint(0, len(words))
        words.insert(random_idx, random_syn)

def random_insertion(words, n=1):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def augment_text(text, num_sr=1, num_ri=1):
    words = text.split()
    words = synonym_replacement(words, n=num_sr)
    words = random_insertion(words, n=num_ri)
    return ' '.join(words)

# Feature Engineering
def extract_features(df):
    keywords = ['+', '-', '=', 'fraction', 'sum', 'product', '/', 'add', 'subtract', 'multiply', 'divide']
    df['question_len'] = df['QuestionText'].fillna('').apply(lambda x: len(x.split()))
    df['explanation_len'] = df['StudentExplanation'].fillna('').apply(lambda x: len(x.split()))
    for kw in keywords:
        if kw:
            df[f'kw_{kw}'] = df['QuestionText'].fillna('').str.contains(re.escape(kw)).astype(int)
    df['question_sentiment_polarity'] = df['QuestionText'].fillna('').apply(lambda x: TextBlob(x).sentiment.polarity)
    df['explanation_sentiment_polarity'] = df['StudentExplanation'].fillna('').apply(lambda x: TextBlob(x).sentiment.polarity)
    df['question_sentiment_subjectivity'] = df['QuestionText'].fillna('').apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df['explanation_sentiment_subjectivity'] = df['StudentExplanation'].fillna('').apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return df

class MultiTaskBertTopK(nn.Module):
    def __init__(self, model_name: str, num_categories: int, num_misconceptions: int, extra_feat_dim: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.category_head = nn.Linear(self.bert.config.hidden_size, num_categories)
        self.misconception_head = nn.Linear(self.bert.config.hidden_size, num_misconceptions)
        self.correct_head = nn.Linear(self.bert.config.hidden_size + extra_feat_dim, 1)
        self.dropout = nn.Dropout(0.35)

        # Define loss functions here
        self.category_loss_fn = nn.CrossEntropyLoss()
        self.misconception_loss_fn = nn.CrossEntropyLoss()
        self.correct_loss_fn = nn.BCEWithLogitsLoss()


    def forward(self, input_ids=None, attention_mask=None, extra_feats=None,
                labels=None, misconception_labels=None, correct=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)  # Use pooled_output and apply dropout

        category_logits = self.category_head(pooled_output)
        misconception_logits = self.misconception_head(pooled_output)
        combined = torch.cat([pooled_output, extra_feats], dim=1)  # Concatenate pooled output with extra features
        correct_logits = self.correct_head(combined)

        loss = None
        if labels is not None and misconception_labels is not None and correct is not None:
            category_loss = self.category_loss_fn(category_logits, labels)
            misconception_loss = self.misconception_loss_fn(misconception_logits, misconception_labels)
            # Ensure correct_logits and correct have compatible shapes (batch_size,)
            correct_loss = self.correct_loss_fn(correct_logits.squeeze(-1), correct)
            loss = category_loss + misconception_loss + correct_loss

        # When using Trainer, the model is expected to return (loss, ...) during training
        # and (logits, ...) during evaluation/prediction.
        # We return the combined loss, and the individual logits.
        # The Trainer will use the first element as the loss for backpropagation.
        return (loss, category_logits, misconception_logits, correct_logits) if loss is not None else (category_logits, misconception_logits, correct_logits)


class MathDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: BertTokenizer, cats: np.ndarray = None, miscons: np.ndarray = None,
                 correct: np.ndarray = None, extra_feats: np.ndarray = None, max_length: int = 128):
        self.texts = texts
        self.cats = cats
        self.miscons = miscons
        self.correct = correct
        self.extra_feats = extra_feats
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        if (self.cats is not None and self.miscons is not None and
            self.correct is not None and self.extra_feats is not None):
            item['labels'] = torch.tensor(self.cats[idx], dtype=torch.long)
            item['misconception_labels'] = torch.tensor(self.miscons[idx], dtype=torch.long)
            # Ensure correct label is float for BCEWithLogitsLoss
            item['correct'] = torch.tensor(self.correct[idx], dtype=torch.float)
            item['extra_feats'] = torch.tensor(self.extra_feats[idx], dtype=torch.float)
        elif self.extra_feats is not None: # Include extra_feats even if no labels (for prediction)
             item['extra_feats'] = torch.tensor(self.extra_feats[idx], dtype=torch.float)
        return item

# Custom collate function
def collate_fn(batch):
    item = {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'extra_feats': torch.stack([x['extra_feats'] for x in batch])
    }
    # Only include labels if they exist in the first item of the batch
    if 'labels' in batch[0]:
        item['labels'] = torch.stack([x['labels'] for x in batch])
        item['misconception_labels'] = torch.stack([x['misconception_labels'] for x in batch])
        item['correct'] = torch.stack([x['correct'] for x in batch])
    return item


# CUSTOM MAP@3 METRIC
def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    map3 = 0
    for i in range(len(labels)):
        if labels[i] in top3[i]:
            ranks = np.where(top3[i] == labels[i])[0] + 1
            for rank in ranks:
                 map3 += 1.0 / rank
    return {"map@3": map3 / len(labels)}


def train_and_save_model(train_file: str, model_name: str = "bert-base-uncased", epochs: int =128, batch_size: int = 128) -> Tuple:
    # Load and preprocess data
    train_df = pd.read_csv(train_file)
    train_df['Misconception'] = train_df['Misconception'].fillna('NA')

    # Feature engineering
    train_df = extract_features(train_df)

    # Data augmentation on 'input_text'
    # train_df['input_text'] = (train_df['QuestionText'].fillna('') + ' ' + train_df['StudentExplanation'].fillna('')).apply(clean_text).apply(augment_text)
    # Augmentation applied to the text used for BERT input
    train_df['text_for_bert'] = (train_df['QuestionText'].fillna('') + ' ' + train_df['StudentExplanation'].fillna('')).apply(clean_text)
    # Apply augmentation to the text used for BERT input
    train_df['augmented_text'] = train_df['text_for_bert'].apply(augment_text)


    # Prepare inputs with concatenated text and original Answer
    # Use the augmented text for BERT input
    train_df['text'] = train_df.apply(lambda x: f"Question: {x['augmented_text']} [SEP] Answer: {x['MC_Answer']}", axis=1)


    # Encode labels
    cat_encoder = LabelEncoder()
    misc_encoder = LabelEncoder()
    train_df['cat_label'] = cat_encoder.fit_transform(train_df['Category'])
    train_df['misc_label'] = misc_encoder.fit_transform(train_df['Misconception'])

    # Create the 'Correct' column
    idx = train_df.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
    correct = train_df.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
    correct = correct.sort_values('c', ascending=False)
    correct = correct.drop_duplicates(['QuestionId'])
    correct['is_correct'] = 1
    train_df = train_df.merge(correct[['QuestionId', 'MC_Answer', 'is_correct']], on=['QuestionId', 'MC_Answer'], how='left')
    train_df['is_correct'] = train_df['is_correct'].fillna(0)

    # Remove classes with insufficient instances
    min_class_size = 2
    train_df = train_df[train_df['Category'].map(train_df['Category'].value_counts()) >= min_class_size]

    # Select features for model input
    feature_cols = ['question_len', 'explanation_len'] + [col for col in train_df.columns if col.startswith('kw_')] + [
                    #'question_sentiment_polarity', 'explanation_sentiment_polarity',
                    'question_sentiment_subjectivity', 'explanation_sentiment_subjectivity']

    # Scale features
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

    # Train-validation split stratified on correctness
    train_data, val_data = train_test_split(
        train_df,
        test_size=0.25,
        stratify=train_df['is_correct'],
        random_state=41
    )

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Datasets
    train_ds = MathDataset(
        train_data['text'].values,
        tokenizer=tokenizer,
        cats=train_data['cat_label'].values,
        miscons=train_data['misc_label'].values,
        correct=train_data['is_correct'].values,
        extra_feats=train_data[feature_cols].values,
        max_length=128
    )

    val_ds = MathDataset(
        val_data['text'].values,
        tokenizer=tokenizer,
        cats=val_data['cat_label'].values,
        miscons=val_data['misc_label'].values,
        correct=val_data['is_correct'].values,
        extra_feats=val_data[feature_cols].values,
        max_length=128
    )

    # Initialize model
    model = MultiTaskBertTopK(
        model_name,
        len(cat_encoder.classes_),
        len(misc_encoder.classes_),
        extra_feat_dim=len(feature_cols)
    ).to(device)

    # Training configuration
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        do_train=True,
        do_eval=True,
        learning_rate=5e-5,
        num_train_epochs=epochs,
        eval_strategy='epoch',  # Evaluate every epoch
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_map@3',  # Use the custom metric for best model
        greater_is_better=True,  # Higher MAP@3 is better
        fp16=True,
        logging_steps=100,
        report_to=["none"],
        bf16=False,
    )


    # Custom compute_metrics function for the Trainer
    def compute_metrics(p):
        # p.predictions will now be (loss, category_logits, misconception_logits, correct_logits) during training
        # During evaluation, p.predictions will be (category_logits, misconception_logits, correct_logits) as loss is None
        # p.label_ids will be a tuple of (labels, misconception_labels, correct)

        # Access category logits and labels
        # In evaluation, p.predictions is a tuple of 3 logits
        category_logits = p.predictions[0] if isinstance(p.predictions, tuple) and len(p.predictions) == 3 else p.predictions
        category_labels = p.label_ids[0] if isinstance(p.label_ids, tuple) else p.label_ids

        # Call the user-provided compute_map3 with logits and labels
        return compute_map3((category_logits, category_labels))


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn, # Use the custom collate_fn
        compute_metrics=compute_metrics # Pass the custom compute_metrics function
    )

    # Train and save
    trainer.train()
    trainer.save_model("best_model")

    # Save encoders and scaler
    with open('cat_encoder.json', 'w') as f:
        json.dump({'classes_': cat_encoder.classes_.tolist()}, f)
    with open('misc_encoder.json', 'w') as f:
        json.dump({'classes_': misc_encoder.classes_.tolist()}, f)
    with open('scaler.json', 'w') as f:
        json.dump({'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'features': feature_cols}, f)


    return model, tokenizer, cat_encoder, misc_encoder, scaler, feature_cols

def predict_and_format(model, tokenizer, cat_enc, misc_enc, scaler, feature_cols, test_file):
    # Load the test dataset
    test_df = pd.read_csv(test_file)

    # Clean and prepare the input text
    test_df['input_text'] = test_df['QuestionText'].fillna('') + ' ' + test_df['StudentExplanation'].fillna('')
    test_df['input_text'] = test_df['input_text'].apply(clean_text)

    # Feature engineering for the test set
    test_df = extract_features(test_df)
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Prepare the text for the model
    test_df['text'] = test_df.apply(lambda x: f"Question: {x['input_text']} [SEP] Answer: {x['MC_Answer']}", axis=1)

    # Create the dataset for predictions
    pred_ds = MathDataset(
        test_df['text'].values,
        tokenizer=tokenizer,
        extra_feats=test_df[feature_cols].values,
        max_length=256
    )

    # Create a DataLoader for the predictions
    pred_loader = DataLoader(pred_ds, batch_size=128, collate_fn=collate_fn) # Use the custom collate_fn

    model.eval()
    category_preds = []
    misconception_preds = []

    with torch.no_grad():
        for batch in tqdm(pred_loader):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'extra_feats': batch['extra_feats'].to(device)
            }
            # Model now returns a tuple including potential loss during training, but only logits during eval
            # Access the logits based on the return structure during evaluation
            outputs = model(**inputs)
            # During evaluation (model.eval()), the loss will be None, and the model returns (category_logits, misconception_logits, correct_logits)
            category_logits, misconception_logits, _ = outputs # Unpack the tuple

            category_preds.append(category_logits.cpu().numpy())
            misconception_preds.append(misconception_logits.cpu().numpy())

    # Stack predictions
    category_preds = np.vstack(category_preds)
    misconception_preds = np.vstack(misconception_preds)

    # Get top K predictions for categories and misconceptions
    top_k_categories = np.argsort(category_preds, axis=1)[:, -3:][:, ::-1]
    top_k_misconceptions = np.argsort(misconception_preds, axis=1)[:, -3:][:, ::-1]

    # Prepare the submission DataFrame
    submission = []
    for i in range(len(test_df)):
        cat_preds = cat_enc.classes_[top_k_categories[i]]
        misc_preds = misc_enc.classes_[top_k_misconceptions[i]]

        combined_predictions = []
        # Combine top predictions for submission
        for j in range(3):
             # Ensure we don't go out of bounds if a class has fewer than 3 unique values
             cat = cat_preds[j] if j < len(cat_preds) else "NA"
             misc = misc_preds[j] if j < len(misc_preds) else "NA"
             combined_predictions.append(f"{cat}:{misc}")

        submission.append({
            'row_id': test_df['row_id'].iloc[i],
            'Category:Misconception': ' '.join(combined_predictions)
        })

    return pd.DataFrame(submission)

### MAIN EXECUTION ###
if __name__ == "__main__":
    train_file = "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"
    test_file = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
    model_name = "/kaggle/input/bert/transformers/bert/1/"

    print("Training model...")
    model, tokenizer, cat_enc, misc_enc, scaler, feature_cols = train_and_save_model(train_file, model_name)

    print("\nGenerating predictions...")
    submission = predict_and_format(model, tokenizer, cat_enc, misc_enc, scaler, feature_cols, test_file)

    submission.set_index('row_id').to_csv("submission.csv", index=True)
    print("\nSubmission saved successfully!")
    print(submission.head())




