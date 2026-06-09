import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42


%%writefile constants.py

# Global
SEED = 42
OUT_DIR = "/kaggle/working/"
BATCH_SIZE = 32
MAX_TOKEN = 256

# BERT
BERT_MODEL_DIR = "/kaggle/input/my-base-bert/my-bert-cls"
BERT_EPOCH = 3

# RoBERTa
ROBERTA_MODEL_DIR = "/kaggle/input/roberta-base"
ROBERTA_EPOCH = 9


df_train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
df_test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")


"""
Data Augmentation using examples in train
"""

# positives
pos = df_train[["positive_example_1", "rule", "subreddit"]].rename(
    columns={"positive_example_1": "body"}
)
pos["rule_violation"] = 1

pos_2 = df_train[["positive_example_2", "rule", "subreddit"]].rename(
    columns={"positive_example_2": "body"}
)
pos_2["rule_violation"] = 1

# negatives
neg = df_train[["negative_example_1", "rule", "subreddit"]].rename(
    columns={"negative_example_1": "body"}
)
neg["rule_violation"] = 0

neg_2 = df_train[["negative_example_2", "rule", "subreddit"]].rename(
    columns={"negative_example_2": "body"}
)
neg_2["rule_violation"] = 0

# combine
df_add = pd.concat([pos, pos_2, neg, neg_2], ignore_index=True)

# optional: drop missing texts, ensure int dtype
df_add = df_add.dropna(subset=["body"]).reset_index(drop=True)
df_add["rule_violation"] = df_add["rule_violation"].astype(int)
df_train = pd.concat([df_train, df_add], ignore_index=True)


"""
Data Augmentation using examples in test
"""

# positives
pos = df_test[["positive_example_1", "rule", "subreddit"]].rename(
    columns={"positive_example_1": "body"}
)
pos["rule_violation"] = 1

pos_2 = df_test[["positive_example_2", "rule", "subreddit"]].rename(
    columns={"positive_example_2": "body"}
)
pos_2["rule_violation"] = 1

# negatives
neg = df_test[["negative_example_1", "rule", "subreddit"]].rename(
    columns={"negative_example_1": "body"}
)
neg["rule_violation"] = 0

neg_2 = df_test[["negative_example_2", "rule", "subreddit"]].rename(
    columns={"negative_example_2": "body"}
)
neg_2["rule_violation"] = 0

# combine
df_add = pd.concat([pos, pos_2, neg, neg_2], ignore_index=True)

# optional: drop missing texts, ensure int dtype
df_add = df_add.dropna(subset=["body"]).reset_index(drop=True)
df_add["rule_violation"] = df_add["rule_violation"].astype(int)
df_train_aug = pd.concat([df_train, df_add], ignore_index=True)


%%writefile utils_bert.py

import urllib.request
import re
from urllib.parse import urlparse
import emoji
import pandas as pd


def add_rule_and_subreddit(df, is_train = True):
    new_df = pd.DataFrame()
    new_df["data"] = (
        "Rule: " + df["rule"] + " [SEP] " +
        "Subreddit: " + df["subreddit"] + " [SEP] " +
        "Comment: " + df["body"]
    )
    if is_train:
        new_df["label"] = df["rule_violation"]
        
    return new_df


def replace_urls_with_features(text):
    urls = re.findall(r"(?:http|https)://[^\s]+", text)

    for url in urls:
        seen_semantics = set()
        all_semantics = []    
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
        except ValueError:
            domain = "invalid"

        # domain
        domain_match = re.search(r"(?:https?://)?([a-z0-9\-\.]+)\.[a-z]{2,}", url.lower())
        if domain_match:
            full_domain = domain_match.group(1)
            parts = full_domain.split('.')
            for part in parts:
                if part and part not in seen_semantics and len(part) > 3:
                    all_semantics.append(f"domain:{part}")
                    seen_semantics.add(part)

        # path
        path = re.sub(r"^(?:https?://)?[a-z0-9\.-]+\.[a-z]{2,}/?", "", url.lower())
        path_parts = [p for p in re.split(r'[/_.-]+', path) if p and p.isalnum()]
        for part in path_parts:
            part_clean = re.sub(r"\.(html?|php|asp|jsp)$|#.*|\?.*", "", part)
            if part_clean and part_clean not in seen_semantics and len(part_clean) > 3:
                all_semantics.append(f"path:{part_clean}")
                seen_semantics.add(part_clean)

        if all_semantics:
            semantic_str = f"\n(URL Keywords: {' '.join(all_semantics)})"
        else:
            semantic_str = ""

        text = text.replace(url, semantic_str)

    return text


def clean_text(text):
    text = replace_urls_with_features(text) # extract semantics from URL
    text = emoji.replace_emoji(text, replace="")  # remove emoji
    text = re.sub(r'\s+', ' ', text).strip() # remove unnecessary space
    return text


def data_processing(df, is_train = True):
    df = add_rule_and_subreddit(df, is_train)
    df["data"] = df['data'].apply(clean_text)
    
    if is_train:
        df = df.drop_duplicates()

    return df


"""
Data Processing
"""
from utils_bert import data_processing

df_train_aug = data_processing(df_train_aug, is_train=True)
df_train_aug = df_train_aug.sample(frac=1).reset_index(drop=True)


"""
Devide train data into 3 
"""
size = len(df_train_aug)
i1 = (size * 1) // 5
i2 = (size * 2) // 5
i3 = (size * 3) // 5

print(f"data size {size}")
print(f"split data into 0-{i1}, {i1}-{i2}, {i2}-{i3}")

df_train_aug_1 = df_train_aug.iloc[0:i1].copy()
df_train_aug_2 = df_train_aug.iloc[i1:i2].copy()
df_train_aug_3 = df_train_aug.iloc[i2:i3].copy()


"""
Save train data
"""
df_train_aug_1.to_csv('/kaggle/working/train_data_1.csv', index=False)  
df_train_aug_2.to_csv('/kaggle/working/train_data_2.csv', index=False)  
df_train_aug_3.to_csv('/kaggle/working/train_data_3.csv', index=False) 


# %%writefile train_bert.py

# import argparse, pandas as pd
# from constants import BATCH_SIZE, MAX_TOKEN, BERT_EPOCH, BERT_MODEL_DIR, OUT_DIR, SEED
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.model_selection import train_test_split
# import copy
# from sklearn.metrics import accuracy_score, f1_score

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# # Token and Encode Function
# def tokenize_and_encode(tokenizer, comments, labels, max_length):
#     # Initialize empty lists to store tokenized inputs and attention masks
#     input_ids = []
#     attention_masks = []

#     # Iterate through each comment in the 'comments' list
#     for comment in comments:

#         # Tokenize and encode the comment using the BERT tokenizer
#         encoded_dict = tokenizer.encode_plus(
#             comment,

#             # Add special tokens like [CLS] and [SEP]
#             add_special_tokens=True,

#             truncation=True,
            
#             # Truncate or pad the comment to 'max_length'
#             max_length=max_length,

#             # Pad the comment to 'max_length' with zeros if needed
#             padding='max_length',

#             # Return attention mask to mask padded tokens
#             return_attention_mask=True,

#             # Return PyTorch tensors
#             return_tensors='pt'
#         )

#         # Append the tokenized input and attention mask to their respective lists
#         input_ids.append(encoded_dict['input_ids'])
#         attention_masks.append(encoded_dict['attention_mask'])

#     # Concatenate the tokenized inputs and attention masks into tensors
#     input_ids = torch.cat(input_ids, dim=0)
#     attention_masks = torch.cat(attention_masks, dim=0)

#     # Convert the labels to a PyTorch tensor with the data type float32
#     labels = torch.tensor(labels, dtype=torch.float32)

#     # Return the tokenized inputs, attention masks, and labels as PyTorch tensors
#     return input_ids, attention_masks, labels

    
# def train_model(model, train_loader, val_loader, device, num_epochs, patience=2):
#     # loss_fn = nn.BCELoss()  # binary cross entropy
#     loss_fn = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=2e-5)

#     # loss_fn = nn.BCEWithLogitsLoss()  
#     # optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
#     best_f1 = -1.0
#     # best_auc = -1.0
#     epochs_no_improve = 0
#     best_state = None

#     train_losses, val_losses, train_f1s, val_f1s = [], [], [], []
#     # train_losses, val_losses, train_aucs, val_aucs = [], [], [], []
    
#     for epoch in range(num_epochs):

#         model.train()
#         total_loss = 0.0
#         all_train_preds, all_train_labels = [], []
#         # train
#         for batch in train_loader:
            
#             input_ids, attention_mask, labels = [t.to(device) for t in batch]
#             labels = labels.float()

#             # prediction (number of batches)
#             output = model(input_ids=input_ids, attention_mask=attention_mask)
#             logits = output.logits.squeeze(-1)
#             probs = torch.sigmoid(logits)  # convert logits to probabilities first
#             # print(logits) # DEBUG
                 
#             # forward pass
#             # loss = loss_fn(probs, labels)
#             loss = loss_fn(logits, labels)
#             # print(loss) # DEBUG
            
#             total_loss += loss.item()

#             # backward pass
#             optimizer.zero_grad()
#             loss.backward()
            
#             # update weights
#             optimizer.step()

#             preds = (probs > 0.5).long()
#             all_train_preds.extend(preds.cpu().tolist())
#             all_train_labels.extend(labels.cpu().tolist())
            
#         train_loss = total_loss / len(train_loader)
#         train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')

#         train_losses.append(train_loss)
#         train_f1s.append(train_f1)
        
#         # Validate
#         model.eval()
#         val_loss = 0.0
#         all_preds, all_labels = [], []
#         with torch.no_grad():
#             for batch in val_loader:
                
#                 input_ids, attention_mask, labels = [t.to(device) for t in batch]
#                 labels = labels.float()
#                 output = model(input_ids=input_ids, attention_mask=attention_mask)
#                 logits = output.logits.squeeze(-1)
#                 probs = torch.sigmoid(logits)
                
#                 # val_loss += loss_fn(probs, labels.float()).item()
#                 val_loss += loss_fn(logits, labels.float()).item()
                
#                 preds = (probs > 0.5).long()
#                 all_preds.extend(preds.cpu().tolist())
#                 all_labels.extend(labels.cpu().tolist())
                
#         val_f1 = f1_score(all_labels, all_preds, average='macro')
#         val_f1s.append(val_f1)
        
#         val_losses.append(val_loss)
#         val_loss /= len(val_loader)
        
#         print(f"Epoch {epoch+1}/{num_epochs} - "
#               f"Train Loss: {train_loss:.4f}, Train Acc: {train_f1:.4f} - "
#               f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

#         if val_f1 > best_f1:
#             best_f1 = val_f1
#             epochs_no_improve = 0
#             best_state = copy.deepcopy(model.state_dict())
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 print(f"Early stopping (no val F1 improvement for {patience} epochs). Best Val F1: {best_f1:.4f}")
#                 break

#     if best_state is not None:
#         model.load_state_dict(best_state)

#     return model


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data", type=str, required=True)
#     args = parser.parse_args()

#     df_train_aug_1 = pd.read_csv(args.data)
#     y = df_train_aug_1["label"]
#     X = df_train_aug_1["data"]
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

#     tokenizer = BertTokenizer.from_pretrained(
#         BERT_MODEL_DIR,
#         do_lower_case=True,
#         local_files_only=True
#     )
    
#     config = BertConfig.from_pretrained(
#         BERT_MODEL_DIR,
#         local_files_only=True
#     )
    
#     model = BertForSequenceClassification.from_pretrained(
#         BERT_MODEL_DIR,
#         config=config,
#         local_files_only=True
#     )
#     model = model.to(device)
    
#     input_ids, attention_masks, labels = tokenize_and_encode(
#         tokenizer,
#         X_train,
#         y_train.values,
#         MAX_TOKEN,
#     )

#     val_input_ids, val_attention_masks, val_labels = tokenize_and_encode(
#         tokenizer,
#         X_val,
#         y_val.values,
#         MAX_TOKEN,
#     )

#     # Creating DataLoader for the balanced dataset
#     train_dataset = TensorDataset(input_ids, attention_masks, labels)
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
#     # validation set 
#     val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
#     model = train_model(model, train_loader, val_loader, device, num_epochs=BERT_EPOCH)
#     torch.save(model.state_dict(), OUT_DIR+"bert")

# if __name__ == "__main__":
#     main()



# !python train_bert.py --data /kaggle/working/train_data_1.csv


%%writefile train_roberta.py

import argparse, pandas as pd
from constants import BATCH_SIZE, MAX_TOKEN, ROBERTA_EPOCH, ROBERTA_MODEL_DIR, OUT_DIR, SEED
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import copy
from sklearn.metrics import accuracy_score, f1_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Token and Encode Function
def tokenize_and_encode(tokenizer, comments, labels, max_length):
    # Initialize empty lists to store tokenized inputs and attention masks
    input_ids = []
    attention_masks = []

    # Iterate through each comment in the 'comments' list
    for comment in comments:

        # Tokenize and encode the comment using the BERT tokenizer
        encoded_dict = tokenizer.encode_plus(
            comment,

            # Add special tokens like [CLS] and [SEP]
            add_special_tokens=True,

            truncation=True,
            
            # Truncate or pad the comment to 'max_length'
            max_length=max_length,

            # Pad the comment to 'max_length' with zeros if needed
            padding='max_length',

            # Return attention mask to mask padded tokens
            return_attention_mask=True,

            # Return PyTorch tensors
            return_tensors='pt'
        )

        # Append the tokenized input and attention mask to their respective lists
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Concatenate the tokenized inputs and attention masks into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Convert the labels to a PyTorch tensor with the data type float32
    labels = torch.tensor(labels, dtype=torch.float32)

    # Return the tokenized inputs, attention masks, and labels as PyTorch tensors
    return input_ids, attention_masks, labels

    
def train_model(model, train_loader, val_loader, device, num_epochs, patience=2):
    # loss_fn = nn.BCELoss()  # binary cross entropy
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.01) # weight_decay=0.01

    # loss_fn = nn.BCEWithLogitsLoss()  
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    best_f1 = -1.0
    # best_auc = -1.0
    epochs_no_improve = 0
    best_state = None

    train_losses, val_losses, train_f1s, val_f1s = [], [], [], []
    # train_losses, val_losses, train_aucs, val_aucs = [], [], [], []
    
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0.0
        all_train_preds, all_train_labels = [], []
        # train
        for batch in train_loader:
            
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            labels = labels.float()

            # prediction (number of batches)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits.squeeze(-1)
            probs = torch.sigmoid(logits)  # convert logits to probabilities first
            # print(logits) # DEBUG
                 
            # forward pass
            # loss = loss_fn(probs, labels)
            loss = loss_fn(logits, labels)
            # print(loss) # DEBUG
            
            total_loss += loss.item()

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # update weights
            optimizer.step()

            preds = (probs > 0.5).long()
            all_train_preds.extend(preds.cpu().tolist())
            all_train_labels.extend(labels.cpu().tolist())
            
        train_loss = total_loss / len(train_loader)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')

        train_losses.append(train_loss)
        train_f1s.append(train_f1)
        
        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                
                input_ids, attention_mask, labels = [t.to(device) for t in batch]
                labels = labels.float()
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output.logits.squeeze(-1)
                probs = torch.sigmoid(logits)
                
                # val_loss += loss_fn(probs, labels.float()).item()
                val_loss += loss_fn(logits, labels.float()).item()
                
                preds = (probs > 0.5).long()
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_f1s.append(val_f1)
        
        val_losses.append(val_loss)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_f1:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping (no val F1 improvement for {patience} epochs). Best Val F1: {best_f1:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--num", type=str, required=True)

    args = parser.parse_args()

    df_train_aug_1 = pd.read_csv(args.data)
    y = df_train_aug_1["label"]
    X = df_train_aug_1["data"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ROBERTA_MODEL_DIR,
        use_fast=True,
        local_files_only=True
    )  
    
    # Config
    config = AutoConfig.from_pretrained(
        ROBERTA_MODEL_DIR,
        num_labels=1,  # 1 --> regression (MSE Loss), 2--> binary classification (Cross-entropy)
        problem_type="single_label_classification",
        local_files_only=True,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
    )
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_MODEL_DIR,
        config=config,
        local_files_only=True
    )
    model = model.to(device)
    
    input_ids, attention_masks, labels = tokenize_and_encode(
        tokenizer,
        X_train,
        y_train.values,
        MAX_TOKEN,
    )

    val_input_ids, val_attention_masks, val_labels = tokenize_and_encode(
        tokenizer,
        X_val,
        y_val.values,
        MAX_TOKEN,
    )

    # Creating DataLoader for the balanced dataset
    train_dataset = TensorDataset(input_ids, attention_masks, labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # validation set 
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = train_model(model, train_loader, val_loader, device, num_epochs=ROBERTA_EPOCH)
    torch.save(model.state_dict(), OUT_DIR+"roberta_"+args.num)

if __name__ == "__main__":
    main()



!python train_roberta.py --data /kaggle/working/train_data_1.csv --num 1


!python train_roberta.py --data /kaggle/working/train_data_2.csv --num 2


%%writefile utils_deberta.py
import pandas as pd
import re
from typing import List

# Pre-compiled regex patterns
URL_PATTERN = re.compile(r'https?://[^\s/$.?#].[^\s]*')
DOMAIN_PATTERN = re.compile(r"(?:https?://)?([a-z0-9\-\.]+)\.[a-z]{2,}")
CLEAN_PATH_PART_PATTERN = re.compile(r"\.(html?|php|asp|jsp)$|#.*|\?.*")


def url_to_semantics(text: str) -> str:
    """
    Extracts semantic keywords from URLs found in a given text.
    Returns a formatted string containing semantic features.
    """
    if not isinstance(text, str):
        return ""

    urls = URL_PATTERN.findall(text)
    if not urls:
        return "" 

    all_semantics = []
    seen_semantics = set()

    for url in urls:
        url_lower = url.lower()

        # Extract domain parts
        domain_match = DOMAIN_PATTERN.search(url_lower)
        if domain_match:
            full_domain = domain_match.group(1)
            for part in full_domain.split('.'):
                if part and part not in seen_semantics and len(part) > 3:
                    all_semantics.append(f"domain:{part}")
                    seen_semantics.add(part)

        # Extract path parts
        path = re.sub(r"^(?:https?://)?[a-z0-9\.-]+\.[a-z]{2,}/?", "", url_lower)
        path_parts = [p for p in re.split(r'[/_.-]+', path) if p and p.isalnum()]
        
        for part in path_parts:
            part_clean = CLEAN_PATH_PART_PATTERN.sub("", part)
            if part_clean and part_clean not in seen_semantics and len(part_clean) > 3:
                all_semantics.append(f"path:{part_clean}")
                seen_semantics.add(part_clean)

    return f"\nURL Keywords: {' '.join(all_semantics)}" if all_semantics else ""


def extract_violation_examples(df: pd.DataFrame, dataset_type: str) -> List[pd.DataFrame]:
    """
    Extracts positive and negative examples from a given dataset.
    Returns a list of cleaned and labeled DataFrames.
    """
    examples = []
    for violation_type in ["positive", "negative"]:
        label = 1 if violation_type == "positive" else 0
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            if col in df.columns:
                sub_df = df[[col, "rule", "subreddit"]].copy()
                sub_df = sub_df.rename(columns={col: "body"})
                sub_df["rule_violation"] = label
                sub_df.dropna(subset=["body"], inplace=True)
                sub_df = sub_df[sub_df["body"].str.strip().str.len() > 0]
                if not sub_df.empty:
                    examples.append(sub_df)
    return examples


def get_dataframe_to_train(data_path: str, seed: int = 42) -> pd.DataFrame:
    """
    Loads train and test datasets, extracts and flattens rule violation examples,
    deduplicates, and returns a shuffled DataFrame ready for training.

    Parameters:
        data_path (str): Path to the dataset.
        seed (int): Random seed for reproducible shuffling.

    Returns:
        pd.DataFrame: Cleaned and shuffled training dataset.
    """
    train_df = pd.read_csv(f"{data_path}/train.csv")
    test_df = pd.read_csv(f"{data_path}/test.csv")

    combined = []

    if {"body", "rule", "subreddit", "rule_violation"}.issubset(train_df.columns):
        combined.append(train_df[["body", "rule", "subreddit", "rule_violation"]].copy())

    combined.extend(extract_violation_examples(train_df, "train"))
    combined.extend(extract_violation_examples(test_df, "test"))

    full_df = pd.concat(combined, axis=0, ignore_index=True)

    # Deduplicate
    full_df.drop_duplicates(subset=["body", "rule", "subreddit"], inplace=True)
    full_df.drop_duplicates(subset=["body", "rule"], keep="first", inplace=True)

    # Shuffle with specified seed
    return full_df.sample(frac=1, random_state=seed).reset_index(drop=True)



%%writefile train_deberta.py
import os
import pandas as pd
import torch
import random
import numpy as np
import argparse
from sklearn.model_selection import train_test_split 
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from utils_deberta import get_dataframe_to_train, url_to_semantics

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CFG:
    model_name_or_path = "/kaggle/input/debertav3base"
#    model_name_or_path = "/kaggle/input/deberta-v3-small/deberta-v3-small"
    data_path = "/kaggle/input/jigsaw-agile-community-rules/"
    output_dir = "./dbv3_base_ens_model"
#    output_dir = "./dbv3_small_ens_model"
  
    EPOCHS = 5
    LEARNING_RATE = 2e-5  
    
    MAX_LENGTH = 246
    BATCH_SIZE = 8

class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def add_url_semantics_column(df):
    return df['body'].apply(lambda x: x + url_to_semantics(x))

def build_input_text(df):
    return df['rule'] + "[SEP]" + df['body_with_url']

def prepare_dataset(df, tokenizer, max_length, is_train=True):
    encodings = tokenizer(
        df['input_text'].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length
    )
    if is_train:
        labels = df['rule_violation'].tolist()
        return JigsawDataset(encodings, labels)
    else:
        return JigsawDataset(encodings)

def train(seed=42):
    seed_everything(seed)
    print(f"\n Loading and preparing training data with seed={seed} ...")
    training_data_df = get_dataframe_to_train(CFG.data_path, seed=seed)
    print(f"Training dataset size: {len(training_data_df)}")

    training_data_df['body_with_url'] = add_url_semantics_column(training_data_df)
    training_data_df['input_text'] = build_input_text(training_data_df)

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path)
    train_dataset = prepare_dataset(training_data_df, tokenizer, CFG.MAX_LENGTH, is_train=True)

    model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name_or_path, num_labels=2)

    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        num_train_epochs=CFG.EPOCHS,
        learning_rate=CFG.LEARNING_RATE,
        per_device_train_batch_size=CFG.BATCH_SIZE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    return trainer, tokenizer

def predict(trainer, tokenizer, seed):
    print(f"\nLoading test data for seed={seed}...")
    test_df = pd.read_csv(f"{CFG.data_path}/test.csv")
    test_df['body_with_url'] = add_url_semantics_column(test_df)
    test_df['input_text'] = build_input_text(test_df)
    
    test_dataset = prepare_dataset(test_df, tokenizer, CFG.MAX_LENGTH, is_train=False)

    print("Running prediction...")
    predictions = trainer.predict(test_dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()

    submission_df = pd.DataFrame({
        "row_id": test_df["row_id"],
        "rule_violation": probs
    })

    output_file = f"submission_deberta_{seed}.csv"
    submission_df.to_csv(output_file, index=False)
    print(f"ðŸ“¤ Saved submission to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Train DeBERTa on rule violation data with multiple seeds.")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated list of seeds to run (e.g. 1,42,1337)"
    )
    args = parser.parse_args()

    seed_list = [int(s) for s in args.seeds.split(",") if s.strip().isdigit()]
    
    for seed in seed_list:
        print(f"\n\n===== Running training and prediction for seed: {seed} =====")
        trainer, tokenizer = train(seed=seed)
        predict(trainer, tokenizer, seed=seed)

if __name__ == "__main__":
    main()



!python train_deberta.py


%%writefile inference_roberta.py

import argparse, pandas as pd
from constants import BATCH_SIZE, MAX_TOKEN, ROBERTA_MODEL_DIR, OUT_DIR, SEED
from utils_bert import data_processing
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    DataCollatorWithPadding
)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def tokenize_test(tokenizer, comments, max_length):
    input_ids, attention_masks = [], []
    for comment in comments:
        enc = tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids.append(enc["input_ids"])
        attention_masks.append(enc["attention_mask"])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


def predict(model, loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask = [t.to(device) for t in batch]
            
            output = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = output.logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().tolist())
            
    return all_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    args = parser.parse_args()
    
    df_test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    submission = pd.DataFrame(df_test["row_id"])
    df_test = data_processing(df_test, is_train=False)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ROBERTA_MODEL_DIR,
        do_lower_case=True, 
        local_files_only=True
    )  
    
    # Config
    config = AutoConfig.from_pretrained(
        ROBERTA_MODEL_DIR,
        num_labels=1,  # 1 --> regression (MSE Loss), 2--> binary classification (Cross-entropy)
        problem_type="single_label_classification",
        local_files_only=True,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
    )
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_MODEL_DIR,
        config=config,
        local_files_only=True
    )

    # loading weight from the trained model
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    test_input_ids, test_attention_masks = tokenize_test(tokenizer, df_test["data"], max_length=MAX_TOKEN)
    test_dataset = TensorDataset(test_input_ids, test_attention_masks)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    res = predict(model, test_loader, device)
    submission["rule_violation"] = res
    submission.to_csv(args.save, index=False)
    print(submission)
    
if __name__ == "__main__":
    main()


!python inference_roberta.py --model /kaggle/working/roberta_1 --save /kaggle/working/submission_roberta_1.csv


!python inference_roberta.py --model /kaggle/working/roberta_2 --save /kaggle/working/submission_roberta_2.csv


res_deberta = pd.read_csv("/kaggle/working/submission_deberta_42.csv")
res_deberta


df1 = pd.read_csv("/kaggle/working/submission_roberta_1.csv")
df2 = pd.read_csv("/kaggle/working/submission_roberta_2.csv")


import pandas as pd
import numpy as np
from scipy.stats import rankdata

df1 = pd.read_csv("/kaggle/working/submission_roberta_1.csv")
df2 = pd.read_csv("/kaggle/working/submission_roberta_2.csv")
df3 = pd.read_csv("/kaggle/working/submission_deberta_42.csv")

r1 = rankdata(df1["rule_violation"])
r2 = rankdata(df2["rule_violation"])
r3 = rankdata(df3["rule_violation"])

final_rank = (r1 + r2 + r3) / 3
final_pred = final_rank / np.max(final_rank) # normalization

df1["rule_violation"] = final_pred
df1.to_csv("submission.csv", index=False)


df1

