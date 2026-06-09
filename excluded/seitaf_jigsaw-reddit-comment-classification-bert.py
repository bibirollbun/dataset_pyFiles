import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
import re

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


SEED = 42
batch_size = 32
token_max_length = 256


df_train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
df_test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
sample_submission = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")

print("train size: ", df_train.shape)
print("test size: ", df_test.shape)


df_train.head(10)


df_test.head()


print('NA in train data:', df_train.isna().values.any())
print('NA in test data:', df_test.isna().values.any())


counts = df_train["rule_violation"].value_counts().sort_index()
labels = ['No Violation (0)', 'Violation (1)']

plt.bar(labels, counts)
plt.title("Rule Violation Distribution")
plt.xlabel("Rule Violation")
plt.ylabel("Count")
plt.show()


import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score


def add_rule_and_subreddit(df):

    new_df = pd.DataFrame()
    new_df["data"] = "Rule: " + df["rule"] + \
              " Subreddit: " + df["subreddit"] + \
              " Comment: " + df['body']
    new_df["label"] = df["rule_violation"]

    return new_df


"""
(Gray Rules): Adding Test data
"""
# df_sample = df_test.sample(frac=0.30, random_state=SEED).reset_index(drop=True)


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


df_train.shape


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


df_train_aug = add_rule_and_subreddit(df_train_aug)


print(df_train_aug["data"].loc[9000,])


"""
Duplicate
"""
duplicates = df_train_aug[df_train_aug.duplicated()]
# print(duplicates)
print(df_train_aug.shape)
df_train_aug = df_train_aug.drop_duplicates()
print(df_train_aug.shape)


"""
Capital letters count
"""
# def find_capitals(text):
#     matches = re.findall(r"[A-Z!]", text)
#     return len(matches)/len(text)
    
# df_train_aug["capital_ratio"] = df_train_aug["data"].apply(find_capitals)
# sns.histplot(data=df_train_aug, x='capital_ratio', hue='label')


"""
URLs count
"""
def find_url(text):
    # Finds a full URL starting with http or https    
    matches = re.findall(r"(?:http|https)://[^\s]+", text)

    if matches:
        return True
    else:
        return False

df_train_aug['url'] = df_train_aug['data'].apply(find_url)

# group label and url 
plot_data = df_train_aug.groupby(['label', 'url']).size().reset_index(name='Count')


# visuzalize
sns.set_style("whitegrid")
plt.figure(figsize=(7, 5))
palette_dict = {True: 'teal', False: 'darkorange'}

bar_plot = sns.barplot(
    data=plot_data,
    x='label',
    y='Count',
    hue='url', 
    palette=palette_dict
)

plt.title('Count of URLs (True/False) per Label Category', fontsize=14)
plt.xlabel('Label Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='URL Present', loc='upper right')

for container in bar_plot.containers:
    bar_plot.bar_label(container)

plt.show()


print(df_train_aug["label"].loc[3,])
print(df_train_aug["data"].loc[3,]) # --> http://sh.ors.it/PALI2 (Not really helpful)

print(df_train_aug["label"].loc[4,])
print(df_train_aug["data"].loc[4,]) # --> www.paypal.com ("paypal")


"""
Word Count
"""
# df_train_aug['word_count'] = df_train_aug['data'].map(lambda calc: len(calc))
# sns.histplot(data=df_train_aug, x='word_count', hue='label')


"""
Space count
"""
def count_spaces_and_newlines(text):
    text = str(text)
    space_count = text.count(" ")
    newline_count = text.count("\n")
    return pd.Series({"space_count": space_count, "newline_count": newline_count})

# 適用例
df_train_aug[["space_count", "newline_count"]] = df_train_aug["data"].apply(count_spaces_and_newlines)

import seaborn as sns
import matplotlib.pyplot as plt

# 各ラベルごとの平均を表示
plot_data = df_train_aug.groupby("label")[["space_count", "newline_count"]].mean().reset_index()
print(plot_data)
sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))

sns.barplot(data=plot_data, x="label", y="space_count", color="teal", label="Spaces")
sns.barplot(data=plot_data, x="label", y="newline_count", color="orange", label="Newlines", alpha=0.7)

plt.title("Average Space and Newline Counts per Label", fontsize=14)
plt.xlabel("Label Category", fontsize=12)
plt.ylabel("Average Count", fontsize=12)
plt.legend()

plt.show()


import urllib.request
import re
from urllib.parse import urlparse
import emoji

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



# df_train_aug["data"] = df_train_aug['data'].apply(replace_urls_with_features)
# print(df_train_aug["data"].loc[4,])


"""
clean text 
"""

def clean_text(text):
    text = replace_urls_with_features(text) # extract semantics from URL
    text = emoji.replace_emoji(text, replace="")  # remove emoji
    text = re.sub(r'\s+', ' ', text).strip() # remove unnecessary space
    
    return text


df_train_aug["data"] = df_train_aug['data'].apply(clean_text)
print(df_train_aug["data"].loc[4,])


def uppercase_ratio(text):
    text = str(text)
    if len(text) == 0:
        return 0
    uppercase_count = sum(1 for c in text if c.isupper())
    total_alpha_count = sum(1 for c in text if c.isalpha())
    if total_alpha_count == 0:
        return 0
    return uppercase_count / total_alpha_count

# 各行の大文字率を計算
df_train_aug["uppercase_ratio"] = df_train_aug["data"].apply(uppercase_ratio)

# 70%以上かどうかのフラグ列
threshold = 0.7
df_train_aug["upper_70"] = df_train_aug["uppercase_ratio"] >= threshold

# ラベルごとの件数集計
label_summary = (
    df_train_aug.groupby("label")["upper_70"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "count_over_70", "count": "total"})
)
label_summary["ratio"] = label_summary["count_over_70"] / label_summary["total"]

print("Labelごとの大文字率 >= 70% の件数と割合")
print(label_summary)


df_train_aug['data'] = df_train_aug['data'].str.lower()


# from transformers import AutoModelForCausalLM, AutoTokenizer

# LLM_MODEL_PATH = "/kaggle/input/qwen2.5/transformers/1.5b-instruct/1"

# model = AutoModelForCausalLM.from_pretrained(
#     LLM_MODEL_PATH,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)


# def build_prompt(url):

#     prompt = (
#         f"Please list 3 to 5 keywords that are likely related to the content of this URL: {url}. "
#         f"Answer the keywords only, separated by commas."
#     )
    
#     messages = [
#         {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#         {"role": "user", "content": prompt}
#     ]
    
#     # Qwen-Chat
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )

#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=100,
#         do_sample=True,
#         temperature=0.2,
#         top_p=0.9
#     )
    
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]
    
#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

#     keywords = re.findall(r"[A-Za-z0-9\-]+", response)
#     return [kw.lower() for kw in keywords if kw.lower() not in ["keywords"]]
    

# def replace_url_to_keywords(text):
#     # Retrieve urls（http/https/www.）
#     urls = re.findall(r'https?://\S+|www\.\S+', text)
#     # print(urls)
#     for url in urls:
#         try:
#             keywords = build_prompt(url)
#             keyword_str = ", ".join(keywords)
#             replacement = f"<URL: {keyword_str}>"
#             text = text.replace(url, replacement)
        
#         except Exception as e:
#             print(f"Error processing {url}: {e}")
#             text = text.replace(url, "<URL: unknown>")
    
#     return text


# # Test
# url = "https://onlyfans.com/user123"
# print(build_prompt(url))


# df_train_aug["data"] = df_train_aug["data"].apply(replace_url_to_keywords)


print(df_train_aug.shape)
df_train_aug = df_train_aug.sample(frac=1).reset_index(drop=True)
print(df_train_aug.shape)

size = len(df_train_aug)
i1 = size // 3
i2 = (size * 2) // 3

print(i1,i2)


df_shuf = df_train_aug.sample(frac=1, random_state=SEED).reset_index(drop=True)

df_train_aug_1 = df_shuf.iloc[0:i1].copy()
df_train_aug_2 = df_shuf.iloc[i1:i2].copy()
df_train_aug_3 = df_shuf.iloc[i2:size].copy()


y = df_train_aug_1["label"]
X = df_train_aug_1["data"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)


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


# from transformers import AutoTokenizer, RobertaForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
# model = RobertaForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=1)
# model.save_pretrained("my-roberta-base-cls")
# tokenizer.save_pretrained("my-roberta-base-cls")
# batch_size = 32


MODEL_DIR = "/kaggle/input/my-base-bert/my-bert-cls"

tokenizer = BertTokenizer.from_pretrained(
    MODEL_DIR,
    do_lower_case=True,
    local_files_only=True
)

config = BertConfig.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = BertForSequenceClassification.from_pretrained(
    MODEL_DIR,
    config=config,
    local_files_only=True
)
batch_size = 32


# """
# Token length distribution
# """
# lengths = [len(tokenizer.encode(t, add_special_tokens=True)) for t in df_train_aug["data"]]
# print("p50:", int(np.percentile(lengths, 50)),
#       "p90:", int(np.percentile(lengths, 90)),
#       "max:", max(lengths))

# plt.hist(lengths, bins=50)
# plt.title("Token Length Distribution")
# plt.xlabel("tokens")
# plt.ylabel("freq")
# plt.show()


# from transformers import (
#     AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
#     DataCollatorWithPadding
# )

# MODEL_DIR = "/kaggle/input/roberta-base"

# # Tokenizer
# tokenizer_2 = AutoTokenizer.from_pretrained(
#     MODEL_DIR,
#     use_fast=True,
#     local_files_only=True
# )  

# # Config
# config_2 = AutoConfig.from_pretrained(
#     MODEL_DIR,
#     num_labels=1,  # 1 --> regression (MSE Loss), 2--> binary classification (Cross-entropy)
#     problem_type="single_label_classification",
#     local_files_only=True,
#     hidden_dropout_prob=0.2,
#     attention_probs_dropout_prob=0.2,
# )

# # Model
# model_2 = AutoModelForSequenceClassification.from_pretrained(
#     MODEL_DIR,
#     config=config_2,
#     local_files_only=True
# )

# # For dynamic padding in your DataLoader/Trainer:
# # data_collator = DataCollatorWithPadding(
# #     tokenizer=tokenizer,
# #     pad_to_multiple_of=8  # helpful for fp16, optional
# # )


# from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# QWEN_MODEL_DIR = "/kaggle/input/qwen2.5/transformers/0.5b/1"

# # load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(
#     QWEN_MODEL_DIR, 
#     padding=True,
#     truncation=True,
#     trust_remote_code=True)
    
# model = AutoModelForSequenceClassification.from_pretrained(
#     QWEN_MODEL_DIR,
#     num_labels=1,
#     trust_remote_code=True
# )


# batch_size = 8


# # prepare the model input
# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# # conduct text completion
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=32768
# )
# output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# # parsing thinking content
# try:
#     # rindex finding 151668 (</think>)
#     index = len(output_ids) - output_ids[::-1].index(151668)
# except ValueError:
#     index = 0

# thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
# content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# print("thinking content:", thinking_content)
# print("content:", content)


# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)


# Tokenize and Encode the comments and labels for the training set
def embedding_text(tokenizer, X_train, y_train, token_max_length=token_max_length):
    input_ids, attention_masks, labels = tokenize_and_encode(
        tokenizer,
        X_train,
        y_train.values,
        token_max_length,
    )

    return input_ids, attention_masks, labels

input_ids, attention_masks, labels = embedding_text(tokenizer, X_train, y_train)

# Tokenize and Encode the comments and labels for the validation set
val_input_ids, val_attention_masks, val_labels = tokenize_and_encode(
    tokenizer,
    X_val,
    y_val.values,
    token_max_length,
)

print('Training Comments :',X_train.shape)
print('Input Ids         :',input_ids.shape)
print('Attention Mask    :',attention_masks.shape)
print('Labels            :',labels.shape)


# Creating DataLoader for the balanced dataset
train_dataset = TensorDataset(input_ids, attention_masks, labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# validation set 
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# def plot_traing(train_losses, val_losses, train_f1s, val_f1s):

#     plt.figure(figsize=(12, 5))

#     # Training Loss & Validation Loss
#     plt.subplot(1, 2, 1)
#     plt.plot(train_losses, label='Train Loss')
#     plt.plot(val_losses, label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Loss over Epochs')
#     plt.legend()

#     # Training Accuracy & Validation F1 Score
#     plt.subplot(1, 2, 2)
#     plt.plot(train_f1s, label='Train Accuracy')
#     plt.plot(val_f1s, label='Val F1 Score')
#     plt.xlabel('Epoch')
#     plt.ylabel('Score')
#     plt.title('Accuracy & F1 Score over Epochs')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()


import copy
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score

def train_model(model, train_loader, val_loader, device, num_epochs, patience=5):
    # loss_fn = nn.BCELoss()  # binary cross entropy
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

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

    # visualize training
    # plot_traing(train_losses, val_losses, train_f1s, val_f1s)
    
    return model


# Call the function to train the model
model = train_model(model, train_loader, val_loader, device, num_epochs=5, patience=2)


# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model_2 = model_2.to(device)


# model_2 = train_model(model_2, train_loader_2, val_loader_2, device, num_epochs=5, patience=2)


def add_rule_and_subreddit(df):

    new_df = pd.DataFrame()
    
    new_df["data"] = "Rule: " + df["rule"] + \
              " Subreddit: " + df["subreddit"] + \
              " Comment: " + df['body']
    
    new_df["row_id"] = df["row_id"]

    return new_df


# # Align with train data 
df_test = add_rule_and_subreddit(df_test)
df_test[["space_count", "newline_count"]] = df_test["data"].apply(count_spaces_and_newlines)
df_test["uppercase_ratio"] = df_test["data"].apply(uppercase_ratio)
threshold = 0.7
df_test["upper_70"] = df_test["uppercase_ratio"] >= threshold
df_test["data"] = df_test['data'].apply(clean_text)


df_test


# import torch
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# # 学習済みモデルをロード
# model_path = "model"  # trainer.save_model("model") で保存したフォルダ
# model = AutoModelForSequenceClassification.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/distilbertbaseuncased")

# # 推論用テキスト（例）
# # texts = [
# #     "This content violates the rule.",
# #     "This comment is perfectly fine."
# # ]

# # トークナイズ
# inputs = tokenizer(
#     df_test["data"],
#     padding=True,
#     truncation=True,
#     return_tensors="pt"
# )

# # 推論モード
# model.eval()
# with torch.no_grad():
#     outputs = model(**inputs)
#     logits = outputs.logits  # shape: (batch_size, num_labels)
#     probs = torch.softmax(logits, dim=1)[:, 1]  # クラス1（違反）の確率を抽出

# # numpy配列に変換
# res = probs.cpu().numpy()

# # 出力表示
# for text, p in zip(texts, res):
#     print(f"Text: {text}")
#     print(f"Probability of rule violation: {p:.4f}")



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


test_input_ids, test_attention_masks = tokenize_test(tokenizer, df_test["data"], max_length=token_max_length)
test_dataset = TensorDataset(test_input_ids, test_attention_masks)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)


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


res = predict(model, test_loader, device)


res = predict(model, test_loader, device)
submission = pd.DataFrame(df_test["row_id"])
submission["rule_violation"] = res


submission.sort_values(by=['rule_violation'])


# space_norm = df_test["space_count"] / df_test["space_count"].max()
# upper_flag = df_test["upper_70"].astype(float)
# syntax_weight = 1 + 0.05 * space_norm + 0.1 * upper_flag
# submission["rule_violation"] = submission["rule_violation"] * syntax_weight
# submission["rule_violation"] /= submission["rule_violation"].max()


submission.sort_values(by=['rule_violation'])


submission.to_csv('submission.csv', index=False)

