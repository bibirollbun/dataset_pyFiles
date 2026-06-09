!pip install transformers peft accelerate \
    -U --no-index --find-links /kaggle/input/lmsys-wheel-files


%%capture
!pip install --no-index /kaggle/input/bitsandbytes0-42-0/bitsandbytes-0.42.0-py3-none-any.whl --find-links=/kaggle/input/bitsandbytes0-42-0
# !pip install --no-index  /kaggle/input/bitsandbytes0-42-0/optimum-1.21.2-py3-none-any.whl --find-links=/kaggle/input/bitsandbytes0-42-0
# !pip install --no-index  /kaggle/input/bitsandbytes0-42-0/auto_gptq-0.7.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --find-links=/kaggle/input/bitsandbytes0-42-0


from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import gc
import pandas as pd
import pickle
import sys
import numpy as np
from tqdm.autonotebook import trange
from sklearn.model_selection import GroupKFold
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from numpy.linalg import norm
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel,BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
)
import json
import copy
import warnings
import os
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv")
test = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv")
sub = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/sample_submission.csv")
misconception_mapping  = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")


train.head()


train.columns


# Check for missing values in the train dataset
missing_values = train[['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']].isnull().sum()

print("Missing values per Misconception column:")
print(missing_values)


# Fill missing misconception values with -1 (indicating no misconception)
train[['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']] = train[['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']].fillna(-1)

# Verify that there are no more missing values
missing_values_after = train[['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']].isnull().sum()

print("Missing values after filling:")
print(missing_values_after)


# Combine all misconception IDs into one series
misconceptions = pd.concat([
    train['MisconceptionAId'],
    train['MisconceptionBId'],
    train['MisconceptionCId'],
    train['MisconceptionDId']
], ignore_index=True)

# Exclude the -1 values (no misconception)
misconceptions = misconceptions[misconceptions != -1]

print(f"Total misconceptions (excluding -1): {len(misconceptions)}")


# Count the occurrences of each misconception ID
misconception_counts = misconceptions.value_counts()

# Plot the top 20 most common misconceptions
plt.figure(figsize=(12,6))
sns.barplot(
    x=misconception_counts.head(20).index.astype(int),
    y=misconception_counts.head(20).values
)
plt.title('Top 20 Misconceptions')
plt.xlabel('Misconception ID')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()


# Merge misconception_counts with misconception_mapping
misconception_counts_df = misconception_counts.reset_index()
misconception_counts_df.columns = ['MisconceptionId', 'Count']
misconception_counts_df = misconception_counts_df.merge(
    misconception_mapping, on='MisconceptionId', how='left'
)

# Display top misconceptions with names
print(misconception_counts_df.head(10))


# Add columns for text lengths
train['QuestionLength'] = train['QuestionText'].apply(lambda x: len(str(x).split()))
train['AnswerALength'] = train['AnswerAText'].apply(lambda x: len(str(x).split()))
train['AnswerBLength'] = train['AnswerBText'].apply(lambda x: len(str(x).split()))
train['AnswerCLength'] = train['AnswerCText'].apply(lambda x: len(str(x).split()))
train['AnswerDLength'] = train['AnswerDText'].apply(lambda x: len(str(x).split()))

# Plot distribution of question lengths
plt.figure(figsize=(10,6))
sns.histplot(train['QuestionLength'], bins=30)
plt.title('Distribution of Question Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()


# Plot distribution of answer lengths
answer_lengths = ['AnswerALength', 'AnswerBLength', 'AnswerCLength', 'AnswerDLength']
for col in answer_lengths:
    plt.figure(figsize=(10,6))
    sns.histplot(data=train, x=col, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()


print("Missing values in 'QuestionText':", train['QuestionText'].isnull().sum())
print("Missing values in 'AnswerAText':", train['AnswerAText'].isnull().sum())
print("Missing values in 'AnswerBText':", train['AnswerBText'].isnull().sum())
print("Missing values in 'AnswerCText':", train['AnswerCText'].isnull().sum())
print("Missing values in 'AnswerDText':", train['AnswerDText'].isnull().sum())


path_prefix = "/kaggle/input/eedi-mining-misconceptions-in-mathematics"
model_path = "/kaggle/input/qwen2.5-14/pytorch/default/1"
lora_path='/kaggle/input/qwen14b-it-lora/lora_weights/adapter.bin'
device='cuda:0'
VALID = False


def apk(actual, predicted, k=25):
    """
    Tính Average Precision at K (AP@K) giữa hai danh sách:
        actual: Danh sách phần tử đúng (đáp án thực sự).
        predicted: Danh sách phần tử dự đoán.
    Tham số:
        actual: Danh sách chứa các phần tử đúng (không cần thứ tự).
        predicted: Danh sách phần tử được dự đoán (thứ tự quan trọng).
        k: Số lượng phần tử tối đa trong danh sách dự đoán cần xem xét.
    """
    
    if not actual:
        return 0.0

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # kiểm tra xem nó có phải là dự đoán hợp lệ
        # kiểm tra xem dự đoán có được lặp lại không
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)

def mapk(actual, predicted, k=25):
    """
    Tính Mean Average Precision at K (MAP@K) trên nhiều danh sách:
        actual: Danh sách các danh sách phần tử đúng.
        predicted: Danh sách các danh sách phần tử dự đoán.
    Tham số:
        actual: Danh sách các danh sách đáp án đúng.
        predicted: Danh sách các danh sách dự đoán.
        k: Số lượng phần tử tối đa cần xem xét.
    """
    # Tính trung bình của tất cả các giá trị AP@K để trả về
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def batch_to_device(batch, target_device):
    """
    Di chuyển một batch dữ liệu PyTorch tới thiết bị cụ thể (CPU/GPU).
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    """
        Trích xuất vector biểu diễn cuối cùng từ trạng thái ẩn của mô hình Transformer.
        Trả về tensor biểu diễn vector của token cuối cùng.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    """
        Tạo chuỗi hướng dẫn chi tiết từ một nhiệm vụ và truy vấn.
    """
    return f'Instruct: {task_description}\nQuery: {query}'

def inference(df, model, tokenizer, device):
    """
        Thực hiện suy luận trên dữ liệu và trả về vector nhúng cho từng truy vấn.
        Trả về một dictionary với pids làm key và embedding làm giá trị.
    """
    batch_size = 16
    max_length = 512
    sentences = list(df['query_text'].values) # danh sách câu truy vấn 
    pids = list(df['order_index'].values)
    all_embeddings = []
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=False):
        sentences_batch = sentences_sorted[start_index: start_index + batch_size]
        features = tokenizer(sentences_batch, max_length=max_length, padding=True, truncation=True,
                             return_tensors="pt")
        features = batch_to_device(features, device)
        with torch.no_grad():
            outputs = model(**features)
            embeddings = last_token_pool(outputs.last_hidden_state, features['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = embeddings.detach().cpu().numpy().tolist()
        all_embeddings.extend(embeddings)

    all_embeddings = [np.array(all_embeddings[idx]).reshape(1, -1) for idx in np.argsort(length_sorted_idx)]

    sentence_embeddings = np.concatenate(all_embeddings, axis=0)
    result = {pids[i]: em for i, em in enumerate(sentence_embeddings)}
    return result


task_description = 'Given a math question with correct answer and a misconcepted incorrect answer, retrieve the most accurate misconception for the incorrect answer.'


if VALID:
    tra = pd.read_parquet("/kaggle/input/val-parquet/v1_val.parquet")
    print(tra.shape)
else:
    tra = pd.read_csv(f"{path_prefix}/test.csv")
    print(tra.shape)
misconception_mapping = pd.read_csv(f"{path_prefix}/misconception_mapping.csv")
if tra.shape[0]<10:
    misconception_mapping = misconception_mapping.sample(n=5,random_state=2023)


if VALID:
    train_data = []
    for _,row in tra.iterrows():
        for c in ['A','B','C','D']:
            if str(row[f"Misconception{c}Id"])!="nan":
                real_answer_id = row['CorrectAnswer']
                real_text = row[f'Answer{real_answer_id}Text']
                query_text = f"### SubjectName: {row['SubjectName']}\n### ConstructName: {row['ConstructName']}\n### Question: {row['QuestionText']}\n### Correct Answer: {real_text}\n### Misconcepte Incorrect answer: {row[f'Answer{c}Text']}"
                row['query_text'] = get_detailed_instruct(task_description,query_text)
                row['answer_id'] = int(row[f"Misconception{c}Id"])
                train_data.append(copy.deepcopy(row))
    train_df = pd.DataFrame(train_data)
    train_df['order_index'] = list(range(len(train_df)))
else:
    train_data = []
    for _,row in tra.iterrows():
        for c in ['A','B','C','D']:
            if c ==row['CorrectAnswer']:
                continue
            if f'Answer{c}Text' not in row:
                continue
            real_answer_id = row['CorrectAnswer']
            real_text = row[f'Answer{real_answer_id}Text']
            query_text = f"### SubjectName: {row['SubjectName']}\n### ConstructName: {row['ConstructName']}\n### Question: {row['QuestionText']}\n### Correct Answer: {real_text}\n### Misconcepte Incorrect answer: {row[f'Answer{c}Text']}"
            row['query_text'] = get_detailed_instruct(task_description,query_text)
            row['answer_name'] = c
            train_data.append(copy.deepcopy(row))
    train_df = pd.DataFrame(train_data)
    train_df['order_index'] = list(range(len(train_df)))
train_df.shape


# Khởi tạo AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(lora_path.replace("/adapter.bin",""))

bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
# Tải mô hình 
model = AutoModel.from_pretrained(model_path, 
                                  quantization_config=bnb_config, 
                                  device_map=device,
                                  trust_remote_code=True)

# Tạo cấu hình LoRA
if lora_path:
    print("loading lora")
    config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(model, config) 
    d = torch.load(lora_path, map_location=model.device) # Tải trọng số LoRA
    model.load_state_dict(d, strict=False)
    model = model.merge_and_unload()
    
# Chuyển mô hình về chế độ suy luận
model = model.eval()



train_embeddings = inference(train_df, model, tokenizer, device)


misconception_mapping['query_text'] = misconception_mapping['MisconceptionName']
misconception_mapping['order_index'] = misconception_mapping['MisconceptionId']
doc_embeddings = inference(misconception_mapping, model, tokenizer, device)


sentence_embeddings = np.concatenate([e.reshape(1, -1) for e in list(doc_embeddings.values())])
index_text_embeddings_index = {index: paper_id for index, paper_id in
                                         enumerate(list(doc_embeddings.keys()))}


predicts_test = []
for _, row in tqdm(train_df.iterrows()):
    query_id = row['order_index']
    query_em = train_embeddings[query_id].reshape(1, -1)
    
    cosine_similarity = np.dot(query_em, sentence_embeddings.T).flatten()
    
    sort_index = np.argsort(-cosine_similarity)[:25]
    pids = [index_text_embeddings_index[index] for index in sort_index]
    predicts_test.append(pids)


if VALID:
    train_df['recall_ids'] = predicts_test
    print(mapk([[data] for data in train_df['answer_id'].values],train_df['recall_ids'].values))
else:
    train_df['MisconceptionId'] = [' '.join(map(str,c)) for c in predicts_test]
    sub = []
    for _,row in train_df.iterrows():
        sub.append(
            {
                "QuestionId_Answer":f"{row['QuestionId']}_{row['answer_name']}",
                "MisconceptionId":row['MisconceptionId']
            }
        )
    submission_df = pd.DataFrame(sub)
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file created successfully!")

