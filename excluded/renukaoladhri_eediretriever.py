# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


bge_large='/kaggle/input/bge-large-en-v1.5/transformers/default/1'
qwen='/kaggle/input/qwen2.5/transformers/qwen2-math-7b-instruct-awq-4bit-smashed/1'


train_df=pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv')
test_df=pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv')
misconceptions_df=pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv')


train_df.head()


train_df.info()


train_df.shape


misconceptions_df.head()


misconceptions_df.info()



import matplotlib.pyplot as plt
import seaborn as sns
miscon_ids = pd.concat([
    train_df['MisconceptionAId'],
    train_df['MisconceptionBId'],
    train_df['MisconceptionCId'],
    train_df['MisconceptionDId']
], ignore_index=True)

# Drop NA (correct answers might have NA)
miscon_ids = miscon_ids.dropna().astype(int)

# Count frequency
miscon_count = miscon_ids.value_counts().rename_axis('MisconceptionId').reset_index(name='Count')

# Merge with names
miscon_count = miscon_count.merge(misconceptions_df, on='MisconceptionId', how='left')

# Plot top 20 most frequent misconceptions
plt.figure(figsize=(12, 6))
sns.barplot(data=miscon_count.head(20), x='Count', y='MisconceptionName', palette='mako')
plt.title('Top 20 Most Frequent Misconceptions')
plt.xlabel('Frequency')
plt.ylabel('Misconception')
plt.tight_layout()
plt.show()

# Summary statistics
print(f"Total unique misconceptions (used): {miscon_ids.nunique()}")
print(f"Total misconceptions in mapping file: {misconceptions_df['MisconceptionId'].nunique()}")

# Distribution across answer choices
answer_miscon = {
    'A': train_df['MisconceptionAId'].dropna().astype(int),
    'B': train_df['MisconceptionBId'].dropna().astype(int),
    'C': train_df['MisconceptionCId'].dropna().astype(int),
    'D': train_df['MisconceptionDId'].dropna().astype(int),
}

# Convert to dataframe
dist_df = pd.DataFrame({k: v.value_counts() for k, v in answer_miscon.items()}).fillna(0).astype(int)
dist_df['Total'] = dist_df.sum(axis=1)
dist_df = dist_df.sort_values('Total', ascending=False)

# View top misconceptions by answer slot
print("\nTop 10 misconceptions by answer slot:\n")
print(dist_df.head(10))


from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch


# # Example function to create (query, label) pairs
# def generate_query_label_pairs(df, misconceptions_df):
#     rows = []
#     for i, row in df.iterrows():
#         for opt in ['A', 'B', 'C']:
#             miscon_id = row[f'Misconception{opt}Id']
#             if pd.notna(miscon_id):
#                 query = f"{row['ConstructName']} | {row['SubjectName']} | {row['QuestionText']} | {row[f'Answer{opt}Text']}"
#                 miscon_name = misconceptions_df.loc[misconceptions_df['MisconceptionId'] == int(miscon_id), 'MisconceptionName'].values[0]
#                 rows.append((query, miscon_name, int(miscon_id)))
#     return pd.DataFrame(rows, columns=['query', 'misconception_name', 'misconception_id'])

# query_df = generate_query_label_pairs(train_df, misconceptions_df)



def process_train_data(train_df):
    rows = []
    for _, row in train_df.iterrows():
        for option in ['A', 'B', 'C', 'D']:
            distractor_text = row[f'Answer{option}Text']
            misconception_id = row[f'Misconception{option}Id']
            if pd.notnull(misconception_id):
                rows.append({
                    'question_id': row['QuestionId'],
                    'question': row['QuestionText'],
                    'construct': row['ConstructName'],
                    'subject': row['SubjectName'],
                    'distractor': distractor_text,
                    'misconception_id': int(misconception_id)
                })
    return pd.DataFrame(rows)

processed_train = process_train_data(train_df)



from transformers import AutoTokenizer, AutoModel
import torch
bge_large='/kaggle/input/bge-large-en-v1.5/transformers/default/1/bge-large-en-v1.5'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
bge_tokenizer = AutoTokenizer.from_pretrained(bge_large)
bge_model = AutoModel.from_pretrained(bge_large).to(device)



def get_embedding(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]  # [CLS] token
    return embeddings.cpu()

misconception_embeddings = {}
batch_size = 64
for i in range(0, len(misconceptions_df), batch_size):
    batch = misconceptions_df.iloc[i:i+batch_size]
    texts = batch['MisconceptionName'].tolist()
    embs = get_embedding(texts, bge_tokenizer, bge_model)
    for idx, emb in zip(batch['MisconceptionId'], embs):
        misconception_embeddings[idx] = emb



from sklearn.metrics.pairwise import cosine_similarity

def evaluate_affinity(example, misconception_embeddings):
    text = f"{example['construct']} {example['subject']} {example['question']} {example['distractor']}"
    emb = get_embedding([text], bge_tokenizer, bge_model)[0]
    
    scores = {}
    for mid, m_emb in misconception_embeddings.items():
        score = cosine_similarity([emb], [m_emb])[0][0]
        scores[mid] = score
    
    # Return top 25 predictions
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in sorted_scores[:25]]



def map_at_25(preds, labels):
    score = 0.0
    for pred, label in zip(preds, labels):
        if label in pred:
            rank = pred.index(label) + 1
            score += 1.0 / rank
    return score / len(preds)



from tqdm import tqdm

# Evaluate on processed_train
print("Evaluating MAP@25 on training data...")
preds = []
labels = []

for _, row in tqdm(processed_train.iterrows(), total=len(processed_train)):
    top_25 = evaluate_affinity(row, misconception_embeddings)
    preds.append(top_25)
    labels.append(row['misconception_id'])

print(f"Total evaluated samples: {len(preds)}")
print("MAP@25:", map_at_25(preds, labels))



def process_test_data(test_df):
    from tqdm import tqdm
    print("Processing test data...")
    rows = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        for option in ['A', 'B', 'C', 'D']:
            answer_text = row.get(f'Answer{option}Text', None)
            if pd.notnull(answer_text):
                rows.append({
                    'id': f"{row['QuestionId']}_{option}",
                    'construct': row['ConstructName'],
                    'subject': row['SubjectName'],
                    'question': row['QuestionText'],
                    'distractor': answer_text
                })
    df = pd.DataFrame(rows)
    print(f"Processed {len(df)} test distractors.")
    return df
test_processed = process_test_data(test_df)



from tqdm import tqdm

print("Generating predictions on test data...")
submission = []

for _, row in tqdm(test_processed.iterrows(), total=len(test_processed)):
    top_25 = evaluate_affinity(row, misconception_embeddings)
    submission.append({
        'Id': row['id'],  # Format: "QuestionId_AnswerOption"
        'Predicted': ' '.join(map(str, top_25))  # space-separated MisconceptionIds
    })

print(f"Total test predictions: {len(submission)}")



import pandas as pd

# Convert to DataFrame
submission_df = pd.DataFrame(submission)

# Optional: sanity check the output
print(submission_df.head())

# Save to CSV
submission_df.to_csv('submission.csv', index=False)
print("✅ Submission file saved as submission.csv")











