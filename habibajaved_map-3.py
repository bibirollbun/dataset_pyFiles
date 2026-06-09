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


import pandas as pd



df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
df.info()


df.shape


df = df.drop(['row_id'], axis=1)





df.shape


# Check for duplicate rows
duplicate_rows = df[df.duplicated()]
print(f"Number of duplicate rows: {duplicate_rows.shape[0]}")


# Remove duplicate rows
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape}")


df["Category"].unique()


print("\nMisconception Column Value Counts:")
print(df['Misconception'].value_counts(dropna=False))


df.head()


df['Misconception'] = df['Misconception'].fillna('NA')

# Print value counts
print("\nMisconception Column Value Counts:")
print(df['Misconception'].value_counts(dropna=False))


# Step 2: Combine question, selected answer, and explanation into one text feature
df['text_input'] = df['QuestionText'] + ' ' + df['MC_Answer'] + ' ' + df['StudentExplanation']

# Step 3: Create target label combining category and misconception
df['label'] = df['Category'] + ':' + df['Misconception']

# Step 4: Flag rows where the category indicates the answer is 'True'
is_true_answer = df['Category'].str.startswith('True')
true_responses = df[is_true_answer].copy()

# Step 5: Count how often each answer appears per question
true_responses['answer_count'] = true_responses.groupby(['QuestionId', 'MC_Answer'])['MC_Answer'].transform('count')

# Step 6: Sort and keep the most common correct answer per question
true_responses = true_responses.sort_values('answer_count', ascending=False)
true_responses = true_responses.drop_duplicates(subset=['QuestionId'])

# Step 7: Prepare correct answer mapping
true_responses = true_responses[['QuestionId', 'MC_Answer']]
true_responses['correct_flag'] = 1

# Step 8: Merge the correct flag into the original DataFrame
df = df.merge(true_responses, on=['QuestionId', 'MC_Answer'], how='left')

# Step 9: Fill missing correct_flag values with 0 (incorrect)
df['correct_flag'] = df['correct_flag'].fillna(0).astype(int)

# ðŸŽ¯ Final useful columns:
# - df['text_input']  â†’ for model input
# - df['label']       â†’ for target classification
# - df['correct_flag'] â†’ binary flag (optional target)



df['features'] = df['QuestionText'] + ' ' + df['MC_Answer'] + ' ' + df['StudentExplanation']


df.info()


df["features"][0]


df["QuestionText"][0]


print(df['label'].value_counts())


df['label'].unique()


import matplotlib.pyplot as plt

df['label'].value_counts().plot(kind='bar', figsize=(16, 4), title='Label distribution')



# Threshold for rare labels
threshold = 5

# Count occurrences
label_counts = df['label'].value_counts()

# Get labels to keep and map rare ones to 'Other'
labels_to_keep = label_counts[label_counts >= threshold].index
df['label'] = df['label'].apply(lambda x: x if x in labels_to_keep else 'Other')




print(labels_to_keep)


df['label'].unique()


import matplotlib.pyplot as plt

df['label'].value_counts().plot(kind='bar', figsize=(16, 4), title='Label distribution')



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])



df.head()


from sklearn.model_selection import train_test_split
X = df['features']
y = df['label_encoded']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)




from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(texts):
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )


train_encodings = tokenize_function(X_train.tolist())
val_encodings = tokenize_function(X_val.tolist())


import torch

class MCQDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(le.classes_)
)


from torch.utils.data import DataLoader

train_dataset = MCQDataset(train_encodings, y_train.tolist())
val_dataset = MCQDataset(val_encodings, y_val.tolist())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)


from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)



import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"\nEpoch {epoch + 1}/{EPOCHS} - Training loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {acc:.4f}")



# Save the model
torch.save(model.state_dict(), 'map@3.pth')
print("Model saved successfully.")


import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# âœ… Load test data
test_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
test_df['features'] = test_df['QuestionText'] + ' ' + test_df['MC_Answer'] + ' ' + test_df['StudentExplanation']

# âœ… Tokenize
test_encodings = tokenizer(
    test_df['features'].tolist(),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors='pt'
)

# âœ… Test dataset
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

test_dataset = TestDataset(test_encodings)
test_loader = DataLoader(test_dataset, batch_size=16)

# âœ… Predict Top-3
model.eval()
top3_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting top-3"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits

        # Top 3 predictions
        topk = torch.topk(logits, k=3, dim=1)
        top_indices = topk.indices.cpu().numpy()

        for row in top_indices:
            labels = le.inverse_transform(row)
            top3_preds.append(" ".join(labels))  # space-separated

# âœ… Final submission
submission = pd.DataFrame({
    "row_id": test_df["row_id"],
    "Category:Misconception": top3_preds
})

submission.to_csv("submission.csv", index=False)
print("âœ… MAP@3 submission file saved as submission.csv")



import pandas as pd

submission_df = pd.read_csv('submission.csv')
submission_df.head()  # show first 5 rows

