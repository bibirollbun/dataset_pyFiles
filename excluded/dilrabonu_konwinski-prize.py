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


import zipfile
import os

zip_path = "/kaggle/input/konwinski-prize/data.a_zip"

extract_path = "/kaggle/working/konwinski_data"

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)
print("Extraction complete!")


extracted_files = os.listdir(extract_path)
print('Extracted files:', extracted_files)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



import zipfile

zip_path = "/kaggle/input/konwinski-prize/data.a_zip"  # Update with actual path
extract_path = "/kaggle/working/konwinski_data/"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Files after extraction:", os.listdir(extract_path))



extract_path = "/kaggle/working/konwinski_data/data"
parquet_path = os.path.join(extract_path, "data.parquet")

df = pd.read_parquet(parquet_path)

print("Dataset Loaded Successfully!")
print(df.head())


df.shape


df.columns


df.info()


df.isnull().sum()


df.describe(include="all")


!pip install nltk transformers datasets

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re

nltk.download('punkt')
nltk.download('stopwords')


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

df['cleaned_problem_statement'] = df['problem_statement'].apply(clean_text)

print("Sample cleaned text:")
print(df['cleaned_problem_statement'].head())


from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

df['problem_embedding'] = df['cleaned_problem_statement'].apply(get_embedding)
print("Embedding generated successfully!")


from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)



print(f"Training Set: {len(train_data)} samples")
print(f'Testing Set: {len(test_data)} samples')


from transformers import T5Tokenizer, T5ForConditionalGeneration

t5_model_name = "t5-small"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)


from torch.utils.data import DataLoader, Dataset

class PatchDataset(Dataset):
    def __init__(self, df):
        self.texts = df["cleaned_problem_statement"].tolist()
        self.labels = df["patch"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

train_dataset = PatchDataset(train_data)
test_dataset = PatchDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


optimizer = torch.optim.Adam(t5_model.parameters(), lr=1e-4)
t5_model.train()

for epoch in range(5):
    for problem, patch in train_loader:
        inputs = t5_tokenizer(problem, return_tensors="pt", padding=True, truncation=True)
        labels = t5_tokenizer(patch, return_tensors="pt", padding=True, truncation=True).input_ids
        optimizer.zero_grad()
        outputs = t5_model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}: Loss = {loss.item()}')


t5_model.eval()

def generate_patch(problem_text):
    inputs = t5_tokenizer(problem_text, return_tensors="pt", padding=True, truncation=True)
    outputs = t5_model.generate(**inputs)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)


sample_problem = test_data.iloc[0]['cleaned_problem_statement']
generated_patch = generate_patch(sample_problem)

print("Porblem Statement:", sample_problem)
print("Generated Patch:", generated_patch)



import torch

# Define the save path
save_path = "/kaggle/working/t5_model"

# Save the trained model and tokenizer
t5_model.save_pretrained(save_path)
t5_tokenizer.save_pretrained(save_path)

print("Model saved successfully!")





predictions = []
for problem in test_data["cleaned_problem_statement"]:
    generated_patch = generate_patch(problem)
    predictions.append(generated_patch)

submission_df = pd.DataFrame({
    "id": test_data.index,  
    "patch": predictions
})


submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Submission file saved successfully:", submission_path)





