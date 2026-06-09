# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install transformers pandas


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("/kaggle/input/drawing-with-llms/train.csv")


train_df.head()


test_df = pd.read_csv("/kaggle/input/drawing-with-llms/kaggle_evaluation/test.csv")


test_df.head()


# Example: Convert everything to lowercase (optional)
train_df["description"] = train_df["description"].str.lower()
test_df["description"] = test_df["description"].str.lower()


train_df["description"]


from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load T5 Model and Tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)



from torch.utils.data import Dataset, DataLoader
import torch

class DrawingDataset(Dataset):
    def __init__(self, texts, svgs, tokenizer, max_length=512):
        self.texts = texts
        self.svgs = svgs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        svg = self.svgs[idx]

        input_encoding = self.tokenizer(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            svg, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }

# Create Dataset and DataLoader
train_dataset = DrawingDataset(train_df["description"].tolist(), train_df["id"].tolist(), tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)



from transformers import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 3  # Change this for longer training
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} Loss: {loss.item()}")



# Function to generate SVGs from text
def generate_svg(text):
    input_encoding = tokenizer(
        text, max_length=512, padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)

    output_ids = model.generate(
        input_ids=input_encoding["input_ids"],
        attention_mask=input_encoding["attention_mask"],
        max_length=512
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Generate predictions
test_df["id"] = test_df["description"].apply(generate_svg)

# Save results to CSV
test_df[["id", "description"]].to_csv("submission.csv", index=False)


