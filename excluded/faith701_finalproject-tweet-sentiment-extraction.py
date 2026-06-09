import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import pandas as pd

# Load the datasets

train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
submission_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

# Display basic information and the first few rows of the training dataset
train_info = train_df.info()
train_head = train_df.head()

train_info, train_head



# Below is a helper Function which generates random colors which can be used to give different colors to your plots.
import re
import string
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

import random

def random_colours(number_of_colors):
    '''
    Simple function for random colours generation.
    Input:
        number_of_colors - integer value indicating the number of colours which are going to be generated.
    Output:
        List of colors in the following format: ['#E86DA4'].
    '''
    colors = []
    for i in range(number_of_colors):
        colors.append("#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]))
    return colors

# Test the function by generating 5 random colors
random_colours(5)



temp = train_df.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)
temp.style.background_gradient(cmap='Purples')


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Drop rows with missing values
train_df.dropna(subset=["text", "selected_text"], inplace=True)

# Add text and selected_text length columns
train_df["text_len"] = train_df["text"].apply(len)
train_df["selected_text_len"] = train_df["selected_text"].apply(len)

# Plot sentiment distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x="sentiment", palette="pastel")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Plot distribution of text and selected_text lengths by sentiment
plt.figure(figsize=(12, 5))
sns.histplot(data=train_df, x="text_len", hue="sentiment", bins=40, kde=True, palette="pastel")
plt.title("Tweet Length Distribution by Sentiment")
plt.xlabel("Tweet Length")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(12, 5))
sns.histplot(data=train_df, x="selected_text_len", hue="sentiment", bins=40, kde=True, palette="muted")
plt.title("Selected Text Length Distribution by Sentiment")
plt.xlabel("Selected Text Length")
plt.ylabel("Frequency")
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import random
import re

# Random color generator
def random_colours(number_of_colors):
    return ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
            for _ in range(number_of_colors)]

# Sample: Load data and filter positive tweets
train_df = train_df.dropna(subset=['text', 'selected_text'])
positive_tweets = train_df[train_df['sentiment'] == 'positive']['selected_text']

# Preprocessing: Remove punctuation and tokenize
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower()).split()

words = []
for tweet in positive_tweets:
    words.extend(clean_text(tweet))

# Count most common words
common_words = Counter(words).most_common(10)
words, freqs = zip(*common_words)

# Plot with random colors
colors = random_colours(len(words))

plt.figure(figsize=(10, 6))
plt.bar(words, freqs, color=colors)
plt.title("Top 10 Words in Positive Tweets")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import re

# Extract emojis, hashtags, and mentions using regex
emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
hashtag_pattern = re.compile(r"#\w+")
mention_pattern = re.compile(r"@\w+")

def extract_multimodal_features(text):
    emojis = emoji_pattern.findall(text)
    hashtags = hashtag_pattern.findall(text)
    mentions = mention_pattern.findall(text)
    return pd.Series({
        "emojis": emojis,
        "hashtags": hashtags,
        "mentions": mentions,
        "emoji_count": len(emojis),
        "hashtag_count": len(hashtags),
        "mention_count": len(mentions)
    })

# Apply the feature extraction to the training data
multimodal_features = train_df["text"].apply(extract_multimodal_features)

# Combine features with original dataframe
train_df = pd.concat([train_df.reset_index(drop=True), multimodal_features.reset_index(drop=True)], axis=1)

# Preview enhanced dataframe
train_df[["text", "emojis", "hashtags", "mentions", "emoji_count", "hashtag_count", "mention_count"]].head()



# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z0-9@#\s" + "\U00010000-\U0010ffff" + "]", "", text)  # Keep emojis, mentions, hashtags
    text = re.sub(r"\s+", " ", text).strip()  # Remove excess whitespace
    return text

# Apply preprocessing to text and selected_text
train_df["clean_text"] = train_df["text"].apply(preprocess_text)
train_df["clean_selected_text"] = train_df["selected_text"].apply(preprocess_text)

# Preview the cleaned data
train_df[["text", "clean_text", "selected_text", "clean_selected_text"]].head()



!pip install transformers --upgrade
from transformers import BertTokenizerFast

# Use BERT's fast tokenizer (pretrained uncased model)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Helper function to tokenize input and map selected text spans
def tokenize_with_labels(row):
    full_text = row['clean_text']
    selected_text = row['clean_selected_text']

    # Tokenize the full tweet
    encoding = tokenizer(full_text, return_offsets_mapping=True, truncation=True, padding="max_length", max_length=128)

    # Locate selected text in full text using character span
    start_idx = full_text.find(selected_text)
    end_idx = start_idx + len(selected_text)

    # Align selected text span to token indices
    token_labels = [0] * len(encoding["offset_mapping"])
    for idx, (start, end) in enumerate(encoding["offset_mapping"]):
        if start >= start_idx and end <= end_idx:
            token_labels[idx] = 1

    return pd.Series({
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "token_type_ids": encoding["token_type_ids"],
        "labels": token_labels
    })

# Apply tokenization and label alignment to a subset for performance
tokenized_subset = train_df.iloc[:1000].apply(tokenize_with_labels, axis=1)

# Convert to dictionary before creating a DataFrame
tokenized_dict = {col: tokenized_subset[col].to_list() for col in tokenized_subset.columns}

# Convert to DataFrame
tokenized_df = pd.DataFrame(tokenized_dict)
tokenized_df.head()


import torch
from torch import nn
from transformers import BertModel

class TweetSentimentExtractor(nn.Module):
    def __init__(self):
        super(TweetSentimentExtractor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # 2 classes: inside or outside the selected text

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        x = self.dropout(sequence_output)
        logits = self.classifier(x)  # (batch_size, seq_len, 2)
        return logits



from torch.utils.data import Dataset, DataLoader

class TweetDataset(Dataset):
    def __init__(self, inputs):
        self.input_ids = list(inputs['input_ids'])
        self.attention_mask = list(inputs['attention_mask'])
        self.token_type_ids = list(inputs['token_type_ids'])
        self.labels = list(inputs['labels'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'token_type_ids': torch.tensor(self.token_type_ids[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Example:
train_dataset = TweetDataset(tokenized_subset)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)



import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

!pip install hf_xet



import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TweetSentimentExtractor().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids)  # shape (batch_size, seq_len, 2)

        # Reshape for loss calculation
        outputs = outputs.view(-1, 2)
        labels = labels.view(-1)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")



from sklearn.metrics import f1_score

def calculate_iou(pred_labels, true_labels):
    pred_set = set([i for i, val in enumerate(pred_labels) if val == 1])
    true_set = set([i for i, val in enumerate(true_labels) if val == 1])

    if not pred_set and not true_set:
        return 1.0  # If both are empty
    if not pred_set or not true_set:
        return 0.0

    intersection = len(pred_set.intersection(true_set))
    union = len(pred_set.union(true_set))
    return intersection / union

def evaluate(model, dataloader):
    model.eval()
    iou_scores = []
    f1_scores = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            predictions = torch.argmax(outputs, dim=-1)

            predictions = predictions.view(-1).cpu().numpy()
            true_labels = labels.view(-1).cpu().numpy()

            f1 = f1_score(true_labels, predictions, average='binary')
            iou = calculate_iou(predictions, true_labels)

            f1_scores.append(f1)
            iou_scores.append(iou)

    print(f"Validation F1 Score: {sum(f1_scores)/len(f1_scores):.4f}")
    print(f"Validation IoU Score: {sum(iou_scores)/len(iou_scores):.4f}")

evaluate(model, train_loader)


# Preprocess test data
test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
test_df["clean_text"] = test_df["text"].apply(preprocess_text)

# Tokenize test set
test_encodings = test_df["clean_text"].apply(lambda x: tokenizer(x, return_offsets_mapping=True,
                                                                 truncation=True, padding="max_length", max_length=128))

class TestDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = [enc['input_ids'] for enc in encodings]
        self.attention_mask = [enc['attention_mask'] for enc in encodings]
        self.token_type_ids = [enc['token_type_ids'] for enc in encodings]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'token_type_ids': torch.tensor(self.token_type_ids[idx], dtype=torch.long)
        }

test_dataset = TestDataset(test_encodings)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Generate predictions
model.eval()
all_selected_texts = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)
        predictions = torch.argmax(outputs, dim=-1)

        for i in range(predictions.shape[0]):
            ids = input_ids[i]
            preds = predictions[i]
            selected_tokens = ids[preds == 1]
            selected_text = tokenizer.decode(selected_tokens, skip_special_tokens=True)
            all_selected_texts.append(selected_text.strip())

# Create submission
submission = pd.DataFrame({
    "textID": test_df["textID"],
    "selected_text": all_selected_texts
})

submission.to_csv("submission.csv", index=False)



# Show a preview of the submission file
print("Submission preview:")
print(submission.head())

# Confirm successful file creation
print("\nâœ… Submission file 'submission.csv' has been created.")


