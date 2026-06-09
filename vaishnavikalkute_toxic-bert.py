# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")


df.head()


df.info()


df.describe()


df["toxic"].value_counts()


import re
def clean_comment(text):
    text=text.lower()
    text=re.sub(r'[^a-z0-9\s]','',text)
    text=re.sub(r'[\s+]',' ',text).strip()
    return text


df["md_comt_text"]=[clean_comment(cmt) for cmt in df["comment_text"]]


df.head()



# Count of each toxic label
toxic_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df[toxic_labels].sum().sort_values(ascending=False)



plt.figure(figsize=(10, 6))
df[toxic_labels].sum().sort_values().plot(kind='barh', color='salmon')
plt.title("Number of Comments per Toxic Category")
plt.xlabel("Count")
plt.ylabel("Category")
plt.grid(axis='x', linestyle='--')
plt.show()


# Count number of labels per comment
df['num_labels'] = df[toxic_labels].sum(axis=1)

# Distribution of number of labels
plt.figure(figsize=(8, 4))
sns.countplot(x='num_labels', data=df, palette='magma')
plt.title("Number of Toxic Labels per Comment")
plt.xlabel("Number of Labels")
plt.ylabel("Number of Comments")
plt.grid(axis='y', linestyle='--')
plt.show()


# Add word count
df['word_count'] = df['comment_text'].apply(lambda x: len(str(x).split()))

# Toxic or not (at least one label)
df['is_toxic'] = (df[toxic_labels].sum(axis=1) > 0).astype(int)

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='word_count', hue='is_toxic', bins=50, kde=True, palette=['green', 'red'], log_scale=(False, True))
plt.title("Word Count Distribution: Toxic vs Non-Toxic Comments")
plt.xlabel("Word Count")
plt.ylabel("Frequency (log scale)")
plt.legend(title='Toxic', labels=['Non-Toxic', 'Toxic'])
plt.show()



# from wordcloud import WordCloud

# # WordCloud for toxic comments
# toxic_text = " ".join(df[df['is_toxic']==1]['comment_text'].dropna().tolist())
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(toxic_text)

# plt.figure(figsize=(12, 6))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.title("WordCloud of Toxic Comments")
# plt.show()




# Load the tokenizer for BERT base model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer


text = "You're such a disgusting person."
tokens = tokenizer.tokenize(text)
print(tokens)

# Convert to input IDs
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)


encoded = tokenizer(
    text,
    padding='max_length',
    truncation=True,
    max_length=20,
    return_tensors='pt'  # returns PyTorch tensors
)
print(encoded)
print(encoded['input_ids'])         # shape: (1, 128)
print(encoded['attention_mask'].shape)    # same


def tokenize_data(comments, tokenizer, max_len=128):
    return tokenizer(
        comments,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )


import torch
from torch.utils.data import Dataset

class ToxicCommentDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_len):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        text = str(self.comments[idx])
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),          # (128,)
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)                     # For multi-label classification
        }


from torch.utils.data import DataLoader

MAX_LEN = 128
BATCH_SIZE = 16

# Example
train_dataset = ToxicCommentDataset(
    comments=df['comment_text'].tolist(),
    labels=df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm


model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=6,  # for 6 toxic categories
    problem_type="multi_label_classification"
)

model = model.to("cuda" if torch.cuda.is_available() else "cpu")


optimizer = AdamW(model.parameters(), lr=2e-5)

EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    print(f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Average loss: {total_loss / len(train_loader):.4f}")


# Save model and tokenizer
model.save_pretrained("/kaggle/working/toxic-bert")
tokenizer.save_pretrained("/kaggle/working/toxic-bert")


from sklearn.model_selection import train_test_split

# Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['comment_text'].tolist(),
    df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values,
    test_size=0.1,
    random_state=42
)

# Dataset and DataLoader
val_dataset = ToxicCommentDataset(val_texts, val_labels, tokenizer, max_len=128)
val_loader = DataLoader(val_dataset, batch_size=16)


from sklearn.metrics import f1_score, accuracy_score

val_preds = []
val_labels_list = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()  # shape: (batch, 6)
        val_preds.append(probs)
        val_labels_list.append(labels.cpu().numpy())

# Combine all batches
val_preds = np.vstack(val_preds)
val_labels = np.vstack(val_labels_list)

# Convert probs to 0/1 based on 0.5 threshold
val_preds_binary = (val_preds >= 0.5).astype(int)

# Metrics
f1 = f1_score(val_labels, val_preds_binary, average="macro")
acc = accuracy_score(val_labels, val_preds_binary)

print(f"Validation F1 Score: {f1:.4f}")
print(f"Validation Accuracy: {acc:.4f}")


def predict_comment(text):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        input_ids = encoding['input_ids'].to(model.device)
        attention_mask = encoding['attention_mask'].to(model.device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        return {label: float(prob) for label, prob in zip(
            ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], probs)}

predict_comment("You are a horrible person!")


import shap
import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline

# === Load BERT model and tokenizer ===
model_path = "/kaggle/input/trained-model/toxic-bert/"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# === Create TextClassification pipeline ===
pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1
)

# === SHAP Explainer ===
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(pipe, masker)

# === Toxic Labels ===
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# === Rule-Based Logic (Pattern Matching) ===
TOXIC_RULES = [
    (re.compile(r"\byou (stupid|idiot|dumb|fool)\b"), "insult"),
    (re.compile(r"\bi will (kill|hurt|destroy) you\b"), "threat"),
    (re.compile(r"\bf\*\*k|\bshit|\basshole\b"), "obscene"),
    (re.compile(r"\byou people\b.*\b(all|always|are|should)\b"), "identity_hate"),
    (re.compile(r"\bhate (you|them|everyone)\b"), "toxic"),
]

def rule_based_detect(comment):
    flags = {label: False for label in labels}
    for pattern, label in TOXIC_RULES:
        if pattern.search(comment.lower()):
            flags[label] = True
    return flags

# === Predict with BERT ===
def predict_comment(text):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    return {label: round(float(prob), 4) for label, prob in zip(labels, probs)}

# === Explain a single input ===
def explain_toxic_comment(text_input):
    print("ğŸ§¾ Input:", text_input)
    
    # Predict with BERT
    model_output = predict_comment(text_input)
    print("\nğŸ¤– BERT Model Output:")
    for label, prob in model_output.items():
        print(f"  {label:15} â†’ {prob}")

    # Rule-based logic
    rule_output = rule_based_detect(text_input)
    print("\nğŸ“� Rule-Based Output:")
    for label, present in rule_output.items():
        print(f"  {label:15} â†’ {present}")

    # SHAP Explanation
    print("\nğŸ“Š SHAP Explanations:")
    shap_values = explainer([text_input])
    for i, _ in enumerate(labels):
        print(f"\nğŸ”� Class: {shap_values[0].output_names[i]}")
        shap.plots.text(shap_values[0][:, i])  # Show token-level explanation for that class

# === Try a sample comment ===
explain_toxic_comment("You stupid piece of trash, I hate you.")




