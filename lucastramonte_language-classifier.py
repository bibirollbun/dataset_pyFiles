pip install pycountry


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pycountry
import os
import io
import zipfile
import logging
import torch
import joblib
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.amp import GradScaler, autocast


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


###############################################################################
# Load the data
###############################################################################

data_path = "/kaggle/input/nlp-cs-2025/"
train_df = pd.read_csv(os.path.join(data_path, "train_submission.csv"))
test_df  = pd.read_csv(os.path.join(data_path, "test_without_labels.csv"))  # Assuming predictions come later

print("\n Train : Missing values in each column: \n\n", train_df.isnull().sum())
print("\n Test : Missing values in each column: \n\n", test_df.isnull().sum())

print("\n Train : Number of duplicated rows:", train_df.duplicated().sum())
print("\n Test : Number of duplicated rows:", test_df.duplicated().sum())

print("\n Train : Number of unique languages:", train_df["Label"].nunique())




###############################################################################
#Train data distribution (Top 20) by frequency
###############################################################################

# Function to get language name from ISO 639-3 code

def get_language_name(iso_code):
    try:
        language_name = pycountry.languages.get(alpha_3=iso_code).name
        return f"{language_name} ({iso_code})"
    except AttributeError:
        return iso_code

train_df["Language Name"] = train_df["Label"].astype(str).apply(get_language_name)
label_counts = train_df["Language Name"].value_counts(normalize=True) * 100
top_labels = label_counts.head(20)

plt.figure(figsize=(14, 8))  
ax = sns.barplot(y=top_labels.index, x=top_labels.values, palette="viridis")
plt.xlabel("Percentage (%)")
plt.ylabel("Language")
plt.title("Top 20 Most Frequent Languages (Percentage)")

for i, value in enumerate(top_labels.values):
    ax.text(value - 0.05, i, f"{value:.2f}%", va="center", ha="right", fontsize=10, color="white", fontweight="bold")

plt.tight_layout()
plt.show()


###############################################################################
# Compute word counts for train and test
###############################################################################
train_df["word_count"] = train_df["Text"].apply(lambda x: len(str(x).split()))
test_df["word_count"]  = test_df["Text"].apply(lambda x: len(str(x).split()))

train_total = len(train_df)
test_total  = len(test_df)

###############################################################################
#  Bin word counts:
#    0-9, 10-19, 20-29, 30-39, 40-99, 100+
###############################################################################
bins   = [0, 10, 20, 30, 40, 100, float("inf")]
labels = ["0-9", "10-19", "20-29", "30-39", "40-99", "100+"]

train_df["wc_bin"] = pd.cut(train_df["word_count"], bins=bins, labels=labels, right=False)
test_df["wc_bin"]  = pd.cut(test_df["word_count"],  bins=bins, labels=labels, right=False)

###############################################################################
# Compute the percentage in each bin for train & test
###############################################################################
train_bin_counts = train_df["wc_bin"].value_counts().reindex(labels, fill_value=0)
test_bin_counts  = test_df["wc_bin"].value_counts().reindex(labels, fill_value=0)

# Convert to percentage
train_bin_perc = (train_bin_counts / train_total) * 100
test_bin_perc  = (test_bin_counts / test_total) * 100

print("Train bin percentages (%):")
print(train_bin_perc)
print("\nTest bin percentages (%):")
print(test_bin_perc)

###############################################################################
# Plot a horizontal bar chart comparing train vs. test side by side
###############################################################################
N     = len(labels)  # 6 bins
ind   = np.arange(N)
width = 0.4

plt.figure(figsize=(7,5))
for i in range(N):
    # Train bars (light blue) on the left side
    plt.barh(ind[i] - width/2, train_bin_perc.iloc[i], height=width, color="lightblue",
             label=None if i>0 else "Train")
    # Test bars (dark green) on the right side
    plt.barh(ind[i] + width/2, test_bin_perc.iloc[i], height=width, color="darkgreen",
             label=None if i>0 else "Test")

plt.yticks(ind, labels)
plt.xlabel("Percentage %")
plt.ylabel("Word Count Range")
plt.title("Word Count Distribution: Train vs. Test (in %)")
plt.legend(loc="lower right")
plt.gca().invert_yaxis()  # So the first bin appears at the top
plt.tight_layout()


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define paths for Kaggle Notebook
MODEL_DIR = "/kaggle/working/xlmroberta_model"
SUBMISSION_DIR = "/kaggle/working/submission.csv"

class LanguageDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)




class LanguageClassifierBERT:
    def __init__(self, model_name='xlm-roberta-base', max_len=128, batch_size=128, epochs=18):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.class_weights = None
        self.model = None
        logger.info("Initialization complete.")

    def preprocess_data(self, data):
        logger.info("Preprocessing data...")
        X = data['Text'].tolist()
        y = data['Label'].tolist()
        y = self.label_encoder.fit_transform(y)
        num_labels = len(self.label_encoder.classes_)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels).to(self.device)
        encodings = self.tokenizer(X, truncation=True, padding=True, max_length=self.max_len, return_tensors='pt')
        class_counts = torch.bincount(torch.tensor(y))
        class_weights = 1.0 / class_counts.float()
        self.class_weights = (class_weights / class_weights.sum()).to(self.device)
        logger.info("Data preprocessing complete.")
        return encodings.to(self.device), torch.tensor(y).to(self.device)

    def train(self, encodings, y):
        logger.info("Starting training...")
        dataset = LanguageDataset(encodings, y)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        scaler = GradScaler('cuda')
        best_loss = float('inf')
        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                with autocast('cuda'):
                    outputs = self.model(**batch)
                    loss = criterion(outputs.logits, batch['labels'])
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} - Training loss: {avg_train_loss:.4f}")
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                self.save_model()
        logger.info("Training complete.")

    def save_model(self):
        self.model.save_pretrained(MODEL_DIR)
        self.tokenizer.save_pretrained(MODEL_DIR)
        joblib.dump(self.label_encoder, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        self.is_trained = True
        logger.info("Model saved successfully.")

    def load_model(self):
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError("Model not found. Train the model first.")
        logger.info("Loading model...")
        self.model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_DIR).to(self.device)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_DIR)
        self.label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        self.is_trained = True
        logger.info("Model loaded successfully.")

    def predict_language(self, text):
        if not self.is_trained:
            self.load_model()
        encodings = self.tokenizer([text], truncation=True, padding=True, max_length=self.max_len, return_tensors='pt')
        inputs = {key: val.to(self.device) for key, val in encodings.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        return self.label_encoder.inverse_transform([prediction])[0]

    def generate_submission(self, test_data_path):
        test_data = pd.read_csv(test_data_path)
        if 'ID' not in test_data.columns:
            test_data['ID'] = range(1, len(test_data) + 1)
        predictions = [self.predict_language(text) for text in tqdm(test_data['Text'], desc="Predicting languages")]
        submission_df = pd.DataFrame({'ID': test_data['ID'], 'Label': predictions})
        submission_df.to_csv(SUBMISSION_DIR, index=False)
        logger.info(f"Submission saved to {SUBMISSION_DIR}")


classifier = LanguageClassifierBERT()
train_encodings, train_labels = classifier.preprocess_data(train_df)
classifier.train(train_encodings, train_labels)



classifier.generate_submission("/kaggle/input/nlp-cs-2025/test_without_labels.csv")

