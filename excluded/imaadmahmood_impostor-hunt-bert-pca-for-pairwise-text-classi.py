from IPython.display import Image, display

img_path = "/kaggle/input/imposter-hunt-logo/logo.png"

display(Image(filename=img_path))


import os
import warnings
import pandas as pd
import numpy as np
import re
from datasets import Dataset
from datasets import DatasetDict
from transformers import AutoTokenizer,AutoModel
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use("fast") 


df = pd.read_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv')
df.head()


def train_data_generator(data_dir, csv_path):
    """
    Yield dictionaries of (text1, text2, label) for training.
    Label = 1 if file_1.txt is the real/original text, else 0.
    """
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        folder_id = row["id"]
        real_text_id = row["real_text_id"]

        folder_path = os.path.join(data_dir, f"article_{folder_id:04d}")
        file1_path = os.path.join(folder_path, "file_1.txt")
        file2_path = os.path.join(folder_path, "file_2.txt")

        with open(file1_path, encoding="utf-8") as f1:
            text1 = f1.read()
        with open(file2_path, encoding="utf-8") as f2:
            text2 = f2.read()

        label = 1 if real_text_id == 1 else 0

        yield {
            "id": folder_id,
            "text1": text1,
            "text2": text2,
            "label": label
        }


def test_data_generator(data_dir):
    """
    Yield dictionaries of (text1, text2) for testing (no labels).
    """
    folders = sorted([
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f)) and re.match(r'article_\d+', f)
    ])

    for folder in folders:
        folder_id = int(folder.split('_')[1])
        folder_path = os.path.join(data_dir, folder)

        file1_path = os.path.join(folder_path, "file_1.txt")
        file2_path = os.path.join(folder_path, "file_2.txt")

        with open(file1_path, encoding="utf-8") as f1:
            text1 = f1.read()
        with open(file2_path, encoding="utf-8") as f2:
            text2 = f2.read()

        yield {
            "id": folder_id,
            "text1": text1,
            "text2": text2
        }


# Paths
train_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
train_csv = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"
test_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"

# Create datasets from generators
train_dataset = Dataset.from_generator(lambda: train_data_generator(train_dir, train_csv))
test_dataset = Dataset.from_generator(lambda: test_data_generator(test_dir))

# Combine into a DatasetDict
raw_datasets = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})


raw_datasets


def extract_mean_pooling_vector(text, tokenizer, model, max_len=512, stride=256, device="cuda"):
    """
    Extracts a mean-pooled vector for a potentially long input text using sliding windows.
    - Handles token overflow via `stride`
    - Removes padding effects via attention mask
    - Returns a single average vector across all chunks
    """
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length"
    )

    input_ids_chunks = encoded["input_ids"]
    attention_mask_chunks = encoded["attention_mask"]

    all_mean_vecs = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for input_ids, attention_mask in zip(input_ids_chunks, attention_mask_chunks):
            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # shape: [1, seq_len, hidden_dim]

            # Apply mean pooling (excluding padded tokens)
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            masked_hidden = last_hidden_state * mask
            summed = masked_hidden.sum(dim=1)
            count = mask.sum(dim=1)
            mean_vec = summed / count

            all_mean_vecs.append(mean_vec.squeeze(0))

    # Average over all chunks to form the final vector
    final_vec = torch.stack(all_mean_vecs).mean(dim=0)

    return final_vec.cpu()


model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
bert_model = AutoModel.from_pretrained(model_checkpoint, num_labels=2)


#Test on a Sample Text
text = raw_datasets['train'][0]["text1"]
vector = extract_mean_pooling_vector(text, tokenizer, bert_model)
print(vector.shape)


def extract_features(dataset, tokenizer, model):
    """
    Extracts interaction-based features from each text pair.

    Returns:
        features: numpy array of shape [num_samples, feature_dim]
        ids: list of sample IDs
    """
    features = []
    ids = []

    for row in tqdm(dataset, desc="Extracting features"):
        vec1 = extract_mean_pooling_vector(row['text1'], tokenizer, model)
        vec2 = extract_mean_pooling_vector(row['text2'], tokenizer, model)

        # Compute interaction vectors
        diff = vec1 - vec2
        prod = vec1 * vec2

        # Concatenate all parts
        final_vec = torch.cat([vec1, vec2, diff, prod])
        features.append(final_vec.numpy())
        ids.append(row['id'])

    return np.array(features), ids


# Extract raw features from training datasets
X_train_raw, train_ids = extract_features(raw_datasets["train"], tokenizer, bert_model)


# Step 2: Apply PCA (fit only on training set)
n_components = 20
pca_model = PCA(n_components=n_components)

# Fit on training features only
X_train = pca_model.fit_transform(X_train_raw)

# Step 3: Extract labels
y_train = np.array([ex["label"] for ex in raw_datasets["train"]])


models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}


# Use stratified 5-fold cross-validation to preserve label balance
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    # Store metrics across folds
    accuracies, precisions, recalls, f1s = [], [], [], []

    # 5-fold CV loop
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Train model and predict
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        # Compute metrics
        acc = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='macro'
        )

        # Append metrics
        accuracies.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # Average metrics across folds
    results[name] = {
        "Accuracy": np.mean(accuracies),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "F1-score": np.mean(f1s)
    }


# Create DataFrame and sort by F1-score
df_results = pd.DataFrame(results).T.sort_values(by="F1-score", ascending=False).round(4)
print(df_results)


# Set visualization style
sns.set(style="whitegrid")

# Create grouped bar chart
ax = df_results.plot(
    kind="bar",
    figsize=(12, 6),
    edgecolor='black',
    linewidth=1.2
)

# Set titles and labels
ax.set_title("Model Evaluation Metrics (5-Fold Cross Validation)", fontsize=16, weight='bold')
ax.set_ylabel("Score", fontsize=14)
ax.set_xlabel("Model", fontsize=14)

# Adjust ticks and grid
plt.xticks(rotation=20)
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Legend formatting
plt.legend(title="Metrics", fontsize=12, title_fontsize=13)

plt.tight_layout()
plt.show()


# Extract raw interaction features from test set
X_test_raw, test_ids = extract_features(raw_datasets["test"], tokenizer, bert_model)

# Use previously fitted PCA model to transform test features
X_test = pca_model.transform(X_test_raw)


# Select the best model based on F1-score from cross-validation
best_model_name = df_results.index[0]
best_model = models[best_model_name]

# Retrain the best model on the entire training set
best_model.fit(X_train, y_train)


# Predict probability that text1 is the real one
test_probs = best_model.predict_proba(X_test)[:, 1]  # probability of class '1'


submission = []

# Loop through test IDs and assign predicted label
for i, pid in enumerate(test_dataset["id"]):
    prob = test_probs[i]
    real_text_id = 1 if prob >= 0.5 else 2
    submission.append((pid, real_text_id))

# Convert to DataFrame and export as CSV
submission_df = pd.DataFrame(submission, columns=["id", "real_text_id"])
submission_df.to_csv("submission.csv", index=False)

# Preview submission
print(submission_df.head())

