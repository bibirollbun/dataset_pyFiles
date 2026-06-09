!pip install fpdf
!pip install pdfplumber



import pandas as pd
import os
import xml.etree.ElementTree as ET
import pdfplumber
from google.colab import drive
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np



import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)



import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


train_label = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/train_labels.csv")
sample_submission = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/sample_submission.csv")



print("train_label.head():")
train_label.head()



print("sample_submission.head():")
sample_submission.head()


print("Label type counts:")
train_label['type'].value_counts()


print("Label type proportions (%):")
train_label['type'].value_counts(normalize=True) * 100



n_articles = train_label['article_id'].nunique()
print(f"Number of unique articles in train set: {n_articles}\n")



# Bar plot: label counts
plt.figure(figsize=(10, 5))
sns.countplot(data=train_label, x="type", order=train_label['type'].value_counts().index, palette="viridis")
plt.title("Label Distribution (Count)")
plt.ylabel("Number of Instances")
plt.xlabel("Label Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Pie chart: label proportions
plt.figure(figsize=(8, 8))
train_label['type'].value_counts().plot.pie(autopct="%1.1f%%", colors=sns.color_palette("pastel"))
plt.title("Label Type Proportions")
plt.ylabel("")
plt.show()


# Filter out 'Missing' types
df_valid = train_label[train_label['type'] != 'Missing']

# Group dataset_id and types per article_id
grouped = df_valid.groupby('article_id').agg({
    'dataset_id': lambda x: list(set(x)),
    'type': lambda x: list(set(x))
}).reset_index()

# Directory paths for XML and PDF files
data_dir = "/kaggle/input/make-data-count-finding-data-references/train/"
xml_dir = os.path.join(data_dir, "XML")
pdf_dir = os.path.join(data_dir, "PDF")



def extract_text_from_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return " ".join(elem.text.strip() for elem in root.iter() if elem.text)
    except Exception as e:
        print(f"[XML] Error reading {xml_path}: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        print(f"[PDF] Error reading {pdf_path}: {e}")
        return None

def load_article_text(article_id):
    xml_path = os.path.join(xml_dir, f"{article_id}.xml")
    pdf_path = os.path.join(pdf_dir, f"{article_id}.pdf")
    if os.path.exists(xml_path):
        print(f"Using XML for {article_id}")
        return extract_text_from_xml(xml_path)
    elif os.path.exists(pdf_path):
        print(f"Using PDF for {article_id}")
        return extract_text_from_pdf(pdf_path)
    else:
        print(f"Missing both XML and PDF for {article_id}")
        return None

# Add extracted text to dataframe
grouped['text'] = grouped['article_id'].apply(load_article_text)
grouped = grouped.dropna(subset=['text'])

# Show an example
if not grouped.empty:
    print("\nExample extracted text:")
    print(grouped[['article_id', 'text']].iloc[0])
else:
    print("⚠️ No text extracted. Please check XML/PDF files.")



# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(grouped['text'])

# Multi-label binarization
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(grouped['type'])

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


clf = OneVsRestClassifier(RandomForestClassifier(class_weight='balanced', n_estimators=100))
clf.fit(X_train, Y_train)


# Predict probabilities
Y_proba_val = clf.predict_proba(X_test)

# Find best threshold for each class
optimal_thresholds = []
for i in range(Y_proba_val.shape[1]):
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (Y_proba_val[:, i] >= thresh).astype(int)
        f1 = f1_score(Y_test[:, i], preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    optimal_thresholds.append(best_thresh)
    print(f"Class {mlb.classes_[i]}: best threshold = {best_thresh:.2f}, F1 = {best_f1:.3f}")

# Apply optimal thresholds
Y_pred_optimal = np.zeros_like(Y_proba_val, dtype=int)
for i, thresh in enumerate(optimal_thresholds):
    Y_pred_optimal[:, i] = (Y_proba_val[:, i] >= thresh).astype(int)

# Final evaluation
print("Macro F1-score:", f1_score(Y_test, Y_pred_optimal, average='macro'))
print("Micro F1-score:", f1_score(Y_test, Y_pred_optimal, average='micro'))

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred_optimal, target_names=mlb.classes_))



# Predict on full dataset
preds = clf.predict(X)
pred_labels = mlb.inverse_transform(preds)

submission_data = []
row_id = 0

for i, labels in enumerate(pred_labels):
    article_id = grouped.iloc[i]['article_id']
    dataset_ids = grouped.iloc[i]['dataset_id']
    for label in labels:
        for dataset_id in dataset_ids:
            submission_data.append({
                "row_id": row_id,
                "article_id": article_id,
                "dataset_id": dataset_id,
                "type": label
            })
            row_id += 1

submission_df = pd.DataFrame(submission_data)

print("\n✅ submission df")
print(submission_df.head())


# ==========================================================
# IMPORTS & SETUP
# ==========================================================

import pandas as pd
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from collections import Counter

# ==========================================================
# LOAD PREDICTIONS AND TRUE LABELS
# ==========================================================

# Load ground truth labels and filter out 'Missing'
true_labels = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/train_labels.csv")
true_labels = true_labels[true_labels['type'] != 'Missing']


# ==========================================================
# GROUP AND ALIGN PREDICTIONS WITH TRUE LABELS
# ==========================================================

# Group predicted and true labels by article_id
# Group labels by article_id
preds_grouped = submission_df.groupby('article_id')['type'].apply(set).reset_index(name='type_pred')
true_grouped = true_labels.groupby('article_id')['type'].apply(set).reset_index(name='type_true')

# Merge predicted and true labels
df = pd.merge(true_grouped, preds_grouped, on='article_id', how='inner')




# ==========================================================
# MULTI-LABEL BINARIZATION
# ==========================================================

mlb = MultiLabelBinarizer()
Y_true = mlb.fit_transform(df["type_true"])
Y_pred = mlb.transform(df["type_pred"])

print("=== SKLEARN CLASSIFICATION REPORT ===")
print(classification_report(Y_true, Y_pred, target_names=mlb.classes_))




# ==========================================================
# FINAL CLASSIFICATION REPORT
# ==========================================================

print(classification_report(Y_true, Y_pred, target_names=mlb.classes_))





# ==========================================================
# CONVERT MULTI-LABEL TO BINARY FORMAT
# ==========================================================

# Create binary flags for each label in true and predicted sets
# Create binary columns for selected labels
df["Primary_true"] = df["type_true"].apply(lambda x: "Primary" in x)
df["Secondary_true"] = df["type_true"].apply(lambda x: "Secondary" in x)
df["Primary_pred"] = df["type_pred"].apply(lambda x: "Primary" in x)
df["Secondary_pred"] = df["type_pred"].apply(lambda x: "Secondary" in x)

# Prepare binary arrays
Y_true = df[["Primary_true", "Secondary_true"]].astype(int).values
Y_pred = df[["Primary_pred", "Secondary_pred"]].astype(int).values


# ==========================================================
# MULTILABEL CONFUSION MATRICES
# ==========================================================

# Compute multilabel confusion matrices
mcm = multilabel_confusion_matrix(Y_true, Y_pred)

# ==========================================================
# DISPLAY CONFUSION MATRICES WITH HEATMAPS
# ==========================================================


for idx, label in enumerate(["Primary", "Secondary"]):
    tn, fp, fn, tp = mcm[idx].ravel()
    cm = [[tp, fn], [fp, tn]]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for "{label}"')
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.show()




# ==========================================================
# EXTRACT UNIQUE LABELS FROM TRUE COLUMN
# ==========================================================

labels = df["type_true"].explode().unique()

# ==========================================================
# INITIALIZE COUNTERS FOR EACH METRIC
# ==========================================================

TP = Counter()
FP = Counter()
FN = Counter()
support = Counter()


# ==========================================================
# COMPUTE TP, FP, FN, AND SUPPORT PER CLASS
# ==========================================================

for label in labels:
    for true, pred in zip(df["type_true"], df["type_pred"]):
        if label in true and label in pred:
            TP[label] += 1
        elif label not in true and label in pred:
            FP[label] += 1
        elif label in true and label not in pred:
            FN[label] += 1
    support[label] = sum(label in x for x in df["type_true"])


# ==========================================================
# COMPUTE PRECISION, RECALL, F1 PER CLASS
# ==========================================================

precision = {}
recall = {}
f1 = {}

for label in labels:
    tp, fp, fn = TP[label], FP[label], FN[label]
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0

    precision[label] = p
    recall[label] = r
    f1[label] = f

# ==========================================================
# MACRO / MICRO / WEIGHTED F1 SCORES
# ==========================================================

# F1 Macro: unweighted average over classes
f1_macro = sum(f1.values()) / len(f1)

# F1 Micro: global counts of TP, FP, FN
total_tp = sum(TP.values())
total_fp = sum(FP.values())
total_fn = sum(FN.values())

precision_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0

# F1 Weighted: average F1 weighted by support
total_samples = sum(support.values())
f1_weighted = sum((support[label] / total_samples) * f1[label] for label in labels)


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("F1 Macro Score     :", round(f1_macro, 4))
print("F1 Micro Score     :", round(f1_micro, 4))
print("F1 Weighted Score  :", round(f1_weighted, 4))
print()
print("Per-Class Metrics:")
for label in labels:
    print(f"- {label}: Precision={precision[label]:.2f}, Recall={recall[label]:.2f}, F1={f1[label]:.2f}, Support={support[label]}")

    # Articles mal classés
df["error"] = df["type_true"] != df["type_pred"]
erreurs = df[df["error"]]

# Afficher les erreurs (premiers exemples)
print(erreurs[["article_id", "type_true", "type_pred"]].head(10))






from sklearn.metrics import jaccard_score

print("Jaccard (micro):", jaccard_score(Y_true, Y_pred, average='micro'))

