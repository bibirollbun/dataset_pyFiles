import os
import zipfile

zip_files = [
    '/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip', 
    '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip', 
    '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip', 
    '/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip'
]

extract_base_folder = '/kaggle/working/jigsaw-toxic-comment-classification-challenge'

os.makedirs(extract_base_folder, exist_ok=True)

for zip_file in zip_files:
    extract_folder = os.path.join(extract_base_folder, os.path.splitext(os.path.basename(zip_file))[0])
    
    if not os.path.exists(extract_folder):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        print(f"Extracted {zip_file} to {extract_folder}")
    else:
        print(f"Skipping {zip_file}, {extract_folder} already exists.")


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

nltk.download('stopwords')

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/working/jigsaw-toxic-comment-classification-challenge/train.csv/train.csv')
df


df.describe()


df.info()


df.isnull().sum()


x = df.iloc[:, 2:].sum() # Chỉ lấy các cột label
x


rowsums = df.iloc[:, 2:].sum(axis=1) # Lấy các cột label và tính tổng theo từng cột
rowsums


no_label_count = 0

for i, count in rowsums.items():
    if count==0:
        no_label_count += 1
        
print('Tổng số lượng comments: ', len(df))
print('Số lượng comment chưa được gán nhãn: ', no_label_count)
print('Số lượng label ', x.sum())


plt.figure(figsize=(6, 4))
ax = sns.barplot(x=x.index, y=x.values, alpha=0.8, palette=['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:red', 'tab:grey'])
plt.title('Phân bố Label của Dataset')
plt.ylabel('Count')
plt.xlabel('Label')

plt.show()


plt.figure(figsize=(6, 4))
ax = sns.countplot(x=rowsums.values, alpha=0.8, palette=['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:red', 'tab:grey'])
plt.title('Phân bố Labels cho mỗi Comment')
plt.ylabel('# of Occurences')
plt.xlabel('# of Labels')

plt.show()


df = df.drop(columns=['id'], axis=1) # Drop cột id


df.head()


# Fill NaN
df['comment_text'] = df['comment_text'].fillna('').astype(str)


# Set STOPWORDS
STOPWORDS = set(stopwords.words('english'))


def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(no_stopword_text)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(' ')
    return text


stemmer = SnowballStemmer('english')

def stemming(sentence):
    stemmed_sentence = ""
    for word in sentence.split():
        stemmed_word = stemmer.stem(word)
        stemmed_sentence += stemmed_word + " "
    stemmed_sentence = stemmed_sentence.strip()
    return stemmed_sentence


df['comment_text'] = df['comment_text'].apply(remove_stopwords)
df['comment_text'] = df['comment_text'].apply(clean_text)
df['comment_text'] = df['comment_text'].apply(stemming)


df.head(15)


# Các cột label gốc cho multi-label classification
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Binary target: comment được coi là toxic nếu có ÍT NHẤT một trong các label trên = 1
df["any_toxic"] = (df[label_cols].max(axis=1) > 0).astype(int)


# Binary classification: any_toxic và non-toxic
X_bin = df["comment_text"]
y_bin = df["any_toxic"]

# Multi-label classification
X_multi = df["comment_text"]
Y_multi = df[label_cols]


# Binary classification
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_bin, y_bin,
    test_size=0.2,
    random_state=42,
    stratify=y_bin
)

# Multi-label classification
X_train_multi, X_test_multi, Y_train_multi, Y_test_multi = train_test_split(
    X_multi, Y_multi,
    test_size=0.2,
    random_state=42
)


# ---- Binary: Confusion Matrix ----
def plot_confusion_matrix_binary(y_true, y_pred, title="Confusion Matrix (binary)"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"]
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ---- Binary: ROC Curve ----
def plot_roc_curve_binary(y_true, y_scores, title="ROC Curve (binary)"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# ---- Multi-label: Confusion Matrices cho từng label ----
def plot_confusion_matrices_multilabel(Y_true, Y_pred, label_names):
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)
    n_labels = len(label_names)

    n_cols = 3
    n_rows = int(np.ceil(n_labels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.ravel()

    for i, label in enumerate(label_names):
        cm = confusion_matrix(Y_true[:, i], Y_pred[:, i])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            ax=axes[i]
        )
        axes[i].set_title(label)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    # Ẩn axes thừa (nếu n_labels không chia hết cho n_cols)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

# ---- Multi-label: ROC Curves cho từng label ----
def plot_roc_curves_multilabel(Y_true, Y_scores, label_names, title_prefix="ROC (multi-label)"):
    Y_true = np.asarray(Y_true)
    Y_scores = np.asarray(Y_scores)
    n_labels = len(label_names)

    plt.figure(figsize=(6, 6))
    for i, label in enumerate(label_names):
        try:
            fpr, tpr, _ = roc_curve(Y_true[:, i], Y_scores[:, i])
            auc = roc_auc_score(Y_true[:, i], Y_scores[:, i])
            plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
        except ValueError:
            # Trường hợp label toàn 0 hoặc toàn 1 → không vẽ được ROC
            continue

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title_prefix)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()


def run_binary(pipeline, model_name, X_train_bin, X_test_bin, y_train_bin, y_test_bin):
    print(f"\n=== {model_name} | Binary classification ===")
    
    # train
    pipeline.fit(X_train_bin, y_train_bin)
    
    # predict labels
    y_pred = pipeline.predict(X_test_bin)
    
    # predict scores for ROC-AUC
    if hasattr(pipeline, "predict_proba"):
        y_scores = pipeline.predict_proba(X_test_bin)
        # lấy xác suất lớp positive (cột 1)
        if y_scores.ndim == 2 and y_scores.shape[1] == 2:
            y_scores = y_scores[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_scores = pipeline.decision_function(X_test_bin)
    else:
        y_scores = None
    
    # metrics
    print("accuracy:", accuracy_score(y_test_bin, y_pred))
    print("f1_binary:", f1_score(y_test_bin, y_pred, average="binary"))
    print("precision:", precision_score(y_test_bin, y_pred, average="binary"))
    print("recall:", recall_score(y_test_bin, y_pred, average="binary"))
    
    if y_scores is not None:
        print("roc_auc:", roc_auc_score(y_test_bin, y_scores))
    
    print("\nclassification report:")
    print(classification_report(y_test_bin, y_pred))
    
    print("confusion matrix (numbers):")
    print(confusion_matrix(y_test_bin, y_pred))
    
    # Plot confusion matrix
    plot_confusion_matrix_binary(y_test_bin, y_pred, title=f"{model_name} - Confusion Matrix")
    
    # Plot ROC curve
    if y_scores is not None:
        plot_roc_curve_binary(y_test_bin, y_scores, title=f"{model_name} - ROC Curve")


def run_multilabel(pipeline, model_name,
                   X_train_multi, X_test_multi,
                   Y_train_multi, Y_test_multi,
                   label_names):
    print(f"\n=== {model_name} | Multi-label classification ===")
    
    # train
    pipeline.fit(X_train_multi, Y_train_multi)
    
    # predict labels
    Y_pred = pipeline.predict(X_test_multi)
    
    # predict scores cho ROC-AUC
    if hasattr(pipeline, "predict_proba"):
        Y_scores = pipeline.predict_proba(X_test_multi)
    elif hasattr(pipeline, "decision_function"):
        Y_scores = pipeline.decision_function(X_test_multi)
    else:
        Y_scores = None
    
    # metrics
    print("accuracy (exact match):", accuracy_score(Y_test_multi, Y_pred))
    print("f1_micro:", f1_score(Y_test_multi, Y_pred, average="micro"))
    print("f1_macro:", f1_score(Y_test_multi, Y_pred, average="macro"))
    
    if Y_scores is not None:
        print("roc_auc_macro:", roc_auc_score(Y_test_multi, Y_scores, average="macro"))
    
    print("\nclassification report (per label):")
    print(classification_report(Y_test_multi, Y_pred, target_names=label_names))
    
    # Plot confusion matrices
    plot_confusion_matrices_multilabel(Y_test_multi, Y_pred, label_names)
    
    # Plot ROC curves
    if Y_scores is not None:
        plot_roc_curves_multilabel(Y_test_multi, Y_scores, label_names,
                                   title_prefix=f"{model_name} - ROC (per label)")


# Multinomial Naive Bayes
NB_pipeline_bin = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb_model', MultinomialNB())
])

# Logistic Regression
LR_pipeline_bin = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('lr_model', LogisticRegression(max_iter=1000))
])

# LinearSVM
SVM_pipeline_bin = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("svm_model", LinearSVC())
])


run_binary(
    LR_pipeline_bin,
    "Logistic Regression (binary)",
    X_train_bin, X_test_bin,
    y_train_bin, y_test_bin
)


run_binary(
    SVM_pipeline_bin,
    "Linear SVM (binary)",
    X_train_bin, X_test_bin,
    y_train_bin, y_test_bin
)


run_binary(
    NB_pipeline_bin,
    "MultinomialNB (binary)",
    X_train_bin, X_test_bin,
    y_train_bin, y_test_bin
)


# MultinomialNB (OneVsRest)
NB_pipeline_multi = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb_model', OneVsRestClassifier(MultinomialNB(), n_jobs=-1))
])

# Logistic Regression (OneVsRest)
LR_pipeline_multi = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('lr_model', OneVsRestClassifier(LogisticRegression(max_iter=1000), n_jobs=-1))
])

# Multi-label: LinearSVC (OneVsRest)
SVM_pipeline_multi = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('svm_model', OneVsRestClassifier(LinearSVC(), n_jobs=-1))
])



run_multilabel(
    LR_pipeline_multi,
    "Logistic Regression (multi-label)",
    X_train_multi, X_test_multi,
    Y_train_multi, Y_test_multi,
    label_cols
)


run_multilabel(
    SVM_pipeline_multi,
    "Linear SVM (multi-label)",
    X_train_multi, X_test_multi,
    Y_train_multi, Y_test_multi,
    label_cols
)


run_multilabel(
    NB_pipeline_multi,
    "MultinomialNB (multi-label)",
    X_train_multi, X_test_multi,
    Y_train_multi, Y_test_multi,
    label_cols
)


test_path = "/kaggle/working/jigsaw-toxic-comment-classification-challenge/test.csv/test.csv"


test_df = pd.read_csv(test_path)
test_df.head(10)


# Xử lý NaN
test_df["comment_text"] = test_df["comment_text"].fillna("").astype(str)

# xóa stopwords -> làm sạch text -> stemming
test_df["comment_text"] = test_df["comment_text"].apply(remove_stopwords)
test_df["comment_text"] = test_df["comment_text"].apply(clean_text)
test_df["comment_text"] = test_df["comment_text"].apply(stemming)

test_df.head(10)


label_cols


# X là cột comment_text đã preproces
X_test_kaggle = test_df["comment_text"]

# Dự đoán xác suất cho từng label với mô hình tốt nhất (Logistic Regression)
test_probs = LR_pipeline_multi.predict_proba(X_test_kaggle)


# Tạo DataFrame submission
submission = pd.DataFrame(test_probs, columns=label_cols)

# Thêm cột id ở đầu để khớp với test.csv
submission.insert(0, "id", test_df["id"].values)

submission.head()


submission_file = "submission.csv"
submission.to_csv(submission_file, index=False)

print("Saved:", submission_file)
print(submission.head())


import joblib

# Save binary models
joblib.dump(NB_pipeline_bin,  "nb_binary.joblib")
joblib.dump(LR_pipeline_bin,  "lr_binary.joblib")
joblib.dump(SVM_pipeline_bin, "svm_binary.joblib")

# Save multi-label models
joblib.dump(NB_pipeline_multi,  "nb_multilabel.joblib")
joblib.dump(LR_pipeline_multi,  "lr_multilabel.joblib")
joblib.dump(SVM_pipeline_multi, "svm_multilabel.joblib")

print("Saved 6 models:")
print("  nb_binary.joblib")
print("  lr_binary.joblib")
print("  svm_binary.joblib")
print("  nb_multilabel.joblib")
print("  lr_multilabel.joblib")
print("  svm_multilabel.joblib")

