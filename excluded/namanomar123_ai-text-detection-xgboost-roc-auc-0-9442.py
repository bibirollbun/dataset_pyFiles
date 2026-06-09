import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
%matplotlib inline


from warnings import filterwarnings
filterwarnings('ignore')


!nvidia-smi


TRAIN_DATA = "/kaggle/input/mercor-ai-detection/train.csv"
TEST_DATA = "/kaggle/input/mercor-ai-detection/test.csv"


train= pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)


print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


# Checking data - imbalanace
train['is_cheating'].value_counts()


# checking missing values
print(train.isnull().sum(),"\n")
print(train.isnull().sum())


train.info()


# Text Length Analysis 
train['text_length'] = train['answer'].astype(str).apply(len)
train['word_count'] = train['answer'].astype(str).apply(lambda x: len(x.split()))

plt.figure(figsize=(10,4))
sns.histplot(train['word_count'], bins=50, kde=True, color='skyblue')
plt.title("Distribution of Word Counts in Answers")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()


print(train.groupby('is_cheating')['word_count'].mean())


if 'is_cheating' in train.columns:
    plt.figure(figsize=(5,4))
    sns.countplot(x='is_cheating', data=train, palette='coolwarm')
    plt.title("Distribution of 'is_cheating' Labels")
    plt.xlabel("is_cheating (0=Human, 1=AI/Cheat)")
    plt.ylabel("Count")
    plt.show()
    
    target_counts = train['is_cheating'].value_counts(normalize=True).rename_axis('label').reset_index(name='proportion')
    print("\nâš–ï¸� Class Balance (Proportion):")
    print(target_counts)


nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    if pd.isnull(text):
        return ""
    
    text = text.lower()                                     # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)              # remove links
    text = re.sub(r"\d+", "", text)                         # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()                # remove extra whitespace
    words = [word for word in text.split() if word not in STOPWORDS] # remove stopwords
    return " ".join(words)


# cleaning the answer
train['clean_answer'] = train['answer'].astype(str).apply(clean_text)
test['clean_answer'] = test['answer'].astype(str).apply(clean_text)


# Combining the topic with answer 
train['combined_text'] = train['topic'].astype(str) + " " + train['clean_answer']
test['combined_text'] = test['topic'].astype(str) + " " + test['clean_answer']


train[['topic', 'answer', 'clean_answer']].head(3)


train['clean_word_count'] = train['clean_answer'].apply(lambda x: len(x.split()))


print("\nğŸ“� Average Clean Word Count:")
if 'is_cheating' in train.columns:
    print(train.groupby('is_cheating')['clean_word_count'].mean())

plt.figure(figsize=(10,4))
sns.histplot(train['clean_word_count'], bins=50, kde=True, color='salmon')
plt.title("Distribution of Clean Word Counts")
plt.xlabel("Word Count (Cleaned)")
plt.ylabel("Frequency")
plt.show()


text_col = "combined_text"   
tfidf = TfidfVectorizer(
    max_features=5000,        # limits features to most informative words
    ngram_range=(1, 2),       # unigrams + bigrams capture style & fluency
    min_df=3,                 # ignore rare terms
    max_df=0.9,               # ignore too-common terms
    sublinear_tf=True         # dampen term frequency effect
)


X_train_tfidf = tfidf.fit_transform(train[text_col])
X_test_tfidf  = tfidf.transform(test[text_col])


print(f"âœ… TF-IDF Vectors Created: {X_train_tfidf.shape[1]} features")


### Numeric (Stylometric) Features
def extract_text_features(df):
    df["char_count"] = df["clean_answer"].apply(len)
    df["word_count"] = df["clean_answer"].apply(lambda x: len(x.split()))
    df["avg_word_length"] = df["clean_answer"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if len(x.split())>0 else 0
    )
    df["punct_count"] = df["answer"].apply(lambda x: len(re.findall(r"[!?,.;:]", str(x))))
    df["digit_count"] = df["answer"].apply(lambda x: len(re.findall(r"\d", str(x))))
    df["uppercase_ratio"] = df["answer"].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / (len(x) + 1))
    return df

train = extract_text_features(train)
test = extract_text_features(test)


if "is_cheating" in train.columns:
    corr_features = ["char_count", "word_count", "avg_word_length", "punct_count", "digit_count", "uppercase_ratio"]
    corr = train[corr_features + ["is_cheating"]].corr()["is_cheating"].sort_values(ascending=False)
    print("\nğŸ“ˆ Correlation of features with `is_cheating`:")
    print(corr)


from scipy.sparse import hstack

numeric_features = ["char_count", "word_count", "avg_word_length", "punct_count", "digit_count", "uppercase_ratio"]

X_train_numeric = train[numeric_features].values
X_test_numeric  = test[numeric_features].values

# Combine sparse (TF-IDF) and dense (numeric) features
X_train_final = hstack([X_train_tfidf, X_train_numeric])
X_test_final  = hstack([X_test_tfidf, X_test_numeric])

print(f"\n Final Feature Matrix Shape: {X_train_final.shape}")


from scipy.sparse import csr_matrix

# Convert sparse matrices to CSR format
X_train_final = csr_matrix(X_train_final)
X_test_final = csr_matrix(X_test_final)


## Catboost
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from scipy.sparse import csr_matrix


y_train = train["is_cheating"].values


# Strarified folding
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))


n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / (n_pos + 1e-9)
print(f"ğŸ“Š Positive: {n_pos}, Negative: {n_neg}, scale_pos_weight = {scale_pos_weight:.3f}")


cat_params = {
    "iterations": 1500,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3,
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "random_seed": 42,
    "verbose": 200,
    "task_type": "CPU",       # change to "GPU" if you have one
    "scale_pos_weight": scale_pos_weight,
    "od_type": "Iter",        # early stopping based on iterations
    "od_wait": 50             # stop if no improvement for 50 rounds
}


fold_aucs = []

fold = 1
for train_idx, val_idx in skf.split(X_train_final, y_train):
    print(f"\nğŸš€ Fold {fold} Training...")
    X_tr, X_val = X_train_final[train_idx], X_train_final[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    X_tr_dense = X_tr.toarray()
    X_val_dense = X_val.toarray()
    
    model = CatBoostClassifier(**cat_params)
    
    model.fit(
        X_tr_dense, y_tr,
        eval_set=(X_val_dense, y_val),
        use_best_model=True
    )
    val_preds = model.predict_proba(X_val_dense)[:, 1]
    oof_preds[val_idx] = val_preds
    
    fold_auc = roc_auc_score(y_val, val_preds)
    fold_aucs.append(fold_auc)
    print(f"âœ… Fold {fold} ROC-AUC: {fold_auc:.4f}")

    test_preds += model.predict_proba(X_test_final.toarray())[:, 1] / skf.n_splits
    fold += 1



cv_auc = roc_auc_score(y_train, oof_preds)
print("\nğŸ�� Overall CV ROC-AUC:", round(cv_auc, 4))
print("Fold AUCs:", [round(a, 4) for a in fold_aucs])


submission = pd.DataFrame({
    "id": test["id"],
    "is_cheating": test_preds
})


submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as submission.csv")

