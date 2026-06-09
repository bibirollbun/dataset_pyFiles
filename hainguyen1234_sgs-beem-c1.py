import numpy as np
import pandas as pd
# Visualization (optional but helpful)
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('/kaggle/input/rmit-hackathon-2025/train.csv')




pd.set_option('display.max_colwidth', 50)


data


data.shape


data.info()


data.isnull().sum()





def label_distribution():
  if 'label' in data.columns:
      plt.figure(figsize=(6,4))
      sns.countplot(x='label', data=data)
      plt.title('Label Distribution')
      plt.show()
  elif 'target' in data.columns:
      plt.figure(figsize=(6,4))
      sns.countplot(x='target', data=data)
      plt.title('Target Distribution')
      plt.show()


label_distribution()



# Create text length column
data['text_length'] = data['text'].str.len()

# Basic statistics
print(data['text_length'].describe())



def text_length_distribution():
  plt.figure(figsize=(6,4))
  plt.hist(data['text_length'], bins=50, color='skyblue', edgecolor='black')
  plt.title('Text Length Distribution')
  plt.xlabel('Number of Characters')
  plt.ylabel('Frequency')
  plt.show()


text_length_distribution()


def text_length_by_label():
  sns.boxplot(x='label', y='text_length', data=data)
  plt.title('Text Length by Label')
  plt.show()


text_length_by_label()


def world_cloud():
  from wordcloud import WordCloud

  for label in data['label'].unique():
      text = ' '.join(data[data['label']==label]['text'])
      wc = WordCloud(width=800, height=400, background_color='white').generate(text)
      plt.figure(figsize=(8,4))
      plt.imshow(wc, interpolation='bilinear')
      plt.axis('off')
      plt.title(f'WordCloud - Label {label}')
      plt.show()


world_cloud()


def special_chars_by_label():
  data['num_symbols'] = data['text'].str.count(r'[!@#$%^&*()]')
  sns.boxplot(x='label', y='num_symbols', data=data)
  plt.title('Special Character Usage by Label')
  plt.show()


special_chars_by_label()


import string
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def processed_text(text):
  text = text.lower()
  text = text.translate(str.maketrans("", "", string.punctuation))
  words = text.split()
  words = [word for word in words if word not in ENGLISH_STOP_WORDS]
  text = ' '.join(words)
  return text


def test_processed_text(row_num):
  sample_text = data['text'].values[row_num]
  print("Before:", sample_text)
  print("After:", processed_text(sample_text))


test_processed_text(100)


# Split text into words and count frequency
all_words = ' '.join(data['text'].astype(str)).lower().split()
word_freq = Counter(all_words)

# Show top 20 most common words
word_freq.most_common(20)


def create_cleaned_text_col(old_col, new_col):
  data[new_col] = data[old_col].astype(str).apply(processed_text)


create_cleaned_text_col('text', 'cleaned_text')


data


print(data['label'].value_counts())


def balance_dataset(df, label_col='label'):
    import builtins
    len = builtins.len

    benign_df = df[df[label_col] == 'benign']
    jailbreak_df = df[df[label_col] == 'jailbreak']

    diff = len(benign_df) - len(jailbreak_df)  # ✅ works after reset

    jailbreak_upsampled = jailbreak_df.sample(diff, replace=True, random_state=42)

    balanced_df = pd.concat([benign_df, jailbreak_df, jailbreak_upsampled])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df


data_balanced = balance_dataset(data)
print(data_balanced['label'].value_counts())


def encode_labels(df, old_col, new_col):
    df[new_col] = df[old_col].map({'benign': 0, 'jailbreak': 1})
    return df


encode_labels(data_balanced,'label', 'label_num')


def remove_column(df, col_name):
    if col_name in df.columns:
        df = df.drop(columns=[col_name], inplace=True)
    return df


remove_column(data_balanced, 'Id')


# data_balanced.to_csv('cleaned_data.csv', index=False)


# 1️⃣ Check actual column names
print(data_balanced.columns.tolist())


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


X = data_balanced['cleaned_text'].astype(str)
y = data_balanced['label_num']


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


from sklearn.pipeline import Pipeline

# Pipeline: TF-IDF -> Logistic Regression
clf = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2)),
    ("logreg", LogisticRegression(max_iter=1000))
])

# Train
clf.fit(X_train, y_train)


# Train performance
y_train_pred = clf.predict(X_train)
y_train_prob = clf.predict_proba(X_train)[:, 1]

train_auc = roc_auc_score(y_train, y_train_prob)
print("TRAIN ROC-AUC:", round(train_auc, 4))
print("TRAIN REPORT:\n", classification_report(y_train, y_train_pred))

# Test performance
y_test_pred = clf.predict(X_test)
y_test_prob = clf.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_test_prob)
print("\nTEST ROC-AUC:", round(test_auc, 4))
print("TEST REPORT:\n", classification_report(y_test, y_test_pred))


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
import numpy as np


# Data
X = data_balanced['cleaned_text'].astype(str)
y = data_balanced['label_num']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Pipeline: TF-IDF -> Logistic Regression
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2)),
    ("logreg", LogisticRegression(max_iter=2000, solver="liblinear"))
])

# Hyperparam search
param_grid = {
    "logreg__C": [0.1, 0.5, 1, 2, 5],
    "logreg__class_weight": [None, "balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV ROC-AUC:", round(grid.best_score_, 4))

# Evaluate on test
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nTest ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob)

# Top features for class 1 (jailbreak)
tfidf = best_model.named_steps["tfidf"]
logreg = best_model.named_steps["logreg"]
feat_names = tfidf.get_feature_names_out()
coefs = logreg.coef_[0]
top_idx = np.argsort(coefs)[-20:]     # most positive -> predict 1
bot_idx = np.argsort(coefs)[:20]      # most negative -> predict 0

print("\nTop + features (→ jailbreak):")
for i in reversed(top_idx):
    print(f"{feat_names[i]} : {coefs[i]:.4f}")

print("\nTop - features (→ benign):")
for i in bot_idx:
    print(f"{feat_names[i]} : {coefs[i]:.4f}")

