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


%%time
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


%%time
import pandas as pd

# Load data
train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sample_sub = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)   


%%time
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(data=train, x='rule_violation', palette='Set2')
plt.title('Class Distribution (rule_violation)')
plt.xlabel('Violation (1 = Yes, 0 = No)')
plt.ylabel('Count')
plt.show()

print("Class distribution (%):\n", train['rule_violation'].value_counts(normalize=True) * 100)   


%%time
print("Missing in train:\n", train.isnull().sum())
print("\nMissing in test:\n", test.isnull().sum())  


%%time
cols = ['body', 'rule', 'rule_violation', 'subreddit']
print("Sample training examples:\n", train[cols].head(3).to_string())   


%%time
import re

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.strip().lower())

train['cleaned_text'] = train['body'].apply(clean_text)
test['cleaned_text'] = test['body'].apply(clean_text)   


%%time
train['input'] = "Rule: " + train['rule'] + " Comment: " + train['cleaned_text']
test['input'] = "Rule: " + test['rule'] + " Comment: " + test['cleaned_text']   


%%time
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Download and save model and tokenizer
model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("/kaggle/working/local_tokenizer/")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.save_pretrained("/kaggle/working/local_model/")   
"""


%%time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(train['input'], train['rule_violation'], test_size=0.2, random_state=42)

# Vectorizer
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
val_preds = model.predict_proba(X_val_vec)[:, 1]
print("Validation AUC:", roc_auc_score(y_val, val_preds))   


%%time
from datasets import Dataset

# Use the 'input' column we created earlier (rule + comment)
train_texts = train['input'].tolist()
train_labels = train['rule_violation'].astype(int).tolist()

# Split into train and validation
train_texts_split, val_texts_split, train_labels_split, val_labels_split = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)

# Create Hugging Face Dataset objects
train_dataset = Dataset.from_dict({"text": train_texts_split, "label": train_labels_split})
val_dataset = Dataset.from_dict({"text": val_texts_split, "label": val_labels_split})

print("Train/val datasets created.")   


%%time
import matplotlib.pyplot as plt
import seaborn as sns

# Rule-wise violation rate
rule_violations = train.groupby("rule")["rule_violation"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=rule_violations.values, y=rule_violations.index)
plt.title("Violation Rate by Rule")
plt.xlabel("Violation Rate")
plt.ylabel("Rule")
plt.show()

# Subreddit-wise violation rate
subreddit_violations = train.groupby("subreddit")["rule_violation"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=subreddit_violations.values, y=subreddit_violations.index)
plt.title("Violation Rate by Subreddit")
plt.xlabel("Violation Rate")
plt.ylabel("Subreddit")
plt.show()   


%%time
import matplotlib.pyplot as plt
import seaborn as sns

# Rule-wise violation rate (unchanged)
rule_violations = train.groupby("rule")["rule_violation"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=rule_violations.values, y=rule_violations.index)
plt.title("Violation Rate by Rule")
plt.xlabel("Violation Rate")
plt.ylabel("Rule")
plt.show()

# Subreddit-wise violation rate (updated to show top N only)
subreddit_violations = train.groupby("subreddit")["rule_violation"].mean().sort_values(ascending=False)

# Limit to top 10 subreddits for readability
top_n = 10
subreddit_violations_top = subreddit_violations.head(top_n)

plt.figure(figsize=(10, 6))
sns.barplot(x=subreddit_violations_top.values, y=subreddit_violations_top.index)
plt.title(f"Violation Rate by Subreddit (Top {top_n})")
plt.xlabel("Violation Rate")
plt.ylabel("Subreddit")
plt.show()   


%%time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Step 1: Extract features and target
X_train = train['body']                    # ✅ Use 'body' as input
y_train = train['rule_violation']          # ✅ Use 'rule_violation' as label
X_test = test['body']                      # ✅ Test input

# Step 2: Build pipeline (TF-IDF + Logistic Regression)
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words='english',
        lowercase=True,
        strip_accents='unicode'
    )),
    ('lr', LogisticRegression(C=1.0, max_iter=500))
])

# Step 3: Train
model.fit(X_train, y_train)

# Step 4: Predict probabilities (required for ROC AUC)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1

# Step 5: Create submission
submission = pd.DataFrame({
    'row_id': test['row_id'],
    'rule_violation': y_pred_proba   # ✅ Must be 'rule_violation', not 'prediction'
})

# Save and preview
submission.to_csv('submission.csv', index=False)
print("✅ Submission file created!")
print("\nSubmission head:")
print(submission.head())
print("\nPrediction range:", submission['rule_violation'].min(), "→", submission['rule_violation'].max())   

