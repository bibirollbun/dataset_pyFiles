import pandas as pd
train=pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test=pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sample=pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')


train


test


sample


import re
import html
import emoji
import string

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Unescape HTML
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove Reddit usernames and subreddit mentions
    text = re.sub(r'u\/\w+|r\/\w+', '', text)

    # Remove Markdown links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove special characters (but keep basic punctuation)
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"-]', '', text)

    # Replace emoji with text (optional)
    text = emoji.demojize(text)

    # Lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv") 
df['clean_text'] = df['body'].apply(clean_text)

# Check label distribution
print(df['rule_violation'].value_counts()) 


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# TF-IDF
tfidf = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=5,
    max_df=0.9,
    max_features=30000
)

X = tfidf.fit_transform(df['clean_text'])
y = df['rule_violation']

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
svm = LinearSVC(C=1.0, max_iter=2000)
svm.fit(X_train, y_train)

# Evaluate
y_pred = svm.predict(X_val)
print(classification_report(y_val, y_pred))


test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
test_df['clean_text'] = test_df['body'].apply(clean_text)
X_test = tfidf.transform(test_df['clean_text'])
test_preds = svm.predict(X_test)

submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': test_preds
})
submission.to_csv("submission.csv", index=False)

