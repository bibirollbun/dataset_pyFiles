import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score


train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
test_labels = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')
sample_submission = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')


print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train[label_cols].sum().sort_values(ascending=False).plot(kind='bar', figsize=(8,5), color='tomato')
plt.title("Number of Comments per Toxic Class")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


train['label_sum'] = train[label_cols].sum(axis=1)

train['label_sum'].value_counts().sort_index().plot(kind='bar', color='steelblue')
plt.title("Multi-Label Distribution")
plt.xlabel("Number of Toxic Tags per Comment")
plt.ylabel("Number of Comments")
plt.show()


train['comment_length'] = train['comment_text'].str.len()

plt.figure(figsize=(8,5))
sns.histplot(train['comment_length'], bins=50, kde=True, color='orchid')
plt.title("Distribution of Comment Lengths")
plt.xlabel("Character Length")
plt.ylabel("Frequency")
plt.show()


for label in label_cols:
    print(f"\n\nðŸ“Œ Example of '{label.upper()}':\n")
    example = train[train[label] == 1]['comment_text'].iloc[0]
    print(example)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


train['clean_text'] = train['comment_text'].apply(clean_text)
test['clean_text'] = test['comment_text'].apply(clean_text)


train.head()


tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2),
    stop_words='english'
)


X_train = tfidf.fit_transform(train['clean_text'])
X_test = tfidf.transform(test['clean_text'])

print("TF-IDF matrix shape (train):", X_train.shape)
print("TF-IDF matrix shape (test):", X_test.shape)


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = train[label_cols]


model = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
model.fit(X_train, y_train)


train_preds = model.predict_proba(X_train)

for i, label in enumerate(label_cols):
    score = roc_auc_score(y_train[label], train_preds[:, i])
    print(f"{label}: ROC AUC = {score:.4f}")

mean_auc = roc_auc_score(y_train, train_preds, average='macro')
print(f"\nMean ROC AUC: {mean_auc:.4f}")


test_preds = model.predict_proba(X_test)  # shape: (test_samples, 6)

submission = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')

for i, label in enumerate(label_cols):
    submission[label] = test_preds[:, i]

submission.to_csv('submission.csv', index=False)




