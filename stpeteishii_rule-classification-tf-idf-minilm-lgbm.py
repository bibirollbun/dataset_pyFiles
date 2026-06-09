


!pip install lightgbm scikit-learn pandas sentence-transformers


# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix


# 2. Load data
train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
print(len(train),len(test))



# 3. Combine text columns into a single input text
def combine_text(row):
    return (
        f"Subreddit: {row['subreddit']} [SEP] "
        f"Rule: {row['rule']} [SEP] "
        f"Positive Example 1: {row['positive_example_1']} [SEP] "
        f"Positive Example 2: {row['positive_example_2']} [SEP] "
        f"Negative Example 1: {row['negative_example_1']} [SEP] "
        f"Negative Example 2: {row['negative_example_2']} [SEP] "
        f"Post: {row['body']}"
    )

train['input_text'] = train.apply(combine_text, axis=1)
test['input_text'] = test.apply(combine_text, axis=1)


# 4. TF-IDF feature extraction
tfidf = TfidfVectorizer(max_features=400)
tfidf_train = tfidf.fit_transform(train['input_text']).toarray()
tfidf_test = tfidf.transform(test['input_text']).toarray()


# 5. MiniLM embedding
model = SentenceTransformer("/kaggle/input/all-minilm-l6-v2/all-MiniLM-L6-v2")
embed_train = model.encode(train['input_text'].tolist(), show_progress_bar=True)
embed_test = model.encode(test['input_text'].tolist(), show_progress_bar=True)


# 6. Combine features
X_train = np.hstack([tfidf_train, embed_train])
X_test = np.hstack([tfidf_test, embed_test])
y_train = train['rule_violation'].values


# 7. Train LightGBM classifier
params = {
    'objective': 'binary',  
    'learning_rate': 0.1,
    'reg_lambda': 1.0,
    'reg_alpha': 0.1,
    'max_depth': 5,
    'n_estimators': 1000,
    'colsample_bytree': 0.5,
    'min_child_samples': 10,
    'subsample_freq': 3,
    'subsample': 0.9,
    'importance_type': 'gain',
    'random_state': 71,
    'num_leaves': 31, 
}
clf = lgb.LGBMClassifier(**params)
clf.fit(X_train, y_train)


# 8. Predict on train set (for validation)
y_pred_train = clf.predict(X_train)
print("Training Performance:")
print(confusion_matrix(y_train, y_pred_train))
print(classification_report(y_train, y_pred_train))


# 9. Predict on test set
test_pred_prob = clf.predict_proba(X_test)[:, 1]
test['rule_violation'] = (test_pred_prob > 0.5).astype(int)

# Save submission
test[['row_id', 'rule_violation']].to_csv("submission.csv", index=False)
print("submission.csv saved!")




