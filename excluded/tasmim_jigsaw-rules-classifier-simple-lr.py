import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
# 1. Load data
def load_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    sample = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    return train, test, sample

# 2. Preprocess: here, simple filling missing bodies and combining rule text

def preprocess(df):
    df['body'] = df['body'].fillna('')
    df['rule_text'] = df['rule'].fillna('').astype(str)
    # Combine comment and rule description for modeling
    df['text'] = df['rule_text'] + ' [SEP] ' + df['body']
    return df

# 3. Build pipeline: TF-IDF + LR
def build_pipeline():
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=50000,
            ngram_range=(1,2),
            stop_words='english'
        )),
        ('clf', LogisticRegression(
            solver='saga',
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            random_state=42
        ))
    ])
    return pipe

# 4. Train and evaluate

def train_and_evaluate(train, pipeline):
    X = train['text']
    y = train['rule_violation']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    val_preds = pipeline.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    print(f'Validation AUC: {auc:.4f}')
    return pipeline

# 5. Predict on test and prepare submission

def predict_and_submit(pipeline, test, sample, output_path='submission.csv'):
    test = preprocess(test)
    preds = pipeline.predict_proba(test['text'])[:, 1]
    sample['rule_violation'] = preds
    sample.to_csv(output_path, index=False)
    print(f'Submission saved to {output_path}')


# 6. Main
data_dir = '/kaggle/input/jigsaw-agile-community-rules'  # adjust to your data path
train, test, sample = load_data(data_dir)
train = preprocess(train)
pipeline = build_pipeline()
pipeline = train_and_evaluate(train, pipeline)
predict_and_submit(pipeline, test, sample)


submission = pd.read_csv('/kaggle/working/submission.csv')
submission.head()




