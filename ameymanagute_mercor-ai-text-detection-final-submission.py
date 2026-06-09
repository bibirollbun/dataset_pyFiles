import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone


train_data=pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
test_data=pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')
sample_submission = pd.read_csv('/kaggle/input/mercor-ai-detection/sample_submission.csv')


train_data


test_data


def add_style_features(df):
    df_out = df.copy()
    s = df_out['answer'].fillna("").astype(str)
    
    df_out['answer_len'] = s.str.len()
    df_out['answer_word_count'] = s.str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
    df_out['answer_sentence_count'] = s.str.count(r'[.!?]') + 1
    df_out['answer_avg_word_len'] = df_out['answer_len'] / (df_out['answer_word_count'] + 1e-6)
    df_out['answer_avg_sent_len'] = df_out['answer_word_count'] / (df_out['answer_sentence_count'] + 1e-6)
    df_out['q_marks'] = s.str.count(r'\?')
    df_out['exclam'] = s.str.count(r'!')
    df_out['commas'] = s.str.count(r',')
    df_out['unique_word_ratio'] = s.str.lower().str.findall(r'\w+').apply(lambda words: len(set(words))/(len(words)+1e-6) if words else 0.0)
    
    return df_out


train_df = add_style_features(train_data)
test_df = add_style_features(test_data)


style_features = [
    'answer_len', 'answer_word_count', 'answer_sentence_count', 
    'answer_avg_word_len', 'answer_avg_sent_len', 'q_marks', 
    'exclam', 'commas', 'unique_word_ratio'
]


text_feature = 'answer'
topic_feature = ['topic']


preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=10000, stop_words='english'), text_feature), ('style', StandardScaler(), style_features),('topic', OneHotEncoder(handle_unknown='ignore'), topic_feature)],remainder='drop',n_jobs=-1)


model = lgb.LGBMClassifier(
    random_state=42,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    n_jobs=-1,
    class_weight='balanced'
)


full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])


X = train_df
y = train_df['is_cheating']
X_test_processed = test_df


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds_folds = []


for fold, (train_index, val_index) in enumerate(skf.split(X, y), start=1):
    print(f"--- Fold {fold}/5 ---")

    fold_pipeline = clone(full_pipeline)
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Fit the *cloned* pipeline
    fold_pipeline.fit(X_train, y_train)
    
    # Predict with the *cloned* pipeline
    val_probs = fold_pipeline.predict_proba(X_val)[:, 1]
    oof_preds[val_index] = val_probs
    
    fold_test_probs = fold_pipeline.predict_proba(X_test_processed)[:, 1]
    test_preds_folds.append(fold_test_probs)
    
    print(f"Fold {fold} AUC: {roc_auc_score(y_val, val_probs):.5f}")


oof_auc = roc_auc_score(y, oof_preds)
print(f"Overall OOF ROC-AUC: {oof_auc:.5f}")


final_test_pred = np.mean(test_preds_folds, axis=0)


sample_submission = pd.read_csv('/kaggle/input/mercor-ai-detection/sample_submission.csv')
sample_submission['is_cheating'] = final_test_pred
sample_submission.to_csv('submission.csv', index=False)

