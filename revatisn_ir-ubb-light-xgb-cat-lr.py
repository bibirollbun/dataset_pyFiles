import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import re
import warnings
warnings.filterwarnings('ignore')


import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/ir-ubb-classification/train_binary.csv")
test = pd.read_csv("/kaggle/input/ir-ubb-classification/test_binary.csv")
sample = pd.read_csv("/kaggle/input/ir-ubb-classification/sample_submission_binary.csv")


train


test


sample #SUBMISSION


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train['cleaned_text'] = train['Lyric'].apply(clean_text)
test['cleaned_text'] = test['Lyric'].apply(clean_text)


tfidf_word = TfidfVectorizer(max_features=5000, ngram_range=(1,2), 
                              min_df=2, max_df=0.9)
tfidf_char = TfidfVectorizer(max_features=3000, analyzer='char', 
                              ngram_range=(2,4))

X_train_word = tfidf_word.fit_transform(train['cleaned_text'])
X_test_word = tfidf_word.transform(test['cleaned_text'])

X_train_char = tfidf_char.fit_transform(train['cleaned_text'])
X_test_char = tfidf_char.transform(test['cleaned_text'])

# Label encoding
le = LabelEncoder()
y_train = le.fit_transform(train['Genre'])


def train_lgb(X_train, y_train, X_test, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros((X_train.shape[0], len(np.unique(y_train))))
    test_preds = np.zeros((X_test.shape[0], len(np.unique(y_train))))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f'LightGBM Fold {fold+1}/{n_splits}')
        
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y_train)),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        
        train_data = lgb.Dataset(X_train[train_idx], label=y_train[train_idx])
        val_data = lgb.Dataset(X_train[val_idx], label=y_train[val_idx])
        
        model = lgb.train(params, train_data, num_boost_round=1000,
                         valid_sets=[val_data], 
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        
        oof_preds[val_idx] = model.predict(X_train[val_idx])
        test_preds += model.predict(X_test) / n_splits
    
    return oof_preds, test_preds



def train_xgb(X_train, y_train, X_test, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros((X_train.shape[0], len(np.unique(y_train))))
    test_preds = np.zeros((X_test.shape[0], len(np.unique(y_train))))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f'XGBoost Fold {fold+1}/{n_splits}')
        
        params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y_train)),
            'eval_metric': 'mlogloss',
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'tree_method': 'hist'
        }
        
        model = xgb.XGBClassifier(**params, n_estimators=1000, early_stopping_rounds=50)
        model.fit(X_train[train_idx], y_train[train_idx],
                 eval_set=[(X_train[val_idx], y_train[val_idx])],
                 verbose=100)
        
        oof_preds[val_idx] = model.predict_proba(X_train[val_idx])
        test_preds += model.predict_proba(X_test) / n_splits
    
    return oof_preds, test_preds



def train_catboost(X_train, y_train, X_test, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros((X_train.shape[0], len(np.unique(y_train))))
    test_preds = np.zeros((X_test.shape[0], len(np.unique(y_train))))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f'CatBoost Fold {fold+1}/{n_splits}')
        
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='MultiClass',
            eval_metric='MultiClass',
            random_seed=42,
            early_stopping_rounds=50,
            verbose=100
        )
        
        model.fit(X_train[train_idx], y_train[train_idx],
                 eval_set=(X_train[val_idx], y_train[val_idx]))
        
        oof_preds[val_idx] = model.predict_proba(X_train[val_idx])
        test_preds += model.predict_proba(X_test) / n_splits
    
    return oof_preds, test_preds


def train_logreg(X_train, y_train, X_test, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros((X_train.shape[0], len(np.unique(y_train))))
    test_preds = np.zeros((X_test.shape[0], len(np.unique(y_train))))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f'LogReg Fold {fold+1}/{n_splits}')
        
        model = LogisticRegression(max_iter=1000, C=1.0, random_state=42, 
                                   solver='saga', multi_class='multinomial')
        model.fit(X_train[train_idx], y_train[train_idx])
        
        oof_preds[val_idx] = model.predict_proba(X_train[val_idx])
        test_preds += model.predict_proba(X_test) / n_splits
    
    return oof_preds, test_preds


# Training with Word TF-IDF features
lgb_oof_word, lgb_test_word = train_lgb(X_train_word, y_train, X_test_word)
xgb_oof_word, xgb_test_word = train_xgb(X_train_word, y_train, X_test_word)
cat_oof_word, cat_test_word = train_catboost(X_train_word, y_train, X_test_word)
lr_oof_word, lr_test_word = train_logreg(X_train_word, y_train, X_test_word)


# Training with Char TF-IDF featureS
lgb_oof_char, lgb_test_char = train_lgb(X_train_char, y_train, X_test_char)
lr_oof_char, lr_test_char = train_logreg(X_train_char, y_train, X_test_char)


weights = [0.15, 0.20, 0.20, 0.10, 0.12, 0.23]

ensemble_test = (
    weights[0] * lgb_test_word +
    weights[1] * xgb_test_word +
    weights[2] * cat_test_word +
    weights[3] * lr_test_word +
    weights[4] * lgb_test_char +
    weights[5] * lr_test_char
)


final_predictions = le.inverse_transform(np.argmax(ensemble_test, axis=1))

submission = pd.DataFrame({
    'Id': test['Id'],
    'Genre': final_predictions
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created!")
print(submission.head())




