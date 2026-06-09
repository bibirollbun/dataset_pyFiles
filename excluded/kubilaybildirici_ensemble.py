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


import pandas as pd
train_data = pd.read_csv("/kaggle/input/train-data/train.csv")
test_data = pd.read_csv("/kaggle/input/test-dataset/test.csv")
train_data.head()


import re
import string


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub('\s+', ' ', text)
    return text


for df in [train_data, test_data]:
    df["body_clean"] = df["body"].apply(clean_text)
    df["rule_clean"] = df["rule"].apply(clean_text)
    df["combined"] = df["body_clean"] + " " + df["rule_clean"] + " " + df["subreddit"].str.lower()


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1,3), stop_words="english")
X_tfidf = tfidf.fit_transform(train_data["combined"])
X_test_tfidf = tfidf.transform(test_data["combined"])

tfidf_feature_names = tfidf.get_feature_names_out()
X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_feature_names)
X_test_tfidf = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_feature_names)


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_cos_sim(df):
    body_vec = tfidf.transform(df["body_clean"])
    rule_vec = tfidf.transform(df["rule_clean"])
    return np.array([cosine_similarity(body_vec[i], rule_vec[i])[0][0] for i in range(len(df))]).reshape(-1, 1)

cos_sim_train = get_cos_sim(train_data)
cos_sim_test = get_cos_sim(test_data)

cos_sim_train_df = pd.DataFrame(cos_sim_train, columns=["cos_sim"])
cos_sim_test_df = pd.DataFrame(cos_sim_test, columns=["cos_sim"])


def extract_features(df):
    return pd.DataFrame({
        "body_len": df["body"].apply(len),
        "word_count": df["body"].apply(lambda x: len(x.split())),
        "has_url": df["body"].str.contains("http").astype(int),
        "caps_ratio": df["body"].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1e-5)),
        "has_question": df["body"].str.contains("\?").astype(int),
        "starts_with_you": df["body"].str.lower().str.startswith("you").astype(int),
    })


meta_train = extract_features(train_data)
meta_test = extract_features(test_data)


X_train = pd.concat([
    X_tfidf_df.reset_index(drop=True),
    meta_train.reset_index(drop=True),
    cos_sim_train_df.reset_index(drop=True)
], axis=1)


X_test = pd.concat([
    X_test_tfidf.reset_index(drop=True),
    meta_test.reset_index(drop=True),
    cos_sim_test_df.reset_index(drop=True)
], axis=1)


y = train_data["rule_violation"]



import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score
import pickle
import numpy as np
#from tensorflow.python.ops.losses.losses_impl import log_loss


def lightGBM_model(X_train, y,X_test):
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42) # etiketi dengeli dagitarak bolme yapar

    test_preds = np.zeros(X_test.shape[0]) # test verisi icin tahminlerin ortalamasi alinacak
    oof_preds = np.zeros(X_train.shape[0]) # her fold daki validation tahminleri buraya yazilir
    #auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)): # 5 kez donucek dongu
        print(f"\nFold {fold + 1}") # hangi fold isleniyor ?

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            metric='auc',
            learning_rate=0.01,
            n_estimators=10000,
            num_leaves=300,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=5,
            min_child_samples=5,
            colsample_bytree=0.8,
            subsample=0.9,
            is_unbalance=True,
            random_state=42,
            verbose=-1
        )

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[early_stopping(100), log_evaluation(100)])

        val_preds = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
        oof_preds[val_idx] = val_preds


        print(f"Fold {fold + 1} AUC: {roc_auc_score(y_val, val_preds):.4f}")

    print(f"\n Overall AUC: {roc_auc_score(y, oof_preds):.4f}")
    #print(f"Overall logLoss: {log_loss(y, oof_preds):.4f}")

    try:
        with open('lgbm_model_extra_futures.pkl', 'wb') as f:
            pickle.dump(model, f)
            print("Model saved successfully")
    except:
        print("Error saving model")

    return oof_preds,test_preds


oof_preds, test_preds = lightGBM_model(X_train, y, X_test)



import xgboost as xgb
print("Before:", xgb.get_config())      
xgb.set_config(verbosity=0)             
print("After:", xgb.get_config())


from xgboost import XGBClassifier
def XGBoost_model(X_train,y, X_test):
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42) # 5 kez donucek dongu

    test_preds = np.zeros(X_test.shape[0])
    oof_preds = np.zeros(X_train.shape[0])

    for fold, (train_idx,val_idx) in enumerate(skf.split(X_train,y)):
        print(f"\nFold {fold + 1}")

        X_tr,X_val = X_train.iloc[train_idx],X_train.iloc[val_idx]
        y_tr,y_val = y.iloc[train_idx],y.iloc[val_idx]

        model = XGBClassifier(
            objective='binary:logistic', # binary classification
            n_estimators=10000, # total tree
            max_depth=10, # max dept
            eval_metric="auc",
            learning_rate=0.01,
            colsample_bytree=0.9,
            subsample=0.9,
            random_state=42,
            use_label_encoder=False,
            verbosity=0,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1,
            #tree_method='gpu_hist',
            early_stopping_rounds=100
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        val_preds = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
        oof_preds[val_idx] = val_preds


        print(f"Fold {fold + 1} AUC: {roc_auc_score(y_val, val_preds):.4f}")

    print(f"\n Overall AUC: {roc_auc_score(y, oof_preds):.4f}")
    #print(f"Overall logLoss: {log_loss(y, oof_preds):.4f}")

    try:
        with open('xgboost_model_extra_features.pkl', 'wb') as f:
            pickle.dump(model, f)
            print("Model saved successfully")
    except:
        print("Error saving model")

    return oof_preds,test_preds


oof_preds, test_preds = XGBoost_model(X_train, y, X_test)



from catboost import CatBoostClassifier

def CatBoostClassifier_Model(X_train,y,X_test):
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    test_preds = np.zeros(X_test.shape[0])
    oof_preds = np.zeros(X_train.shape[0])

    for fold, (train_idx,val_idx) in enumerate(skf.split(X_train,y)):
        print(f"\nFold {fold + 1}")

        X_tr,X_val = X_train.iloc[train_idx],X_train.iloc[val_idx]
        y_tr,y_val = y.iloc[train_idx],y.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=8,
            loss_function="Logloss",
            eval_metric="AUC",
            early_stopping_rounds=100,
            random_seed=42,
            verbose=100
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        val_preds = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
        oof_preds[val_idx] = val_preds

        print(f"Fold {fold + 1} AUC: {roc_auc_score(y_val, val_preds):.4f}")

    print(f"\n Overall AUC: {roc_auc_score(y, oof_preds):.4f}")
    #print(f"Overall logLoss: {log_loss(y, oof_preds):.4f}")

    try:
        with open('CatBoostClassifier.pkl', 'wb') as f:
            pickle.dump(model, f)
            print("Model saved successfully")
    except:
        print("Error saving model")

    return oof_preds, test_preds


oof_preds, test_preds = CatBoostClassifier_Model(X_train, y, X_test)


def ensemble_model(X_train,y,X_test, w=(1/3,1/3,1/3)):
    with open("/kaggle/working/lgbm_model_extra_futures.pkl","rb") as f:
        lgbm = pickle.load(f)

    with open("/kaggle/working/xgboost_model_extra_features.pkl","rb") as f:
        xgboost = pickle.load(f)

    with open("/kaggle/working/CatBoostClassifier.pkl","rb") as f:
        catboost = pickle.load(f)

    p_lgbm = lgbm.predict_proba(X_test)[:,1]
    p_xgboost = xgboost.predict_proba(X_test)[:,1]
    p_catboost = catboost.predict_proba(X_test)[:,1]

    w = np.array(w,dtype=float); w = w / w.sum()
    test_preds = w[0]*p_lgbm + w[1]*p_xgboost + w[2]*p_catboost
    print(test_preds)

    oof_preds = None
    return oof_preds,test_preds


oof_preds, ensemble_test_preds = ensemble_model(X_train, y, X_test)


ensemble_test_preds


sample = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")
sample


sample["rule_violation"] = ensemble_test_preds
sample


sample.to_csv("submission.csv", index=False)
print('submission.csv')

