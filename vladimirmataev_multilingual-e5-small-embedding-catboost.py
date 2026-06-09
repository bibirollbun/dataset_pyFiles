import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import os
import string
import optuna
import gc
import torch

import sklearn
import sentence_transformers

from sentence_transformers import SentenceTransformer

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
from xgboost import XGBClassifier
import catboost as catb
from catboost import CatBoostClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")


RANDOM_STATE = 42


train_idx_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"
train_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
test_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"


def read_data(data_path: str) -> pd.DataFrame:
    data = []
    for dir in os.listdir(data_path):
        dir_path = os.path.join(data_path, dir)
        file_1_path = os.path.join(dir_path, "file_1.txt")
        file_2_path = os.path.join(dir_path, "file_2.txt")
        
        try:
            with open(file_1_path, "r", encoding="utf-8") as f:
                text1 = f.read().strip()
            with open(file_2_path, "r", encoding="utf-8") as f:
                text2 = f.read().strip()
            
            idx = int(dir_path[-4:])
            data.append((idx, text1, text2))        
        except Exception as e:
            print(f"Error reading directory {dir}: {e}")
        
    df = pd.DataFrame(data, columns=["id", "file1", "file2"])
    df = df.sort_values("id").reset_index(drop=True)
    
    return df


def clean_text(text: str) -> str:
    clean_punc = str.maketrans("", "", string.punctuation + "\n\t\r")
    clean_digit = str.maketrans("", "", string.digits)
    
    cleaned_text = text.translate(clean_punc)
    cleaned_text = cleaned_text.translate(clean_digit)
    cleaned_text = cleaned_text.lower()
    
    return cleaned_text


def clean_data(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = data.copy()
    
    for col in cols:
        df[col] = df[col].apply(lambda row: clean_text(row))
    
    return df
    
    
def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    
    real_text = []
    fake_text = []
    
    for _, row in data.iterrows():
        if row["real_text_id"] == 1:
            real_text.append(row["file1"])
            fake_text.append(row["file2"])
        else:
            real_text.append(row["file2"])
            fake_text.append(row["file1"])
    
    real_df = pd.DataFrame({"text": real_text, "label": 1})
    fake_df = pd.DataFrame({"text": fake_text, "label": 0})
    
    df = pd.concat([real_df, fake_df]).sample(frac=1, random_state=RANDOM_STATE, ignore_index=True)
    
    return df


def get_embeddings(texts: np.ndarray, embedding_model) -> np.ndarray:
    embeddings = embedding_model.encode(texts,
                                        batch_size=64,
                                        show_progress_bar=True,
                                        convert_to_numpy=True,
                                        normalize_embeddings=True)
    return embeddings

def add_features(texts):
    features = defaultdict(list)
    
    for text in texts:
        features["count_chars"].append(len(text))
        features["count_words"].append(len(text.split()))
        features["count_unique_words"].append(len(set(text.split())))
    
    return np.vstack(list(features.values())).T

def cross_val_skf(model,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 n_splits: int=5,
                 shuffle: bool=True,
                 random_state: int=None) -> list:
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    scores = []
    
    for i, (tr_idx, vl_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_vl = X_train[tr_idx], X_train[vl_idx]
        y_tr, y_vl = y_train[tr_idx], y_train[vl_idx]
        
        model.fit(X_tr,
              y_tr,
              eval_set=[(X_vl, y_vl)],
              use_best_model=True,
              verbose=False)
        
        y_pred = model.predict(X_vl)
            
        acc = accuracy_score(y_vl, y_pred)
        scores.append(acc)
        
        print(f"FOLD {i} | Accuracy: {acc}")
        
        
    print(f"\nAverage accuracy: {np.mean(scores):.6f}\n")
    
    return scores


def oof_prediction(model,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test_file1: np.ndarray,
                   X_test_file2: np.ndarray,
                   n_splits: int=5,
                   shuffle: bool=True,
                   random_state: int=None) -> dict:
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    oof_train = np.zeros(X_train.shape[0])
    oof_test = np.zeros((X_test_file1.shape[0], 2))
    
    for tr_idx, vl_idx in skf.split(X_train, y_train):
        X_tr, X_vl = X_train[tr_idx], X_train[vl_idx]
        y_tr, y_vl = y_train[tr_idx], y_train[vl_idx]
        

        model.fit(X_tr,
              y_tr,
              eval_set=[(X_vl, y_vl)],
              use_best_model=True,
              verbose=False)
        
        oof_train[vl_idx] = model.predict_proba(X_vl)[:, 1]
        oof_test[:, 0] += model.predict_proba(X_test_file1)[:, 1] / n_splits
        oof_test[:, 1] += model.predict_proba(X_test_file2)[:, 1] / n_splits
            
            
    return {"train": oof_train,
            "test": oof_test}


train_idx = pd.read_csv(train_idx_path)
train = read_data(train_path).set_index("id")
test_df = read_data(test_path).set_index("id")
train_df = train.merge(train_idx, how="left", on="id").set_index("id")


display(train_idx)
display(train)
display(test_df)
display(train_df)


FEATURES = ["file1", "file2"]

train = clean_data(train, FEATURES)
train_df = clean_data(train_df, FEATURES)
test_df = clean_data(test_df, FEATURES)

prepare_train = prepare_data(train_df)


embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")


torch.manual_seed(RANDOM_STATE)

embed_train = get_embeddings(prepare_train["text"].values, embedding_model)
new_features_train = add_features(prepare_train["text"])

embed_test_file1 = get_embeddings(test_df["file1"].values, embedding_model)
embed_test_file2 = get_embeddings(test_df["file2"].values, embedding_model)
new_features_test_file1 = add_features(test_df["file1"])
new_features_test_file2 = add_features(test_df["file2"])

X_train = np.hstack([embed_train, new_features_train])
y_train = prepare_train["label"]
X_test_file1= np.hstack([embed_test_file1, new_features_test_file1])
X_test_file2 = np.hstack([embed_test_file2, new_features_test_file2])

X_train.shape, y_train.shape, X_test_file1.shape, X_test_file2.shape


catb_params = {"loss_function": "Logloss",
                "eval_metric": "Logloss",
                'iterations': 469,
                'depth': 3,
                'learning_rate': 0.024564140562987967,
                'reg_lambda': 0.2991408652582348,
                "task_type": "GPU",
                "early_stopping_rounds": 50,
                "random_state": RANDOM_STATE}



model = CatBoostClassifier(**catb_params)
scores_model = []
oof_preds = []

scores_model = cross_val_skf(clone(model),
                             X_train,
                             y_train,
                             random_state=RANDOM_STATE)

oof_preds = oof_prediction(model,
                           X_train,
                           y_train,
                           X_test_file1,
                           X_test_file2,
                           random_state=RANDOM_STATE)


y_pred = [1 if p[0] > p[1] else 2 for p in oof_preds["test"]]
pred_test = pd.DataFrame({"id": range(len(test_df)),
                          "real_text_id": y_pred})
pred_test.to_csv("catb.csv", index=False)
pred_test.head()

