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


from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import joblib
import optuna
import shutil
import glob
import json
import gc

warnings.filterwarnings("ignore")



class CFG:
    train_path = "/kaggle/input/playground-series-s5e7/train.csv"
    test_path = "/kaggle/input/playground-series-s5e7/test.csv"
    sample_sub_path = "/kaggle/input/playground-series-s5e7/sample_submission.csv"

    original_path = "/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv"
    
    target = "Personality"
    n_folds = 5
    seed = 42



train = pd.read_csv(CFG.train_path)
test = pd.read_csv(CFG.test_path)

# original dataset with renaming and deduplication
original = (
    pd.read_csv(CFG.original_path)
    .rename(columns={
        'Personality': 'match_p'
    })
    .drop_duplicates([
        'Time_spent_Alone',
        'Stage_fear',
        'Social_event_attendance',
        'Going_outside',
        'Drained_after_socializing',
        'Friends_circle_size',
        'Post_frequency'
    ])
)

train.head(5)



train = train.merge(original, how='left')
test = test.merge(original, how='left')

train.head(5)



# Encode target
label_encoder = LabelEncoder()
train[CFG.target] = label_encoder.fit_transform(train[CFG.target])
# original[CFG.target] = label_encoder.transform(original[CFG.target])



X = train.drop(columns=[CFG.target, "id"])
y = train[CFG.target]
X_test = test.drop(columns=["id"])



# Encode categorical columns
combined = pd.concat([X, X_test], axis=0)
cat_cols = combined.select_dtypes(include="object").columns.tolist()

encoder = OrdinalEncoder()
combined[cat_cols] = encoder.fit_transform(combined[cat_cols])

X_len = len(X)
X_test_len = len(X_test)

X = combined.iloc[:X_len].reset_index(drop=True)
X_test = combined.iloc[X_len:].reset_index(drop=True)



class Trainer:
    def __init__(self, model, config=CFG):
        self.model = model
        self.config = config

    def fit_predict(self, X, y, X_test, X_original=None, y_original=None, fit_args={}):
        print(f"Training {self.model.__class__.__name__}\n")

        scores = []
        oof_pred_probs = np.zeros((X.shape[0], 2))
        test_pred_probs = np.zeros((X_test.shape[0], 2))

        skf = StratifiedKFold(n_splits=self.config.n_folds, random_state=self.config.seed, shuffle=True)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if X_original is not None and y_original is not None:
                X_train = pd.concat([X_train, X_original])
                y_train = pd.concat([y_train, y_original])

            model = clone(self.model)

            if fit_args:
                model.fit(X_train, y_train, **fit_args, eval_set=[(X_val, y_val)])
            else:
                model.fit(X_train, y_train)

            y_pred_probs = self._safe_predict_proba(model, X_val, y_train)
            oof_pred_probs[val_idx] = y_pred_probs

            temp_test_pred_probs = self._safe_predict_proba(model, X_test, y_train)
            test_pred_probs += temp_test_pred_probs / self.config.n_folds

            score = accuracy_score(y_val, np.argmax(y_pred_probs, axis=1))
            scores.append(score)

            del model, X_train, y_train, X_val, y_val, y_pred_probs
            gc.collect()

            print(f"--- Fold {fold_idx + 1} - Accuracy: {score:.6f}")

        overall_score = accuracy_score(y, np.argmax(oof_pred_probs, axis=1))
        print(f"\n------ Overall Accuracy: {overall_score:.6f} | Average Accuracy: {np.mean(scores):.6f} ± {np.std(scores):.6f}")

        return oof_pred_probs, test_pred_probs, scores

    def tune(self, X, y):
        scores = []
        skf = StratifiedKFold(n_splits=self.config.n_folds, random_state=self.config.seed, shuffle=True)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = clone(self.model)
            model.fit(X_train, y_train)

            y_pred_probs = self._safe_predict_proba(model, X_val, y_train)
            score = accuracy_score(y_val, np.argmax(y_pred_probs, axis=1))
            scores.append(score)

            del model, X_train, y_train, X_val, y_val, y_pred_probs
            gc.collect()

        return np.mean(scores)

    def _safe_predict_proba(self, model, X, y_train):
        unique_classes = np.unique(y_train)
        if len(unique_classes) == 1:
            single_class = unique_classes[0]
            preds = model.predict(X)
            probs = np.zeros((len(preds), 2))
            probs[:, single_class] = 1.0
            return probs
        else:
            return model.predict_proba(X)



# setup model
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

lgbm_goss_params = {
    "boosting_type": "goss",
    "device": "gpu",
    "colsample_bytree": 0.32751831793031183,
    "learning_rate": 0.006700715059604966,
    "max_depth": 12,
    "min_child_samples": 84,
    "n_estimators": 10000,
    "n_jobs": -1,
    "num_leaves": 229,
    "random_state": 42,
    "reg_alpha": 6.879977008084246,
    "reg_lambda": 4.739518466581721,
    "subsample": 0.5411572049978781,
    "verbose": -1
}

lgbm_params = {
    "boosting_type": "gbdt",
    "device": "gpu",
    "colsample_bytree": 0.4366677273946288,
    "learning_rate": 0.016164161953515117,
    "max_depth": 12,
    "min_child_samples": 67,
    "n_estimators": 10000,
    "n_jobs": -1,
    "num_leaves": 243,
    "random_state": 42,
    "reg_alpha": 6.38288560443373,
    "reg_lambda": 9.392999314379155,
    "subsample": 0.7989164499431718,
    "verbose": -1
}

rf_params = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_leaf": 16,
    "verbose": 0
}

hgb_params = {
    "learning_rate": 0.03,
    "min_samples_leaf": 12,
    "max_iter": 500,
    "max_depth": 5,
    "l2_regularization": 0.75,
}



scores = {}
oof_pred_probs = {}
test_pred_probs = {}



xgb_model = XGBClassifier(**xgb_params)
xgb_trainer = Trainer(xgb_model, config=CFG)

fit_args = {
    "verbose": 1000
}

oof_pred_probs["XGBoost"], test_pred_probs["XGBoost"], scores["XGBoost"] = xgb_trainer.fit_predict(X, y, X_test, fit_args)



lgbm_model = LGBMClassifier(**lgbm_params)
lgbm_trainer = Trainer(lgbm_model)

fit_args = {
    "callbacks": [
        log_evaluation(period=1000), 
        early_stopping(stopping_rounds=100)
    ]
}

oof_pred_probs["LightGBM (gbdt)"], test_pred_probs["LightGBM (gbdt)"], scores["LightGBM (gbdt)"] = lgbm_trainer.fit_predict(X, y, X_test, fit_args)



lgb_goss_model = LGBMClassifier(**lgbm_goss_params)
lgb_goss_trainer = Trainer(lgb_goss_model)

fit_args = {
    "callbacks": [
        log_evaluation(period=1000), 
        early_stopping(stopping_rounds=100)
    ]
}

oof_pred_probs["LightGBM (goss)"], test_pred_probs["LightGBM (goss)"], scores["LightGBM (goss)"] = lgb_goss_trainer.fit_predict(X, y, X_test, fit_args)



from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score
from sklearn.utils.parallel import Parallel, delayed
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, clone

X = pd.read_csv(CFG.train_path, index_col='id')
X_test = pd.read_csv(CFG.test_path, index_col='id')

le = LabelEncoder()
y = X.pop('Personality')
y = le.fit_transform(y)

X = X.astype(str)
X_test = X_test.astype(str)

def adjusted_mutual_info(x, y, n_iter=5):
    x, y = x.astype(str), y.astype(str)
    m0 = mutual_info_score(x, y)
    m1 = Parallel(n_jobs=-1)(
        delayed(lambda rs: mutual_info_score(
            y, np.random.default_rng(rs).permutation(x)
        ))(rs)
        for rs in range(n_iter)
    )
    return m0 - np.mean(m1)

mi = {}
e = mutual_info_score(y, y)
for c1, c2, c3 in tqdm(list(combinations(list(X.columns), 3))):
    c = c1+'|'+c2+'|'+c3
    mi[c] = adjusted_mutual_info(X[c1]+'|'+X[c2]+'|'+X[c3], y)/e

comb3 = sorted(mi, key=mi.get, reverse=True)

def Augmented(model, weight_arg, weight=1.0):
    class AugmentedModel(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            sample_weight = np.array([1.0]*len(X))
            self.m = clone(model).fit(X, y, **{weight_arg: sample_weight})
            self.classes_ = self.m.classes_
            return self
        def predict_proba(self, X):
            return self.m.predict_proba(X)
    return AugmentedModel()

X_o = None
y_o = None

X_all = pd.concat([X, X_o]).astype(str)

X_all_e = X_all.copy()
X_test_e = X_test.copy()
for c1, c2 in combinations(X_all.columns, 2):
    X_all_e[c1+'|'+c2] = X_all[c1]+'|'+X_all[c2]
    X_test_e[c1+'|'+c2] = X_test[c1]+'|'+X_test[c2]

topk = 25
for c1_c2_c3 in comb3[:topk]:
    c1, c2, c3 = c1_c2_c3.split('|')
    X_all_e[c1_c2_c3] = X_all[c1]+'|'+X_all[c2]+'|'+X_all[c3]
    X_test_e[c1_c2_c3] = X_test[c1]+'|'+X_test[c2]+'|'+X_test[c3]

X_e = X_all_e.iloc[:len(X)]
X_o_e = X_all_e.iloc[len(X):]



lr_model = Augmented(
    make_pipeline(
        OneHotEncoder(handle_unknown='ignore'),
        LogisticRegression(C=1e-2, max_iter=10000, random_state=0)
    ), 
    weight_arg='logisticregression__sample_weight', 
    weight=4.0
)

y = pd.DataFrame(y)
lr_trainer = Trainer(lr_model)
oof_pred_probs["LogisticRegression"], test_pred_probs["LogisticRegression"], scores["LogisticRegression"] = lr_trainer.fit_predict(X_e, y, X_test_e)



X = pd.DataFrame(np.concatenate(list(oof_pred_probs.values()), axis=1))
X_test = pd.DataFrame(np.concatenate(list(test_pred_probs.values()), axis=1))



joblib.dump(X, "oof_pred_probs.pkl")
joblib.dump(X_test, "test_pred_probs.pkl")



lr_model = LogisticRegression(
    random_state=42, 
    max_iter=1000, 
    solver='liblinear', 
    penalty='l2', 
    C=32.89802104596641,
    tol=0.0029878837974181643,
    fit_intercept=True
) 

lr_trainer = Trainer(lr_model)
_, lr_test_pred_probs, scores["Ensemble"] = lr_trainer.fit_predict(X, y, X_test)



final_predictions = []
for i in np.argsort(lr_test_pred_probs)[:, -1:][:, ::-1]:
    prediction = label_encoder.inverse_transform(i)
    final_predictions.append(" ".join(prediction))



sub = pd.read_csv(CFG.sample_sub_path)
sub[CFG.target] = final_predictions
sub.to_csv("submission.csv", index=False)
sub.head()





