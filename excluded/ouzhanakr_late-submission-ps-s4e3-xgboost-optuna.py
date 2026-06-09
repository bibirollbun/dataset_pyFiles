# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from numpy.random import default_rng
import optuna
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s4e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e3/test.csv')
s_s = pd.read_csv('/kaggle/input/playground-series-s4e3/sample_submission.csv')


pd.set_option('display.max_columns', None)
train.head()


train.info()


train.isnull().sum()


s_s.head()


train.drop(['id'],inplace=True,axis=1)
test.drop(['id'],inplace=True,axis=1)


target_columns = ['Pastry','Z_Scratch', 'K_Scatch',   'Stains',   'Dirtiness','Bumps','Other_Faults']


target_classes = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
targets_bin = train[target_classes]

train = train.drop(target_classes, axis="columns")


num_cols = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]

scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])




from sklearn.ensemble import RandomForestClassifier
def adversarial_validation(df_train,df_test):
    X_test  = df_test.select_dtypes(include=['number']).copy()
    X_train = df_train.select_dtypes(include=['number']).copy()

    drop_ = ['Pastry','Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults']

    X_train = X_train.drop(drop_, axis=1)
    print(X_train.shape)
    print(X_test.shape)
    X_train["Adv_Val_label"] = 0
    X_test["Adv_Val_label"]  = 1
    all_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    # shuffle
    all_data = all_data.sample(frac=1)
    forest = RandomForestClassifier(random_state=42,max_depth=2,class_weight='balanced')

    X = all_data.drop(['Adv_Val_label'], axis=1).fillna(-1)
    y = all_data['Adv_Val_label']

    clf = RandomForestClassifier(random_state=42).fit(X, y)
    from sklearn.metrics import roc_auc_score
    auc_score = roc_auc_score(y, clf.predict_proba(X)[:,1])
    print(auc_score)



adversarial_validation(targets_bin,test)


targets_bin


USE_GPU = False
def create_xgb_params(trial, seed):
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "verbosity": 0,
        **({"tree_method": "gpu_hist", "predictor": "gpu_predictor"} if USE_GPU else {}),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.1, 1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "random_state": seed,
        "use_label_encoder": False,
    }

def tune_for_label(X, y, seed=42, n_trials=5):
    def objective(trial):
        params = create_xgb_params(trial, seed)
        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

def train_oof_and_predict(X, y, test_X, params, seed=42):
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(test_X))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model_params = params.copy()
        model = xgb.XGBClassifier(**model_params)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )

        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(test_X)[:, 1] / n_splits

    score = roc_auc_score(y, oof)
    return oof, test_preds, score

final_test_preds = pd.DataFrame()
oof_scores = {}
n_trials = 5  

for target in target_columns:
    print(f"\n=== Processing target: {target} ===")
    y = targets_bin[target].reset_index(drop=True)
    X = train.reset_index(drop=True)
    X_test = test.reset_index(drop=True)

    
    best_params, best_cv = tune_for_label(X, y, seed=42, n_trials=n_trials)
    print(f"Optuna best CV AUC (estimate): {best_cv:.5f}")
    print("Best params:", best_params)

    oof, test_pred, score = train_oof_and_predict(X, y, X_test, best_params, seed=42)
    print(f"OOF ROC-AUC for {target}: {score:.5f}")

    oof_scores[target] = score
    final_test_preds[target] = test_pred







submission = s_s.copy()
for col in target_columns:
    submission[col] = final_test_preds[col].values

submission[target_columns] = submission[target_columns].clip(0, 1)

submission.to_csv("submission.csv", index=False)
print("\nAll done! Mean OOF ROC-AUC:", np.mean(list(oof_scores.values())))


ls



submission

