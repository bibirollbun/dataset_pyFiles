import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from collections import Counter

import warnings
warnings.simplefilter('ignore')

SEED = 30


df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


df_train.head()


def mapk(actual, predicted, k=3):
    total_score = 0.0
    actual = le.inverse_transform(actual)
    for a, p in zip(actual, predicted):
        if a in p[:k]:
            index = p.index(a)
            total_score += 1.0 / (index + 1)
    return total_score / len(actual)


le = LabelEncoder()
le.fit(df_train['Fertilizer Name'])

def make_features(df, test=False, original=False):
    df_temp = df.copy()
    if not original:
        df_temp.drop(columns=['id'], inplace=True)
    cat_cols = df_temp.select_dtypes(include=['object']).columns
    df_temp[cat_cols] = df_temp[cat_cols].astype('category')

    # adding binning of numerical features
    numerical_features = [col for col in df_temp.select_dtypes(include=['int64', 'float64']).columns]
    for col in numerical_features:
        df_temp[f'{col}_Binned'] = df_temp[col].astype(str).astype('category')

    if not test:
        df_temp['Fertilizer Name'] = le.transform(df_temp['Fertilizer Name'])
    
    return df_temp


df_train1 = make_features(df_train)
df_original1 = make_features(df_original, original=True)
df_test1 = make_features(df_test, test=True)


X = df_train1.drop(columns=['Fertilizer Name'])
y = df_train1['Fertilizer Name']

X_original = df_original1.drop(columns=['Fertilizer Name'])
y_original = df_original1['Fertilizer Name']


NUM_CLASSES = 7

def cross_val(X, y, params, K=10, debug=True, predict_test=False, weight_original=7, X_original=X_original, y_original=y_original):
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    fold_scores = []

    if predict_test:
        test_pred = np.zeros((len(df_test), NUM_CLASSES))

    X_original_copy = X_original.copy()
    y_original_copy = y_original.copy()

    for i in range(weight_original):
        X_original = pd.concat([X_original, X_original_copy])
        y_original = pd.concat([y_original, y_original_copy])

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_train_fold = pd.concat([X.iloc[train_idx], X_original])
        y_train_fold = pd.concat([y.iloc[train_idx], y_original])
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )

        val_probs = model.predict_proba(X_val_fold)
        val_top3 = np.argsort(val_probs, axis=1)[:, -3:][:, ::-1]

        if predict_test:
            test_probs_fold = model.predict_proba(df_test1)
            test_pred += test_probs_fold

        val_pred_labels = [[le.classes_[j] for j in row] for row in val_top3]
        score = mapk(y_val_fold, val_pred_labels)
        fold_scores.append(score)

        if debug:
            print(f'Fold {fold} MAP@3: {score:.4f}')

    avg_score = np.mean(fold_scores)
    if debug:
        print(f'========== Average Validation Score: {avg_score:.4f} ==========')

    if predict_test:
        test_pred /= K
        test_pred = np.argsort(test_pred, axis=1)[:, -3:][:, ::-1]
        test_pred = [[le.classes_[j] for j in row] for row in test_pred]
        test_pred = [' '.join(row) for row in test_pred]
        return avg_score, test_pred

    return avg_score


import optuna


def objective(trial):
    params = {
        "device": 'cuda',
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, step=0.01),
        "max_depth": trial.suggest_int("max_depth", 3, 13),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, step=0.1),
        "max_bin": trial.suggest_int("max_bin", 256, 2048),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 0.1, step=0.01),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "n_estimators": 3000,
        "enable_categorical": True,
        "early_stopping_rounds":50,
        "random_state": SEED
    }
    weight_original = trial.suggest_int('weight_original', 1.0, 10.0)

    score = cross_val(X, y, params=params, debug=False, weight_original=weight_original, K=5)
    return score


# %%time
# study = optuna.create_study(direction='maximize',
#                             sampler = optuna.samplers.RandomSampler(seed=SEED),
#                             study_name = "BIG BLUE FIN TUNA!!")
# study.optimize(objective, n_trials=50, show_progress_bar=True)


# best_params = study.best_params
# print(f'Best Trial Params: {best_params}')

# print(f'Best Trial Value: {study.best_trial.value}')


# # # These parameters are from 33 Rounds of Optuna only
best_params = {'learning_rate': 0.05,
               'max_depth': 7,
               'subsample': 0.8,
               'colsample_bytree': 0.4,
               'max_bin': 1323,
               'min_child_weight': 4,
               'gamma': 0.03,
               'lambda': 0.00953514350973168,
               'alpha': 0.6191568184269528,
               'max_delta_step': 3,
               }
weight_original = 3


best_params["n_estimators"] =  3000
best_params["enable_categorical"] = True
best_params["early_stopping_rounds"] = 50
best_params["random_state"] = SEED
best_params["device"] = 'cuda'


# best param cross validation
print('========== XGB BEST PARAMS CROSS VALIDATION ==========')
avg_score, test_pred = cross_val(X, y, params=best_params, K=10, predict_test=True, weight_original=3)


submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission['Fertilizer Name'] = test_pred
submission.to_csv('submission.csv', index=False)
submission.head()

