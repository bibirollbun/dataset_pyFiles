pip install --upgrade xgboost lightgbm scikit-learn


import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score as AUC
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
target = df.columns.tolist()[-1]
print(df.shape)
df.head()


def create_frequency_features(df, df_test):
    """
    Add frequency and binning features efficiently.

    - For each categorical column, create <col>_freq = how often each value appears in train data.
    - For numeric columns, split values into 5, 10, 15 quantile bins.
    """
    # Pre-allocate DataFrames for new features to avoid fragmentation
    freq_features_train = pd.DataFrame(index=df.index)
    freq_features_test = pd.DataFrame(index=df_test.index)
    bin_features_train = pd.DataFrame(index=df.index)
    bin_features_test = pd.DataFrame(index=df_test.index)

    for col in cols:
        # --- Frequency encoding ---
        freq = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(freq)
        freq_features_test[f"{col}_freq"] = df_test[col].map(freq).fillna(freq.mean())

        # --- Quantile binning for numeric columns ---
        if col in num:
            for q in [5, 10, 15]:
                try:
                    train_bins, bins = pd.qcut(df[col], q=q, labels=False, retbins=True, duplicates="drop")
                    bin_features_train[f"{col}_bin{q}"] = train_bins
                    bin_features_test[f"{col}_bin{q}"] = pd.cut(df_test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    bin_features_train[f"{col}_bin{q}"] = 0
                    bin_features_test[f"{col}_bin{q}"] = 0

    # Concatenate all new features at once
    df = pd.concat([df, freq_features_train, bin_features_train], axis=1)
    df_test = pd.concat([df_test, freq_features_test, bin_features_test], axis=1)

    return df, df_test


def target_encoding(train, predict, n_splits=10):
    """
    The function turns categorical columns into numbers by replacing
    each category with its average target value, using K-Folds to avoid
    leaking information from the training data.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mean_features_train = pd.DataFrame(index=train.index)
    mean_features_test = pd.DataFrame(index=predict.index)

    for col in cols:
        # --- K-Fold Target Mean Encoding ---
        mean_encoded = np.zeros(len(train))
        for tr_idx, val_idx in kf.split(train):
            tr_fold = train.iloc[tr_idx]
            val_fold = train.iloc[val_idx]
            mean_map = tr_fold.groupby(col)[target].mean()
            mean_encoded[val_idx] = val_fold[col].map(mean_map)

        mean_features_train[f'mean_{col}'] = mean_encoded

        # --- Apply global mean mapping to prediction/test data ---
        global_mean = train.groupby(col)[target].mean()
        mean_features_test[f'mean_{col}'] = predict[col].map(global_mean)

    # --- Concatenate new features at once to avoid fragmentation ---
    train = pd.concat([train, mean_features_train], axis=1)
    predict = pd.concat([predict, mean_features_test], axis=1)

    # Defragment
    train = train.copy()
    predict = predict.copy()
    return train, predict


# Rounding the values
for c in ['annual_income', 'loan_amount']:
    for s, l in {'1s': 0, '10s': -1}.items():
        for g in [df, df_test]:
            g[f'{c}_ROUND_{s}'] = g[c].round(l).astype(int)

# Specific feature engineering
for gf in [df, df_test]:
    gf['subgrade'] = gf['grade_subgrade'].str[1:].astype(int)
    gf['grade'] = gf['grade_subgrade'].str[0]
    gf['total_debt_burden'] = (gf['loan_amount'] * gf['interest_rate'] / 100) / (gf['annual_income'] + 1) 


cols = df.drop(columns=[target,"id"]).columns.tolist()
cat = [c for c in cols if df[c].dtype in ["object","category"]]
num = [c for c in cols if df[c].dtype not in ["object","category","bool"]]

# Creating new features based on the frequency of numerical features
df, df_test = target_encoding(df, df_test)
df, df_test = create_frequency_features(df, df_test)

# Preparing categorical features
df[cat], df_test[cat] = df[cat].astype("category"), df_test[cat].astype("category")


remove = [
    'annual_income_ROUND_10s_bin10','annual_income_ROUND_1s_bin10','annual_income_ROUND_1s_bin15','annual_income_ROUND_1s_bin5',
    'annual_income_bin10','annual_income_bin5','credit_score_bin10','credit_score_bin5','debt_to_income_ratio_bin15','debt_to_income_ratio_bin5',
    'education_level_freq','gender_freq','interest_rate_bin10','interest_rate_bin5','loan_amount_ROUND_10s_bin5','loan_amount_ROUND_1s_bin10',
    'loan_amount_ROUND_1s_bin15','loan_amount_ROUND_1s_bin5','loan_amount_bin10','loan_amount_bin15','loan_amount_bin5','marital_status_freq',
    'subgrade','subgrade_bin10','subgrade_bin15','subgrade_bin5','subgrade_freq',"mean_total_debt_burden",'grade_subgrade',
    'annual_income_ROUND_1s', 'mean_annual_income', 'mean_gender', 'mean_marital_status', 'mean_education_level',
    'mean_employment_status', 'mean_grade_subgrade', 'mean_subgrade', 'interest_rate_freq', 'loan_amount_ROUND_1s_freq',
    'grade_freq', 'total_debt_burden_freq', 'annual_income_bin15', 'annual_income_ROUND_10s_bin5', 'total_debt_burden_bin5'
]

df, df_test = df.drop(columns = remove+["id"]), df_test.drop(columns = remove)
cat = [c for c in df.columns if df[c].dtype in ["object","category"]]


print(f"Number of columns {len(df.columns.tolist())}")
print(df.columns.tolist())


df.isnull().sum()[lambda x: x>0] # Null values count


def LGB(X_train, X_test, y_train, y_test=None, iteration=3000):
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'max_depth': 6, 'num_leaves': 50, 'learning_rate': 0.01,
        'colsample_bytree': 0.8, 'subsample': 0.8,
        'subsample_freq': 1, 'min_child_samples': 20, 'reg_alpha': 0.05,
        'reg_lambda': 0.1, 'random_state': 42,
        'n_jobs': -1, 'device': 'gpu','verbose': -1,"n_estimators":iteration,
    }
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    if y_test is not None:
        model = LGBMClassifier(**lgb_params,
            early_stopping_rounds=100,
        )
        model.fit(
            X_train, y_train, 
            categorical_feature=cat, 
            eval_set=[(X_test, y_test)], 
        )
        print("Early stopping triggered at iteration:", model.best_iteration_)
        pred = model.predict_proba(X_test)[:, 1]
        return pred, model.best_iteration_
    else:
        model = LGBMClassifier(**lgb_params)
        model.fit(X_train, y_train, categorical_feature=cat)
        pred = model.predict_proba(X_test)[:, 1]
        return pred


def XGB(X_train, X_test, y_train, y_test=None, iteration=3000):
    xgb_params = {
        'tree_method': 'hist', 'device': 'cuda','eval_metric': 'auc',
        'objective': 'binary:logistic','random_state': 42,
        'min_child_weight': 89,"max_leaves":4,"reg_alpha":3.2,
        "reg_lambda":5,"eta":0.1,"enable_categorical":True, "n_estimators":iteration,
    }
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    if y_test is not None:
        model = XGBClassifier(**xgb_params,
            early_stopping_rounds=100
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        print("Early stopping triggered at iteration:", model.best_iteration)
        pred = model.predict_proba(X_test)[:, 1]
        return pred, model.best_iteration
    else:
        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_test)[:, 1]
        return pred


X = df.drop(columns=target)
y = df[target]

fold = 7

kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)

XGB_OOF, XGB_iter = np.zeros(len(y)), np.zeros(fold)
LGB_OOF, LGB_iter = np.zeros(len(y)), np.zeros(fold)

for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f"Fold {i+1}")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    XGB_OOF[test_index], XGB_iter[i] = XGB(X_train, X_test, y_train, y_test)
    LGB_OOF[test_index], LGB_iter[i] = LGB(X_train, X_test, y_train, y_test)

XGB_AUC = AUC(y, XGB_OOF)
LGB_AUC = AUC(y, LGB_OOF)

print(f"XGBoost AUC: {XGB_AUC}")
print(f"LGBoost AUC: {LGB_AUC}")

stack_X = np.vstack([XGB_OOF, LGB_OOF]).T

log_model = LogisticRegression()
log_model.fit(stack_X, y)

log_preds = log_model.predict_proba(stack_X)[:, 1]

Stacked_AUC = AUC(y, log_preds)
print(f"Stacked(log) AUC: {Stacked_AUC}")


X_train = df.drop(columns = target)
y_train = df[target]
X_test = df_test

xgb_pred = XGB(X_train, X_test.drop(columns = "id"), y_train, iteration=2700)
lgb_pred = LGB(X_train, X_test.drop(columns = 'id'), y_train, iteration=2800)

stack_X = np.vstack([xgb_pred, lgb_pred]).T
stack_train_preds = log_model.predict_proba(stack_X)[:, 1]

sub = df_test["id"].copy()
sub = pd.DataFrame(sub)

sub[target] = stack_train_preds
sub.to_csv("submission.csv", index = False)

