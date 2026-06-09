import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, roc_auc_score


import optuna
import torch
print(torch.cuda.is_available())


df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


df_train.head()


print('df_train.shape:', df_train.shape)
print('df_test.shape:', df_test.shape)


print(df_train.info())


print(df_test.info())


print(df_train.describe())


cols = ['id', 'annual_income', 'debt_to_income_ratio', 'credit_score',
       'loan_amount', 'interest_rate', 'gender', 'marital_status',
       'education_level', 'employment_status', 'loan_purpose',
       'grade_subgrade', 'loan_paid_back']


for col in cols:
    print(col, df_train[col].nunique())


cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
num_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']


target_col = "loan_paid_back"


counts = df_train[target_col].value_counts()
labels = counts.index
values = counts.values

plt.figure(figsize=(15,5.5)) 

bars = plt.barh(labels, values, color = 'crimson')
plt.ylabel("loan_paid_back")
plt.xlabel("Frequency")
plt.title("The Distribution of the Target Column 'loan_paid_back'")


plt.yticks([1, 0])

total = values.sum()
for bar, count in zip(bars, values):
    width = bar.get_width()
    pct = count / total * 100
    plt.text(width, bar.get_y() + bar.get_height()/2,
             f"{count}\n({pct:.1f}%)",
             ha='left', va='center')
plt.show()


n_vars = len(num_cols)
fig, axes = plt.subplots(n_vars, 2, figsize=(12, n_vars * 3))

for i, col in enumerate(num_cols):

    axes[i, 0].hist(df_train[col], bins=60, edgecolor='black', color = 'crimson')
    axes[i, 0].set_title(f"{col}'s Histogram")
    
    axes[i, 1].boxplot(df_train[col], vert=False)
    axes[i, 1].set_title(f"{col}'s Boxplot")

plt.tight_layout()
plt.show()


n_vars = len(num_cols)
n_cols = 2 
n_rows = (n_vars + 1) // 2  

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows)) 

for i, col in enumerate(num_cols):
    row = i // 2  
    col_idx = i % 2  
    sns.boxplot(x='loan_paid_back', y=col, data=df_train, ax=axes[row, col_idx], palette='pastel')
    axes[row, col_idx].set_title(f"{col} by loan_paid_back")

if n_vars % 2 != 0:
    fig.delaxes(axes[n_rows-1, 1])

plt.tight_layout()
plt.show()


n_vars = len(cat_cols)
fig, axes = plt.subplots(n_vars, 2, figsize=(14, n_vars * 5))

for i, col in enumerate(cat_cols):
    sns.countplot(x=df_train[col], ax=axes[i, 0],
                  order=df_train[col].value_counts().index,
                  palette='pastel')
    axes[i, 0].set_title(f"{col}'s Countplot")
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel('Count')
    axes[i, 0].tick_params(axis='x', rotation=45)

    df_train[col].value_counts().plot.pie(
        ax=axes[i, 1],
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette('pastel'),
        legend=False,
        ylabel='' 
    )
    axes[i, 1].set_title(f"{col}'s Pie Chart")

plt.tight_layout()
plt.show()


n_vars = len(cat_cols)
n_cols = 2
n_rows = (n_vars + 1) // 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))

for i, col in enumerate(cat_cols):
    row = i // 2
    col_idx = i % 2
    sns.countplot(x=col, hue='loan_paid_back', data=df_train, ax=axes[row, col_idx], palette='pastel')
    axes[row, col_idx].set_title(f"{col} by loan_paid_back")
    axes[row, col_idx].tick_params(axis='x', rotation=45)
    axes[row, col_idx].set_xlabel(col)
    axes[row, col_idx].set_ylabel('Count')

if n_vars % 2 != 0:
    fig.delaxes(axes[n_rows - 1, 1])

plt.tight_layout()
plt.show()


def preprocess_pipeline(df):

    df = df.copy()
    eps = 1e-6

    clip_rules = {
        "annual_income": 0.995,
        "loan_amount": 0.995,
        "debt_to_income_ratio": 0.99,
        "interest_rate": 0.995,
        "credit_score": 0.99
    }

    for col, q in clip_rules.items():
        upper = df[col].quantile(q)

        if col == "credit_score":
            lower = df[col].quantile(1 - q)
            df[col] = df[col].clip(lower=lower, upper=upper)
        else:
            df[col] = df[col].clip(lower=0, upper=upper)

    df["annual_income"] = df["annual_income"].clip(lower=eps)
    df["loan_amount"]   = df["loan_amount"].clip(lower=eps)

    df["interest_rate_real"] = df["interest_rate"] / 100.0

    df["interest_burden"] = df["loan_amount"] * df["interest_rate_real"]
    
    df["monthly_income"] = df["annual_income"] / 12

    df["estimated_monthly_payment"] = (df["interest_burden"] / 12)

    df["monthly_payment_ratio"] = (df["estimated_monthly_payment"] / (df["monthly_income"] + eps))


    df["real_dti"] = (df["estimated_monthly_payment"] / (df["annual_income"] + eps))

    df["credit_per_income"] = (df["credit_score"] / (df["annual_income"] + eps))

    df["grade_group"] = df["grade_subgrade"].str[0]

    df.drop(columns=["estimated_monthly_payment"], inplace=True)

    return df


df_train = preprocess_pipeline(df_train)
df_test = preprocess_pipeline(df_test)


df_train.head()


df_test.head()


def analyze_feature_importance(df):

    X = df.drop(["loan_paid_back", "id"], axis=1)
    y = df["loan_paid_back"]

    X = pd.get_dummies(X, drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train)

    feature_importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(10, 18))
    sns.barplot(data=feature_importances.head(30),
                x="importance", y="feature")
    plt.title("Top 30 Feature Importances (LightGBM)")
    plt.tight_layout()
    plt.show()

    return feature_importances


fi_df = analyze_feature_importance(df_train)
fi_df.head()


rs = 42


num_cols = ['annual_income', 'monthly_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate', 'monthly_payment_ratio', 'interest_burden', 'real_dti', 'credit_per_income']

cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade', 'grade_group']

onehot_cols = ['gender', 'marital_status']

ordinal_cols = ['loan_purpose', 'education_level', 'employment_status', 'grade_subgrade', 'grade_group']


def get_target_rate_order(df, col, target_col):
    rate_df = (
        df.groupby(col)[target_col]
        .mean()
        .sort_values()   
    )
    return rate_df.index.astype(str).tolist()


ordinal_target_cols = ['loan_purpose', "education_level", "employment_status", "grade_subgrade", "grade_group"]
target_col = "loan_paid_back" 

ordinal_categories = []

for col in ordinal_target_cols:
    order = get_target_rate_order(df_train, col, target_col)
    ordinal_categories.append(order)

    print(f"\n[{col}] unpaid back order:")
    print(order)


def prepare_data(df_train, target_col, num_cols, onehot_cols, ordinal_cols):
    
    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=rs
    )

    ordinal_categories = [
        ['Education', 'Medical', 'Vacation', 'Debt consolidation', 'Car', 'Other', 'Business', 'Home'],  # loan_purpose
        ["Bachelor's", "Master's", 'Other', 'High School', 'PhD'],                                        # education_level
        ['Unemployed', 'Student', 'Employed', 'Self-employed', 'Retired'],                                # employment_status
        [
            'F3', 'F2', 'F1', 'F4', 'F5',
            'E3', 'E4', 'E1', 'E2', 'E5',
            'D3', 'D5', 'D4', 'D2', 'D1',
            'C3', 'C4', 'C5', 'C2', 'C1',
            'B1', 'B4', 'B5', 'B2', 'B3',
            'A5', 'A1', 'A2', 'A3', 'A4'
        ],                                                                                                 # grade_subgrade
        ['F', 'E', 'D', 'C', 'B', 'A']                                                                     # grade_group
    ]

    num_transformer = "passthrough"

    onehot_transformer = OneHotEncoder(
        handle_unknown='ignore', 
        sparse_output=False
    )

    ordinal_transformer = OrdinalEncoder(
        categories=ordinal_categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('onehot', onehot_transformer, onehot_cols),
            ('ordinal', ordinal_transformer, ordinal_cols)
        ],
        remainder='drop'
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    return X_train_processed, X_val_processed, y_train, y_val, preprocessor


print("âœ… education_level Encoding Mapping:",
      dict(zip(
          ordinal_categories[1],
          range(len(ordinal_categories[1]))
      )))


print("num :", set(num_cols) - set(df_train.columns))
print("cat :", set(cat_cols) - set(df_train.columns))
print("onehot :", set(onehot_cols) - set(df_train.columns))
print("ordinal :", set(ordinal_cols) - set(df_train.columns))


def get_oof_predictions(models, X, y, X_test, n_splits=5):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)

    oof_train = np.zeros((X.shape[0], len(models)))
    oof_test = np.zeros((X_test.shape[0], len(models)))

    for i, model in enumerate(models):
        test_fold_preds = []

        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            model.fit(X_tr, y_tr)

            oof_train[val_idx, i] = model.predict_proba(X_val)[:, 1]
            test_fold_preds.append(model.predict_proba(X_test)[:, 1])

        oof_test[:, i] = np.mean(test_fold_preds, axis=0)

    return oof_train, oof_test


def optimize_lgb(trial, X_train_p, y_train, X_val_p, y_val):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1400, 2100),
        "learning_rate": trial.suggest_float("learning_rate", 0.022, 0.033),
        "num_leaves": trial.suggest_int("num_leaves", 380, 550),
        "max_depth": trial.suggest_int("max_depth", 12, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 90, 160),
        "subsample": trial.suggest_float("subsample", 0.55, 0.70),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 0.65),
        "reg_alpha": trial.suggest_float("reg_alpha", 6.0, 9.5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.8, 2.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.45, 0.80),
        "random_state": rs,
        "verbose": -1,
        "verbosity": -1,
        "silent": True
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_p, y_train)
    proba = model.predict_proba(X_val_p)[:, 1]
    return roc_auc_score(y_val, proba)


def optimize_xgb(trial, X_train_p, y_train, X_val_p, y_val):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 900, 1600),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.055),
        "max_depth": trial.suggest_int("max_depth", 6, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 7, 15),
        "subsample": trial.suggest_float("subsample", 0.90, 0.99),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 0.60),
        "gamma": trial.suggest_float("gamma", 0.5, 1.8),
        "reg_alpha": trial.suggest_float("reg_alpha", 7.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 3.0, 9.0),

        "tree_method": "hist",
        "eval_metric": "auc",
        "random_state": rs
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_p, y_train)

    proba = model.predict_proba(X_val_p)[:, 1]
    return roc_auc_score(y_val, proba)


def optimize_meta(trial, oof_train, y_full):

    params = {
        "C": trial.suggest_float("C", 2.6, 3.2), 
        "solver": "lbfgs",
        "penalty": "l2",
        "max_iter": 1000,
        "n_jobs": -1,
        "random_state": rs
    }

    model = LogisticRegression(**params)
    model.fit(oof_train, y_full)

    proba = model.predict_proba(oof_train)[:, 1]
    return roc_auc_score(y_full, proba)


def main_stacking(df_train, df_test, target_col, num_cols, onehot_cols, ordinal_cols):

    X_train_p, X_val_p, y_train, y_val, preprocessor = prepare_data(
        df_train, target_col, num_cols, onehot_cols, ordinal_cols
    )

    X_full = df_train.drop(columns=[target_col])
    y_full = df_train[target_col]

    X_full_p = preprocessor.transform(X_full)
    X_test_p = preprocessor.transform(df_test)

    print("\n[Optuna] LGBM Optimizing...")
    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(
        lambda trial: optimize_lgb(trial, X_train_p, y_train, X_val_p, y_val),
        n_trials=50,
        n_jobs=2
    )

    print("\n[Optuna] XGBoost Optimizing...")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(
        lambda trial: optimize_xgb(trial, X_train_p, y_train, X_val_p, y_val),
        n_trials=50,
        n_jobs=2
    )

    best_lgb = study_lgb.best_params.copy()
    print("best_lgb:", best_lgb)
    best_lgb.update({"random_state": rs, "gpu_use_dp": False, "verbose": -1, "verbosity": -1, "silent": True})

    best_xgb = study_xgb.best_params.copy()
    print("best_xgb:", best_xgb)
    best_xgb.update({"random_state": rs, "eval_metric": "auc","tree_method": "hist", "device": "cuda"})

    lgb_model = lgb.LGBMClassifier(**best_lgb)
    xgb_model = xgb.XGBClassifier(**best_xgb)

    base_models = [lgb_model, xgb_model]
    model_names = ["LGBM", "XGB"]

    print("\n[Stacking] OOF ~ing (n_splits=5)...\n")
    oof_train, oof_test = get_oof_predictions(
        base_models, X_full_p, y_full, X_test_p, n_splits=5
    )

    print("\nðŸ“Š Base Model OOF AUC Scores:")
    for i, name in enumerate(model_names):
        auc = roc_auc_score(y_full, oof_train[:, i])
        print(f"{name} â†’ OOF AUC: {auc:.5f}")

    print("\n[Optuna] Stacking Meta Logisitc ~ing...")
    study_meta = optuna.create_study(direction="maximize")
    study_meta.optimize(
        lambda trial: optimize_meta(trial, oof_train, y_full),
        n_trials=50,
        n_jobs=4
    )

    meta_best = study_meta.best_params.copy()
    meta_best.update({
        "solver": "lbfgs",
        "penalty": "l2",
        "max_iter": 1000,
        "n_jobs": -1,
        "random_state": rs
    })

    meta_model = LogisticRegression(**meta_best)

    meta_model.fit(oof_train, y_full)

    stacked_oof_proba = meta_model.predict_proba(oof_train)[:, 1]
    stacked_auc = roc_auc_score(y_full, stacked_oof_proba)
    print(f"\nðŸ”¥ Stacking Final OOF AUC: {stacked_auc:.5f}")

    test_proba_lgb = lgb_model.fit(X_full_p, y_full, callbacks=[lgb.log_evaluation(0)]).predict_proba(X_test_p)[:, 1]
    test_proba_xgb = xgb_model.fit(X_full_p, y_full).predict_proba(X_test_p)[:, 1]

    test_proba_stacking = meta_model.predict_proba(oof_test)[:, 1]

    print("\nâœ… Stacking with Optimized Logistic Meta Model completed.")

    return {
        "lgbm": test_proba_lgb,
        "xgb": test_proba_xgb,
        "stacking": test_proba_stacking
    }, preprocessor, base_models, meta_model


test_proba_dict, preprocessor, models_list, meta_model = main_stacking(
    df_train=df_train,
    df_test=df_test,
    target_col=target_col,
    num_cols=num_cols,
    onehot_cols=onehot_cols,
    ordinal_cols=ordinal_cols
)


submission_lgb = pd.DataFrame({
    'id': df_sub['id'],
    'loan_paid_back': test_proba_dict["lgbm"]
})
submission_lgb.to_csv('submission_lgbm.csv', index=False)

submission_xgb = pd.DataFrame({
    'id': df_sub['id'],
    'loan_paid_back': test_proba_dict["xgb"]
})
submission_xgb.to_csv('submission_xgb.csv', index=False)

submission_stack = pd.DataFrame({
    'id': df_sub['id'],
    'loan_paid_back': test_proba_dict["stacking"]
})
submission_stack.to_csv('submission_stacking.csv', index=False)

print("âœ… submission_lgbm.csv saved")
print("âœ… submission_xgb.csv saved")
print("âœ… submission_stacking.csv saved")


preds = test_proba_dict

pred_df = pd.DataFrame({
    "LGBM": preds["lgbm"],
    "XGB": preds["xgb"],
    "STACKING": preds["stacking"]
})

corr_pearson = pred_df.corr(method="pearson")
corr_spearman = pred_df.corr(method="spearman")

print("\nðŸ“Š [Pearson Correlation]")
print(corr_pearson)

print("\nðŸ“Š [Spearman Correlation]")
print(corr_spearman)




