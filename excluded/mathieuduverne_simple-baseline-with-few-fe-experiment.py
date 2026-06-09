import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from copy import deepcopy
from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from sklearn import linear_model
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time
import shap
import lightgbm as lgb 
from lightgbm import LGBMClassifier

# -*- coding: utf-8 -*-
from __future__ import annotations
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix,log_loss
import pandas as pd

import os
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Style plots (discret, lisible)
sns.set(style="whitegrid", context="notebook")
warnings.filterwarnings('ignore')


df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv", index_col="id")
print(df_train.shape)
df_train.head()


num_col = ["annual_income","debt_to_income_ratio","credit_score","loan_amount","interest_rate"]
cat_col = ["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade"]
target = "loan_paid_back"


# null value
print(df_train.isnull().sum(), "\n")
print(df_train.min(numeric_only=True))

# unique category
print("unique values in category column")
for col in cat_col:
    print(f"{col}: {df_train[col].nunique()} unique values -> {df_train[col].unique()[:30]}")


# distribution
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()

for i, col in enumerate(num_col):
    sns.histplot(df_train[col], kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='both', labelsize=8)

for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout(pad=1.0)
plt.show()


outlier_summary = []

for col in num_col:
    Q1 = df_train[col].quantile(0.25)
    Q3 = df_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df_train[col] < lower) | (df_train[col] > upper)).sum()
    outlier_pct = n_outliers / len(df_train) * 100
    outlier_summary.append([col, n_outliers, round(outlier_pct, 2)])
    
outlier_df = pd.DataFrame(outlier_summary, columns=["Variable", "outliers", "% outliers"])
print(outlier_df, "\n")


fig, axes = plt.subplots(2, 3, figsize=(12, 5))
axes = axes.flatten()

for i, col in enumerate(num_col):
    sns.boxplot(y=df_train[col], ax=axes[i], color='lightcoral', fliersize=2)
    axes[i].set_title(col, fontsize=10)
    axes[i].tick_params(axis='y', labelsize=8)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.suptitle("outliers", fontsize=12, fontweight='bold')
plt.tight_layout(pad=1.0)
plt.show()


def feature_engineer(df):
    df = df.copy()
    
    # --- Grade & Subgrade
    df["grade"] = df["grade_subgrade"].astype(str).str[0]
    df["subgrade"] = pd.to_numeric(df["grade_subgrade"].astype(str).str[1:])
        
    # interactions
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
    df["income_to_loan"] = df["annual_income"] / df["loan_amount"]
    df["interest_x_loan"] = df["interest_rate"] * df["loan_amount"]
    df["interest_x_income"] = df["interest_rate"] * df["annual_income"]
    df["interest_x_credit"] = df["interest_rate"] * df["credit_score"]
    df["dti_x_interest"] = df["debt_to_income_ratio"] * df["interest_rate"]
    df["credit_x_income"] = df["credit_score"] * df["annual_income"]
    df["credit_x_loan"] = df["credit_score"] * df["loan_amount"]
    df["income_x_dti"] = df["annual_income"] * df["debt_to_income_ratio"]
    df["loan_x_dti"] = df["loan_amount"] * df["debt_to_income_ratio"]

    # --- Polynomial & ratios
    df["loan_sq"] = df["loan_amount"] ** 2
    df["income_sq"] = df["annual_income"] ** 2
    df["credit_sq"] = df["credit_score"] ** 2
    df["dti_sq"] = df["debt_to_income_ratio"] ** 2
    df["interest_sq"] = df["interest_rate"] ** 2

    df["loan_over_credit"] = df["loan_amount"] / df["credit_score"]
    df["income_over_credit"] = df["annual_income"] / df["credit_score"]
    df["credit_over_income"] = df["credit_score"] / df["annual_income"]

    # --- Credit score bucket
    df["credit_score_bucket"] = pd.cut(
        df["credit_score"],
        bins=[0, 580, 670, 740, 800, 900],
        labels=["poor", "fair", "good", "very_good", "excellent"]
    )

    # --- Aggregated features
    df["total_income_debt_interest"] = df["annual_income"] + df["debt_to_income_ratio"] + df["interest_rate"]
    df["income_minus_loan"] = df["annual_income"] - df["loan_amount"]
    df["credit_minus_dti"] = df["credit_score"] - df["debt_to_income_ratio"]
    df["loan_interest_ratio"] = df["loan_amount"] / df["interest_rate"]
    df["income_interest_ratio"] = df["annual_income"] / df["interest_rate"]
    df["credit_interest_ratio"] = df["credit_score"] / df["interest_rate"]

    # grade_subgrade corresponds to a risk category assigned to loan before encoding (A1 < B2) -> after encoding (11, 22) 
    df['grade_encoded'] = df['grade_subgrade'].apply(lambda x: (ord(x[0]) - 64) * 10 + int(x[1]))

    # Encoding gender: 3 unique values -> ['Male', 'Female', 'Other']
    df['gender_encoded'] = df['gender'].map({'Male': 2, 'Female': 1, 'Other': 0})

    # marital_status: 4 unique values -> ['Single' 'Married' 'Divorced' 'Widowed']
    df['marital_status_encoded'] = df['marital_status'].map({'Single': 1, 'Married': 2, 'Divorced': 3, 'Widowed': 4})

    #education_level: 5 unique values -> ['High School' "Master's" "Bachelor's" 'PhD' 'Other']
    df['education_level_encoded'] = df['education_level'].map({"PhD": 1, "Master's": 2, "Bachelor's": 3, "High School": 4, "Other": 5})
    
    # employment_status: 5 unique values -> ['Self-employed' 'Employed' 'Unemployed' 'Retired' 'Student']
    df['employment_status_encoded'] = df['employment_status'].map({"Employed": 1, "Self-employed": 2, "Bachelor's": 3, "Student": 4, "Retired": 5, "Unemployed": 6})

    # loan_purpose: 8 unique values -> ['Other' 'Debt consolidation' 'Home' 'Education' 'Vacation' 'Car' 'Medical' 'Business']
    df['loan_purpose_encoded'] = df['loan_purpose'].map({"Home": 1, "Education": 2, "Business": 3, "Medical": 4, "Car": 5, "Debt consolidation": 6, "Other": 7, "Vacation": 8})

    # credit_score_backet: 5 unique values -> ["poor", "fair", "good", "very_good", "excellent"]
    df['credit_score_bucket_encoded'] = df['credit_score_bucket'].map({"excellent": 1, "very_good": 2, "good": 3, "fair": 4, "poor": 5})
    return df


df = feature_engineer(df_train)
df.head()


feature_col = [
    "annual_income", "debt_to_income_ratio", "credit_score", "loan_amount",
    "interest_rate", "grade_encoded", "marital_status_encoded",
    "education_level_encoded", "employment_status_encoded", "loan_purpose_encoded",
    "credit_minus_dti","dti_x_interest","total_income_debt_interest",
    "credit_x_loan","credit_x_income","loan_over_credit","interest_x_credit","interest_x_loan",
    "loan_interest_ratio","income_x_dti","income_to_loan","credit_interest_ratio",
    "credit_score_bucket_encoded"
]


y = df["loan_paid_back"]
X = df[feature_col].copy()
X.head()


N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 100
MAX_BOOST_ROUNDS = 2000
RANDOM_STATE = 42

def run_experiment(
    X, y,
    exp_name,
    feature_set_name,
    encoder_name,
    params=None,
    verbose=False,
    save_results=True,
    use_shap="one_fold",
):

    if params is None:
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbosity": -1,
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
            "n_estimators": MAX_BOOST_ROUNDS
        }
    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(100)
    ]
    
    results, importances = [], []
    shap_values_all, shap_X_all = [], []

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    start_total = time.time()

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        start = time.time()
        if encoder_name == "target_encoding":
            model = build_target_encoding_pipeline(X_train, params)
        else:
            model = LGBMClassifier(**params)
            
        # model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric="auc",
            callbacks=callbacks,
        )
        train_time = time.time() - start

        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        logloss = log_loss(y_val, y_pred)
        cm = confusion_matrix(y_val, (y_pred > 0.5).astype(int))

        feat_imp = pd.DataFrame({
            "feature": X.columns,
            "importance_gain": model.booster_.feature_importance(importance_type='gain'),
            "importance_split": model.booster_.feature_importance(importance_type='split')
        })
        feat_imp["fold"] = fold
        importances.append(feat_imp)

        perm = permutation_importance(
            model, X_val, y_val,
            n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1
        )
        perm_imp = pd.DataFrame({
            "feature": X.columns,
            "perm_importance": perm["importances_mean"]
        }).sort_values("perm_importance", ascending=False).head(20)

        if use_shap.lower() != "none":
            if use_shap.lower() == "one_fold" and fold == 0:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_val)
                shap_values_all.append(shap_vals)
                shap_X_all.append(X_val)
            elif use_shap.lower() == "all_folds":
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_val)
                shap_values_all.append(shap_vals)
                shap_X_all.append(X_val)

        results.append({
            "exp_name": exp_name,
            "feature_set": feature_set_name,
            "encoder": encoder_name,
            "fold": fold,
            "auc": auc,
            "logloss": logloss,
            "train_time": round(train_time, 2),
            "best_iter": model.best_iteration_,
            "cm": cm.tolist(),
            "top_perm_features": perm_imp["feature"].tolist(),
        })

        if verbose:
            print(f"Fold {fold} | AUC={auc:.5f} | Logloss={logloss:.5f} "
                  f"| Iter={model.best_iteration_} | Time={train_time:.1f}s")

    total_time = time.time() - start_total
    df_results = pd.DataFrame(results)
    df_importances = (
        pd.concat(importances)
        .groupby("feature")[["importance_gain", "importance_split"]]
        .mean()
        .reset_index()
    )

    df_summary = (
        df_results.groupby(["exp_name", "feature_set", "encoder"])
        .agg({"auc": ["mean", "std"], "logloss": "mean", "train_time": "sum"})
        .reset_index()
    )
    df_summary.columns = [
        "exp_name", "feature_set", "encoder",
        "auc_mean", "auc_std", "logloss_mean", "train_time_total"
    ]
    df_summary["total_time_sec"] = total_time

    if save_results:
        df_results.to_csv(f"results_{exp_name}_folds.csv", index=False)
        df_importances.to_csv(f"importances_{exp_name}.csv", index=False)
        df_summary.to_csv(f"summary_{exp_name}.csv", index=False)
        if shap_values_all:
            np.save(f"shap_values_{exp_name}.npy", np.array(shap_values_all, dtype=object))
            pd.concat(shap_X_all).to_csv(f"shap_X_{exp_name}.csv", index=False)

    return df_summary, df_results, df_importances, (shap_X_all, shap_values_all)


def analyze_features(importances, shap_data, top_k=30):
    shap_X_all, shap_values_all = shap_data
    X_shap = shap_X_all[0]
    shap_values_raw = shap_values_all[0]

    if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
        shap_values = shap_values_raw[1]
    else:
        shap_values = shap_values_raw

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "feature": X_shap.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    merged = importances.merge(shap_importance, on="feature", how="outer").fillna(0)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        y="feature",
        x="mean_abs_shap",
        data=shap_importance.head(top_k),
        color="royalblue"
    )
    plt.title(f"Top {top_k} Features — Importance SHAP")
    plt.xlabel("Mean |SHAP| value")
    plt.ylabel("")
    plt.show()

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x="importance_gain", y="mean_abs_shap", data=merged)
    plt.title("Corrélation between importances LightGBM & SHAP")
    plt.xlabel("Gain Importance (LightGBM)")
    plt.ylabel("Mean |SHAP|")
    plt.show()

    # --- Résumé lisible ---
    top_shap = shap_importance.head(35)
    top_gain = importances.sort_values("importance_gain", ascending=False).head(35)

    print("------------------------------------------------")
    print(f"Top {len(top_shap)} SHAP :")
    for i, row in top_shap.iterrows():
        print(f"  • {row['feature']} — mean impact = {row['mean_abs_shap']:.4f}")

    print(f"\nTop {len(top_gain)} gain features LightGBM  :")
    for i, row in top_gain.iterrows():
        print(f"  • {row['feature']} — mean gain = {row['importance_gain']:.0f}")

    return shap_importance, merged


def build_target_encoding_pipeline(X, params):
    cat_cols = [
        "gender",
        "marital_status",
        "education_level",
        "employment_status",
        "loan_purpose",
        "grade_subgrade",
        "credit_score_bucket",
    ]

    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("target_enc", TargetEncoder(cols=cat_cols, cv=5, smoothing=1.0), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = Pipeline([
        ("preprocess", preprocess),
        ("clf", LGBMClassifier(**params))
    ])

    return model


base_params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    n_estimators=2000,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=2.0,
    verbosity=-1,
    n_jobs=-1,
    random_state=42,
)

def build_feature_version(X,df, version="baseline"):
    X_new = X.copy()

    if version == "baseline":
        return X_new

    elif version == "nonlinear_transforms":
        for col in ["debt_to_income_ratio", "loan_amount", "annual_income", "credit_score", "debt_to_income_ratio", "interest_rate"]:
            X_new[f"log_{col}"] = np.log1p(X_new[col])
            X_new[f"sqrt_{col}"] = np.sqrt(np.abs(X_new[col]))

        X_new["log_income_x_interest"] = X_new["log_annual_income"] * X_new["interest_rate"]
        X_new["log_loan_x_credit"] = X_new["log_loan_amount"] * X_new["credit_score"]
        return X_new

    elif version == "grade_intercation":
        X_new["subgrade_x_credit"] = X_new["subgrade"] * X_new["credit_score"]
        X_new["subgrade_x_interest"] = X_new["subgrade"] * X_new["interest_rate"]
        X_new["subgrade_x_income"] = X_new["subgrade"] * X_new["annual_income"]
        return X_new

    elif version == "target_encoding":
        X_new = X_new[feature_col].copy()

        to_remove = [
            "grade_encoded", "marital_status_encoded", "education_level_encoded",
            "employment_status_encoded", "loan_purpose_encoded", "gender_encoded",
            "credit_score_bucket_encoded"
    ]
        for col in to_remove:
            if col in X_new.columns:
                X_new = X_new.drop(columns=[col])

        cats = ["gender", "marital_status", "education_level",
            "employment_status", "loan_purpose",
            "grade_subgrade", "credit_score_bucket"]

        for c in cats:
            X_new[c] = df[c]

        return X_new
 
    else:
        raise ValueError(f"Version inconnue : {version}")

experiments = [
    ("exp_baseline", "base_features", "label_encoding", "baseline", "one_fold"),
    ("exp_nonlinear", "base_features_nonlinear_features", "label_encoding", "nonlinear_transforms", "one_fold"),
    ("exp_grade_interaction", "base_features_grade_interaction", "label_encoding", "grade_interaction", "one_fold"),
    ("exp_target_encoding", "target_encoding_features", "target_encoding", "target_encoding", "one_fold"),
]

all_summaries = []

for exp_name, feat_set, encoder, version, shap_mode in experiments:
    print(f"\n Experience : {exp_name} ({version})")

    X_exp = build_feature_version(X, df, version)

    summary, results, importances, shap_data = run_experiment(
        X_exp, y,
        exp_name=exp_name,
        feature_set_name=feat_set,
        encoder_name=encoder,
        params=base_params,
        verbose=True,
        use_shap=shap_mode,
    )

    print(summary)

    print(f"\n Analyse {exp_name}")
    shap_imp, merged = analyze_features(importances, shap_data)

    summary["feature_strategy"] = version
    all_summaries.append(summary)


final_summary = pd.concat(all_summaries)
final_summary = final_summary.sort_values("auc_mean", ascending=False)
display(final_summary)

