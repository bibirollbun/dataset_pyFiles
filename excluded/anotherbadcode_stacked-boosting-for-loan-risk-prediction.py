import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss
from catboost import CatBoostClassifier, Pool
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from xgboost import XGBClassifier



TEST = False

if TEST:
    train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv", nrows=10000)
    test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv", nrows=10000)
else:
    train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

train.head(3)


train['loan_paid_back'].value_counts(normalize=True) # Imbalanced Class


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame = None, target_col: str = "loan_paid_back"):

    train_df = train_df.copy()
    if test_df is not None:
        test_df = test_df.copy()

    print(f"Before Feature Engineering: {train_df.shape[1]} features in train", end="")
    if test_df is not None:
        print(f", {test_df.shape[1]} features in test")
    else:
        print()

    for df in [train_df, test_df] if test_df is not None else [train_df]:
        df["loan_to_income_ratio"] = df["loan_amount"] / (df["annual_income"] + 1e-5)
        df["income_per_credit"] = df["annual_income"] / (df["credit_score"] + 1e-5)
        df["effective_interest"] = df["interest_rate"] * df["debt_to_income_ratio"]
        df["grade"] = df["grade_subgrade"].str[0]
        df["subgrade_num"] = df["grade_subgrade"].str[1].map(
            lambda x: ord(x.upper()) - 64 if isinstance(x, str) else np.nan
        )

    num_cols = train_df.select_dtypes(include=np.number).columns
    for col in num_cols:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        if test_df is not None and col in test_df.columns:
            test_df[col] = test_df[col].fillna(median_val)

    cat_cols = train_df.select_dtypes(include="object").columns.tolist() + ["grade"]
    cat_cols = [c for c in cat_cols if c in train_df.columns]

    freq_maps = {}
    for col in cat_cols:
        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        freq_maps[col] = freq_map
        train_df[f"{col}_freq"] = train_df[col].map(freq_map)
        if test_df is not None:
            test_df[f"{col}_freq"] = test_df[col].map(freq_map).fillna(0)

    target_means = {}
    global_mean = train_df[target_col].mean() if target_col in train_df.columns else 0.5

    for col in cat_cols:
        if target_col in train_df.columns:
            target_mean = train_df.groupby(col)[target_col].mean().to_dict()
            target_means[col] = target_mean
            train_df[f"{col}_target_mean"] = train_df[col].map(target_mean)
        else:
            target_means[col] = {}

        if test_df is not None:
            test_df[f"{col}_target_mean"] = test_df[col].map(target_means.get(col, {})).fillna(global_mean)

    for col in cat_cols:
        train_df[col] = train_df[col].fillna("Unknown")
        if test_df is not None:
            test_df[col] = test_df[col].fillna("Unknown")

    print(f"After Feature Engineering: {train_df.shape[1]} features in train", end="")
    if test_df is not None:
        print(f", {test_df.shape[1]} features in test\n")
    else:
        print("\n")

    if test_df is not None:
        return train_df, test_df, cat_cols, freq_maps, target_means
    else:
        return train_df, cat_cols, freq_maps, target_means



def plot_func(train_df: pd.DataFrame, target_col: str = "loan_paid_back"):

    plt.figure(figsize=(20, 16))
    plt.suptitle("Loan Repayment EDA:", fontsize=16, fontweight='bold')

    plt.subplot(4, 4, 1)
    sns.countplot(x=target_col, data=train_df, palette="viridis")
    plt.title("Target Distribution")
    plt.xlabel("Loan Paid Back")
    plt.ylabel("Count")

    plt.subplot(4, 4, 2)
    num_cols = train_df.select_dtypes(include=np.number).columns
    if target_col in num_cols:
        corr = (
                train_df[num_cols]
                .corr()[target_col]
                .drop(target_col)
                .sort_values(ascending=False)
               )
        sns.barplot(x=corr.values[:10], y=corr.index[:10], palette="coolwarm")
        plt.title("Top 10 Correlations with Target")
    else:
        plt.axis('off')
        plt.text(0.5, 0.5, "No numeric correlation data", ha='center')

    # Distribution of Annual Income
    plt.subplot(4, 4, 3)
    if "annual_income" in train_df.columns:
        sns.kdeplot(data=train_df, x="annual_income", hue=target_col, fill=True)
        plt.title("Annual Income by Target")
    else:
        plt.axis('off')

    # Distribution of Loan Amount
    plt.subplot(4, 4, 4)
    if "loan_amount" in train_df.columns:
        sns.kdeplot(data=train_df, x="loan_amount", hue=target_col, fill=True)
        plt.title("Loan Amount by Target")
    else:
        plt.axis('off')

    # Debt-to-Income Ratio
    plt.subplot(4, 4, 5)
    if "debt_to_income_ratio" in train_df.columns:
        sns.kdeplot(data=train_df, x="debt_to_income_ratio", hue=target_col, fill=True)
        plt.title("Debt-to-Income Ratio by Target")
    else:
        plt.axis('off')

    # Credit Score
    plt.subplot(4, 4, 6)
    if "credit_score" in train_df.columns:
        sns.kdeplot(data=train_df, x="credit_score", hue=target_col, fill=True)
        plt.title("Credit Score by Target")
    else:
        plt.axis('off')

    # Interest Rate
    plt.subplot(4, 4, 7)
    if "interest_rate" in train_df.columns:
        sns.kdeplot(data=train_df, x="interest_rate", hue=target_col, fill=True)
        plt.title("Interest Rate by Target")
    else:
        plt.axis('off')

    # Loan-to-Income Ratio
    plt.subplot(4, 4, 8)
    if "loan_to_income_ratio" in train_df.columns:
        sns.kdeplot(data=train_df, x="loan_to_income_ratio", hue=target_col, fill=True)
        plt.title("Loan-to-Income Ratio by Target")
    else:
        plt.axis('off')

    # Grade vs Target
    plt.subplot(4, 4, 9)
    if "grade" in train_df.columns:
        sns.barplot(x="grade", y=target_col, data=train_df, order=sorted(train_df["grade"].unique()))
        plt.title("Repayment Rate by Grade")
    else:
        plt.axis('off')

    # Employment Status
    plt.subplot(4, 4, 10)
    if "employment_status" in train_df.columns:
        top_emp = train_df["employment_status"].value_counts().index[:6]
        sns.barplot(x="employment_status", y=target_col, data=train_df[train_df["employment_status"].isin(top_emp)], palette="crest")
        plt.title("Repayment Rate by Employment Status")
        plt.xticks(rotation=30)
    else:
        plt.axis('off')

    # Loan Purpose
    plt.subplot(4, 4, 11)
    if "loan_purpose" in train_df.columns:
        top_purpose = train_df["loan_purpose"].value_counts().index[:6]
        sns.barplot(x="loan_purpose", y=target_col, data=train_df[train_df["loan_purpose"].isin(top_purpose)], palette="magma")
        plt.title("Repayment Rate by Loan Purpose")
        plt.xticks(rotation=30)
    else:
        plt.axis('off')

    # Marital Status
    plt.subplot(4, 4, 12)
    if "marital_status" in train_df.columns:
        sns.barplot(x="marital_status", y=target_col, data=train_df, palette="coolwarm")
        plt.title("Repayment Rate by Marital Status")
    else:
        plt.axis('off')

    # Education Level
    plt.subplot(4, 4, 13)
    if "education_level" in train_df.columns:
        sns.barplot(x="education_level", y=target_col, data=train_df, palette="viridis")
        plt.title("Repayment Rate by Education Level")
        plt.xticks(rotation=45)
    else:
        plt.axis('off')

    # Effective Interest
    plt.subplot(4, 4, 14)
    if "effective_interest" in train_df.columns:
        sns.kdeplot(data=train_df, x="effective_interest", hue=target_col, fill=True)
        plt.title("Effective Interest by Target")
    else:
        plt.axis('off')

    # Income per Credit
    plt.subplot(4, 4, 15)
    if "income_per_credit" in train_df.columns:
        sns.kdeplot(data=train_df, x="income_per_credit", hue=target_col, fill=True)
        plt.title("Income per Credit by Target")
    else:
        plt.axis('off')

    # Subgrade vs Target
    plt.subplot(4, 4, 16)
    if "grade_subgrade" in train_df.columns:
        top_subgrades = train_df["grade_subgrade"].value_counts().index[:8]
        sns.barplot(x="grade_subgrade", y=target_col, data=train_df[train_df["grade_subgrade"].isin(top_subgrades)], palette="flare")
        plt.title("Repayment Rate by Subgrade")
        plt.xticks(rotation=45)
    else:
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

train_proc, test_proc, cat_cols, freq_maps, target_means = engineer_features(train, test)
plot_func(train_proc.sample(frac=0.1))


def fit_kfold_and_predict_ensemble_voting(X, y, X_test, seed=42, verbose=True):

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    X_pd, X_test_pd = X.copy(), X_test.copy()
    y_np = y.values if isinstance(y, pd.Series) else y
    scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]

    cat_features = [c for c in X_pd.columns if X_pd[c].dtype == "object"]

    oof_pred = np.zeros(len(y_np))
    test_pred = np.zeros(len(X_test_pd))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_pd, y_np)):

        X_tr, X_va = X_pd.iloc[tr_idx].copy(), X_pd.iloc[va_idx].copy()
        y_tr, y_va = y_np[tr_idx], y_np[va_idx]

        # if verbose: print(f"[Fold {fold+1}] : ")

        X_tr_voting = X_tr.copy()
        X_va_voting = X_va.copy()
        X_test_voting = X_test_pd.copy()

        for col in cat_features:
            X_tr_voting[col] = X_tr_voting[col].astype("category")
            X_va_voting[col] = X_va_voting[col].astype("category")
            X_test_voting[col] = X_test_voting[col].astype("category")

        voting_clf = VotingClassifier(
        estimators=[
            ("cat", CatBoostClassifier(
                iterations=2000,
                learning_rate=0.05,
                depth=8,
                eval_metric="Logloss",
                loss_function="Logloss",
                random_seed=seed,
                verbose=False,
                task_type="GPU",
                cat_features=cat_features
            )),
            ("xgb", XGBClassifier(
                n_estimators=2000,
                tree_method="hist",
                predictor="gpu_predictor",
                device="cuda",
                objective="binary:logistic",
                eval_metric="auc",
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=0.8,
                reg_alpha=0.3,
                random_state=seed,
                enable_categorical=True,
                verbose=False,
            ))
        ],
            voting="soft",
            weights=[0.6, 0.4],
            n_jobs=1,
            verbose=False
        )

        voting_clf.fit(X_tr_voting, y_tr)

        oof_pred[va_idx] = voting_clf.predict_proba(X_va_voting)[:, 1]
        test_pred += voting_clf.predict_proba(X_test_voting)[:, 1] / kf.n_splits

        auc_fold = roc_auc_score(y_va, oof_pred[va_idx])
        if verbose:
            print(f"Fold {fold+1} AUC: {auc_fold:.5f}")


    metrics = {
        "auc": float(roc_auc_score(y_np, oof_pred)),
        "logloss": float(log_loss(y_np, np.clip(oof_pred, 1e-6, 1 - 1e-6)))
    }

    if verbose:
        print("\n[Final Ensemble CV metrics]:", metrics)

    oof_meta = pd.DataFrame({
        "ensemble_pred": oof_pred
    })

    return test_pred, metrics, oof_meta



def fit_kfold_and_predict_ensemble(X, y, X_test, seed=42, verbose=True):

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    X_pd, X_test_pd = X.copy(), X_test.copy()
    y_np = y.values if isinstance(y, pd.Series) else y
    scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]

    cat_features = [c for c in X_pd.columns if X_pd[c].dtype == "object"]

    X_xgb = X_pd.copy()
    X_test_xgb = X_test_pd.copy()
    for col in cat_features:
        X_xgb[col] = X_xgb[col].astype("category")
        X_test_xgb[col] = X_test_xgb[col].astype("category")

    oof_cat, oof_xgb = np.zeros(len(y_np)), np.zeros(len(y_np))
    test_cat, test_xgb = np.zeros(len(X_test_pd)), np.zeros(len(X_test_pd))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_pd, y_np)):

        X_tr, X_va = X_pd.iloc[tr_idx].copy(), X_pd.iloc[va_idx].copy()
        X_tr_xgb, X_va_xgb = X_xgb.iloc[tr_idx].copy(), X_xgb.iloc[va_idx].copy()
        y_tr, y_va = y_np[tr_idx], y_np[va_idx]

        if verbose: print(f"[Fold {fold+1}] : ")

        cat_model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=8,
            eval_metric="Logloss",
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            # class_weights=[1, scale_pos_weight],
            task_type="GPU"
        )
        cat_model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            use_best_model=True,
            cat_features=cat_features
        )
        oof_cat[va_idx] = cat_model.predict_proba(X_va)[:, 1]
        test_cat += cat_model.predict_proba(X_test_pd)[:, 1] / kf.n_splits

        X_tr_d = xgb.DMatrix(
            X_tr_xgb,
            label=y_tr,
            enable_categorical=True
        )
        X_va_d = xgb.DMatrix(
            X_va_xgb,
            label=y_va,
            enable_categorical=True
        )
        X_test_d = xgb.DMatrix(
            X_test_xgb,
            enable_categorical=True
        )

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "gpu_hist",
            "device": "cuda",
            "predictor": "gpu_predictor",
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 0.8,
            "alpha": 0.3,
            "random_state": seed,
        }

        # params["scale_pos_weight"] = scale_pos_weight

        evals = [(X_tr_d, "train"), (X_va_d, "valid")]
        xgb_model = xgb.train(
            params,
            X_tr_d,
            num_boost_round=2000,
            evals=evals,
            early_stopping_rounds=100,
            verbose_eval=False
        )

        oof_xgb[va_idx] = xgb_model.predict(X_va_d)
        test_xgb += xgb_model.predict(X_test_d) / kf.n_splits

        auc_cat = roc_auc_score(y_va, oof_cat[va_idx])
        auc_xgb = roc_auc_score(y_va, oof_xgb[va_idx])
        if verbose:
            print(f"CatBoost AUC: {auc_cat:.5f} | XGBoost AUC: {auc_xgb:.5f}")

    oof_meta = pd.DataFrame({
        "cat": oof_cat,
        "xgb": oof_xgb
    })
    test_meta = pd.DataFrame({
        "cat": test_cat,
        "xgb": test_xgb
    })

    meta_model = LogisticRegression(
                                    solver="lbfgs",
                                    max_iter=2000,
                                    random_state=seed
                                    )
    meta_model.fit(oof_meta, y_np)
    final_oof = meta_model.predict_proba(oof_meta)[:, 1]
    final_test = meta_model.predict_proba(test_meta)[:, 1]

    metrics = {
        "auc": float(roc_auc_score(y_np, final_oof)),
        "logloss": float(log_loss(y_np, np.clip(final_oof, 1e-6, 1 - 1e-6)))
    }

    if verbose:
        print("\nMeta-Blender Feature Importances:")
        print(pd.DataFrame({
                            "Feature": oof_meta.columns,
                            "Coefficient": meta_model.coef_.flatten()
                          }).sort_values("Coefficient", ascending=False))

        print("[Meta-Blender] CV metrics:", metrics)

    return final_test, metrics, oof_meta



features = [c for c in train_proc.columns if c not in ["id", "loan_paid_back"]]
X = train_proc[features]
y = train_proc["loan_paid_back"]
X_test = test_proc[features]

# final_pred, metrics, oof_meta = fit_kfold_and_predict_ensemble(X, y, X_test)
final_pred, metrics, oof_meta = fit_kfold_and_predict_ensemble_voting(X, y, X_test)


submission = pd.DataFrame({
    "id": test_proc["id"],
    "loan_paid_back": np.clip(final_pred, 0, 1)
})

submission.to_csv("submission.csv", index=False)
print(submission.head(7))

