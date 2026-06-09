import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin

# === Feature Engineering Function ===
def add_features_numeric_only(df):
    df = df.copy()
    df['is_senior'] = (df['age'] >= 60).astype(int)
    df['is_young_adult'] = df['age'].between(18, 30).astype(int)
    df['age_decade'] = (df['age'] // 10) * 10
    df['age_zscore'] = (df['age'] - df['age'].mean()) / df['age'].std()
    df['age_bin'] = pd.cut(df['age'], bins=range(15, 100, 5), labels=False)

    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str))

    job_freq = df['job'].value_counts(normalize=True)
    df['job_freq'] = df['job'].map(job_freq).fillna(0)
    df['job_is_high_profile'] = df['job'].isin(['management', 'admin.', 'technician']).astype(int)
    df['is_self_employed'] = df['job'].isin(['self-employed', 'entrepreneur']).astype(int)

    df['is_married'] = (df['marital'] == 'married').astype(int)
    df['is_single_or_divorced'] = df['marital'].isin(['single', 'divorced']).astype(int)
    df['is_educated'] = df['education'].isin(['tertiary', 'secondary']).astype(int)
    df['unknown_education'] = (df['education'] == 'unknown').astype(int)
    df['edu_job_match'] = ((df['education'] == 'tertiary') & df['job_is_high_profile'].astype(bool)).astype(int)

    df['balance_log'] = df['balance'].apply(lambda x: np.log1p(x) if x > 0 else 0)
    df['is_balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_zscore'] = (df['balance'] - df['balance'].mean()) / df['balance'].std()
    df['balance_bucket'] = pd.qcut(df['balance'], 5, labels=False, duplicates='drop')
    df['has_high_balance'] = (df['balance'] > df['balance'].quantile(0.75)).astype(int)

    df['has_any_loan'] = ((df['loan'] == 'yes') | (df['housing'] == 'yes')).astype(int)
    df['has_both_loans'] = ((df['loan'] == 'yes') & (df['housing'] == 'yes')).astype(int)

    loan_indicator = (df['loan'] == 'yes').astype(int).replace(0, np.nan)
    df['loan_balance_ratio'] = df['balance'] / loan_indicator

    df['is_mobile_contact'] = (df['contact'] == 'cellular').astype(int)
    df['unknown_contact'] = (df['contact'] == 'unknown').astype(int)
    df['preferred_contact_score'] = df['contact'].map({'cellular': 2, 'telephone': 1, 'unknown': 0}).fillna(0)

    df['calls_per_day'] = df['campaign'] / df['day'].replace(0, np.nan)
    df['multiple_contacts_flag'] = (df['campaign'] > 3).astype(int)
    df['pdays_flag'] = (df['pdays'] != -1).astype(int)
    df['days_since_last_contact_bucket'] = pd.cut(df['pdays'], bins=[-2,0,30,90,180,999], labels=False)
    df['previous_contact_ratio'] = df['previous'] / df['campaign'].replace(0, np.nan)

    month_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
    df['month_enc'] = df['month'].map(month_map).fillna(0).astype(int)
    season_map = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}
    df['season'] = df['month_enc'].map(season_map).fillna(-1).astype(int)
    df['is_month_end'] = (df['day'] > 25).astype(int)
    df['day_of_week_estimate'] = df['day'] % 7
    df['is_q2'] = df['month'].isin(['apr','may','jun']).astype(int)

    df['age_x_balance'] = df['age'] * df['balance']
    df['campaign_x_previous'] = df['campaign'] * df['previous']
    df['pdays_x_poutcome'] = df['pdays'] * df['poutcome_enc'].fillna(0)
    df['balance_per_campaign'] = df['balance'] / (df['campaign'].replace(0, np.nan))
    df['duration_per_campaign'] = df['duration'] / (df['campaign'].replace(0, np.nan))
    df['age_bin_x_is_married'] = df['age_bin'].fillna(-1).astype(int) * df['is_married']
    df['job_enc_x_loan'] = df['job_enc'] * (df['loan'] == 'yes').astype(int)
    df['balance_log_x_is_balance_pos'] = df['balance_log'] * df['is_balance_positive']
    df['calls_per_day_x_multiple_contacts'] = df['calls_per_day'] * df['multiple_contacts_flag']
    df['season_x_is_q2'] = df['season'] * df['is_q2']

    original_cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    df = df.drop(columns=[c for c in original_cat_cols if c in df.columns])
    df = df.fillna(0)

    for c in df.columns:
        if df[c].dtype == 'bool':
            df[c] = df[c].astype(int)
        elif df[c].dtype.name == 'category':
            df[c] = df[c].astype(int)
    return df



# Enable GPU acceleration
%load_ext cuml.accel

# Import as usual—cuML will intercept supported calls
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# === Load Data ===
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

X = train.drop(columns=["id", "y"])
y = train["y"]
X_test = test.drop(columns=["id"])

X = add_features_numeric_only(X)
X_test = add_features_numeric_only(X_test)

numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()


import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols)
])

def objective(trial):
    # Numeric hyperparameters
    base_C = trial.suggest_float("base_C", 1e-3, 10.0, log=True)
    meta_C = trial.suggest_float("meta_C", 1e-3, 10.0, log=True)

    # Categorical hyperparameters
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None])
    class_weight_opt = trial.suggest_categorical("class_weight", ["balanced", None])

    # Compatibility checks
    if solver in ["newton-cg", "sag", "lbfgs"] and penalty not in ["l2", None]:
        raise optuna.exceptions.TrialPruned()
    if solver == "liblinear" and penalty not in ["l1", "l2"]:
        raise optuna.exceptions.TrialPruned()
    if solver == "saga" and penalty not in ["l1", "l2", "elasticnet", None]:
        raise optuna.exceptions.TrialPruned()

    # For elasticnet, suggest l1_ratio
    kwargs = {}
    if penalty == "elasticnet":
        kwargs["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

    # Build base pipeline
    base_clf = LogisticRegression(
        solver=solver,
        penalty=penalty,
        C=base_C,
        class_weight=class_weight_opt,
        max_iter=3000,
        n_jobs=-1,
        **kwargs
    )
    base_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", base_clf)
    ])

    # Out-of-fold predictions
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for tr_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        base_pipe.fit(X_tr, y_tr)
        oof_preds[val_idx] = base_pipe.predict_proba(X_val)[:, 1]
        test_preds += base_pipe.predict_proba(X_test)[:, 1] / cv.n_splits

    # Meta-model training
    X_meta = pd.DataFrame({"oof": oof_preds})
    X_test_meta = pd.DataFrame({"oof": test_preds})
    meta_clf = LogisticRegression(C=meta_C, solver="lbfgs", max_iter=1000)
    meta_clf.fit(X_meta, y)
    meta_auc = roc_auc_score(y, meta_clf.predict_proba(X_meta)[:, 1])

    return meta_auc

# Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Export full trial logs to CSV
df_trials = study.trials_dataframe()
df_trials.to_csv("optuna_trials_full.csv", index=False)

print(" All trial parameters and values saved to optuna_trials_full.csv")
print(" Best params:", study.best_params)
print(" Best ROC‑AUC:", study.best_value)


submission = pd.DataFrame({
    "id": test["id"],
    "y": meta_clf.predict_proba(X_test_meta)[:, 1]
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv is saved.")


