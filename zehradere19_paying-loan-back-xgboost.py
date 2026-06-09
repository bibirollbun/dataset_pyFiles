import numpy as np
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
from sklearn.utils.class_weight import compute_class_weight



df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df.head()


df['grade'] = df['grade_subgrade'].str[0]
df['subgrade'] = df['grade_subgrade'].str[1]


df['interest_burden'] = df['interest_rate'] * df['loan_amount']
df['risk_adjusted_loan_rate'] = df['loan_amount'] / df['credit_score']


df['loan_paid_back'].value_counts()


df.isnull().sum()


def detect_outliers_iqr_all(df):
    outlier_results = {}

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.05)
        Q3 = df[col].quantile(0.95)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        outlier_results[col] = {
            "outlier_count": len(outliers),
            "lower_bound": lower,
            "upper_bound": upper,
            "outliers": outliers
        }

    return outlier_results

results = detect_outliers_iqr_all(df)

for col, info in results.items():
    print(f"\nColumn: {col}")
    print("Outlier count:", info["outlier_count"])




def remove_outliers_iqr(df):
    df_clean = df.copy()

    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.05)
        Q3 = df_clean[col].quantile(0.95)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    return df_clean

df = remove_outliers_iqr(df)
print(df.shape)


X = df.drop(columns=["loan_paid_back", "id", "gender", "marital_status", "subgrade"])
y = df["loan_paid_back"]  

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

for col in cat_cols:
    X[col] = X[col].astype("category").cat.codes  

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)


classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)

sample_weight_train = y_train.map(class_weight_dict)


"""def objective(trial):

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist", 
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
    }

    model = XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight_train,
        verbose=False
    )

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    return auc


study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42),
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

print("Best AUC:", study.best_value)
print("Best Params:", study.best_params)"""


model_xgb = XGBClassifier(
    objective="binary:logistic",
    n_estimators=563,
    learning_rate=0.04,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=0.5,
    n_jobs=-1,
    eval_metric="auc",
    tree_method="hist",  
    random_state=42
)

model_xgb.fit(
    X_train,
    y_train,
    sample_weight=sample_weight_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred_proba = model_xgb.predict_proba(X_test)[:, 1]  

auc_roc = roc_auc_score(y_test, y_pred_proba)
auc_pr  = average_precision_score(y_test, y_pred_proba)

print("AUC-ROC:", auc_roc)
print("AUC-PR :", auc_pr)



test_df  = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")



test_df['grade'] = test_df['grade_subgrade'].str[0]
test_df['subgrade'] = test_df['grade_subgrade'].str[1]


test_df['interest_burden'] = test_df['interest_rate'] * test_df['loan_amount']
test_df['risk_adjusted_loan_rate'] = test_df['loan_amount'] / test_df['credit_score']


X_fin = test_df.drop(columns=["id", "gender", "marital_status", "subgrade"])
cat_cols = X_fin.select_dtypes(include=["object"]).columns.tolist()

for col in cat_cols:
    X_fin[col] = X_fin[col].astype("category").cat.codes 


y_pred_proba = model_xgb.predict_proba(X_fin)[:, 1] 



ids = test_df['id']
subm = pd.DataFrame({'id': ids,'loan_paid_back':y_pred_proba})
subm.to_csv('submission.csv',index=False)

