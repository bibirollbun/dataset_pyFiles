import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import optuna
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier


df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


df_train.head()


df_train.info()


categorical_cols = []
num_cols = []

for col in df_train.columns:
    if df_train[col].dtype not in ["int64", "float64"]:
        categorical_cols.append(col)
    else:
        num_cols.append(col)
        


def frequency_encode(train_df, test_df, columns):
    for col in columns:
        freq = train_df[col].value_counts() / len(train_df)
        train_df[col + "_freq"] = train_df[col].map(freq)
        test_df[col + "_freq"] = test_df[col].map(freq).fillna(0)
    return train_df, test_df

df_train, df_test = frequency_encode(df_train, df_test, categorical_cols)


education_map = {
    "No formal": 0,
    "Highschool": 1,
    "Graduate": 2,
    "Postgraduate": 3
}

income_map = {
    "Low": 0,
    "Lower-Middle": 1,
    "Middle": 2,
    "Upper-Middle": 3,
    "High": 4
}

smoking_map = {
    "Never": 0,
    "Former": 1,
    "Current": 2
}

employment_map = {
    "Student": 0,
    "Unemployed": 1,
    "Employed": 2,
    "Retired": 3
}

ordinal_cols = {
    "education_level": education_map,
    "income_level": income_map,
    "smoking_status": smoking_map,
    "employment_status": employment_map
}

for col, mapping in ordinal_cols.items():
    df_train[col] = df_train[col].map(mapping)
    df_test[col] = df_test[col].map(mapping)



df_train['physical_activity_minutes_per_week_log'] = (
    np.log1p(df_train['physical_activity_minutes_per_week'])
)

df_test['physical_activity_minutes_per_week_log'] = (
    np.log1p(df_test['physical_activity_minutes_per_week'])
)

# df_train.drop()


df_train.info()


df_train.info()


cat_cols = df_train.select_dtypes(include="object").columns.tolist()
for col in df_test.columns:
    df_train[col] = df_train[col].astype("category")
    df_test[col] = df_test[col].astype("category")


df_train.info()


test_ids = df_test["id"].copy()
df_train.drop(columns=['id'], inplace = True)
df_test.drop(columns=['id'], inplace = True)


X_train = df_train.drop(columns=["diagnosed_diabetes"])
y_train = df_train["diagnosed_diabetes"]


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 20, 256),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "device": "gpu",
        "random_state": 42,
        "verbose": -1
    }

    model = LGBMClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        categorical_feature=categorical_cols,
    )

    preds = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, preds)
    return auc


# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)


lgb_params = {'n_estimators': 2993, 'learning_rate': 0.013436964724333552, 'max_depth': 9, 'num_leaves': 68, 'min_data_in_leaf': 137, 'feature_fraction': 0.5096337484904506, 'bagging_fraction': 0.859675489462859, 'bagging_freq': 2, 'lambda_l1': 2.4780337254374047, 'lambda_l2': 3.148440274240726, 'verbose':-1}
# 0.7277226307166502.
xgb_params = {'max_depth': 7, 'learning_rate': 0.041051535254900795, 'subsample': 0.8918591748067689, 'colsample_bytree': 0.6870058089135236, 'min_child_weight': 11, 'reg_alpha': 4.427944890304783, 'reg_lambda': 2.4063419811501303, 'n_estimators': 2999, 'tree_method':"hist", # Required for categorical support
    'enable_categorical':True}


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from lightgbm import LGBMClassifier


def run_cv_model(model, model_name, X, y, df_test, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(df_test))

    fold_scores = []

    print(f"\n==================== {model_name} CV START ====================\n")

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        model = LGBMClassifier(**lgb_params)
        model.fit(X_tr, y_tr)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred

        auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(auc)
        print(f"Fold {fold+1} AUC: {auc:.5f}")

        test_preds += model.predict_proba(df_test)[:, 1] / n_splits

    full_auc = roc_auc_score(y, oof_preds)
    print(f"\n>>> {model_name} Full OOF AUC: {full_auc:.5f}")
    print("===========================================================\n")

    return oof_preds, test_preds



lgb_model = LGBMClassifier(**lgb_params)

oof_lgb, test_lgb = run_cv_model(
    model=lgb_model,
    model_name="LightGBM",
    X=X_train,
    y=y_train,
    df_test=df_test,
    n_splits=5
)



# xgb = XGBClassifier(**xgb_params)

# # oof_xgb, test_xgb = run_cv_model(
# #     model=xgb,
# #     model_name="XGBoost",
# #     X=X_train,
# #     y=y_train,
# #     df_test=df_test,
# #     n_splits=10
# # )

# xgb.fit(X_train, y_train)


# from sklearn.linear_model import LogisticRegression
# st = StandardScaler()
# X_train_scaled = st.fit_transform(X_train)
# X_test_scaled = st.transform(df_test)
# lr = LogisticRegression()
# lr.fit(X_train_scaled, y_train)


submission = pd.DataFrame({
    'id':test_ids,
    'diagnosed_diabetes': test_lgb
})
submission.to_csv("submission.csv", index=False)

submission.head()


