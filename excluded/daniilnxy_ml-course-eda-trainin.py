import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


import optuna
from optuna.pruners import SuccessiveHalvingPruner 




train_df = pd.read_csv('/kaggle/input/playground-series-s4e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e7/test.csv')
sub_df = pd.read_csv('/kaggle/input/playground-series-s4e7/sample_submission.csv')


train_df.shape


train_df.head()


train_df.isna().sum()



train_df.nunique()



train_df.describe()



train_df.dtypes



train_df.shape



test_df.head()



test_df.info()



test_df.isna().sum()



test_df.shape



plt.figure(figsize=(8, 6))

sns.set_style('whitegrid')
sns.set_palette('pastel')

ax = sns.countplot(x='Response', data=train_df, order=[0, 1]) 
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.title('Response distribution',fontsize=16)
plt.xlabel('0 or 1', fontsize=14)
plt.ylabel('Колво чел', fontsize=14)

plt.show()


train_df['Response'].value_counts(normalize=True)


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

sns.countplot(data=train_df, x="Gender", ax=axes[0])
axes[0].set_title("Gender (train)")
axes[0].set_xlabel("Gender")

sns.countplot(data=test_df, x="Gender", ax=axes[1])
axes[1].set_title("Gender (test)")
axes[1].set_xlabel("Gender")

plt.tight_layout()
plt.show()



gender_resp = (
    train_df.groupby("Gender")["Response"]
    .mean()
    .reset_index()
    .rename(columns={"Response": "response_rate"})
)

fig, ax = plt.subplots()
sns.barplot(data=gender_resp, x="Gender", y="response_rate", ax=ax)
ax.set_title("доля Response=1 по полу")
ax.set_ylabel("Response rate")
plt.show()

print(gender_resp)



top_regions = (
    train_df["Region_Code"].value_counts()
    .head(20)
    .index
)

fig, ax = plt.subplots(figsize=(10, 4))
sns.countplot(
    data=train_df[train_df["Region_Code"].isin(top_regions)],
    x="Region_Code",
    order=top_regions,
    ax=ax
)
ax.set_title("топ 20 регионов по числу клиентов на основе трейн данных")
ax.set_xlabel("Region_Code")
ax.set_ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



region_resp = (
    train_df[train_df["Region_Code"].isin(top_regions)]
    .groupby("Region_Code")["Response"]
    .mean()
    .reset_index()
    .rename(columns={"Response": "response_rate"})
)

region_resp_sorted = region_resp.sort_values("response_rate", ascending=False)

order_sorted = region_resp_sorted["Region_Code"].tolist()

plt.figure(figsize=(12, 5))
sns.barplot(
    data=region_resp_sorted,
    x="Region_Code",
    y="response_rate",
    order=order_sorted
)
plt.title("response rate по топ-20 Region_Code")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

sns.countplot(data=train_df, x="Vehicle_Age", ax=axes[0])
axes[0].set_title("Vehicle_Age (train)")
axes[0].set_xlabel("Vehicle_Age")
axes[0].tick_params(axis="x", rotation=45)

sns.countplot(data=test_df, x="Vehicle_Age", ax=axes[1])
axes[1].set_title("Vehicle_Age (test)")
axes[1].set_xlabel("Vehicle_Age")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

va_resp = (
    train_df.groupby("Vehicle_Age")["Response"]
    .mean()
    .reset_index()
    .rename(columns={"Response": "response_rate"})
)

fig, ax = plt.subplots()
sns.barplot(data=va_resp, x="Vehicle_Age", y="response_rate", ax=ax)
ax.set_title("Response rate по Vehicle_Age")
ax.set_ylabel("Response rate")
ax.tick_params(axis="x", rotation=45)
plt.show()

print(va_resp)



top_channels = (
    train_df["Policy_Sales_Channel"].value_counts()
    .head(20)
    .index
)

fig, ax = plt.subplots(figsize=(10, 4))
sns.countplot(
    data=train_df[train_df["Policy_Sales_Channel"].isin(top_channels)],
    x="Policy_Sales_Channel",
    order=top_channels,
    ax=ax
)
ax.set_title("топ 20 каналов продаж (train)")
ax.set_xlabel("Policy_Sales_Channel")
ax.set_ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



num_features = [
    "Age",
    "Annual_Premium",
    "Vintage",
    "Driving_License",
    "Previously_Insured",
    "Response",
]

corr_matrix = train_df[num_features].corr()

corr_with_target = (
    corr_matrix["Response"]
    .drop("Response")
    .sort_values(ascending=False)
)


print("крреляция признаков с Response:\n")
print(corr_with_target)

plt.figure(figsize=(7, 4))
sns.barplot(
    x=corr_with_target.index,
    y=corr_with_target.values,
    palette="viridis"
)
plt.title("Корреляция признаков с Response (отсортировано)")
plt.ylabel("Correlation")
plt.xlabel("Feature")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



tmp = train_df.copy()

resp_vd = (
    tmp.groupby("Vehicle_Damage")["Response"]
    .mean()
    .reset_index()
    .rename(columns={"Response": "response_rate"})
    .sort_values("response_rate", ascending=False)
)

print(resp_vd)

plt.figure(figsize=(8, 4))
sns.barplot(data=resp_vd, x="Vehicle_Damage", y="response_rate")
plt.title("Response rate по Vehicle_Damage")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



tmp = train_df.copy()

resp_va = (
    tmp.groupby("Vehicle_Age")["Response"]
    .mean()
    .reset_index()
    .rename(columns={"Response": "response_rate"})
    .sort_values("response_rate", ascending=False)
)

print(resp_va)

plt.figure(figsize=(8, 4))
sns.barplot(data=resp_va, x="Vehicle_Age", y="response_rate")
plt.title("Response rate по Vehicle_Age")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




feature_cols = [
    "Gender",
    "Age",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage",
]

num_cols = train_df[feature_cols].select_dtypes(include=["int64", "float64"]).columns

corr = train_df[num_cols].corr().abs()

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

high_corr_pairs = [
    (col, row, upper.loc[row, col])
    for col in upper.columns
    for row in upper.index
    if not np.isnan(upper.loc[row, col]) and upper.loc[row, col] > 0.999
]

high_corr_pairs



plt.figure(figsize=(8, 4))
sns.histplot(train_df["Annual_Premium"], bins=100, kde=True)
plt.title("Распределение Annual_Premium (сырое)")
plt.xlabel("Annual_Premium")
plt.ylabel("Count")
plt.show()



cat_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()




for feat in cat_cols:
    print(train_df[feat].nunique())


id_col = 'id'
target_col = "Response"


test_ids = test_df[id_col].copy()


train_df = train_df.drop(columns=[id_col])
test_df = test_df.drop(columns=[id_col])


feature_cols = [c for c in train_df.columns if c != target_col]



X_raw = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test_raw = test_df[feature_cols].copy()


cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()


if "Annual_Premium" in X_raw.columns:
    X_raw["Annual_Premium_log"] = np.log1p(X_raw["Annual_Premium"])
    X_test_raw["Annual_Premium_log"] = np.log1p(X_test_raw["Annual_Premium"])
    if "Annual_Premium_log" not in num_cols:
        num_cols.append("Annual_Premium_log")



N_TUNE = 800_000 

rng = np.random.RandomState(42)
tune_idx = rng.choice(len(X_raw), size=N_TUNE, replace=False)

X_tune = X_raw.iloc[tune_idx].reset_index(drop=True)
y_tune = y.iloc[tune_idx].reset_index(drop=True)



skf_tune = StratifiedKFold(
    n_splits=3, 
    shuffle=True,
    random_state=42
)



preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)


def objective_logreg(trial: optuna.Trial) -> float:
    C = trial.suggest_float("C", 1e-3, 100.0, log=True)
    class_weight_choice = trial.suggest_categorical("class_weight", [None, "balanced"])

    logreg_clf = LogisticRegression(
        C=C,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        n_jobs=-1,
        class_weight=class_weight_choice
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", logreg_clf),
    ])

    scores = []

    for fold_idx, (trn_idx, val_idx) in enumerate(skf_tune.split(X_tune, y_tune), start=1):
        X_tr, X_val = X_tune.iloc[trn_idx], X_tune.iloc[val_idx]
        y_tr, y_val = y_tune.iloc[trn_idx], y_tune.iloc[val_idx]
    
        model.fit(X_tr, y_tr)
        val_pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, val_pred)
        scores.append(score)

        trial.report(score, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


study_logreg = optuna.create_study(
    direction="maximize",
    pruner=SuccessiveHalvingPruner()
)


study_logreg.optimize(
    objective_logreg,
    n_trials=30,              
    show_progress_bar=True
)



best_params_logreg = study_logreg.best_params
best_score_logreg = study_logreg.best_value
print("лучшицй CV AUC:", best_score_logreg)
print("лучшие гиперпарамы логрега:", best_params_logreg)


final_logreg_clf = LogisticRegression(
    C=best_params_logreg["C"],
    penalty="l2",
    solver="lbfgs",
    max_iter=2000,
    n_jobs=-1,
    class_weight=best_params_logreg["class_weight"],
)

final_logreg_model = Pipeline([
    ("prep", preprocessor),
    ("clf", final_logreg_clf),
])

final_logreg_model.fit(X_raw, y)
test_pred_logreg = final_logreg_model.predict_proba(X_test_raw)[:, 1]



test_pred_logreg


def objective_xgb(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 700),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }

    xgb_clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",   
        device="cuda",        
        random_state=42,
        n_jobs=-1,
        **params
    )


    model = Pipeline([
        ("prep", preprocessor),
        ("clf", xgb_clf),
    ])

    scores = []

    for fold_idx, (trn_idx, val_idx) in enumerate(skf_tune.split(X_tune, y_tune), start=1):
        X_tr, X_val = X_tune.iloc[trn_idx], X_tune.iloc[val_idx]
        y_tr, y_val = y_tune.iloc[trn_idx], y_tune.iloc[val_idx]
    
        model.fit(X_tr, y_tr)
        val_pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, val_pred)
        scores.append(score)

        trial.report(score, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


study_xgb = optuna.create_study(
    direction="maximize",
    pruner=SuccessiveHalvingPruner()
)

study_xgb.optimize(
    objective_xgb,
    n_trials=30,
    show_progress_bar=True
)



best_params_xgb = study_xgb.best_params
best_score_xgb = study_xgb.best_value
print("лучший AUC:", best_score_xgb)
print("XGB лучшие парамы:", best_params_xgb)


final_xgb_clf = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    device="cuda",
    random_state=42,
    n_jobs=-1,
    **best_params_xgb
)

final_xgb_model = Pipeline([
    ("prep", preprocessor),
    ("clf", final_xgb_clf),
])

final_xgb_model.fit(X_raw, y)
test_pred_xgb = final_xgb_model.predict_proba(X_test_raw)[:, 1]


test_pred_xgb


def objective_lgbm(trial: optuna.Trial) -> float:

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 700),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", -1, 12), 
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.05),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),

        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "boosting_type": "gbdt",
    }

    clf = LGBMClassifier(
        objective="binary",
        n_jobs=-1,
        verbosity=-1,
        **params,
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", clf)
    ])

    scores = []

    for fold_idx, (trn_idx, val_idx) in enumerate(skf_tune.split(X_tune, y_tune), start=1):

        X_tr, X_val = X_tune.iloc[trn_idx], X_tune.iloc[val_idx]
        y_tr, y_val = y_tune.iloc[trn_idx], y_tune.iloc[val_idx]

        model.fit(
            X_tr,
            y_tr,
            clf__eval_metric="auc"
        )

        preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        scores.append(score)

        trial.report(score, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


study_lgbm = optuna.create_study(
    direction="maximize",
    pruner=SuccessiveHalvingPruner()
)

study_lgbm.optimize(
    objective_lgbm,
    n_trials=30,
    show_progress_bar=True
)



best_params_lgbm = study_lgbm.best_params
best_score_lgbm = study_lgbm.best_value
print("лучший AUC:", best_score_lgbm)
print("LGBM лучшие парамы:", best_params_lgbm)


from sklearn.feature_selection import VarianceThreshold

safe_best_params = {k: v for k, v in best_params_lgbm.items()
                    if k not in ("device", "gpu_platform_id", "gpu_device_id")}

final_lgbm_clf = LGBMClassifier(
    objective="binary",
    n_jobs=-1,
    boosting_type="gbdt",
    verbosity=-1,
    **safe_best_params,
)

final_lgbm_model = Pipeline([
    ("prep", preprocessor),
    ("var", VarianceThreshold(threshold=0.0)),   
    ("clf", final_lgbm_clf),
])

final_lgbm_model.fit(
    X_raw,
    y,
    clf__eval_metric="auc"
)

test_pred_lgbm = final_lgbm_model.predict_proba(X_test_raw)[:, 1]



cat_features_idx = [X_raw.columns.get_loc(c) for c in cat_cols]

def objective_catboost(trial: optuna.Trial) -> float:
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "iterations": trial.suggest_int("iterations", 200, 800),
    }

    scores = []

    for fold_idx, (trn_idx, val_idx) in enumerate(skf_tune.split(X_tune, y_tune), start=1):
        X_tr, X_val = X_tune.iloc[trn_idx], X_tune.iloc[val_idx]
        y_tr, y_val = y_tune.iloc[trn_idx], y_tune.iloc[val_idx]
    
        cat_clf = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            task_type="GPU",
            devices="0",
            **params
        )

        cat_clf.fit(
            X_tr, y_tr,
            cat_features=cat_features_idx
        )
        val_pred = cat_clf.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, val_pred)
        scores.append(score)

        trial.report(score, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


study_cat = optuna.create_study(
    direction="maximize",
    pruner=SuccessiveHalvingPruner()
)

study_cat.optimize(
    objective_catboost,
    n_trials=30,
    show_progress_bar=True
)



best_params_cat = study_cat.best_params
best_score_cat = study_cat.best_value
print("лушчи йAUC:", best_score_cat)
print("лучшие параы кэтбуста:", best_params_cat)


final_cat_clf = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    task_type="GPU",
    devices="0",
    random_seed=42,
    verbose=False,
    **best_params_cat
)

final_cat_clf.fit(
    X_raw,
    y,
    cat_features=cat_features_idx
)

test_pred_cat = final_cat_clf.predict_proba(X_test_raw)[:, 1]



sub_df.head(3)




