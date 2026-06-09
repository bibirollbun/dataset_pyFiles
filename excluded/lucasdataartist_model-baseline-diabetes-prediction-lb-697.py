# Packages 
# Data Processing 
import numpy as np 
import pandas as pd 
import pandas.api.types
# Visualization 
import matplotlib.pyplot as plt 
plt.rcParams['figure.dpi'] = 200 
import seaborn as sns 
# Statistics 
import math 
from scipy import stats 
from scipy.stats import norm 
# File Path 
import os 
for dirname, _, filenames in os.walk('/kaggle/input'): 
    for filename in filenames: 
        print(os.path.join(dirname, filename))


# version check
print(f"numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")

# # ignore Warning
# import warnings
# warnings.filterwarnings("ignore")

# setting
path_root = "/kaggle/input/playground-series-s5e12/"
seed = 394
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


# load data
df_train = pd.read_csv(path_root + "train.csv")
print("Train shape:",df_train.shape)
df_test = pd.read_csv(path_root + "test.csv")
print("Test shape:", df_test.shape)


# features
target = 'diagnosed_diabetes'
list_not_features = ['id', 'diagnosed_diabetes']
list_features = [c for c in df_train.columns if not c in list_not_features]

# categorical features
list_categorical_features = df_train.select_dtypes(include=["object", "category"]).columns.tolist()
list_categorical_features = list(set(list_features).intersection(set(list_categorical_features)))
list_numeric_features = list(set(list_features) - set(list_categorical_features))

print(f"Numeric features({len(list_numeric_features)}): {list_numeric_features}")
print(f"Categorical features({len(list_categorical_features)}): {list_categorical_features}")


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder


# preprocess pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            list_categorical_features
        ),
        (
            "num",
            "passthrough",
            list_numeric_features
        )
    ]
)


from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# models
dict_models = {
    "lgbm": LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=seed,
        n_jobs=-1,
        verbose=-1
    ),
    "xgb": XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        use_label_encoder=False
    ),
    "cat": CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_state=seed,
        verbose=False
    ),
}


# pipeline
dict_pipelines = {
    "lgbm": Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", dict_models["lgbm"])
        ]
    ),
    "xgb": Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", dict_models["xgb"])
        ]
    ),
    "cat": Pipeline(
        steps=[
            ("model", dict_models["cat"])
        ]
    )
}


from sklearn.model_selection import StratifiedKFold


n_folds = 10
skf = StratifiedKFold(
    n_splits=n_folds,
    shuffle=True,
    random_state=seed
)


list_splits = list(
    skf.split(
        df_train[list_features],
        df_train[target]
    )
)


from sklearn.metrics import roc_auc_score, roc_curve


%%time

# model fitting

dict_oof_pred  = {}
dict_test_pred = {}
dict_auc_score = {}
arr_y = df_train[target].values
df_X_test = df_test[list_features].copy()

for model_name in dict_models.keys():
    print(f"\n==============================")
    print(f"model: {model_name}")
    print(f"==============================")

    # OOF, test fold
    arr_oof_pred = np.zeros(len(df_train), dtype=float)
    arr_test_pred_folds = np.zeros((len(df_test), skf.n_splits))

    fold = 0

    for arr_idx_tr, arr_idx_va in list_splits:
        fold += 1
        print(f"[{model_name}] Fold {fold} Start")

        df_X_tr = df_train.iloc[arr_idx_tr][list_features]
        df_X_va = df_train.iloc[arr_idx_va][list_features]
        arr_y_tr = arr_y[arr_idx_tr]
        arr_y_va = arr_y[arr_idx_va]

        # fitting
        pipeline = dict_pipelines[model_name]
        
        if model_name == "cat":
            model_cat = dict_models["cat"]
            # column index for CatBoost model
            list_cat_idx = [list_features.index(col) for col in list_categorical_features]
            model_cat.fit(df_X_tr, arr_y_tr, cat_features=list_cat_idx)
            # validation
            arr_pred_va = model_cat.predict_proba(df_X_va)[:, 1]
            arr_oof_pred[arr_idx_va] = arr_pred_va
            # predict test data
            arr_test_pred_folds[:, fold - 1] = model_cat.predict_proba(df_X_test)[:, 1]
        else:
            pipeline.fit(df_X_tr, arr_y_tr)
            # validation
            arr_pred_va = pipeline.predict_proba(df_X_va)[:, 1]
            arr_oof_pred[arr_idx_va] = arr_pred_va
            # predict test data
            arr_test_pred_folds[:, fold - 1] = pipeline.predict_proba(df_X_test)[:, 1]

        # AUC
        auc_fold = roc_auc_score(arr_y_va, arr_pred_va)
        print(f"[{model_name}] Fold {fold} AUC: {auc_fold:.5f}")
        print(f"--------------------")

    # OOF AUC
    auc_oof = roc_auc_score(arr_y, arr_oof_pred)
    dict_oof_pred[model_name] = arr_oof_pred
    dict_test_pred[model_name] = arr_test_pred_folds.mean(axis=1)
    dict_auc_score[model_name] = auc_oof

    print(f"[{model_name}] OOF AUC: {auc_oof:.5f}")


# mean
arr_test_pred_ensemble = np.mean(
    [
        dict_test_pred["lgbm"],
        dict_test_pred["xgb"],
        dict_test_pred["cat"],
    ],
    axis=0
)


df_submission = pd.read_csv(path_root + "sample_submission.csv")
print("Submission shape:",df_submission.shape)


df_submission[target] = dict_test_pred["lgbm"]
df_submission.to_csv("submission_pipeline_baseline_lgbm.csv", index=False)


df_submission[target] = dict_test_pred["xgb"]
df_submission.to_csv("submission_pipeline_baseline_xgb.csv", index=False)


df_submission[target] = dict_test_pred["cat"]
df_submission.to_csv("submission_pipeline_baseline_cat.csv", index=False)


df_submission[target] = arr_test_pred_ensemble
df_submission.to_csv("submission_pipeline_baseline_ensemble_softvoting.csv", index=False)

