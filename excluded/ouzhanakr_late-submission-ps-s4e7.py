# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import polars as pl
from catboost import CatBoostClassifier

import optuna
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.simplefilter("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


trainpl = pl.read_csv('/kaggle/input/playground-series-s4e7/train.csv')
testpl = pl.read_csv('/kaggle/input/playground-series-s4e7/test.csv')
subpl = pl.read_csv('/kaggle/input/playground-series-s4e7/sample_submission.csv')


train = trainpl.to_pandas()
test  = testpl.to_pandas()


ID_COL = "id"
TARGET = "Response"


train.head()


train.isnull().sum()


train.info()


class InsurancePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.map_gender = {"Male": 0, "Female": 1}
        self.map_vehicle_age = {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}
        self.map_vehicle_damage = {"No": 0, "Yes": 1}

        self.cross_base_cols = ["Annual_Premium", "Vehicle_Age", "Vehicle_Damage", "Vintage"]

        self.cross_maps_ = {}
        self.output_columns_ = None
        self.categorical_columns_ = None

    def basic_encodings(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if "Gender" in X:
            X["Gender"] = (
                X["Gender"].astype(str).str.strip().str.title().map(self.map_gender)
                .fillna(-1).astype("int32")
            )

        if "Vehicle_Age" in X:
            X["Vehicle_Age"] = X["Vehicle_Age"].map(self.map_vehicle_age).fillna(-1).astype("int32")

        if "Vehicle_Damage" in X:
            X["Vehicle_Damage"] = X["Vehicle_Damage"].map(self.map_vehicle_damage).fillna(-1).astype("int32")

        for col in ["Region_Code", "Annual_Premium", "Policy_Sales_Channel"]:
            if col in X:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(-1).astype("int32")

        for col in ["Age", "Vintage"]:
            if col in X:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(-1).astype("int32")

        for col in ["Driving_License", "Previously_Insured"]:
            if col in X:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(-1).astype("int32")

        return X

    def build_cross_series(self, X: pd.DataFrame, base_col: str) -> pd.Series:
        return X["Previously_Insured"].astype(str) + "_" + X[base_col].astype(str)

    def fit(self, X: pd.DataFrame, y=None):
        Xk = self.basic_encodings(X)

        for base in self.cross_base_cols:
            key_name = f"Previously_Insured_{base}"
            combo = self.build_cross_series(Xk, base)
            uniques = pd.unique(combo)
            self.cross_maps_[key_name] = {v: i for i, v in enumerate(uniques)}

        self.output_columns_ = list(Xk.columns) + list(self.cross_maps_.keys())

        self.categorical_columns_ = [
            "Gender", "Driving_License", "Region_Code", "Previously_Insured",
            "Vehicle_Age", "Vehicle_Damage", "Policy_Sales_Channel",
            "Previously_Insured_Annual_Premium", "Previously_Insured_Vehicle_Age",
            "Previously_Insured_Vehicle_Damage", "Previously_Insured_Vintage"
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xk = self.basic_encodings(X)

        for key_name, mapping in self.cross_maps_.items():
            base = key_name.replace("Previously_Insured_", "")  
            combo = self.build_cross_series(Xk, base)
            Xk[key_name] = combo.map(mapping).fillna(-1).astype("int32")  

        cols_present = [i for i in self.output_columns_ if i in Xk.columns]  
        return Xk[cols_present]



features = [c for c in train.columns if c not in [ID_COL, TARGET]]
X_all = train[features]
y_all = train[TARGET].astype("int32")


skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
aucs = []
test_pred_list = []



X_test_raw = test[features]


for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all), start=1):
    print(f"### Fold {fold} Training ###")

    X_tr_raw, y_tr = X_all.iloc[tr_idx].copy(), y_all.iloc[tr_idx].copy()
    X_va_raw, y_va = X_all.iloc[va_idx].copy(), y_all.iloc[va_idx].copy()

    prep = InsurancePreprocessor().fit(X_tr_raw, y_tr)
    X_tr = prep.transform(X_tr_raw)
    X_va = prep.transform(X_va_raw)
    X_te = prep.transform(X_test_raw)

    cat_cols = [c for c in prep.categorical_columns_ if c in X_tr.columns]

    model = CatBoostClassifier(
        task_type="CPU",
        loss_function="Logloss",
        eval_metric="AUC",
        learning_rate=0.1,          
        iterations=200,              
        depth=4,                     
        l2_leaf_reg=2.0,
        bootstrap_type="Bernoulli",  
        subsample=0.5,
        rsm=0.8,                     
        max_ctr_complexity=1,        
        border_count=64,             
        thread_count=-1,
        allow_writing_files=False,
        verbose=100
    )


    model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=200,
        cat_features=cat_cols
    )

    p_val = model.predict_proba(X_va)[:, 1]
    p_te  = model.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_va, p_val)
    aucs.append(auc)
    test_pred_list.append(p_te)

    print(f"Fold {fold} AUC: {auc:.5f}\n")

print(f"\nOverall AUC: {np.mean(aucs):.5f} +/- {np.std(aucs):.5f}")



# test: 0.8667194	best: 0.8667194 (0)	total: 20.2s	remaining: 11h 12m 31s


# for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all), start=1):
#     print(f"### Fold {fold} Training ###")

#     X_tr_raw, y_tr = X_all.iloc[tr_idx].copy(), y_all.iloc[tr_idx].copy()
#     X_va_raw, y_va = X_all.iloc[va_idx].copy(), y_all.iloc[va_idx].copy()

#     prep = InsurancePreprocessor().fit(X_tr_raw, y_tr)
#     X_tr = prep.transform(X_tr_raw)
#     X_va = prep.transform(X_va_raw)
#     X_te = prep.transform(X_test_raw)

#     cat_cols = [c for c in prep.categorical_columns_ if c in X_tr.columns]

#     model = CatBoostClassifier(
#         task_type="CPU",
#         loss_function="Logloss",
#         eval_metric="AUC",
#         learning_rate=0.05,
#         iterations=2000,          
#         depth=8,                  
#         random_strength=0.0,
#         l2_leaf_reg=1.0,          
#         random_seed=42,
#         verbose=500,             
#         thread_count=-1,          
#         allow_writing_files=False 
#     )

#     model.fit(
#         X_tr, y_tr,
#         eval_set=(X_va, y_va),
#         early_stopping_rounds=200,
#         cat_features=cat_cols
#     )

#     p_val = model.predict_proba(X_va)[:, 1]
#     p_te  = model.predict_proba(X_te)[:, 1]

#     auc = roc_auc_score(y_va, p_val)
#     aucs.append(auc)
#     test_pred_list.append(p_te)

#     print(f"Fold {fold} AUC: {auc:.5f}\n")

# print(f"\nOverall AUC: {np.mean(aucs):.5f} +/- {np.std(aucs):.5f}")



#----------ooooooooooooo-----------


# features = [c for c in train.columns if c not in [ID_COL, TARGET]]
# X_all = train[features]
# y_all = train[TARGET].astype("int32")
# X_test_raw = test[features]



# pos_ratio = float(y_all.mean())
# scale_pos_weight = ((1.0 - pos_ratio) / pos_ratio) if pos_ratio > 0 else 1.0






# features [i for i in train.coolumns if i not in [ID_COL, TARGET]]
# X_all = train[features]
# y_all = train[TARGET].astype('int32')

# def cv_mean_auc(params, n_splits=5, use_gpu=Fasle, verbose_eval=1000):
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     aucs = []

#     for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all), start=1):
#         X_tr_raw, y_tr = X_all.iloc[tr_idx], y_all.iloc[tr_idx]
#         X_va_raw, y_va = X_all.iloc[va_idx], y_all.iloc[va_idx]

#         prep = InsurancePreprocessor().fit(X_tr_raw, y_tr)
#         X_tr = prep.transform(X_tr_raw)
#         X_va = prep.transform(X_va_raw)

#         cat_cols = [c for c in prep.categorical_columns_ if c in X_tr.columns]
#         cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]

#         model = CatBoostClassifier(
#             loss_function="Logloss",
#             eval_metric="AUC",
#             task_type="GPU" if use_gpu else "CPU",
#             random_seed=42,
#             verbose=False,
#             iterations=params.get("iterations", 5000),
#             depth=params["depth"],
#             learning_rate=params["learning_rate"],
#             l2_leaf_reg=params["l2_leaf_reg"],
#             random_strength=params["random_strength"],
#             bagging_temperature=params["bagging_temperature"],
#             border_count=params["border_count"],
#             scale_pos_weight=params.get("scale_pos_weight", scale_pos_weight)
#         )

#         model.fit(
#             X_tr, y_tr,
#             eval_set=(X_va, y_va),
#             early_stopping_rounds=200,
#             verbose=verbose_eval,
#             cat_features=cat_idx
#         )

#         p_va = model.predict_proba(X_va)[:, 1]
#         aucs.append(roc_auc_score(y_va, p_va))

#     return float(np.mean(aucs))


# def run_cv(
#     params: dict,
#     X: pd.DataFrame,
#     y: pd.Series,
#     X_test: pd.DataFrame | None = None,
#     n_splits: int = 5,
#     seed: int = 42,
#     use_gpu: bool = True,
#     verbose_eval: int = 500,
#     early_stopping_rounds: int = 200,
#     trial: optuna.trial.Trial | None = None
# ):
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     oof = np.full(len(X), np.nan, dtype=float)
#     aucs = []
#     test_pred = np.zeros(len(X_test)) if X_test is not None else None

#     for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
#         X_tr_raw, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
#         X_va_raw, y_va = X.iloc[va_idx], y.iloc[va_idx]

#         prep = InsurancePreprocessor().fit(X_tr_raw, y_tr)
#         X_tr = prep.transform(X_tr_raw)
#         X_va = prep.transform(X_va_raw)
#         if X_test is not None:
#             X_te = prep.transform(X_test)

#         cat_cols = [c for c in prep.categorical_columns_ if c in X_tr.columns]
#         cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]

#         model = CatBoostClassifier(
#             loss_function="Logloss",
#             eval_metric="AUC",
#             task_type="GPU" if use_gpu else "CPU",
#             random_seed=seed,
#             verbose=False,
#             iterations=5000,  
#             depth=params["depth"],
#             learning_rate=params["learning_rate"],
#             l2_leaf_reg=params["l2_leaf_reg"],
#             random_strength=params["random_strength"],
#             bagging_temperature=params["bagging_temperature"],  
#             bootstrap_type="Bayesian",
#             border_count=params["border_count"],
#             scale_pos_weight=params.get("scale_pos_weight", scale_pos_weight),
#         )

#         model.fit(
#             X_tr, y_tr,
#             eval_set=(X_va, y_va),
#             early_stopping_rounds=early_stopping_rounds,
#             verbose=verbose_eval,
#             cat_features=cat_idx
#         )

#         p_va = model.predict_proba(X_va)[:, 1]
#         oof[va_idx] = p_va
#         fold_auc = roc_auc_score(y_va, p_va)
#         aucs.append(fold_auc)
#         print(f"[Fold {fold}] AUC: {fold_auc:.6f}")

#         if trial is not None:
#             trial.report(fold_auc, step=fold)
#             if trial.should_prune():
#                 raise optuna.TrialPruned()

#         if X_test is not None:
#             p_te = model.predict_proba(X_te)[:, 1]
#             test_pred += p_te / n_splits

#     return {
#         "mean_auc": float(np.mean(aucs)),
#         "std_auc": float(np.std(aucs)),
#         "oof": oof,
#         "fold_aucs": aucs,
#         "test_pred": test_pred
#     }



# def suggest_params(trial):
#     return {
#         "depth": trial.suggest_int("depth", 5, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
#         "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
#         "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
#         "border_count": trial.suggest_int("border_count", 64, 255),
#         "scale_pos_weight": scale_pos_weight,  
#     }



# def objective(trial):
#     params = suggest_params(trial)
#     result = run_cv(
#         params=params,
#         X=X_all,
#         y=y_all,
#         X_test=None,           
#         n_splits=5,
#         seed=42,
#         use_gpu=False,
#         verbose_eval=0,        
#         early_stopping_rounds=200,
#         trial=trial            
#     )
#     return result["mean_auc"]



# sampler = optuna.samplers.TPESampler(seed=42)
# pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)

# study = optuna.create_study(
#     direction='maximize',
#     sampler=sampler,
#     pruner=pruner,
#     study_name='catboost_insurance_optuna_single_cv'
# )
# study.optimize(objective, n_trials= 1, show_progress_bar=True)


# print("Best AUC:", study.best_value)
# print("Best params:", study.best_trial.params)


# best_params = study.best_trial.params
# final = run_cv(
#     params=best_params,
#     X=X_all,
#     y=y_all,
#     X_test=X_test_raw,   
#     n_splits=5,
#     seed=42,
#     use_gpu=True,
#     verbose_eval=500,
#     early_stopping_rounds=200,
#     trial=None
# )


submission = pd.DataFrame({
    "id": test[ID_COL],
    "Response": np.mean(test_pred_list, axis=0)
})
submission.to_csv("submission.csv", index=False)



submission.head()




