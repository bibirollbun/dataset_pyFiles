# load packages
import pandas as pd
import numpy as np

import optuna
import gc

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans


def cleanup_gpu(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def get_cat_cols(df: pd.DataFrame):
    """CatBoost needs column names for cat features."""
    cat_cols = []
    for c in df.columns:
        dt = df[c].dtype
        if str(dt) == "category" or dt == object or str(dt) == "bool":
            cat_cols.append(c)
    return cat_cols



train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sub   = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

target_col = "diagnosed_diabetes"
id_col = "id"


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        use_kmeans=True,
        kmeans_clusters=(6, 12),
        kmeans_max_features=12,
        interaction_pairs=None,
        random_state=42,
    ):
        self.use_kmeans = use_kmeans
        self.kmeans_clusters = kmeans_clusters
        self.kmeans_max_features = kmeans_max_features
        self.random_state = random_state
        self.interaction_pairs = interaction_pairs or [
            ("gender", "smoking_status"),
            ("education_level", "income_level"),
            ("gender", "ethnicity"),
            ("smoking_status", "family_history_diabetes"),
        ]

        self.numeric_cols_ = []
        self.cat_cols_ = []
        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()

        self.kmeans_models_ = {}
        self.kmeans_feature_cols_ = []

        self.feature_names_ = []

    def _infer_cols(self, X: pd.DataFrame):
        df = X.copy()
        drop_like = set([c for c in ["id", "diagnosed_diabetes"] if c in df.columns])

        num_cols = []
        cat_cols = []

        for c in df.columns:
            if c in drop_like:
                continue

            dtype = df[c].dtype

            if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
                cat_cols.append(c)
            elif pd.api.types.is_bool_dtype(dtype):
                cat_cols.append(c)
            elif pd.api.types.is_numeric_dtype(dtype):
                nun = df[c].nunique(dropna=True)
                if nun <= 20 and pd.api.types.is_integer_dtype(dtype):
                    cat_cols.append(c)
                else:
                    num_cols.append(c)
            else:
                cat_cols.append(c)

        self.numeric_cols_ = sorted(list(dict.fromkeys(num_cols)))
        self.cat_cols_ = sorted(list(dict.fromkeys(cat_cols)))

    @staticmethod
    def _safe_div(a, b, eps=1e-6):
        return a / (b + eps)

    def _add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if {"systolic_bp", "diastolic_bp"}.issubset(out.columns):
            out["pulse_pressure"] = out["systolic_bp"] - out["diastolic_bp"]
            out["mean_arterial_pressure"] = (2.0 * out["diastolic_bp"] + out["systolic_bp"]) / 3.0

            sys_ = out["systolic_bp"]
            dia_ = out["diastolic_bp"]
            bp_stage = np.select(
                [
                    (sys_ < 120) & (dia_ < 80),
                    (sys_.between(120, 129, inclusive="both")) & (dia_ < 80),
                    (sys_.between(130, 139, inclusive="both")) | (dia_.between(80, 89, inclusive="both")),
                    (sys_ >= 140) | (dia_ >= 90),
                ],
                ["normal", "elevated", "stage1", "stage2"],
                default="unknown",
            )
            out["bp_stage"] = pd.Series(bp_stage, index=out.index).astype("category")

        if {"cholesterol_total", "hdl_cholesterol"}.issubset(out.columns):
            out["chol_hdl_ratio"] = self._safe_div(out["cholesterol_total"], out["hdl_cholesterol"])
            out["non_hdl_chol"] = out["cholesterol_total"] - out["hdl_cholesterol"]

        if {"ldl_cholesterol", "hdl_cholesterol"}.issubset(out.columns):
            out["ldl_hdl_ratio"] = self._safe_div(out["ldl_cholesterol"], out["hdl_cholesterol"])

        if {"triglycerides", "hdl_cholesterol"}.issubset(out.columns):
            out["tg_hdl_ratio"] = self._safe_div(out["triglycerides"], out["hdl_cholesterol"])
            out["aip_log_tg_hdl"] = np.log(self._safe_div(out["triglycerides"], out["hdl_cholesterol"]))

        for col in ["triglycerides", "cholesterol_total", "ldl_cholesterol", "heart_rate", "screen_time_hours_per_day"]:
            if col in out.columns:
                out[f"log1p_{col}"] = np.log1p(out[col].clip(lower=0))

        if "bmi" in out.columns:
            bmi_class = pd.cut(
                out["bmi"],
                bins=[-np.inf, 18.5, 25.0, 30.0, np.inf],
                labels=["under", "normal", "over", "obese"],
            )
            out["bmi_class"] = bmi_class.astype("category")

        if "age" in out.columns:
            out["age_decade"] = (out["age"] // 10).astype("int64").astype("category")

        if {"physical_activity_minutes_per_week", "bmi"}.issubset(out.columns):
            out["activity_per_bmi"] = self._safe_div(out["physical_activity_minutes_per_week"], out["bmi"].clip(lower=10))

        if {"screen_time_hours_per_day", "sleep_hours_per_day"}.issubset(out.columns):
            out["screen_sleep_ratio"] = self._safe_div(out["screen_time_hours_per_day"], out["sleep_hours_per_day"].clip(lower=1))

        if {"waist_to_hip_ratio", "gender"}.issubset(out.columns):
            whr = out["waist_to_hip_ratio"]
            g = out["gender"].astype(str).str.lower()
            whr_risk = np.where(
                ((g.str.contains("male")) & (whr > 0.90)) | ((g.str.contains("female")) & (whr > 0.85)),
                "high",
                "ok",
            )
            out["whr_risk"] = pd.Series(whr_risk, index=out.index).astype("category")

        flags = []
        if "bmi" in out.columns:
            flags.append((out["bmi"] >= 30).astype(int))
        if {"systolic_bp", "diastolic_bp"}.issubset(out.columns):
            flags.append(((out["systolic_bp"] >= 130) | (out["diastolic_bp"] >= 85)).astype(int))
        if "triglycerides" in out.columns:
            flags.append((out["triglycerides"] >= 150).astype(int))
        if "family_history_diabetes" in out.columns:
            flags.append(out["family_history_diabetes"].fillna(0).astype(int))

        if flags:
            out["risk_flag_sum"] = np.sum(np.vstack(flags), axis=0)

        return out

    def _add_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for a, b in self.interaction_pairs:
            if a in out.columns and b in out.columns:
                out[f"{a}_x_{b}"] = (
                    out[a].astype(str).fillna("NA") + "__" + out[b].astype(str).fillna("NA")
                ).astype("category")
        return out

    def fit(self, X: pd.DataFrame, y=None):
        self._infer_cols(X)
        df = X.copy()

        if "id" in df.columns:
            df = df.drop(columns=["id"])

        df = self._add_domain_features(df)
        df = self._add_interactions(df)

        self._infer_cols(df)

        if self.use_kmeans and self.numeric_cols_:
            num_df = df[self.numeric_cols_].copy()
            num_imputed = pd.DataFrame(
                self.imputer_.fit_transform(num_df),
                columns=self.numeric_cols_,
                index=df.index,
            )
            variances = num_imputed.var(axis=0).sort_values(ascending=False)
            self.kmeans_feature_cols_ = variances.head(min(self.kmeans_max_features, len(variances))).index.tolist()

            Z = num_imputed[self.kmeans_feature_cols_].to_numpy()
            Zs = self.scaler_.fit_transform(Z)

            self.kmeans_models_.clear()
            for k in self.kmeans_clusters:
                km = MiniBatchKMeans(
                    n_clusters=int(k),
                    random_state=self.random_state,
                    batch_size=4096,
                    n_init="auto",
                    reassignment_ratio=0.01,
                )
                km.fit(Zs)
                self.kmeans_models_[int(k)] = km

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "id" in df.columns:
            df = df.drop(columns=["id"])

        df = self._add_domain_features(df)
        df = self._add_interactions(df)

        for c in self.cat_cols_:
            if c in df.columns:
                df[c] = df[c].astype("category")

        if self.use_kmeans and self.kmeans_models_ and self.kmeans_feature_cols_:
            num_df = df[self.numeric_cols_].copy() if self.numeric_cols_ else pd.DataFrame(index=df.index)
            num_imputed = pd.DataFrame(
                self.imputer_.transform(num_df),
                columns=self.numeric_cols_,
                index=df.index,
            )
            Z = num_imputed[self.kmeans_feature_cols_].to_numpy()
            Zs = self.scaler_.transform(Z)

            for k, km in self.kmeans_models_.items():
                d = km.transform(Zs)
                for j in range(d.shape[1]):
                    df[f"km{k}_dist_{j}"] = d[:, j].astype(np.float32)
                df[f"km{k}_label"] = pd.Series(km.predict(Zs), index=df.index).astype("int64").astype("category")

        self.feature_names_ = df.columns.tolist()
        return df


y = train[target_col].astype(int)
X_raw = train.drop(columns=[target_col, id_col], errors="ignore")

X_train_raw, X_hold_raw, y_train, y_hold = train_test_split(
    X_raw, y, test_size=0.10, random_state=42, stratify=y
)

def fe_factory():
    return DiabetesFeatureEngineer(
        use_kmeans=True,
        kmeans_clusters=(6, 12),
        kmeans_max_features=12,
        random_state=42,
    )


splits = 5
n_trial = 15
skf = StratifiedKFold(n_splits=splits, random_state=42, shuffle=True)


def xgb_objective_auc(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",

        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),

        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),

        "max_cat_to_onehot": trial.suggest_int("max_cat_to_onehot", 1, 16),
        "n_estimators": trial.suggest_int("n_estimators", 2000, 30000),

        "enable_categorical": True,
        "tree_method": "hist",
        "device": "cuda:0",
        "random_state": 42,
        "verbosity": 0,
        "early_stopping_rounds":500
    }

    aucs = []
    for tr_idx, va_idx in skf.split(X_train_raw, y_train):
        X_tr_raw = X_train_raw.iloc[tr_idx]
        y_tr     = y_train.iloc[tr_idx]
        X_va_raw = X_train_raw.iloc[va_idx]
        y_va     = y_train.iloc[va_idx]

        fe = fe_factory()
        X_tr = fe.fit_transform(X_tr_raw, y_tr)
        X_va = fe.transform(X_va_raw)

        model = XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )

        p = model.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y_va, p))

        cleanup_gpu(model, fe, X_tr, X_va, X_tr_raw, X_va_raw, y_tr, y_va)

    return float(np.mean(aucs))


print("Starting XGBoost AUC Optimization...")
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(xgb_objective_auc, n_trials=n_trial)
print("Best XGB params:", study_xgb.best_params)


def cat_objective_auc(trial):
    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",          
        "custom_metric": ["AUC"],          
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "iterations": trial.suggest_int("iterations", 500, 8000),

        "task_type": "GPU",
        "devices": "0",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "early_stopping_rounds":300
    }

    aucs = []
    for tr_idx, va_idx in skf.split(X_train_raw, y_train):
        X_tr_raw = X_train_raw.iloc[tr_idx]
        y_tr     = y_train.iloc[tr_idx]
        X_va_raw = X_train_raw.iloc[va_idx]
        y_va     = y_train.iloc[va_idx]

        fe = fe_factory()
        X_tr = fe.fit_transform(X_tr_raw, y_tr)
        X_va = fe.transform(X_va_raw)

        cat_cols = get_cat_cols(X_tr)

        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            cat_features=cat_cols,
            eval_set=(X_va, y_va),
            verbose=False,
            use_best_model=True,
        )

        p = model.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y_va, p))

        cleanup_gpu(model, fe, X_tr, X_va, X_tr_raw, X_va_raw, y_tr, y_va)

    return float(np.mean(aucs))


print("Starting CatBoost AUC Optimization...")
study_cat = optuna.create_study(direction="maximize") 
study_cat.optimize(cat_objective_auc, n_trials=n_trial)
print("Best Cat params:", study_cat.best_params)



#ensemble
fe_final = fe_factory()
X_train_fe = fe_final.fit_transform(X_train_raw, y_train)
X_hold_fe  = fe_final.transform(X_hold_raw)

X_test_raw = test.drop(columns=[id_col], errors="ignore")
X_test_fe  = fe_final.transform(X_test_raw)

# --- XGB final ---
best_xgb_params = dict(study_xgb.best_params)
best_xgb_params.update({
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "enable_categorical": True,
    "tree_method": "hist",
    "device": "cuda:0",
    "random_state": 42,
    "verbosity": 0,
})

xgb_final = XGBClassifier(**best_xgb_params)
xgb_final.fit(
    X_train_fe, y_train,
    eval_set=[(X_hold_fe, y_hold)],
    verbose=False,
    early_stopping_rounds=800,
)

# --- Cat final ---
best_cat_params = dict(study_cat.best_params)
best_cat_params.update({
    "loss_function": "Logloss",
    "eval_metric": "Logloss",    
    "custom_metric": ["AUC"],    
    "task_type": "GPU",
    "devices": "0",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
})

cat_cols_final = get_cat_cols(X_train_fe)

cat_final = CatBoostClassifier(**best_cat_params)
cat_final.fit(
    X_train_fe, y_train,
    cat_features=cat_cols_final,
    eval_set=(X_hold_fe, y_hold),
    verbose=False,
    early_stopping_rounds=800,
    use_best_model=True,
)



# plot evaluation
p_xgb = xgb_final.predict_proba(X_hold_fe)[:, 1]
cleanup_gpu()
p_cat = cat_final.predict_proba(X_hold_fe)[:, 1]
cleanup_gpu()

xgb_auc = roc_auc_score(y_hold, p_xgb)
cat_auc = roc_auc_score(y_hold, p_cat)

w_xgb, w_cat = 0.55, 0.45
p_ens = w_xgb * p_xgb + w_cat * p_cat
ens_auc = roc_auc_score(y_hold, p_ens)

print(f"Holdout AUC XGB: {xgb_auc:.5f}")
print(f"Holdout AUC CAT: {cat_auc:.5f}")
print(f"Holdout AUC ENS: {ens_auc:.5f}")

fpr, tpr, _ = roc_curve(y_hold, p_ens)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, lw=2, label=f"Ensemble AUC={ens_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", lw=2)
plt.title("Holdout ROC")
plt.legend()
plt.show()


p_sub_xgb = xgb_final.predict_proba(X_test_fe)[:, 1]
cleanup_gpu()
del xgb_final
p_sub_cat = cat_final.predict_proba(X_test_fe)[:, 1]
cleanup_gpu()

submission = pd.DataFrame({
    "id": test[id_col],
    "target": w_xgb * p_sub_xgb + w_cat * p_sub_cat
})
submission.to_csv("submission_splitfirst_fefold.csv", index=False)
print("saved: submission_splitfirst_fefold.csv")

