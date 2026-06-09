VERSION = "019"
TIME_LIMIT_HOURS = 11.5
TIME_LIMIT = TIME_LIMIT_HOURS * 3600

# L1 MODELS (Original dataset models)
L1_MODELS = [
    "cat",
    "lgb",
    "xgb",
    "lr",
    "ridge",
    "rf",
    "et",
    "knn",
    "blend"   # special case → blend of all enabled L1 models
]

# L2 MODELS (Residual models)
L2_MODELS = [
    "xgb3",
    "xgb7",
    "xgb12",
    "lgb3",
    "lgb7",
    "lgb12",
    "cat",
    # "rf_small",
    # "rf_big",
    # "et"
]

# L3 STACKER MODELS
L3_MODELS = [
    "lgb3",
    "lgb7",
    "lgb12",
    "xgb3",
    "xgb7",
    "xgb12",
    "cat",
    "logreg",
    "ridge",
    "mlp"
]

# L3 ENSEMBLES
L3_ENSEMBLES = [
    "equal",
    "rank",
    "optimal"
]


# Helper — timer
import time
START_TIME = time.time()

def time_up():
    return (time.time() - START_TIME) >= TIME_LIMIT

print("Running Version:", VERSION)
print("Time Limit (hours):", TIME_LIMIT_HOURS)
print("L1 Models:", L1_MODELS)
print("L2 Models:", L2_MODELS)
print("L3 Models:", L3_MODELS)
print("L3 Ensembles:", L3_ENSEMBLES)


import pandas as pd
import numpy as np

# Competition data
COMP_TRAIN_PATH = "/kaggle/input/playground-series-s5e11/train.csv"
COMP_TEST_PATH  = "/kaggle/input/playground-series-s5e11/test.csv"

# Original dataset (any structure allowed)
ORIG_PATH       = "/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv"

train = pd.read_csv(COMP_TRAIN_PATH)
test  = pd.read_csv(COMP_TEST_PATH)
orig  = pd.read_csv(ORIG_PATH)

TARGET = "loan_paid_back"

# Only use train's predictors
features = [c for c in train.columns if c not in ["id", TARGET]]

# Drop extra columns in original (strict drop)
orig = orig[[c for c in orig.columns if c in features + [TARGET]]]

# Drop id from train and test
train.drop('id', inplace = True, axis = 1)
test.drop('id', inplace = True, axis = 1)

print("Train:", train.shape, "| Test:", test.shape, "| Original:", orig.shape)


from sklearn.preprocessing import LabelEncoder

cat_cols = [c for c in features if train[c].dtype == "object"]

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col], orig[col]], axis=0).astype(str)
    le.fit(combined)

    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))
    orig[col]  = le.transform(orig[col].astype(str))

    encoders[col] = le

print("Categorical columns:", cat_cols)


from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

if time_up():
    raise SystemExit("Time limit reached before L1.")

# Storage
L1_train_preds = {}
L1_test_preds  = {}
L1_results     = []

def save_L1_submission(name, preds):
    test_sub = pd.read_csv(COMP_TEST_PATH)
    out = pd.DataFrame({"id": test_sub["id"], "loan_paid_back": preds})
    fname = f"submission_L1_{name}_v{VERSION}.csv"
    out.to_csv(fname, index=False)
    print("Saved:", fname)


print("\n===================================")
print("        LEVEL 1 MODELS START")
print("===================================\n")

# 1) CATBOOST
if "cat" in L1_MODELS:
    print("\n[ L1 - CatBoost ]")
    cb = CatBoostClassifier(
        iterations=2000,
        depth=6,
        learning_rate=0.03,
        eval_metric="AUC",
        loss_function="Logloss",
        task_type="GPU",
        random_seed=42,
        verbose=False,
    )
    cb.fit(orig[features], orig[TARGET])

    train["pL1_cat"] = cb.predict_proba(train[features])[:, 1]
    test["pL1_cat"]  = cb.predict_proba(test[features])[:, 1]

    auc = roc_auc_score(train[TARGET], train["pL1_cat"])
    print("CatBoost L1 AUC:", auc)

    L1_results.append(("catboost", auc))
    save_L1_submission("cat", test["pL1_cat"])

# 2) LIGHTGBM
if "lgb" in L1_MODELS:
    print("\n[ L1 - LightGBM ]")
    lgbm = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.8,
        metric="auc",
        random_state=42
    )
    lgbm.fit(orig[features], orig[TARGET])

    train["pL1_lgb"] = lgbm.predict_proba(train[features])[:, 1]
    test["pL1_lgb"]  = lgbm.predict_proba(test[features])[:, 1]

    auc = roc_auc_score(train[TARGET], train["pL1_lgb"])
    print("LightGBM L1 AUC:", auc)

    L1_results.append(("lightgbm", auc))
    save_L1_submission("lgb", test["pL1_lgb"])

# 3) XGBOOST
if "xgb" in L1_MODELS:
    print("\n[ L1 - XGBoost ]")
    xgb_L1 = xgb.XGBClassifier(
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="auc",
        learning_rate=0.03,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.8,
        n_estimators=2500,
        random_state=42
    )
    xgb_L1.fit(orig[features], orig[TARGET])

    train["pL1_xgb"] = xgb_L1.predict_proba(train[features])[:, 1]
    test["pL1_xgb"]  = xgb_L1.predict_proba(test[features])[:, 1]

    auc = roc_auc_score(train[TARGET], train["pL1_xgb"])
    print("XGBoost L1 AUC:", auc)

    L1_results.append(("xgboost", auc))
    save_L1_submission("xgb", test["pL1_xgb"])

# 4) LOGISTIC REGRESSION
if "lr" in L1_MODELS:
    print("\n[ L1 - Logistic Regression ]")
    logr = LogisticRegression(max_iter=500)
    logr.fit(orig[features], orig[TARGET])

    train["pL1_lr"] = logr.predict_proba(train[features])[:, 1]
    test["pL1_lr"]  = logr.predict_proba(test[features])[:, 1]

    auc = roc_auc_score(train[TARGET], train["pL1_lr"])
    print("LR L1 AUC:", auc)

    L1_results.append(("logreg", auc))
    save_L1_submission("lr", test["pL1_lr"])

# 5) RIDGE CLASSIFIER
if "ridge" in L1_MODELS:
    print("\n[ L1 - Ridge Classifier ]")

    ridge = RidgeClassifier()
    ridge.fit(orig[features], orig[TARGET])

    def ridge_to_prob(model, X):
        z = model.decision_function(X)
        return 1/(1 + np.exp(-z))

    train["pL1_ridge"] = ridge_to_prob(ridge, train[features])
    test["pL1_ridge"]  = ridge_to_prob(ridge, test[features])

    auc = roc_auc_score(train[TARGET], train["pL1_ridge"])
    print("Ridge L1 AUC:", auc)

    L1_results.append(("ridge", auc))
    save_L1_submission("ridge", test["pL1_ridge"])

# 6) RANDOM FOREST
if "rf" in L1_MODELS:
    print("\n[ L1 - RandomForest ]")

    rf_L1 = RandomForestClassifier(
        n_estimators=600,
        max_depth=18,
        n_jobs=-1,
        random_state=42
    )
    rf_L1.fit(orig[features], orig[TARGET])

    train["pL1_rf"] = rf_L1.predict_proba(train[features])[:, 1]
    test["pL1_rf"]  = rf_L1.predict_proba(test[features])[:, 1]

    auc = roc_auc_score(train[TARGET], train["pL1_rf"])
    print("RF L1 AUC:", auc)

    L1_results.append(("rf", auc))
    save_L1_submission("rf", test["pL1_rf"])

# 7) EXTRA TREES
if "et" in L1_MODELS:
    print("\n[ L1 - ExtraTrees ]")

    et_L1 = ExtraTreesClassifier(
        n_estimators=600,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
    et_L1.fit(orig[features], orig[TARGET])

    train["pL1_et"] = et_L1.predict_proba(train[features])[:, 1]
    test["pL1_et"]  = et_L1.predict_proba(test[features])[:, 1]

    auc = roc_auc_score(train[TARGET], train["pL1_et"])
    print("ET L1 AUC:", auc)

    L1_results.append(("extratrees", auc))
    save_L1_submission("et", test["pL1_et"])

# 8) KNN
if "knn" in L1_MODELS:
    print("\n[ L1 - KNN ]")

    knn = KNeighborsClassifier(n_neighbors=25)
    knn.fit(orig[features], orig[TARGET])

    train["pL1_knn"] = knn.predict_proba(train[features])[:, 1]
    test["pL1_knn"]  = knn.predict_proba(test[features])[:, 1]

    auc = roc_auc_score(train[TARGET], train["pL1_knn"])
    print("KNN L1 AUC:", auc)

    L1_results.append(("knn", auc))
    save_L1_submission("knn", test["pL1_knn"])


# 05 — LEVEL 1 SUMMARY
print("\n==========================")
print("   LEVEL 1 SUMMARY TABLE  ")
print("==========================")

# Collect L1 results into a DataFrame
df_L1 = pd.DataFrame(L1_results, columns=["Model", "AUC"]).sort_values("AUC", ascending=False)
display(df_L1)


# Identify available L1 prediction columns
all_L1_cols = {
    "cat":   "pL1_cat",
    "lgb":   "pL1_lgb",
    "xgb":   "pL1_xgb",
    "lr":    "pL1_lr",
    "ridge": "pL1_ridge",
    "rf":    "pL1_rf",
    "et":    "pL1_et",
    "knn":   "pL1_knn"
}

# Only include columns for active models
L1_cols = [all_L1_cols[m] for m in L1_MODELS if m in all_L1_cols]

print("\nUsing these active L1 model columns for blending:")
print(L1_cols)

# Assign blending weights dynamically
default_weights = {
    "pL1_cat":   1.8,
    "pL1_lgb":   2.0,
    "pL1_xgb":   2.2,
    "pL1_lr":    0.8,
    "pL1_ridge": 0.8,
    "pL1_rf":    1.0,
    "pL1_et":    1.0,
    "pL1_knn":   0.6
}

# Filter weights for active models only
weights = {col: default_weights[col] for col in L1_cols}

total_w = sum(weights.values())

# Weighted blend for L1 prior probability
train["pL1_blend"] = sum(train[col] * weights[col] for col in L1_cols) / total_w
test ["pL1_blend"] = sum(test[col]  * weights[col] for col in L1_cols) / total_w

auc_blend = roc_auc_score(train[TARGET], train["pL1_blend"])
print(f"\nBlended L1 AUC (pL1_blend) = {auc_blend:.5f}")

# Save blended L1 submission
sub_blend = pd.DataFrame({
    "id": pd.read_csv(COMP_TEST_PATH)["id"],
    "loan_paid_back": test["pL1_blend"]
})

fname = f"submission_L1_BLEND_v{VERSION}.csv"
sub_blend.to_csv(fname, index=False)
print("Saved:", fname)


def to_logit(p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# true logits
train["true_logit"] = to_logit(train[TARGET])

# blended L1 logits
train["logit_L1_blend"] = to_logit(train["pL1_blend"])
test ["logit_L1_blend"] = to_logit(test ["pL1_blend"])

# residual target for L2
train["L2_target"] = train["true_logit"] - train["logit_L1_blend"]

print("\nResiduals ready for Level-2 models!")
print(train[["true_logit", "logit_L1_blend", "L2_target"]].head())


import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

features_L2 = features + ["pL1_blend", "logit_L1_blend"]

# Storage for leaderboard
L2_results = []
L2_pred_cols = []

# Store Level-2 predictions for stacking input
for col in ["pL2_xgb3","pL2_xgb7","pL2_xgb12",
            "pL2_lgb3","pL2_lgb7","pL2_lgb12",
            "pL2_cat","pL2_rf_small","pL2_rf_big","pL2_et"]:
    train[col] = 0.0
    test[col]  = 0.0

# Helper for model evaluation
def eval_and_save_L2(name, oof_pred, test_pred):
    train[name] = oof_pred
    test[name]  = test_pred

    final_prob = sigmoid(train["logit_L1_blend"] + oof_pred)
    auc = roc_auc_score(train[TARGET], final_prob)

    L2_results.append([name, auc])
    L2_pred_cols.append(name)

    print(f"{name}: AUC = {auc:.5f}")

    # Save submission
    sub_file = f"submission_L2_{name}_v{VERSION}.csv"
    pd.DataFrame({
        "id": pd.read_csv(COMP_TEST_PATH)["id"],
        "loan_paid_back": sigmoid(test["logit_L1_blend"] + test_pred)
    }).to_csv(sub_file, index=False)

    print("Saved:", sub_file)
    print("-------------------------------------------")



XGB_DEPTHS = {
    "xgb3": 3,
    "xgb7": 7,
    "xgb12": 12
}

LGB_DEPTHS = {
    "lgb3": 3,
    "lgb7": 7,
    "lgb12": 12
}

# RUN XGB MODELS (if selected)
if any(m in L2_MODELS for m in XGB_DEPTHS.keys()):

    xgb_base_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "seed": 42,
    }

    for name, depth in XGB_DEPTHS.items():
        if name not in L2_MODELS:
            continue

        print(f"\n========== Training {name} ==========")

        oof = np.zeros(len(train))
        test_pred = np.zeros(len(test))

        for tr, va in skf.split(train[features], train[TARGET]):
            params = xgb_base_params.copy()
            params["max_depth"] = depth

            dtr = xgb.DMatrix(train.iloc[tr][features_L2], label=train.iloc[tr]["L2_target"])
            dva = xgb.DMatrix(train.iloc[va][features_L2], label=train.iloc[va]["L2_target"])
            dte = xgb.DMatrix(test[features_L2])

            model = xgb.train(
                params, dtr,
                num_boost_round=2000,
                evals=[(dva, "valid")],
                early_stopping_rounds=150,
                verbose_eval=False
            )

            oof[va] = model.predict(dva)
            test_pred += model.predict(dte) / FOLDS

        eval_and_save_L2(f"pL2_{name}", oof, test_pred)

# RUN LGBM MODELS (if selected)
if any(m in L2_MODELS for m in LGB_DEPTHS.keys()):

    lgb_base_params = {
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "metric": "rmse",
        "random_state": 42
    }

    for name, depth in LGB_DEPTHS.items():
        if name not in L2_MODELS:
            continue

        print(f"\n========== Training {name} ==========")

        oof = np.zeros(len(train))
        test_pred = np.zeros(len(test))

        for tr, va in skf.split(train[features], train[TARGET]):
            params = lgb_base_params.copy()
            params["num_leaves"] = 2 ** depth

            model = lgb.LGBMRegressor(**params, n_estimators=3000)
            model.fit(
                train.iloc[tr][features_L2], train.iloc[tr]["L2_target"],
                eval_set=[(train.iloc[va][features_L2], train.iloc[va]["L2_target"])],
                callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
            )

            oof[va] = model.predict(train.iloc[va][features_L2])
            test_pred += model.predict(test[features_L2]) / FOLDS

        eval_and_save_L2(f"pL2_{name}", oof, test_pred)


if "cat" in L2_MODELS:
    print("\n========== CatBoostRegressor ==========")

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    cat_model = CatBoostRegressor(
        iterations=2000,
        depth=8,
        learning_rate=0.03,
        loss_function="RMSE",
        task_type="GPU",
        verbose=False
    )

    for tr, va in skf.split(train[features], train[TARGET]):
        cat_model.fit(train.iloc[tr][features_L2], train.iloc[tr]["L2_target"])
        oof[va] = cat_model.predict(train.iloc[va][features_L2])
        test_pred += cat_model.predict(test[features_L2]) / FOLDS

    eval_and_save_L2("pL2_cat", oof, test_pred)

# RANDOM FOREST SMALL / BIG
if "rf_small" in L2_MODELS:
    print("\n========== RandomForest SMALL ==========")

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    model = RandomForestRegressor(n_estimators=500, max_depth=14, n_jobs=-1, random_state=42)

    for tr, va in skf.split(train[features], train[TARGET]):
        model.fit(train.iloc[tr][features_L2], train.iloc[tr]["L2_target"])
        oof[va] = model.predict(train.iloc[va][features_L2])
        test_pred += model.predict(test[features_L2]) / FOLDS

    eval_and_save_L2("pL2_rf_small", oof, test_pred)

if "rf_big" in L2_MODELS:
    print("\n========== RandomForest BIG ==========")

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    model = RandomForestRegressor(n_estimators=2000, max_depth=20, n_jobs=-1, random_state=42)

    for tr, va in skf.split(train[features], train[TARGET]):
        model.fit(train.iloc[tr][features_L2], train.iloc[tr]["L2_target"])
        oof[va] = model.predict(train.iloc[va][features_L2])
        test_pred += model.predict(test[features_L2]) / FOLDS

    eval_and_save_L2("pL2_rf_big", oof, test_pred)

# EXTRA TREES REGRESSOR
if "et" in L2_MODELS:
    print("\n========== ExtraTreesRegressor ==========")

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    model = ExtraTreesRegressor(n_estimators=1200, random_state=42, n_jobs=-1)

    for tr, va in skf.split(train[features], train[TARGET]):
        model.fit(train.iloc[tr][features_L2], train.iloc[tr]["L2_target"])
        oof[va] = model.predict(train.iloc[va][features_L2])
        test_pred += model.predict(test[features_L2]) / FOLDS

    eval_and_save_L2("pL2_et", oof, test_pred)

print("\n===================================")
print("  LEVEL 2 — MODEL PERFORMANCE")
print("===================================\n")

df_L2 = pd.DataFrame(L2_results, columns=["Model", "AUC"]).sort_values("AUC", ascending=False)
display(df_L2)


# LEVEL 3 STACKERS — FULLY MODULAR
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

print("\n===================================")
print("           LEVEL 3 STACKERS")
print("===================================\n")

# DEFINE LEVEL-3 FEATURES
features_L3 = (
    features +
    ["pL1_blend", "logit_L1_blend"] +
    L2_pred_cols  
)

X_L3 = train[features_L3]
T_L3 = train[TARGET]
Xte_L3 = test[features_L3]

L3_results = []
L3_preds = {}

# HELPER — TRAIN, EVALUATE, SAVE SUBMISSIONS
def fit_L3(name, model):
    print(f"\n========== Training {name} ==========")

    model.fit(X_L3, T_L3)
    train_pred = model.predict_proba(X_L3)[:, 1]
    test_pred  = model.predict_proba(Xte_L3)[:, 1]

    auc = roc_auc_score(T_L3, train_pred)
    L3_results.append([name, auc])
    L3_preds[name] = test_pred

    print(f"✔ {name} AUC = {auc:.5f}")

    # Save submission
    fname = f"submission_L3_{name}_v{VERSION}.csv"
    pd.DataFrame({
        "id": pd.read_csv(COMP_TEST_PATH)["id"],
        "loan_paid_back": test_pred
    }).to_csv(fname, index=False)
    print("Saved:", fname)

    return auc

# LGB STACKERS
if "lgb3" in L3_MODELS:
    fit_L3("lgb3", lgb.LGBMClassifier(num_leaves=8, n_estimators=2000, learning_rate=0.02))

if "lgb7" in L3_MODELS:
    fit_L3("lgb7", lgb.LGBMClassifier(num_leaves=128, n_estimators=2000, learning_rate=0.02))

if "lgb12" in L3_MODELS:
    fit_L3("lgb12", lgb.LGBMClassifier(num_leaves=2048, n_estimators=2000, learning_rate=0.02))

# XGB STACKERS
if "xgb3" in L3_MODELS:
    fit_L3("xgb3", xgb.XGBClassifier(max_depth=3, n_estimators=1500, learning_rate=0.03, tree_method="hist"))

if "xgb7" in L3_MODELS:
    fit_L3("xgb7", xgb.XGBClassifier(max_depth=7, n_estimators=1500, learning_rate=0.03, tree_method="hist"))

if "xgb12" in L3_MODELS:
    fit_L3("xgb12", xgb.XGBClassifier(max_depth=12, n_estimators=1500, learning_rate=0.03, tree_method="hist"))

# CATBOOST STACKER
if "cat" in L3_MODELS:
    fit_L3(
        "cat",
        CatBoostClassifier(
            iterations=2000,
            depth=8,
            learning_rate=0.03,
            loss_function="Logloss",
            task_type="GPU",
            verbose=False
        )
    )

# LOGISTIC REGRESSION STACKER
if "logreg" in L3_MODELS:
    fit_L3("logreg", LogisticRegression(max_iter=5000))

# RIDGE STACKER
if "ridge" in L3_MODELS:
    print("\n========== Training ridge ==========")

    model = RidgeClassifier()
    model.fit(X_L3, T_L3)

    # Convert decision_function → probability
    train_scores = model.decision_function(X_L3)
    train_pred   = sigmoid(train_scores)

    test_scores  = model.decision_function(Xte_L3)
    test_pred    = sigmoid(test_scores)

    auc = roc_auc_score(T_L3, train_pred)
    L3_results.append(["ridge", auc])
    L3_preds["ridge"] = test_pred

    print(f"ridge AUC = {auc:.5f}")

    # Save submission
    fname = f"submission_L3_ridge_v{VERSION}.csv"
    pd.DataFrame({
        "id": pd.read_csv(COMP_TEST_PATH)["id"],
        "loan_paid_back": test_pred
    }).to_csv(fname, index=False)
    print("Saved:", fname)

# MLP STACKER
if "mlp" in L3_MODELS:
    fit_L3(
        "mlp",
        MLPClassifier(
            hidden_layer_sizes=(256,128,64),
            activation="relu",
            solver="adam",
            max_iter=30,
            random_state=42
        )
    )


# ENSEMBLES
print("\n========== ENSEMBLE MODELS ==========\n")

# ENSEMBLES (CONTROLLED BY L3_ENSEMBLES)
def save_ensemble(name, pred):
    sub_file = f"submission_{name}_v{VERSION}.csv"
    pd.DataFrame({
        "id": pd.read_csv(COMP_TEST_PATH)["id"],
        "loan_paid_back": pred
    }).to_csv(sub_file, index=False)
    print("Saved ensemble:", sub_file)

# Equal-weight
if "equal" in L3_ENSEMBLES:
    ens = np.mean(np.column_stack(list(L3_preds.values())), axis=1)
    save_ensemble("L3_equal", ens)

# Rank-avg
if "rank" in L3_ENSEMBLES:
    ens = np.mean(np.column_stack([
        pd.Series(p).rank(pct=True).values
        for p in L3_preds.values()
    ]), axis=1)
    save_ensemble("L3_rank", ens)

# Optimal weighted blend (based on CV)
if "optimal" in L3_ENSEMBLES:
    w = np.array([auc for _, auc in L3_results])
    w = w / w.sum()
    ens = np.sum(np.column_stack(list(L3_preds.values())) * w, axis=1)
    save_ensemble("L3_optimal", ens)


# Show Leaderboard
df_L3 = pd.DataFrame(L3_results, columns=["Model", "AUC"]).sort_values("AUC", ascending=False)

print("\n===================================")
print("       LEVEL 3 — LEADERBOARD")
print("===================================\n")
display(df_L3)


# # MASTER ENSEMBLER
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score

# print("\n==== MASTER ENSEMBLER START ====\n")

# test_sub = pd.read_csv(COMP_TEST_PATH)
# base_sub = pd.DataFrame({"id": test_sub["id"].values})

# # COLLECT TRAIN/TEST PREDICTIONS
# # L1 and L2 predictions from train dataframe
# L1_cols = [c for c in train.columns if c.startswith("pL1_")]
# L2_cols = [c for c in train.columns if c.startswith("pL2_")]

# # L3 predictions
# L3_cols      = list(L3_preds.keys())         # TEST predictions
# L3_cols_tr   = list(L3_train_preds.keys())   # TRAIN predictions (new)

# # BUILD TRAIN MATRIX
# train_preds_df = pd.DataFrame()

# # L1/L2
# for col in (L1_cols + L2_cols):
#     train_preds_df[col] = train[col]

# # L3 TRAIN predictions (added)
# for col in L3_cols_tr:
#     train_preds_df[col] = L3_train_preds[col]

# # BUILD TEST MATRIX
# test_preds_df = pd.DataFrame()

# for col in (L1_cols + L2_cols):
#     test_preds_df[col] = test[col]

# # L3 TEST predictions
# for col in L3_cols:
#     test_preds_df[col] = L3_preds[col]

# y_true = train[TARGET].values

# print(f"Collected {train_preds_df.shape[1]} models for ensembling.")

# # EQUAL-WEIGHT
# pred_equal = test_preds_df.mean(axis=1).values
# cv_equal = roc_auc_score(y_true, train_preds_df.mean(axis=1).values)

# sub_equal = base_sub.copy()
# sub_equal["loan_paid_back"] = pred_equal
# sub_equal.to_csv(f"submission_{VERSION}_equal.csv", index=False)
# print(f"Equal-weight ensemble AUC = {cv_equal:.5f}")

# # SOFTMAX WEIGHTS
# cv_scores = np.array([roc_auc_score(y_true, train_preds_df[c]) for c in train_preds_df.columns])
# softmax = np.exp(cv_scores) / np.sum(np.exp(cv_scores))

# pred_softmax = np.sum(test_preds_df.values * softmax[:, None], axis=1)
# cv_softmax = roc_auc_score(y_true, np.sum(train_preds_df.values * softmax[:, None], axis=1))

# sub_softmax = base_sub.copy()
# sub_softmax["loan_paid_back"] = pred_softmax
# sub_softmax.to_csv(f"submission_{VERSION}_softmax.csv", index=False)
# print(f"Softmax-weight ensemble AUC = {cv_softmax:.5f}")

# # CORRELATION-DIVERSIFIED BLEND
# corr_mat = train_preds_df.corr().values
# div_scores = 1 / (1 + corr_mat.mean(axis=1))
# div_w = div_scores / div_scores.sum()

# pred_div = np.sum(test_preds_df.values * div_w[:, None], axis=1)
# cv_div = roc_auc_score(y_true, np.sum(train_preds_df.values * div_w[:, None], axis=1))

# sub_div = base_sub.copy()
# sub_div["loan_paid_back"] = pred_div
# sub_div.to_csv(f"submission_{VERSION}_diverse.csv", index=False)
# print(f"Diversity ensemble AUC = {cv_div:.5f}")

# # RANK AVERAGE
# pred_rank = test_preds_df.rank(pct=True).mean(axis=1).values
# cv_rank = roc_auc_score(y_true, train_preds_df.rank(pct=True).mean(axis=1).values)

# sub_rank = base_sub.copy()
# sub_rank["loan_paid_back"] = pred_rank
# sub_rank.to_csv(f"submission_{VERSION}_rankavg.csv", index=False)
# print(f"Rank-average AUC = {cv_rank:.5f}")

# # TOP-3 MODELS
# top3_cols = train_preds_df.columns[np.argsort(cv_scores)[-3:]]
# pred_top3 = (
#     0.3 * test_preds_df[top3_cols[0]] +
#     0.3 * test_preds_df[top3_cols[1]] +
#     0.4 * test_preds_df[top3_cols[2]]
# )

# cv_top3 = roc_auc_score(
#     y_true,
#     0.3 * train_preds_df[top3_cols[0]] +
#     0.3 * train_preds_df[top3_cols[1]] +
#     0.4 * train_preds_df[top3_cols[2]]
# )

# sub_top3 = base_sub.copy()
# sub_top3["loan_paid_back"] = pred_top3
# sub_top3.to_csv(f"submission_{VERSION}_top3.csv", index=False)
# print(f"Top-3 AUC = {cv_top3:.5f}")

# # OPTIMIZED 3-MODEL BLEND
# best_auc = 0
# best_w = None
# best_cols = None

# cols = train_preds_df.columns

# for i in range(len(cols)):
#     for j in range(i+1, len(cols)):
#         for k in range(j+1, len(cols)):
#             c1,c2,c3 = cols[i], cols[j], cols[k]

#             for w1 in [0.2,0.3,0.4]:
#                 for w2 in [0.2,0.3,0.4]:
#                     w3 = 1 - w1 - w2
#                     if w3 <= 0: continue

#                     pred_tr = w1*train_preds_df[c1] + w2*train_preds_df[c2] + w3*train_preds_df[c3]
#                     auc = roc_auc_score(y_true, pred_tr)

#                     if auc > best_auc:
#                         best_auc = auc
#                         best_w = (w1, w2, w3)
#                         best_cols = (c1,c2,c3)

# print("\nOptimized 3-model blend:")
# print("Models:", best_cols)
# print("Weights:", best_w)
# print("AUC:", best_auc)

# pred_opt3 = (
#     best_w[0] * test_preds_df[best_cols[0]] +
#     best_w[1] * test_preds_df[best_cols[1]] +
#     best_w[2] * test_preds_df[best_cols[2]]
# )

# sub_opt3 = base_sub.copy()
# sub_opt3["loan_paid_back"] = pred_opt3
# sub_opt3.to_csv(f"submission_{VERSION}_opt3.csv", index=False)


# cv_df = pd.DataFrame({
#     # "Model": (
#     #     all_pred_cols +
#     #     ["Equal", "Softmax", "Diverse", "RankAvg", "Top3", "Opt3"]
#     # ),
#     "Model": (
#         ["Equal", "Softmax", "Diverse", "RankAvg", "Top3", "Opt3"]
#     ),
#     # "CV_AUC": (
#     #     list(cv_scores) +
#     #     [cv_equal, cv_softmax, cv_div, cv_rank, cv_top3, best_auc]
#     # )
#     "CV_AUC": (
#         [cv_equal, cv_softmax, cv_div, cv_rank, cv_top3, best_auc]
#     )
# }).sort_values("CV_AUC", ascending=False)

# display(cv_df)

# # Save table
# cv_df.to_csv(f"comparison_CV_{VERSION}.csv", index=False)


# # PLOT CV RESULTS
# plt.figure(figsize=(11, 7))
# plt.barh(cv_df["Model"], cv_df["CV_AUC"], color="royalblue")
# plt.gca().invert_yaxis()
# plt.title(f"Version {VERSION} — Comparison of ALL Model AUC Scores", fontsize=15)
# plt.xlabel("AUC")
# plt.grid(axis="x", linestyle="--", alpha=0.4)
# plt.show()




