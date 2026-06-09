import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)



train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

train.shape, test.shape



y = train["diagnosed_diabetes"]
train.drop(["diagnosed_diabetes"], axis=1, inplace=True)

test_ids = test["id"]

train = train.drop("id", axis=1)
test = test.drop("id", axis=1)



cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
cat_cols



# Safe ordinal-encoding
for c in cat_cols:
    train[c] = train[c].astype("category").cat.codes
    test[c] = test[c].astype("category").cat.codes



X = train.copy()
X_test = test.copy()

X.shape, X_test.shape



N_FOLDS = 10
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))



def run_cv_model(model_name, model_init_fn, X, y, X_test, folds):
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(folds):
        print(f"\n=== {model_name} Fold {fold+1}/{len(folds)} ===")
        
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = model_init_fn()

        if model_name == "catboost":
            model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                verbose=False,
                use_best_model=True
            )

        elif model_name == "lightgbm":
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
            )

        elif model_name == "xgb":
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=200,
                verbose=False
            )

        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / len(folds)

        fold_auc = roc_auc_score(y_val, oof[val_idx])
        print(f"Fold AUC: {fold_auc:.5f}")

    cv_score = roc_auc_score(y, oof)
    print(f"\n=== Full {model_name} CV AUC: {cv_score:.5f} ===")

    return oof, test_preds, cv_score



def cat_init():
    return CatBoostClassifier(
        iterations=2000,
        depth=8,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED
    )

cat_oof, cat_test, cat_cv = run_cv_model(
    "catboost", cat_init, X, y, X_test, folds
)

cat_cv



def lgb_init():
    return lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1
    )

lgb_oof, lgb_test, lgb_cv = run_cv_model(
    "lightgbm", lgb_init, X, y, X_test, folds
)

lgb_cv



def xgb_init():
    return XGBClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED
    )

xgb_oof, xgb_test, xgb_cv = run_cv_model(
    "xgb", xgb_init, X, y, X_test, folds
)

xgb_cv



def blend_objective(trial):
    w1 = trial.suggest_float("w1", 0, 1)
    w2 = trial.suggest_float("w2", 0, 1)
    w3 = trial.suggest_float("w3", 0, 1)

    w = np.array([w1, w2, w3])
    w = w / w.sum()

    blended = w[0]*cat_oof + w[1]*lgb_oof + w[2]*xgb_oof
    return roc_auc_score(y, blended)

study = optuna.create_study(direction="maximize")
study.optimize(blend_objective, n_trials=200)

study.best_params, study.best_value



best = study.best_params
w = np.array([best["w1"], best["w2"], best["w3"]])
w = w / w.sum()
w



final_test_pred = (
    w[0]*cat_test +
    w[1]*lgb_test +
    w[2]*xgb_test
)



submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": final_test_pred
})

submission.to_csv("clean_blended_submission.csv", index=False)
submission.head()



final_oof_blend = (
    w[0]*cat_oof +
    w[1]*lgb_oof +
    w[2]*xgb_oof
)

print("Final Blended OOF AUC:", roc_auc_score(y, final_oof_blend))



print("OOF mean:", final_oof_blend.mean())
print("Test mean:", final_test_pred.mean())

import numpy as np
print("OOF percentiles:", np.percentile(final_oof_blend, [1,5,25,50,75,95,99]))
print("Test percentiles:", np.percentile(final_test_pred, [1,5,25,50,75,95,99]))



print("Base AUCs:")
print("Cat:", cat_cv)
print("LGB:", lgb_cv)
print("XGB:", xgb_cv)
print("Blend:", roc_auc_score(y, final_oof_blend))





