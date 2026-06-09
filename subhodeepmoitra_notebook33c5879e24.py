import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


TARGET = "accident_risk"
ID = "id"


X = train.drop(columns=[TARGET])
y = train[TARGET]
test_data = test.copy()


print(train.columns.tolist())


bool_cols = ["road_signs_present", "public_road", "holiday", "school_season"]
for col in bool_cols:
    X[col] = X[col].astype(int)
    test_data[col] = test_data[col].astype(int)


cat_cols = ["road_type", "lighting", "weather", "time_of_day"]
num_cols = [c for c in X.columns if c not in cat_cols + bool_cols + [ID]]


from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    test_data[col] = le.transform(test_data[col].astype(str))


lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.02,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "verbose": -1,
    "n_estimators": 3000
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.02,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 2.0,
    "alpha": 0.5,
    "n_estimators": 3000
}

cat_params = {
    "loss_function": "RMSE",
    "learning_rate": 0.03,
    "depth": 7,
    "l2_leaf_reg": 3,
    "iterations": 3000,
    "random_strength": 1.5,
    "early_stopping_rounds": 150,
    "verbose": 0
}



'''kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds_lgb = np.zeros(len(test_data))
test_preds_xgb = np.zeros(len(test_data))
test_preds_cat = np.zeros(len(test_data))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"===== Fold {fold + 1} =====")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # lightgmb
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100)]
    )
    preds_lgb = lgb_model.predict(X_valid)
    oof_preds[valid_idx] += preds_lgb / 3
    test_preds_lgb += lgb_model.predict(test_data) / kf.n_splits / 3

    # xgboost
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
        early_stopping_rounds=100
    )
    preds_xgb = xgb_model.predict(X_valid)
    oof_preds[valid_idx] += preds_xgb / 3
    test_preds_xgb += xgb_model.predict(test_data) / kf.n_splits / 3

    # CatBoost
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        cat_features=[X.columns.get_loc(c) for c in cat_cols],
        use_best_model=True
    )
    preds_cat = cat_model.predict(X_valid)
    oof_preds[valid_idx] += preds_cat / 3
    test_preds_cat += cat_model.predict(test_data) / kf.n_splits / 3'''


'''# using GPU
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds_lgb = np.zeros(len(test_data))
test_preds_xgb = np.zeros(len(test_data))
test_preds_cat = np.zeros(len(test_data))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"===== Fold {fold + 1} =====")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # ----------------- LightGBM -----------------
    lgb_model = lgb.LGBMRegressor(
        **lgb_params,
        device='gpu'  # Enable GPU
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100)]
    )
    preds_lgb = lgb_model.predict(X_valid)
    oof_preds[valid_idx] += preds_lgb / 3
    test_preds_lgb += lgb_model.predict(test_data) / kf.n_splits / 3

    # ----------------- XGBoost -----------------
    xgb_model = xgb.XGBRegressor(
       **xgb_params,
       tree_method="hist",
       device="cuda"
     )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
        early_stopping_rounds=100
    )
    preds_xgb = xgb_model.predict(X_valid)
    oof_preds[valid_idx] += preds_xgb / 3
    test_preds_xgb += xgb_model.predict(test_data) / kf.n_splits / 3

    # ----------------- CatBoost -----------------
    cat_model = CatBoostRegressor(
        **cat_params,
        task_type='GPU',     # using the GPU
        devices='0:1'        # ID of the gpu device
    )
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        cat_features=[X.columns.get_loc(c) for c in cat_cols],
        use_best_model=True
    )
    preds_cat = cat_model.predict(X_valid)
    oof_preds[valid_idx] += preds_cat / 3
    test_preds_cat += cat_model.predict(test_data) / kf.n_splits / 3
'''


# by using stacking
# List of seeds for averaging
seeds = [42, 1337, 2025]
#seeds = [42]
n_splits = 10

# Initialize final arrays
oof_preds_avg = np.zeros(len(X))
test_preds_avg = np.zeros(len(test_data))

for seed in seeds:
    print(f"\n===== Seed {seed} =====")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # OOF and test predictions for this seed
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))

    test_preds_lgb = np.zeros(len(test_data))
    test_preds_xgb = np.zeros(len(test_data))
    test_preds_cat = np.zeros(len(test_data))

    # === Base Models with GPU ===
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"\n===== Fold {fold + 1} =====")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        # ----- LightGBM -----
        lgb_model = lgb.LGBMRegressor(**lgb_params, device='gpu')
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100)]
        )
        oof_lgb[valid_idx] = lgb_model.predict(X_valid)
        test_preds_lgb += lgb_model.predict(test_data) / n_splits

        # ----- XGBoost -----
        xgb_model = xgb.XGBRegressor(**xgb_params, tree_method="hist", device="cuda")
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=False
        )
        oof_xgb[valid_idx] = xgb_model.predict(X_valid)
        test_preds_xgb += xgb_model.predict(test_data) / n_splits

        # ----- CatBoost -----
        cat_model = CatBoostRegressor(**cat_params, task_type='GPU', devices='0')
        cat_model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=[X.columns.get_loc(c) for c in cat_cols],
            use_best_model=True,
            verbose=False
        )
        oof_cat[valid_idx] = cat_model.predict(X_valid)
        test_preds_cat += cat_model.predict(test_data) / n_splits

    # === Meta Stacking Model for this seed ===
    stack_train = np.vstack([oof_lgb, oof_xgb, oof_cat]).T
    stack_test = np.vstack([test_preds_lgb, test_preds_xgb, test_preds_cat]).T

    meta_model = lgb.LGBMRegressor(
        num_leaves=7,
        learning_rate=0.03,
        n_estimators=1500,
        subsample=0.9,
        colsample_bytree=0.9,
        device='gpu'
    )
    meta_model.fit(stack_train, y, eval_metric="rmse")
    
    # Average predictions over seeds
    oof_preds_avg += meta_model.predict(stack_train) / len(seeds)
    test_preds_avg += meta_model.predict(stack_test) / len(seeds)

print(f"\nFinal OOF RMSE: {mean_squared_error(y, oof_preds_avg, squared=False):.5f}")


''''# with seeding average and extra tree
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# ----------------- Config -----------------
seed = 42  # use a single seed for efficiency
n_splits = 2
cat_cols = ["road_type", "lighting", "weather", "time_of_day"]

# ----------------- Model Params -----------------
lgb_params = {
    'num_leaves': 31, 'learning_rate': 0.01, 'n_estimators': 1500,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'device': 'gpu'
}
xgb_params = {
    'learning_rate': 0.01, 'n_estimators': 1500, 'tree_method': 'gpu_hist', 'device': 'cuda'
}
cat_params = {
    'iterations': 1500, 'learning_rate': 0.01, 'depth': 6, 'task_type': 'GPU', 'devices': '0:1', 'silent': True
}

# ----------------- Storage -----------------
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_et = np.zeros(len(X))

test_preds_lgb = np.zeros(len(test_data))
test_preds_xgb = np.zeros(len(test_data))
test_preds_cat = np.zeros(len(test_data))
test_preds_et = np.zeros(len(test_data))

# ----------------- KFold -----------------
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n===== Fold {fold+1} =====")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # --------- LightGBM ---------
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100)]
    )
    oof_lgb[valid_idx] = lgb_model.predict(X_valid)
    test_preds_lgb += lgb_model.predict(test_data) / n_splits

    # --------- XGBoost ---------
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=False
    )
    oof_xgb[valid_idx] = xgb_model.predict(X_valid)
    test_preds_xgb += xgb_model.predict(test_data) / n_splits

    # --------- CatBoost (multi-GPU) ---------
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        cat_features=[X.columns.get_loc(c) for c in cat_cols],
        use_best_model=True
    )
    oof_cat[valid_idx] = cat_model.predict(X_valid)
    test_preds_cat += cat_model.predict(test_data) / n_splits

    # --------- ExtraTrees (CPU) ---------
    et_model = ExtraTreesRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
    et_model.fit(X_train, y_train)
    oof_et[valid_idx] = et_model.predict(X_valid)
    test_preds_et += et_model.predict(test_data) / n_splits

# ----------------- Stack Meta-Model (LightGBM GPU) -----------------
stack_train = np.vstack([oof_lgb, oof_xgb, oof_cat, oof_et]).T
stack_test = np.vstack([test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_et]).T

meta_model = lgb.LGBMRegressor(
    num_leaves=7, learning_rate=0.03, n_estimators=1500,
    subsample=0.9, colsample_bytree=0.9, device='gpu'
)
meta_model.fit(stack_train, y, eval_metric="rmse", callbacks=[lgb.early_stopping(100)])
final_preds = meta_model.predict(stack_test)

    # Optional MLP meta-model (uncomment to use)
    # mlp_meta = MLPRegressor(hidden_layer_sizes=(16,8),
    #                         activation='relu', solver='adam',
    #                         max_iter=1000, random_state=seed)
    # mlp_meta.fit(stack_train, y)
    # oof_preds_avg += mlp_meta.predict(stack_train) / len(seeds)
    # test_preds_avg += mlp_meta.predict(stack_test) / len(seeds)

# ----------------- Evaluate -----------------
'''
'''
# ----------------- Submission -----------------
submission = pd.DataFrame({
    "id": test_data.index,
    "accident_risk": np.clip(test_preds_avg, 0, 1)
})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved successfully!")
'''


# eval for non-stacking
#rmse = mean_squared_error(y, oof_preds, squared=False)
#print(f"\nOOF RMSE: {rmse:.5f}")


# eval for stacking
oof_stack = meta_model.predict(stack_train)
rmse = mean_squared_error(y, oof_stack, squared=False)
print(f"\nFinal OOF RMSE after stacking: {rmse:.5f}")


'''# eval for  seeding average and extra tree
rmse = mean_squared_error(y, oof_preds_avg, squared=False)
print(f"\nOOF RMSE (Seed Averaged + Stacking): {rmse:.5f}")'''


import pandas as pd
import numpy as np

# Clip the predictions to valid range [0, 1]
final_preds = np.clip(test_preds_avg, 0, 1)

# Ensure you have the correct ID column from your test data
submission = pd.DataFrame({
    "id": test_data[ID],   # replace ID with the actual column name, e.g., "id"
    "accident_risk": final_preds
})

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("submission.csv saved successfully!")





