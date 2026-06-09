import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
import warnings, time

warnings.filterwarnings("ignore")

print("âœ… Libraries loaded successfully!")



def strong_fe(df, encoders=None, scalers=None, fit=True):
    df = df.copy()

    # ===========================
    # ğŸ”¤ Encode categorical
    # ===========================
    cat_cols = ["road_type", "lighting", "weather", "time_of_day"]
    if encoders is None: encoders = {}
    for c in cat_cols:
        if fit:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le
        else:
            df[c] = encoders[c].transform(df[c].astype(str))

    # ===========================
    # âš™ï¸� Boolean â†’ int
    # ===========================
    bool_cols = ["road_signs_present", "public_road"]
    for b in bool_cols:
        df[b] = df[b].astype(int)

    # ===========================
    # ğŸ§® Polynomial + Interactions
    # ===========================
    df['curvature_squared'] = df['curvature'] ** 2
    df['curvature_cubed'] = df['curvature'] ** 3
    df['speed_squared'] = df['speed_limit'] ** 2
    df['speed_curvature'] = df['speed_limit'] * df['curvature']
    df['lanes_curvature'] = df['num_lanes'] * df['curvature']
    df['speed_lanes'] = df['speed_limit'] * df['num_lanes']
    df['accidents_curvature'] = df['num_reported_accidents'] * df['curvature']
    df['accidents_speed'] = df['num_reported_accidents'] * df['speed_limit']

    # ===========================
    # ğŸ§  Domain features
    # ===========================
    df['high_risk_combo'] = ((df['curvature'] > 0.5) & (df['speed_limit'] >= 60)).astype(int)
    df['weather_lighting_risk'] = (
        ((df['weather'] == 'foggy') | (df['weather'] == 'rainy')) &
        ((df['lighting'] == 'dim') | (df['lighting'] == 'night'))
    ).astype(int)
    df['is_night'] = (df['lighting'] == 'night').astype(int)
    df['is_bad_weather'] = df['weather'].isin(['foggy', 'rainy']).astype(int)
    df['is_highway'] = (df['road_type'] == 'highway').astype(int)
    df['is_urban'] = (df['road_type'] == 'urban').astype(int)
    df['is_peak_time'] = df['time_of_day'].isin(['morning', 'evening']).astype(int)
    df['is_weekend'] = df['holiday'].astype(int)

    df['safety_score'] = (
        df['road_signs_present'].astype(int) * 2 +
        (df['lighting'] == 'daylight').astype(int) +
        (df['weather'] == 'clear').astype(int)
    )

    df['danger_score'] = (
        (df['curvature'] > 0.6).astype(int) +
        (df['speed_limit'] >= 60).astype(int) +
        df['is_bad_weather'] +
        df['is_night'] +
        (df['num_reported_accidents'] >= 2).astype(int)
    )

    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['risk_intensity'] = df['curvature'] * df['speed_limit'] / 50

    # ===========================
    # ğŸ”¢ Scaling
    # ===========================
    cont_cols = df.select_dtypes(include=['int', 'float']).columns
    cont_cols = [c for c in cont_cols if c not in ['id', 'accident_risk', 'target']]
    if scalers is None: scalers = {}
    
    if fit:
        qt = QuantileTransformer(output_distribution="normal", random_state=42)
        df[cont_cols] = qt.fit_transform(df[cont_cols])
        sc = StandardScaler()
        df[cont_cols] = sc.fit_transform(df[cont_cols])
        scalers["quantile"], scalers["standard"] = qt, sc
    else:
        qt, sc = scalers["quantile"], scalers["standard"]
        df[cont_cols] = qt.transform(df[cont_cols])
        df[cont_cols] = sc.transform(df[cont_cols])
    print(f"âœ… FE done. shape={df.shape}")
    return df, encoders, scalers




# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

# FE
train_fe, encoders, scalers = strong_fe(train, fit=True)
test_fe, _, _ = strong_fe(test, encoders=encoders, scalers=scalers, fit=False)

X = train_fe.drop(columns=["id", "accident_risk"])
y = train_fe["accident_risk"]
X_test = test_fe.drop(columns=["id"])
print("âœ… Data ready:", X.shape, y.shape)



# # LightGBM
# lgb_params = {
#     "num_leaves": np.arange(20, 100, 5),
#     "learning_rate": np.linspace(0.01, 0.5, 30),
#     "feature_fraction": np.linspace(0.6, 1.0, 20),
#     "bagging_fraction": np.linspace(0.6, 1.0, 20),
#     "min_data_in_leaf": np.arange(10, 100, 10),
#     "lambda_l1": np.linspace(0, 2, 20),
# }

# best_lgb_score, best_lgb_params = 1e9, None
# for i in range(40):
#     params = {k: np.random.choice(v) for k, v in lgb_params.items()}
#     model = lgb.LGBMRegressor(**params, random_state=42)
#     model.fit(X, y)
#     preds = model.predict(X)
#     rmse = mean_squared_error(y, preds, squared=False)
#     if rmse < best_lgb_score:
#         best_lgb_score, best_lgb_params = rmse, params
# print("ğŸ�† Best LGB params:", best_lgb_params)




from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# ================================================
# âš¡ Giai Ä‘oáº¡n 1: Random Search nhanh
# ================================================
print("ğŸš€ Random Search báº¯t Ä‘áº§u (200 iterations)...")

cat_param_space = {
    "depth": np.arange(4, 10),
    "learning_rate": np.linspace(0.01, 0.3, 20),
    "l2_leaf_reg": np.linspace(1, 10, 20),
    "border_count": np.arange(32, 256, 32)
}

best_cat_rmse = float("inf")
best_cat_params = None

for i in range(40):  # thá»­ 20 bá»™ tham sá»‘ ngáº«u nhiÃªn
    params = {k: np.random.choice(v) for k, v in cat_param_space.items()}
    model = CatBoostRegressor(
        iterations=200,                 # chá»‰ 200 vÃ²ng cho nhanh
        early_stopping_rounds=30,       # dá»«ng sá»›m náº¿u khÃ´ng cáº£i thiá»‡n
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        border_count=params["border_count"],
        random_state=42,
        verbose=1
    )
    model.fit(X, y)
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)
    print(f"ğŸ§© Trial {i+1:02d} | RMSE={rmse:.5f} | Params={params}")
    if rmse < best_cat_rmse:
        best_cat_rmse = rmse
        best_cat_params = params

print("\nğŸ�† Best CatBoost params (from Random Search):")
print(best_cat_params)
print(f"âœ… RMSE (train, quick eval): {best_cat_rmse:.5f}")



import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

kf = KFold(n_splits=7, shuffle=True, random_state=42)

# OOF & Test predictions
oof_lgb, oof_cat = np.zeros(len(X)), np.zeros(len(X))
test_preds_lgb, test_preds_cat = np.zeros(len(X_test)), np.zeros(len(X_test))

print("ğŸš€ Start 7-Fold CV training...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n==================== Fold {fold+1}/7 ====================")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # ============================
    # âš¡ LightGBM (upgraded)
    # ============================
    
    lgb_model = LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )

    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
    )

    preds_val_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)
    preds_test_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration_)
    oof_lgb[val_idx] = preds_val_lgb
    test_preds_lgb += preds_test_lgb / kf.n_splits

    # ============================
    # âš¡ CatBoost (upgraded)
    # ============================
    cat_model = CatBoostRegressor(
        **best_cat_params,
        iterations=3000,                   # tÄƒng sá»‘ vÃ²ng há»�c
        early_stopping_rounds=100,
        loss_function="RMSE",
        random_seed=42,
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    preds_val_cat = cat_model.predict(X_val)
    preds_test_cat = cat_model.predict(X_test)
    oof_cat[val_idx] = preds_val_cat
    test_preds_cat += preds_test_cat / kf.n_splits

    # ğŸ�¯ Fold summary
    rmse_lgb_fold = mean_squared_error(y_val, preds_val_lgb, squared=False)
    rmse_cat_fold = mean_squared_error(y_val, preds_val_cat, squared=False)
    print(f"ğŸ“˜ Fold {fold+1} â€” LGB RMSE: {rmse_lgb_fold:.5f} | Cat RMSE: {rmse_cat_fold:.5f}")

# ============================
# âœ… Final results
# ============================
rmse_lgb = mean_squared_error(y, oof_lgb, squared=False)
rmse_cat = mean_squared_error(y, oof_cat, squared=False)
print("\n==================== Summary ====================")
print(f"âœ… CV RMSE â€” LightGBM: {rmse_lgb:.5f}")
print(f"âœ… CV RMSE â€” CatBoost:  {rmse_cat:.5f}")



w_lgb = 1 / rmse_lgb
w_cat = 1 / rmse_cat
w_sum = w_lgb + w_cat
w_lgb /= w_sum
w_cat /= w_sum

final_preds = test_preds_lgb * w_lgb + test_preds_cat * w_cat

print(f"ğŸ“Š Blend weights â€” LGB: {w_lgb:.3f}, CatBoost: {w_cat:.3f}")
sub = pd.DataFrame({
    "id": test["id"],
    "accident_risk": final_preds
})
sub.to_csv("submission.csv", index=False)
print("âœ… Submission saved as submission.csv")


