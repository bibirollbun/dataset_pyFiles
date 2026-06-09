import pandas as pd
data = '/kaggle/input/playground-series-s5e10/train.csv'

df = pd.read_csv(data)


df.isnull().sum()


df.shape
df.head()


import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(df['accident_risk'], bins=20, kde=True)
plt.title("Distribution of Accident Risk")


for col in ['road_type','num_lanes','curvature','speed_limit','road_signs_present', 'lighting', 'weather', 'time_of_day', 'holiday','num_reported_accidents','public_road']:
    plt.figure()
    sns.boxplot(x=col, y='accident_risk', data=df)
    plt.title(f"Accident Risk by {col}")



import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('accident_risk ~ C(road_type) + C(lighting) + C(weather) + C(time_of_day) + C(holiday)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table.sort_values('F', ascending=False))


ols('accident_risk ~ C(lighting) * C(weather)', data=df).fit().summary()



df['low_visibility'] = (
    (df['lighting'].isin(['dim','night'])) |
    (df['weather'].isin(['foggy','rainy']))
).astype(int)

df['lighting_weather'] = df['lighting'] + '_' + df['weather']


# speed: mph? km/h? (dataset của bạn giống hạn mức kiểu 35, 60, 70 → coi là mph hoặc km/h tùy đề;
# chỉ cần nhất quán. Ví dụ coi là km/h -> đổi sang m/s):
v = df['speed_limit'] * 1000/3600.0  # km/h -> m/s

# reaction time theo lighting/time_of_day (có lý do con người):
tau_map = {'daylight': 1.0, 'dim': 1.2, 'night': 1.5}
df['tau'] = df['lighting'].map(tau_map).fillna(1.2)

# friction theo weather (ma sát giảm khi mưa/sương):
mu_map = {'clear': 0.8, 'rainy': 0.55, 'foggy': 0.40}
df['mu'] = df['weather'].map(mu_map).fillna(0.7)

g = 9.81
df['stop_dist'] = v*df['tau'] + (v**2)/(2*df['mu']*g)



df['lateral_load'] = (v**2) * df['curvature']


light_score   = {'daylight': 0.0, 'dim': 0.4, 'night': 1.0}
weather_score = {'clear':   0.0, 'rainy': 0.5, 'foggy': 0.8}
df['light_sc']   = df['lighting'].map(light_score).astype(float)
df['weather_sc'] = df['weather'].map(weather_score).astype(float)

# visibility_index càng cao → tầm nhìn càng kém
df['visibility_index'] = df['light_sc'] + df['weather_sc'] + 0.5*df['light_sc']*df['weather_sc']



lanes_effect   = 1.0 / (1.0 + 0.15*df['num_lanes'].clip(lower=1))   # nhiều làn → giảm
signs_effect   = (1.0 - 0.15*df['road_signs_present'].astype(int))  # có biển báo → giảm
public_effect  = (1.0 + 0.10*df['public_road'].astype(int))         # đường công cộng → tăng



# kết hợp hai rủi ro + các effect + visibility
df['hazard_core'] = 0.6*df['stop_dist'] + 0.4*df['lateral_load']
df['hazard_score'] = (
    df['hazard_core'] *
    (1.0 + 0.35*df['visibility_index']) *
    lanes_effect * signs_effect * public_effect
)


from catboost.utils import get_gpu_device_count
print("Available GPUs:", get_gpu_device_count())



# from sklearn.model_selection import KFold
# from catboost import CatBoostRegressor, Pool
# import numpy as np

# TARGET = 'accident_risk'
# ID_COLS = ['id'] if 'id' in df.columns else []

# # (tuỳ dữ liệu đã tạo) liệt kê các cột categorical
# cat_cols = ['road_type','lighting','weather','time_of_day','holiday','school_season','lighting_weather']
# features = [c for c in df.columns if c not in ID_COLS + [TARGET]]

# X = df[features].copy()
# y = df[TARGET].values

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# oof = np.zeros(len(df))
# models = []

# params = dict(
#     loss_function='RMSE',
#     task_type='GPU', devices='0',
#     iterations=5000,           # nhiều vòng + LR nhỏ + early stopping
#     learning_rate=0.02,
#     depth=8,
#     l2_leaf_reg=6.0,
#     random_strength=0.8,
#     bootstrap_type='Bernoulli', subsample=0.8,
#     eval_metric='RMSE',
#     od_type='Iter', od_wait=300,  # early stopping
#     verbose=200
# )

# for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y), 1):
#     X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
#     y_trn, y_val = y[trn_idx], y[val_idx]
#     train_pool = Pool(X_trn, y_trn, cat_features=[X.columns.get_loc(c) for c in cat_cols if c in X.columns])
#     valid_pool = Pool(X_val, y_val, cat_features=[X.columns.get_loc(c) for c in cat_cols if c in X.columns])

#     model = CatBoostRegressor(**params)
#     model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
#     oof[val_idx] = model.predict(valid_pool)
#     models.append(model)

# rmse = (( (oof - y)**2 ).mean())**0.5
# print("OOF RMSE:", rmse)



import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor, Pool

# ==== Cấu hình chung ====
TARGET = 'accident_risk'
# các cột categorical gốc của bạn (phải tồn tại trong df)
base_cat_cols = ['road_type','lighting','weather','time_of_day','holiday','school_season','lighting_weather']
FOLDS = 5
SEED = 42

n = len(df)

# ==== Search space 3 cấu hình (chọn best theo OOF) ====
search_space = [
    dict(iterations=10000, learning_rate=0.012, depth=8,  l2_leaf_reg=20, subsample=0.75, random_strength=1.5),
]

best_oof = 1e9
best_cfg = None
best_models = None
best_feature_order = None
best_fold_rmses = None
best_cat_in_use = None
best_oof_pred = None

for cfg in search_space:
    print("\n=== Trying config:", cfg, "===\n")
    oof_pred = np.zeros(n)
    fold_rmses = []
    models_tmp = []

    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (trn_idx, val_idx) in enumerate(kf.split(df), 1):
        trn = df.iloc[trn_idx].copy()
        val = df.iloc[val_idx].copy()

        # --- Stage 1: y_phys (Ridge trên hazard_score) -> giữ làm feature ---
        base = Ridge(alpha=1.0)
        base.fit(trn[['hazard_score']], trn[TARGET])
        trn['y_phys'] = base.predict(trn[['hazard_score']]).clip(0, 1)
        val['y_phys'] = base.predict(val[['hazard_score']]).clip(0, 1)

        # --- Stage 2: CatBoost dự đoán trực tiếp y (y_phys là 1 feature) ---
        X_trn = trn.drop(columns=[TARGET]).copy()
        X_val = val.drop(columns=[TARGET]).copy()
        y_trn = trn[TARGET].values
        y_val = val[TARGET].values

        # categorical -> str (tránh lỗi convert float)
        cat_in_use = [c for c in base_cat_cols if c in X_trn.columns]
        for c in cat_in_use:
            X_trn[c] = X_trn[c].astype(str)
            X_val[c] = X_val[c].astype(str)

        # (optional) bool -> int nếu tồn tại
        for b in X_trn.select_dtypes(include=['bool']).columns:
            X_trn[b] = X_trn[b].astype(int)
        for b in X_val.select_dtypes(include=['bool']).columns:
            X_val[b] = X_val[b].astype(int)

        cat_idx = [X_trn.columns.get_loc(c) for c in cat_in_use]

        train_pool = Pool(X_trn, y_trn, cat_features=cat_idx)
        valid_pool = Pool(X_val, y_val, cat_features=cat_idx)

        cb = CatBoostRegressor(
            task_type='GPU', devices='0',
            loss_function='RMSE', eval_metric='RMSE',
            leaf_estimation_method='Newton', leaf_estimation_iterations=2,
            one_hot_max_size=16, max_ctr_complexity=2, ctr_target_border_count=128,
            bootstrap_type='Bernoulli', od_type='Iter', od_wait=600, verbose=200,
            **cfg
        )
        cb.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        models_tmp.append(cb)

        val_pred = cb.predict(valid_pool)
        oof_pred[val_idx] = val_pred

        rmse_fold = mean_squared_error(y_val, val_pred, squared=False)
        fold_rmses.append(rmse_fold)
        print(f"[Fold {fold}] RMSE: {rmse_fold:.6f}")

        # lưu feature order và cat list từ fold đầu (dùng cho predict test)
        if fold == 1:
            feature_order = X_trn.columns.tolist()
            cat_list_for_test = cat_in_use[:]

    oof_rmse = mean_squared_error(df[TARGET].values, oof_pred, squared=False)
    print("Config OOF RMSE:", f"{oof_rmse:.6f} | Folds:", [f"{x:.6f}" for x in fold_rmses])

    if oof_rmse < best_oof:
        best_oof = oof_rmse
        best_cfg = cfg
        best_models = models_tmp
        best_feature_order = feature_order
        best_cat_in_use = cat_list_for_test
        best_fold_rmses = fold_rmses
        best_oof_pred = oof_pred.copy()

print("\n==============================")
print("Best OOF:", f"{best_oof:.6f}")
print("Best config:", best_cfg)
print("Fold RMSEs:", [f"{x:.6f}" for x in best_fold_rmses])

# Lưu các biến để dùng cho predict test
models = best_models
feature_order_for_test = best_feature_order
cat_cols_for_test = best_cat_in_use
cb_best_config = best_cfg
oof_pred = best_oof_pred  # OOF dự đoán tốt nhất (nếu bạn muốn lưu)



from sklearn.metrics import mean_squared_error

# chỉ mô hình vật lý
rmse_phys = mean_squared_error(df[TARGET], oof_y_phys, squared=False)

# chỉ mô hình residual (CatBoost)
rmse_res_only = mean_squared_error(df[TARGET], oof_y_phys + oof_pred_res, squared=False)

# mô hình full (y_phys + residual)
rmse_full = mean_squared_error(df[TARGET], oof_pred_final, squared=False)

print(f"RMSE y_phys only: {rmse_phys:.6f}")
print(f"RMSE y_phys + residual: {rmse_full:.6f}")
print(f"Giảm được: {rmse_phys - rmse_full:.6f}")



import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from catboost import Pool

# ===== 1) Base model (Ridge) trên FULL train để lấy y_phys cho test =====
TARGET = 'accident_risk'

base_full = Ridge(alpha=1.0)
base_full.fit(df[['hazard_score']], df[TARGET])

df_test = df_test.copy()
df_test['y_phys'] = base_full.predict(df_test[['hazard_score']]).clip(0, 1)

# ===== 2) Chuẩn bị X_test đúng thứ tự cột như lúc train residual =====
# features_all, drop_cols, cat_cols phải là đúng các biến đã dùng trong KFold training
X_test = df_test[[c for c in features_all if c in df_test.columns]].copy()

# đảm bảo thứ tự cột khớp hoàn toàn với lúc train
X_test = X_test.reindex(columns=features_all)

# index các cột categorical theo X_test hiện tại
cat_idx_test = [X_test.columns.get_loc(c) for c in cat_cols if c in X_test.columns]
test_pool = Pool(X_test, cat_features=cat_idx_test)

# ===== 3) Predict residual trên test bằng các model theo fold rồi average =====
# 'models' là list CatBoostRegressor đã lưu từ vòng K-Fold train residual
test_res_preds = [m.predict(test_pool) for m in models]
residual_pred_test = np.mean(np.column_stack(test_res_preds), axis=1)

# ===== 4) Dự đoán cuối cùng cho test =====
test_pred = np.clip(df_test['y_phys'].values + residual_pred_test, 0, 1)

# ===== 5) Tạo submission =====
sub = pd.DataFrame({
    'id': df_test['id'],
    'accident_risk': test_pred
})
sub.to_csv('submission.csv', index=False)
print(sub.head())
print("Saved -> submission.csv")


