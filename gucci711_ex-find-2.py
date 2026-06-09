import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


# 特徴量生成
def add_custom_features(df):
    df["Elevation_binned"] = df["Elevation"] // 50
    df["Elevation_minus_Hydro"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    df["Hillshade_diff"] = df["Hillshade_9am"] - df["Hillshade_3pm"]
    df["Soil_Type_Combo_1"] = df["Soil_Type23"] + df["Soil_Type22"] + df["Soil_Type32"] + df["Soil_Type33"]
    df["Hydro_plus_Fire"] = df["Horizontal_Distance_To_Hydrology"] + df["Horizontal_Distance_To_Fire_Points"]
    df["Fire_div_Road"] = df["Horizontal_Distance_To_Fire_Points"] / (df["Horizontal_Distance_To_Roadways"] + 1)
    df["Hillshade_mean"] = (df["Hillshade_9am"] + df["Hillshade_Noon"] + df["Hillshade_3pm"]) / 3
    df["Slope_mul_Elevation"] = df["Slope"] * df["Elevation"]
    df["Euclid_Distance_Hydro"] = np.sqrt(df["Horizontal_Distance_To_Hydrology"]**2 + df["Vertical_Distance_To_Hydrology"]**2)
    df["Soil38_Wild1"] = df["Soil_Type38"] * df["Wilderness_Area1"]
    return df


# 読み込み・前処理
df_train = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")
df_test = pd.read_csv("/kaggle/input/forest-cover-type-prediction/test.csv")
df_train = add_custom_features(df_train)
df_test = add_custom_features(df_test)

rare = ['Soil_Type15','Soil_Type7','Soil_Type36','Soil_Type8','Soil_Type37','Soil_Type14','Soil_Type25','Soil_Type21','Soil_Type28','Soil_Type27']
df_train.drop(columns=rare, inplace=True)
df_test.drop(columns=rare, inplace=True)

features = [c for c in df_train.columns if c not in ["Id", "Cover_Type"]]
X = df_train[features]
y = df_train["Cover_Type"] - 1
X_test = df_test[features]





# モデル定義
lgb_params = dict(objective='multiclass', num_class=7, learning_rate=0.05,
                  num_leaves=64, max_depth=10, min_child_samples=20,
                  subsample=0.8, colsample_bytree=0.8, random_state=71)
rf_params = dict(n_estimators=300, max_depth=15, random_state=71, n_jobs=-1)
xgb_params = dict(objective='multi:softprob', num_class=7, learning_rate=0.05,
                  max_depth=10, subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss',
                  use_label_encoder=False, random_state=71, n_estimators=500)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=71)

oof_pred = np.zeros((len(X), 7))
test_pred = np.zeros((len(X_test), 7))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # LightGBM
    lgb_clf = lgb.LGBMClassifier(**lgb_params, n_estimators=2000)
    lgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    p_lgb = lgb_clf.predict_proba(X_va)

    # RandomForest
    rf_clf = RandomForestClassifier(**rf_params)
    rf_clf.fit(X_tr, y_tr)
    p_rf = rf_clf.predict_proba(X_va)

    # XGBoost
    xgb_clf = XGBClassifier(**xgb_params)
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, early_stopping_rounds=50)
    p_xgb = xgb_clf.predict_proba(X_va)

    # 平均アンサンブル（3モデル）
    p_avg = (p_lgb + p_rf + p_xgb) / 3

    oof_pred[va_idx] = p_avg
    test_pred += (lgb_clf.predict_proba(X_test) + rf_clf.predict_proba(X_test) + xgb_clf.predict_proba(X_test)) / 3

    print(f"Fold {fold} logloss:", log_loss(pd.get_dummies(y_va), p_avg))

print("OOF logloss:", log_loss(pd.get_dummies(y), oof_pred))
test_pred /= skf.n_splits


# 提出ファイル
submission = pd.DataFrame({
    "Id": df_test["Id"],
    "Cover_Type": np.argmax(test_pred, axis=1) + 1
})
submission.to_csv("submission_xgboost_ensemble.csv", index=False)

# 特徴量重要度（XGBoost, Gainベース）
importances = xgb_clf.feature_importances_
feat_importance = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feat_importance.sort_values(by="Importance", ascending=False).plot(kind="bar", x="Feature", figsize=(14, 5), color="skyblue")
plt.title("XGBoost Feature Importance (Gain)")
plt.tight_layout()
plt.show()

