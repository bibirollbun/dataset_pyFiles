
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


y = train["accident_risk"]

train["is_train"] = 1
test["is_train"] = 0

combined = pd.concat([train.drop(columns=["accident_risk"]), test], ignore_index=True)

categorical_cols = ['road_type', 'lighting', 'weather', 'road_signs_present',
                    'public_road', 'time_of_day', 'holiday', 'school_season']

numeric_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']


le = LabelEncoder()
for col in categorical_cols:
    combined[col] = le.fit_transform(combined[col])

scaler = MinMaxScaler()
combined[numeric_cols] = scaler.fit_transform(combined[numeric_cols])

train_processed = combined[combined["is_train"] == 1].drop(columns=["is_train"])
test_processed  = combined[combined["is_train"] == 0].drop(columns=["is_train"])

X = train_processed.drop(columns=["id"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)


cat_model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    random_seed=42,
    verbose=0
)

lgb_model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


ensemble = VotingRegressor([
    ('cat', cat_model),    
    ('lgb', lgb_model)
])

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE (Ensemble): {rmse:.6f}")


ensemble.fit(X, y)
test_preds = ensemble.predict(test_processed.drop(columns=["id"]))

submission = pd.DataFrame({
    "id": test["id"],                
    "accident_risk": test_preds
})

submission.to_csv("submission.csv", index=False)

print(submission.head())

