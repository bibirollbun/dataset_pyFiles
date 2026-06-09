import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import time


# === Load Data ===
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


# === Define Columns ===
target_col = "Listening_Time_minutes"
id_col = "id"
cat_cols = ["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]


# === Feature Engineering ===
def feature_engineering(df):
    df = df.copy()
    df["Episode_Length_minutes"] = df["Episode_Length_minutes"].fillna(df["Episode_Length_minutes"].median())
    df["Guest_Popularity_percentage"] = df["Guest_Popularity_percentage"].fillna(0)
    df["Length_per_Ad"] = df["Episode_Length_minutes"] / (df["Number_of_Ads"] + 1)
    df["Popularity_Diff"] = df["Host_Popularity_percentage"] - df["Guest_Popularity_percentage"]
    df["Host_x_Guest"] = df["Host_Popularity_percentage"] * df["Guest_Popularity_percentage"]
    df["is_weekend"] = df["Publication_Day"].isin(["Saturday", "Sunday"]).astype(int)
    df["is_night"] = df["Publication_Time"].isin(["Night"]).astype(int)
    return df

train = feature_engineering(train)
test = feature_engineering(test)


# === Prepare Features ===
features = [col for col in train.columns if col not in [target_col, id_col]]
X = train[features].copy()
y = train[target_col].copy()
X_test = test[features].copy()



# === Encode Categorical Features ===
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

X[cat_cols] = preprocessor.fit_transform(X[cat_cols])
X_test[cat_cols] = preprocessor.transform(X_test[cat_cols])


# === Fill Remaining Numeric Missing Values ===
numeric_cols = [col for col in X.columns if col not in cat_cols]
num_imputer = SimpleImputer(strategy='median')
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])


# === Apply log1p transformation to skewed numeric features ===
skewed_features = ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Length_per_Ad"]
for col in skewed_features:
    X[col] = np.log1p(X[col])
    X_test[col] = np.log1p(X_test[col])


# === Define Base Models ===
base_models = [
    ("xgb", XGBRegressor(n_estimators=2500, learning_rate=0.01, max_depth=12, subsample=0.85, colsample_bytree=0.7, random_state=42, tree_method='hist', verbosity=0)),
    ("lgb", LGBMRegressor(n_estimators=2500, learning_rate=0.01, max_depth=12, subsample=0.85, colsample_bytree=0.7, random_state=42, device_type='cpu')),
    ("cat", CatBoostRegressor(iterations=2500, learning_rate=0.01, depth=12, subsample=0.85, random_state=42, task_type='CPU', verbose=0))
]


# === Define Stacking Model ===
stack = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=1.0, random_state=42)
)


# === K-Fold Cross Validation ===
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\u25b6\ufe0f Fold {fold}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    stack.fit(X_train, y_train)

    oof_preds[val_idx] = stack.predict(X_val)
    test_preds += stack.predict(X_test) / FOLDS


# === Score ===
rmse = np.sqrt(mean_squared_error(y, oof_preds))


# === Create Submission ===
submission = pd.DataFrame({
    "id": test[id_col],
    "Listening_Time_minutes": test_preds
})

start_time = time.time()
try:
    submission.to_csv("submission.csv", index=False)
    elapsed_time = time.time() - start_time
    print(f"Validation RMSE: {rmse:.5f}")
    print(f"ğŸš€ Submission saved in {elapsed_time:.2f} seconds!")
except Exception as e:
    print(f"â�Œ Error saving submission: {e}")


