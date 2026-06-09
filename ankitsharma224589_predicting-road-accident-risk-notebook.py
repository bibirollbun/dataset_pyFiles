# Basic libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from lightgbm import early_stopping, log_evaluation
# Ignore warnings for clean output
import warnings
warnings.filterwarnings("ignore")



# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

train.head()



# Target distribution
plt.figure(figsize=(7,5))
sns.histplot(train['accident_risk'], bins=30, kde=True, color="royalblue")
plt.title("Distribution of Accident Risk")
plt.show()

# Missing values check
train.isnull().sum().sort_values(ascending=False).head(10)



# Separate features and target
TARGET = "accident_risk"
ID = "id"

X = train.drop([ID, TARGET], axis=1)
y = train[TARGET]
X_test = test.drop(ID, axis=1)

print("Features:", X.shape, " Target:", y.shape)



# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print(f"Number of categorical features: {len(cat_cols)}")
print(f"Number of numerical features: {len(num_cols)}")
print("\nCategorical columns:", cat_cols)
print("\nNumerical columns:", num_cols[:10], "...")  # show only first 10 if many




# Copy train/test
X_encoded = X.copy()
X_test_encoded = X_test.copy()

# ===== Missing value handling =====
for col in X_encoded.columns:
    if X_encoded[col].dtype == "object":
        X_encoded[col] = X_encoded[col].fillna("missing")
        X_test_encoded[col] = X_test_encoded[col].fillna("missing")
    else:
        X_encoded[col] = X_encoded[col].fillna(X_encoded[col].median())
        X_test_encoded[col] = X_test_encoded[col].fillna(X_encoded[col].median())

# ===== Label Encoding for categoricals =====
for col in cat_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    X_test_encoded[col] = le.transform(X_test_encoded[col])

# ===== Scale numerical features =====
scaler = StandardScaler()
X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])
X_test_encoded[num_cols] = scaler.transform(X_test_encoded[num_cols])

print("âœ… Encoding + Scaling Done")
print("X_encoded shape:", X_encoded.shape)
print("X_test_encoded shape:", X_test_encoded.shape)



rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

X_train, X_valid, y_train, y_valid = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_valid)

rmse = mean_squared_error(y_valid, y_pred, squared=False)
print("ðŸŒ² Random Forest RMSE:", rmse)

# Plot feature importances
feat_importances = pd.Series(rf.feature_importances_, index=X_encoded.columns)
feat_importances.nlargest(20).plot(kind="barh", figsize=(8,6))
plt.title("Top 20 Feature Importances - RandomForest")
plt.show()




kf = KFold(n_splits=10, shuffle=True, random_state=42)
lgb_preds = np.zeros(len(X_test_encoded))
lgb_oof = np.zeros(len(X_encoded))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_encoded, y)):
    print(f"Fold {fold+1}")
    
    X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": 42,
        "learning_rate": 0.015,
        "num_leaves": 128,
        "max_depth": -1,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 1e-3,
        "lambda_l2": 1e-3,
        "min_child_weight": 15,
    }
    
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=30000,
        callbacks=[early_stopping(stopping_rounds=400), log_evaluation(200)]
    )
    
    lgb_oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    lgb_preds += model.predict(X_test_encoded, num_iteration=model.best_iteration) / kf.n_splits



xgb_preds = np.zeros(len(X_test_encoded))
xgb_oof = np.zeros(len(X_encoded))

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_encoded, y)):
    print(f"Fold {fold+1}")
    
    X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=30000,
        learning_rate=0.015,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        colsample_bylevel=0.9,
        colsample_bynode=0.9,
        reg_alpha=1e-3,
        reg_lambda=1e-3,
        min_child_weight=10,
        gamma=0.1,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=400,
        verbose=200
    )
    
    xgb_oof[val_idx] = model.predict(X_val, iteration_range=(0, model.best_iteration))
    xgb_preds += model.predict(X_test_encoded, iteration_range=(0, model.best_iteration)) / kf.n_splits



# Stack OOF preds
stack_train = np.vstack([lgb_oof, xgb_oof]).T
stack_test = np.vstack([lgb_preds, xgb_preds]).T

meta_model = Ridge(alpha=1e-3)
meta_model.fit(stack_train, y)

final_preds = meta_model.predict(stack_test)

print("âœ… Stacking done. This usually improves 0.0001â€“0.0003 RMSE over plain avg.")



best_rmse = 999
best_w = None

for w in np.linspace(0, 1, 21):  # weights from 0.0 to 1.0 in 0.05 steps
    blend = w * lgb_oof + (1 - w) * xgb_oof   # blend OOF preds
    rmse = mean_squared_error(y, blend, squared=False)
    if rmse < best_rmse:
        best_rmse = rmse
        best_w = w

print(f"Best weight for LGB: {best_w:.2f}, RMSE: {best_rmse:.5f}")

# Use best weight on test predictions
final_preds = best_w * lgb_preds + (1 - best_w) * xgb_preds



submission = pd.DataFrame({
    "id": test[ID],
    "accident_risk": final_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()


