import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.simplefilter('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col='id')
train.head(3)


import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

df = train.copy()
cat_features = ["Brand", "Material", "Size", "Compartments", "Laptop Compartment", 
                "Waterproof", "Style", "Color"]
df[cat_features] = df[cat_features].astype("category")

X = df.drop(columns=["Price"])
y = df["Price"]

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

rmse_scores = []
rmse_shuffled_scores = []
oof_predictions = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}/{n_splits} - Normal Training")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        objective="reg:squarederror",
        enable_categorical=True,
        tree_method="hist",
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=100
    )

    y_val_pred = model.predict(X_val)
    oof_predictions[val_idx] = y_val_pred

    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    rmse_scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

y_shuffled = shuffle(y, random_state=42).reset_index(drop=True)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}/{n_splits} - Shuffled Training")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_shuffled.iloc[train_idx], y_shuffled.iloc[val_idx]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        objective="reg:squarederror",
        enable_categorical=True,
        tree_method="hist",
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=100
    )

    y_val_pred = model.predict(X_val)

    rmse_shuffled = np.sqrt(mean_squared_error(y_val, y_val_pred))
    rmse_shuffled_scores.append(rmse_shuffled)
    print(f"Fold {fold+1} Shuffled RMSE: {rmse_shuffled:.4f}")

mean_rmse = np.mean(rmse_scores)
mean_rmse_shuffled = np.mean(rmse_shuffled_scores)

print(f"\nMean RMSE (Normal): {mean_rmse:.4f}")
print(f"Mean RMSE (Shuffled): {mean_rmse_shuffled:.4f}")

if abs(mean_rmse - mean_rmse_shuffled) < 1.0:
    print("\nðŸ”¹ The prediction accuracy is nearly the same as random â†’ Features likely do not function as meaningful explanatory variables!")
else:
    print("\nâœ… There is a noticeable difference in prediction accuracy â†’ Features have some explanatory power.")




