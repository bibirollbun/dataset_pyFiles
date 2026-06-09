import numpy as np
import pandas as pd
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train_df.info()


test_df.info()


train_df


# EDA

# Target variable distribution
plt.figure(figsize=(8,5))
sns.histplot(train_df['accident_risk'], bins=50, kde=True, color="teal")
plt.title("Distribution of Target: accident_risk")
plt.show()



# Correlation heatmap for numerical features
plt.figure(figsize=(10,6))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()



# Countplots for categorical features
categorical_cols = ["road_type", "lighting", "weather", "time_of_day"]

fig, axes = plt.subplots(2, 2, figsize=(14,10))
for col, ax in zip(categorical_cols, axes.flatten()):
    sns.countplot(data=train_df, x=col, palette="Set2", ax=ax, order=train_df[col].value_counts().index)
    ax.set_title(f"Distribution of {col}")
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()



# Boxplots (categorical features vs target)
fig, axes = plt.subplots(2, 2, figsize=(14,10))
for col, ax in zip(categorical_cols, axes.flatten()):
    sns.boxplot(data=train_df, x=col, y="accident_risk", palette="Set3", ax=ax,
                order=train_df[col].value_counts().index)
    ax.set_title(f"Accident Risk by {col}")
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()



# Boolean features impact on target
bool_cols = ["road_signs_present", "public_road", "holiday", "school_season"]

fig, axes = plt.subplots(2, 2, figsize=(12,8))
for col, ax in zip(bool_cols, axes.flatten()):
    sns.barplot(data=train_df, x=col, y="accident_risk", ax=ax, palette="muted")
    ax.set_title(f"Mean Accident Risk by {col}")
plt.tight_layout()
plt.show()



# Encode boolean columns (True/False -> 1/0)
for col in bool_cols:
    train_df[col] = train_df[col].astype(int)


# One-hot encode categorical features
train_df_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)

print("Original shape:", train_df.shape)
print("Encoded shape:", train_df_encoded.shape)
train_df_encoded.head()


# Feature scaling
from sklearn.preprocessing import StandardScaler

num_cols = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents"]
scaler = StandardScaler()
train_df_encoded[num_cols] = scaler.fit_transform(train_df_encoded[num_cols])

train_df_encoded.head()



# Check for multicollinearity

corr_matrix = train_df_encoded.corr(numeric_only=True)
high_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
high_corr = high_corr[(high_corr < 1.0) & (high_corr > 0.8)]
print(high_corr.head(10))



X = train_df_encoded.drop(columns=['accident_risk', 'id'])
y = train_df_encoded['accident_risk']


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


# Define RMSE function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Setup KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)



# Corrected XGBoost K-Fold (use callbacks for early stopping)
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

xgb_oof = np.zeros(len(X))
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 8,
    "n_estimators": 3000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    print(f"XGBoost Fold {fold}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBRegressor(**xgb_params)

    # use callbacks for early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
        verbose=True
    )

    preds = model.predict(X_val)
    xgb_oof[val_idx] = preds
    print("Fold RMSE:", rmse(y_val, preds))

print("Overall XGBoost CV RMSE:", rmse(y, xgb_oof))



from lightgbm import early_stopping

# Train LightGBM with K-Fold (fixed)
lgb_oof = np.zeros(len(X))
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 256,
    "max_depth": 8,
    "n_estimators": 3000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"LightGBM Fold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=50, verbose=1)]
    )
    
    preds = model.predict(X_val)
    lgb_oof[val_idx] = preds
    print("Fold RMSE:", rmse(y_val, preds))

print("Overall LightGBM CV RMSE:", rmse(y, lgb_oof))



# Corrected CatBoost K-Fold
cat_oof = np.zeros(len(X))
cat_params = {
    "loss_function": "RMSE",
    "learning_rate": 0.03,
    "depth": 8,
    "iterations": 3000,
    "random_seed": 42,
    "logging_level": "Silent"   # replaces verbose=-1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    print(f"CatBoost Fold {fold}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostRegressor(**cat_params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,             # uses early stopping internally
        early_stopping_rounds=50,
        verbose=1
    )
    
    preds = model.predict(X_val)
    cat_oof[val_idx] = preds
    print("Fold RMSE:", rmse(y_val, preds))

print("Overall CatBoost CV RMSE:", rmse(y, cat_oof))



# Compare CV Scores
print("Final CV RMSEs:")
print("XGBoost:", rmse(y, xgb_oof))
print("LightGBM:", rmse(y, lgb_oof))
print("CatBoost:", rmse(y, cat_oof))



# Stacking with Ridge Regression
# Build meta-features from OOF predictions

from sklearn.linear_model import Ridge

# Stack out-of-fold predictions into a matrix (n_samples x n_models)
stack_X = np.vstack([xgb_oof, lgb_oof, cat_oof]).T
stack_y = y

print("Stack_X shape:", stack_X.shape)  # should be (num_samples, 3)



# Train a meta-learner
# Ridge regression as meta-learner
stack_model = Ridge(alpha=1.0, random_state=42)
stack_model.fit(stack_X, stack_y)

# Meta-learner OOF predictions
stack_oof = stack_model.predict(stack_X)
print("Stacked OOF RMSE:", rmse(stack_y, stack_oof))

# Check learned weights
print("Model Weights (XGB, LGBM, CatBoost):", stack_model.coef_)



# Retrain each base model on full dataset
final_xgb = xgb.XGBRegressor(**xgb_params)
final_xgb.fit(X, y)

final_lgb = lgb.LGBMRegressor(**lgb_params)
final_lgb.fit(X, y)

final_cat = CatBoostRegressor(**cat_params)
final_cat.fit(X, y, verbose=100)  # keep logging



# Encode boolean columns
for col in bool_cols:
    test_df[col] = test_df[col].astype(int)

# One-hot encode categoricals
test_df_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

# Ensure same columns as train
test_df_encoded = test_df_encoded.reindex(columns=X.columns, fill_value=0)

# Apply scaling (use the SAME scaler fitted on train)
test_df_encoded[num_cols] = scaler.transform(test_df_encoded[num_cols])



# Predict with final models
xgb_test = final_xgb.predict(test_df_encoded)
lgb_test = final_lgb.predict(test_df_encoded)
cat_test = final_cat.predict(test_df_encoded)

# Stack for meta-model
stack_test = np.vstack([xgb_test, lgb_test, cat_test]).T

# Final stacked predictions
final_preds = stack_model.predict(stack_test)



submission = pd.DataFrame({
    "id": test_df['id'],
    "accident_risk": final_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission saved!")


