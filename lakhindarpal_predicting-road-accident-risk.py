# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# load datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
og_df = pd.read_csv("/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv")


print("Train data")
print("Shape", train_df.shape)
train_df.head()


print("Test data")
print("Shape", test_df.shape)
test_df.head()


print("Original data")
print("Shape", og_df.shape)
og_df.head()


# compare with original dataset
og_df.columns.equals(train_df.drop("id", axis=1).columns)


# Merge with original dataset
train_df = pd.concat([train_df.drop("id", axis=1), og_df], ignore_index=True)
tran_df = train_df.drop_duplicates()

print("Train data shape", train_df.shape)


print("Missing Values")
print("Train data:", train_df.isna().sum().sum())
print("Test data:", test_df.isna().sum().sum())


train_df.info()


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(train_df['accident_risk'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_title('Accident Risk Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Accident Risk')
axes[0].set_ylabel('Frequency')
axes[0].axvline(train_df['accident_risk'].mean(), color='red', linestyle='--', label=f'Mean: {train_df["accident_risk"].mean():.3f}')
axes[0].legend()

axes[1].boxplot(train_df['accident_risk'], vert=True)
axes[1].set_title('Accident Risk Boxplot', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Accident Risk')

stats.probplot(train_df['accident_risk'], dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()


train_df["accident_risk"].describe()


train_df.describe(include='object')


cat_features = ['road_type', 'lighting', 'weather', 'time_of_day', 'road_signs_present', 
                        'public_road', 'holiday', 'school_season']
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for idx, col in enumerate(cat_features):
    if train_df[col].dtype == 'bool':
        data = train_df.groupby(col)['accident_risk'].mean().sort_values()
    else:
        data = train_df.groupby(col)['accident_risk'].mean().sort_values()
    
    data.plot(kind='bar', ax=axes[idx], edgecolor='black', alpha=0.8)
    axes[idx].set_title(f'Avg Accident Risk by {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Avg Accident Risk')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


num_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, col in enumerate(num_features):
    # Distribution
    axes[0, idx].hist(train_df[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
    axes[0, idx].set_xlabel(col)
    axes[0, idx].set_ylabel('Frequency')
    
    # Relationship with target
    if col in ['num_lanes', 'speed_limit']:
        grouped = train_df.groupby(col)['accident_risk'].agg(['mean', 'std', 'count'])
        axes[1, idx].bar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                        color='lightcoral', edgecolor='black', alpha=0.8, capsize=5)
        axes[1, idx].set_xlabel(col)
    else:
        axes[1, idx].scatter(train_df[col], train_df['accident_risk'], alpha=0.3, s=1, color='green')
        z = np.polyfit(train_df[col], train_df['accident_risk'], 2)
        p = np.poly1d(z)
        x_line = np.linspace(train_df[col].min(), train_df[col].max(), 100)
        axes[1, idx].plot(x_line, p(x_line), "r-", linewidth=2, label='Trend')
        axes[1, idx].legend()
        axes[1, idx].set_xlabel(col)
    
    axes[1, idx].set_ylabel('Accident Risk')
    axes[1, idx].set_title(f'{col} vs Accident Risk', fontsize=12, fontweight='bold')
    axes[1, idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# Heatmap of the correlation matrix between categorical features (one-hot encoded)
data_encoded = pd.get_dummies(test_df[['road_type', 'lighting', 'weather', 'time_of_day']], drop_first=True)

# Compute correlation matrix
cat_corr_matrix = data_encoded.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cat_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Categorical Variables')
plt.show()


num_corr_matrix = train_df[num_features + ['accident_risk']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(num_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Numerical Feature Correlation Heatmap')
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

target = "accident_risk"

X = train_df.drop(columns=[target])
y = train_df[target]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_test = test_df.drop(columns=["id"], errors="ignore")

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("Categrical Columns", cat_cols)


for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_valid[col] = X_valid[col].astype('category')
    X_test[col] = X_test[col].astype('category')


# CatBoost Model
cat_model = CatBoostRegressor(
    iterations=4000,
    learning_rate=0.01,
    depth=7,
    eval_metric="RMSE",
    task_type="GPU",
    devices="0:1",
    random_seed=42,
    verbose=100
)
cat_model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    cat_features=cat_cols,
    early_stopping_rounds=50
)
preds_cat = cat_model.predict(X_valid)
rmse_cat = mean_squared_error(y_valid, preds_cat, squared=False)

print(f"CatBoost RMSE: {rmse_cat:.5f}")


# LightGBM Model
lgb_model = LGBMRegressor(
    n_estimators=4000,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    device="gpu",
    early_stopping_rounds=50,
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    categorical_feature=cat_cols,
)
preds_lgb = lgb_model.predict(X_valid)
rmse_lgb = mean_squared_error(y_valid, preds_lgb, squared=False)

print(f"LightGBM RMSE: {rmse_lgb:.5f}")


# XGBoost Model
xgb_model = XGBRegressor(
    enable_categorical=True,
    n_estimators=4000,
    learning_rate=0.01,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
    device="cuda",
    eval_metric="rmse",
    early_stopping_rounds=50
)

# Fit the model
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=100
)

# Make predictions
preds_xgb = xgb_model.predict(X_valid)
rmse_xgb = mean_squared_error(y_valid, preds_xgb, squared=False)

print(f"XGBoost RMSE: {rmse_xgb:.5f}")


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

stacked_X = np.column_stack([preds_xgb, preds_lgb, preds_cat])
stacked_y = y_valid

# Scale meta features for regularized models
scaler = StandardScaler()
stacked_X_scaled = scaler.fit_transform(stacked_X)

# --- Meta Models ---
meta_models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(alpha=0.001, random_state=42),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)
}

best_rmse = float("inf")
best_model_name = None
best_meta_model = None

for name, model in meta_models.items():
    model.fit(stacked_X_scaled, stacked_y)
    preds = model.predict(stacked_X_scaled)
    rmse = mean_squared_error(stacked_y, preds, squared=False)
    print(f"{name} RMSE: {rmse:.5f}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model_name = name
        best_meta_model = model

print(f"\nBest Meta Model: {best_model_name} (RMSE: {best_rmse:.5f})")


import joblib

# Save all models for reuse (Streamlit)
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(lgb_model, "lgb_model.pkl")
joblib.dump(cat_model, "cat_model.pkl")
joblib.dump(best_meta_model, f"meta_{best_model_name.lower()}.pkl")
joblib.dump(scaler, "meta_scaler.pkl")

print("✅ Models saved successfully.")


# Predictions from each model on the test data
preds_cat_test = cat_model.predict(X_test)
preds_lgb_test = lgb_model.predict(X_test)
preds_xgb_test = xgb_model.predict(X_test)

# Stack the predictions from all models
stacked_preds_test = np.column_stack([preds_xgb_test, preds_lgb_test, preds_cat_test])
# Scale meta features
stacked_preds_scaled = scaler.transform(stacked_preds_test)

# Use the best meta model to make the final predictions
final_preds_test = best_meta_model.predict(stacked_preds_scaled)

# Create the submission file
submission = pd.DataFrame({
    "id": test_df["id"],
    "accident_risk": final_preds_test
})

# Save the submission to a CSV file
submission.to_csv("submission.csv", index=False)

print("✅ submission.csv created!")




