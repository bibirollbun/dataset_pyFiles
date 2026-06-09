## importing libraries 
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge


## reading the data
road_accident_train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
road_accident_train_df.head()


road_accident_train_df['accident_risk'].describe()


road_accident_train_df['road_type'].unique()


road_accident_train_df['num_lanes'].unique()


road_accident_train_df['curvature'].describe()


road_accident_train_df['curvature'].plot(kind = 'box')


road_accident_train_df['speed_limit'].unique()


road_accident_train_df


road_accident_train_df['lighting'].unique()


road_accident_train_df['weather'].unique()


road_accident_train_df['road_signs_present'].unique()


road_accident_train_df['public_road'].unique()


road_accident_train_df['time_of_day'].unique()


road_accident_train_df['holiday'].unique()


road_accident_train_df['school_season'].unique()


road_accident_train_df['num_reported_accidents'].describe()


road_accident_train_df['num_reported_accidents'].plot(kind = 'box')


# Identify categorical and numerical columns
categorical_cols = road_accident_train_df.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_cols = road_accident_train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove id and target from features
categorical_cols = [col for col in categorical_cols if col not in ['id', 'accident_risk']]
numerical_cols = [col for col in numerical_cols if col not in ['id', 'accident_risk']]

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")


# ## Remove id and target from features
# def identify_column_types(df, exclude_cols=['id', 'accident_risk']):
#     """Identify categorical and numerical columns"""
#     categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
#     numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
#     categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
#     numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
#     return categorical_cols, numerical_cols


# ## label encoder feature extraction
# def label_encode_features(df, categorical_cols, fitted_encoders=None):
#     """
#     Label encode categorical columns
#     Returns: encoded dataframe and encoder dictionary
#     """
#     df_encoded = df.copy()
#     encoders = fitted_encoders if fitted_encoders else {}
    
#     for col in categorical_cols:
#         if fitted_encoders is None:
#             le = LabelEncoder()
#             df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
#             encoders[col] = le
#         else:
#             df_encoded[col] = fitted_encoders[col].transform(df_encoded[col].astype(str))
    
#     return df_encoded, encoders


# ## onehot encoder
# def onehot_encode_features(X_train, X_test, categorical_cols, numerical_cols):
#     """
#     One-hot encode categorical columns using ColumnTransformer
#     Returns: transformed arrays, preprocessor, and feature names
#     """
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numerical_cols),
#             # ('num', MinMaxScaler(), numerical_cols),
#             ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
#         ])
    
#     X_train_processed = preprocessor.fit_transform(X_train)
#     X_test_processed = preprocessor.transform(X_test)
    
#     # Get feature names
#     ohe_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
#     feature_names = numerical_cols + list(ohe_features)
    
#     return X_train_processed, X_test_processed, preprocessor, feature_names


# Identify column types
categorical_cols = road_accident_train_df.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_cols = road_accident_train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

categorical_cols = [col for col in categorical_cols if col not in ['id', 'accident_risk']]
numerical_cols = [col for col in numerical_cols if col not in ['id', 'accident_risk']]

# Prepare features
X = road_accident_train_df.drop(['id', 'accident_risk'], axis=1).copy()
y = road_accident_train_df['accident_risk']

# Label encode categorical columns
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

# Scale numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print(f"Shape: {X.shape}")
X.head()


# Split
X_train_std, X_val_std, y_train_std, y_val_std = train_test_split(
    X, y, test_size=0.2, random_state=42
)


## minmax scaler
X_minmax = road_accident_train_df.drop(['id', 'accident_risk'], axis=1).copy()

# Label encode
le_dict_minmax = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_minmax[col] = le.fit_transform(X_minmax[col].astype(str))
    le_dict_minmax[col] = le

# MinMax scaling
scaler_minmax = MinMaxScaler()
X_minmax[numerical_cols] = scaler_minmax.fit_transform(X_minmax[numerical_cols])


# Split
X_train_mm, X_val_mm, y_train_mm, y_val_mm = train_test_split(
    X_minmax, y, test_size=0.2, random_state=42
)





# Method 1: Correlation
print("\n--- Correlation Importance (Standard Scaler) ---")
df_corr_std = X_train_std.copy()
df_corr_std['target'] = y_train_std
corr_std = df_corr_std.corr()['target'].drop('target').abs().sort_values(ascending=False)
corr_std.head()


# Method 2: Random Forest
print("\n--- Random Forest Importance (Standard Scaler) ---")
rf_std = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_std.fit(X_train_std, y_train_std)
rf_imp_std = pd.DataFrame({
    'feature': X_train_std.columns,
    'importance': rf_std.feature_importances_
}).sort_values('importance', ascending=False)
print(rf_imp_std.head(10))


# Method 3: Permutation Importance
print("\n--- Permutation Importance (Standard Scaler) ---")
perm_std = permutation_importance(
    rf_std, X_val_std, y_val_std, 
    n_repeats=5,  # Reduced from 10
    random_state=42, 
    n_jobs=1  # Changed from -1 to avoid disk space issue
)
perm_imp_std = pd.DataFrame({
    'feature': X_train_std.columns,
    'importance': perm_std.importances_mean
}).sort_values('importance', ascending=False)
print(perm_imp_std.head(10))


# Method 1: Correlation
print("\n--- Correlation Importance (MinMax Scaler) ---")
df_corr_mm = X_train_mm.copy()
df_corr_mm['target'] = y_train_mm
corr_mm = df_corr_mm.corr()['target'].drop('target').abs().sort_values(ascending=False)
print(corr_mm.head(10))

# Method 2: Random Forest
print("\n--- Random Forest Importance (MinMax Scaler) ---")
rf_mm = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_mm.fit(X_train_mm, y_train_mm)
rf_imp_mm = pd.DataFrame({
    'feature': X_train_mm.columns,
    'importance': rf_mm.feature_importances_
}).sort_values('importance', ascending=False)
print(rf_imp_mm.head(10))

# Method 3: Permutation Importance
print("\n--- Permutation Importance (MinMax Scaler) ---")
perm_mm = permutation_importance(rf_mm, X_val_mm, y_val_mm, n_repeats=10, random_state=42, n_jobs=1)
perm_imp_mm = pd.DataFrame({
    'feature': X_train_mm.columns,
    'importance': perm_mm.importances_mean
}).sort_values('importance', ascending=False)
print(perm_imp_mm.head(10))


fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Standard Scaler plots
axes[0, 0].barh(corr_std.head(10).index, corr_std.head(10).values)
axes[0, 0].set_title('Correlation (Standard)')
axes[0, 0].invert_yaxis()

axes[0, 1].barh(rf_imp_std['feature'][:10], rf_imp_std['importance'][:10])
axes[0, 1].set_title('RF Importance (Standard)')
axes[0, 1].invert_yaxis()

axes[0, 2].barh(perm_imp_std['feature'][:10], perm_imp_std['importance'][:10])
axes[0, 2].set_title('Permutation (Standard)')
axes[0, 2].invert_yaxis()

# MinMax Scaler plots
axes[1, 0].barh(corr_mm.head(10).index, corr_mm.head(10).values)
axes[1, 0].set_title('Correlation (MinMax)')
axes[1, 0].invert_yaxis()

axes[1, 1].barh(rf_imp_mm['feature'][:10], rf_imp_mm['importance'][:10])
axes[1, 1].set_title('RF Importance (MinMax)')
axes[1, 1].invert_yaxis()

axes[1, 2].barh(perm_imp_mm['feature'][:10], perm_imp_mm['importance'][:10])
axes[1, 2].set_title('Permutation (MinMax)')
axes[1, 2].invert_yaxis()

plt.tight_layout()
plt.show()


# Create subplots
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        'Correlation (Standard)', 'RF Importance (Standard)', 'Permutation (Standard)',
        'Correlation (MinMax)', 'RF Importance (MinMax)', 'Permutation (MinMax)'
    ),
    horizontal_spacing=0.1,
    vertical_spacing=0.12
)

# Row 1: Standard Scaler
# Correlation
fig.add_trace(
    go.Bar(
        y=corr_std.head(10).index[::-1],
        x=corr_std.head(10).values[::-1],
        orientation='h',
        marker=dict(color='steelblue'),
        showlegend=False
    ),
    row=1, col=1
)

# RF Importance
fig.add_trace(
    go.Bar(
        y=rf_imp_std['feature'][:10][::-1],
        x=rf_imp_std['importance'][:10][::-1],
        orientation='h',
        marker=dict(color='darkgreen'),
        showlegend=False
    ),
    row=1, col=2
)

# Permutation
fig.add_trace(
    go.Bar(
        y=perm_imp_std['feature'][:10][::-1],
        x=perm_imp_std['importance'][:10][::-1],
        orientation='h',
        marker=dict(color='darkorange'),
        showlegend=False
    ),
    row=1, col=3
)

# Row 2: MinMax Scaler
# Correlation
fig.add_trace(
    go.Bar(
        y=corr_mm.head(10).index[::-1],
        x=corr_mm.head(10).values[::-1],
        orientation='h',
        marker=dict(color='steelblue'),
        showlegend=False
    ),
    row=2, col=1
)

# RF Importance
fig.add_trace(
    go.Bar(
        y=rf_imp_mm['feature'][:10][::-1],
        x=rf_imp_mm['importance'][:10][::-1],
        orientation='h',
        marker=dict(color='darkgreen'),
        showlegend=False
    ),
    row=2, col=2
)

# Permutation
fig.add_trace(
    go.Bar(
        y=perm_imp_mm['feature'][:10][::-1],
        x=perm_imp_mm['importance'][:10][::-1],
        orientation='h',
        marker=dict(color='darkorange'),
        showlegend=False
    ),
    row=2, col=3
)

# Update layout
fig.update_layout(
    height=800,
    width=1400,
    title_text="Feature Importance Comparison: Standard vs MinMax Scaler",
    title_font_size=20
)

# Update axes labels
for i in range(1, 7):
    row = (i-1) // 3 + 1
    col = (i-1) % 3 + 1
    fig.update_xaxes(title_text="Importance", row=row, col=col)

fig.show()





## presets

# Use Standard Scaler data from previous steps
X_train = X_train_std
X_val = X_val_std
y_train = y_train_std
y_val = y_val_std

baseline_models = {
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=2),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, n_jobs=2),
    'LightGBM': LGBMRegressor(random_state=42, n_jobs=2, verbose=-1)
}


## BaseLine Model Performance

baseline_results = {}
for name, model in baseline_models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    baseline_results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print(f"{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

baseline_df = pd.DataFrame(baseline_results).T
print("\n", baseline_df)






# ============================================
# HYPERPARAMETER TUNING - RANDOM FOREST
# ============================================
print("HYPERPARAMETER TUNING - RANDOM FOREST")

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}

rf_random = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=1),
    param_distributions=rf_param_grid,
    n_iter=10,  # Reduced from 20
    cv=2,  # Reduced from 3
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=1,  # No parallel processing
    verbose=2
)

print("Starting Random Forest tuning...")
rf_random.fit(X_train, y_train)
print(f"Best parameters: {rf_random.best_params_}")
print(f"Best CV RMSE: {-rf_random.best_score_:.4f}")

rf_best = rf_random.best_estimator_
y_pred_rf = rf_best.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, y_pred_rf))
rf_mae = mean_absolute_error(y_val, y_pred_rf)
rf_r2 = r2_score(y_val, y_pred_rf)
print(f"Validation - RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}, R2: {rf_r2:.4f}")


# ============================================
# HYPERPARAMETER TUNING - XGBOOST
# ============================================
print("\n" + "="*60)
print("HYPERPARAMETER TUNING - XGBOOST")
print("="*60)

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5]
}

xgb_random = RandomizedSearchCV(
    XGBRegressor(random_state=42, n_jobs=2),
    param_distributions=xgb_param_grid,
    n_iter=10,
    cv=2,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=1,
    verbose=1
)

print("Starting XGBoost tuning...")
xgb_random.fit(X_train, y_train)
print(f"Best parameters: {xgb_random.best_params_}")
print(f"Best CV RMSE: {-xgb_random.best_score_:.4f}")

# Evaluate
xgb_best = xgb_random.best_estimator_
y_pred_xgb = xgb_best.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
xgb_mae = mean_absolute_error(y_val, y_pred_xgb)
xgb_r2 = r2_score(y_val, y_pred_xgb)
print(f"Validation - RMSE: {xgb_rmse:.4f}, MAE: {xgb_mae:.4f}, R2: {xgb_r2:.4f}")


# ============================================
# HYPERPARAMETER TUNING - LIGHTGBM
# ============================================
print("\n" + "="*60)
print("HYPERPARAMETER TUNING - LIGHTGBM")
print("="*60)

lgbm_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 70],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_samples': [10, 20, 30]
}

lgbm_random = RandomizedSearchCV(
    LGBMRegressor(random_state=42, n_jobs=2, verbose=-1),
    param_distributions=lgbm_param_grid,
    n_iter=20,
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=1,
    verbose=1
)

print("Starting LightGBM tuning...")
lgbm_random.fit(X_train, y_train)
print(f"Best parameters: {lgbm_random.best_params_}")
print(f"Best CV RMSE: {-lgbm_random.best_score_:.4f}")

# Evaluate
lgbm_best = lgbm_random.best_estimator_
y_pred_lgbm = lgbm_best.predict(X_val)
lgbm_rmse = np.sqrt(mean_squared_error(y_val, y_pred_lgbm))
lgbm_mae = mean_absolute_error(y_val, y_pred_lgbm)
lgbm_r2 = r2_score(y_val, y_pred_lgbm)
print(f"Validation - RMSE: {lgbm_rmse:.4f}, MAE: {lgbm_mae:.4f}, R2: {lgbm_r2:.4f}")





# ============================================
# COMPARE TUNED MODELS
# ============================================
print("\n" + "="*60)
print("TUNED MODELS COMPARISON")
print("="*60)

tuned_results = {
    'Random Forest': {'RMSE': rf_rmse, 'MAE': rf_mae, 'R2': rf_r2},
    'XGBoost': {'RMSE': xgb_rmse, 'MAE': xgb_mae, 'R2': xgb_r2},
    'LightGBM': {'RMSE': lgbm_rmse, 'MAE': lgbm_mae, 'R2': lgbm_r2}
}

tuned_df = pd.DataFrame(tuned_results).T
print(tuned_df)


# ============================================
# METHOD 1: WEIGHTED AVERAGE (BASED ON VALIDATION PERFORMANCE)
# ============================================
print("\n" + "="*60)
print("METHOD 1: WEIGHTED ENSEMBLE (Performance-Based)")
print("="*60)

# Calculate weights based on inverse RMSE (better models get higher weight)
weight_rf = 1 / rf_rmse
weight_xgb = 1 / xgb_rmse
weight_lgbm = 1 / lgbm_rmse

# Normalize weights to sum to 1
total_weight = weight_rf + weight_xgb + weight_lgbm
weight_rf = weight_rf / total_weight
weight_xgb = weight_xgb / total_weight
weight_lgbm = weight_lgbm / total_weight

print(f"Weights - RF: {weight_rf:.3f}, XGB: {weight_xgb:.3f}, LGBM: {weight_lgbm:.3f}")

# Validation predictions
y_pred_weighted = (weight_rf * y_pred_rf + 
                   weight_xgb * y_pred_xgb + 
                   weight_lgbm * y_pred_lgbm)

weighted_rmse = np.sqrt(mean_squared_error(y_val, y_pred_weighted))
weighted_r2 = r2_score(y_val, y_pred_weighted)
print(f"Weighted Ensemble - RMSE: {weighted_rmse:.4f}, R2: {weighted_r2:.4f}")


# ============================================
# METHOD 2: OPTIMIZED WEIGHTS (GRID SEARCH)
# ============================================
print("\n" + "="*60)
print("METHOD 2: OPTIMIZED WEIGHTS (Grid Search)")
print("="*60)

best_rmse = float('inf')
best_weights = None

# Try different weight combinations
for w1 in np.arange(0.1, 0.8, 0.1):
    for w2 in np.arange(0.1, 0.8, 0.1):
        w3 = 1.0 - w1 - w2
        if w3 >= 0.1 and w3 <= 0.8:
            y_pred_test = w1 * y_pred_rf + w2 * y_pred_xgb + w3 * y_pred_lgbm
            rmse = np.sqrt(mean_squared_error(y_val, y_pred_test))
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = (w1, w2, w3)

print(f"Best weights - RF: {best_weights[0]:.3f}, XGB: {best_weights[1]:.3f}, LGBM: {best_weights[2]:.3f}")
print(f"Optimized Ensemble - RMSE: {best_rmse:.4f}")

y_pred_optimized = (best_weights[0] * y_pred_rf + 
                    best_weights[1] * y_pred_xgb + 
                    best_weights[2] * y_pred_lgbm)


# ============================================
# METHOD 3: STACKING (META-LEARNER)
# ============================================
print("\n" + "="*60)
print("METHOD 3: STACKING WITH META-LEARNER")
print("="*60)

# Create meta-features from base model predictions
meta_features_train = np.column_stack([y_pred_rf, y_pred_xgb, y_pred_lgbm])

# Train meta-learner (Ridge regression)
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_features_train, y_val)

# Get stacked predictions
y_pred_stacked = meta_model.predict(meta_features_train)

stacked_rmse = np.sqrt(mean_squared_error(y_val, y_pred_stacked))
stacked_r2 = r2_score(y_val, y_pred_stacked)

print(f"Meta-learner weights: {meta_model.coef_}")
print(f"Stacking Ensemble - RMSE: {stacked_rmse:.4f}, R2: {stacked_r2:.4f}")


# ============================================
# COMPARE ALL ENSEMBLE METHODS
# ============================================
print("\n" + "="*60)
print("ENSEMBLE COMPARISON")
print("="*60)

ensemble_comparison = {
    'Simple Average': ensemble_rmse,
    'Weighted (Performance)': weighted_rmse,
    'Optimized Weights': best_rmse,
    'Stacking': stacked_rmse
}

for method, rmse in sorted(ensemble_comparison.items(), key=lambda x: x[1]):
    print(f"{method:25s}: RMSE = {rmse:.5f}")





## reading the test data
road_accident_test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
road_accident_test_df.head()


## preprocessing and basic EDA
X_test = road_accident_test_df.drop(['id'], axis=1).copy()

# Apply encoding and scaling - FIXED VARIABLE NAMES
for col in categorical_cols:
    X_test[col] = le_dict[col].transform(X_test[col].astype(str))  # Changed from le_dict_standard

X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])  # Changed from scaler_standard


# Get predictions from all models
test_pred_rf = rf_best.predict(X_test)
test_pred_xgb = xgb_best.predict(X_test)
test_pred_lgbm = lgbm_best.predict(X_test)


stacked_rmse


# Apply BEST ensemble method (optimized weights)
test_pred_final = (best_weights[0] * test_pred_rf + 
                   best_weights[1] * test_pred_xgb + 
                   best_weights[2] * test_pred_lgbm)


# Alternative: Stacking method
meta_features_test = np.column_stack([test_pred_rf, test_pred_xgb, test_pred_lgbm])
test_pred_stacked = meta_model.predict(meta_features_test)


# Clip predictions to valid range [0, 1]
test_pred_final_clipped = np.clip(test_pred_final, 0, 1)
test_pred_stacked_clipped = np.clip(test_pred_stacked, 0, 1)


# Create submission with optimized weights
submission_optimized = pd.DataFrame({
    'id': road_accident_test_df['id'],
    'accident_risk': test_pred_final_clipped
})
submission_optimized.to_csv('submission_optimized_clipped.csv', index=False)
print("Clipped submissions created!")
print(f"\nOptimized - Min: {test_pred_final_clipped.min():.4f}, Max: {test_pred_final_clipped.max():.4f}")


# Create submission with stacking weights
submission_stacking = pd.DataFrame({
    'id': road_accident_test_df['id'],
    'accident_risk': test_pred_stacked_clipped
})
submission_stacking.to_csv('submission_stacking_clipped.csv', index=False)
print(f"Stacking - Min: {test_pred_stacked_clipped.min():.4f}, Max: {test_pred_stacked_clipped.max():.4f}")




