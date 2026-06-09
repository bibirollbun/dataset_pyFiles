import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Modeling libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load train and test datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print("=== DATASET SHAPES ===")
print(f"Train Data Shape: {train_df.shape}")
print(f"Test Data Shape: {test_df.shape}")

# Display first few rows
train_df.head()


print("=== DATA TYPES & NULL VALUES ===")
print(train_df.info())

print("\n=== NULL VALUES CHECK ===")
print(train_df.isnull().sum())

print("\n=== BASIC STATISTICS ===")
train_df.describe()


plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(train_df['accident_risk'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Distribution of Accident Risk', fontsize=14, fontweight='bold')
plt.xlabel('Accident Risk', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Box plot
plt.subplot(1, 2, 2)
plt.boxplot(train_df['accident_risk'], vert=True)
plt.title('Accident Risk - Box Plot', fontsize=14, fontweight='bold')
plt.ylabel('Accident Risk', fontsize=12)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


numeric_cols = ['num_lanes','curvature','speed_limit','num_reported_accidents','accident_risk']
categorical_cols = ['road_type','lighting','weather','time_of_day']
boolean_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']


# Correlation Matrix
correlation_matrix = train_df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Correlation Coefficient'})

plt.title('Correlation Matrix - Numerical Features', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()


# Categorical Field Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, col in enumerate(categorical_cols):
    train_df[col].value_counts().plot(kind='bar', ax=axes[idx], color='steelblue')
    axes[idx].set_title(f'{col.upper()} Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('Count')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))

sns.boxplot(data=train_df, x='num_reported_accidents', y='accident_risk', palette='Blues')
plt.title('Distribution of Accident Risk by Number of Reported Accidents', 
          fontsize=14, fontweight='bold')
plt.xlabel('Number of Reported Accidents', fontsize=12)
plt.ylabel('Accident Risk', fontsize=12)

plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


X = train_df.drop(['id', 'accident_risk'], axis=1)
y = train_df['accident_risk']

label_encoders = {}

X_encoded = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"âœ… Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

for col in boolean_cols:
    X_encoded[col] = X_encoded[col].astype(int)

print(f"\nâœ… Preprocessing complete!")
print(f"Feature shape: {X_encoded.shape}")


X_fe = X_encoded.copy()

# 1. Curvature Ã— Speed (high-risk combination)
X_fe['curvature_x_speed'] = X_fe['curvature'] * X_fe['speed_limit']

# 2. Lighting Ã— Weather (poor visibility)
X_fe['lighting_x_weather'] = X_fe['lighting'] * X_fe['weather']

# 3. Curvature Ã— Lighting (curved + dark)
X_fe['curvature_x_lighting'] = X_fe['curvature'] * X_fe['lighting']

# 4. Speed Ã— Weather (high speed + bad weather)
X_fe['speed_x_weather'] = X_fe['speed_limit'] * X_fe['weather']

# 5. High risk combo flag
X_fe['high_risk_combo'] = (
    (X_fe['curvature'] > 0.5).astype(int) * 
    (X_fe['speed_limit'] > 50).astype(int) * 
    (X_fe['lighting'] == 1).astype(int)  # Assuming 1 = night
)

# 6. Polynomial features
X_fe['curvature_squared'] = X_fe['curvature'] ** 2
X_fe['speed_squared'] = X_fe['speed_limit'] ** 2

print("=== FEATURE ENGINEERING COMPLETE ===")
print(f"Original features: {X_encoded.shape[1]}")
print(f"New features: {X_fe.shape[1]}")
print(f"Added: {X_fe.shape[1] - X_encoded.shape[1]} features")

print("\nğŸ“� New features created:")
new_features = [col for col in X_fe.columns if col not in X_encoded.columns]
for feat in new_features:
    print(f"  âœ… {feat}")


X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

X_train_fe, X_val_fe, y_train_fe, y_val_fe = train_test_split(
    X_fe, y, test_size=0.2, random_state=42
)

print("=== DATA SPLIT COMPLETE ===")
print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Training set (FE): {X_train_fe.shape}")
print(f"Validation set (FE): {X_val_fe.shape}")


rf_baseline = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("Training baseline Random Forest...")
rf_baseline.fit(X_train, y_train)

# Predictions
y_train_pred = rf_baseline.predict(X_train)
y_val_pred = rf_baseline.predict(X_val)

# Evaluation
print("\n=== BASELINE MODEL PERFORMANCE ===\n")
print("TRAINING SET:")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.6f}")
print(f"  MAE:  {mean_absolute_error(y_train, y_train_pred):.6f}")
print(f"  RÂ²:   {r2_score(y_train, y_train_pred):.6f}")

print("\nVALIDATION SET:")
baseline_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
baseline_mae = mean_absolute_error(y_val, y_val_pred)
baseline_r2 = r2_score(y_val, y_val_pred)

print(f"  RMSE: {baseline_rmse:.6f}")
print(f"  MAE:  {baseline_mae:.6f}")
print(f"  RÂ²:   {baseline_r2:.6f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': rf_baseline.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== TOP 10 FEATURES ===")
print(feature_importance.head(10).to_string(index=False))


rf_fe = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("Training Random Forest with engineered features...")
rf_fe.fit(X_train_fe, y_train_fe)

# Predictions
y_val_pred_fe = rf_fe.predict(X_val_fe)

# Evaluation
rf_fe_rmse = np.sqrt(mean_squared_error(y_val_fe, y_val_pred_fe))
rf_fe_r2 = r2_score(y_val_fe, y_val_pred_fe)

print("\n=== RF WITH FEATURE ENGINEERING ===")
print(f"  RMSE: {rf_fe_rmse:.6f}")
print(f"  RÂ²:   {rf_fe_r2:.6f}")

# Feature importance
feature_importance_fe = pd.DataFrame({
    'feature': X_fe.columns,
    'importance': rf_fe.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== TOP 10 FEATURES (with FE) ===")
print(feature_importance_fe.head(10).to_string(index=False))


X_catboost = train_df.drop(['id', 'accident_risk'], axis=1)
y_catboost = train_df['accident_risk']

cat_features = ['road_type', 'lighting', 'weather', 'time_of_day']

X_train_cat, X_val_cat, y_train_cat, y_val_cat = train_test_split(
    X_catboost, y_catboost, test_size=0.2, random_state=42
)

catboost_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=8,
    loss_function='RMSE',
    cat_features=cat_features,
    random_state=42,
    verbose=False
)

catboost_model.fit(X_train_cat, y_train_cat)

y_val_pred_cat = catboost_model.predict(X_val_cat)

cat_rmse = np.sqrt(mean_squared_error(y_val_cat, y_val_pred_cat))
cat_r2 = r2_score(y_val_cat, y_val_pred_cat)

print("\n=== CATBOOST PERFORMANCE ===")
print(f"  RMSE: {cat_rmse:.6f}")
print(f"  RÂ²:   {cat_r2:.6f}")


lgbm_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

lgbm_model.fit(X_train_fe, y_train_fe)

y_val_pred_lgbm = lgbm_model.predict(X_val_fe)

lgbm_rmse = np.sqrt(mean_squared_error(y_val_fe, y_val_pred_lgbm))
lgbm_r2 = r2_score(y_val_fe, y_val_pred_lgbm)

print("\n=== LIGHTGBM PERFORMANCE ===")
print(f"  RMSE: {lgbm_rmse:.6f}")
print(f"  RÂ²:   {lgbm_r2:.6f}")


results = pd.DataFrame({
    'Model': ['Baseline RF', 'RF + Feature Eng', 'CatBoost', 'LightGBM'],
    'RMSE': [baseline_rmse, rf_fe_rmse, cat_rmse, lgbm_rmse],
    'RÂ²': [baseline_r2, rf_fe_r2, cat_r2, lgbm_r2]
}).sort_values('RMSE')

print("=== MODEL COMPARISON ===\n")
print(results.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(results['Model'], results['RMSE'], color='coral')
axes[0].set_xlabel('RMSE (lower is better)', fontweight='bold')
axes[0].set_title('Model Comparison - RMSE', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

axes[1].barh(results['Model'], results['RÂ²'], color='lightgreen')
axes[1].set_xlabel('RÂ² Score (higher is better)', fontweight='bold')
axes[1].set_title('Model Comparison - RÂ²', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()


print("=== TESTING ENSEMBLE COMBINATIONS ===\n")

weight_configs = [
    (0.3, 0.5, 0.2, 'RF 30% + Cat 50% + LGBM 20%'),
    (0.2, 0.5, 0.3, 'RF 20% + Cat 50% + LGBM 30%'),
    (0.25, 0.5, 0.25, 'RF 25% + Cat 50% + LGBM 25%'),
    (0.3, 0.4, 0.3, 'RF 30% + Cat 40% + LGBM 30%'),
]

ensemble_results = []

for w_rf, w_cat, w_lgbm, label in weight_configs:
    y_val_ensemble = (w_rf * y_val_pred_fe + 
                     w_cat * y_val_pred_cat + 
                     w_lgbm * y_val_pred_lgbm)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_val_ensemble))
    r2 = r2_score(y_val, y_val_ensemble)
    
    ensemble_results.append({
        'Configuration': label,
        'RMSE': rmse,
        'RÂ²': r2
    })
    
    print(f"{label}")
    print(f"  RMSE: {rmse:.6f}, RÂ²: {r2:.6f}\n")

# Find best
ensemble_df = pd.DataFrame(ensemble_results).sort_values('RMSE')
print("\nğŸ�† BEST ENSEMBLE:")
print(ensemble_df.iloc[0].to_string())

best_ensemble_rmse = ensemble_df.iloc[0]['RMSE']


all_results = pd.DataFrame({
    'Model': ['Baseline RF', 'RF + FE', 'CatBoost', 'LightGBM', 'Ensemble (Best)'],
    'RMSE': [baseline_rmse, rf_fe_rmse, cat_rmse, lgbm_rmse, best_ensemble_rmse]
}).sort_values('RMSE')

plt.figure(figsize=(12, 6))
colors = ['lightcoral', 'lightsalmon', 'lightblue', 'lightgreen', 'gold']
bars = plt.barh(all_results['Model'], all_results['RMSE'], color=colors)

# Highlight best model
bars[-1].set_color('darkgreen')
bars[-1].set_edgecolor('black')
bars[-1].set_linewidth(2)

plt.xlabel('RMSE', fontsize=12, fontweight='bold')
plt.title('Final Model Comparison - All Models', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(all_results.iterrows()):
    plt.text(row['RMSE'] + 0.00005, i, f"{row['RMSE']:.6f}", 
             va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Calculate improvement
improvement = ((baseline_rmse - best_ensemble_rmse) / baseline_rmse) * 100
print(f"\nğŸ“ˆ TOTAL IMPROVEMENT: {improvement:.2f}% from baseline")


X_test = test_df.drop(['id'], axis=1)
X_test_encoded = X_test.copy()

for col in categorical_cols:
    X_test_encoded[col] = label_encoders[col].transform(X_test[col])

for col in boolean_cols:
    X_test_encoded[col] = X_test_encoded[col].astype(int)

X_test_fe = X_test_encoded.copy()
X_test_fe['curvature_x_speed'] = X_test_fe['curvature'] * X_test_fe['speed_limit']
X_test_fe['lighting_x_weather'] = X_test_fe['lighting'] * X_test_fe['weather']
X_test_fe['curvature_x_lighting'] = X_test_fe['curvature'] * X_test_fe['lighting']
X_test_fe['speed_x_weather'] = X_test_fe['speed_limit'] * X_test_fe['weather']
X_test_fe['high_risk_combo'] = (
    (X_test_fe['curvature'] > 0.5).astype(int) * 
    (X_test_fe['speed_limit'] > 50).astype(int) * 
    (X_test_fe['lighting'] == 1).astype(int)
)
X_test_fe['curvature_squared'] = X_test_fe['curvature'] ** 2
X_test_fe['speed_squared'] = X_test_fe['speed_limit'] ** 2

X_test_cat = test_df.drop(['id'], axis=1)

print(f"âœ… Test data prepared!")
print(f"RF/LGBM input shape: {X_test_fe.shape}")
print(f"CatBoost input shape: {X_test_cat.shape}")


# RF predictions
print("Random Forest predicting...")
test_pred_rf = rf_fe.predict(X_test_fe)

# CatBoost predictions
print("CatBoost predicting...")
test_pred_cat = catboost_model.predict(X_test_cat)

# LightGBM predictions
print("LightGBM predicting...")
test_pred_lgbm = lgbm_model.predict(X_test_fe)

# Ensemble with best weights (30%, 50%, 20%)
final_predictions = (0.3 * test_pred_rf + 
                    0.5 * test_pred_cat + 
                    0.2 * test_pred_lgbm)

print("\n Predictions complete!")


submission = pd.DataFrame({
    'id': test_df['id'],
    'accident_risk': final_predictions
})

print("=== SUBMISSION STATISTICS ===")
print(f"Total predictions: {len(submission)}")
print(f"Prediction range: [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")
print(f"Prediction mean: {final_predictions.mean():.4f}")
print(f"Prediction std: {final_predictions.std():.4f}")

print("\n=== SAMPLE PREDICTIONS ===")
print(submission.head(10))

# Visualize prediction distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(final_predictions, bins=50, edgecolor='black', alpha=0.7, color='green')
plt.axvline(final_predictions.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {final_predictions.mean():.4f}')
plt.xlabel('Predicted Accident Risk', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Test Predictions Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(final_predictions, vert=True)
plt.ylabel('Predicted Accident Risk', fontsize=12)
plt.title('Test Predictions - Box Plot', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Save submission
submission.to_csv('submission.csv', index=False)
print("\nâœ… Submission file saved as 'submission.csv'")




