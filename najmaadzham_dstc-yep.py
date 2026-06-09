# 1 - library
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

train = pd.read_csv("/kaggle/input/dstc-kaggle-practice-2025/train.csv")
test = pd.read_csv("/kaggle/input/dstc-kaggle-practice-2025/test.csv")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print(f"Train: {train.shape}, Test: {test.shape}")
print(train.columns.tolist())

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 2 - Melihat 5 baris pertama data training untuk memahami struktur data
train.head()


# 3 - Mengecek informasi struktur data dan missing values
train.info()


# 4 - Melihat statistik deskriptif untuk kolom numerik
train.describe()


# 5 - Menghitung total missing values di setiap kolom
train.isnull().sum()


# 6 - Mengecek dimensi dataset dan daftar kolom yang tersedia
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain columns:", train.columns.tolist())


# 7 - Analisis komprehensif struktur data dan target variable

print("=== Data Types ===")
print(train.info())

print("\n=== Target Variable (accident_risk) ===")
print(train['accident_risk'].describe())

print("\n=== Cek missing values ===")
print("Train missing values:")
print(train.isnull().sum())
print("\nTest missing values:")
print(test.isnull().sum())


# 8 - Visualisasi distribusi variabel target dan fitur terkait kecelakaan
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(train['accident_risk'], kde=True, bins=50)
plt.title('Distribution of Accident Risk')

plt.subplot(1, 3, 2)
sns.boxplot(y=train['accident_risk'])
plt.title('Boxplot of Accident Risk')

plt.subplot(1, 3, 3)
sns.histplot(train['num_reported_accidents'], kde=True, bins=30)
plt.title('Distribution of Reported Accidents')

plt.tight_layout()
plt.show()


# 9 - Analisis fitur kategorikal dan korelasi fitur numerik dengan target
print("=== Categorical Features ===")
categorical_features = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']

for col in categorical_features:
    print(f"\n{col}:")
    print(train[col].value_counts().head())

print("\n" + "="*50)
print("=== NUMERICAL FEATURES CORRELATION ===")
print("="*50)

numeric_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents', 'accident_risk']
corr_with_target = train[numeric_features].corr()['accident_risk'].sort_values(ascending=False)

print("Correlation with accident_risk:")
for feature, corr in corr_with_target.items():
    print(f"{feature:25} : {corr:+.4f}")

plt.figure(figsize=(8, 6))
sns.heatmap(train[numeric_features].corr(), 
            annot=True, cmap='coolwarm', center=0, fmt='.3f',
            square=True, linewidths=0.5)
plt.title('Correlation Matrix', fontsize=14, pad=20)
plt.tight_layout()
plt.show()


# 10 - Preprocessing 
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=== PREPROCESSING ===")

# Prepare features
X = train.drop(['id', 'accident_risk'], axis=1)
y = train['accident_risk']
X_test = test.drop(['id'], axis=1)

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X_test shape: {X_test.shape}")

# One-hot encoding 
categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']
binary_features = ['road_signs_present', 'public_road', 'holiday', 'school_season']

print("\nApplying one-hot encoding...")
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

X_encoded, X_test_encoded = X_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

print(f"After encoding - X shape: {X_encoded.shape}, X_test shape: {X_test_encoded.shape}")
print(f"Features: {X_encoded.columns.tolist()}")


# 11 - BASELINE MODELING FRAMEWORK

from sklearn.metrics import mean_absolute_error, r2_score

print("=== BASELINE MODELING FRAMEWORK ===")
print("Initializing model evaluation protocol...\n")

# Professional data splitting with stratification
def create_stratified_bins(y, n_bins=10):
    """Create stratified bins for continuous target"""
    return pd.cut(y, bins=n_bins, labels=False)

strat_bins = create_stratified_bins(y)
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, 
    stratify=strat_bins, shuffle=True
)

print("Dataset Partitioning:")
print(f"   Training Set   : {X_train.shape}")
print(f"   Validation Set : {X_val.shape}")
print(f"   Stratification : Applied (10 bins)")

# Initialize model registry with robust parameters
models = {
    'XGBoost': XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='rmse'
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        verbose=0
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
}

print("\nModel Configuration Complete")
print("   Models: XGBoost, Random Forest, Gradient Boosting")
print("   Ensemble Methods: Bagging, Boosting")

# Comprehensive evaluation function
def evaluate_model(model, X_train, X_val, y_train, y_val, X_full, y_full, name):
    """Comprehensive model evaluation with cross-validation"""
    
    # Training phase
    train_start = pd.Timestamp.now()
    model.fit(X_train, y_train)
    train_time = (pd.Timestamp.now() - train_start).total_seconds()
    
    # Prediction phase
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Calculate metrics
    metrics = {
        'RMSE_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'RMSE_val': np.sqrt(mean_squared_error(y_val, y_pred_val)),
        'MAE_val': mean_absolute_error(y_val, y_pred_val),
        'R2_val': r2_score(y_val, y_pred_val),
        'Train_Time': train_time
    }
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_full, y_full, 
                               cv=5, scoring='neg_mean_squared_error',
                               n_jobs=-1)
    metrics['CV_RMSE_mean'] = np.sqrt(-cv_scores.mean())
    metrics['CV_RMSE_std'] = np.sqrt(cv_scores.std())
    
    return metrics, y_pred_val

# Execute model evaluation
print("\nInitiating Model Training & Evaluation...")
print("=" * 80)

results = {}
predictions = {}

for name, model in models.items():
    print(f"Evaluating {name}")
    
    metrics, y_pred = evaluate_model(model, X_train, X_val, y_train, y_val, X_encoded, y, name)
    results[name] = metrics
    predictions[name] = y_pred
    
    print("COMPLETED")
    print(f"   RMSE (Val)  : {metrics['RMSE_val']:.4f}")
    print(f"   RMSE (Train): {metrics['RMSE_train']:.4f}")
    print(f"   CV RMSE     : {metrics['CV_RMSE_mean']:.4f} +- {metrics['CV_RMSE_std']:.4f}")
    print(f"   R2 Score    : {metrics['R2_val']:.4f}")
    print(f"   Training Time: {metrics['Train_Time']:.2f}s")

# Performance comparison
print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)

performance_df = pd.DataFrame(results).T
performance_df['Overfitting_Gap'] = performance_df['RMSE_train'] - performance_df['RMSE_val']
performance_df = performance_df.sort_values('RMSE_val')

print(performance_df.round(4))

# Identify best model
best_model_name = performance_df.index[0]
best_model = models[best_model_name]
best_predictions = predictions[best_model_name]

print(f"\nRECOMMENDED MODEL: {best_model_name}")
print(f"   Validation RMSE: {performance_df.loc[best_model_name, 'RMSE_val']:.4f}")
print(f"   Cross-Val RMSE : {performance_df.loc[best_model_name, 'CV_RMSE_mean']:.4f}")
print(f"   R2 Score       : {performance_df.loc[best_model_name, 'R2_val']:.4f}")

# Feature importance analysis
print(f"\nFeature Importance Analysis - {best_model_name}")
print("-" * 50)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Display top features
    print("Top 10 Most Predictive Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {i+1:2d}. {row['feature']:30} : {row['importance']:.4f}")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance.head(15), x='importance', y='feature', palette='viridis')
    plt.title(f'Feature Importance - {best_model_name}\n(Validation RMSE: {performance_df.loc[best_model_name, "RMSE_val"]:.4f})', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Feature Importance Score', fontsize=12)
    plt.tight_layout()
    plt.show()

print("\nBaseline Modeling Complete. Ready for Hyperparameter Optimization.")


# 12 - MODEL DIAGNOSTICS & RESIDUAL ANALYSIS
print("=== MODEL DIAGNOSTICS AND RESIDUAL ANALYSIS ===")

# Get predictions from best model (XGBoost)
y_pred = best_model.predict(X_val)
residuals = y_val - y_pred

# Diagnostic plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Residuals vs Predicted
axes[0,0].scatter(y_pred, residuals, alpha=0.5)
axes[0,0].axhline(y=0, color='red', linestyle='--')
axes[0,0].set_xlabel('Predicted Values')
axes[0,0].set_ylabel('Residuals')
axes[0,0].set_title('Residuals vs Predicted')

# 2. Q-Q Plot for normality
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[0,1])
axes[0,1].set_title('Q-Q Plot (Normality Check)')

# 3. Actual vs Predicted
axes[0,2].scatter(y_val, y_pred, alpha=0.5)
axes[0,2].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'red', lw=2)
axes[0,2].set_xlabel('Actual')
axes[0,2].set_ylabel('Predicted')
axes[0,2].set_title('Actual vs Predicted')

# 4. Residual distribution
axes[1,0].hist(residuals, bins=50, edgecolor='black')
axes[1,0].set_xlabel('Residuals')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('Residual Distribution')

# 5. Prediction error distribution
prediction_error = np.abs(residuals)
axes[1,1].hist(prediction_error, bins=50, edgecolor='black', color='orange')
axes[1,1].set_xlabel('Absolute Prediction Error')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Prediction Error Distribution')

# 6. Feature importance (already done)
axes[1,2].axis('off')
axes[1,2].text(0.1, 0.5, f"Best Model: XGBoost\nValidation RMSE: {0.0563:.4f}\nR² Score: {0.8860:.4f}", 
               fontsize=12, transform=axes[1,2].transAxes)

plt.tight_layout()
plt.show()

# Statistical diagnostics
print("\nRESIDUAL ANALYSIS:")
print(f"Mean of residuals: {residuals.mean():.6f}")
print(f"Std of residuals: {residuals.std():.6f}")
print(f"Residual skewness: {stats.skew(residuals):.4f}")
print(f"Residual kurtosis: {stats.kurtosis(residuals):.4f}")

# Check for patterns in large errors
large_errors = np.abs(residuals) > residuals.std() * 2
print(f"Large errors (>2σ): {large_errors.sum()} ({large_errors.mean()*100:.1f}%)")


# 13 - HYPERPARAMETER TUNING
print("=== HYPERPARAMETER OPTIMIZATION ===")

# Since model is well-behaved, we can proceed with tuning
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [1, 1.5, 2.0]
}

xgb = XGBRegressor(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(
    xgb, param_dist, n_iter=50, scoring='neg_mean_squared_error',
    cv=5, random_state=42, n_jobs=-1, verbose=1
)

print("Starting RandomizedSearchCV...")
random_search.fit(X_encoded, y)

print(f"Best RMSE: {np.sqrt(-random_search.best_score_):.4f}")
print("Best parameters:", random_search.best_params_)


# 14 - STRATEGIC OPTIMIZATION
print("=== STRATEGIC OPTIMIZATION ===")

# 1. BUAT FEATURE ENGINEERING DULU SEBELUM TRAINING
X_advanced = X_encoded.copy()
X_test_advanced = X_test_encoded.copy()

# Tambahkan feature engineering
X_advanced['night_fog_risk'] = X_encoded['lighting_night'] * X_encoded['weather_foggy']
X_advanced['curve_speed_risk'] = X_encoded['curvature'] * X_encoded['speed_limit']

X_test_advanced['night_fog_risk'] = X_test_encoded['lighting_night'] * X_test_encoded['weather_foggy'] 
X_test_advanced['curve_speed_risk'] = X_test_encoded['curvature'] * X_test_encoded['speed_limit']

print(f"Feature shapes - Advanced: {X_advanced.shape}, Test Advanced: {X_test_advanced.shape}")

# 2. TRAIN MODEL DENGAN FEATURE SET YANG SESUAI

# Model dengan original features
model1 = XGBRegressor(**random_search.best_params_)
model1.fit(X_encoded, y)  # Train dengan original features

# Model dengan advanced features  
model_advanced = XGBRegressor(**random_search.best_params_)
model_advanced.fit(X_advanced, y)  # Train dengan advanced features

# Model alternatif
model2 = XGBRegressor(
    n_estimators=1000, 
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model2.fit(X_encoded, y)

# 3. PREDICT DENGAN FEATURE SET YANG SESUAI
pred1 = model1.predict(X_test_encoded)           # Original features
pred_advanced = model_advanced.predict(X_test_advanced)  # Advanced features  
pred2 = model2.predict(X_test_encoded)           # Original features

print("Predictions completed successfully!")

# 4. BLENDING
weights = [0.4, 0.4, 0.2]  # Adjust these weights
final_pred = (weights[0] * pred1 + 
              weights[1] * pred_advanced + 
              weights[2] * pred2)

# Ensure reasonable range
final_pred = np.clip(final_pred, 0, 1)

# 5. CREATE SUBMISSION
submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': final_pred
})

submission.to_csv('submission_optimized.csv', index=False)

print("Optimized submission created!")
print(f"Final prediction range: [{final_pred.min():.6f}, {final_pred.max():.6f}]")


# 15 - booster

X_ultimate = X_encoded.copy()
X_test_ultimate = X_test_encoded.copy()

X_ultimate['extreme_risk'] = (
    X_encoded['lighting_night'] * 
    X_encoded['weather_foggy'] * 
    X_encoded['curvature'] * 
    (X_encoded['speed_limit'] / 100)
)

X_ultimate['accident_intensity'] = X_encoded['num_reported_accidents'] * X_encoded['curvature']
X_ultimate['night_speed_risk'] = X_encoded['lighting_night'] * X_encoded['speed_limit']

X_test_ultimate['extreme_risk'] = (
    X_test_encoded['lighting_night'] * 
    X_test_encoded['weather_foggy'] * 
    X_test_encoded['curvature'] * 
    (X_test_encoded['speed_limit'] / 100)
)
X_test_ultimate['accident_intensity'] = X_test_encoded['num_reported_accidents'] * X_test_encoded['curvature']
X_test_ultimate['night_speed_risk'] = X_test_encoded['lighting_night'] * X_test_encoded['speed_limit']

model_ult = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42
)

model_ult.fit(X_ultimate, y)
pred_ult = model_ult.predict(X_test_ultimate)

prev_best = pd.read_csv('submission_optimized.csv')['accident_risk'].values

final_ultimate = 0.3 * prev_best + 0.7 * pred_ult
final_ultimate = np.clip(final_ultimate, 0, 1)

pd.DataFrame({
    'id': test['id'],
    'accident_risk': final_ultimate
}).to_csv('submission10.csv', index=False)


# 16

X_advanced = X_encoded.copy()
X_test_advanced = X_test_encoded.copy()

# High-impact features only
X_advanced['night_fog_risk'] = X_encoded['lighting_night'] * X_encoded['weather_foggy']
X_advanced['curve_speed_risk'] = X_encoded['curvature'] * X_encoded['speed_limit']
X_advanced['accident_density'] = X_encoded['num_reported_accidents'] / (X_encoded['num_lanes'] + 1)

X_test_advanced['night_fog_risk'] = X_test_encoded['lighting_night'] * X_test_encoded['weather_foggy']
X_test_advanced['curve_speed_risk'] = X_test_encoded['curvature'] * X_test_encoded['speed_limit']
X_test_advanced['accident_density'] = X_test_encoded['num_reported_accidents'] / (X_test_encoded['num_lanes'] + 1)

print(f"Advanced features: {X_advanced.shape}")

best_params = {
    'n_estimators': 300,
    'max_depth': 10,
    'learning_rate': 0.05,
    'subsample': 1.0,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 1.5,
    'random_state': 42
}

model_advanced = XGBRegressor(**best_params)
model_advanced.fit(X_advanced, y)

model_original = XGBRegressor(**best_params)
model_original.fit(X_encoded, y)

# Predictions
pred_advanced = model_advanced.predict(X_test_advanced)
pred_original = model_original.predict(X_test_encoded)

final_pred = 0.7 * pred_original + 0.3 * pred_advanced
final_pred = np.clip(final_pred, 0, 1)

submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': final_pred
})

submission.to_csv('submission.csv', index=False)

print(f"Prediction range: [{final_pred.min():.6f}, {final_pred.max():.6f}]")

X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
model_original.fit(X_train, y_train)
val_pred = model_original.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))

