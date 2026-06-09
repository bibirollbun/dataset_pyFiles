# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)


# Load data
train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv(r'/kaggle/input/playground-series-s5e9/test.csv')
sample_submission = pd.read_csv(r'/kaggle/input/bpm-prediction-challenge/Train.csv')

# Display info
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nTrain columns:", train_df.columns.tolist())
print("Test columns:", test_df.columns.tolist())
print("\nTrain info:")
print(train_df.info())
print("\nFirst few rows of train:")
print(train_df.head())

# Target stats
if 'BeatsPerMinute' in train_df.columns:
    print("\nTarget statistics:")
    print(train_df['BeatsPerMinute'].describe())
else:
    print("\nError: 'BeatsPerMinute' not found. Check columns above.")


# Robust ID detection
def find_id_column(df):
    # Check for 'ID', 'id', or columns containing 'id'
    for col in df.columns:
        if col.lower() == 'id' or 'id' in col.lower():
            return col
    # Fallback: look for unique integer column
    for col in df.columns:
        if df[col].dtype in [np.int32, np.int64] and df[col].nunique() == len(df):
            return col
    return None

id_col = find_id_column(test_df)
target_col = 'BeatsPerMinute'

if id_col is None:
    print("Error: No ID column found in test data. Columns:", test_df.columns.tolist())
    raise SystemExit
if target_col not in train_df.columns:
    print(f"Error: Target '{target_col}' not found. Columns: {train_df.columns.tolist()}")
    raise SystemExit

print(f"Detected ID column: {id_col}")

# Select features (all except ID and target)
exclude_cols = [id_col, target_col] if id_col else [target_col]
features = [col for col in train_df.columns if col not in exclude_cols]
print(f"Detected features: {features}")

# Prepare X and y
X = train_df[features].copy()
y = train_df[target_col]
X_test = test_df[features].copy()

# Feature engineering based on actual columns
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

# Music-specific feature engineering with actual column names
# Energy/Rhythm ratio (Energy over RhythmScore)
if 'Energy' in X.columns and 'RhythmScore' in X.columns:
    X['Energy_over_RhythmScore'] = X['Energy'] / (X['RhythmScore'] + 1e-6)
    X_test['Energy_over_RhythmScore'] = X_test['Energy'] / (X_test['RhythmScore'] + 1e-6)
    features.append('Energy_over_RhythmScore')

# Energy * MoodScore interaction
if 'Energy' in X.columns and 'MoodScore' in X.columns:
    X['Energy_x_MoodScore'] = X['Energy'] * X['MoodScore']
    X_test['Energy_x_MoodScore'] = X_test['Energy'] * X_test['MoodScore']
    features.append('Energy_x_MoodScore')

# RhythmScore * AcousticQuality interaction
if 'RhythmScore' in X.columns and 'AcousticQuality' in X.columns:
    X['RhythmScore_x_AcousticQuality'] = X['RhythmScore'] * X['AcousticQuality']
    X_test['RhythmScore_x_AcousticQuality'] = X_test['RhythmScore'] * X_test['AcousticQuality']
    features.append('RhythmScore_x_AcousticQuality')

# Log transforms for skewed features
if 'TrackDurationMs' in X.columns:
    X['Log_TrackDurationMs'] = np.log1p(X['TrackDurationMs'])
    X_test['Log_TrackDurationMs'] = np.log1p(X_test['TrackDurationMs'])
    features.append('Log_TrackDurationMs')

if 'AudioLoudness' in X.columns:
    X['Log_AudioLoudness'] = np.log1p(np.abs(X['AudioLoudness']))  # Handle negative values
    X_test['Log_AudioLoudness'] = np.log1p(np.abs(X_test['AudioLoudness']))
    features.append('Log_AudioLoudness')

# InstrumentalScore transformations
if 'InstrumentalScore' in X.columns:
    # Log transform for skewness
    X['Log_InstrumentalScore'] = np.log1p(X['InstrumentalScore'])
    X_test['Log_InstrumentalScore'] = np.log1p(X_test['InstrumentalScore'])
    features.append('Log_InstrumentalScore')
    
    # Square for non-linear relationship
    X['InstrumentalScore_Squared'] = X['InstrumentalScore'] ** 2
    X_test['InstrumentalScore_Squared'] = X_test['InstrumentalScore'] ** 2
    features.append('InstrumentalScore_Squared')

# Polynomial features for top 2 numeric features
if len(numeric_features) >= 2:
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_cols = numeric_features[:2]  # Use first 2 features
    poly_train = pd.DataFrame(poly.fit_transform(X[poly_cols]), index=X.index)
    poly_test = pd.DataFrame(poly.transform(X_test[poly_cols]), index=X_test.index)
    poly_names = [f'Poly_{name}' for name in poly.get_feature_names_out(poly_cols)]
    X[poly_names] = poly_train
    X_test[poly_names] = poly_test
    features.extend(poly_names)

# Handle categorical features (none expected based on data info, but keeping for robustness)
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_features:
    print(f"Categorical features detected: {categorical_features}")
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)
    X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)
    features = X.columns.tolist()

# Check missing values
print("\nMissing values in train:", X.isnull().sum().sum())
print("Missing values in test:", X_test.isnull().sum().sum())
print(f"\nFinal features (top 10): {features[:10]}{'...' if len(features) > 10 else ''}")
print(f"Total engineered features: {len(features)}")


# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Validation set shape: {X_val_scaled.shape}")


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42,
        'objective': 'reg:squarederror',
        'early_stopping_rounds': 50
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
    y_pred = model.predict(X_val_scaled)
    return np.sqrt(mean_squared_error(y_val, y_pred))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
best_params = study.best_params
print("\nBest parameters:", best_params)
print(f"Best validation RMSE: {study.best_value:.4f}")


# Train final model
best_params['early_stopping_rounds'] = 50
model = xgb.XGBRegressor(**best_params)
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=100
)

# Validation score
y_val_pred = model.predict(X_val_scaled)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"\nValidation RMSE: {val_rmse:.4f}")

# Clipped validation
y_val_pred_clipped = np.clip(y_val_pred, 60, 200)
val_rmse_clipped = np.sqrt(mean_squared_error(y_val, y_val_pred_clipped))
print(f"Validation RMSE (clipped 60-200): {val_rmse_clipped:.4f}")

# Cross-validation (without early stopping for CV)
cv_params = best_params.copy()
cv_params.pop('early_stopping_rounds', None)  # Remove early stopping for CV
cv_model = xgb.XGBRegressor(**cv_params)
X_scaled = scaler.transform(X)
cv_scores = cross_val_score(cv_model, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
print(f"CV RMSE (mean): {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=15)
plt.title('Top 15 Feature Importances')
plt.show()


if id_col is None:
    print("Error: Cannot generate submission without ID column.")
else:
    # Predict and clip
    test_predictions = model.predict(X_test_scaled)
    test_predictions = np.clip(test_predictions, 60, 200)

    # Create submission
    submission = pd.DataFrame({
        'id': test_df[id_col],
        'BeatsPerMinute': test_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file 'submission.csv' created!")
    print(submission.head())
    print(f"\nPrediction stats: mean={test_predictions.mean():.2f}, std={test_predictions.std():.2f}, min={test_predictions.min():.2f}, max={test_predictions.max():.2f}")

    # Verify submission format
    if set(submission.columns) == {'id', 'BeatsPerMinute'}:
        print("Submission format correct.")
    else:
        print("Warning: Submission format may be incorrect. Expected columns: ['id', 'BeatsPerMinute']")


# Correlation heatmap (original features only)
# Use only original features that exist in train_df
original_features = [col for col in train_df.columns if col not in [id_col, target_col]]
plot_features = original_features + [target_col]
numeric_plot_features = train_df[plot_features].select_dtypes(include=[np.number]).columns
corr_data = train_df[numeric_plot_features].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation with BeatsPerMinute (Original Features)')
plt.show()

# Pairplot for top correlated features
top_features = corr_data['BeatsPerMinute'].abs().sort_values(ascending=False).index[1:4]
sns.pairplot(train_df[list(top_features) + [target_col]], diag_kind='kde')
plt.suptitle('Pairplot of Top Features vs BeatsPerMinute', y=1.02)
plt.show()

# Target distribution and outliers
plt.figure(figsize=(8, 4))
sns.histplot(y, kde=True, bins=30)
plt.title('Distribution of BeatsPerMinute')
plt.xlabel('BPM')
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x=y)
plt.title('Boxplot of BeatsPerMinute')
plt.xlabel('BPM')
plt.show()

# Additional: Show engineered features correlation (if model exists)
if 'model' in locals() and hasattr(model, 'feature_importances_'):
    plt.figure(figsize=(12, 8))
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Top 15 Most Important Features (Including Engineered)')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    print("Top 10 most important features:")
    for i, (feat, imp) in enumerate(zip(feature_importance['feature'].head(10), 
                                       feature_importance['importance'].head(10)), 1):
        print(f"{i:2d}. {feat}: {imp:.4f}")

