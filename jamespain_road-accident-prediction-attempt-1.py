import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM (install if needed)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print(f"XGBoost available: {XGBOOST_AVAILABLE}")
print(f"LightGBM available: {LIGHTGBM_AVAILABLE}")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


X = train_df.drop(['id', 'accident_risk'], axis=1)
y = train_df['accident_risk']
X_test = test_df.drop('id', axis=1)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")


# Encode categorical variables
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    combined = pd.concat([X[col].astype(str), X_test[col].astype(str)])
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

print("Categorical encoding completed.")


def create_engineered_features(df):
    """Create interaction and polynomial features"""
    df_eng = df.copy()
    
    # Interaction features for top predictors
    df_eng['curvature_x_speed'] = df['curvature'] * df['speed_limit']
    df_eng['curvature_x_lighting'] = df['curvature'] * df['lighting']
    df_eng['lighting_x_weather'] = df['lighting'] * df['weather']
    df_eng['speed_x_weather'] = df['speed_limit'] * df['weather']
    df_eng['curvature_x_weather'] = df['curvature'] * df['weather']
    
    # Polynomial features for curvature (most important)
    df_eng['curvature_squared'] = df['curvature'] ** 2
    df_eng['curvature_cubed'] = df['curvature'] ** 3
    
    # Speed limit categories (risk zones)
    df_eng['speed_category'] = pd.cut(df['speed_limit'], bins=[0, 35, 55, 100], labels=[0, 1, 2]).astype(int)
    
    # High risk combination: high curvature + high speed
    df_eng['high_risk_curve'] = ((df['curvature'] > df['curvature'].median()) & 
                                  (df['speed_limit'] > df['speed_limit'].median())).astype(int)
    
    # Poor visibility: dim/night lighting + bad weather
    df_eng['poor_visibility'] = ((df['lighting'] >= 1) & (df['weather'] >= 1)).astype(int)
    
    # Three-way interaction for most important features
    df_eng['curvature_x_speed_x_lighting'] = df['curvature'] * df['speed_limit'] * df['lighting']
    
    return df_eng

# Apply feature engineering
X_engineered = create_engineered_features(X)
X_test_engineered = create_engineered_features(X_test)

print(f"Original features: {X.shape[1]}")
print(f"Engineered features: {X_engineered.shape[1]}")
print(f"New features added: {X_engineered.shape[1] - X.shape[1]}")


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_engineered)
X_test_scaled = scaler.transform(X_test_engineered)

X_scaled = pd.DataFrame(X_scaled, columns=X_engineered.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_engineered.columns)

print("Feature scaling completed.")


# Split data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")


def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    """Train and evaluate a model"""
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print(f"Training   - RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}, RÂ²: {train_r2:.6f}")
    print(f"Validation - RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}, RÂ²: {val_r2:.6f}")
    print(f"Overfit gap: {(train_r2 - val_r2):.4f}")
    
    return model, val_rmse, val_r2, y_val_pred


models_performance = {}
trained_models = {}


# Random Forest (baseline)
rf_model, rf_rmse, rf_r2, rf_pred = evaluate_model(
    RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    X_train, y_train, X_val, y_val,
    "Random Forest (Baseline)"
)
models_performance['RF_Baseline'] = {'rmse': rf_rmse, 'r2': rf_r2}
trained_models['RF_Baseline'] = rf_model


# Gradient Boosting (baseline)
gb_model, gb_rmse, gb_r2, gb_pred = evaluate_model(
    GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    X_train, y_train, X_val, y_val,
    "Gradient Boosting (Baseline)"
)
models_performance['GB_Baseline'] = {'rmse': gb_rmse, 'r2': gb_r2}
trained_models['GB_Baseline'] = gb_model


if XGBOOST_AVAILABLE:
    xgb_model, xgb_rmse, xgb_r2, xgb_pred = evaluate_model(
        xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        X_train, y_train, X_val, y_val,
        "XGBoost"
    )
    models_performance['XGBoost'] = {'rmse': xgb_rmse, 'r2': xgb_r2}
    trained_models['XGBoost'] = xgb_model
else:
    print("\nXGBoost not available - skipping")


if LIGHTGBM_AVAILABLE:
    lgb_model, lgb_rmse, lgb_r2, lgb_pred = evaluate_model(
        lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        X_train, y_train, X_val, y_val,
        "LightGBM"
    )
    models_performance['LightGBM'] = {'rmse': lgb_rmse, 'r2': lgb_r2}
    trained_models['LightGBM'] = lgb_model
else:
    print("\nLightGBM not available - skipping")


# Find current best model
best_model_name = min(models_performance, key=lambda x: models_performance[x]['rmse'])
print(f"Current best model: {best_model_name}")
print(f"RMSE: {models_performance[best_model_name]['rmse']:.6f}")
print(f"RÂ²: {models_performance[best_model_name]['r2']:.6f}")


# Hyperparameter tuning for Random Forest
print("\nTuning Random Forest hyperparameters...")

param_grid = {
    'n_estimators': [150, 200, 250],
    'max_depth': [20, 25, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_tuned = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid,
    n_iter=20,
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_tuned.fit(X_train, y_train)

print(f"\nBest parameters: {rf_tuned.best_params_}")
print(f"Best CV RMSE: {-rf_tuned.best_score_:.6f}")

# Evaluate tuned model
rf_tuned_model, rf_tuned_rmse, rf_tuned_r2, rf_tuned_pred = evaluate_model(
    rf_tuned.best_estimator_,
    X_train, y_train, X_val, y_val,
    "Random Forest (Tuned)"
)
models_performance['RF_Tuned'] = {'rmse': rf_tuned_rmse, 'r2': rf_tuned_r2}
trained_models['RF_Tuned'] = rf_tuned_model


# Create weighted ensemble of top 3 models
print("\nCreating ensemble of top models...")

# Get top 3 models by RÂ²
top_models = sorted(models_performance.items(), key=lambda x: x[1]['r2'], reverse=True)[:3]
print("\nTop 3 models for ensemble:")
for name, metrics in top_models:
    print(f"  {name}: RÂ² = {metrics['r2']:.6f}")

# Simple average ensemble
ensemble_pred = np.zeros(len(y_val))
for model_name, _ in top_models:
    model = trained_models[model_name]
    ensemble_pred += model.predict(X_val)
ensemble_pred /= len(top_models)

ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
ensemble_r2 = r2_score(y_val, ensemble_pred)
ensemble_mae = mean_absolute_error(y_val, ensemble_pred)

print(f"\n{'='*60}")
print("Ensemble Model (Average of Top 3)")
print(f"{'='*60}")
print(f"Validation - RMSE: {ensemble_rmse:.6f}, MAE: {ensemble_mae:.6f}, RÂ²: {ensemble_r2:.6f}")

models_performance['Ensemble'] = {'rmse': ensemble_rmse, 'r2': ensemble_r2}


# Create comparison dataframe
comparison_df = pd.DataFrame(models_performance).T
comparison_df = comparison_df.sort_values('r2', ascending=False)

print("\n" + "="*70)
print("MODEL PERFORMANCE COMPARISON")
print("="*70)
print(comparison_df.to_string())

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# RMSE comparison
axes[0].barh(comparison_df.index, comparison_df['rmse'], color='skyblue', edgecolor='black')
axes[0].set_xlabel('Validation RMSE (lower is better)')
axes[0].set_title('Model Comparison - RMSE')
axes[0].invert_yaxis()

# RÂ² comparison
axes[1].barh(comparison_df.index, comparison_df['r2'], color='lightgreen', edgecolor='black')
axes[1].set_xlabel('Validation RÂ² (higher is better)')
axes[1].set_title('Model Comparison - RÂ²')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

best_final_model = comparison_df.index[0]
print(f"\nğŸ�† Best Model: {best_final_model}")
print(f"   RMSE: {comparison_df.loc[best_final_model, 'rmse']:.6f}")
print(f"   RÂ²: {comparison_df.loc[best_final_model, 'r2']:.6f}")


# Get feature importance from best tree-based model
if 'RF' in best_final_model or best_final_model == 'Ensemble':
    importance_model = trained_models['RF_Tuned'] if 'RF_Tuned' in trained_models else rf_model
    feature_importance = pd.DataFrame({
        'feature': X_engineered.columns,
        'importance': importance_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string())
    
    # Visualize top 20 features
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# Train final model on full training data
print(f"\nTraining final {best_final_model} model on full training dataset...")

if best_final_model == 'Ensemble':
    # Retrain top 3 models on full data
    final_predictions = np.zeros(len(X_test_scaled))
    for model_name, _ in top_models:
        model = trained_models[model_name]
        model.fit(X_scaled, y)
        final_predictions += model.predict(X_test_scaled)
    final_predictions /= len(top_models)
else:
    final_model = trained_models[best_final_model]
    final_model.fit(X_scaled, y)
    final_predictions = final_model.predict(X_test_scaled)

print("Training completed!")
print(f"\nPrediction Statistics:")
print(f"Mean: {final_predictions.mean():.4f}")
print(f"Std: {final_predictions.std():.4f}")
print(f"Min: {final_predictions.min():.4f}")
print(f"Max: {final_predictions.max():.4f}")


submission = pd.DataFrame({
    'id': test_df['id'],
    'accident_risk': final_predictions
})

submission.to_csv('submission_enhanced.csv', index=False)
print("Enhanced submission file created: submission_enhanced.csv")
print(f"\nSubmission preview:")
print(submission.head(10))

