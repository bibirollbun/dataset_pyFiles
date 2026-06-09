#system handling
import os
import time
import warnings
warnings.filterwarnings('ignore')

#data handling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize

#model handling
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold,cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.shape


test.shape


train.info()


train.dtypes


print("Target column statistics (accident_risk):")

train['accident_risk'].describe()


train.isnull().sum()


print("Duplicated Rows:",train.duplicated().sum())


train_num_cols = train.select_dtypes(include='number').columns.tolist()
train_num_cols.remove('id')
correlation_matrix = train[train_num_cols].corr()

plt.figure(figsize=(6,5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

nominalFeatures = ['road_type', 'weather']

ohe = OneHotEncoder(drop='first', sparse_output=False)
encoded_train = ohe.fit_transform(train[nominalFeatures])
encoded_test = ohe.fit_transform(test[nominalFeatures])

feature_names_dropped = ohe.get_feature_names_out(nominalFeatures)

encoded_train_df = pd.DataFrame(encoded_train, columns=feature_names_dropped)
encoded_test_df = pd.DataFrame(encoded_test, columns=feature_names_dropped)
train = pd.concat([train.drop(columns=nominalFeatures), encoded_train_df], axis=1)
test = pd.concat([test.drop(columns=nominalFeatures), encoded_test_df], axis=1)

train.head()


ordinal_features = ['lighting', 'time_of_day']

oe = OrdinalEncoder()
for feature in ordinal_features: 
    train[feature] = oe.fit_transform(train[feature].values.reshape(-1,1))
    test[feature] = oe.fit_transform(test[feature].values.reshape(-1,1))

train.head()


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_cols = train.select_dtypes(include='number').columns.tolist()
num_cols.remove('id')
num_cols.remove('accident_risk')

train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.fit_transform(test[num_cols])

train.head()


CATEGORICAL_FEATURES = ['lighting', 'time_of_day']
BOOLEAN_FEATURES = ['road_signs_present', 'public_road', 'holiday', 'school_season']
NUMERICAL_FEATURES = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
TARGET = 'accident_risk'
ID_COL = 'id'


def engineer_features(df):
    """
    Create domain-informed feature interactions.
    """
    df_eng = df.copy()
    
    # Core interactions
    df_eng['curv_speed'] = df_eng['curvature'] * df_eng['speed_limit']
    df_eng['lane_speed'] = df_eng['num_lanes'] * df_eng['speed_limit']
    df_eng['accidents_speed'] = df_eng['num_reported_accidents'] * df_eng['speed_limit']
    df_eng['accidents_curv'] = df_eng['num_reported_accidents'] * df_eng['curvature']
    
    # Polynomial features
    df_eng['curvature_sq'] = df_eng['curvature'] ** 2
    df_eng['curvature_cube'] = df_eng['curvature'] ** 3
    df_eng['speed_sq'] = df_eng['speed_limit'] ** 2
    
    # Risk scores
    df_eng['risk_intensity'] = (df_eng['curvature'] * df_eng['speed_limit']) / 50
    df_eng['lane_capacity_risk'] = (5 - df_eng['num_lanes']) * df_eng['speed_limit']
    df_eng['accidents_per_lane'] = df_eng['num_reported_accidents'] / (df_eng['num_lanes'] + 1)
    
    # Binary indicators
    df_eng['high_risk_combo'] = ((df_eng['curvature'] > 0.5) & 
                                  (df_eng['speed_limit'] >= 60)).astype(int)
    
    return df_eng





# Preprocessing
train_processed = train.copy()
test_processed = test.copy()

# Convert booleans
for col in BOOLEAN_FEATURES:
    train_processed[col] = train_processed[col].astype(int)
    test_processed[col] = test_processed[col].astype(int)

# Label encode categoricals
label_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    train_processed[f'{col}_enc'] = le.fit_transform(train_processed[col])
    test_processed[f'{col}_enc'] = le.transform(test_processed[col])
    label_encoders[col] = le

# Apply feature engineering
train = engineer_features(train_processed)
test = engineer_features(test_processed)

print(f"Feature engineering complete")
print(f"Original features: {len(CATEGORICAL_FEATURES + BOOLEAN_FEATURES + NUMERICAL_FEATURES)}")
print(f"Engineered features: {train.shape[1]}")
print(f"New features created: {test.shape[1] - train_processed.shape[1]}")


train.head()



xgb_params = {
    "objective": "reg:squarederror",
    "tree_method": "auto",
    "device": "cuda",
    "learning_rate": 0.0126,
    "n_estimators": 803,
    "max_depth": 11,
    "subsample": 0.801,
    "colsample_bytree": 0.813,
    "reg_alpha": 1.60,
    "reg_lambda": 7.52,
    "verbosity": 1,
    "eval_metric": "rmse",
    "random_state": 42,
    "enable_categorical": True
}


lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "n_estimators": 1200,
    "num_leaves": 127,
    "max_depth": 9,
    "min_child_samples": 40,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,
    "reg_alpha": 1.2,
    "reg_lambda": 5.0,
    "random_state": 42,
    "verbosity": -1,
    "device": "gpu"
}

cat_params = {
    "loss_function": "RMSE",
    "learning_rate": 0.01,
    "iterations": 1200,
    "depth": 9,
    "l2_leaf_reg": 6.0,
    "bootstrap_type": "MVS",
    "subsample": 0.85,
    "bagging_temperature": 0.7,
    "random_state": 42,
    "verbose": False,
    "task_type": "GPU"
}




# Prepare features
features = train.columns
features = features.drop(['id', 'accident_risk'])

X = train[features]
y = train['accident_risk']

print("Starting Cross-Validation Training...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store out-of-fold predictions for evaluation
oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

# Store models from each fold
xgb_models = []
lgb_models = []
cat_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n{'='*40}")
    print(f"Training Fold {fold + 1}/5")
    print(f"{'='*40}")
    
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]
    
    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train_fold, y_train_fold)
    oof_xgb[val_idx] = xgb_model.predict(X_val_fold)
    xgb_models.append(xgb_model)
    
    # Train LightGBM
    print("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train_fold, y_train_fold)
    oof_lgb[val_idx] = lgb_model.predict(X_val_fold)
    lgb_models.append(lgb_model)
    
    # Train CatBoost
    print("Training CatBoost...")
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_train_fold, y_train_fold)
    oof_cat[val_idx] = cat_model.predict(X_val_fold)
    cat_models.append(cat_model)
    
    # Fold results
    fold_xgb_rmse = np.sqrt(mean_squared_error(y_val_fold, oof_xgb[val_idx]))
    fold_lgb_rmse = np.sqrt(mean_squared_error(y_val_fold, oof_lgb[val_idx]))
    fold_cat_rmse = np.sqrt(mean_squared_error(y_val_fold, oof_cat[val_idx]))
    
    print(f"Fold {fold+1} RMSE - XGBoost: {fold_xgb_rmse:.6f}")
    print(f"Fold {fold+1} RMSE - LightGBM: {fold_lgb_rmse:.6f}")
    print(f"Fold {fold+1} RMSE - CatBoost: {fold_cat_rmse:.6f}")

# Calculate overall CV scores
print(f"\n{'='*40}")
print("Overall Cross-Validation Results:")
print(f"{'='*40}")


xgb_cv_rmse = np.sqrt(mean_squared_error(y, oof_xgb))
lgb_cv_rmse = np.sqrt(mean_squared_error(y, oof_lgb))
cat_cv_rmse = np.sqrt(mean_squared_error(y, oof_cat))

xgb_cv_r2 = r2_score(y, oof_xgb)
lgb_cv_r2 = r2_score(y, oof_lgb)
cat_cv_r2 = r2_score(y, oof_cat)

print(f"XGBoost  - CV RMSE: {xgb_cv_rmse:.6f}, R2: {xgb_cv_r2:.6f}")
print(f"LightGBM - CV RMSE: {lgb_cv_rmse:.6f}, R2: {lgb_cv_r2:.6f}")
print(f"CatBoost - CV RMSE: {cat_cv_rmse:.6f}, R2: {cat_cv_r2:.6f}")




# Optimize ensemble weights

def ensemble_rmse(weights):
    ensemble = weights[0]*oof_xgb + weights[1]*oof_lgb + weights[2]*oof_cat
    return np.sqrt(mean_squared_error(y, ensemble))

result = minimize(
    ensemble_rmse,
    x0=[0.60, 0.20, 0.20],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
)

optimal_weights = result.x
print(f"\nOptimal Ensemble Weights: [{optimal_weights[0]:.2f}, {optimal_weights[1]:.2f}, {optimal_weights[2]:.2f}]")
print(f"Ensemble CV RMSE: {result.fun:.6f}")


print(f"\n{'='*40}")
print("Retraining on Full Dataset for Submission...")
print(f"{'='*40}")

# Train final models on ALL data
final_xgb = xgb.XGBRegressor(**xgb_params)
final_xgb.fit(X, y)

final_lgb = lgb.LGBMRegressor(**lgb_params)
final_lgb.fit(X, y)

final_cat = CatBoostRegressor(**cat_params)
final_cat.fit(X, y)

print("Full data training completed!")


# Regression metrics function
def print_regression_metrics(y_true, y_pred, model_name="Model", dataset="CV"):
    from sklearn.metrics import mean_absolute_error, r2_score
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"==== {model_name} {dataset} Metrics ====")
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"R2   : {r2:.6f}")
    print()


# Evaluate models using out-of-fold predictions
print("\n=== Cross-Validation Performance ===\n")

# Individual model CV performance
print_regression_metrics(y, oof_xgb, "XGBoost", "CV")
print_regression_metrics(y, oof_lgb, "LightGBM", "CV")
print_regression_metrics(y, oof_cat, "CatBoost", "CV")

# Optimized ensemble CV performance
optimized_ensemble_pred = (optimal_weights[0] * oof_xgb + 
                           optimal_weights[1] * oof_lgb + 
                           optimal_weights[2] * oof_cat)

print("="*40)
print_regression_metrics(y, optimized_ensemble_pred, "Optimized Ensemble", "CV")
print("="*40)

# Show weight comparison
print(f"\nEnsemble Weight Optimization:")
print(f"  XGBoost:  {optimal_weights[0]:.3f}")
print(f"  LightGBM: {optimal_weights[1]:.3f}")
print(f"  CatBoost: {optimal_weights[2]:.3f}")
print(f"{'='*40}\n")


# Visualization function
def plot_predictions(y_true, y_pred, model_name="Model", dataset="CV"):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=20)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel("Actual Accident Risk", fontsize=12)
    plt.ylabel("Predicted Accident Risk", fontsize=12)
    plt.title(f"{model_name} - Predicted vs Actual ({dataset})", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




print("\nGenerating prediction plots...")

# Plot out-of-fold predictions
plot_predictions(y, oof_xgb, "XGBoost", "Cross-Validation")
plot_predictions(y, oof_lgb, "LightGBM", "Cross-Validation")
plot_predictions(y, oof_cat, "CatBoost", "Cross-Validation")

# Plot optimized ensemble
optimized_ensemble_pred = (optimal_weights[0] * oof_xgb + 
                           optimal_weights[1] * oof_lgb + 
                           optimal_weights[2] * oof_cat)

plot_predictions(y, optimized_ensemble_pred, 
                f"Optimized Ensemble [{optimal_weights[0]:.2f}, {optimal_weights[1]:.2f}, {optimal_weights[2]:.2f}]", 
                "Cross-Validation")


# Feature importance visualization
def plot_feature_importance(model, model_name="Model", max_features=30, importance_type='weight'):
    
    plt.figure(figsize=(10, 8))
    
    if model_name == "XGBoost":
        xgb.plot_importance(model, importance_type=importance_type, max_num_features=max_features)
    elif model_name == "LightGBM":
        lgb.plot_importance(model, importance_type='gain', max_num_features=max_features, figsize=(10, 8))
    elif model_name == "CatBoost":
        feature_importance = model.get_feature_importance()
        feature_names = model.feature_names_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=True).tail(max_features)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Features')
    elif model_name == "Optimized Ensemble":
       
        xgb.plot_importance(model, importance_type='weight', max_num_features=max_features)
        plt.title(f"Ensemble - Feature Importance (Based on XGBoost)\nTop {max_features}", 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        return
    
    plt.title(f"{model_name} - Feature Importance (Top {max_features})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()




# Calculate CV RMSE for all models
xgb_rmse = np.sqrt(mean_squared_error(y, oof_xgb))
lgb_rmse = np.sqrt(mean_squared_error(y, oof_lgb))
cat_rmse = np.sqrt(mean_squared_error(y, oof_cat))

# Optimized ensemble prediction
optimized_ensemble_pred = (optimal_weights[0] * oof_xgb + 
                           optimal_weights[1] * oof_lgb + 
                           optimal_weights[2] * oof_cat)
ensemble_rmse = np.sqrt(mean_squared_error(y, optimized_ensemble_pred))


model_scores = {
    "XGBoost": (final_xgb, xgb_rmse),
    "LightGBM": (final_lgb, lgb_rmse),
    "CatBoost": (final_cat, cat_rmse),
    "Optimized Ensemble": (final_xgb, ensemble_rmse) 
}

best_model_name = min(model_scores, key=lambda x: model_scores[x][1])
best_model, best_rmse = model_scores[best_model_name]

print(f"\n{'='*40}")
print("Model Ranking (CV RMSE):")
print(f"{'='*40}")
for name, (_, rmse) in sorted(model_scores.items(), key=lambda x: x[1][1]):
    print(f"{name:20s}: {rmse:.6f}")
print(f"{'='*40}")
print(f"Best Model: {best_model_name} (CV RMSE: {best_rmse:.6f})")
print(f"{'='*40}\n")

# Plot feature importance for best model
plot_feature_importance(best_model, best_model_name, max_features=30)



print("\n" + "="*40)
print("Training Enhanced Stacking Meta-Model...")
print("="*40)

alphas = [0.009, 0.01, 0.02, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
best_alpha_score = float('inf')
best_alpha = None

print("\nTuning Ridge alpha parameter...")
meta_features_train = np.column_stack([oof_xgb, oof_lgb, oof_cat])

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(meta_features_train, y)
    pred = ridge.predict(meta_features_train)
    rmse = np.sqrt(mean_squared_error(y, pred))
    print(f"Alpha {alpha:5.2f}: RMSE = {rmse:.6f}")
    
    if rmse < best_alpha_score:
        best_alpha_score = rmse
        best_alpha = alpha

print(f"\nBest Ridge alpha: {best_alpha} (RMSE: {best_alpha_score:.6f})")


print("\nTesting enhanced meta-features...")

weighted_avg_xgb_lgb = 0.6 * oof_xgb + 0.4 * oof_lgb
weighted_avg_xgb_cat = 0.6 * oof_xgb + 0.4 * oof_cat

meta_features_enhanced = np.column_stack([
    oof_xgb, oof_lgb, oof_cat,
    oof_xgb * oof_lgb,
    oof_xgb * oof_cat,
    weighted_avg_xgb_lgb,
    weighted_avg_xgb_cat,
    (oof_xgb + oof_lgb) / 2, 
    (oof_xgb + oof_cat) / 2, 
    (oof_lgb + oof_cat) / 2,
    np.minimum(oof_xgb, np.minimum(oof_lgb, oof_cat)),
    np.maximum(oof_xgb, np.maximum(oof_lgb, oof_cat))
])

ridge_enhanced = Ridge(alpha=best_alpha)
ridge_enhanced.fit(meta_features_enhanced, y)
pred_enhanced = ridge_enhanced.predict(meta_features_enhanced)
rmse_enhanced = np.sqrt(mean_squared_error(y, pred_enhanced))
print(f"Enhanced features RMSE: {rmse_enhanced:.6f}")


print("\nTesting multiple meta-models...")
meta_models = {
    'Ridge_tuned': Ridge(alpha=best_alpha),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
    'BayesianRidge': BayesianRidge(),
    'LightGBM_shallow': lgb.LGBMRegressor(
        n_estimators=50, learning_rate=0.001, max_depth=3,
        num_leaves=4, min_child_samples=100, random_state=42, verbosity=-1
    )
}

best_meta_model = None
best_meta_rmse = float('inf')
best_meta_name = None
use_enhanced = False

# Test on basic features
for meta_name, meta_model in meta_models.items():
    meta_model.fit(meta_features_train, y)
    pred = meta_model.predict(meta_features_train)
    rmse = np.sqrt(mean_squared_error(y, pred))
    print(f"{meta_name:20s} - RMSE: {rmse:.6f}")
    
    if rmse < best_meta_rmse:
        best_meta_rmse = rmse
        best_meta_model = meta_model
        best_meta_name = meta_name

# Check if enhanced features are better
if rmse_enhanced < best_meta_rmse:
    best_meta_rmse = rmse_enhanced
    best_meta_model = ridge_enhanced
    best_meta_name = "Ridge_enhanced"
    use_enhanced = True

print(f"\n{'='*40}")
print(f"Best Stacking Approach: {best_meta_name}")
print(f"Stacking CV RMSE: {best_meta_rmse:.6f}")
print(f"Optimized Ensemble CV RMSE: {result.fun:.6f}")
print(f"Improvement: {result.fun - best_meta_rmse:.6f}")
print(f"{'='*40}")

# Decide whether to use stacking
use_stacking = best_meta_rmse < result.fun


# PREPARE TEST DATA FOR SUBMISSION

print("\nPreparing test data for submission...")
test_data = test.drop(columns=["id"]).copy()

# Generate predictions from all models FIRST
print("Generating predictions on test dataset...")
xgb_test_predictions = final_xgb.predict(test_data)
lgb_test_predictions = final_lgb.predict(test_data)
cat_test_predictions = final_cat.predict(test_data)



if use_stacking:
    print(f"\nStacking wins! Using {best_meta_name}")
    

    if use_enhanced:
        
        weighted_avg_xgb_lgb = 0.6 * xgb_test_predictions + 0.4 * lgb_test_predictions
        weighted_avg_xgb_cat = 0.6 * xgb_test_predictions + 0.4 * cat_test_predictions
        
        meta_features_test = np.column_stack([
            xgb_test_predictions, 
            lgb_test_predictions, 
            cat_test_predictions,
            xgb_test_predictions * lgb_test_predictions,
            xgb_test_predictions * cat_test_predictions,
            weighted_avg_xgb_lgb,
            weighted_avg_xgb_cat,
            (xgb_test_predictions + lgb_test_predictions) / 2, 
            (xgb_test_predictions + cat_test_predictions) / 2, 
            (lgb_test_predictions + cat_test_predictions) / 2,
            np.minimum(xgb_test_predictions, np.minimum(lgb_test_predictions, cat_test_predictions)),
            np.maximum(xgb_test_predictions, np.maximum(lgb_test_predictions, cat_test_predictions))
        ])
    else:
        meta_features_test = np.column_stack([
            xgb_test_predictions, lgb_test_predictions, cat_test_predictions
        ])
    
    ensemble_predictions = best_meta_model.predict(meta_features_test)
    method_used = f"Stacking ({best_meta_name})"

else:
    print(f"\nEnsemble wins! Using optimized weights")
    ensemble_predictions = (
        optimal_weights[0] * xgb_test_predictions + 
        optimal_weights[1] * lgb_test_predictions + 
        optimal_weights[2] * cat_test_predictions
    )
    method_used = "Optimized Ensemble"



# Create submission
submission = pd.DataFrame({
    "id": test["id"],          
    "accident_risk": ensemble_predictions    
})

submission.to_csv("submission.csv", index=False)
print(f"\nSubmission dataset saved as 'submission.csv'!")
print(f"Method used: {method_used}")
print(f"Number of predictions: {len(submission)}")
print(f"Prediction range: [{ensemble_predictions.min():.4f}, {ensemble_predictions.max():.4f}]")

# Show model details
if use_stacking:
    if 'LightGBM' in best_meta_name:
        print(f"\nMeta-Model Feature Importance:")
        importances = best_meta_model.feature_importances_
        print(f"  XGBoost weight:  {importances[0]:.3f}")
        print(f"  LightGBM weight: {importances[1]:.3f}")
        print(f"  CatBoost weight: {importances[2]:.3f}")
    elif best_meta_name in ['Ridge', 'Ridge_tuned', 'Ridge_enhanced', 'ElasticNet', 'BayesianRidge']:
        print(f"\nMeta-Model Coefficients:")
        coeffs = best_meta_model.coef_
        if use_enhanced:
            print(f"  Base models and {len(coeffs)-3} interaction features")
            print(f"  XGBoost coefficient:  {coeffs[0]:.3f}")
            print(f"  LightGBM coefficient: {coeffs[1]:.3f}")
            print(f"  CatBoost coefficient: {coeffs[2]:.3f}")
        else:
            print(f"  XGBoost:  {coeffs[0]:.3f}")
            print(f"  LightGBM: {coeffs[1]:.3f}")
            print(f"  CatBoost: {coeffs[2]:.3f}")
else:
    print(f"\nOptimized Ensemble Weights:")
    print(f"  XGBoost:  {optimal_weights[0]:.3f}")
    print(f"  LightGBM: {optimal_weights[1]:.3f}")
    print(f"  CatBoost: {optimal_weights[2]:.3f}")

