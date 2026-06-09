import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


# Custom RMSLE metric
def rmsle(y_true, y_pred):
    # Ensure positive values for log calculation
    y_true = np.maximum(y_true, 0.001)
    y_pred = np.maximum(y_pred, 0.001)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

def rmsle_objective(y_true, y_pred):
    """Custom RMSLE for gradient boosting"""
    return 'rmsle', rmsle(y_true, y_pred), False



# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

print("Data Info:")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Target stats: mean={train['Calories'].mean():.2f}, std={train['Calories'].std():.2f}")
print(f"Target range: min={train['Calories'].min():.2f}, max={train['Calories'].max():.2f}")


# Check for negative or zero values
negative_calories = (train['Calories'] <= 0).sum()
if negative_calories > 0:
    print(f"Warning: {negative_calories} negative or zero calorie values detected")
    # Clip negative values to small positive number
    train['Calories'] = np.maximum(train['Calories'], 0.001)

# Global encoder
sex_encoder = LabelEncoder()


# EDA and Feature Engineering
def create_features(df, is_train=True):
    df = df.copy()
    
    # Encode Sex
    if is_train:
        df['Sex_encoded'] = sex_encoder.fit_transform(df['Sex'])
    else:
        df['Sex_encoded'] = sex_encoder.transform(df['Sex'])
    
    # BMI (Body Mass Index)
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    
    # Age groups
    df['Age_group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4])
    df['Age_group'] = df['Age_group'].astype(int)
    
    # Heart rate zones (based on age)
    df['Max_HR'] = 220 - df['Age']
    df['HR_Zone'] = df['Heart_Rate'] / df['Max_HR']
    
    # Body temp categories
    df['Temp_high'] = (df['Body_Temp'] > 40.0).astype(int)
    df['Temp_normal'] = ((df['Body_Temp'] >= 37.0) & (df['Body_Temp'] <= 40.0)).astype(int)
    
    # Interaction features
    df['Weight_Duration'] = df['Weight'] * df['Duration']
    df['HR_Duration'] = df['Heart_Rate'] * df['Duration']
    df['BMI_Duration'] = df['BMI'] * df['Duration']
    df['Age_Weight'] = df['Age'] * df['Weight']
    df['HR_Weight'] = df['Heart_Rate'] * df['Weight']
    
    # Metabolic features
    df['Metabolic_rate'] = df['Weight'] * df['Heart_Rate'] / df['Age']
    df['Intensity'] = df['Heart_Rate'] * df['Duration'] / df['Weight']
    
    # Polynomial features for key variables
    df['Duration_sq'] = df['Duration'] ** 2
    df['Weight_sq'] = df['Weight'] ** 2
    df['HR_sq'] = df['Heart_Rate'] ** 2
    
    # Log transforms
    df['log_Duration'] = np.log1p(df['Duration'])
    df['log_Weight'] = np.log1p(df['Weight'])
    df['log_HR'] = np.log1p(df['Heart_Rate'])
    
    return df


# Apply feature engineering
print("Creating features...")
train_fe = create_features(train, is_train=True)
test_fe = create_features(test, is_train=False)

# Feature selection
feature_cols = [col for col in train_fe.columns if col not in ['id', 'Sex', 'Calories']]
X = train_fe[feature_cols]
y = train_fe['Calories']
X_test = test_fe[feature_cols]

print(f"Number of features: {len(feature_cols)}")
print("Features:", feature_cols)


# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Model 1: XGBoost
print("\nTraining XGBoost...")
xgb_params = {
    'objective': 'reg:squaredlogerror',  # Direct RMSLE optimization
    'eval_metric': 'rmsle',
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_estimators': 2000
}

xgb_scores = []
xgb_predictions = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        # early_stopping_rounds=100,
        # verbose=False
    )
    
    val_pred = model.predict(X_val_fold)
    score = rmsle(y_val_fold, val_pred)
    xgb_scores.append(score)
    
    xgb_predictions += model.predict(X_test) / 5
    print(f"Fold {fold+1} RMSLE: {score:.5f}")

print(f"XGBoost CV RMSLE: {np.mean(xgb_scores):.5f} (+/- {np.std(xgb_scores)*2:.5f})")

# Model 2: LightGBM
print("\nTraining LightGBM...")
lgb_params = {
    'objective': 'regression',
    'metric': 'None',  # We'll use custom RMSLE
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 100,
    'max_depth': 8,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_estimators': 2000
}

lgb_scores = []
lgb_predictions = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        eval_metric=lambda y_true, y_pred: ('rmsle', rmsle(y_true, y_pred), False),
        # early_stopping_rounds=100,
        # verbose=False
    )
    
    val_pred = model.predict(X_val_fold)
    score = rmsle(y_val_fold, val_pred)
    lgb_scores.append(score)
    
    lgb_predictions += model.predict(X_test) / 5
    print(f"Fold {fold+1} RMSLE: {score:.5f}")

print(f"LightGBM CV RMSLE: {np.mean(lgb_scores):.5f} (+/- {np.std(lgb_scores)*2:.5f})")

# Model 3: CatBoost
print("\nTraining CatBoost...")
cat_params = {
    'loss_function': 'RMSE',  # Will optimize manually for RMSLE
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 10,
    'random_seed': 42,
    'iterations': 2000,
    'verbose': False
}

cat_scores = []
cat_predictions = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostRegressor(**cat_params)
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        # early_stopping_rounds=100,
        # verbose=False
    )
    
    val_pred = model.predict(X_val_fold)
    score = rmsle(y_val_fold, val_pred)
    cat_scores.append(score)
    
    cat_predictions += model.predict(X_test) / 5
    print(f"Fold {fold+1} RMSLE: {score:.5f}")

print(f"CatBoost CV RMSLE: {np.mean(cat_scores):.5f} (+/- {np.std(cat_scores)*2:.5f})")


# Ensemble (weighted average based on CV performance)
xgb_weight = 1 / np.mean(xgb_scores)
lgb_weight = 1 / np.mean(lgb_scores)
cat_weight = 1 / np.mean(cat_scores)

total_weight = xgb_weight + lgb_weight + cat_weight
xgb_weight /= total_weight
lgb_weight /= total_weight
cat_weight /= total_weight

ensemble_predictions = (xgb_weight * xgb_predictions + 
                       lgb_weight * lgb_predictions + 
                       cat_weight * cat_predictions)

print(f"\nEnsemble weights: XGB={xgb_weight:.3f}, LGB={lgb_weight:.3f}, CAT={cat_weight:.3f}")

# Create submission with positive values
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': np.maximum(ensemble_predictions, 0.001)  # Ensure positive predictions
})

submission.to_csv('calories_prediction_ensemble.csv', index=False)
print(f"\nSubmission created! Predicted calories range: {submission['Calories'].min():.2f} - {submission['Calories'].max():.2f}")

# Additional: Single best model submission
best_model_name = ['XGBoost', 'LightGBM', 'CatBoost'][np.argmin([np.mean(xgb_scores), np.mean(lgb_scores), np.mean(cat_scores)])]
best_predictions = [xgb_predictions, lgb_predictions, cat_predictions][np.argmin([np.mean(xgb_scores), np.mean(lgb_scores), np.mean(cat_scores)])]

submission_single = pd.DataFrame({
    'id': test['id'],
    'Calories': np.maximum(best_predictions, 0.001)  # Ensure positive predictions
})

submission_single.to_csv(f'calories_prediction_{best_model_name.lower()}.csv', index=False)
print(f"Best single model: {best_model_name} with CV RMSLE: {min(np.mean(xgb_scores), np.mean(lgb_scores), np.mean(cat_scores)):.5f}")

