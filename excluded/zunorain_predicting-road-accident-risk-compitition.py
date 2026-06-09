import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
pd.set_option('display.max_columns', None)

print("Libraries loaded successfully!")


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTrain columns: {train.columns.tolist()}")
print(f"\nFirst few rows:")
train.head()


print("Data Info:")
print(train.info())
print("\nMissing Values:")
print(train.isnull().sum())
print("\nTarget Statistics:")
print(train['accident_risk'].describe())


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(train['accident_risk'], bins=50, edgecolor='black')
plt.title('Distribution of Accident Risk')
plt.xlabel('Accident Risk')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.boxplot(train['accident_risk'])
plt.title('Accident Risk Box Plot')
plt.ylabel('Accident Risk')
plt.tight_layout()
plt.show()


def advanced_feature_engineering(df, is_train=True):
    df = df.copy()
    
    if is_train:
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        if 'accident_risk' in df.columns:
            target = df['accident_risk']
            df = df.drop('accident_risk', axis=1)
    else:
        test_ids = df['id'].copy() if 'id' in df.columns else None
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    if len(numeric_cols) > 1:
        for i, col1 in enumerate(numeric_cols[:5]):
            for col2 in numeric_cols[i+1:6]:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-5)
                df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
    
    for col in numeric_cols[:10]:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
        df[f'{col}_log'] = np.log1p(np.abs(df[col]))
    
    if is_train:
        return df, target
    else:
        return df, test_ids

print("Feature engineering function created!")


X_train, y_train = advanced_feature_engineering(train, is_train=True)
X_test, test_ids = advanced_feature_engineering(test, is_train=False)

print(f"\nAfter feature engineering:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")


n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))

predictions_xgb = np.zeros(len(X_test))
predictions_lgb = np.zeros(len(X_test))
predictions_cat = np.zeros(len(X_test))

print(f"Starting {n_folds}-Fold Cross-Validation...")


xgb_params = {
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
}

xgb_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"\nXGBoost - Fold {fold}/{n_folds}")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    oof_xgb[val_idx] = model.predict(X_val)
    predictions_xgb += model.predict(X_test) / n_folds
    
    fold_score = np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx]))
    xgb_scores.append(fold_score)
    print(f"Fold {fold} RMSE: {fold_score:.6f}")

xgb_oof_score = np.sqrt(mean_squared_error(y_train, oof_xgb))
print(f"\n{'='*50}")
print(f"XGBoost OOF RMSE: {xgb_oof_score:.6f}")
print(f"XGBoost Mean CV RMSE: {np.mean(xgb_scores):.6f} ± {np.std(xgb_scores):.6f}")
print(f"{'='*50}")


lgb_params = {
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'regression',
    'metric': 'rmse',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

lgb_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"\nLightGBM - Fold {fold}/{n_folds}")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    
    oof_lgb[val_idx] = model.predict(X_val)
    predictions_lgb += model.predict(X_test) / n_folds
    
    fold_score = np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx]))
    lgb_scores.append(fold_score)
    print(f"Fold {fold} RMSE: {fold_score:.6f}")

lgb_oof_score = np.sqrt(mean_squared_error(y_train, oof_lgb))
print(f"\n{'='*50}")
print(f"LightGBM OOF RMSE: {lgb_oof_score:.6f}")
print(f"LightGBM Mean CV RMSE: {np.mean(lgb_scores):.6f} ± {np.std(lgb_scores):.6f}")
print(f"{'='*50}")


cat_params = {
    'iterations': 3000,
    'learning_rate': 0.01,
    'depth': 7,
    'l2_leaf_reg': 3,
    'subsample': 0.8,
    'colsample_bylevel': 0.8,
    'random_strength': 0.5,
    'bagging_temperature': 0.2,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': False,
    'task_type': 'CPU',
    'thread_count': -1
}

cat_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"\nCatBoost - Fold {fold}/{n_folds}")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = CatBoostRegressor(**cat_params)
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=False
    )
    
    oof_cat[val_idx] = model.predict(X_val)
    predictions_cat += model.predict(X_test) / n_folds
    
    fold_score = np.sqrt(mean_squared_error(y_val, oof_cat[val_idx]))
    cat_scores.append(fold_score)
    print(f"Fold {fold} RMSE: {fold_score:.6f}")

cat_oof_score = np.sqrt(mean_squared_error(y_train, oof_cat))
print(f"\n{'='*50}")
print(f"CatBoost OOF RMSE: {cat_oof_score:.6f}")
print(f"CatBoost Mean CV RMSE: {np.mean(cat_scores):.6f} ± {np.std(cat_scores):.6f}")
print(f"{'='*50}")


print("\n" + "="*60)
print("INDIVIDUAL MODEL PERFORMANCE")
print("="*60)
print(f"XGBoost OOF RMSE:   {xgb_oof_score:.6f}")
print(f"LightGBM OOF RMSE:  {lgb_oof_score:.6f}")
print(f"CatBoost OOF RMSE:  {cat_oof_score:.6f}")
print("="*60)


from scipy.optimize import minimize

def ensemble_rmse(weights, *args):
    y_true, predictions = args
    weighted_pred = sum(w * p for w, p in zip(weights, predictions))
    return np.sqrt(mean_squared_error(y_true, weighted_pred))

initial_weights = [1/3, 1/3, 1/3]
bounds = [(0, 1)] * 3
constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}

result = minimize(
    ensemble_rmse,
    initial_weights,
    args=(y_train, [oof_xgb, oof_lgb, oof_cat]),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x
print(f"\nOptimal Weights:")
print(f"XGBoost:  {optimal_weights[0]:.4f}")
print(f"LightGBM: {optimal_weights[1]:.4f}")
print(f"CatBoost: {optimal_weights[2]:.4f}")

oof_ensemble = (
    optimal_weights[0] * oof_xgb +
    optimal_weights[1] * oof_lgb +
    optimal_weights[2] * oof_cat
)

ensemble_oof_score = np.sqrt(mean_squared_error(y_train, oof_ensemble))

print(f"\n{'='*60}")
print(f"OPTIMIZED ENSEMBLE OOF RMSE: {ensemble_oof_score:.6f}")
print(f"{'='*60}")


final_predictions = (
    optimal_weights[0] * predictions_xgb +
    optimal_weights[1] * predictions_lgb +
    optimal_weights[2] * predictions_cat
)

final_predictions = np.clip(final_predictions, 0, 1)

print(f"Prediction Statistics:")
print(f"Min: {final_predictions.min():.6f}")
print(f"Max: {final_predictions.max():.6f}")
print(f"Mean: {final_predictions.mean():.6f}")
print(f"Std: {final_predictions.std():.6f}")


fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].scatter(y_train, oof_ensemble, alpha=0.3)
axes[0, 0].plot([0, 1], [0, 1], 'r--', lw=2)
axes[0, 0].set_xlabel('True Values')
axes[0, 0].set_ylabel('Predictions')
axes[0, 0].set_title(f'Ensemble Predictions vs True Values\nOOF RMSE: {ensemble_oof_score:.6f}')
axes[0, 0].grid(True, alpha=0.3)

residuals = y_train - oof_ensemble
axes[0, 1].hist(residuals, bins=50, edgecolor='black')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Residuals')
axes[0, 1].grid(True, alpha=0.3)

model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'Ensemble']
scores = [xgb_oof_score, lgb_oof_score, cat_oof_score, ensemble_oof_score]
colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4']
axes[1, 0].bar(model_names, scores, color=colors, edgecolor='black')
axes[1, 0].set_ylabel('RMSE')
axes[1, 0].set_title('Model Comparison')
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(scores):
    axes[1, 0].text(i, v + 0.0001, f'{v:.6f}', ha='center', va='bottom')

axes[1, 1].hist(final_predictions, bins=50, edgecolor='black', alpha=0.7, label='Test Predictions')
axes[1, 1].hist(y_train, bins=50, edgecolor='black', alpha=0.5, label='Train Target')
axes[1, 1].set_xlabel('Accident Risk')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution: Test Predictions vs Train Target')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': final_predictions
})

submission.to_csv('submission.csv', index=False)

print("\n" + "="*60)
print("SUBMISSION FILE CREATED: submission.csv")
print("="*60)
print(f"\nSubmission shape: {submission.shape}")
print(f"\nFirst 10 rows:")
print(submission.head(10))
print(f"\nLast 10 rows:")
print(submission.tail(10))

print(f"\n{'='*60}")
print("FINAL MODEL SUMMARY")
print(f"{'='*60}")
print(f"Total Features Used: {X_train.shape[1]}")
print(f"Cross-Validation Folds: {n_folds}")
print(f"\nModel Performance (OOF RMSE):")
print(f"  XGBoost:  {xgb_oof_score:.6f}")
print(f"  LightGBM: {lgb_oof_score:.6f}")
print(f"  CatBoost: {cat_oof_score:.6f}")
print(f"  Ensemble: {ensemble_oof_score:.6f} ⭐")
print(f"\nEnsemble Weights:")
print(f"  XGBoost:  {optimal_weights[0]:.4f}")
print(f"  LightGBM: {optimal_weights[1]:.4f}")
print(f"  CatBoost: {optimal_weights[2]:.4f}")
print(f"{'='*60}")
print("\n✅ Ready to submit to Kaggle!")




