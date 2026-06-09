
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor

import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

SEED = 42
np.random.seed(SEED)




# Load training and test data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print('Training data shape:', train_df.shape)
print('Test data shape:', test_df.shape)

# Separate target variable
y = train_df['accident_risk']
X_raw = train_df.drop(columns=['accident_risk'])




# Display the first few rows
train_df.head()




# Summary statistics for numeric features
numeric_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents', 'accident_risk']
train_df[numeric_cols].describe()




# Plot distribution of the target variable
plt.figure(figsize=(8, 5))
sns.histplot(train_df['accident_risk'], bins=50, kde=True, color='dodgerblue')
plt.title('Distribution of accident_risk')
plt.xlabel('accident_risk')
plt.ylabel('Count')
plt.show()




# Bar plots for categorical variables
categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for idx, col in enumerate(categorical_cols):
    sns.countplot(x=train_df[col], ax=axes[idx], palette='Set2')
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Count')
    axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()




# Compute correlations (convert booleans to int and categories to label codes)
eda_df = train_df.copy()
for col in ['road_signs_present','public_road','holiday','school_season']:
    eda_df[col] = eda_df[col].astype(int)

for col in ['road_type','lighting','weather','time_of_day']:
    le = LabelEncoder()
    eda_df[col] = le.fit_transform(eda_df[col])

corr = eda_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation matrix')
plt.show()




# Function to create engineered features
def engineer_features(df):
    data = df.copy()
    # Convert boolean flags to integers
    for col in ['road_signs_present','public_road','holiday','school_season']:
        data[col] = data[col].astype(int)
    # Label encode categorical variables
    encoders = {}
    for col in ['road_type','lighting','weather','time_of_day']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le
    # Interaction and polynomial features
    data['curvature_speed'] = data['curvature'] * data['speed_limit']
    data['accidents_per_lane'] = data['num_reported_accidents'] / (data['num_lanes'] + 1)
    data['speed_per_lane'] = data['speed_limit'] / (data['num_lanes'] + 1)
    data['curvature_squared'] = data['curvature'] ** 2
    data['speed_squared'] = data['speed_limit'] ** 2
    data['weather_lighting'] = data['weather'] * 3 + data['lighting']
    return data

# Apply feature engineering to train and test
X_eng = engineer_features(X_raw)
X_test_eng = engineer_features(test_df)

# Remove low‑correlation features (<0.02 absolute correlation)
full_tmp = X_eng.copy()
full_tmp['target'] = y
correlations = full_tmp.corr()['target'].abs().sort_values()
low_corr_cols = [col for col in correlations.index if col != 'target' and correlations[col] < 0.02]
print('Removing low‑correlation features:', low_corr_cols)
X_eng_reduced = X_eng.drop(columns=low_corr_cols)
X_test_eng_reduced = X_test_eng.drop(columns=low_corr_cols)

print('Shape after feature engineering and reduction:', X_eng_reduced.shape)




# LightGBM parameter grid
lgb_param_grid = [
    {'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8},
    {'num_leaves': 127, 'learning_rate': 0.03, 'feature_fraction': 0.9, 'bagging_fraction': 0.8}
]

kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
lgb_results = []

for i, params in enumerate(lgb_param_grid):
    print(f'LightGBM parameter set {i+1}:', params)
    oof = np.zeros(len(X_eng_reduced))
    test_pred = np.zeros(len(X_test_eng_reduced))
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_eng_reduced)):
        X_tr, X_val = X_eng_reduced.iloc[train_idx], X_eng_reduced.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Convert categorical columns to 'category'
        X_tr_cat = X_tr.copy(); X_val_cat = X_val.copy(); X_test_cat = X_test_eng_reduced.copy()
        for col in ['road_type','lighting','weather','time_of_day','weather_lighting']:
            if col in X_tr_cat.columns:
                X_tr_cat[col] = X_tr_cat[col].astype('category')
                X_val_cat[col] = X_val_cat[col].astype('category')
                X_test_cat[col] = X_test_cat[col].astype('category')

        train_set = lgb.Dataset(X_tr_cat, y_tr, categorical_feature=[c for c in ['road_type','lighting','weather','time_of_day','weather_lighting'] if c in X_tr_cat.columns])
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': params['learning_rate'],
            'num_leaves': params['num_leaves'],
            'feature_fraction': params['feature_fraction'],
            'bagging_fraction': params['bagging_fraction'],
            'bagging_freq': 5,
            'seed': SEED,
            'verbosity': -1
        }
        model = lgb.train(lgb_params, train_set, num_boost_round=500)
        oof[val_idx] = model.predict(X_val_cat)
        test_pred += model.predict(X_test_cat) / kf.n_splits

        rmse = mean_squared_error(y_val, oof[val_idx], squared=False)
        fold_scores.append(rmse)
        print(f'  Fold {fold+1} RMSE: {rmse:.6f}')
    overall_rmse = mean_squared_error(y, oof, squared=False)
    lgb_results.append({'params': params, 'oof_rmse': overall_rmse, 'fold_scores': fold_scores, 'test_pred': test_pred})
    print(f'Overall OOF RMSE for this set: {overall_rmse:.6f}')

# Select best LightGBM configuration
best_lgb = min(lgb_results, key=lambda x: x['oof_rmse'])
print('Best LightGBM configuration:', best_lgb['params'])
print('Best LightGBM OOF RMSE:', best_lgb['oof_rmse'])




# XGBoost parameter grid
xgb_param_grid = [
    {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 400},
    {'max_depth': 8, 'learning_rate': 0.03, 'n_estimators': 600}
]

kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
xgb_results = []

for i, params in enumerate(xgb_param_grid):
    print(f'XGBoost parameter set {i+1}:', params)
    oof = np.zeros(len(X_eng_reduced))
    test_pred = np.zeros(len(X_test_eng_reduced))
    fold_scores = []

    # One-hot encode categorical variables
    X_train_xgb = pd.get_dummies(X_eng_reduced, columns=[c for c in ['road_type','lighting','weather','time_of_day','weather_lighting'] if c in X_eng_reduced.columns], drop_first=False)
    X_test_xgb = pd.get_dummies(X_test_eng_reduced, columns=[c for c in ['road_type','lighting','weather','time_of_day','weather_lighting'] if c in X_test_eng_reduced.columns], drop_first=False)
    X_test_xgb = X_test_xgb.reindex(columns=X_train_xgb.columns, fill_value=0)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_xgb)):
        X_tr, X_val = X_train_xgb.iloc[train_idx], X_train_xgb.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            eval_metric='rmse',
            tree_method='hist',
            random_state=SEED
        )
        model.fit(X_tr, y_tr)

        oof[val_idx] = model.predict(X_val)
        test_pred += model.predict(X_test_xgb) / kf.n_splits

        rmse = mean_squared_error(y_val, oof[val_idx], squared=False)
        fold_scores.append(rmse)
        print(f'  Fold {fold+1} RMSE: {rmse:.6f}')
    overall_rmse = mean_squared_error(y, oof, squared=False)
    xgb_results.append({'params': params, 'oof_rmse': overall_rmse, 'fold_scores': fold_scores, 'test_pred': test_pred})
    print(f'Overall OOF RMSE for this set: {overall_rmse:.6f}')

# Select best XGBoost configuration
best_xgb = min(xgb_results, key=lambda x: x['oof_rmse'])
print('Best XGBoost configuration:', best_xgb['params'])
print('Best XGBoost OOF RMSE:', best_xgb['oof_rmse'])




# Train CatBoost on engineered features (single configuration)
kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
cat_oof = np.zeros(len(X_eng_reduced))
cat_test_pred = np.zeros(len(X_test_eng_reduced))
cat_fold_scores = []

# Determine categorical feature indices
cat_indices = [X_eng_reduced.columns.get_loc(c) for c in ['road_type','lighting','weather','time_of_day','weather_lighting'] if c in X_eng_reduced.columns]

for fold, (train_idx, val_idx) in enumerate(kf.split(X_eng_reduced)):
    X_tr, X_val = X_eng_reduced.iloc[train_idx], X_eng_reduced.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=10,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=SEED,
        verbose=False
    )
    model.fit(X_tr, y_tr, cat_features=cat_indices, eval_set=(X_val, y_val))

    cat_oof[val_idx] = model.predict(X_val)
    cat_test_pred += model.predict(X_test_eng_reduced) / kf.n_splits

    rmse = mean_squared_error(y_val, cat_oof[val_idx], squared=False)
    cat_fold_scores.append(rmse)
    print(f'CatBoost fold {fold+1} RMSE: {rmse:.6f}')

cat_oof_rmse = mean_squared_error(y, cat_oof, squared=False)
print('CatBoost OOF RMSE:', cat_oof_rmse)




# Neural network (MLP) – prepare data
# One-hot encode all categorical features
X_mlp = pd.get_dummies(X_eng_reduced, drop_first=False)
X_test_mlp = pd.get_dummies(X_test_eng_reduced, drop_first=False)
X_test_mlp = X_test_mlp.reindex(columns=X_mlp.columns, fill_value=0)

# Split into training and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(X_mlp, y, test_size=0.2, random_state=SEED)

# Scale features for MLP
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_mlp)

# Define a deeper MLP model
mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam',
                   learning_rate_init=0.001, max_iter=40, random_state=SEED, verbose=True)

mlp.fit(X_tr_scaled, y_tr)

# Compute validation RMSE
val_pred = mlp.predict(X_val_scaled)
mlp_rmse = mean_squared_error(y_val, val_pred, squared=False)
print('MLP validation RMSE:', mlp_rmse)

# Predict on full training set to get OOF (approximation)
mlp_oof = mlp.predict(scaler.transform(X_mlp))
mlp_oof_rmse = mean_squared_error(y, mlp_oof, squared=False)
print('MLP OOF RMSE:', mlp_oof_rmse)

# Predict on test set
mlp_test_pred = mlp.predict(X_test_scaled)




# Collect all best predictions
best_lgb_pred = best_lgb['test_pred']
best_xgb_pred = best_xgb['test_pred']

# Ensemble: average of best LGBM, best XGB and CatBoost predictions
ensemble_test_pred = (best_lgb_pred + best_xgb_pred + cat_test_pred) / 3
ensemble_test_pred = np.clip(ensemble_test_pred, 0, 1)

# Create a summary DataFrame with OOF scores
results = pd.DataFrame({
    'Model': ['LightGBM_best', 'XGBoost_best', 'CatBoost', 'MLP'],
    'OOF_RMSE': [best_lgb['oof_rmse'], best_xgb['oof_rmse'], cat_oof_rmse, mlp_oof_rmse]
})
print('Summary of out-of-fold RMSE (lower is better):')
print(results)

# Save submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'accident_risk': ensemble_test_pred
})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print('Saved submission.csv')
submission.head()


