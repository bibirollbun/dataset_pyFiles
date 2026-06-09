# =======================
# Calorie Prediction - Feature Importance & Ridge Stacking
# =======================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter('ignore')


# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


# Feature Engineering (basic BMI and interaction features)
train['BMI'] = train['Weight'] / (train['Height']/100)**2
test['BMI'] = test['Weight'] / (test['Height']/100)**2

train['Intensity'] = train['Heart_Rate'] * train['Duration']
test['Intensity'] = test['Heart_Rate'] * test['Duration']


# Encode Sex
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])


# Prepare X and y
features = [col for col in train.columns if col not in ['id', 'Calories']]
X = train[features]
y = np.log1p(train['Calories'])
X_test = test[features]


# Define models
models = {
    'CatBoost': CatBoostRegressor(verbose=0, random_seed=42, cat_features=['Sex'], early_stopping_rounds=100),
    'XGBoost': XGBRegressor(max_depth=10, colsample_bytree=0.7, subsample=0.9, n_estimators=1000, learning_rate=0.03,
                            gamma=0.01, max_delta_step=2, early_stopping_rounds=50, eval_metric='rmse',
                            enable_categorical=True, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=1000, learning_rate=0.03, max_depth=10, colsample_bytree=0.7,
                              subsample=0.9, random_state=42, verbose=-1)
}


# K-Fold
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
results = {name: {'oof': np.zeros(len(X)), 'pred': np.zeros(len(X_test)), 'rmsle': []} for name in models}

for name, model in models.items():
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        if name == 'XGBoost':
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
        elif name == 'CatBoost':
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            model.fit(X_train, y_train)
        
        oof_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        results[name]['oof'][val_idx] = oof_pred
        results[name]['pred'] += test_pred / FOLDS
        rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(oof_pred)))
        results[name]['rmsle'].append(rmsle)


# Ridge Stacking
stack_X = pd.DataFrame({name: results[name]['oof'] for name in models})
stack_test = pd.DataFrame({name: results[name]['pred'] for name in models})

ridge = Ridge(alpha=1.0)
ridge.fit(stack_X, y)
stacked_preds = ridge.predict(stack_test)
final_preds = np.clip(np.expm1(stacked_preds), 1, 314)

submission['Calories'] = final_preds
submission.to_csv("submission_stacked_ridge.csv", index=False)


# Feature Importance Visualization
best_model_name = min(results, key=lambda x: np.mean(results[x]['rmsle']))
best_model = models[best_model_name]

if best_model_name == 'CatBoost':
    importances = best_model.get_feature_importance()
elif best_model_name == 'XGBoost':
    importances = best_model.feature_importances_
elif best_model_name == 'LightGBM':
    importances = best_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)


# Barplot
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(15), x='Importance', y='Feature')
plt.title(f'{best_model_name} Feature Importance')
plt.tight_layout()
plt.show()

