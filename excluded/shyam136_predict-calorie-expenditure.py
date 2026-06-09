# Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

# Models & Tools
import lightgbm as lgb
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import VotingRegressor
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder


# ==================================
# ğŸ“ˆ Load and Prepare Data
# ==================================
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

train['Log_Calories'] = np.log1p(train['Calories'])
X = train.drop(columns=['id', 'Calories', 'Log_Calories'])
y = train['Log_Calories']
X_test = test.drop(columns=['id'])

# Label Encoding for object columns
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category').cat.codes
        X_test[col] = X_test[col].astype('category').cat.codes


# ==================================
# ğŸ”� Define LightGBM Cross-Validation Function
# ==================================
def lgb_cv(num_leaves, max_depth, learning_rate, subsample, colsample_bytree):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'num_leaves': int(num_leaves),
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        preds = np.maximum(0, np.expm1(preds))
        y_val_exp = np.expm1(y_val)
        rmse = np.sqrt(mean_squared_error(y_val_exp, preds))
        rmses.append(rmse)
    
    return np.mean(rmses)



# ==================================
# ğŸ”� Run Bayesian Optimization + Retraining Loop
# ==================================
from bayes_opt import BayesianOptimization

pbounds = {
    'num_leaves': (20, 150),
    'max_depth': (5, 30),
    'learning_rate': (0.01, 0.2),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1)
}

bo = BayesianOptimization(
    f=lgb_cv,
    pbounds=pbounds,
    random_state=42
)

bo.maximize(init_points=5, n_iter=25)

best_params = bo.max['params']
best_params['num_leaves'] = int(best_params['num_leaves'])
best_params['max_depth'] = int(best_params['max_depth'])

print("Best Parameters from Bayesian Optimization:")
print(best_params)


# ==================================
# ğŸ�† Retrain Final Model on Entire Data
# ==================================
final_lgb_train = lgb.Dataset(X, label=y)
final_model = lgb.train(
    best_params,
    final_lgb_train,
    num_boost_round=1000,
    callbacks=[lgb.log_evaluation(100)]
)

lgb_preds = final_model.predict(X_test)
lgb_preds = np.maximum(0, np.expm1(lgb_preds))

# Save final predictions to submission
submission['Calories'] = lgb_preds
submission.to_csv('submission.csv', index=False)
print("âœ… Final submission file created: submission.csv")


