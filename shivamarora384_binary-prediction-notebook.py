import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from scipy.optimize import minimize

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


# Feature engineering (same as before)
def create_features(df):
    # Interaction features
    df['temp_humidity'] = df['temparature'] * df['humidity']
    df['pressure_cloud'] = df['pressure'] * df['cloud']
    df['temp_pressure_ratio'] = df['temparature'] / (df['pressure'] + 1e-5)
    
    # Polynomial features
    df['temp_squared'] = df['temparature'] ** 2
    df['humidity_squared'] = df['humidity'] ** 2
    
    # Additional interactions
    df['temp_wind'] = df['temparature'] * df['windspeed']
    df['humidity_pressure'] = df['humidity'] * df['pressure']
    
    return df.drop(columns=['id'])

# Process data
X = create_features(train)
y = X.pop('rainfall')
X_test = create_features(test)

# Cross-validation setup
FOLDS = 7
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'colsample_bytree': 0.9,
    'subsample': 0.9,
    'learning_rate': 0.05,
    'eval_metric': 'auc',
    'seed': 42,
    'verbosity': 0
}

# Training loop for XGBoost
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        evals=watchlist,
        early_stopping_rounds=100,
        verbose_eval=500
    )
    
    # Record OOF predictions and accumulate test predictions
    if hasattr(model, 'best_ntree_limit'):
        best_iteration = model.best_ntree_limit
    else:
        best_iteration = None
    
    oof_preds[val_idx] = model.predict(dval, ntree_limit=best_iteration) if best_iteration is not None else model.predict(dval)
    test_preds += (model.predict(dtest, ntree_limit=best_iteration) if best_iteration is not None else model.predict(dtest)) / FOLDS

    
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold} AUC: {fold_auc:.4f}")

final_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall OOF AUC: {final_auc:.4f}")

# -------------------------------------------
# Ensemble Section: Finding Optimal Ensemble Weights
# -------------------------------------------
# Here we assume you have another model's OOF and test predictions.
# For demonstration, we simulate these arrays by adding a bit of noise
# to the XGBoost predictions. Replace these with your actual predictions.
other_oof = oof_preds + np.random.normal(0, 0.01, size=oof_preds.shape)
other_test_preds = test_preds + np.random.normal(0, 0.01, size=test_preds.shape)

# Function to optimize ensemble weight for maximum AUC
def ensemble_auc(weight):
    # weight for XGBoost model; (1-weight) for the other model
    ensemble_pred = weight * oof_preds + (1 - weight) * other_oof
    return -roc_auc_score(y, ensemble_pred)  # negative since we maximize AUC

# Optimize weight starting from 0.5
result = minimize(ensemble_auc, x0=0.5, bounds=[(0,1)], method='L-BFGS-B')
optimal_weight = result.x[0]
print(f"Optimal ensemble weight for XGBoost: {optimal_weight:.4f}")

# Generate ensemble test predictions using the optimized weight
ensemble_test_preds = optimal_weight * test_preds + (1 - optimal_weight) * other_test_preds

# -------------------------------------------
# Submission
# -------------------------------------------
sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sub['rainfall'] = ensemble_test_preds
sub.to_csv("submission.csv", index=False)
print("Submission created!")

