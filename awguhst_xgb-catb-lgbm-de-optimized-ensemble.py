pip install tqdm


# General libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Machine learning libraries
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb

# Optimization library
from scipy.optimize import differential_evolution

# Ignore warnings
warnings.filterwarnings('ignore')


# Define base path 
PATH = "/kaggle/input/playground-series-s5e10/"

# Load the data
train = pd.read_csv(PATH + "train.csv")
test = pd.read_csv(PATH + "test.csv")

# Preview the datasets
train.head()


# Separate features and target
X = train.drop(columns=["accident_risk", "id"])  
y = train["accident_risk"]

# Identify categorical columns
categorical_columns = X.select_dtypes(include=["object"]).columns

# Initialize OrdinalEncoder
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Apply Ordinal Encoding to categorical columns 
X_encoded = X.copy()
X_encoded[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])

# Apply the same encoding to the test data 
X_test_encoded = test.drop(columns=["id"])  
X_test_encoded[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# Feature scaling (StandardScaler)
scaler = StandardScaler()
X_encoded_scaled = scaler.fit_transform(X_encoded)
X_test_scaled = scaler.transform(X_test_encoded) 


# Initialize model hyperparameters
params_xgb = {
    'learning_rate': 0.1335212800698337,
    'n_estimators': 581,
    'max_depth': 7,
    'subsample': 0.5901039939725466,
    'colsample_bytree': 0.9727141365907552,
    'reg_alpha': 1.3196160166681845,
    'reg_lambda': 8.181830748649771,
    'min_child_weight': 9,
    'gamma': 0.007156830970017944,
    'max_delta_step': 10
}

params_lgbm = {
    'n_estimators': 1000,           
    'learning_rate': 0.05,           
    'max_depth': 7,                
    'num_leaves': 31,               
    'min_child_samples': 20,         
    'subsample': 0.8,               
    'colsample_bytree': 0.8,        
    'reg_alpha': 0.1,                
    'reg_lambda': 0.1           
}

params_catboost = {
    'iterations': 1000,             
    'learning_rate': 0.05,          
    'depth': 7,                     
    'l2_leaf_reg': 3            
}

# Define the models with optimized hyperparameters
model_xgb = xgb.XGBRegressor(**params_xgb)
model_catboost = CatBoostRegressor(**params_catboost, silent=True)
model_lgbm = lgb.LGBMRegressor(**params_lgbm)

def ensemble_loss_oof(weights, X_train, y_train, X_test, y_test, models_preds_oof):
    # Normalize the weights to sum to 1
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Weighted average of OOF predictions
    ensemble_preds = (weights[0] * models_preds_oof['xgb'] +
                      weights[1] * models_preds_oof['catboost'] +
                      weights[2] * models_preds_oof['lgbm'])
    
    # Calculate MSE loss on OOF predictions
    return mean_squared_error(y_test, ensemble_preds)

def optimize_ensemble_oof(X, y, cv_splits=5):
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    models_preds_oof = {'xgb': [], 'catboost': [], 'lgbm': []}
    y_oof = []

    # Store cross-validation results
    for train_index, test_index in tqdm(kf.split(X), desc='Cross-validation folds', total=kf.get_n_splits(), ncols=100, leave=False):
        # Split the data into train/test sets for the current fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train models on the current fold
        model_xgb.fit(X_train, y_train)
        model_catboost.fit(X_train, y_train)
        model_lgbm.fit(X_train, y_train)
        
        # Generate predictions for the test set
        models_preds_oof['xgb'].append(model_xgb.predict(X_test))
        models_preds_oof['catboost'].append(model_catboost.predict(X_test))
        models_preds_oof['lgbm'].append(model_lgbm.predict(X_test))
        y_oof.append(y_test)

    # Concatenate OOF predictions across all folds
    for model_name in models_preds_oof:
        models_preds_oof[model_name] = np.concatenate(models_preds_oof[model_name])
    y_oof = np.concatenate(y_oof)

    # DE Optimization for the OOF predictions
    bounds = [(0, 1), (0, 1), (0, 1)]  
    result = differential_evolution(
        ensemble_loss_oof, bounds, args=(X, y, X, y_oof, models_preds_oof),
        strategy='currenttobest1exp', maxiter=1000, popsize=60, mutation=(0.5, 1), recombination=0.8
    )
    
    optimal_weights = result.x / np.sum(result.x)  
    return optimal_weights

# Apply the optimization using DE
de_weights = optimize_ensemble_oof(X_encoded_scaled, y, cv_splits=5)

print("Optimized Weights for OOF: ", de_weights)


# Train models on the training set
model_xgb.fit(X_encoded_scaled, y) 
model_catboost.fit(X_encoded_scaled, y) 
model_lgbm.fit(X_encoded_scaled, y)  

# Generate predictions for the test set 
preds_xgb_test = model_xgb.predict(X_test_scaled)
preds_catboost_test = model_catboost.predict(X_test_scaled)
preds_lgbm_test = model_lgbm.predict(X_test_scaled)

# Create a weighted average of the predictions using the optimized weights
ensemble_preds_test = (
    de_weights[0] * preds_xgb_test + 
    de_weights[1] * preds_catboost_test + 
    de_weights[2] * preds_lgbm_test
)

# Create submission
submission_df = pd.DataFrame({
    'id': test['id'],  
    'prediction': ensemble_preds_test
})

# Save the predictions to a CSV file
submission_df.to_csv('submission.csv', index=False)

