import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train.head()


train.info()


train.isnull().sum()


cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']


for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])


train['Compartments'] = train['Compartments'].astype(int)
test['Compartments'] = test['Compartments'].astype(int)


train['Waterproof'] = train['Waterproof'].map({'Yes': 1, 'No': 0})
train['Laptop Compartment'] = train['Laptop Compartment'].map({'Yes': 1, 'No': 0})

size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
train['Size'] = train['Size'].map(size_mapping)


test['Waterproof'] = test['Waterproof'].map({'Yes': 1, 'No': 0})
test['Laptop Compartment'] = test['Laptop Compartment'].map({'Yes': 1, 'No': 0})

size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
test['Size'] = test['Size'].map(size_mapping)


cat_col = ['Brand', 'Material','Style', 'Color']


le = LabelEncoder()

for col in cat_col:
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])


train.head()


X = train.drop(columns=['Price'])
y = train['Price']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def objective_xg(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 5000),  # Increased from 300 to 5000
        "max_depth": trial.suggest_int("max_depth", 3, 15),  # Increased from 10 to 15
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.3),  # Wider range
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),  # Lower bound decreased
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),  # Increased range
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 0.001, 10),  # More flexible range
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 0.001, 10),
        "gamma": trial.suggest_loguniform("gamma", 0.001, 10),  # Added gamma tuning
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),  # Increased range
        "tree_method": "gpu_hist",  
        "predictor": "gpu_predictor"
    }

    model_xgb = xgb.XGBRegressor(objective="reg:squarederror", **params, random_state=42)
    
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  
        eval_metric="rmse",  
        early_stopping_rounds=50,  
        verbose=False
    )

    y_pred = model_xgb.predict(X_val)
    return mean_squared_error(y_val, y_pred, squared=False)


#study = optuna.create_study(direction="minimize") 
#study.optimize(objective_xg, n_trials=100)  

#print("Best Hyperparameters:", study.best_params)


best_params = {
    'n_estimators': 4026,
    'max_depth': 3,
    'learning_rate': 0.04788313490661605,
    'subsample': 0.3176293462903142,
    'colsample_bytree': 0.3997772883492928,
    'reg_alpha': 0.11236453231081674,
    'reg_lambda': 0.046172570438178684,
    'gamma': 0.9896560395098459,
    'min_child_weight': 20,
    'tree_method': 'gpu_hist',  
    'predictor': 'gpu_predictor',
    'objective': 'reg:squarederror',
    'random_state': 42
}

model = xgb.XGBRegressor(**best_params)
model.fit(X, y) 
# Model Performance as 38.8810


test.head()


submission_ids = test['id']
predictions = model.predict(test)


submission = pd.DataFrame({
    'id': submission_ids,
    'num_sold': predictions 
})


submission.to_csv('submission.csv', index=False)
print("File Saved!")
print(submission.head())

