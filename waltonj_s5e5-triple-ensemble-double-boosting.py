# Import Appropriate Libraries
# For your reading pleasure in response to sns
import warnings

# Data Manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Graphing
import matplotlib.pyplot as plt # general graphing
import seaborn as sns # pretty & easy graphing based off plt

# Feature Engineering
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures, StandardScaler

# Model
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV, cross_val_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor

# Weight Optimization
from sklearn.metrics import mean_squared_log_error
from scipy.optimize import minimize

# Training time metrics
import time

# Beautifying Kaggle Notebooks
warnings.filterwarnings("ignore")


# Data input
dir_name = '/kaggle/input/playground-series-s5e5/'
train_file = dir_name +  'train.csv'
test_file = dir_name +  'test.csv'

# Instantiating Dataframes
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

# Quick look
display(train.head(1), test.head(1))


train.select_dtypes(include='number').drop(columns=['id', 'Calories']).columns


# Feature Engineering
def create_features(df):
    # Creating static variables
    df['Sex'] = df['Sex'].replace({'male': 1, 'female': 0})
    df['BMI'] = df['Weight'] / np.square(df['Height'] / 100)
    df['Is_female'] = 1 - df['Sex']
    df['Intensity'] = df['Heart_Rate'] / df['Duration']
    df['Height_Weight_Ratio'] = df['Weight'] / df['Height']

    # Creating robust cross-product features to have the model learn non-linear interactions
    features = df.columns
    if 'Calories' in features:
        num_features = df.select_dtypes(include='number').drop(columns=['id', 'Calories']).columns
    else:
        num_features = df.select_dtypes(include='number').drop(columns=['id']).columns
    for i in range(len(num_features)):
        for j in range(i + 1, len(num_features)):
            df[f'{num_features[i]}_x_{num_features[j]}'] = df[num_features[i]] * df[num_features[j]]

    # Creating categorical items for durations
    for duration in df['Duration'].unique():
        df[f'HR_Duration_{int(duration)}'] = np.where(df['Duration'] == duration, df['Heart_Rate'], 0)
        df[f'Temp_Duration_{int(duration)}'] = np.where(df['Duration'] == duration, df['Body_Temp'], 0)

    # Get individual items for age
    for age in df['Age'].unique():
        df[f'HR_Age_{int(age)}'] = np.where(df['Age'] == age, df['Heart_Rate'], 0)
        df[f'Temp_Age_{int(age)}'] = np.where(df['Age'] == age, df['Body_Temp'], 0)

    for feature_i in ['Duration', 'Heart_Rate', 'Body_Temp']:
        for feature_ii in ['Sex', 'Is_female']:
            df[f'{feature_i}_{feature_ii}'] = df[feature_i] * df[feature_ii]

    for col in ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']:
        for agg in ['min', 'mean', 'std', 'max']:
            agg_value = df.groupby('Sex')[col].agg(agg).rename(f'Sex{col}_{agg}')
            df = df.merge(agg_value, on='Sex', how='left')

    df.drop(columns=['Is_female'], inplace=True)

    return df

train = create_features(train)
test = create_features(test)

print(train.shape, test.shape)


# Instantiating X & y
X = train.drop(columns=['id','Calories'])
X_test = test.drop(columns=['id'])
# Order columns
X_test = X_test[X.columns]

y = np.log1p(train['Calories'])



params = {
    'cat_params': {
        'iterations': 3000,
        'learning_rate': 0.01,
        'loss_function': 'RMSE',
        'depth': 10,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 200,
        'cat_features': ['Sex'],
        'verbose': 0,
        'task_type': 'GPU'
    },

    'xgb_params': {
        'max_depth': 9,
        'colsample_bytree': 0.7,
        'subsample': 0.9,
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'gamma': 0.01,
        'max_delta_step': 2,
        'eval_metric': 'rmse',
        'enable_categorical': 0,
        'random_state': 42,
        'early_stopping_rounds': 100,
        'verbose': 0,
        'tree_method': 'gpu_hist'
    },

    'lgbm_params': {
            'n_estimators': 3000,
            'max_depth': 10,
            'learning_rate': 0.01,
            'num_leaves': 20,
            'colsample_bytree': 0.7,
            'subsample': 0.9,
            'device': 'gpu',
            'verbose': 0,
            'random_state': 42
        }
}

models = {
    'CatBoost': CatBoostRegressor(**params['cat_params']),
    'XGBoost': XGBRegressor(**params['xgb_params']),
    'LightGBM': LGBMRegressor(**params['lgbm_params'])
}


folds = 5 # Reusable code

# Instantiating KFold
kf = KFold(n_splits = folds, shuffle=True, random_state=42)

# Results or 
results = {
    name: {
        'oof': np.zeros(len(train)),
        'pred': np.zeros(len(test)),
        'rmsle': []
    } for name in models
}

# Going through model by model
for name, model in models.items():
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        assert(y>= 0).all()

        # Training time tracker
        start = time.time()

        # Fitting based on model
        if name == 'LightGBM':
            model.fit(X_train, y_train)
        elif name == 'XGBoost':
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        #Out-of-Fold prediction & final prediction
        oof_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        # Saving results
        results[name]['oof'][val_idx] = oof_pred
        results[name]['pred'] += test_pred / folds

        # Root Mean Square Error
        rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(oof_pred)))
        results[name]['rmsle'].append(rmsle) # acts as a list within the dict

        print(f'Fold {i + 1} - ({name}) RMSLE: {rmsle:.6f}')
        print(f'Training time: {time.time() - start:.2f} seconds (or {(time.time() - start) / 60:.2f} minutes)')

print('\n============ Model Comparison ============')
for name in models:
    mean_rmsle = np.mean(results[name]['rmsle'])
    std_rmsle = np.std(results[name]['rmsle'])
    print(f'{name} - Mean RMSLE: {mean_rmsle:.5f} ± {std_rmsle:.5f}')
        
        


# Create lists to attach results to for voting
oof_preds = {name: np.expm1(results[name]['oof']) for name in results}
test_preds = {name: np.expm1(results[name]['pred']) for name in results}

# Translate back to linear
y_true = np.expm1(y)

def weight_decider(weights):
    blended = (
        weights[0] * oof_preds['CatBoost'] + 
        weights[1] * oof_preds['XGBoost'] +
        weights[2] * oof_preds['LightGBM']
    )
    return np.sqrt(mean_squared_log_error(y_true, blended))

# Parameters
initial_weights = [0.35, 0.55, 0.10]
constraints = ({
    'type': 'eq',
    'fun': lambda w: 1 - sum(w)
})
bounds = [(0, 1)] * 3

# Minimization optimization
res = minimize(weight_decider, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints,
               options={'disp': True})
best_weights = res.x

# Fantastic
print(f'\nOptimized Weights:')
print(f"CatBoost = {best_weights[0]:.4f}")
print(f"XGBoost  = {best_weights[1]:.4f}")
print(f"LightGBM = {best_weights[2]:.4f}")


def weight_application(final_weights):
    f_blend = (
    final_weights[0] * test_preds['CatBoost'] + 
    final_weights[1] * test_preds['XGBoost'] +
    final_weights[2] * test_preds['LightGBM'])

    return f_blend

# Apply test predictions
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': weight_application(best_weights)
})

