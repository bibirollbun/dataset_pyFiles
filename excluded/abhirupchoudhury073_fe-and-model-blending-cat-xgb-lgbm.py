import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


import itertools
from sklearn.preprocessing import LabelEncoder


def add_feature_cross_terms(df, features):
    df = df.copy()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    return df

def add_statistical_features(df, features):
    df_new = df.copy()
    df_new["row_mean"] = df[features].mean(axis=1)
    df_new["row_std"] = df[features].std(axis=1)
    df_new["row_max"] = df[features].max(axis=1)
    df_new["row_min"] = df[features].min(axis=1)
    df_new["row_median"] = df[features].median(axis=1)
    return df_new

def add_interaction_features(df, features):
    df_new = df.copy()
    for f1, f2 in itertools.combinations(features, 2):
        df_new[f"{f1}_plus_{f2}"] = df_new[f1] + df_new[f2]
        df_new[f"{f1}_minus_{f2}"] = df_new[f1] - df_new[f2]
        df_new[f"{f2}_minus_{f1}"] = df_new[f2] - df_new[f1]
        df_new[f"{f1}_div_{f2}"] = df_new[f1] / (df_new[f2] + 1e-6)
        df_new[f"{f2}_div_{f1}"] = df_new[f2] / (df_new[f1] + 1e-6)
    return df_new


train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)

train = add_interaction_features(train, numerical_features)
test = add_interaction_features(test, numerical_features)

train = add_statistical_features(train, numerical_features)
test = add_statistical_features(test, numerical_features)


le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

train['Sex'] = train['Sex'].astype('category')
test['Sex'] = test['Sex'].astype('category')


X = train.drop(columns=['id', 'Calories'])
y_log = np.log1p(train['Calories'])
y = train['Calories']
X_test = test.drop(columns=['id'])


categorical_columns = ['Sex']
cat_features = [list(X.columns).index(col) for col in categorical_columns]
cat_features


FOLDS = 10
SEED = 40


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def rmsle(y_true,y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


cat_params = {
     'iterations': 2000,
     'learning_rate': 0.04702799520841546,
     'depth': 10,
     'l2_leaf_reg': 6.543117388416962,
     'random_strength': 18.987839255325074,
     'bagging_temperature': 0.004332917984131254,
     'border_count': 255,
     'task_type': 'GPU',
     'devices': '0:1',
     'cat_features': cat_features, #,<----------CHECK THIS
     'early_stopping_rounds': 100,
     'eval_metric': 'RMSE',
     'random_seed': SEED,
     'verbose': False,
    }

xgb_params = final_params = {
    'n_estimators': 1000, 
    'learning_rate': 0.02793372246452372, 
    'max_depth': 10, 
    'subsample': 0.9666864431101987, 
    'colsample_bytree': 0.7757830943581432, 
    'gamma': 0.001311174387642583, 
    'reg_alpha': 1.846055326249069, 
    'reg_lambda': 3.7492694725505253, 
    'enable_categorical': True, 
    'eval_metric': 'rmse',
    'device': 'cuda', 
    'tree_method': 'gpu_hist', 
    'predictor': 'gpu_predictor',
    'random_state': SEED,
    'verbosity' : 0
    }

lgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'max_depth': 9,
    'objective': 'rmse',
    'metric': 'rmse',
    'early_stopping_rounds': 100,
    'feature_fraction': 0.7,
    'num_leaves': 20,
    'eval_metric': 'rmse',
    'device': 'gpu',
    'verbosity': 0,
    'random_state': SEED
    }


kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
models = {
    'cat': CatBoostRegressor(**cat_params),
    'xgb': XGBRegressor(**xgb_params),
    'lgb': LGBMRegressor(**lgb_params)
}

results = {name: {'oof': np.zeros(len(train)), 'pred': np.zeros(len(test)), 'rmsle': []} for name in models}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    for i, (train_idx, valid_idx) in enumerate(kf.split(X)):
        print(f"\nFold {i+1}")
        x_train, y_train = X.iloc[train_idx], y_log[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y_log[valid_idx]
        
        if name == 'xgb':
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
        elif name == 'cat':
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], cat_features=cat_features)
        else:
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])

        # Get OOF logged predictions
        oof_pred_logged = model.predict(x_valid)
        # Inverse Log OOF
        oof_pred = np.expm1(oof_pred_logged)
        # Save the OOF predictions in array
        results[name]['oof'][valid_idx] = oof_pred
        
        # GET test logged prediction
        test_pred_logged = model.predict(X_test)
        # Inverse log test
        test_pred = np.expm1(test_pred_logged)
        # Save test predictions in array
        results[name]['pred'] += test_pred / FOLDS

        # Calculate RMSLE b/w actual y and inverse-log oof_pred
        y_actual_valid = y[valid_idx]
        oof_rmsle = rmsle(y_actual_valid, oof_pred)
        results[name]['rmsle'].append(oof_rmsle)
        
        print(f"Fold {i+1} RMSLE: {oof_rmsle:.4f}")

print("\n=== Model Comparison ===")
for name in models:
    mean_rmsle = np.mean(results[name]['rmsle'])
    std_rmsle = np.std(results[name]['rmsle'])
    print(f"{name} - Mean RMSLE: {mean_rmsle:.4f} ± {std_rmsle:.4f}")


# Save the results as a CSV
oof_df = pd.DataFrame({f'{m}': results[m]['oof'] for m in results})
oof_df.to_csv('oof_results.csv', index=False)
print("saved OOF results")

pred_df = pd.DataFrame({f'{m}': results[m]['pred'] for m in results})
df.to_csv('submission_results.csv', index=False)
print("saved submission predictions")


from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import optuna


# load oof results
oof_df = pd.read_csv("/kaggle/input/fe-and-model-blending/oof_results.csv")
# load preictions
pred_df = pd.read_csv("/kaggle/input/fe-and-model-blending/submission_results.csv")
# REMOVE NEXT 2 LINES NEXT TIME
pred_df.drop(columns=['id'],inplace=True)
pred_df.columns = oof_df.columns
# load y
y = train["Calories"]


oof_df.shape,pred_df.shape,y.shape


X_train = np.log1p(oof_df)
y_log = np.log1p(y)
X_test = np.log1p(pred_df)


# run optuna on ridge
# Define Optuna objective function
def objective(trial):
    params = {
        "alpha": trial.suggest_float("alpha", 0.001, 100.0, log=True),
        "tol": trial.suggest_float("tol", 1e-6, 1e-2)
    }

    model = Ridge(**params)
    
    rmse = -np.mean(cross_val_score(model, X_train, y_log, cv=FOLDS, scoring="neg_root_mean_squared_error"))
    
    return rmse

# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Get best alpha value
best_params = study.best_params
stack_rmse = study.best_value

print(f"Best params using Optuna: {best_params}\n Stacking RMSE: {stack_rmse}")


# final rdige model using cv
stack_oof_pred = np.zeros(len(X_train))
stack_test_pred = np.zeros(len(X_test))
scores = []
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for i, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
        print(f"\nFold {i+1}")
        x_train, y_train = X_train.iloc[train_idx], y_log[train_idx]
        x_valid, y_valid = X_train.iloc[valid_idx], y_log[valid_idx]

        model = Ridge(**best_params)
        model.fit(x_train,y_train)
    
        # Get OOF logged predictions
        oof_pred_logged = model.predict(x_valid)
        # Inverse Log OOF
        oof_pred = np.expm1(oof_pred_logged)
        # Save the OOF predictions in array
        stack_oof_pred[valid_idx] = oof_pred
        
        # GET test logged prediction
        test_pred_logged = model.predict(X_test)
        # Inverse log test
        test_pred = np.expm1(test_pred_logged)
        # Save test predictions in array
        stack_test_pred += test_pred / FOLDS

        # Calculate RMSLE b/w actual y and inverse-log oof_pred
        y_actual_valid = y[valid_idx]
        oof_rmsle = rmsle(y_actual_valid, oof_pred)
        scores.append(oof_rmsle)
        
        print(f"Fold {i+1} RMSLE: {oof_rmsle:.4f}")

stack_oof_score = np.mean(scores)
print(f'OOF RMSLE: {stack_oof_score:.4f}')


# save submission.csv
# already did expm1 during fitting model
# stack_test_pred = np.expm1(stack_test_pred)
submission['Calories'] = np.clip(stack_test_pred,1,314)
submission['Calories'] = round(submission['Calories'])

submission.to_csv('submission.csv',index=False)

