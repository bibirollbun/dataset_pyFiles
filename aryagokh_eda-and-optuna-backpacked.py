import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

from tqdm.notebook import tqdm
import optuna

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv').set_index('id')
train.head()


# Uniques
print("Number of Unique values : \n")
for col in train.columns:
    print(f'{col} ----> {train[col].nunique()}\n')


# NaNs
print("% of NaN values : \n")
for col in train.columns:
    print(f'{col} ----> {(train[col].isna().sum()/len(train)*100):.2f} %\n')


train.info()


train.duplicated().sum()


sns.histplot(data=train, x='Price', kde=True, bins=50);
max(train['Price']), min(train['Price'])


sns.countplot(x='Brand', data=train);


sns.boxplot(data=train, x='Brand', y='Price')
plt.xticks(rotation=45)
plt.show()


sns.histplot(data=train, x='Price', kde=True, bins=70, hue='Brand');


sns.countplot(x='Material', data=train);


sns.histplot(data=train, x='Price', kde=True, bins=70, hue='Material');


sns.heatmap(pd.crosstab(train['Brand'], train['Material']), annot=True, cmap='coolwarm', fmt='d')
plt.yticks(rotation=0);


sns.countplot(x='Size', data=train);


sns.histplot(data=train, x='Price', kde=True, bins=70, hue='Size');


sns.countplot(x='Compartments', data=train);


sns.countplot(x='Laptop Compartment', data=train);


sns.histplot(data=train, hue='Laptop Compartment', x='Price', kde=True);


sns.countplot(x='Waterproof', data=train);


sns.countplot(x='Style', data=train);


sns.histplot(data=train, x='Price', hue='Style', kde=True, bins=50);


sns.countplot(x='Color', data=train);


sns.histplot(data=train, x='Price', hue='Color', kde=True, bins=50);


sns.histplot(data=train, x='Weight Capacity (kg)', kde=True);


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

# Your categorical columns
biv_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
            'Waterproof', 'Style', 'Color']

# Generate all unique pairs of columns
column_pairs = list(itertools.combinations(biv_cols, 2))

# Number of heatmap pairs (even number of pairs to avoid having an odd one out)
n_pairs = len(column_pairs)

# Set up the plot with two subplots side by side
for num in range(0, n_pairs, 2):  # Process two at a time
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns
    

    col1, col2 = column_pairs[num]
    cross_tab1 = pd.crosstab(train[col1], train[col2])
    sns.heatmap(cross_tab1, annot=True, fmt="d", linewidths=0.5, ax=axs[0])
    axs[0].set_title(f'No. {num+1} - Heatmap of {col1} vs {col2}')
    axs[0].set_xlabel(col2)
    axs[0].set_ylabel(col1)


    if num + 1 < n_pairs:  # Check if there’s another pair
        col1, col2 = column_pairs[num + 1]
        cross_tab2 = pd.crosstab(train[col1], train[col2])
        sns.heatmap(cross_tab2, annot=True, fmt="d", linewidths=0.5, ax=axs[1])
        axs[1].set_title(f'No. {num+2} - Heatmap of {col1} vs {col2}')
        axs[1].set_xlabel(col2)
        axs[1].set_ylabel(col1)
    
    plt.tight_layout() 
    plt.show()


# Will start will mode imputation


obj_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
            'Waterproof', 'Style', 'Color']
num_cols = ['Weight Capacity (kg)']
tgt_cols = ['Price']

le_cols = ['Size', 'Laptop Compartment', 'Waterproof', ]
ohe_cols = ['Brand', 'Material', 'Style', 'Color']



def impute_cols(train_data, test_data, obj_cols, num_cols):
    for col in obj_cols:
        fill_value = train_data[col].mode()[0]
        train_data[col] = train_data[col].fillna(fill_value)
        if test_data is not None:
            test_data[col] = test_data[col].fillna(fill_value)

        print(f"{col} imputation successful! - Fill value : {fill_value}")
    for col in num_cols:
        fill_value = train_data[col].mode()[0]
        train_data[col] = train_data[col].fillna(fill_value)
        if test_data is not None:
            test_data[col] = test_data[col].fillna(fill_value)

        print(f"{col} imputation successful! - Fill value : {fill_value}\n")

    print("Done Imputation!!")

    if test_data is None:
        return train_data
    if test_data is not None:
        return train_data, test_data


train_imputed = impute_cols(train_data=train, obj_cols=obj_cols, num_cols=num_cols, test_data=None)
train_imputed.head()


train_imputed.isna().sum()


def encode_and_scale(train_data, test_data, le_cols, ohe_cols, num_cols):
    le = LabelEncoder()
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    scaler = StandardScaler()

    for col in le_cols:
        train_data[col] = le.fit_transform(train_data[col])
        if test_data is not None:
            test_data[col] = test_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    if len(ohe_cols) > 0:
        ohe_train = ohe.fit_transform(train_data[ohe_cols])
        if test_data is not None:
            ohe_test = ohe.transform(test_data[ohe_cols])

        ohe_cols_new = ohe.get_feature_names_out(ohe_cols)
        
        ohe_train_df = pd.DataFrame(ohe_train, columns=ohe_cols_new)
        if test_data is not None:
            ohe_test_df = pd.DataFrame(ohe_test, columns=ohe_cols_new)
        
        train_data = train_data.drop(columns=ohe_cols).reset_index(drop=True)
        if test_data is not None:
            test_data = test_data.drop(columns=ohe_cols).reset_index(drop=True)
        
        train_data = pd.concat([train_data, ohe_train_df], axis=1)
        if test_data is not None:
            test_data = pd.concat([test_data, ohe_test_df], axis=1)

    if len(num_cols) > 0:
        train_data[num_cols] = scaler.fit_transform(train_data[num_cols])
        if test_data is not None:
            test_data[num_cols] = scaler.transform(test_data[num_cols])

    if test_data is not None:
        return train_data, test_data
    if test_data is None:
        return train_data


X_train, X_test, y_train, y_test = train_test_split(train_imputed.drop(tgt_cols, axis=1), train_imputed[tgt_cols],
                                                    test_size=0.2, random_state=42)


X_train_, X_test_ = encode_and_scale(train_data=X_train, test_data=X_test, le_cols=le_cols,
                                     ohe_cols=ohe_cols, num_cols=num_cols)


X_train_.columns = X_train_.columns.str.replace(' ', '_')
X_test_.columns = X_test_.columns.str.replace(' ', '_')


def evaluate_models(X_train, X_test, y_train, y_test):
    results = {}

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet Regression': ElasticNet(),
        'Random Forest': RandomForestRegressor(n_jobs=-1),
        'XGBoost': XGBRegressor(tree_method='hist', device="cuda"),
        'LightGBM': lgb.LGBMRegressor(n_jobs=-1),
        'CatBoost': CatBoostRegressor(task_type='GPU', verbose=100),
        'KNeighbors Regressor': KNeighborsRegressor(n_jobs=-1),
        'Decision Tree': DecisionTreeRegressor(),
        'AdaBoost Regressor': AdaBoostRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        # 'SVR': SVR(),
    }

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        rmse_cv = -cv_scores.mean()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[model_name] = {'RMSE': rmse, 'CV RMSE': rmse_cv}

        print(f"{model_name} RMSE: {rmse:.4f}, CV RMSE: {rmse_cv:.4f}\n")
    
    results_df = pd.DataFrame(results).T
    return results_df


results = evaluate_models(X_train_, X_test_, y_train, y_test)


print(results)


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

def tune_linear_regression(X_train, y_train):
    linear_reg = LinearRegression()

    param_grid = {
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'n_jobs': [-1], 
        'positive': [True, False] 
    }

    grid_search = GridSearchCV(estimator=linear_reg, param_grid=param_grid,
                               scoring='neg_root_mean_squared_error', cv=5, verbose=1, 
                               n_jobs=-1)

    total_iter = len(param_grid['fit_intercept']) * len(param_grid['copy_X']) * len(param_grid['positive'])
    
    with tqdm(total=total_iter, dynamic_ncols=True) as pbar:
        grid_search.fit(X_train, y_train)
        pbar.update(total_iter) 

    print(f"Best parameters for Linear Regression: {grid_search.best_params_}")
    print(f"Best RMSE for Linear Regression: {-grid_search.best_score_}")

    return grid_search.best_params_


best_params = tune_linear_regression(X_train_, y_train)


def tune_elastic_net(X_train, y_train):
    elastic_net = ElasticNet()

    param_grid = {
        'alpha': np.logspace(-4, 4, 9),  # Regularization parameter
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1],  # ElasticNet mixing parameter
        'fit_intercept': [True, False],
        'selection': ['cyclic', 'random']
    }

    grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid,
                               scoring='neg_root_mean_squared_error', cv=5, verbose=1, 
                               n_jobs=-1)

    total_iter = len(param_grid['alpha']) * len(param_grid['l1_ratio']) * len(param_grid['fit_intercept']) * len(param_grid['selection'])
    
    with tqdm(total=total_iter, dynamic_ncols=True) as pbar:
        grid_search.fit(X_train, y_train)
        pbar.update(total_iter)

    print(f"Best parameters for ElasticNet: {grid_search.best_params_}")
    print(f"Best RMSE for ElasticNet: {-grid_search.best_score_}")

    return grid_search.best_params_


best_params = tune_elastic_net(X_train_, y_train)


# optuna.logging.set_verbosity(optuna.logging.WARNING)

# best_rmse = float('inf')

# def objective(trial):
#     global best_rmse

#     # param = {
#     #     'objective': 'reg:squarederror',
#     #     'eval_metric': 'rmse',
#     #     'tree_method': 'gpu_hist',
#     #     'device': 'cuda',
#     #     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
#     #     'max_depth': trial.suggest_int('max_depth', 3, 10),
#     #     'n_estimators': 200,
#     #     'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#     #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#     #     'gamma': trial.suggest_float('gamma', 0, 1),
#     #     'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#     # }
#     param = {
#         'objective': 'reg:squarederror',
#         'eval_metric': 'rmse',
#         'tree_method': 'gpu_hist',
#         'device': 'cuda',
#         'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5),
#         'max_depth': trial.suggest_int('max_depth', 2, 15),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 800),
#         'subsample': trial.suggest_float('subsample', 0.4, 1.0), 
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
#         'gamma': trial.suggest_float('gamma', 0, 5), 
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
#     }

#     model = XGBRegressor(**param)
#     model.fit(X_train_, y_train)
#     y_pred = model.predict(X_test_)
#     rmse = mean_squared_error(y_test, y_pred, squared=False)

#     if rmse < best_rmse:
#         best_rmse = rmse
#         print(f"New Best RMSE: {rmse:.5f} | Params: {param}")

#     return rmse

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=200, n_jobs=-1, show_progress_bar=True)


# import optuna
# from catboost import CatBoostRegressor
# from sklearn.metrics import mean_squared_error

# # Set Optuna logging verbosity
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# best_rmse = float('inf')

# def objective(trial):
#     global best_rmse

#     param = {
#         'objective': 'RMSE',
#         'eval_metric': 'RMSE',
#         'iterations': trial.suggest_int('iterations', 100, 800),
#         'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5),
#         'depth': trial.suggest_int('depth', 2, 15),
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
#         'subsample': trial.suggest_float('subsample', 0.4, 1.0),
#         # 'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 1.0),
#         'random_seed': 42,
#         'verbose': 0,  # No verbose output from CatBoost
#         'task_type': 'GPU',  # Specify GPU usage
#         'devices': '0',  # Specify GPU device, '0' is typically the first GPU
#         'bootstrap_type': 'Bernoulli',  # Correct bootstrap type for subsample
#     }

#     model = CatBoostRegressor(**param)
#     model.fit(X_train_, y_train)
#     y_pred = model.predict(X_test_)
#     rmse = mean_squared_error(y_test, y_pred, squared=False)

#     if rmse < best_rmse:
#         best_rmse = rmse
#         print(f"New Best RMSE: {rmse:.5f} | Params: {param}")

#     return rmse

# # Create and optimize the study
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=2000, n_jobs=1, show_progress_bar=True)


test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv').set_index('id')
test.head()


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv').set_index('id')
train.head()


train_imputed, test_imputed = impute_cols(train_data=train, obj_cols=obj_cols, num_cols=num_cols, test_data=test)
test_imputed.head()


train_imputed.head()


test_imputed['Size'].value_counts()


train_copy_enc, test_enc = encode_and_scale(train_data=train_imputed, test_data=test_imputed, le_cols=le_cols,
                                     ohe_cols=ohe_cols, num_cols=num_cols)

test_enc.head()


train_copy_enc.columns = train_copy_enc.columns.str.replace(' ', '_')
test_enc.columns = test_enc.columns.str.replace(' ', '_')


sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
sample_sub.head()


def make_preds(models, params, train_data, test_data, sample_sub, name, **kwargs):
    model = models(**params, **kwargs)
    print("Fitting....")
    model.fit(train_data.drop(['Price'], axis=1), train_data['Price'])
    print("Predicting....")
    preds = model.predict(test_data)
    print("Saving....")
    sample_sub['Price'] = preds
    sample_sub.to_csv(f'{str(name)}', index=False)
    print("Done!")


# 1. Linear Regression
params = {'copy_X': True, 'fit_intercept': True, 'n_jobs': -1, 'positive': False}
model = LinearRegression

make_preds(models=model, params=params, train_data=train_copy_enc, test_data=test_enc, 
           sample_sub=sample_sub, name='GridSearch_LinearRegression.csv')

# Score - 39.14773


# 2. XGBoost Optuna
params = {'learning_rate': 0.03440598938090024,
          'max_depth': 2,
          'n_estimators': 597,
          'subsample': 0.49942492435199753,
          'colsample_bytree': 0.7094069370929462,
          'gamma': 4.4792903658045065,
          'min_child_weight': 11}
model = XGBRegressor
make_preds(models=model, params=params, train_data=train_copy_enc, test_data=test_enc, 
           sample_sub=sample_sub, name='XGB_Optuna_1.csv')

#  Score - 39.13840


params = {'objective': 'RMSE', 'eval_metric': 'RMSE', 'iterations': 101, 
          'learning_rate': 0.19437423506677795, 'depth': 3, 
          'l2_leaf_reg': 7.035083804063252, 'subsample': 0.7052725830955228, 
          'random_seed': 42, 'verbose': 0, 
          'task_type': 'GPU', 'devices': '0', 
          'bootstrap_type': 'Bernoulli'}
model = CatBoostRegressor
make_preds(models=model, params=params, train_data=train_copy_enc, test_data=test_enc, 
           sample_sub=sample_sub, name='Cat_Optuna_1.csv')

# Score: 39.14265


# Stacking regressor - 1
xgb_1 = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 
         'tree_method': 'gpu_hist', 'device': 'cuda', 'learning_rate': 0.03843586377106862, 
         'max_depth': 2, 'n_estimators': 587, 'subsample': 0.48953769882659876, 
         'colsample_bytree': 0.7018454672720531, 'gamma': 3.593736567606315, 
         'min_child_weight': 10}

xgb_2 = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 
         'tree_method': 'gpu_hist', 'device': 'cuda', 'learning_rate': 0.03440598938090024, 
         'max_depth': 2, 'n_estimators': 597, 'subsample': 0.49942492435199753, 
         'colsample_bytree': 0.7094069370929462, 'gamma': 4.4792903658045065,
         'min_child_weight': 11}

cat_1 = {'objective': 'RMSE', 'eval_metric': 'RMSE', 
         'iterations': 237, 'learning_rate': 0.1490862870726785, 
         'depth': 2, 'l2_leaf_reg': 5.259516388580975, 'subsample': 0.43573983606694344, 
         'random_seed': 42, 'verbose': 0, 'task_type': 'GPU', 'devices': '0', 
         'bootstrap_type': 'Bernoulli'}

cat_2 =  {'objective': 'RMSE', 'eval_metric': 'RMSE', 
          'iterations': 101, 'learning_rate': 0.19437423506677795, 
          'depth': 3, 'l2_leaf_reg': 7.035083804063252, 
          'subsample': 0.7052725830955228, 'random_seed': 42,
          'verbose': 0, 'task_type': 'GPU', 'devices': '0', 
          'bootstrap_type': 'Bernoulli'}

catboost_model1 = CatBoostRegressor(**cat_1)
catboost_model2 = CatBoostRegressor(**cat_2)  
xgb_model1 = XGBRegressor(**xgb_1)  
xgb_model2 = XGBRegressor(**xgb_2)  

final_model = LinearRegression()

base_learners = [
    ('catboost1', catboost_model1),
    ('catboost2', catboost_model2),
    ('xgb1', xgb_model1),
    ('xgb2', xgb_model2)
]

stacking_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=final_model,
    cv=5,  
    verbose=1
)

stacking_model.fit(X_train_, y_train)

y_pred = stacking_model.predict(X_test_)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE of stacked model: {rmse:.4f}')


preds = stacking_model.predict(test_enc)
new_sub = sample_sub
new_sub['Price'] = preds
new_sub.to_csv('Stacked_XGBnCat.csv', index=False)
new_sub.head()




