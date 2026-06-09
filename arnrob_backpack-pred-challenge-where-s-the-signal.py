import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
import optuna


def get_ori_files(extra=False):
    
    path = '/kaggle/input/playground-series-s5e2/'
    
    train_df = pd.read_csv(path + 'train.csv')
    train_extra_df = None
    if extra==True:
        train_extra_df = pd.read_csv(path + 'training_extra.csv')
    test_df = pd.read_csv(path + 'test.csv')

    return train_df, train_extra_df, test_df


train_df, train_extra_df, test_df = get_ori_files(extra=True)


train_df.head(3)


train_extra_df.head(3)


test_df.head(3)


train_df.info()


train_extra_df.info(show_counts=True)


test_df.info()


train_df[['Compartments','Weight Capacity (kg)','Price']].describe() #same stats for train_extra_df


train_df, train_extra_df, test_df = get_ori_files(extra=True)


def plot_feature(f):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    datasets = [train_df, train_extra_df, test_df]
    titles = ['Distrib on train_df', 'Distrib on train_extra_df', 'Distrib on test_df']

    # Combine values across all datasets to get the global order if needed
    combined = pd.concat([df[[f]] for df in datasets])
    
    # Check if feature is categorical
    is_categorical = combined[f].dtype == 'object' or pd.api.types.is_categorical_dtype(combined[f])

    # Get consistent order of categories if categorical
    order = sorted(combined[f].dropna().unique()) if is_categorical else None

    for ax, data, title in zip(axes, datasets, titles):
        if is_categorical:
            sns.countplot(data=data, x=f, order=order, ax=ax)
        else:
            sns.histplot(data=data, x=f, discrete=True, ax=ax)

        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


plot_feature('Brand')


plot_feature('Weight Capacity (kg)')


def plot_target():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    datasets = [train_df, train_extra_df]
    titles = ['Distrib on train_df', 'Distrib on train_extra_df']
    
    # Combine values across all datasets to get the global order if needed
    combined = pd.concat([df[['Price']] for df in datasets])
        
    for ax, data, title in zip(axes, datasets, titles):
        sns.histplot(data=data, x='Price', discrete=True, ax=ax)
    
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


plot_target()


train_df, _, test_df = get_ori_files()


train_df['Weight Capacity (kg)'] = train_df['Weight Capacity (kg)'].round().astype('str')
train_df['Compartments'] = train_df['Compartments'].astype(int).astype(str)
train_df.head(3)


def plot_feature_vs_price(df, f, rotation=0):
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    sns.boxplot(data=df[[f,'Price']], x=f, y='Price')
    plt.xticks(rotation=rotation)
    plt.show()


plot_feature_vs_price(train_df, 'Material')


plot_feature_vs_price(train_df, 'Weight Capacity (kg)')


train_df, _, test_df = get_ori_files()


train_df['Weight Capacity (kg)'] = train_df['Weight Capacity (kg)'].round().astype('str')
train_df['Compartments'] = train_df['Compartments'].astype(int).astype(str)
train_df.head(3)


features = [
    'Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
    'Waterproof', 'Style', 'Color', 'Weight Capacity (kg)'
]

# GÃ©nÃ©rer toutes les combinaisons de 2 colonnes
for combo in combinations(features, 2):
    col_name = '_'.join(combo)
    train_df[col_name] = train_df[combo[0]] + '-' + train_df[combo[1]]  

pd.set_option('display.max_columns', None)
train_df.head(3)


plot_feature_vs_price(train_df, 'Material_Compartments', rotation=90)


plot_feature_vs_price(train_df, 'Compartments_Style', rotation=90)


train_df, _, test_df = get_ori_files()


mean_price = train_df['Price'].mean()

test_df['Price'] = mean_price

test_df.head(3)


test_df[['id','Price']].to_csv('submission_one_price_for_all.csv',index=False)


train_df, train_extra_df, test_df = get_ori_files(extra=True)

train_df['Weight Capacity (kg)'] = train_df['Weight Capacity (kg)'].round().astype('Int64')
train_extra_df['Weight Capacity (kg)'] = train_extra_df['Weight Capacity (kg)'].round().astype('Int64')

train_all_df = pd.concat([train_df, train_extra_df])
train_all_df.head(3)


price_means_df = train_all_df.groupby(['Brand', 'Material', 'Size', 'Compartments', 
                                'Laptop Compartment', 'Waterproof', 'Style', 
                                'Color', 'Weight Capacity (kg)'])['Price'].mean().reset_index()
price_means_df.head(3)


test_df['Weight Capacity (kg)'] = test_df['Weight Capacity (kg)'].round().astype('Int64')

test_df = test_df.merge(price_means_df, on=['Brand', 'Material', 'Size', 'Compartments',
                                         'Laptop Compartment', 'Waterproof', 'Style', 
                                         'Color', 'Weight Capacity (kg)'], 
                        how='left')

# Fill the missing values with the mean price if needed
mean_price = train_all_df['Price'].mean()
test_df['Price'].fillna(mean_price, inplace=True)

test_df.head(3)


test_df[['id','Price']].to_csv('submission_one_price_per_cat.csv',index=False)


train_df, _, test_df = get_ori_files()


def add_stats(df):

    features = [
        'Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
        'Waterproof', 'Style', 'Color', 'Weight Capacity (kg)'
    ]
    target = 'Price'
    
    new_df = df.copy()
    
    # GÃ©nÃ©rer toutes les combinaisons de 1, 2 colonnes
    for r in [1, 2]:
        for combo in combinations(features, r):
            col_name = '_'.join(combo)
            mean_col_name = f'mean_{col_name}'
            
            # Calcul de la moyenne groupÃ©e
            group_mean = (
                df.groupby(list(combo))[target]
                .mean()
                .reset_index()
                .rename(columns={target: mean_col_name})
            )
            
            # Merge avec le DataFrame d'origine
            new_df = new_df.merge(group_mean, on=list(combo), how='left')
    
    # Ã€ ce stade, new_df contient toutes les nouvelles features de moyennes groupÃ©es
    return new_df


def cv_loop(train_df, model):

    # Supprimer la colonne 'id' car elle ne sert pas Ã  la prÃ©diction
    train_df = train_df.drop(columns=['id'])
    
    train_df['Weight Capacity (kg)'] = train_df['Weight Capacity (kg)'].round().astype('Int64')
    
    X_train = train_df
    y_train = train_df['Price']
    
    # Initializing variables
    n_splits = 5 # used for Cross-Validation, can be changed.
    rmse_all = 0 # rmse_all = sum RMSE by fold / nb of folds
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(X_train, y_train):
        
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        # Add the engineered columns
        X_train_fold = add_stats(X_train_fold)
        X_val_fold = add_stats(X_val_fold)
    
        # Removing of categorical features (replaced by Target Encoding)
        cat_cols = X_train_fold.select_dtypes(include=['object']).columns.tolist()
        X_train_fold = X_train_fold.drop(columns=cat_cols)
        X_val_fold = X_val_fold.drop(columns=cat_cols)
        
        # Remove the target column from the training and validation sets
        X_train_fold = X_train_fold.drop(columns=['Price'])
        X_val_fold = X_val_fold.drop(columns=['Price'])
    
        # Train the model
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], early_stopping_rounds=100, verbose=False)
    
        # Calculate RMSE for this fold
        y_val_preds = model.predict(X_val_fold)
        rmse = mean_squared_error(y_val_fold, y_val_preds, squared=False)
    
        # Calculate avg RMSE for all the folds
        rmse_all = rmse_all + rmse
    
    rmse_all = rmse_all / n_splits
    return rmse_all


def objective(trial):
    
    param_grid = {
        'max_depth': trial.suggest_int('max_depth', 3, 15), # default = 6 range = [0,âˆž]
        'n_estimators': trial.suggest_int('n_estimators', 100, 5000), # default = 100
        'eta': trial.suggest_loguniform('eta', 0.001, 0.3), # default = 0.3 range = [0,1]
        'subsample': trial.suggest_discrete_uniform('subsample', 0.1, 1.0, 0.1), # default = 1 range = (0, 1]
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.2, 1.0, 0.1), # default = 1 range = (0, 1]
        'colsample_bylevel': trial.suggest_discrete_uniform('colsample_bylevel', 0.2, 1.0, 0.1), # default = 1 range = (0, 1]
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 0.1, 10), # default = 1 range: [0,âˆž]
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.1, 100), # default = 1
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.0001, 10), # default=0
        'gamma': trial.suggest_loguniform('gamma', 0.001, 10), # default = 0 range: [0,âˆž]
    }  
    
    model = XGBRegressor(
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        n_jobs=4,
        **param_grid
    )

    rmse = cv_loop(train_df, model)

    return rmse


'''
study = optuna.create_study(direction='minimize', study_name='hyperparameter tuning for xgb')
study.optimize(objective, timeout=7200)
    
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('\tValue: {}'.format(trial.value))
print('\tParams: ')
for key, value in trial.params.items():
    print('\t\t{}: {}'.format(key, value))
'''


def add_stats_on_test(df_train, df_test):

    features = [
        'Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
        'Waterproof', 'Style', 'Color', 'Weight Capacity (kg)'
    ]
    target = 'Price'
    
    new_df = df_train.copy()
    
    # GÃ©nÃ©rer toutes les combinaisons de 1, 2 colonnes
    for r in [1, 2]:
        for combo in combinations(features, r):
            col_name = '_'.join(combo)
            mean_col_name = f'mean_{col_name}'
            
            # Calcul de la moyenne groupÃ©e
            group_mean = (
                df_train.groupby(list(combo))[target]
                .mean()
                .reset_index()
                .rename(columns={target: mean_col_name})
            )
            
            # Merge avec le DataFrame d'origine
            df_test = df_test.merge(group_mean, on=list(combo), how='left')
    
    # Ã€ ce stade, new_df contient toutes les nouvelles features de moyennes groupÃ©es
    return df_test


test_ids = test_df['id'] # save the ids for the final submission
test_df = test_df.drop(columns=['id'])


test_df = add_stats_on_test(train_df, test_df)

cat_cols = test_df.select_dtypes(include=['object']).columns.tolist()
test_df = test_df.drop(columns=cat_cols)


# Supprimer la colonne 'id' car elle ne sert pas Ã  la prÃ©diction
train_df = train_df.drop(columns=['id'])

train_df['Weight Capacity (kg)'] = train_df['Weight Capacity (kg)'].round().astype('Int64')

X_train = train_df
y_train = train_df['Price']

# Initializing variables
n_splits = 5 # used for Cross-Validation, can be changed.
rmse_all = 0
y_test_all = None
best_hp = {
        'max_depth': 3,
		'n_estimators': 2032,
		'eta': 0.014936571747636355,
		'subsample': 0.8,
		'colsample_bytree': 0.6000000000000001,
		'colsample_bylevel': 0.9000000000000001,
		'min_child_weight': 6.804233963577628,
		'reg_lambda': 94.59277010880174,
		'reg_alpha': 0.6142619872644968,
		'gamma': 0.0020707420006130604,  
}

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
for train_index, val_index in kf.split(X_train, y_train):

    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Add the engineered columns
    X_train_fold = add_stats(X_train_fold)
    X_val_fold = add_stats(X_val_fold)
    
    # Removing of categorical features (replaced by Target Encoding)
    cat_cols = X_train_fold.select_dtypes(include=['object']).columns.tolist()
    X_train_fold = X_train_fold.drop(columns=cat_cols)
    X_val_fold = X_val_fold.drop(columns=cat_cols)
    
    # Remove the target column from the training and validation sets
    X_train_fold = X_train_fold.drop(columns=['Price'])
    X_val_fold = X_val_fold.drop(columns=['Price'])
    
    # Train the model
    model = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', n_jobs=4, **best_hp)
    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], early_stopping_rounds=100, verbose=False)
    
    # Calculate preds for this fold
    y_val_preds = model.predict(X_val_fold)
    rmse = mean_squared_error(y_val_fold, y_val_preds, squared=False)
    print(f'RMSE for this fold {rmse}')
    
    # Calculate avg RMSE for all the folds
    rmse_all = rmse_all + rmse
    
    # Calculate preds with this model
    y_test = model.predict(test_df)
    
    # Calculate avg preds
    if y_test_all is None:
        y_test_all = y_test
    else:
        y_test_all += y_test
    
rmse_all = rmse_all / n_splits
print(f'Average RMSE for all folds {rmse_all}')

y_test_all = y_test_all / n_splits


test_df['id'] = test_ids
test_df['Price'] = y_test_all
test_df[['id','Price']].to_csv('submission_xgb_fe_hp.csv',index=False)

