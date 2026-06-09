import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb


train = pd.read_csv('/kaggle/input/neoai-2025-tricy-table-data/train_tables.csv')
test = pd.read_csv('/kaggle/input/neoai-2025-tricy-table-data/test_tables.csv')


def clf_train(train, test, target, weight_col, id_col, name_file = 'sub.csv', func_inv = None):

    param = {
    'learning_rate': 0.1,
    'num_leaves': 48,
    'lambda_l1' : 1,
    'lambda_l2' : 1,
    'min_data_in_leaf' : 200,
    'boosting_type':    'gbdt',
    'objective': 'mae',
    'verbosity':-1,
    }
    
    predict_test = np.zeros(len(test))

    tr = lgb.Dataset(train, target, weight=weight_col)
    bst = lgb.train(param, tr, num_boost_round=1000)
    predict_test = bst.predict(test)
    if func_inv:
        predict_test = func_inv(predict_test)
    sub = pd.DataFrame()
    sub['id'] = id_col
    sub['target'] = predict_test
    sub.to_csv(name_file, index = None)


def func_inv(x):
    return x 


drop_cols = ['target']
train_cols = [c for c in train.columns if c not in drop_cols]
feat_imp = np.zeros(len(train_cols))
srt_cols = [int(x[1].split('_')[0]) for x in sorted(zip(feat_imp, train_cols))[::-1] if 'flm' in x[1]]
bad_cols = [f'{x}_flm' for x in srt_cols]
weight = np.ones(len(train))
test_sub = clf_train(train[train_cols], test[train_cols], train['target'] , weight, test['id'].tolist(), 'submission.csv', func_inv = func_inv)


test_sub = clf_train(train[train_cols], test[train_cols], train['target'] , weight, test['id'].tolist(), 'submission2.csv', func_inv = func_inv)


drop_cols = ['target']
train_cols = prepare_data(train, drop_cols)
weight = np.ones(len(train))
model, feature_importance = train_model(train, test, train['target'], weight, test['id'].tolist(), 'submission3.csv', func_inv=func_inv)

# Проверка важности переменных 'hour', 'minute' и 'target'
important_features = {col: imp for col, imp in zip(train_cols, feature_importance)}
print("Importance of 'hour':", important_features.get('hour', 'Not available'))
print("Importance of 'minute':", important_features.get('minute', 'Not available'))
print("Importance of 'target':", important_features.get('target', 'Not available'))


def clf_train(train, test, target, weight_col, id_col, name_file='sub.csv', func_inv=None):
    # Updated parameters
    param = {
        'learning_rate': 0.1,
        'num_leaves': 48,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'min_data_in_leaf': 100,
        'boosting_type': 'gbdt',
        'objective': 'mae',
        'verbosity': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'extra_trees': True,
        'reg_sqrt': True,
        'scale_pos_weight': 1.0,
        'is_unbalance': False
    }

    # Example feature selection mechanism
    score_dict = {col: np.random.rand() * 100 for col in train.columns if col != 'target'}
    bad_cols_new = [col for col in score_dict if score_dict[col] < 53.3]
    train_cols_new = [col for col in train.columns if col not in bad_cols_new]

    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    predict_test = np.zeros(len(test))
    predict_train = np.zeros(len(train))
    bst_list = []

    for train_index, valid_index in kf.split(train):
        tr = lgb.Dataset(train.iloc[train_index][train_cols_new], target.iloc[train_index], weight=weight_col.iloc[train_index])
        va = lgb.Dataset(train.iloc[valid_index][train_cols_new], target.iloc[valid_index], weight=weight_col.iloc[valid_index], reference=tr)

        bst = lgb.train(param, tr, valid_sets=[va], num_boost_round=1000, early_stopping_rounds=50, verbose_eval=10)
        bst_list.append(bst)

        predict_train[valid_index] = bst.predict(train.iloc[valid_index][train_cols_new])
        predict_test += bst.predict(test[train_cols_new]) / kf.n_splits

    if func_inv:
        predict_test = func_inv(predict_test)

    sub = pd.DataFrame()
    sub['id'] = id_col
    sub['target'] = predict_test
    sub.to_csv(name_file, index=False)

    # Evaluate the model
    mse = mean_squared_error(target, predict_train)
    print(f'Mean Squared Error: {mse}')

# Define a dummy inverse function as needed
def func_inv(x):
    return x


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Create an imputer object with a strategy of your choice
imputer = SimpleImputer(strategy='mean')

# Load data
train = pd.read_csv('/kaggle/input/neoai-2025-tricy-table-data/train_tables.csv')
test = pd.read_csv('/kaggle/input/neoai-2025-tricy-table-data/test_tables.csv')

# Fit the imputer on the training data and transform both train and test data
train_imputed = imputer.fit_transform(train[train_cols])
test_imputed = imputer.transform(test[train_cols])


# Drop rows with missing values
train_dropped = train.dropna(subset=train_cols)
test_dropped = test.dropna(subset=train_cols)

# Update target and weight arrays accordingly
target_dropped = train_dropped['target']
weight_dropped = np.ones(len(train_dropped))

# Update train_cols to exclude target
train_cols_dropped = [c for c in train_dropped.columns if c != 'target']

def clf_train(train, test, target, weight_col, id_col, name_file='sub.csv', func_inv=None):
    # Feature selection
    selector = SelectKBest(score_func=f_regression, k='all')  # You can choose 'k' based on your needs
    train_selected = selector.fit_transform(train, target)
    test_selected = selector.transform(test)

    # Cross-validation
    kf = KFold(n_splits=1, shuffle=True, random_state=42)
    predict_test = np.zeros(len(test))
    oof_predictions = np.zeros(len(train))

    param = {
        'learning_rate': 0.05,  # Reduced learning rate
        'num_leaves': 80,  # Increased number of leaves
        'lambda_l1': 1,
        'lambda_l2': 1,
        'min_data_in_leaf': 25,  # Reduced to allow for more complex models
        'objective': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'metric': 'mae',
    }

    for train_index, val_index in kf.split(train_selected):
        X_train, X_val = train_selected[train_index], train_selected[val_index]
        y_train, y_val = target[train_index], target[val_index]

        tr = lgb.Dataset(X_train, y_train, weight=weight_col[train_index])
        val = lgb.Dataset(X_val, y_val, weight=weight_col[val_index], reference=tr)

        bst = lgb.train(
            param,
            tr,
            valid_sets=[tr, val], num_boost_round=1000))

        oof_predictions[val_index] = bst.predict(X_val, num_iteration=bst.best_iteration)
        predict_test += bst.predict(test_selected, num_iteration=bst.best_iteration) / kf.n_splits

    # Calculate out-of-fold MAE
    oof_mae = mean_absolute_error(target, oof_predictions)
    print(f'Out-of-Fold MAE: {oof_mae}')

    if func_inv:
        predict_test = func_inv(predict_test)

    sub = pd.DataFrame()
    sub['id'] = id_col
    sub['target'] = predict_test
    sub.to_csv(name_file, index=False)

    return sub

def func_inv(x):
    return x

# Prepare data
drop_cols = ['target']
train_cols = [c for c in train.columns if c not in drop_cols]
weight = np.ones(len(train))
# Impute missing values
imputer = SimpleImputer(strategy='mean')
train_imputed = imputer.fit_transform(train[train_cols])
test_imputed = imputer.transform(test[train_cols])

# Train model with imputed data
submission = clf_train(train_imputed, test_imputed, train['target'], weight, test['id'].tolist(), 'submission.csv', func_inv=func_inv)


import lightgbm as lgb
print(lgb.__version__)


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv('/kaggle/input/neoai-2025-tricy-table-data/train_tables.csv')
test = pd.read_csv('/kaggle/input/neoai-2025-tricy-table-data/test_tables.csv')

# Define the columns to be used for training
drop_cols = ['target']
train_cols = [c for c in train.columns if c not in drop_cols]

# Create an imputer object with a strategy of your choice
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data and transform both train and test data
train_imputed = imputer.fit_transform(train[train_cols])
test_imputed = imputer.transform(test[train_cols])

# Drop rows with missing values
train_dropped = train.dropna(subset=train_cols)
test_dropped = test.dropna(subset=train_cols)

# Update target and weight arrays accordingly
target_dropped = train_dropped['target']
weight_dropped = np.ones(len(train_dropped))

# Update train_cols to exclude target
train_cols_dropped = [c for c in train_dropped.columns if c != 'target']

def clf_train(train, test, target, weight_col, id_col, name_file='sub.csv', func_inv=None):
    # Feature selection
    selector = SelectKBest(score_func=f_regression, k='all')  # You can choose 'k' based on your needs
    train_selected = selector.fit_transform(train, target)
    test_selected = selector.transform(test)

    # Cross-validation
    kf = KFold(n_splits=200, shuffle=True, random_state=42)
    predict_test = np.zeros(len(test))
    oof_predictions = np.zeros(len(train))

    param = {
        'learning_rate': 0.1,  # Reduced learning rate
        'num_leaves': 80,  # Increased number of leaves
        'lambda_l1': 1,
        'lambda_l2': 1,
        'min_data_in_leaf': 25,  # Reduced to allow for more complex models
        'objective': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'metric': 'mae',
    }

    for train_index, val_index in kf.split(train_selected):
        X_train, X_val = train_selected[train_index], train_selected[val_index]
        y_train, y_val = target[train_index], target[val_index]

        tr = lgb.Dataset(X_train, y_train, weight=weight_col[train_index])
        val = lgb.Dataset(X_val, y_val, weight=weight_col[val_index], reference=tr)

        bst = lgb.train(
            param,
            tr,
            valid_sets=[tr, val], num_boost_round=1000)

        oof_predictions[val_index] = bst.predict(X_val, num_iteration=bst.best_iteration)
        predict_test += bst.predict(test_selected, num_iteration=bst.best_iteration) / kf.n_splits

    # Calculate out-of-fold MAE
    oof_mae = mean_absolute_error(target, oof_predictions)
    print(f'Out-of-Fold MAE: {oof_mae}')

    if func_inv:
        predict_test = func_inv(predict_test)

    sub = pd.DataFrame()
    sub['id'] = id_col
    sub['target'] = predict_test
    sub.to_csv(name_file, index=False)

    return sub

def func_inv(x):
    return x

# Prepare data
weight = np.ones(len(train))
train_imputed = imputer.fit_transform(train[train_cols])
test_imputed = imputer.transform(test[train_cols])

# Train model with imputed data
submission = clf_train(train_imputed, test_imputed, train['target'], weight, test['id'].tolist(), 'submission.csv', func_inv=func_inv)


from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Define the parameters for the LightGBM model
param = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': -1,
    'num_leaves': 31
    # Add other parameters as needed
}

def determine_feature_importance(train, target):
    # Обучение модели для получения значимости фичей
    model = lgb.LGBMRegressor(**param)
    model.fit(train, target)
    importance = model.feature_importances_
    print("Feature importances from model:", importance)
    return importance

def select_features_rfe(train, target, n_features_to_select=10):
    # Использование RFE для выбора фичей
    model = lgb.LGBMRegressor(**param)
    selector = RFE(model, n_features_to_select=n_features_to_select, step=1)
    selector = selector.fit(train, target)
    print("RFE support:", selector.support_)
    print("RFE ranking:", selector.ranking_)
    return selector.transform(train), selector.transform(test_imputed)

def select_features_mutual_info(train, target):
    # Использование взаимной информации
    mutual_info = mutual_info_regression(train, target)
    print("Mutual information:", mutual_info)
    return mutual_info

def apply_pca(train, test, n_components=10):
    # Использование PCA для уменьшения размерности
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)
    
    print("Explained variance ratio by PCA:", pca.explained_variance_ratio_)
    return train_pca, test_pca

# Пример использования:
importance = determine_feature_importance(train_imputed, train['target'])
train_selected_rfe, test_selected_rfe = select_features_rfe(train_imputed, train['target'])
mutual_info = select_features_mutual_info(train_imputed, train['target'])
train_pca, test_pca = apply_pca(train_imputed, test_imputed)

# Используйте один из методов для выбора фичей перед обучением модели
submission = clf_train(train_pca, test_pca, train['target'], weight, test['id'].tolist(), 'submission.csv', func_inv=func_inv)


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

# Load data
train = pd.read_csv('/kaggle/input/neoai-2025-tricy-table-data/train_tables.csv')
test = pd.read_csv('/kaggle/input/neoai-2025-tricy-table-data/test_tables.csv')

# Define the columns to be used for training
drop_cols = ['target']
train_cols = [c for c in train.columns if c not in drop_cols]

# Create an imputer object with a strategy of your choice
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data and transform both train and test data
train_imputed = imputer.fit_transform(train[train_cols])
test_imputed = imputer.transform(test[train_cols])

# Drop rows with missing values
train_dropped = train.dropna(subset=train_cols)
test_dropped = test.dropna(subset=train_cols)

# Update target and weight arrays accordingly
target_dropped = train_dropped['target']
weight_dropped = np.ones(len(train_dropped))

# Update train_cols to exclude target
train_cols_dropped = [c for c in train_dropped.columns if c != 'target']

def clf_train(train, test, target, weight_col, id_col, name_file='sub.csv', func_inv=None):
    # Feature selection
    selector = SelectKBest(score_func=f_regression, k='all')  # You can choose 'k' based on your needs
    train_selected = selector.fit_transform(train, target)
    test_selected = selector.transform(test)

    # Cross-validation
    kf = KFold(n_splits=200, shuffle=True, random_state=42)
    predict_test = np.zeros(len(test))
    oof_predictions = np.zeros(len(train))

    param = {
        'learning_rate': 0.1,  # Reduced learning rate
        'num_leaves': 80,  # Increased number of leaves
        'lambda_l1': 1,
        'lambda_l2': 1,
        'min_data_in_leaf': 25,  # Reduced to allow for more complex models
        'objective': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'metric': 'mae',
    }

    for train_index, val_index in kf.split(train_selected):
        X_train, X_val = train_selected[train_index], train_selected[val_index]
        y_train, y_val = target[train_index], target[val_index]

        tr = lgb.Dataset(X_train, y_train, weight=weight_col[train_index])
        val = lgb.Dataset(X_val, y_val, weight=weight_col[val_index], reference=tr)

        bst = lgb.train(
            param,
            tr,
            valid_sets=[tr, val], num_boost_round=1000)

        oof_predictions[val_index] = bst.predict(X_val, num_iteration=bst.best_iteration)
        predict_test += bst.predict(test_selected, num_iteration=bst.best_iteration) / kf.n_splits

    # Calculate out-of-fold MAE
    oof_mae = mean_absolute_error(target, oof_predictions)
    print(f'Out-of-Fold MAE: {oof_mae}')

    if func_inv:
        predict_test = func_inv(predict_test)

    sub = pd.DataFrame()
    sub['id'] = id_col
    sub['target'] = predict_test
    sub.to_csv(name_file, index=False)

    return sub

def func_inv(x):
    return x

# Random Forest Feature Importance
def determine_feature_importance_rf(train, target):
    # Обучение модели для получения значимости фичей с помощью Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train, target)
    importance = model.feature_importances_
    print("Feature importances from Random Forest:", importance)
    return importance

# Example usage of Random Forest feature importance:
importance_rf = determine_feature_importance_rf(train_imputed, train['target'])

# CatBoost Feature Importance
def determine_feature_importance_catboost(train, target):
    # Обучение модели для получения значимости фичей с помощью CatBoost
    model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=0)
    model.fit(train, target)
    importance = model.get_feature_importance()
    print("Feature importances from CatBoost:", importance)
    return importance

importance_catboost = determine_feature_importance_catboost(train_imputed, train['target'])

weight = np.ones(len(train))
train_imputed = imputer.fit_transform(train[train_cols])
test_imputed = imputer.transform(test[train_cols])

submission = clf_train(train_imputed, test_imputed, train['target'], weight, test['id'].tolist(), 'submission.csv', func_inv=func_inv)


# Train model with imputed data
submission = clf_train(train_imputed, test_imputed, train['target'], weight, test['id'].tolist(), 'submission.csv', func_inv=func_inv)




