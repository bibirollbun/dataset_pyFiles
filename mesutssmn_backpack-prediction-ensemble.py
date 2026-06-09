import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
from cuml.preprocessing import TargetEncoder
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from lightgbm.callback import early_stopping 


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
extra_train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col='id')

train_data = pd.concat([train_data, extra_train_data], axis=0, ignore_index=True)


def transform_dataset(data):
    missing_columns = ['Material', 'Style', 'Brand', 'Size', 'Waterproof', 'Color', 'Laptop Compartment']
    data.fillna({col: 'Unknown' for col in missing_columns}, inplace=True)
    
    for col in missing_columns:
        data[f'Missing_{col}'] = (data[col] == 'NAN').astype(int)
    
    data['Total_Missing_Values'] = data[[f'Missing_{col}' for col in missing_columns]].sum(axis=1)
    
    return data

train_data = transform_dataset(train_data)
test_data = transform_dataset(test_data)


feature_columns = [col for col in train_data.columns if col != "Price"]
categorical_columns = [col for col in feature_columns if col not in ["Weight Capacity (kg)"]]


validation_predictions_lgbm = np.zeros(len(train_data))
validation_predictions_catboost = np.zeros(len(train_data))
test_predictions_lgbm = np.zeros(len(test_data))
test_predictions_catboost = np.zeros(len(test_data))
validation_predictions_xgb = np.zeros(len(train_data))
test_predictions_xgb = np.zeros(len(test_data))


def get_kfold_splits(train_data, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(kfold.split(train_data))

def prepare_data(train_data, test_data, feature_columns, categorical_columns, train_idx, valid_idx):
    X_train, y_train = train_data.loc[train_idx, feature_columns].copy(), train_data.loc[train_idx, "Price"]
    X_valid, y_valid = train_data.loc[valid_idx, feature_columns].copy(), train_data.loc[valid_idx, "Price"]
    X_test = test_data[feature_columns].copy()
    return X_train, y_train, X_valid, y_valid, X_test

def encode_features(X_train, X_valid, X_test, y_train, feature_columns):
    encoder = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')
    for col in feature_columns:
        encoder.fit(X_train[col], y_train)
        X_train[f"Encoded_{col}"] = encoder.transform(X_train[col])
        X_valid[f"Encoded_{col}"] = encoder.transform(X_valid[col])
        X_test[f"Encoded_{col}"] = encoder.transform(X_test[col])
    return X_train, X_valid, X_test

def preprocess_categorical(X_train, X_valid, X_test, categorical_columns):
    for df in [X_train, X_valid, X_test]:
        df[categorical_columns] = df[categorical_columns].fillna('Missing').astype('string').astype('category')
    return X_train, X_valid, X_test

def train_catboost(X_train, y_train, X_valid, y_valid, all_model_features, categorical_columns):
    catboost_params = {
        'learning_rate': 0.08,
        'depth': 7,
        'l2_leaf_reg': 7.88,
        'iterations': 830,
        'early_stopping_rounds': 200,
        'random_strength': 1.44, 
        'bagging_temperature': 0.53, 
        'border_count': 245,
        'task_type': 'GPU',
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': 0
    }

    model = CatBoostRegressor(**catboost_params, cat_features=categorical_columns)
    model.fit(
        X_train[all_model_features], y_train,
        eval_set=[(X_valid[all_model_features], y_valid)]
    )
    return model

def cross_validation_catboost(train_data, test_data, feature_columns, categorical_columns, n_splits=5):
    kfold_splits = get_kfold_splits(train_data, n_splits)
    validation_predictions = np.zeros(len(train_data))
    test_predictions = np.zeros(len(test_data))

    for fold, (train_idx, valid_idx) in enumerate(kfold_splits):
        print(f"### Processing Fold {fold+1} ###")
        
        X_train, y_train, X_valid, y_valid, X_test = prepare_data(train_data, test_data, feature_columns, categorical_columns, train_idx, valid_idx)

        X_train, X_valid, X_test = encode_features(X_train, X_valid, X_test, y_train, feature_columns)

        X_train, X_valid, X_test = preprocess_categorical(X_train, X_valid, X_test, categorical_columns)

        all_model_features = feature_columns + [f"Encoded_{col}" for col in feature_columns]

        model = train_catboost(X_train, y_train, X_valid, y_valid, all_model_features, categorical_columns)

        validation_predictions[valid_idx] = model.predict(X_valid[all_model_features])
        test_predictions += model.predict(X_test[all_model_features]) / n_splits

    rmse_score = np.sqrt(mean_squared_error(train_data["Price"], validation_predictions))
    print(f"Validation RMSE: {rmse_score}")

    return validation_predictions, test_predictions


validation_preds_cat, test_preds_cat = cross_validation_catboost(train_data, test_data, 
                                                                 feature_columns, categorical_columns)



def get_kfold_splits(train_data, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(kfold.split(train_data))

def prepare_data(train_data, test_data, feature_columns, categorical_columns, train_idx, valid_idx):
    X_train, y_train = train_data.loc[train_idx, feature_columns].copy(), train_data.loc[train_idx, "Price"]
    X_valid, y_valid = train_data.loc[valid_idx, feature_columns].copy(), train_data.loc[valid_idx, "Price"]
    X_test = test_data[feature_columns].copy()
    return X_train, y_train, X_valid, y_valid, X_test

def encode_features(X_train, X_valid, X_test, y_train, feature_columns):
    encoder = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')
    for col in feature_columns:
        encoder.fit(X_train[col], y_train)
        X_train[f"Encoded_{col}"] = encoder.transform(X_train[col])
        X_valid[f"Encoded_{col}"] = encoder.transform(X_valid[col])
        X_test[f"Encoded_{col}"] = encoder.transform(X_test[col])
    return X_train, X_valid, X_test

def preprocess_categorical(X_train, X_valid, X_test, categorical_columns):
    for df in [X_train, X_valid, X_test]:
        df[categorical_columns] = df[categorical_columns].fillna('Missing').astype('string').astype('category')
    return X_train, X_valid, X_test

def train_lightgbm(X_train, y_train, X_valid, y_valid, all_model_features, categorical_columns):
    model = LGBMRegressor(
        n_estimators=693,
        max_depth=8,
        colsample_bytree=0.5075,
        subsample=0.8331,
        learning_rate=0.07,
        min_child_samples=23,
        random_state=42,
        verbose=-1,
        device='gpu'
    )
    model.fit(
        X_train[all_model_features], y_train,
        eval_set=[(X_valid[all_model_features], y_valid)]
    )
    return model

def cross_validation_lgbm(train_data, test_data, feature_columns, categorical_columns, n_splits=5):
    kfold_splits = get_kfold_splits(train_data, n_splits)
    validation_predictions = np.zeros(len(train_data))
    test_predictions = np.zeros(len(test_data))

    for fold, (train_idx, valid_idx) in enumerate(kfold_splits):
        print(f"### Processing Fold {fold+1} ###")
        
        X_train, y_train, X_valid, y_valid, X_test = prepare_data(train_data, test_data, feature_columns, categorical_columns, train_idx, valid_idx)

        X_train, X_valid, X_test = encode_features(X_train, X_valid, X_test, y_train, feature_columns)

        X_train, X_valid, X_test = preprocess_categorical(X_train, X_valid, X_test, categorical_columns)

        all_model_features = feature_columns + [f"Encoded_{col}" for col in feature_columns]

        model = train_lightgbm(X_train, y_train, X_valid, y_valid, all_model_features, categorical_columns)

        validation_predictions[valid_idx] = model.predict(X_valid[all_model_features])
        test_predictions += model.predict(X_test[all_model_features]) / n_splits

    rmse_score = np.sqrt(mean_squared_error(train_data["Price"], validation_predictions))
    print(f"Validation RMSE: {rmse_score}")

    return validation_predictions, test_predictions


validation_preds_lgbm, test_preds_lgbm = cross_validation_lgbm(train_data, test_data, feature_columns, categorical_columns)


def get_kfold_splits(train_data, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(kfold.split(train_data))

def prepare_data(train_data, test_data, feature_columns, categorical_columns, train_idx, valid_idx):
    X_train, y_train = train_data.loc[train_idx, feature_columns].copy(), train_data.loc[train_idx, "Price"]
    X_valid, y_valid = train_data.loc[valid_idx, feature_columns].copy(), train_data.loc[valid_idx, "Price"]
    X_test = test_data[feature_columns].copy()
    return X_train, y_train, X_valid, y_valid, X_test

def encode_features(X_train, X_valid, X_test, y_train, feature_columns):
    encoder = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')
    for col in feature_columns:
        encoder.fit(X_train[col], y_train)
        X_train[f"Encoded_{col}"] = encoder.transform(X_train[col])
        X_valid[f"Encoded_{col}"] = encoder.transform(X_valid[col])
        X_test[f"Encoded_{col}"] = encoder.transform(X_test[col])
    return X_train, X_valid, X_test

def preprocess_categorical(X_train, X_valid, X_test, categorical_columns):
    for df in [X_train, X_valid, X_test]:
        df[categorical_columns] = df[categorical_columns].fillna('Missing').astype('string').astype('category')
    return X_train, X_valid, X_test

def train_xgboost(X_train, y_train, X_valid, y_valid, all_model_features, categorical_columns):
    model = XGBRegressor(
        n_estimators=700,
        max_depth=9,
        colsample_bytree=0.8,
        subsample=0.6,
        learning_rate=0.08,
        min_child_weight=20,
        random_state=42,
        verbosity=0,
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        enable_categorical=True
    )
    model.fit(
        X_train[all_model_features], y_train,
        eval_set=[(X_valid[all_model_features], y_valid)],
        verbose=False,
        early_stopping_rounds=50
    )
    return model
def cross_validation_xgb(train_data, test_data, feature_columns, categorical_columns, n_splits=5):
    kfold_splits = get_kfold_splits(train_data, n_splits)
    validation_predictions = np.zeros(len(train_data))
    test_predictions = np.zeros(len(test_data))

    for fold, (train_idx, valid_idx) in enumerate(kfold_splits):
        print(f"### Processing Fold {fold+1} ###")
        
        X_train, y_train, X_valid, y_valid, X_test = prepare_data(train_data, test_data, feature_columns, categorical_columns, train_idx, valid_idx)

        X_train, X_valid, X_test = encode_features(X_train, X_valid, X_test, y_train, feature_columns)

        X_train, X_valid, X_test = preprocess_categorical(X_train, X_valid, X_test, categorical_columns)

        all_model_features = feature_columns + [f"Encoded_{col}" for col in feature_columns]

        model = train_xgboost(X_train, y_train, X_valid, y_valid, all_model_features, categorical_columns)

        validation_predictions[valid_idx] = model.predict(X_valid[all_model_features])
        test_predictions += model.predict(X_test[all_model_features]) / n_splits

    rmse_score = np.sqrt(mean_squared_error(train_data["Price"], validation_predictions))
    print(f"Validation RMSE: {rmse_score}")

    return validation_predictions, test_predictions


validation_preds_xgb, test_preds_xgb = cross_validation_xgb(train_data, test_data, feature_columns, categorical_columns, n_splits=5)


ensemble_val = (0.4 * validation_preds_lgbm) + (0.5 * validation_preds_cat) + (0.1 * validation_preds_xgb)
ensemble_test_preds = (0.4 * test_preds_lgbm) + (0.5 * test_preds_cat) + (0.1 * test_preds_xgb)

rmse_lgbm = np.sqrt(mean_squared_error(train_data["Price"], validation_preds_lgbm))
rmse_catboost = np.sqrt(mean_squared_error(train_data["Price"], validation_preds_cat))
rmse_xgboost = np.sqrt(mean_squared_error(train_data["Price"], validation_preds_xgb))
rmse_ensemble = np.sqrt(mean_squared_error(train_data["Price"], ensemble_val))

print(f"LGBM RMSE: {rmse_lgbm}")
print(f"CatBoost RMSE: {rmse_catboost}")
print(f"XGBoost RMSE: {rmse_xgboost}")
print(f"Ensemble Model RMSE: {rmse_ensemble}")


sub = pd.DataFrame({"id": test_data.index, "Price": ensemble_test_preds})
sub.head()


sub.to_csv("sub_19.csv", index=False)







