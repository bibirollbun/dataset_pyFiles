import pandas as pd, numpy as np
from cuml.preprocessing import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
train = pd.concat([train, train_extra], axis=0, ignore_index=True)

test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col='id')

target = "Price"

features = [col for col in train.columns if col != target]
CATS = [col for col in train.columns if col not in ["Price", "Compartments", "Weight Capacity (kg)"]]


train.head()


def train_and_predict(Model, params, n_folds = 5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
        print(f"### Fold {fold+1} ###")
    
        X_train, y_train = train.loc[train_idx, features].copy(), train.loc[train_idx, target]
        X_valid, y_valid = train.loc[valid_idx, features].copy(), train.loc[valid_idx, target]
        X_test = test[features].copy()
    
        TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')
        for col in features:
            TE.fit(X_train[col], y_train)
            X_train[f"TE_{col}"] = TE.transform(X_train[col])
            X_valid[f"TE_{col}"] = TE.transform(X_valid[col])
            X_test[f"TE_{col}"] = TE.transform(X_test[col])
    
        X_train[CATS] = X_train[CATS].fillna('Missing').astype('category')
        X_valid[CATS] = X_valid[CATS].fillna('Missing').astype('category')
        X_test[CATS] = X_test[CATS].fillna('Missing').astype('category')
    
        all_features = features + [f"TE_{col}" for col in features]
    
        model = Model(**params)
        try:
            model.fit(
                X_train[all_features], y_train,
                eval_set=[(X_valid[all_features], y_valid)],
                verbose=500
            )

        except Exception as e:
            # LGBM doesn't have verbose parameter under the model.fit
            model.fit(
                X_train[all_features], y_train,
                eval_set=[(X_valid[all_features], y_valid)]
            )

        oof_preds[valid_idx] = model.predict(X_valid[all_features])
        test_preds += model.predict(X_test[all_features]) / n_folds

    rmse = np.sqrt(mean_squared_error(train[target], oof_preds))
    print(f"Validation RMSE: {rmse}")
    return test_preds


xgb_params = {
    "device": "cuda",
    "max_depth": 3,
    "colsample_bytree": 0.5,
    "subsample": 0.8,
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "min_child_weight": 80,
    "enable_categorical": True,
    "verbosity" : 2
}

xgb_preds = train_and_predict(XGBRegressor, xgb_params)


cat_params = {
     'cat_features': CATS,
     'early_stopping_rounds': 60,
     'eval_metric': "RMSE",
     'n_estimators' : 3000,
     'objective': 'RMSE', 
     'depth': 4,
     'min_data_in_leaf': 20,
     'l2_leaf_reg': 0.3349374242775052,
     'bagging_temperature': 0.8315027960954179, 
     'random_strength': 0.309798135191685,
     'learning_rate': 0.01,
     'max_bin': 8000,
     'bootstrap_type': 'Poisson',
     "task_type": "GPU",
}

cat_preds = train_and_predict(CatBoostRegressor, cat_params)


lgb_params = {
    'random_state': 42,
      'early_stopping_round': 60,
      'categorical_feature': CATS,
      'verbosity':-1,
      'boosting_type': 'gbdt',
      'n_estimators': 3000,
      'eval_metric': 'rmse',
      'objective': 'regression_l2',
      'max_depth': 12,
      'num_leaves': 8,
      'min_child_samples': 21,
      'min_child_weight': 11,
      'colsample_bytree': 0.4759506289207658, 
      'reg_alpha': 0.28461417683987383, 
      'reg_lambda': 0.6555944495127437,
      'learning_rate': 0.01,
    "device": "gpu",
}

lgb_preds = train_and_predict(LGBMRegressor, lgb_params)


# Average

test_preds = np.mean([lgb_preds, cat_preds, xgb_preds], axis = 0)
test_preds


sub = pd.DataFrame({"id": test.index, "Price": cat_preds})
sub.to_csv("submission.csv", index=False)
sub.head()

