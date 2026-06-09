import pandas as pd
import numpy as np
from cuml.preprocessing import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings

warnings.simplefilter('ignore')

# ASK: Define the problem and target variable
target = "Price"
FOLDS = 5
xgb_params = {
    "device": "cuda",
    "max_depth": 7,
    "colsample_bytree": 0.7,
    "subsample": 0.85,
    "n_estimators": 3000,
    "learning_rate": 0.015,
    "min_child_weight": 50,
    "enable_categorical": True,
    "reg_lambda": 1.2,
    "reg_alpha": 0.8
}



# PREPARE: Load and clean data
def load_data():
    train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
    train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
    test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col='id')
    return pd.concat([train, train_extra], axis=0, ignore_index=True), test

def preprocess(df):
    na_features = ['Material', 'Style', 'Brand', 'Size', 'Waterproof', 'Color', 'Laptop Compartment']
    df[na_features] = df[na_features].fillna('NaN')
    for col in na_features:
        df[f'_NaN_{col}'] = (df[col] == 'NaN').astype(int)
    df['_Total_NaNs'] = df.filter(like='_NaN_').sum(axis=1)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())  # Fill only numeric NaNs with median
    
    return df

train, test = load_data()
train, test = preprocess(train), preprocess(test)

features = [col for col in train.columns if col != target]
CATS = [col for col in train.columns if col not in [target, "Weight Capacity (kg)", "_Total_NaNs"]]


# PROCESS: Prepare cross-validation
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)



# ANALYZE: Train model using KFold validation
for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
    print(f"### Fold {fold+1} ###")
    X_train, y_train = train.loc[train_idx, features].copy(), train.loc[train_idx, target]
    X_valid, y_valid = train.loc[valid_idx, features].copy(), train.loc[valid_idx, target]
    X_test = test[features].copy()

    TE = TargetEncoder(n_folds=5, smooth=15, split_method='random', stat='mean')
    for col in features:
        TE.fit(X_train[col], y_train)
        X_train[f"TE_{col}"] = TE.transform(X_train[col])
        X_valid[f"TE_{col}"] = TE.transform(X_valid[col])
        X_test[f"TE_{col}"] = TE.transform(X_test[col])
    
    X_train[CATS] = X_train[CATS].fillna('--').astype('category')
    X_valid[CATS] = X_valid[CATS].fillna('--').astype('category')
    X_test[CATS] = X_test[CATS].fillna('--').astype('category')
    
    model = XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=500
    )
    
    oof_preds[valid_idx] = model.predict(X_valid)
    test_preds += model.predict(X_test) / FOLDS


# SHARE: Evaluate model
rmse = np.sqrt(mean_squared_error(train[target], oof_preds))
print(f"Validation RMSE: {rmse}")


# ACT: Save predictions
sub = pd.DataFrame({"id": test.index, "Price": test_preds})
sub.to_csv("submission.csv", index=False)
print(sub.head())


