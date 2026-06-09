import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})
train = train.drop_duplicates(subset=train.columns).reset_index(drop=True)
cols = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
train = train.groupby(by=cols)['Calories'].min().reset_index()


for df in [train, test]:
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['Intensity'] = df['Heart_Rate'] / df['Duration']


def add_interactions_onehot(df, features, gender_col='Sex'):
    df['Male'] = df[gender_col]
    df['Female'] = 1 - df[gender_col]
    for feat in features:
        df[f'{feat}_x_Male'] = df[feat] * df['Male']
        df[f'{feat}_x_Female'] = df[feat] * df['Female']
    df.drop(['Male', 'Female'], axis=1, inplace=True)
    return df


train = add_interactions_onehot(train, features=['Duration', 'Heart_Rate', 'Body_Temp', 'Age'])
test = add_interactions_onehot(test, features=['Duration', 'Heart_Rate', 'Body_Temp', 'Age'])


def add_categorical_aggregations(df):
    categorical_cols = ['Sex']
    numerical_cols = ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']
    for cat_col in categorical_cols:
        aggs = df.groupby(cat_col)[numerical_cols].agg(['min', 'max'])
        aggs.columns = [f"{cat_col}_{num_col}_{stat}" for num_col, stat in aggs.columns]
        df = df.merge(aggs, on=cat_col, how='left')
    return df


train = add_categorical_aggregations(train)
test = add_categorical_aggregations(test)


common_cols = [col for col in test.columns if col in train.columns and col != 'Calories']
train = train[common_cols + ['Calories']]
test = test[common_cols]


X = train.drop(columns='Calories')
y = np.log1p(train['Calories'])
cat_features = ['Sex']
X[cat_features] = X[cat_features].astype('category')
test[cat_features] = test[cat_features].astype('category')


bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
duration_bins = bins.fit_transform(train[['Duration']]).astype(int).flatten()


catboost_params = {
    'iterations': 3500,
    'learning_rate': 0.02,
    'depth': 12,
    'loss_function': 'RMSE',
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'eval_metric': 'RMSE',
    'early_stopping_rounds': 200,
    'cat_features': cat_features,
    'verbose': 1000,
    'task_type': 'GPU'
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cat_oof = np.zeros(len(train))
cat_test = np.zeros(len(test))
cat_scores = []


for fold, (train_idx, val_idx) in enumerate(skf.split(X, duration_bins)):
    model = CatBoostRegressor(**catboost_params)
    model.fit(X.iloc[train_idx], y.iloc[train_idx], eval_set=(X.iloc[val_idx], y.iloc[val_idx]), use_best_model=True)
    val_preds = np.expm1(model.predict(X.iloc[val_idx]))
    cat_oof[val_idx] = val_preds
    cat_test += np.expm1(model.predict(test)) / skf.n_splits
    score = np.sqrt(mean_squared_error(np.log1p(train.iloc[val_idx]['Calories']), np.log1p(val_preds)))
    cat_scores.append(score)
    print(f"Fold {fold+1} - CatBoost RMSLE: {score:.5f}")


print(f"\nCatBoost RMSLE Mean: {np.mean(cat_scores):.5f} ± {np.std(cat_scores):.5f}")


X_xgb = X.copy()
test_xgb = test.copy()
X_xgb['Sex'] = X_xgb['Sex'].astype(int)
test_xgb['Sex'] = test_xgb['Sex'].astype(int)


xgb_params = {
    'max_depth': 9,
    'colsample_bytree': 0.7,
    'subsample': 0.9,
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'gamma': 0.01,
    'max_delta_step': 2,
    'eval_metric': 'rmse',
    'enable_categorical': False,
    'random_state': 42,
    'early_stopping_rounds': 100,
    'tree_method': 'gpu_hist'
}


kf = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_oof = np.zeros(len(train))
xgb_test = np.zeros(len(test))
xgb_scores = []


for fold, (train_idx, val_idx) in enumerate(kf.split(X_xgb)):
    model = XGBRegressor(**xgb_params)
    model.fit(X_xgb.iloc[train_idx], y.iloc[train_idx], eval_set=[(X_xgb.iloc[val_idx], y.iloc[val_idx])], verbose=False)
    val_preds = np.expm1(model.predict(X_xgb.iloc[val_idx]))
    xgb_oof[val_idx] = val_preds
    xgb_test += np.expm1(model.predict(test_xgb)) / kf.n_splits
    score = np.sqrt(mean_squared_error(np.log1p(train.iloc[val_idx]['Calories']), np.log1p(val_preds)))
    xgb_scores.append(score)
    print(f"Fold {fold+1} - XGBoost RMSLE: {score:.5f}")


print(f"\nXGBoost RMSLE Mean: {np.mean(xgb_scores):.5f} ± {np.std(xgb_scores):.5f}")


submission['Calories'] = np.clip((cat_test + xgb_test) / 2, 1, 314)
submission['Calories'] *= 76.46 / submission['Calories'].median()
submission.to_csv('submission.csv', index=False)
print("\n Final submission.csv saved")

