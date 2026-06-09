# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


print(f'Training Data Shape: {train_df.shape}')
print(f'Testing Data Shape: {test_df.shape}')


train_df.head()


train_df.info()


train_df['Sex'].unique()


train_df.drop('id', axis=1).describe()


train_df.isna().sum()


numeric_cols = train_df.drop('id', axis=1).select_dtypes(include='number').columns


corr = train_df[numeric_cols].corr()

plt.figure(figsize=(14, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
plt.show()


fig, axes = plt.subplots(6, 2, figsize=(10, 14))

for i, col in enumerate(numeric_cols.drop('Calories')):
    
    sns.histplot(train_df[col], kde=True, ax=axes[i, 0], color='skyblue')
    axes[i, 0].set_title(f'Histogram of {col}')

    sns.boxplot(x=train_df[col], ax=axes[i, 1], color='skyblue')
    axes[i, 1].set_title(f'Box Plot of {col}')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.countplot(x='Sex', data=train_df, palette='viridis', ax=axes[0])
sns.boxplot(x='Calories', y='Sex', data=train_df, palette='viridis', ax=axes[1])

plt.tight_layout()
plt.show()



numeric_cols = numeric_cols.drop('Calories')


numeric_cols


def encode_sex(df):
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    return df


def cross_cols(df, numeric_cols):
    for col_2 in numeric_cols:
        new_col = f"Sex_{col_2}"
        df[new_col] = df[col_2] * df['Sex'] 
        
    return df


def add_sex_aggregations(df, numeric_cols):
    agg_df = df.groupby('Sex')[numeric_cols]\
        .agg(['mean', 'std', 'max', 'min']).reset_index()

    agg_df.columns = ['Sex'] + [f'{col}_{stat}' for col, stat in agg_df.columns[1:]]

    df = df.merge(agg_df, on='Sex', how='left')

    return df


def bmi(df):
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2 
    return df


def hr_temp_ratio(df):
    df['HR_Temp_Ratio'] = df['Heart_Rate'] / df['Body_Temp']
    return df


def hr_age_ratio(df):
    df['HR_Age_Ratio'] = df['Heart_Rate'] / df['Age']
    return df


def hr_duration_ratio(df):
    df['HR_Durtaion_Ratio'] = df['Heart_Rate'] / df['Duration']
    return df


def temp_age_ratio(df):
    df['Temp_Age_Ratio'] = df['Body_Temp'] / df['Age']
    return df


def temp_duration_ratio(df):
    df['Temp_Duration_Ratio'] = df['Body_Temp'] / df['Duration']
    return df


def age_duration_ratio(df):
    df['Age_Duration_Ratio'] = df['Age'] / df['Duration']
    return df


def age_category(df):
    df['Age_Category'] = pd.cut(df['Age'],
                                bins=[0, 18, 35, 55, 85],
                                labels=['Teen', 'Young_Adult', 'Adult', 'Elderly'])

    encoder = LabelEncoder()
    df['Age_Category'] = encoder.fit_transform(df['Age_Category'])
    return df


def bmi_age(df):
    df['BMI_Age'] = df['BMI'] * df['Age']
    return df


def heartrate_bodytemp(df):
    df['HeartRate_BodyTemp'] = df['Heart_Rate'] * df['Body_Temp']
    return df


def duration_squared(df):
    df['Duration_Squared'] = df['Duration'] ** 2
    return df


train_df = encode_sex(train_df)
test_df = encode_sex(test_df)

train_df = cross_cols(train_df, numeric_cols)
test_df = cross_cols(test_df, numeric_cols)

train_df = add_sex_aggregations(train_df, numeric_cols)
test_df = add_sex_aggregations(test_df, numeric_cols)

train_df = bmi(train_df)
test_df = bmi(test_df)

train_df = hr_temp_ratio(train_df)
test_df = hr_temp_ratio(test_df)

train_df = hr_age_ratio(train_df)
test_df = hr_age_ratio(test_df)

train_df = hr_duration_ratio(train_df)
test_df = hr_duration_ratio(test_df)

train_df = temp_age_ratio(train_df)
test_df = temp_age_ratio(test_df)

train_df = temp_duration_ratio(train_df)
test_df = temp_duration_ratio(test_df)

train_df = age_duration_ratio(train_df)
test_df = age_duration_ratio(test_df)

train_df = age_category(train_df)
test_df = age_category(test_df)

train_df = bmi_age(train_df)
test_df = bmi_age(test_df)

train_df = heartrate_bodytemp(train_df)
test_df = heartrate_bodytemp(test_df)

train_df = duration_squared(train_df)
test_df = duration_squared(test_df)


cat_features = ['Sex', 'Age_Category']
for col in cat_features:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')


train_df.info()


test_df.info()


X = train_df.drop('Calories', axis=1)
y = train_df['Calories']


y_log = np.log1p(y)


# Hyperparameters
xgb_params = {
    'max_depth': 8,
    'colsample_bytree': 0.9253237022869346,
    'subsample': 0.8918610216635463,
    'n_estimators': 2821,
    'learning_rate': 0.011328587649473263,
    'gamma': 0.010726096438638005,
    'reg_lambda': 25.957010875964592,
    'reg_alpha': 1.5959377083913109,
    'eval_metric': 'rmse',
    'enable_categorical': True,
    'random_state': 42,
    'early_stopping_rounds': 100,
    'tree_method': 'hist',
    'device': 'cuda'
}

cat_params = {
    'iterations': 2687,
    'learning_rate': 0.013502135568553198,
    'depth': 11,
    'loss_function': 'RMSE',
    'l2_leaf_reg': 3.8745871477021074,
    'random_seed': 42,
    'eval_metric': 'RMSE',
    'early_stopping_rounds': 200,
    'cat_features': cat_features,
    'verbose': 1000,
    'task_type': 'GPU'
}

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.01594400757385793,
    "n_estimators": 2418, 
    "num_leaves": 103,  
    "max_depth": 9, 
    "feature_fraction": 0.8059292523242455, 
    "bagging_fraction": 0.9833172989350026,
    "bagging_freq": 2,
    "lambda_l1":  2.972869029247061,
    "lambda_l2": 64.91782113630072,
    "random_state": 42,
    "verbosity": -1,
    'enable_categorical': True,
    "force_col_wise": True
}


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


kf = KFold(n_splits=10, shuffle=True, random_state=42)


def train(X, test_df):
    oof_preds_xgb = np.zeros(len(X))
    oof_preds_cat = np.zeros(len(X))
    oof_preds_lgb = np.zeros(len(X))
    
    test_preds_xgb = []
    test_preds_cat = []
    test_preds_lgb = []
    
    xgb_rmsle_scores = []
    cat_rmsle_scores = []
    lgb_rmsle_scores = []
    avg_rmsle_scores = []
    
    xgb_importances = []
    lgb_importances = []
    cat_importances = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1} ---")
    
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
    
        model_xgb = XGBRegressor(**xgb_params)
        model_lgb = LGBMRegressor(**lgb_params)
        model_cat = CatBoostRegressor(**cat_params)

        # Train and predict XGB
        model_xgb.fit(X_train, y_train, 
                      eval_set=[(X_val, y_val)], 
                      verbose=False
                     )
        xgb_score = model_xgb.predict(X_val)
        oof_preds_xgb[val_idx] = xgb_score
        test_preds_xgb.append(model_xgb.predict(test_df))
        xgb_importances.append(model_xgb.feature_importances_)

         # Train and predict CAT
        model_cat.fit(X_train, y_train, 
                      eval_set=(X_val, y_val),
                      cat_features=cat_features,
                      verbose=0)
        cat_score = model_cat.predict(X_val)
        oof_preds_cat[val_idx] = cat_score
        test_preds_cat.append(model_cat.predict(test_df))
        cat_importances.append(model_cat.get_feature_importance())
    
        # Train and predict LGB
        model_lgb.fit(X_train, y_train, 
                      eval_set=[(X_val, y_val)]
                     )
        lgb_score = model_lgb.predict(X_val)
        oof_preds_lgb[val_idx] = lgb_score
        test_preds_lgb.append(model_lgb.predict(test_df))
        lgb_importances.append(model_lgb.feature_importances_)
    
    
        y_val_true = np.expm1(y_val)
        xgb_rmsle = rmsle(y_val_true, np.expm1(xgb_score))
        cat_rmsle = rmsle(y_val_true, np.expm1(cat_score))
        lgb_rmsle = rmsle(y_val_true, np.expm1(lgb_score))
        avg_score = (np.expm1(xgb_score) + np.expm1(cat_score) + np.expm1(lgb_score)) / 3
        avg_rmsle = rmsle(y_val_true, avg_score)
    
        xgb_rmsle_scores.append(xgb_rmsle)
        cat_rmsle_scores.append(cat_rmsle)
        lgb_rmsle_scores.append(lgb_rmsle)
        avg_rmsle_scores.append(avg_rmsle)
    
        print(f'âœ… Average RMSLE: {avg_rmsle:.5f}')
        print(f'ðŸ“Š XGBoost RMSLE: {xgb_rmsle:.5f}')
        print(f'ðŸ“Š CatBoost RMSLE: {cat_rmsle:.5f}')
        print(f'ðŸ“Š LightGBM RMSLE: {lgb_rmsle:.5f}')
    
    stacked_train = np.vstack([oof_preds_xgb, oof_preds_cat, oof_preds_lgb]).T
    stacked_test = np.vstack([
        np.mean(test_preds_xgb, axis=0),
        np.mean(test_preds_cat, axis=0),
        np.mean(test_preds_lgb, axis=0)
    ]).T

    features = X.columns
    xgb_mean_importance = np.mean(xgb_importances, axis=0)
    lgb_mean_importance = np.mean(lgb_importances, axis=0)
    cat_mean_importance = np.mean(cat_importances, axis=0)
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'XGB Importance': xgb_mean_importance,
        'LGB Importance': lgb_mean_importance,
        'CatBoost Importance': cat_mean_importance
    })
    
    importance_df['Mean Importance'] = importance_df[
        ['XGB Importance', 'LGB Importance', 'CatBoost Importance']
    ].mean(axis=1)
    
    importance_df = importance_df.sort_values(by='Mean Importance', ascending=False)
    print("\nðŸŽ¯ Top 20 Important Features:\n", importance_df.head(20))

    return stacked_train, stacked_test


stacked_train, stacked_test = train(X, test_df)


# low_importance = importance_df[importance_df['Mean Importance'] < 1e-3]['Feature'].tolist()
# X_reduced = X.drop(columns=low_importance)
# test_df_reduced = test_df.drop(columns=low_importance)


# stacked_train, stacked_test = train(X_reduced, test_df_reduced)


from sklearn.linear_model import ElasticNetCV

meta_model = ElasticNetCV(alphas=[0.01, 0.1, 1, 10], l1_ratio=[.1, .5, .9], cv=5)
meta_model.fit(stacked_train, y_log)

meta_prids = meta_model.predict(stacked_test)

final_preds = np.expm1(meta_prids)


from sklearn.linear_model import RidgeCV

ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0, 100.0], cv=5)
ridge.fit(stacked_train, y_log)  

ridge_preds = ridge.predict(stacked_test)

final_preds = np.expm1(ridge_preds)


submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': final_preds
})
submission.to_csv("submission.csv", index=False)




