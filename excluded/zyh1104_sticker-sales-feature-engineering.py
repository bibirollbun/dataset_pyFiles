import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import logging

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')
import gc


#load data
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
gc.collect()


#Set id as index

train.set_index('id', inplace=True)
test.set_index('id', inplace=True)


# Delete rows containing NaN in the num_sold column
train = train.dropna(subset=['num_sold'])


train.shape


test.shape


def process_date_features(df):
    df['date'] = pd.to_datetime(df['date'])

    #df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.day_name()
    df['week_of_year'] = df['date'].dt.isocalendar().week

    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)  # Assume a maximum of 31 days per month
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    #df['year_sin'] = np.sin(2 * np.pi * df['year'] / 7)  # Let's say we have a seven-year cycle
    #df['year_cos'] = np.cos(2 * np.pi * df['year'] / 7)

    #df['group'] = (df['year'] - 2020) * 48 + df['month'] * 4 + df['day'] // 7

    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    df.drop('date', axis=1, inplace=True)

    return df

train = process_date_features(train)
test = process_date_features(test)



train.head()


test.head()


train.info()


def get_categorical_features(df):
    return df.select_dtypes(include=['object', 'bool']).columns.tolist()

cat_cols = get_categorical_features(train)
print(f"Categorical Features: {cat_cols}")


#Average MAPE across folds: 0.0423
"""
def one_hot_encode(train, test, cat_columns):
    combined = pd.concat([train[cat_columns], test[cat_columns]], axis=0)
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoder.fit(combined)
    train_encoded = encoder.transform(train[cat_columns])
    test_encoded = encoder.transform(test[cat_columns])
    train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(cat_columns), index=train.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(cat_columns), index=test.index)
    train = pd.concat([train.drop(cat_columns, axis=1), train_encoded_df], axis=1)
    test = pd.concat([test.drop(cat_columns, axis=1), test_encoded_df], axis=1)
    
    return train, test

train, test = one_hot_encode(train, test, cat_cols)
"""


#Average MAPE across folds: 0.0424
"""
def target_encode_inplace(train, test, target_col, cat_cols):
   
    encoder = TargetEncoder(cols=cat_cols)

    train[cat_cols] = encoder.fit_transform(train[cat_cols], train[target_col])
    test[cat_cols] = encoder.transform(test[cat_cols])

target_encode_inplace(train, test, target_col='num_sold', cat_cols=cat_cols)
test = test.reindex(columns=train.columns, fill_value=0)
"""


# Average MAPE across folds: 0.0422
def label_encode_train_test(train, test, cat_cols):
    label_encoders = {}
    
    for feature in cat_cols:
        le = LabelEncoder()
        train[feature] = le.fit_transform(train[feature])
        test[feature] = le.transform(test[feature])
        label_encoders[feature] = le
    
    return train, test, label_encoders

train, test, train_label_encoders = label_encode_train_test(train, test, cat_cols)


train.head()


test.head()



logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

X = train.drop(columns=['num_sold'])
y = np.log1p(train['num_sold'])
test = test[X.columns]

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

def cross_val_lgbm_mape(X, y, test, n_splits=5, **params):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mape_scores = []
    preds = []

    for train_index, valid_index in kf.split(X):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        model = lgb.LGBMRegressor(
            **params,
            #n_estimators=2000,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_valid)
        score = mape(np.expm1(y_valid), np.expm1(y_pred))
        mape_scores.append(score)

        preds.append(model.predict(test))

    test_preds_mean = np.mean(preds, axis=0)

    return np.mean(mape_scores), test_preds_mean

model_params = {
    "objective": "regression",
    "metric": "mape",
    "verbose": -1  
}

"""
model_params = {
    'n_estimators': 3946, 
    'learning_rate': 0.10203344298643195, 
    'max_depth': 12, 
    'num_leaves': 20, 
    'min_child_samples': 39, 
    'subsample': 0.7786665459484634, 
    'colsample_bytree': 0.7352055562065795, 
    'reg_alpha': 0.2840216195298897, 
    'reg_lambda': 6.583320975256993, 
    "verbosity": -1
}
"""

average_mape, lgb_preds = cross_val_lgbm_mape(X, y, test, n_splits=5, **model_params)
print(f"Average MAPE across folds: {average_mape:.4f}")


test_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
lgb_preds_original = np.expm1(lgb_preds)
submission = pd.DataFrame({
    'id': test_submission['id'],
    'num_sold': lgb_preds_original
})
submission.to_csv('submission_lgb.csv', index=False)
print("Submission file created:")
print(submission.head())

