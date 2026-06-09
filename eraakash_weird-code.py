import pandas as pd
import numpy as np 


train_init = pd.read_csv('../data/train.csv')
train_ext = pd.read_csv('../data/training_extra.csv')

train = pd.concat([train_init, train_ext], axis=0)
train = train.reset_index(drop=True)

test = pd.read_csv('../data/test.csv')


if 'id' in test.columns:
    train.drop('id', axis=1, inplace=True)
    test.drop('id', axis=1, inplace=True)


def add_nan_indicators(df):
    nan_cols = df.columns[df.isnull().any()].tolist()

    for col in nan_cols:
        if col == 'Price':
            continue 
        
        df[col + '_is_nan'] = df[col].isnull().astype(int)

        if df[col].dtype == 'object':
            df[col] = df[col].fillna('missing')

        else:
            df[col] = df[col].fillna(-99)

    return df 


train = add_nan_indicators(train)
test = add_nan_indicators(test)


def OHE(df):
    obj_cols = df.select_dtypes(include='object').columns.tolist()

    # perform OHE, drop_first=True 
    df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    return df

train = OHE(train)
test = OHE(test)


len(train.columns) - len(test.columns) == 1


train.head()


def feature_engineering(df):
    df['Compartments x Weight Capacity'] = df['Compartments'] * df['Weight Capacity (kg)']
    return df

train = feature_engineering(train)
test = feature_engineering(test)


from xgboost import XGBRegressor 
from sklearn.model_selection import cross_val_score as cvs 

X = train.drop(columns=['Price']) 
y = train['Price'] 

model = XGBRegressor(n_jobs=-1) 
model.fit(X, y) 

cv = cvs(model, X, y, cv=5, scoring='neg_mean_squared_error') 
print(np.sqrt(-cv).mean()) 


from lightgbm import LGBMRegressor

lgb = LGBMRegressor(n_jobs=-1, verbosity=-1)
lgb.fit(X, y)

cv = cvs(lgb, X, y, cv=5, scoring='neg_mean_squared_error') 
print(np.sqrt(-cv).mean())


test = test[[col for col in train.columns.tolist() if col != 'Price']]
q = lgb.predict(test)

sample_sub = pd.read_csv('../data/sample_submission.csv')
sample_sub['Price'] = np.round(q) 

sample_sub.to_csv('./lgb_rounded.csv', index=False)

