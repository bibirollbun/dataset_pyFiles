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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)


train_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')


train_df.sample(5)


train_df.info()


train_df.isna().sum()


#Filling the null values


train_df['sale_nbr'] = train_df['sale_nbr'].fillna(train_df['sale_nbr'].median())

test_df['sale_nbr'] = test_df['sale_nbr'].fillna(test_df['sale_nbr'].median())


train_df['subdivision'] = train_df['subdivision'].fillna(train_df['subdivision'].mode()[0])

test_df['subdivision'] = test_df['subdivision'].fillna(test_df['subdivision'].mode()[0])



train_df['submarket'] = train_df['submarket'].fillna(train_df['submarket'].mode()[0])

test_df['submarket'] = test_df['submarket'].fillna(test_df['submarket'].mode()[0])


train_df.isna().sum()


train_df.select_dtypes('object').columns


train_df.select_dtypes(include=['int','float']).columns


train_df['sale_date'] = pd.to_datetime(train_df['sale_date'])

train_df["sale_year"] = train_df["sale_date"].dt.year
train_df["sale_month"] = train_df["sale_date"].dt.month
train_df["sale_day"] = train_df["sale_date"].dt.day



test_df['sale_date'] = pd.to_datetime(test_df['sale_date'])
test_df["sale_year"] = test_df["sale_date"].dt.year
test_df["sale_month"] = test_df["sale_date"].dt.month
test_df["sale_day"] = test_df["sale_date"].dt.day


# Building age and renovation
train_df["age_at_sale"] = train_df["sale_year"] - train_df["year_built"]
train_df["reno_age_at_sale"] = train_df["sale_year"] - train_df["year_reno"]
train_df["is_renovated"] = (train_df["year_reno"] > train_df["year_built"]).astype(int)


test_df["age_at_sale"] = test_df["sale_year"] - test_df["year_built"]
test_df["reno_age_at_sale"] = test_df["sale_year"] - test_df["year_reno"]
test_df["is_renovated"] = (test_df["year_reno"] > test_df["year_built"]).astype(int)



# Size ratios
train_df["lot_sqft_ratio"] = train_df["sqft"] / (train_df["sqft_lot"] + 1)
train_df["garage_total"] = train_df["garb_sqft"] + train_df["gara_sqft"]
train_df["bath_total"] = train_df["bath_full"] + train_df["bath_3qtr"] + 0.5 * train_df["bath_half"]
train_df["sqft_per_bed"] = train_df["sqft"] / (train_df["beds"] + 1)
train_df["sqft_per_bath"] = train_df["sqft"] / (train_df["bath_total"] + 1)




test_df["lot_sqft_ratio"] = test_df["sqft"] / (test_df["sqft_lot"] + 1)
test_df["garage_total"] = test_df["garb_sqft"] + test_df["gara_sqft"]
test_df["bath_total"] = test_df["bath_full"] + test_df["bath_3qtr"] + 0.5 * test_df["bath_half"]
test_df["sqft_per_bed"] = test_df["sqft"] / (test_df["beds"] + 1)
test_df["sqft_per_bath"] = test_df["sqft"] / (test_df["bath_total"] + 1)


train_df.info()


cols = train_df.select_dtypes('object').columns


import seaborn as sns


sns.set_style("whitegrid")

for col in cols:
    plt.figure(figsize=(10, 6))

    plt.hist(train_df[col], bins=30, orientation='horizontal', color='skyblue', edgecolor='black')

    plt.title(f'Distribution of {col}', fontsize=14)
    plt.ylabel(col, fontsize=12)
    plt.xlabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.show()



train_df.drop(columns=['sale_warning','id','sale_date','join_status'],inplace=True)
ide = test_df['id'].copy()
test_df.drop(columns=['sale_warning','id','sale_date','join_status'],inplace=True)


cat_col = train_df.select_dtypes('object').columns


from sklearn.preprocessing import OrdinalEncoder

ord = OrdinalEncoder()

for col in cat_col:
    train_df[col] = ord.fit_transform(train_df[[col]])
    test_df[col] = ord.fit_transform(test_df[[col]])


train_df.select_dtypes('object').columns


train_df.corr()['sale_price'].sort_values(ascending=False)


plt.figure(figsize=(25, 20))  
sns.set(font_scale=0.9)      

sns.heatmap(
    train_df.corr(numeric_only=True), 
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={'shrink': 0.6}  
)

plt.title("Correlation Heatmap", fontsize=18)
plt.tight_layout()
plt.show()



train_df.drop(columns=['sale_day'],inplace=True)
test_df.drop(columns=['sale_day'],inplace=True)


X = train_df.drop(columns=['sale_price'])
y = train_df['sale_price']


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


import xgboost as xgb

xg = xgb.XGBRegressor(n_estimators = 2000,learning_rate = 0.06)

xg.fit(X_train,y_train)

from sklearn.metrics import r2_score

y_pred = xg.predict(X_test)

r2_score(y_test,y_pred)


import lightgbm
lgm = lightgbm.LGBMRegressor(n_estimators = 2000,learning_rate = 0.06)
lgm.fit(X_train,y_train)
y_pred = lgm.predict(X_test)
r2_score(y_test,y_pred)


import lightgbm as lgb
lower_model = lgb.LGBMRegressor(
    objective='quantile',
    alpha=0.1,
    n_estimators=2000,
    learning_rate=0.06
)

lower_model.fit(X,y)






upper_model = lgb.LGBMRegressor(
    objective='quantile',
    alpha=0.95,
    n_estimators=2000,
    learning_rate=0.06
)

upper_model.fit(X,y)


pi_lower = lower_model.predict(test_df)
pi_upper = upper_model.predict(test_df)


pi_lower


pi_upper


submission = pd.DataFrame({
    'id':ide,
    'pi_lower':pi_lower,
    'pi_upper':pi_upper
})

submission.to_csv('submission.csv',index=False)

