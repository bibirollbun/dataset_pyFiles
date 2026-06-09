# Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import *
from sklearn.model_selection import *
from matplotlib import pyplot as plt
import seaborn as sns
import optuna 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

# Finding File Paths

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

train.dropna(inplace = True)


train



# 1. Group and convert to a DataFrame
df_agg = train.groupby(['date', 'country'], as_index=False)['num_sold'].sum()

# 2. Plot using seaborn
plt.figure(figsize=(28, 6))
sns.lineplot(data=df_agg, x='date', y='num_sold', hue = 'country')

# 3. Customize labels and title
plt.title('Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Products Sold')
plt.grid(True)

plt.show()



plt.figure(figsize=(28, 6)) # credit https://www.kaggle.com/code/sunilkumarmuduli/sticker-time-series-eda-optuna-0-008-explain
train.groupby('date')['num_sold'].sum().plot(title='Total Sales Over Time', xlabel='Date', ylabel='Number of Products Sold')
plt.grid()
plt.show()


def date_extraction(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month.astype('str')
    df['day'] = df['date'].dt.day

    df["qtr"] = df["date"].dt.quarter.astype('str')
    df["day_of_week"] = df["date"].dt.dayofweek.astype('str')
    df["week_nb"] = df["date"].dt.isocalendar().week.astype('str')
    df["year"] = df["date"].dt.year

    # Features below credit: https://www.kaggle.com/code/ravi20076/playgrounds5e01-public-baseline-v1#MODEL-TRAINING

    df['group']       = (df['year'] - 2010 ) *48 + df['month'].astype(int) * 4 + df['day'].astype(int) // 7
    
    df["month_sin"]   = np.sin(df['month'].astype(int) * (2 * np.pi / 12))
    df["month_cos"]   = np.cos(df['month'].astype(int) * (2 * np.pi / 12))
    df["day_sin"]     = np.sin(df['day'] * (2 * np.pi / 365))
    df["day_cos"]     = np.cos(df['day'] * (2 * np.pi / 365))
    df["week_sin"]    = np.sin(df["week_nb"].astype(int) * (2 * np.pi/ 53))
    df["week_cos"]    = np.cos(df["week_nb"].astype(int) * (2 * np.pi/ 53))       
        

date_extraction(train)
date_extraction(test)


# Sort by year (ascending), country (ascending), and num_sold (descending)
# then group by both 'year' and 'country', and take the top 10 rows for each group.
top_10_each_year_country = (
    train
    .sort_values(['year', 'country', 'num_sold'], ascending=[True, True, False])
    .groupby(['year', 'country'])
    .head(1)
)

top_10_each_year_country



# Sort by year, country, and num_sold
top_10_each_year_country = (
    train
    .sort_values(['year', 'country', 'num_sold'], ascending=[True, True, False])
    .groupby(['year', 'country'])
    .head(10)
    .assign(
        # Format the datetime column as YYYY-MM-DD
        date_str=lambda df: df['date'].dt.strftime('%m-%d')
    )
)

# Select and display only relevant columns
top_10_each_year_country[['year', 'country', 'date_str', 'num_sold']]




top_10_each_year_country['date_str'].nunique()


# 1. Compute the annual average per country and store it in a new column 'yearly_avg'.
train_eda = train.copy()

train_eda['yearly_avg'] = train_eda.groupby(['year', 'country'])['num_sold'].transform('mean')

# 2. Sort the data and pick top 10 per (year, country).
top_10_each_year_country = (
    train_eda
    .sort_values(['year', 'country', 'num_sold'], ascending=[True, True, False])
    .groupby(['year', 'country'])
    .head(10)
    .assign(
        # 3. Format the date as YYYY-MM-DD
        date_str=lambda df: df['date'].dt.strftime('%m-%d'),
        # 4. Calculate difference from the yearly average of that country
        diff_from_baseline=lambda df: df['num_sold'] - df['yearly_avg']
    )
)

# 5. Select desired columns for viewing
result = top_10_each_year_country[
    ['year', 'country', 'date_str', 'num_sold', 'yearly_avg', 'diff_from_baseline']
]

result.sort_values('diff_from_baseline', ascending = False)



result[['date_str']].value_counts().head(30)


# getting the 29 most important dates

important = result[['date_str']].value_counts().head(29).index.tolist()
important


def special_days(df):

    df['date_mo_da'] = df['month'].astype(str) + '-' + train['day'].astype(str)
    df['new_year'] = df['date_mo_da'].apply(lambda x: 1 if x in ['12-26', '12-27', '12-28', '12-29', '12-30', '12-31', '1-1', '1-2'] else 0)
    df['early_mid_jan'] = df['date_mo_da'].apply(lambda x: 1 if x in ['1-5', '1-6', '1-7', '1-8', '1-9', '1-10'] else 0)
    

special_days(train)
special_days(test)


def gdp_feat(df):

    gdp_ppp_mapping = {
        "Canada": 56000,     # Approx. range: 55kâ€“57k
        "Finland": 53000,    # Approx. range: 52kâ€“54k
        "Italy": 46000,      # Approx. range: 45kâ€“48k
        "Kenya": 5500,       # Approx. range (PPP): 5kâ€“6k
        "Norway": 85000,     # Approx. range: 80kâ€“90k
        "Singapore": 115000  # Approx. range: 100kâ€“130k
    }
    
    df['GDP_per_capita_PPP'] = df['country'].map(gdp_ppp_mapping)

gdp_feat(train)
gdp_feat(test)


# Cat(egorical) and Cont(inuous) columns identification

train.drop('date', axis = 1, inplace = True)
test.drop('date', axis = 1, inplace = True)

train.drop('id', axis = 1, inplace = True)
test.drop('id', axis = 1, inplace = True)

train.drop('date_mo_da', axis = 1, inplace = True)
test.drop('date_mo_da', axis = 1, inplace = True)

cats = test.select_dtypes(include=["object_"]).columns.tolist()
conts = [col for col in test.columns if col not in cats]


def pp(df):
    
    for col in cats:
        df[col] = df[col].astype('category')

    for col in conts:
        df[col] = df[col].astype('float32')

pp(train)
pp(test)


X = train.drop('num_sold', axis = 1).copy().reset_index(drop = True)
y = train['num_sold'].copy().reset_index(drop = True)

y = np.log1p(y)


lgbm = lgb.LGBMRegressor(verbose = -1)


def cross_val(model, X, y):
    kfold = KFold(n_splits = 5)
    scores = []

    oof_preds = np.zeros(len(X))

    for i, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train = X.iloc[train_idx, :].copy()
        X_val = X.iloc[val_idx, :].copy()

        y_train = y[train_idx].copy()
        y_val = y[val_idx].copy()

        model.fit(X_train, y_train)
        fold_preds = model.predict(X_val)

        oof_preds[val_idx] = fold_preds


        score = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(fold_preds))
        scores.append(score)

        print(f"Fold {i} Score: {score}")

    print(f"Overall CV score: {np.mean(scores)}")

    return np.mean(scores), oof_preds


score, off_preds = cross_val(lgbm, X, y)


score


lgbm.fit(X, y)

preds = lgbm.predict(test)


lgb.plot_importance(lgbm, max_num_features=20, importance_type='split') 
plt.title("Feature Importances")
plt.show()


sub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


X.columns


sub['num_sold'] = np.expm1(preds) * 1.01 # predicting with the log transformation tends to under-predict


sub.to_csv('submission.csv', index = False)


sub

