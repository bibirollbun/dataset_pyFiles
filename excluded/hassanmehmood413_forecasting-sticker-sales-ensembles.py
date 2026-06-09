import pandas as pd
import numpy as np

# For Data Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# For Encoding
from sklearn.preprocessing import OneHotEncoder

# For Feature Engineering
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# For Model Training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Metrics for regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# To save the model
import pickle

# Extra
import importlib
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', 0)            # no line‐wrapping (or use a big number, e.g. 2000)
pd.set_option('display.max_colwidth', None)  # don't truncate long strings

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


df_sample = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')

df_train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'])
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv',  parse_dates=['date'])

# make sure numeric cols are numeric
df_train['num_sold'] = pd.to_numeric(df_train['num_sold'], errors='coerce').astype('Int64')

df_train.head()  # should print without the warnings now



df_test.shape


df_train.shape


df_sample.shape


df_test.head()


df_train.info()


df_train.describe()


df_train.duplicated().sum()


# Droping some null values that we have in the dataset
df_train = df_train.dropna()
# df_test = df_test.dropna() # Removed to prevent data leakage


def format_date(df_train):
    # Ensure the 'date' column is a datetime object
    if 'date' in df_train.columns:
        df_train['date'] = pd.to_datetime(df_train['date'], errors='coerce')  # Convert to datetime
        if df_train['date'].isna().any():
            raise ValueError("The 'date' column contains invalid datetime values.")
    else:
        raise KeyError("The DataFrame does not have a 'date' column.")

    # Extract date-related components
    df_train['year'] = df_train['date'].dt.year
    df_train['month'] = df_train['date'].dt.month
    df_train['day'] = df_train['date'].dt.day
    df_train['dayOfYear'] = df_train['date'].dt.dayofyear
    df_train['weekday'] = df_train['date'].dt.weekday

    return df_train

df_train = format_date(df_train)
df_test = format_date(df_test)


df_train.head()


df_train.info()


df_train = df_train.reset_index(drop=True)

y = df_train['num_sold'].astype('float64')

for c in ['year','month','day','dayOfYear','weekday']:
    df_train[c] = df_train[c].astype('int16')

for col in ['country','store','product']:
    df_train[col] = df_train[col].astype('category')


df_train.info()


def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df_train['season'] = df_train['month'].apply(get_season)
df_test['season'] = df_test['month'].apply(get_season)


df_train.head()


# Shape of data
print(f"The shape of train data is " , df_train.shape)
print(f"The shape of test data is " , df_test.shape)



plt.hist(df_train['num_sold'])


# Check for zeros/negatives
print("zeros:", (df_train['num_sold'] == 0).sum())
print("negatives:", (df_train['num_sold'] < 0).sum())

# Safer transform
df_train['num_sold_log'] = np.log1p(df_train['num_sold'])


plt.hist(df_train['num_sold_log'])


plt.figure(figsize=(28, 6))
df_train.groupby('date')['num_sold_log'].sum().plot(title='Total Sales Over Time', xlabel='Date', ylabel='Number of Products Sold')
plt.grid()
plt.show()


plt.figure(figsize=(20,5))
fig = sns.lineplot(data=df_train,x='country',y='num_sold_log',hue='year')
plt.title('Sales Trends by Country Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Country')
plt.show()
plt.show()


plt.figure(figsize=(20,5))
fig = sns.lineplot(data=df_train,x='year',y='num_sold_log',hue='country')
plt.title('Different countries performed in terms of sales year-over-year.')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Country')
plt.show()


plt.figure(figsize=(20,5))
fig = sns.lineplot(data=df_train,x='month',y='num_sold_log',hue='product')
plt.title('Sales Trends by Product Year-Wise')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Product')
plt.show()


plt.figure(figsize=(20,5))
fig = sns.lineplot(data=df_train,x='year',y='num_sold_log',hue='store')
plt.title('Sales Trends by Stores')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Stores')
plt.show()


sns.histplot(data=df_train, x=df_train['num_sold_log'], bins=10, kde=False)


df_train.head()


df_test.head()


categorical_features = ['country', 'product', 'store', 'season']
numerical_features   = ['year', 'month', 'day', 'dayOfYear', 'weekday']

X = df_train.drop(columns=['num_sold', 'num_sold_log', 'date', 'id'], errors='ignore').copy()
y = df_train['num_sold'].astype(float)


X


try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:  # older sklearn
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

pre = ColumnTransformer([('cat', ohe, categorical_features)], remainder='drop')

# fit on train, transform both
Xtr_cat = pre.fit_transform(df_train[categorical_features])
Xte_cat = pre.transform(df_test[categorical_features])

cols = pre.get_feature_names_out()
Xtr_cat = pd.DataFrame(Xtr_cat, columns=cols, index=df_train.index)
Xte_cat = pd.DataFrame(Xte_cat, columns=cols, index=df_test.index)

# if you want to keep other (non-categorical) columns:
X_train_encoded = pd.concat([df_train.drop(columns=categorical_features), Xtr_cat], axis=1)
X_test_encoded  = pd.concat([df_test.drop(columns=categorical_features),  Xte_cat], axis=1)


models = [
    ('LinearRegression',  LinearRegression()),
    ('KNeighborsRegressor', KNeighborsRegressor(n_neighbors=5, weights='distance')),
    ('DecisionTreeRegressor', DecisionTreeRegressor(random_state=42)),
    ('RandomForestRegressor', RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
    ('AdaBoostRegressor',  AdaBoostRegressor(random_state=42)),
    ('XGBRegressor',       XGBRegressor(
         random_state=42, n_estimators=600, learning_rate=0.05,
         max_depth=8, subsample=0.8, colsample_bytree=0.8, tree_method='hist',
         eval_metric='rmse')),
    ('CatBoostRegressor',  CatBoostRegressor(
         random_state=42, iterations=800, depth=8, learning_rate=0.05, verbose=0))
]


# 2) Build encoded feature tables (drop target & non-features)
X_train_encoded = pd.concat([
    df_train.drop(columns=categorical_features + ['num_sold','num_sold_log','date','id'], errors='ignore'),
    Xtr_cat
], axis=1)

X_test_encoded = pd.concat([
    df_test.drop(columns=categorical_features + ['date','id'], errors='ignore'),
    Xte_cat
], axis=1)

# 3) Target
y = df_train['num_sold'].astype(float)

# 4) Simple chronological holdout (last 20% for validation)
split_idx = int(len(X_train_encoded) * 0.80)
Xtr, Xva = X_train_encoded.iloc[:split_idx], X_train_encoded.iloc[split_idx:]
ytr, yva = y.iloc[:split_idx], y.iloc[split_idx:]

# 5) Fit & evaluate each model

y_mean = yva.mean()
print("Model                 | RMSE(K) | NRMSE% |   MAE  |  R^2 ")
print("-"*60)

for name, model in models:
    model.fit(Xtr, ytr)
    pred = model.predict(Xva)

    rmse = mean_squared_error(yva, pred, squared=False)
    mae  = mean_absolute_error(yva, pred)
    r2   = r2_score(yva, pred)

    print(f"{name:<20s} | {rmse/1000:7.3f} | {(rmse/(y_mean+1e-9))*100:6.2f} | {mae:7.2f} | {r2:5.3f}")

# 6) Train best-once on ALL rows and predict test (example picks RandomForest)
best_model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
best_model.fit(X_train_encoded, y)
test_pred = best_model.predict(X_test_encoded)  # <- on original scale



from catboost import  Pool

cat_cols = ['country','product','store','season']
num_cols = ['year','month','day','dayOfYear','weekday']

Xtr = df_train[cat_cols + num_cols].iloc[:split_idx]
Xva = df_train[cat_cols + num_cols].iloc[split_idx:]
ytr = df_train['num_sold'].astype(float).iloc[:split_idx]
yva = df_train['num_sold'].astype(float).iloc[split_idx:]

train_pool = Pool(Xtr, ytr, cat_features=[Xtr.columns.get_loc(c) for c in cat_cols])
valid_pool = Pool(Xva, yva, cat_features=[Xva.columns.get_loc(c) for c in cat_cols])

cb = CatBoostRegressor(
    loss_function='RMSE',          # try 'Poisson' next for counts
    iterations=2000,               # allow plenty; early stopping will cut it
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,               # a bit of regularization
    random_seed=42,
    eval_metric='RMSE',
    verbose=200
)

cb.fit(
    train_pool,
    eval_set=valid_pool,
    use_best_model=True,           # keep the best iteration found on valid
    early_stopping_rounds=200      # stop if no improvement for 200 iters
)

pred = cb.predict(valid_pool)      # uses the best iteration


