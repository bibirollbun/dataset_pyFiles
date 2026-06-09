import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape

import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'], index_col=0)
original_train = train.copy()
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'], index_col=0)


train.info()


train.head(6)


train.describe(include='O')


train['num_sold'].describe()


# Missing Data
train.isna().sum()


train[train['num_sold'].isna()].describe(include='O')


for col in train.select_dtypes('object').columns.to_list():
    uniq = train[train['num_sold'].isna()][col].unique()
    print(col + ' with missing values : ' + str(uniq))


na_groups = train[train['num_sold'].isna()].groupby(['country', 'store', 'product'])['date'].count()
na_groups


all_na = []
for index, row in na_groups[na_groups == 2557].reset_index().iterrows():
    all_na.append([row['country'], row['store'], row['product']])
all_na


def plot_with_missing_vals(c, s, p):
    canstholna = train.loc[(train['country'] == c) & (train['store'] == s) & 
    (train['product'] == p) & train['num_sold'].isna()]['date']
    
    fig, ax = plt.subplots()
    train.loc[(train['country'] == c) & (train['store'] == s) & 
    (train['product'] == p)].plot(x='date', y='num_sold', ax=ax)
    plt.scatter(x=canstholna.values, y=np.zeros(len(canstholna)), s=2, c='red', alpha=0.25, label='missing values')
    plt.title(f'Plot of Number of Products Sold over Time with Missing Values Shown as 0 \n in {c} at {s} for the {p} Sticker'.title())
    plt.legend()
    plt.show()

plot_with_missing_vals('Canada', 'Premium Sticker Mart', 'Holographic Goose')
plot_with_missing_vals('Canada', 'Stickers for Less', 'Holographic Goose')
plot_with_missing_vals('Kenya', 'Discount Stickers', 'Kerneler')
plot_with_missing_vals('Kenya', 'Premium Sticker Mart', 'Holographic Goose')
plot_with_missing_vals('Kenya', 'Stickers for Less', 'Holographic Goose')


# Values in Each Categorical Column
for col in train.select_dtypes('object').columns.to_list():
    vals = train[col].unique()
    print(f'{col.capitalize()} : {vals}')


for col in train.select_dtypes('object').columns.to_list():
    
    train[col].value_counts().plot(kind='bar')
    
    plt.title(f'Bar Chart of {col.capitalize()} Counts')
    plt.xticks(rotation=45)
    plt.show()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios': [2, 1]})

train['num_sold'].plot(kind='hist', bins=100, title='Histogram of Number of Products Sold', ax=ax1)

train['num_sold'].plot(kind='box', vert=False, title='Box Plot of Number of Products Sold', ax=ax2)
plt.yticks([])
plt.tight_layout()
plt.show()


# Groupby/Subset graphs
for col in train.select_dtypes('object').columns.to_list():
    
    train.groupby([col])['num_sold'].sum().plot(kind='bar')
    
    plt.title(f'Bar Chart of Number of Products Sold Split by {col.capitalize()}')
    plt.xticks(rotation=45)
    plt.show()


cross = pd.crosstab(train['country'], [train['store'], train['product']], values=train['num_sold'], aggfunc='sum', rownames=['Country'], colnames=['Store', 'Product'])
cross


sns.heatmap(cross.T)
plt.title('Heatmap of Total Products Sold Split by Store, Product and Country', loc='right')
plt.show()


def make_norm_crosstab(index, col1, col2, fig_ax):
    cross_norm = pd.crosstab(train[index], [train[col1], train[col2]], values=train['num_sold'], 
                             aggfunc='sum', normalize='index', rownames=[index.capitalize()], 
                             colnames=[col1.capitalize(), col2.capitalize()])
    sns.heatmap(cross_norm.T, ax=fig_ax)
    fig_ax.set_title(f'Heatmap of Total Products Sold Split by Store, Product and Country Normalized Per {index.capitalize()}', loc='right')
    fig_ax.tick_params(axis='x', rotation=20)

fig, ax = plt.subplots(3, figsize=(6,12))
make_norm_crosstab('country', 'store', 'product', ax[0])
make_norm_crosstab('store', 'country', 'product', ax[1])
make_norm_crosstab('product', 'country', 'store', ax[2])
plt.subplots_adjust(hspace=0.5)


def time_plot_fixed_c_s(c, s):
    fig, ax = plt.subplots()
    i, j = c, s
    for k in train['product'].unique():
        train.loc[(train['country'] == i) & (train['store'] == j) & 
        (train['product'] == k) & (train['date'].dt.year >= 2015),  
        ['date', 'num_sold']].plot(y='num_sold', x='date', label=k, ax=ax)
    plt.title(f'Number of Products Sold Over Time in the Last Two Years \n in {i} from {s}'.title())
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

time_plot_fixed_c_s('Canada', 'Discount Stickers')
time_plot_fixed_c_s('Singapore', 'Premium Sticker Mart')
time_plot_fixed_c_s('Norway', 'Stickers for Less')


# # set num_sold to 0 if no records for the country, store, product group
# for n in all_na:
#     train.loc[(train['country'] == n[0]) & (train['store'] == n[1]) & 
#             (train['product'] == n[2])] = train.loc[(train['country'] == n[0]) & 
#             (train['store'] == n[1]) & (train['product'] == n[2])].fillna(0)


# drop all other rows with no num_sold value
train = train.dropna()


train['num_sold'].isna().sum()


gdp_pc_df = pd.read_csv("/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv")

years =  ['201'+ str(i) for i in range(10)]
gdp_pc_filtered_df = gdp_pc_df.loc[gdp_pc_df['Country Name'].isin(train['country'].unique()), ['Country Name'] + years]#.set_index('Country Name')
# normalize the values within the year
gdp_pc_filtered_df = gdp_pc_filtered_df.set_index('Country Name').apply(lambda x: x/x.max(), axis=0).reset_index()
gdp_pc_filtered_df = gdp_pc_filtered_df.melt(id_vars='Country Name')
gdp_pc_filtered_df.columns = ['country', 'year', 'gdp_per_capita']
gdp_pc_filtered_df['year'] = pd.to_datetime(gdp_pc_filtered_df['year'], format='%Y')
gdp_pc_filtered_df['year'] = gdp_pc_filtered_df['year'].dt.year
gdp_pc_filtered_df.T


def extract_features(df):
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.day_of_week
    df['week'] = df['date'].dt.isocalendar().week
    # some features that capture the cyclical nature of days in the month and months in the year (ie end links to beginning)
    df['month_sin'] = np.sin(2*np.pi*df['day']/31).round(decimals=5)
    df['month_cos'] = np.cos(2*np.pi*df['month']/31).round(decimals=5)
    df['year_sin'] = np.sin(2*np.pi*df['month']/12).round(decimals=5)
    df['year_cos'] = np.cos(2*np.pi*df['day']/12).round(decimals=5)
    
    return df

train = extract_features(train)
test = extract_features(test)


train = pd.merge(train, gdp_pc_filtered_df, on=['year', 'country'])
train.sample(3)


test = pd.merge(test, gdp_pc_filtered_df, on=['year', 'country'])
test.sample(3)


def preprocess(df, to_drop=[]):
    df = df.drop(columns=to_drop)
    df = pd.get_dummies(df, drop_first=True, dtype=float)
    
    return df

train = preprocess(train, ['date'])
test = preprocess(test, ['date'])


X = train.drop(['num_sold'], axis=1)
y = np.log1p(train['num_sold'])


tscv = TimeSeriesSplit(n_splits=4)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    reg = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=5
    )
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    mape_score = mape(y_test, y_pred)
    print("MAPE:", mape_score)


def pred_v_actual_train_plot(model, smoothing):
    fig, ax = plt.subplots()
    pred_df = pd.DataFrame(np.expm1(model.predict(X_train)))
    test_df = np.expm1(y_train)
    test_df = test_df.reset_index().drop(columns=['index'])
    
    test_df.rolling(smoothing).mean().plot(ax=ax)
    pred_df.rolling(smoothing).mean().plot(ax=ax)
    plt.title('Training Data vs Model Prediction of Number Sold vs Time')
    plt.legend(labels=['Actual', 'Predicted'])
    plt.show()

pred_v_actual_train_plot(reg, 3000)


def pred_v_actual_plot(smoothing):
    fig, ax = plt.subplots()
    pred_df = pd.DataFrame(np.expm1(y_pred))
    test_df = np.expm1(y_test)
    test_df = test_df.reset_index().drop(columns=['index'])
    
    test_df.rolling(smoothing).mean().plot(ax=ax)
    pred_df.rolling(smoothing).mean().plot(ax=ax)
    plt.title('Validation Data vs Model Prediction of Number Sold vs Time')
    plt.legend(labels=['Actual', 'Predicted'])
    plt.show()

pred_v_actual_plot(2000)


# param_grid = {
#     'n_estimators': [700],
#     'max_depth': [5],
#     'min_child_weight': [4, 5, 6, 9],
#     'colsample_bytree': [0.3, 0.5, 0.6, 0.7],
#     'learning_rate': [0.075] 
# }

# model = xgb.XGBRegressor(random_state=42)

# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_percentage_error', cv=tscv, verbose=3)
# grid_search.fit(X_train, y_train)

# print(f"Best Parameters: {grid_search.best_params_}")
# print(f"Best Score: {grid_search.best_score_}")


model = xgb.XGBRegressor(n_estimators=700, max_depth=5, min_child_weight=6, 
                         colsample_bytree=0.6, learning_rate=0.075, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mape_score = mape(y_test, y_pred)
print("MAPE:", mape_score)


pred_v_actual_train_plot(model, 3000)


pred_v_actual_plot(2000)





test_pred = model.predict(test)
test_pred = np.expm1(test_pred) 


sample = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
sample


test['num_sold_pred'] = test_pred
print(test[['num_sold_pred']].reset_index())

test[['num_sold_pred']].reset_index().to_csv('submission.csv', index=False)




