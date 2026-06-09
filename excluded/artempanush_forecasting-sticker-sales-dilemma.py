import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from datetime import datetime
import holidays
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message="underflow encountered*")
warnings.simplefilter(action='ignore', category=FutureWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
train


test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
test.head()


train.shape


train.isnull().sum()


isn = train[train.isnull().any(axis=1)]
isn.head()


isn['country'].value_counts()


isn['store'].value_counts()


isn['product'].value_counts()


train = train.dropna()


train


train['country'].value_counts()


train['store'].value_counts()


train['product'].value_counts()


plt.figure(figsize=(12, 3))
plt.subplots_adjust(wspace=0.5)

plt.subplot(1,3,1)
ax = sns.countplot(x='country', hue='country', data=train, palette = "ch:s=.25,rot=-.25")
plt.title('Countries Count', fontsize=12)
ax.legend().remove()
plt.xticks(rotation = 35)

plt.subplot(1,3,2)
ax = sns.countplot(x='store', hue='store', data=train, palette = "ch:s=.25,rot=-.25")
plt.title('Stores Count', fontsize=12)
plt.xticks(rotation = 25)
ax.legend().remove()
plt.tick_params(labelsize=8)

plt.subplot(1,3,3)
ax = sns.countplot(data=train, x='product',  hue='product', palette = "ch:s=.25,rot=-.25")
plt.title('Products Count', fontsize=12)
plt.xticks(rotation = 25)
plt.tick_params(labelsize=8)
ax.legend().remove()
plt.show()


plt.figure(figsize=(11, 3))
# plt.subplots_adjust(wspace=0.5)

sns.catplot(data = train, y='store', hue='product', kind='count', orient = 'y', palette = "ch:s=.25,rot=-.25")

plt.show()


train['num_sold'].value_counts(bins=10)


# train['num_sold'].plot()

sns.histplot(data=train, x = 'num_sold', kde=True, bins=40, color='steelblue')
plt.title('Sold Distribution', fontsize=16)
plt.xticks(fontsize=10, rotation=45)
plt.show()


mean_sales_country = train.groupby('country')['num_sold'].agg('mean').reset_index()
mean_sales_store = train.groupby('store')['num_sold'].agg('mean').reset_index()
mean_sales_product = train.groupby('product')['num_sold'].agg('mean').reset_index()

plt.figure(figsize=(12, 3))
plt.subplots_adjust(wspace=0.5)

plt.subplot(1,3,1)
ax = sns.barplot(data = mean_sales_country, x='country', hue='country', y='num_sold', palette = "ch:s=.25,rot=-.25")
plt.title('Sales by Country', fontsize=12)
ax.legend().remove()
plt.xticks(rotation = 45)

plt.subplot(1,3,2)
ax = sns.barplot(data = mean_sales_store, x='store', hue='store', y='num_sold', palette = "ch:s=.25,rot=-.25")
plt.title('Sales By Store', fontsize=12)
ax.legend().remove()
plt.xticks(rotation = 25)

plt.subplot(1,3,3)
ax = sns.barplot(data = mean_sales_product, x='product', hue='product', y='num_sold', palette = "ch:s=.25,rot=-.25")
plt.xticks(rotation = 25)
ax.legend().remove()
plt.title('Sales By Product', fontsize=12)

plt.show()


mean_sales_store


mean_sales_product


# train['num_sold'] = train['num_sold'].fillna(train['num_sold'].mean())


def is_holiday(df):

    '''
    We receive input holidays for each mentioned country from the imported module.
    Mark these days in the dataset.
    '''

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    countries = np.sort(df['country'].unique())
    code_countries = ['CA', 'FI', 'IT', 'KE', 'NO', 'SG']
    
    dict = {countries[i]: holidays.country_holidays(code_countries[i], years=range(2010, 2020)) for i in range(len(countries))}
    df['is_holiday'] = 0

    for c in countries:
        df.loc[df['country']== c, 'is_holiday'] = df['date'].isin(dict[c]).astype(int)

    return df


train = is_holiday(train)
test = is_holiday(test)


# ndf = train.copy()
# countries = np.sort(ndf['country'].unique())
# code_countries = ['CA', 'FI', 'IT', 'KE', 'NO', 'SG']


# dict = {countries[i]: holidays.country_holidays(code_countries[i], years=range(2010, 2020)) for i in range(len(countries))}
# dict


train['is_holiday'].value_counts()


def complete_feature(df):

    '''transform date feature and create new datetime features'''

    df = df.copy()

    # #change type of column
    # df['date'] = pd.to_datetime(df['date'])

    #create new features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['per_month'] = df['date'].dt.to_period('M')
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek  # Monday = 0, Sunday = 6
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].apply(lambda x: x in [5, 6])
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)  
    df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
    df['is_year_end'] = df['date'].dt.is_year_end.astype(int)    

    df['Season'] = df['date'].dt.month.map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})

    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365.0)
    df['day_sin2'] = np.sin(4 * np.pi * df['day'] / 365.0)
    df['day_cos2'] = np.cos(4 * np.pi * df['day'] / 365.0)
    df['day_sin3'] = np.sin(6 * np.pi * df['day'] / 365.0)
    df['day_cos3'] = np.cos(6 * np.pi * df['day'] / 365.0)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['month_sin2'] = np.sin(4 * np.pi * df['month'] / 12.0)
    df['month_cos2'] = np.cos(4 * np.pi * df['month'] / 12.0)
    df['sin_year'] = np.sin(2*np.pi*df['year']/365)
    df['cos_year'] = np.cos(2*np.pi*df['year']/365)
    df['group'] = (df['year'] - 2020) * 48 + df['month'] * 4 + df['day'] // 7

    df['month_country'] = df['month'].astype(str) + "_" + df['country']
    df['day_country'] = df['day'].astype(str) + "_" + df['country']

    df['continents'] = df['country'].map({'Finland':1, 'Norway':1, 'Italy':1, 'Canada':2, 'Kenya':3, 'Singapore':4})
    df['parts'] = df['country'].map({'Finland':1, 'Norway':1, 'Italy':2, 'Canada':1, 'Kenya':3, 'Singapore':2})

    #drop columns
    df.drop('date', axis=1, inplace=True)
    
    return df


train = complete_feature(train)
test = complete_feature(test)


#look on the new view of data

train.head()


sales_year = train.groupby('year')['num_sold'].agg('mean').reset_index()
sales_month = train.groupby('month')['num_sold'].agg('mean').reset_index()
sales_days = train.groupby('day_of_week')['num_sold'].agg('mean').reset_index()

plt.figure(figsize=(12, 3))
plt.subplots_adjust(wspace=0.5)

plt.subplot(1,3,1)
ax = sns.barplot(data = sales_year, x='year', hue='year', y='num_sold', palette = "ch:s=.25,rot=-.25")
plt.title('Sales by Years', fontsize=12)
ax.legend().remove()
plt.xticks(rotation = 45)

plt.subplot(1,3,2)
ax = sns.barplot(data = sales_month, x='month', hue='month', y='num_sold', palette = "ch:s=.25,rot=-.25")
ax.legend().remove()
plt.title('Sales By Months', fontsize=12)

plt.subplot(1,3,3)
ax = sns.barplot(data = sales_days, x='day_of_week', hue='day_of_week', y='num_sold', palette = "ch:s=.25,rot=-.25")
ax.legend().remove()
plt.title('Sales By Days', fontsize=12)
plt.show()


sales_per_month = train.groupby('per_month')['num_sold'].agg('mean').reset_index()
sales_per_month['per_month'] = sales_per_month['per_month'].astype(str)
plt.figure(figsize=(11, 4))
sns.lineplot(data = sales_per_month, x='per_month', y='num_sold', color = "#00008b", linewidth = 1.5)
plt.xticks(range(0, sales_per_month.shape[0], sales_per_month.shape[0]//10), sales_per_month['per_month'][::sales_per_month.shape[0]//10], rotation=45, ha='right')
plt.tight_layout()
plt.show()


# sales_per_month_country = train.groupby(['per_month', 'country'])['num_sold'].agg('mean').reset_index()
# sales_per_month_country['per_month'] = sales_per_month['per_month'].astype(str)
# sales_per_month_country


# facet_grid = sns.FacetGrid(data=sales_per_month_country, 
#                            # col='year',
#                            col='country', 
#                            height=3,
#                            col_wrap=3)
# facet_grid.map(sns.scatterplot, 'per_month', 'num_sold')
# plt.xticks(range(0, sales_per_month.shape[0], sales_per_month.shape[0]//10), sales_per_month['per_month'][::sales_per_month.shape[0]//10], rotation=45, ha='right')


def trendlines(x,xx,hue):

    '''create the scatter plots for different features by the year
    return plot'''
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1,2,1)
    sns.lineplot(x=x, y="num_sold", data=train, palette='ch:s=.25,rot=-.25', hue=hue)
    plt.title('{x} to {hue}', fontsize=12)
    plt.xticks(rotation = 45)

    plt.subplot(1,2,2)
    sns.lineplot(x=xx, y="num_sold", data=train, palette='ch:s=.25,rot=-.25', hue=hue)
    plt.title('{xx} to {hue}', fontsize=12)

    return plt.show()


trendlines('year', 'month', 'country')


trendlines('year', 'month', 'store')


trendlines('year', 'month', 'product')


def scatter_fg(col,time):

    '''create the scatter plots for different features by the year
    return plot'''
    
    facet_grid = sns.FacetGrid(data=train, 
                           # col='year',
                           col=col, 
                           height=4.5,
                           col_wrap=3)
    facet_grid.map(sns.scatterplot, time, 'num_sold')

    return plt.show()


# facet_grid = sns.FacetGrid(data=train, 
#                            # col='year', 
#                            col='country', 
#                            height=4.5,
#                            col_wrap=3)
# facet_grid.map(sns.scatterplot, 'year', 'num_sold')

# plt.show()

scatter_fg('country', 'year')


scatter_fg('store', 'year')


scatter_fg('product', 'year')


plt.figure(figsize=(11,6))
sold_pivot = train.pivot_table(index='month', columns='year', values='num_sold') 
sold_pivot = round(sold_pivot,2)
sns.heatmap(sold_pivot, annot=True, cmap="Blues", fmt='.2f', linewidths=.5)
plt.show()


# choose object types

categorical = train.select_dtypes(include=['object', 'category']).columns


categorical


#encoding

oe = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)
train[categorical] = oe.fit_transform(train[categorical])
test[categorical] = oe.transform(test[categorical])


train.columns


#choose columns for moodel (combine by adding and removing lines)

X_col = ['country', 'store', 'product', 
         'is_holiday',
         'year', 'month', 'day', 
         'day_of_week', 'quarter', 'is_weekend', 'day_of_year',  
        'day_sin', 'day_cos', 
         'day_sin2', 'day_cos2', 
         'day_sin3', 'day_cos3', 
         'month_sin', 'month_cos', 
         'month_sin2', 'month_cos2', 
         'is_month_start', 'is_month_end', 
         'is_year_start', 'is_year_end',
         'month_country', 'day_country',
         'sin_year', 'cos_year',
         'day_of_week_sin', 'day_of_week_cos',
         'group', 'continents',  
         # 'parts',
         'Season']
y_col = np.log1p(train['num_sold'])


y_col.value_counts(bins=10)


X_train, X_valid, y_train, y_valid = train_test_split(train[X_col], y_col, test_size=0.15, random_state=42)


def modelling(model):
    
    ''' Fit the model, make predictions and calculate rmse '''

    model.fit(X_train, y_train)
    model_predict = model.predict(X_valid)
    model_mape = mean_absolute_percentage_error(np.expm1(y_valid), np.expm1(model_predict))
        
    return model_mape, model_predict


models_predictions = []
models_names = []


cat_params = {'iterations': 800, # Number of boosting iterations (trees)
              'learning_rate': 0.27, # Step size shrinkage for preventing overfitting
              'depth': 7, # Maximum depth of each tree
              'l2_leaf_reg': 0.005, # L2 regularization on leaf values
              'border_count': 250, # Number of splits to consider for features
              'subsample': 0.64, # Fraction of data used for each tree (bagging)
              'random_strength': 5 # Controls randomness in feature splits
             }

cat_model = CatBoostRegressor(**cat_params, verbose=False)


models_names.append('CAT')
mape_cat, pred_cat = modelling(cat_model)
models_predictions.append(mape_cat)


cat_fi = cat_model.get_feature_importance(prettified=True)


plt.figure(figsize=(16, 5))

plt.subplot(1,2,1)
plt.scatter(y_valid, pred_cat, color='steelblue')
plt.xlabel('Y_valid', fontsize=12)
plt.ylabel('Predictions', fontsize=12)
plt.title('Pred vs Valid for Catboost')

plt.subplot(1,2,2)
sns.barplot(x='Importances', y='Feature Id', data=cat_fi, color='#a7c2e1', orient='h')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.yticks(fontsize=8)
plt.title('Feature Importance', fontsize=16)

plt.show()


lgbm_params = {'n_estimators': 1500, # The number of boosting rounds or boosting trees to be built
               'learning_rate': 0.11, # Step size shrinkage used to prevent overfitting
               'max_depth': 17, # Maximum depth of a tree, controls model complexity
               'lambda_l2': 0.01,
               'lambda_l2': 0.18, # L2 regularization term added to weights to prevent overfitting
               'min_child_samples': 68, # Minimum number of data needed in a leaf, used to prevent overfitting
               'colsample_bytree': 0.71, # Fraction of features (columns) used for each tree, can prevent overfitting
               'subsample': 0.95 # Fraction of data (rows) used for each tree, helps with generalization
        }

lgbm_model = LGBMRegressor(**lgbm_params, verbose=-1)


models_names.append('LGBM')
mape_lgbm, pred_lgbm = modelling(lgbm_model)
models_predictions.append(mape_lgbm)


lgbm_fi = pd.DataFrame({'Feature Id':X_col, 'LGBM_Importances':lgbm_model.feature_importances_})
lgbm_fi = lgbm_fi.sort_values(by='LGBM_Importances', ascending=False)


plt.figure(figsize=(15, 6))

plt.subplot(1,2,1)
plt.scatter(y_valid, pred_lgbm, color='steelblue')
plt.xlabel('Y_valid', fontsize=14)
plt.ylabel('Predictions', fontsize=14)
plt.title('Pred vs Valid for LightGBM')

plt.subplot(1,2,2)
sns.barplot(x='LGBM_Importances', y='Feature Id', data=lgbm_fi, color='#a7c2e1', orient='h')
plt.xlabel('Importances', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Feature Importance', fontsize=16)

plt.show()


xgb_params = {
    'n_estimators':700, # The number of boosting rounds (trees) to be built
    'learning_rate':0.07, # Step size shrinkage used to prevent overfitting
    'max_depth':7, # Maximum depth of a tree, determines the complexity of the model
}

xgb_model = XGBRegressor(**xgb_params)


models_names.append('XGB')
mape_xgb, pred_xgb = modelling(xgb_model)
models_predictions.append(mape_xgb)


xgb_fi = pd.DataFrame({'Feature Id':X_col, 'XGB_Importances':xgb_model.feature_importances_})
xgb_fi = xgb_fi.sort_values(by='XGB_Importances', ascending=False)


plt.figure(figsize=(15, 6))

plt.subplot(1,2,1)
plt.scatter(y_valid, pred_xgb, color='steelblue')
plt.xlabel('Y_valid', fontsize=14)
plt.ylabel('Predictions', fontsize=14)
plt.title('Pred vs Valid for XGBBoost')

plt.subplot(1,2,2)
sns.barplot(x='XGB_Importances', y='Feature Id', data=xgb_fi, color='#a7c2e1', orient='h')
plt.xlabel('Importances', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Feature Importance', fontsize=16)

plt.show()


voting_model = VotingRegressor(estimators = [
    ('lgbm', lgbm_model), 
    ('xgb', xgb_model), 
    ('cat', cat_model)
], n_jobs = -1)


models_names.append('VOT')
mape_voting, pred_voting = modelling(voting_model)
models_predictions.append(mape_voting)


models = pd.DataFrame({'model':models_names, 'prediction':models_predictions})
models


display(test)


idd = test['id']


scores = cat_model.predict(test[X_col])
scores = np.expm1(scores)
submission = pd.DataFrame({'id':idd, 'num_sold':scores})
submission.to_csv('sub_cat_1ver.csv', index=False)
submission


scores2 = lgbm_model.predict(test[X_col])
scores2 = np.expm1(scores2)
submission = pd.DataFrame({'id':idd, 'num_sold':scores2})
submission.to_csv('sub_lgbm_1ver.csv', index=False)


submission


scores3 = xgb_model.predict(test[X_col])
scores3 = np.expm1(scores3)
submission = pd.DataFrame({'id':idd, 'num_sold':scores3})
submission.to_csv('sub_xgb_1ver.csv', index=False)


submission

