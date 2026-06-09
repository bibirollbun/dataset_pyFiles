# import base required packages, other package may also be needed
import numpy as np
import pandas as pd
import zipfile, os, shutil
import warnings
warnings.filterwarnings('ignore')
import holidays
from decimal import Decimal, Context

# packages for visualizing the data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style = 'whitegrid')

# import and install modeling packages
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# install tabnet
!pip install pytorch-tabnet
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from pytorch_tabnet.metrics import Metric
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# run this cells to clear working path
try:
    shutil.rmtree("/kaggle/working/")
except:
    pass


# read the data train and save it to variabel named 'df_ori'
df_ori = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv').drop('id', axis=1)

# don't forget the data test
df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

# show the head of data
df_ori.head()


# make a copy of the data frame
# it will help you a lot if you are messing around with the original data
df = df_ori.copy()

# is there any data duplicates?
df.duplicated().sum()


# what about the missing values?
print(df.isnull().sum())

# then let's drop it
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)


# perform date-based feature engineering
def date_feature_engineering(df):
    # regular datetime
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_in_week'] = df['date'].dt.dayofweek

    # add holiday
    year_min = min(df['year'])
    year_max = max(df['year'])
    init_country = dict(zip(np.sort(df['country'].unique()), ['CA',
                                                              'FI',
                                                              'IT',
                                                              'KE',
                                                              'NO',
                                                              'SG']))
    holidays_dict = {
        c: holidays.country_holidays(a, years=range(year_min, year_max))
        for c, a in init_country.items()
        }
    df['is_holiday'] = 0
    for c in holidays_dict:
        df.loc[df.country == c, 'is_holiday'] = df['date'].isin(holidays_dict[c]).astype(int)

    # cyclical features
    df['day_in_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_in_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    df['month_in_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_in_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['year_in_sin'] = np.sin(2 * np.pi * df['year'] / 7)
    df['year_in_cos'] = np.cos(2 * np.pi * df['year'] / 7)

    # group calculation
    df['group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7

    # return as new dataframe
    return df

# execute the function
df = date_feature_engineering(df)
df_test = date_feature_engineering(df_test)


# perform the label encoding for object datatypes
obj_data = list(df.select_dtypes('object').columns)
encoded = {col: LabelEncoder() for col in obj_data}

# apply labelencoder to each categorical column
for col in obj_data:
    df[col] = encoded[col].fit_transform(df[col])
    df_test[col] = encoded[col].transform(df_test[col])


# # perform the one-hot encoding for object datatypes
# obj_data = list(df.select_dtypes('object').columns)
# encoded = OneHotEncoder(sparse_output=False)

# # apply one-hot encoder to each categorical column
# encoded_train = encoded.fit_transform(df[obj_data])
# encoded_test = encoded.transform(df_test[obj_data])

# # modify the dataframe
# one_hot_train = pd.DataFrame(encoded_train,
#                              columns=encoded.get_feature_names_out(obj_data))
# one_hot_test = pd.DataFrame(encoded_test,
#                             columns=encoded.get_feature_names_out(obj_data))
# df = pd.concat([df, one_hot_train], axis=1)
# df_test = pd.concat([df_test, one_hot_test], axis=1)

# # drop the specific columns
# df = df.drop(obj_data, axis=1)
# df_test = df_test.drop(obj_data, axis=1)


# create inter quartile range procedure
def iqr(df):
    for i in df:
        try:
            # calculate the upper and lower limits
            Q1 = df[i].quantile(0.25)
            Q3 = df[i].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5*IQR
            upper = Q3 + 1.5*IQR

            # create arrays of boolean values indicating the outlier rows
            upper_array = np.where(df[i]>=upper)[0]
            lower_array = np.where(df[i]<=lower)[0]

            # removing the outliers
            df = df.drop(index=upper_array)
            df = df.drop(index=lower_array)
        except:
            pass

    # return as new dataframe
    return df

# execute the function
df = iqr(df)


# # encode the country, store, month, and year
# # this one will help us to extract the median value for each country, store, month, and year
# csmy = []

# for i in range(len(df)):
#     temp = str(df['country'].iloc[i])
#     temp += str(df['store'].iloc[i])
#     temp += str(df['month'].iloc[i])
#     temp += str(df['year'].iloc[i])
#     csmy.append(temp)

# # add to the training dataframe
# df['csmy'] = csmy


# # perform the imputation on the 'num_sold' with the help of 'csmy'
# def imp_feat(df):
#     unique_cmy = list(df['csmy'].unique()) # take the unique value of country-month-year
#     temp_df = df[df['csmy'] == unique_cmy[0]].copy() # save the first 'csmy'
#     temp_mode = temp_df['num_sold'].mode()[0] # calculate the chosen metric
#     temp_df.fillna(temp_mode, inplace=True) # fill the na with the median (if any)
#     for i in unique_cmy[1:]:
#         temp_df2 = df[df['csmy'] == i].copy() # redoing the previous steps
#         temp_mode = temp_df['num_sold'].mode()[0]
#         temp_df2.fillna(temp_mode, inplace=True)
#         temp_df = pd.concat([temp_df, temp_df2]) # then concatenate it

#     # return as the new dataframe
#     return temp_df

# # execute the median imputation
# df = imp_feat(df)


# # perform the median imputation on the 'num_sold' for each month and year
# def median_imp(df):
#     unique_year = list(df['year'].unique())
#     for i in range(len(unique_year)):
#         if i == 0:
#             temp = df[df['year'] == unique_year[i]].copy() # initial df
#             temp['num_sold'].fillna(temp['num_sold'].median(), inplace=True) 
#         temp2 = df[df['year'] == unique_year[i]].copy() # the rest df
#         temp2['num_sold'].fillna(temp2['num_sold'].median(), inplace=True)
#         temp = pd.concat([temp, temp2]) # merge the row

#     return temp

# # execute the function
# df = median_imp(df)


# check the dataframe information
df.info()


# correlation check for numerical features
fig, axs = plt.subplots(figsize = (14,10))
plt.title('Correlation Table')
sns.heatmap(df.select_dtypes((int, float)).corr(), annot=True,
            fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=axs)
plt.show()


# perform the inverse transform for categorical columns
inver = df.copy()
inver_test = df_test.copy()
for col in obj_data:
    inver[col] = encoded[col].inverse_transform(inver[col])
    inver_test[col] = encoded[col].inverse_transform(inver_test[col])

# show the sold item in each year
fig, axs = plt.subplots(figsize = (10,3))
plt.title('Sold Item (yearly)')
sns.lineplot(data=inver, x='date', y='num_sold',
             errorbar=None, linewidth=0.4, ax=axs)
plt.show()


# show the sold item, divide by year and store
fig, axs = plt.subplots(figsize = (10,4))
plt.title('Sold Item (yearly each store)')
sns.lineplot(data=inver, x='date', y='num_sold',
             hue='store', palette='bright',
             errorbar=None, linewidth=0.4, ax=axs)
plt.show()


# show the sold item, divide by year and country
fig, axs = plt.subplots(figsize = (10,4))
plt.title('Sold Item (yearly each country)')
sns.lineplot(data=inver, x='date', y='num_sold',
             hue='country', palette='muted',
             errorbar=None, linewidth=0.4, ax=axs)
plt.show()


# separate dependent and independent features
x = df.drop(['num_sold','date'], axis=1)
y = df['num_sold']

# create mape calculation
class Mape(Metric):
    def __init__(self):
        self._name = "mape"
        self._maximize = False

    def __call__(self, y_true, y_score):
        mape = mean_absolute_percentage_error(y_true, y_score)
        return mape

# define tabular network parameters
tabnet_params = dict(
    n_d=16, n_a=16, n_steps=5,
    gamma=1.9, n_independent=5,
    n_shared=5, mask_type="entmax",
    scheduler_fn = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    scheduler_params = dict(T_0=200,
                            T_mult=1,
                            eta_min=1e-4,
                            last_epoch=-1,
                            verbose=False),
    seed=42,
    optimizer_fn = torch.optim.Adam,
    device_name=DEVICE
)


# # show the sold item in each year
# fig, axs = plt.subplots(figsize = (14,3))
# plt.title('Sold Item (yearly)')
# sns.lineplot(data=df.loc[x_train.index.tolist()], x='date', y='num_sold',
#              errorbar=None, linewidth=0.4, ax=axs, label='Training')
# sns.lineplot(data=df.loc[x_test.index.tolist()], x='date', y='num_sold',
#              errorbar=None, linewidth=0.4, ax=axs, label='Testing')
# plt.show()


# # find the feature importance with random forest
# from sklearn.ensemble import RandomForestClassifier

# # create random forest model
# rf = RandomForestClassifier(n_estimators=5,
#                             random_state=42)

# # train the model and save the prediction result
# rf.fit(x_train, y_train)
# rf_pred = rf.predict(x_test)

# # print the mape
# rf_mape = mean_absolute_percentage_error(y_test, rf_pred)
# print('Random Forest model have an MAPE of:', rf_mape)

# # see the feature importance
# pd.DataFrame({'feat': x_train.columns,
#               'importance': rf.feature_importances_}).sort_values(by='importance',
#                                                                   ascending=False)


# create multiple tabnet models, x, and y, for different country
models, xs, ys = {}, {}, {}
for i in df['country'].unique():
    models[i] = TabNetRegressor(**tabnet_params)
    xs[i] = x[x['country'] == i] # filter for specific country
    ys[i] = y.loc[xs[i].index] # and the label
    xs[i] = np.array(xs[i])
    ys[i] = np.array(ys[i]).reshape((-1, 1))
# # model = TabNetRegressor(**tabnet_params)

# fit the model
for i in models:
    models[i].fit(
        xs[i], ys[i],
        eval_set=[(xs[i], ys[i])],
        eval_name=['train_'+str(i)],
        eval_metric=[Mape],
        max_epochs=100, patience=100,
        batch_size=1024, virtual_batch_size=256,
        num_workers=4,
    )


# data test preparation
x_preds = {}
for i in df_test['country'].unique():
    preds = []
    x_preds[i] = df_test[df_test['country'] == i].drop('date', axis=1)
    temp = models[i].predict(np.array(x_preds[i].drop('id', axis=1)))
    x_preds[i]['num_sold'] = temp

# create new prediction dataframe
pred_df = pd.concat([x_preds[i] for i in x_preds]).sort_values('id')

# add the prediction to inverse dataframe
inver_test['num_sold'] = pred_df['num_sold']

# show the sold item in each year
fig, axs = plt.subplots(figsize = (10,3))
plt.title('Sold Item (with TabNet Modeling)')
sns.lineplot(data=inver, x='date', y='num_sold',
             errorbar=None, linewidth=0.4,
             ax=axs, label='Actual')
sns.lineplot(data=inver_test, x='date', y='num_sold',
             errorbar=None, linewidth=0.4,
             ax=axs, label='Prediction')
plt.show()


# show the sold item in each year and store
fig, axs = plt.subplots(figsize = (10,3))
plt.title('Sold Item (with TabNet Modeling, divide by store)')
sns.lineplot(data=inver, x='date', y='num_sold', hue='store',
             errorbar=None, linewidth=0.4, ax=axs)
sns.lineplot(data=inver_test, x='date', y='num_sold',
             hue='store', errorbar=None, linewidth=0.4,
             ax=axs, legend=False, palette='muted')
plt.axvline(pd.to_datetime('2017-01-01'), color='red',
            linestyle='--', label='Actual | Prediction')
plt.legend(loc='upper left')
plt.show()


# show the sold item in each year and country
fig, axs = plt.subplots(figsize = (10,3))
plt.title('Sold Item (with TabNet Modeling, divide by store)')
sns.lineplot(data=inver, x='date', y='num_sold', hue='country',
             errorbar=None, linewidth=0.4, ax=axs)
sns.lineplot(data=inver_test, x='date', y='num_sold',
             hue='country', errorbar=None, linewidth=0.4,
             ax=axs, legend=False, palette='muted')
plt.axvline(pd.to_datetime('2017-01-01'), color='red',
            linestyle='--', label='Actual | Prediction')
plt.legend(loc='upper left')
plt.show()


# create the prediction
pd.set_option("display.precision", 3) # # lower the precision to 3 decimal
temp2 = pd.DataFrame({'id': list(df_test['id']),
                     'num_sold': list(pred_df['num_sold'])})
temp2.head()


# submit it!
temp2.to_csv('submission.csv', index = False)

