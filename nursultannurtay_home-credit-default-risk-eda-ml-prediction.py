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


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.express as px

import plotly.figure_factory as ff
import plotly.graph_objects as go

# ----------------------------------------------------
pd.set_option('display.max_rows', 200)
import warnings
warnings.filterwarnings("ignore")


application_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
application_train.head()


bureau_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
bureau_balance.head()


POS_CASH_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')
POS_CASH_balance.head()


previous_application = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')
previous_application.head()


credit_card_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
credit_card_balance.head()


installments_payments = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')
installments_payments.head()


bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
bureau.head()


files_names = ['application_train', 'bureau_balance', 'POS_CASH_balance', 'previous_application',
        'credit_card_balance', 'installments_payments', 'bureau']

files = [application_train, bureau_balance, POS_CASH_balance, previous_application,
        credit_card_balance, installments_payments, bureau]


for i in range(len(files)):
    print('{}: \n'.format(files_names[i]))
    print('Shape is: ',files[i].shape,'\n')
    print(files[i].info())
    print('\n\n*****************************************\n\n')


concat_application_train = pd.concat([application_train.isnull().sum().reset_index(name ='sum_of_nulls'), 
                                      application_train.dtypes.reset_index(name="dtypes")], 
                                     axis=1).T.drop_duplicates().T

concat_application_train.sort_values('sum_of_nulls', ascending = False)


concat_bureau_balance = pd.concat([bureau_balance.isnull().sum().reset_index(name ='sum_of_nulls'), 
                                      bureau_balance.dtypes.reset_index(name="dtypes")], 
                                     axis=1).T.drop_duplicates().T

concat_bureau_balance.sort_values('sum_of_nulls', ascending = False)


concat_POS_CASH_balance = pd.concat([POS_CASH_balance.isnull().sum().reset_index(name ='sum_of_nulls'), 
                                     POS_CASH_balance.dtypes.reset_index(name="dtypes")], 
                                     axis=1).T.drop_duplicates().T

concat_POS_CASH_balance.sort_values('sum_of_nulls', ascending = False)


concat_previous_application = pd.concat([previous_application.isnull().sum().reset_index(name ='sum_of_nulls'), 
                                         previous_application.dtypes.reset_index(name="dtypes")], 
                                        axis=1).T.drop_duplicates().T

concat_previous_application.sort_values('sum_of_nulls', ascending = False)


concat_credit_card_balance = pd.concat([credit_card_balance.isnull().sum().reset_index(name ='sum_of_nulls'), 
                                        credit_card_balance.dtypes.reset_index(name="dtypes")], 
                                        axis=1).T.drop_duplicates().T

concat_credit_card_balance.sort_values('sum_of_nulls', ascending = False)


concat_installments_payments = pd.concat([installments_payments.isnull().sum().reset_index(name ='sum_of_nulls'), 
                                          installments_payments.dtypes.reset_index(name="dtypes")], 
                                         axis=1).T.drop_duplicates().T

concat_installments_payments.sort_values('sum_of_nulls', ascending = False)


concat_bureau = pd.concat([bureau.isnull().sum().reset_index(name ='sum_of_nulls'), 
                           bureau.dtypes.reset_index(name="dtypes")], 
                          axis=1).T.drop_duplicates().T

concat_bureau.sort_values('sum_of_nulls', ascending = False)


fig = px.histogram(application_train, x="TARGET",
                   barmode='group').update_xaxes(categoryorder = "total descending")
fig.update_layout(bargap=0.2)
fig.show()


fig = px.histogram(application_train, x="CODE_GENDER",
                   barmode='group').update_xaxes(categoryorder = "total descending")
fig.update_layout(bargap=0.2)
fig.show()


fig = px.histogram(application_train, x="TARGET",
             color='CODE_GENDER', barmode='group').update_xaxes(categoryorder = "total descending")
fig.show()


fig = px.histogram(application_train, x="TARGET",
                   color="CODE_GENDER", barmode='group', 
                   facet_col='NAME_CONTRACT_TYPE').update_xaxes(categoryorder = "total descending")
fig.show()


fig = px.histogram(application_train, x="TARGET",
                   color="CODE_GENDER", barmode='group', 
                   facet_col='FLAG_OWN_CAR').update_xaxes(categoryorder = "total descending")
fig.show()


fig = px.histogram(application_train, x="TARGET",
                   color="CODE_GENDER", barmode='group', 
                   facet_col='FLAG_OWN_REALTY').update_xaxes(categoryorder = "total descending")
fig.show()


fig = px.histogram(application_train, x="NAME_CONTRACT_TYPE",
             color="FLAG_OWN_CAR", barmode='group', 
                   facet_col='TARGET').update_xaxes(categoryorder = "total descending")

fig.show()


fig = px.histogram(application_train, x="FLAG_OWN_REALTY",
             color="FLAG_OWN_CAR", barmode='group', 
                   facet_col='TARGET').update_xaxes(categoryorder = "total descending")

fig.show()


fig = px.histogram(application_train, x="NAME_CONTRACT_TYPE",
             color="FLAG_OWN_REALTY", barmode='group', 
                   facet_col='TARGET').update_xaxes(categoryorder = "total descending")
fig.show()


fig = px.histogram(application_train, x="LIVE_CITY_NOT_WORK_CITY",
             color="TARGET", barmode='group').update_xaxes(categoryorder = "total descending")

fig.show()


fig = px.histogram(application_train, x="REG_REGION_NOT_WORK_REGION",
             color="TARGET", barmode='group').update_xaxes(categoryorder = "total descending")

fig.show()


fig = px.histogram(application_train, x="LIVE_REGION_NOT_WORK_REGION",
             color="TARGET", barmode='group').update_xaxes(categoryorder = "total descending")

fig.show()


fig = px.histogram(application_train, x="REG_CITY_NOT_LIVE_CITY",
             color="TARGET", barmode='group').update_xaxes(categoryorder = "total descending")

fig.show()



fig = px.histogram(application_train, x="REG_CITY_NOT_WORK_CITY",
             color="TARGET", barmode='group').update_xaxes(categoryorder = "total descending")
fig.show()


fig = px.histogram(application_train, x="LIVE_CITY_NOT_WORK_CITY",
             color="TARGET", barmode='group').update_xaxes(categoryorder = "total descending")
fig.show()


pie = go.Figure(data=[go.Pie(labels = application_train['NAME_EDUCATION_TYPE'].value_counts().keys(),
                             values = application_train['NAME_EDUCATION_TYPE'].value_counts().values)])
iplot(pie)


fig = px.histogram(application_train, x="CNT_CHILDREN", color="TARGET", barmode='group')
fig.show()


fig = px.histogram(application_train, x="CNT_FAM_MEMBERS", color="TARGET", barmode='group')
fig.show()


fig = ff.create_distplot([application_train[application_train['TARGET']==0]['AMT_CREDIT'], 
                          application_train[application_train['TARGET']==1]['AMT_CREDIT']], 
                         ['target=> 0', 'target=> 1'],
                         bin_size = [2, 2],
                         show_rug=False,
                         show_hist=False,
                         show_curve=True 
                        )
fig.update_layout(xaxis = dict(title='AMT_CREDIT'),
                  yaxis =dict(title='Density'),
                  barmode='overlay')


fig.show()


fig = px.violin(application_train, x='CODE_GENDER', y='AMT_CREDIT', 
                color='TARGET', box = True)
fig.show()


fig = px.violin(application_train, x='FLAG_OWN_REALTY', y='AMT_CREDIT', 
                color='TARGET', box = True)
fig.show()


fig = ff.create_distplot([application_train[application_train['TARGET']==0]['AMT_ANNUITY'].dropna(), 
                          application_train[application_train['TARGET']==1]['AMT_ANNUITY'].dropna()], 
                         ['target=> 0', 'target=> 1'],
                         bin_size = [2, 2],
                         show_rug=False,
                         show_hist=False,
                         show_curve=True 
                        )
fig.update_layout(xaxis = dict(title='AMT_ANNUITY'),
                  yaxis =dict(title='Density'),
                  barmode='overlay')


fig.show()


fig = px.violin(application_train, x='CODE_GENDER', y='AMT_ANNUITY', 
                color='TARGET', box = True)
fig.show()


fig = px.violin(application_train, x='FLAG_OWN_REALTY', y='AMT_ANNUITY', 
                color='TARGET', box = True)
fig.show()


fig = ff.create_distplot([application_train[application_train['TARGET']==0]['AMT_GOODS_PRICE'].dropna(), 
                          application_train[application_train['TARGET']==1]['AMT_GOODS_PRICE'].dropna()], 
                         ['target=> 0', 'target=> 1'],
                         bin_size = [2, 2],
                         show_rug=False,
                         show_hist=False,
                         show_curve=True 
                        )
fig.update_layout(xaxis = dict(title='AMT_GOODS_PRICE'),
                  yaxis =dict(title='Density'),
                  barmode='overlay')


fig.show()


fig = px.violin(application_train, x='CODE_GENDER', y='AMT_GOODS_PRICE', 
                color='TARGET', box = True)
fig.show()



fig = px.violin(application_train, x='FLAG_OWN_REALTY', y='AMT_GOODS_PRICE', 
                color='TARGET', box = True)
fig.show()


pie = go.Figure(data=[go.Pie(labels = application_train['NAME_HOUSING_TYPE'].value_counts().keys(),
                             values = application_train['NAME_HOUSING_TYPE'].value_counts().values)])
iplot(pie)


pie = go.Figure(data=[go.Pie(labels = application_train['NAME_TYPE_SUITE'].value_counts().keys(),
                             values = application_train['NAME_TYPE_SUITE'].value_counts().values)])
iplot(pie)


pie = go.Figure(data=[go.Pie(labels = application_train['NAME_INCOME_TYPE'].value_counts().keys(),
                             values = application_train['NAME_INCOME_TYPE'].value_counts().values)])

iplot(pie)


pie = go.Figure(data=[go.Pie(labels = application_train['NAME_FAMILY_STATUS'].value_counts().keys(),
                             values = application_train['NAME_FAMILY_STATUS'].value_counts().values)])

iplot(pie)


fig = px.histogram(application_train, x="OCCUPATION_TYPE",
                  barmode='group', text_auto='.2s').update_xaxes(categoryorder = "total descending")
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig.show()


fig = px.histogram(application_train, x="OCCUPATION_TYPE", 
                   color="TARGET", barmode='group').update_xaxes(categoryorder = "total descending")
fig.show()


fig = px.histogram(application_train, x="ORGANIZATION_TYPE", 
                   color="TARGET", barmode='group').update_xaxes(categoryorder = "total descending")
fig.show()


fig = px.histogram(x= np.abs(application_train['DAYS_BIRTH'])/365,
                  barmode='group', nbins=15).update_xaxes(categoryorder = "total descending")
fig.update_layout(bargap=0.05,
                  xaxis = dict(title='Age (years)'))
fig.show()


fig = ff.create_distplot([np.abs(application_train.loc[application_train['TARGET'] == 0, 'DAYS_BIRTH']) / 365, 
                         np.abs(application_train.loc[application_train['TARGET'] == 1, 'DAYS_BIRTH']) / 365], 
                         ['target=> 0', 'target=> 1'],
                         bin_size = [2, 2],
                         show_rug=False,
                         show_hist=False,
                         show_curve=True)
fig.update_layout(xaxis = dict(title='Age (years)'),
                  yaxis =dict(title='Density'),
                  barmode='overlay')
fig.show()


fig = ff.create_distplot([application_train[application_train['TARGET']==0]['EXT_SOURCE_1'].dropna(), 
                          application_train[application_train['TARGET']==1]['EXT_SOURCE_1'].dropna()], 
                         ['target=> 0', 'target=> 1'],
                         bin_size = [2, 2],
                         show_rug=False,
                         show_hist=False,
                         show_curve=True)
fig.update_layout(xaxis = dict(title='EXT_SOURCE_1'),
                  yaxis =dict(title='Density'),
                  barmode='overlay')
fig.show()


fig = ff.create_distplot([application_train[application_train['TARGET']==0]['EXT_SOURCE_2'].dropna(), 
                          application_train[application_train['TARGET']==1]['EXT_SOURCE_2'].dropna()], 
                         ['target=> 0', 'target=> 1'],
                         bin_size = [2, 2],
                         show_rug=False,
                         show_hist=False,
                         show_curve=True)
fig.update_layout(xaxis = dict(title='EXT_SOURCE_2'),
                  yaxis =dict(title='Density'),
                  barmode='overlay')
fig.show()


fig = ff.create_distplot([application_train[application_train['TARGET']==0]['EXT_SOURCE_3'].dropna(), 
                          application_train[application_train['TARGET']==1]['EXT_SOURCE_3'].dropna()], 
                         ['target=> 0', 'target=> 1'],
                         bin_size = [2, 2],
                         show_rug=False,
                         show_hist=False,
                         show_curve=True)
fig.update_layout(xaxis = dict(title='EXT_SOURCE_3'),
                  yaxis =dict(title='Density'),
                  barmode='overlay')
fig.show()


join_1 = application_train.join(previous_application, how='left', on='SK_ID_CURR', 
                                rsuffix='_previous')
join_1.head()


join_1.shape


pie = go.Figure(data=[go.Pie(labels = join_1['NAME_CONTRACT_TYPE_previous'].value_counts().keys(),
                             values = join_1['NAME_CONTRACT_TYPE_previous'].value_counts().values)])

iplot(pie)


pie = go.Figure(data=[go.Pie(labels = join_1['NAME_CONTRACT_STATUS'].value_counts().keys(),
                             values = join_1['NAME_CONTRACT_STATUS'].value_counts().values)])

iplot(pie)


fig = px.histogram(join_1, x="NAME_CONTRACT_STATUS",
             color="TARGET", barmode='group').update_xaxes(categoryorder = "total descending")
fig.show()


fig = px.histogram(join_1, x="NAME_CONTRACT_STATUS",
                   color="FLAG_OWN_REALTY", barmode='group', 
                   facet_col='TARGET').update_xaxes(categoryorder = "total descending")
fig.show()


pie = go.Figure(data=[go.Pie(labels = join_1['NAME_PAYMENT_TYPE'].value_counts().keys(),
                             values = join_1['NAME_PAYMENT_TYPE'].value_counts().values)])

iplot(pie)


fig = px.histogram(join_1, x="NAME_PAYMENT_TYPE",
             color="TARGET", barmode='group').update_xaxes(categoryorder = "total descending")
fig.show()


pie = go.Figure(data=[go.Pie(labels = join_1['CODE_REJECT_REASON'].value_counts().keys(),
                             values = join_1['CODE_REJECT_REASON'].value_counts().values)])

iplot(pie)


fig = px.histogram(join_1, x="CODE_REJECT_REASON",
             color="TARGET", barmode='group').update_xaxes(categoryorder = "total descending")
fig.show()


fig = px.histogram(join_1, x="CODE_REJECT_REASON",
             color="FLAG_OWN_REALTY", barmode='group').update_xaxes(categoryorder = "total descending")
fig.show()


fig = px.histogram(join_1, x="CODE_REJECT_REASON",
                   color="FLAG_OWN_REALTY", barmode='group', 
                   facet_col='TARGET').update_xaxes(categoryorder = "total descending")
fig.show()


pie = go.Figure(data=[go.Pie(labels = join_1['NAME_TYPE_SUITE_previous'].value_counts().keys(),
                             values = join_1['NAME_TYPE_SUITE_previous'].value_counts().values)])

iplot(pie)


fig = px.histogram(join_1, x="NAME_TYPE_SUITE_previous",
             color="TARGET", barmode='group').update_xaxes(categoryorder = "total descending")
fig.show()


pie = go.Figure(data=[go.Pie(labels = join_1['NAME_PORTFOLIO'].value_counts().keys(),
                             values = join_1['NAME_PORTFOLIO'].value_counts().values)])

iplot(pie)


fig = px.histogram(join_1, x="NAME_PORTFOLIO",
             color="TARGET", barmode='group').update_xaxes(categoryorder = "total descending")
fig.show()


pie = go.Figure(data=[go.Pie(labels = join_1['CHANNEL_TYPE'].value_counts().keys(),
                             values = join_1['CHANNEL_TYPE'].value_counts().values)])

iplot(pie)


fig = px.histogram(join_1, x="CHANNEL_TYPE",
                   barmode='group', text_auto='.2s').update_xaxes(categoryorder = "total descending")
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig.show()


fig = ff.create_distplot([join_1[join_1['TARGET']==0]['AMT_APPLICATION'], 
                          join_1[join_1['TARGET']==1]['AMT_APPLICATION']], 
                         ['target=> 0', 'target=> 1'],
                         show_rug=False, 
                         show_hist=False,
                         show_curve=True)
fig.update_layout(xaxis = dict(title='AMT_APPLICATION'),
                  yaxis =dict(title='Density'),
                  barmode='overlay')
fig.show()


fig = px.histogram(join_1, x="CNT_PAYMENT", color="TARGET", barmode='group')
fig.show()


fig = ff.create_distplot([join_1[join_1['TARGET']==0]['AMT_CREDIT_previous'].dropna(), 
                          join_1[join_1['TARGET']==1]['AMT_CREDIT_previous'].dropna()], 
                         ['target=> 0', 'target=> 1'],
                         show_rug=False,
                         show_hist=False,
                         show_curve=True )

fig.update_layout(xaxis = dict(title='AMT_CREDIT_previous'),
                  yaxis =dict(title='Density'),
                  barmode='overlay')
fig.show()


fig = px.violin(join_1, x='FLAG_OWN_REALTY', y='AMT_CREDIT_previous', 
                color='TARGET', box = True)
fig.show()


join_2 = join_1.join(POS_CASH_balance, how='left', on='SK_ID_CURR', 
                                rsuffix='_POS_CASH')
join_2.head()


join_2.shape


fig = px.histogram(join_2, x="CNT_INSTALMENT", color="TARGET", barmode='group')
fig.show()


fig = px.histogram(join_2, x="CNT_INSTALMENT_FUTURE", color="TARGET", barmode='group')
fig.show()


pie = go.Figure(data=[go.Pie(labels = join_2['NAME_CONTRACT_STATUS_POS_CASH'].value_counts().keys(),
                             values = join_2['NAME_CONTRACT_STATUS_POS_CASH'].value_counts().values)])

iplot(pie)


fig = px.histogram(join_2, x="NAME_CONTRACT_STATUS_POS_CASH",
             color="TARGET", barmode='group').update_xaxes(categoryorder = "total descending")
fig.show()


join_3 = join_2.join(credit_card_balance, how='left', on='SK_ID_CURR', 
                                rsuffix='_credit')
join_3.head()


join_3.shape


fig = ff.create_distplot([join_3[join_3['TARGET']==0]['AMT_BALANCE'].dropna(), 
                          join_3[join_3['TARGET']==1]['AMT_BALANCE'].dropna()], 
                         ['target=> 0', 'target=> 1'],
                         show_rug=False,
                         show_hist=False,
                         show_curve=True )

fig.update_layout(xaxis = dict(title='AMT_BALANCE'),
                  yaxis =dict(title='Density'),
                  barmode='overlay')
fig.show()


fig = ff.create_distplot([join_3[join_3['TARGET']==0]['AMT_CREDIT_LIMIT_ACTUAL'].dropna(), 
                          join_3[join_3['TARGET']==1]['AMT_CREDIT_LIMIT_ACTUAL'].dropna()], 
                         ['target=> 0', 'target=> 1'],
                         show_rug=False,
                         show_hist=False,
                         show_curve=True )

fig.update_layout(xaxis = dict(title='AMT_CREDIT_LIMIT_ACTUAL'),
                  yaxis =dict(title='Density'),
                  barmode='overlay')

fig.show()


fig = ff.create_distplot([join_3[join_3['TARGET']==0]['AMT_RECEIVABLE_PRINCIPAL'].dropna(), 
                          join_3[join_3['TARGET']==1]['AMT_RECEIVABLE_PRINCIPAL'].dropna()], 
                         ['target=> 0', 'target=> 1'],
                         show_rug=False,
                         show_hist=False,
                         show_curve=True )

fig.update_layout(xaxis = dict(title='AMT_RECEIVABLE_PRINCIPAL'),
                  yaxis =dict(title='Density'),
                  barmode='overlay')

fig.show()


fig = ff.create_distplot([join_3[join_3['TARGET']==0]['AMT_RECIVABLE'].dropna(), 
                          join_3[join_3['TARGET']==1]['AMT_RECIVABLE'].dropna()], 
                         ['target=> 0', 'target=> 1'],
                         show_rug=False,
                         show_hist=False,
                         show_curve=True )

fig.update_layout(xaxis = dict(title='AMT_RECIVABLE'),
                  yaxis =dict(title='Density'),
                  barmode='overlay')

fig.show()


fig = ff.create_distplot([join_3[join_3['TARGET']==0]['AMT_TOTAL_RECEIVABLE'].dropna(), 
                          join_3[join_3['TARGET']==1]['AMT_TOTAL_RECEIVABLE'].dropna()], 
                         ['target=> 0', 'target=> 1'],
                         show_rug=False,
                         show_hist=False,
                         show_curve=True )

fig.update_layout(xaxis = dict(title='AMT_TOTAL_RECEIVABLE'),
                  yaxis =dict(title='Density'),
                  barmode='overlay')

fig.show()


pie = go.Figure(data=[go.Pie(labels = join_3['NAME_CONTRACT_STATUS_credit'].value_counts().keys(),
                             values = join_3['NAME_CONTRACT_STATUS_credit'].value_counts().values)])

iplot(pie)


join_4 = join_3.join(bureau, how='left', on='SK_ID_CURR', 
                                rsuffix='_bureau')
join_4.head()


join_4.shape


pie = go.Figure(data=[go.Pie(labels = join_4['CREDIT_ACTIVE'].value_counts().keys(),
                             values = join_4['CREDIT_ACTIVE'].value_counts().values)])
iplot(pie)


fig = px.histogram(join_4, x="TARGET",color='CREDIT_ACTIVE',
                  barmode='group', text_auto='.2s').update_xaxes(categoryorder = "total descending")
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig.show()


fig = px.histogram(join_4, x="CREDIT_CURRENCY",
                  barmode='group', text_auto='.2s').update_xaxes(categoryorder = "total descending")
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig.show()


fig = px.histogram(join_4, x="CREDIT_CURRENCY", color="TARGET",
                  barmode='group', text_auto='.2s').update_xaxes(categoryorder = "total descending")
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig.show()


fig = ff.create_distplot([np.abs(join_4[join_4['TARGET']==0]['DAYS_CREDIT']).dropna(), 
                          np.abs(join_4[join_4['TARGET']==1]['DAYS_CREDIT']).dropna()], 
                         ['target=> 0', 'target=> 1'],
                         show_rug=False,
                         show_hist=False,
                         show_curve=True )

fig.update_layout(xaxis = dict(title='DAYS_CREDIT'),
                  yaxis =dict(title='Density'),
                  barmode='overlay')

fig.show()


pie = go.Figure(data=[go.Pie(labels = join_4['CREDIT_TYPE'].value_counts().keys(),
                             values = join_4['CREDIT_TYPE'].value_counts().values)])
iplot(pie)


fig = px.histogram(join_4, x="CREDIT_TYPE",
                  barmode='group', text_auto='.2s').update_xaxes(categoryorder = "total descending")
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

# ----------------------------------------------------
import sklearn
import scipy
import statsmodels.api as sm 
from scipy.stats import shapiro

# ----------------------------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

# ----------------------------------------------------
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

# ----------------------------------------------------
from sklearn.metrics import auc, roc_curve, roc_auc_score
from collections import Counter

# ----------------------------------------------------
import warnings
warnings.filterwarnings("ignore")


def outlier_detect(df, col):
    q1_col = Q1[col]
    iqr_col = IQR[col]
    q3_col = Q3[col]
    return df[((df[col] < (q1_col - 1.5 * iqr_col)) |(df[col] > (q3_col + 1.5 * iqr_col)))]

# ----------------------------------------------------------
def lower_outlier(df, col):
    q1_col = Q1[col]
    iqr_col = IQR[col]
    q3_col = Q3[col]
    lower = df[(df[col] < (q1_col - 1.5 * iqr_col))]
    return lower

# ----------------------------------------------------------
def upper_outlier(df, col):
    q1_col = Q1[col]
    iqr_col = IQR[col]
    q3_col = Q3[col]
    upper = df[(df[col] > (q3_col + 1.5 * iqr_col))]
    return upper

# ----------------------------------------------------------
def preprocess(df, col):
    print("*********************** {} ***********************\n".format(col))
    print("lower outlier: {} ****** upper outlier: {}\n".format(lower_outlier(df,col).shape[0], upper_outlier(df,col).shape[0]))
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    df[col].plot(kind='box', subplots=True, sharex=False, vert=False)
    plt.subplot(2,1,2)
    df[col].plot(kind='density', subplots=True, sharex=False)
    plt.show()

# ----------------------------------------------------------
def preprocess_cat(df, col):
    print("******************** {} ********************\n".format(col))
    df[col].value_counts().plot(kind='bar')
    plt.xticks(rotation='vertical')
    plt.show()
    
# ----------------------------------------------------------
def replace_upper(df, col):
    q1_col = Q1[col]
    iqr_col = IQR[col]
    q3_col = Q3[col]
    tmp = 9999999
    upper = q3_col + 1.5 * iqr_col
    df[col] = df[col].where(lambda x: (x < (upper)), tmp)
    df[col] = df[col].replace(tmp, upper)

# ----------------------------------------------------------
def replace_lower(df, col):
    q1_col = Q1[col]
    iqr_col = IQR[col]
    q3_col = Q3[col]
    tmp = 1111111
    lower = q1_col - 1.5 * iqr_col
    df[col] = df[col].where(lambda x: (x > (lower)), tmp)
    df[col] = df[col].replace(tmp, lower)

# ----------------------------------------------------------
def replace_mode(df, col):
    df[col] = df[col].fillna(df[col].mode()[0])
    print("NaN in {} raplaced with {}".format(col, df[col].mode()[0]))

# ----------------------------------------------------------
def replace_mean(df, col):
    df[col] = df[col].fillna(df[col].mean())
    print("NaN in {} raplaced with {}".format(col, df[col].mean()))
    

def replace_median(df, col):
    df[col] = df[col].fillna(df[col].median())
    print("NaN in {} raplaced with {}".format(col, df[col].median()))
    

# ----------------------------------------------------------
kfold = StratifiedKFold(n_splits=5, random_state=100, shuffle=True)

def cross_validation(x, y, model):
    result= cross_val_score(model, x, y, cv=kfold, scoring="roc_auc", n_jobs=-1)
    print("Score: %f" % result.mean())
    
# ----------------------------------------------------------
def RndSrch_Tune(model, X, y, params):
    
    clf = RandomizedSearchCV(model, params, scoring ='roc_auc', cv = kfold, n_jobs=-1, random_state=100)
    clf.fit(X, y)
    print("best score is :" , clf.best_score_)
    print("best estimator is :" , clf.best_estimator_)
    print("best Params is :" , clf.best_params_)
    return (clf.best_score_)


train = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
test = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")
train.head()


train.columns


train.shape


sns.countplot(x = "TARGET", data = train)
train.loc[:, 'TARGET'].value_counts()


print(train.info())
print("*******************************")
print(test.info())


pd.set_option('display.max_rows', train.shape[0])
train.describe().T


pd.DataFrame(train.isnull().sum().sort_values(ascending = False))


pd.DataFrame(test.isnull().sum().sort_values(ascending = False))


threshold_train = len(train) * 0.60
int(threshold_train)


threshold_test = len(test) * 0.60
int(threshold_test)


print("In train data:\n")
print(train.columns[train.isnull().sum() > int(threshold_train)])
print("******************************************")
print("In test data:\n")
print(test.columns[test.isnull().sum() > int(threshold_test)])


train_new = train.dropna(axis=1, thresh=threshold_train)
print(train_new.shape)
print("******************************************")
test_new = test.dropna(axis=1, thresh=threshold_test)
print(test_new.shape)


numeric_feature = train_new.dtypes!=object
final_numeric_feature = train_new.columns[numeric_feature].tolist()

#----------------------------------------------------
numeric_feature_test = test_new.dtypes!=object
final_numeric_feature_test = test_new.columns[numeric_feature_test].tolist()


numeric = train_new[final_numeric_feature]

#-------------------------------------------
numeric_test = test_new[final_numeric_feature_test]
numeric.head()


discrete_features = numeric.dtypes==int
final_discrete_feature = numeric.columns[discrete_features].tolist()
discrete = numeric[final_discrete_feature]

#-------------------------------------------
discrete_features_test = numeric_test.dtypes==int
final_discrete_feature_test = numeric_test.columns[discrete_features_test].tolist()
discrete_test = numeric_test[final_discrete_feature_test]

discrete.head()


pd.DataFrame(discrete.isnull().sum().sort_values(ascending = False))



pd.DataFrame(discrete_test.isnull().sum().sort_values(ascending = False))


continuous_features = numeric.dtypes==float
final_continuous_feature = numeric.columns[continuous_features].tolist()
continuous = numeric[final_continuous_feature]

#-------------------------------------------
continuous_features_test = numeric_test.dtypes==float
final_continuous_feature_test = numeric_test.columns[continuous_features_test].tolist()
continuous_test = numeric_test[final_continuous_feature_test]

continuous.head()


pd.DataFrame(continuous.isnull().sum().sort_values(ascending = False))


pd.DataFrame(continuous_test.isnull().sum().sort_values(ascending = False))


continuous_col = continuous.columns


Q1 = train_new.select_dtypes(include='number').quantile(0.25)
Q3 = train_new.select_dtypes(include='number').quantile(0.75)
IQR = Q3 - Q1


for i in range(len(continuous_col)):
    preprocess(continuous[continuous_col], continuous_col[i])


continuous_is_null = continuous.isnull().sum() != 0
final_continuous_feature = continuous.columns[continuous_is_null].tolist()
print("In train: \n",final_continuous_feature)

print("****************************************")
continuous_is_null_test = continuous_test.isnull().sum() != 0
final_continuous_feature_test = continuous_test.columns[continuous_is_null_test].tolist()
print("In test: \n",final_continuous_feature_test)


print("In train:\n")
for i in range(len(final_continuous_feature)):
    replace_mean(continuous, final_continuous_feature[i])

print("************************************")
print("In test:\n")
for i in range(len(final_continuous_feature_test)):
    replace_mean(continuous_test, final_continuous_feature_test[i])


pd.DataFrame(continuous.isnull().sum().sort_values(ascending = False))


pd.DataFrame(continuous_test.isnull().sum().sort_values(ascending = False))


numeric[continuous_col] = continuous[continuous_col]

# ----------------------------------------------
numeric_test[continuous_col] = continuous_test[continuous_col]


col_names = numeric.columns

# ------------------------------------
col_names_test = numeric_test.columns


print("In train:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(outlier_detect(numeric,col_names[i]).shape[0])))
    
print("\n\n***************************************\n")
print("In test:\n")
for i in range(len(col_names_test)):
    print("{}: {}".format(col_names_test[i],(outlier_detect(numeric_test,col_names_test[i]).shape[0])))


outlier = []
for i in range(len(final_numeric_feature)):
    if outlier_detect(numeric[final_numeric_feature],final_numeric_feature[i]).shape[0] !=0:
        outlier.append(final_numeric_feature[i])

outlier_test = []
for i in range(len(final_numeric_feature_test)):
    if outlier_detect(numeric_test[final_numeric_feature_test],final_numeric_feature_test[i]).shape[0] !=0:
        outlier_test.append(final_numeric_feature_test[i])


# without TARGET field
col_names = outlier_test


print("In train:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(upper_outlier(numeric,col_names[i]).shape[0])))
    
print("\n\n****************************************\n")
print("In test:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(upper_outlier(numeric_test,col_names[i]).shape[0])))


for i in range(len(col_names)):
    replace_upper(numeric, col_names[i])   
    
#------------------------------------------------------
for i in range(len(col_names)):
    replace_upper(numeric_test, col_names[i])   


print("In train:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(upper_outlier(numeric,col_names[i]).shape[0])))
    
print("\n\n****************************************\n")
print("In test:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(upper_outlier(numeric_test,col_names[i]).shape[0])))


print("In train:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(lower_outlier(numeric,col_names[i]).shape[0])))
    
print("\n\n****************************************\n")
print("In test:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(lower_outlier(numeric_test,col_names[i]).shape[0])))


for i in range(len(col_names)):
    replace_lower(numeric, col_names[i])
    
# #--------------------------------------------------
for i in range(len(col_names)):
    replace_lower(numeric_test, col_names[i])


print("In train:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(lower_outlier(numeric,col_names[i]).shape[0])))
    
print("\n\n****************************************\n")
print("In test:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(lower_outlier(numeric_test,col_names[i]).shape[0])))


categorical_feature = train_new.dtypes==object
final_categorical_feature = train_new.columns[categorical_feature].tolist()

#----------------------------------------------------
categorical_feature_test = test_new.dtypes==object
final_categorical_feature_test = test_new.columns[categorical_feature_test].tolist()


categorical = train_new[final_categorical_feature]

#---------------------------------------------
categorical_test = test_new[final_categorical_feature_test]
categorical.head()


pd.DataFrame(categorical.isnull().sum().sort_values(ascending = False))


pd.DataFrame(categorical_test.isnull().sum().sort_values(ascending = False))


col_names_cat = categorical.columns


for i in range(len(col_names_cat)):
    preprocess_cat(categorical, col_names_cat[i])


print("unique number is = {}\nunique values are: \n{} ".format(len(train_new['ORGANIZATION_TYPE'].unique()), train_new['ORGANIZATION_TYPE'].unique()))


print("In train:\n")
for i in range(len(col_names_cat)):
    replace_mode(categorical, col_names_cat[i])

print("\n\n****************************************\n")
print("In test:\n")
for i in range(len(col_names_cat)):
    replace_mode(categorical_test, col_names_cat[i])


pd.DataFrame(categorical.isnull().sum().sort_values(ascending = False))


pd.DataFrame(categorical_test.isnull().sum().sort_values(ascending = False))


categorical.drop(['ORGANIZATION_TYPE'], axis=1, inplace=True)
# ---------------------------------------------
categorical_test.drop(['ORGANIZATION_TYPE'], axis=1, inplace=True)


le = LabelEncoder() 
categorical = categorical.apply(lambda col_names_cat: le.fit_transform(col_names_cat)) 
categorical_test = categorical_test.apply(lambda col_names_cat: le.fit_transform(col_names_cat)) 
categorical.head()


print("In train: ",categorical.shape)
print("In test: ",categorical_test.shape)


col_names_cat = categorical.columns
col_names = numeric_test.columns


train_new[col_names_cat] = categorical[col_names_cat]
train_new[col_names] = numeric[col_names]

# ----------------------------------------------------
test_new[col_names] = numeric_test[col_names]
test_new[col_names_cat] = categorical_test[col_names_cat]


train_new.drop(['ORGANIZATION_TYPE'], axis=1, inplace=True)
test_new.drop(['ORGANIZATION_TYPE'], axis=1, inplace=True)


print("In train: ",train_new.loc[train.duplicated()].shape)
#--------------------------------------------------
print("In test: ",test_new.loc[test.duplicated()].shape)


x_train = train_new.drop("TARGET", axis = 1)
y = train_new['TARGET']


scaler=MinMaxScaler()
col = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',
       'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT',
       'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
       'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
       'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
       'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'FLAG_MOBIL', 'FLAG_EMP_PHONE',
       'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
       'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START',
       'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
       'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
       'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
       'LIVE_CITY_NOT_WORK_CITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
       'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
       'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
       'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
       'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
       'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

x_train[col] = pd.DataFrame(scaler.fit_transform(x_train[col]))
test_new[col] = pd.DataFrame(scaler.transform(test_new[col]))


# Value of hyperparameters for random search

# param_lgb = {'learning_rate':[0.2,0.1,0.01,0.05,0.001],
#              'num_leaves':range(10,100,10),
#              'min_child_samples':range(500,1000,100),
#              'reg_alpha':[0.1,0.01,0.2,0.3],
#              'reg_lambda':[0.1,0.01,0.2,0.3],
#             'n_estimators':range(50,300,50),
#              'max_bin': range(500,1500,100)}

# RndSrch_Tune(LGBMClassifier(random_state = 100, n_jobs=-1, class_weight = 'balanced'), 
#              x_train, y, param_lgb)


lgb = LGBMClassifier(**{'reg_lambda': 0.1, 
                        'reg_alpha': 0.2, 
                        'num_leaves': 70, 
                        'n_estimators': 250, 
                        'min_child_samples': 800, 
                        'learning_rate': 0.05,
                        'max_bin': 500,
                        'objective': 'binary',
                        'n_jobs': -1,
                        'class_weight':'balanced',
                        'random_state':100})

scores = cross_validation(x_train, y, lgb)
print(scores)


counter = Counter(y)
estimate = counter[0] / counter[1]
print('Estimate: %.3f' % estimate)


# Value of hyperparameters for random search

# param_xgbc = {'learning_rate':[0.2,0.1,0.01,0.05,0.001],
#              'subsample':[1,0.5,0.2,0.1],
#              'max_depth' : range(2,11,1),
#              'n_estimators':range(50,300,50)}

# RndSrch_Tune(XGBClassifier(random_state = 100, n_jobs=-1, scale_pos_weight=estimate), 
#              x_train, y, param_xgbc)


xgbc = XGBClassifier(learning_rate=0.2, 
                     max_depth=4, 
                     n_jobs=-1, 
                     random_state=100,
                     scale_pos_weight=11.387150050352467)

scores_xgbc = cross_validation(x_train, y, xgbc)
print(scores_xgbc)


# Value of hyperparameters for random search

# param_ada = {'learning_rate':[0.2,0.1,0.01,0.05,0.001],
#              'algorithm': ['SAMME', 'SAMME.R'],
#              'n_estimators':range(50,300,50)}

# RndSrch_Tune(AdaBoostClassifier(random_state = 100), x_train, y, param_ada)


ada = AdaBoostClassifier(learning_rate=0.2, 
                         algorithm = 'SAMME.R',
                         n_estimators=200, 
                         random_state=100)
scores_ada = cross_validation(x_train, y, ada)
print(scores_ada)


classifiers = [lgb, xgbc, ada]

# Collect results in a list of dicts
results = []

for cls in classifiers:
    model = cls.fit(x_train, y)
    yproba = model.predict_proba(x_train)[:, 1]
    
    fpr, tpr, _ = roc_curve(y, yproba)
    auc = roc_auc_score(y, yproba)
    
    results.append({
        'classifiers': cls.__class__.__name__,
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc
    })

# Convert list of results into a DataFrame
result_table = pd.DataFrame(results)
result_table.set_index('classifiers', inplace=True)


fig = plt.figure(figsize=(10,8))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc'])
             )
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()


lgb.fit(x_train, y)
y_pred_LGB = lgb.predict(x_train)
y_pred_LGB_test = lgb.predict(test_new)


output = pd.DataFrame({'SK_ID_CURR': test_new.SK_ID_CURR, 
                       'TARGET': lgb.predict_proba(test_new)[:,1]})
output.head()


output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")




























































































































