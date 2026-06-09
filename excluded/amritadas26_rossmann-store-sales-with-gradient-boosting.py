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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'iframe'

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


ross_df=pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv',low_memory=False)
test_df=pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')
store_df=pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')


ross_df


test_df


store_df


merged_df = ross_df.merge(store_df, how='left', on='Store')
merged_test_df = test_df.merge(store_df, how='left', on='Store')


merged_df


sns.histplot(data=merged_df, x='Sales')


merged_df[merged_df.Open == 0].Sales.value_counts()


merged_df = merged_df[merged_df.Open == 1].copy()


sns.histplot(data=merged_df, x='Sales')


merged_df.info()


def split_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = df.Date.dt.isocalendar().week


split_date(merged_df)
split_date(merged_test_df)


merged_df


sns.barplot(data=merged_df, x='Year', y='Sales',palette="husl")


sns.barplot(data=merged_df, x='Month', y='Sales',palette="husl")





def comp_months(df):
    df['CompetitionOpen']=12 * (df.Year - df.CompetitionOpenSinceYear) + (df.Month - df.CompetitionOpenSinceMonth)
    df['CompetitionOpen']=df['CompetitionOpen'].map(lambda x:0 if x<0 else x).fillna(0)


comp_months(merged_df)
comp_months(merged_test_df)


merged_df[['Date', 'CompetitionDistance', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'CompetitionOpen']].sample(20)


def check_promo_month(row):
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',              
                 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    try:
        months = (row['PromoInterval'] or '').split(',')
        if row['Promo2Open'] and month2str[row['Month']] in months:
            return 1
        else:
            return 0
    except Exception:
        return 0

def promo_cols(df):
    # Months since Promo2 was open
    df['Promo2Open'] = 12 * (df.Year - df.Promo2SinceYear) +  (df.WeekOfYear - df.Promo2SinceWeek)*7/30.5
    df['Promo2Open'] = df['Promo2Open'].map(lambda x: 0 if x < 0 else x).fillna(0) * df['Promo2']
    # Whether a new round of promotions was started in the current month
    df['IsPromo2Month'] = df.apply(check_promo_month, axis=1) * df['Promo2']


promo_cols(merged_df)
promo_cols(merged_test_df)


merged_df[['Date', 'Promo2', 'Promo2SinceYear', 'Promo2SinceWeek', 'PromoInterval', 'Promo2Open', 'IsPromo2Month']].sample(20)


merged_df.columns


input_cols = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 
              'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpen', 
              'Day', 'Month', 'Year', 'WeekOfYear',  'Promo2', 
              'Promo2Open', 'IsPromo2Month']
target_col = 'Sales'



inputs=merged_df[input_cols].copy()
targets=merged_df[target_col].copy()


test_inputs=merged_test_df[input_cols].copy()
#no target columns for test set


inputs.select_dtypes(np.number).columns


inputs.select_dtypes(object).columns


merged_df.info()


numeric_cols = ['Store', 'Promo', 'SchoolHoliday', 
              'CompetitionDistance', 'CompetitionOpen', 'Promo2', 'Promo2Open', 'IsPromo2Month',
              'Day', 'Month', 'Year', 'WeekOfYear',  ]

#we will consider DayOfWeek also as a categorical_cols
categorical_cols = ['DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment']


inputs[numeric_cols].isna().sum()


test_inputs[numeric_cols].isna().sum()


max_distance =inputs.CompetitionDistance.max()


inputs['CompetitionDistance'].fillna(max_distance*2,inplace=True)
test_inputs['CompetitionDistance'].fillna(max_distance*2,inplace=True)


test_inputs[numeric_cols].isna().sum()


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler().fit(inputs[numeric_cols])


inputs[numeric_cols]=scaler.transform(inputs[numeric_cols])
test_inputs[numeric_cols]=scaler.transform(test_inputs[numeric_cols])


from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore').fit(inputs[categorical_cols])



encoded_cols=list(encoder.get_feature_names_out(categorical_cols))


encoded_cols


inputs[encoded_cols] = encoder.transform(inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


X=inputs[numeric_cols+encoded_cols]
X_test=test_inputs[numeric_cols+encoded_cols]


from xgboost import XGBRegressor


model =XGBRegressor(random_state=42, n_jobs=-1, n_estimators=20, max_depth=4)


model.fit(X,targets) # Fitting training data


preds = model.predict(X)
preds


from sklearn.metrics import mean_squared_error

def rmse(pred, target):
    return np.sqrt(mean_squared_error(pred, target))


def rmspe(pred, target):
    target = np.array(target)
    pred = np.array(pred)
    
    # Filter out zero values in target
    non_zero_mask = target != 0
    target = target[non_zero_mask]
    pred = pred[non_zero_mask]
    
    # Calculate RMSPE
    rmspe_value = np.sqrt(np.mean(((target - pred) / target) ** 2)) * 100
    return f"{rmspe_value:.2f}%"


rmse(preds, targets),rmspe(preds, targets)


import matplotlib.pyplot as plt
from xgboost import plot_tree
from matplotlib.pylab import rcParams
%matplotlib inline

rcParams['figure.figsize'] = 30,30


plot_tree(model, rankdir='LR');



plot_tree(model, rankdir='LR', num_trees=1);



plot_tree(model, rankdir='LR', num_trees=19);



trees = model.get_booster().get_dump()
len(trees)


print(trees[0])


importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)


importance_df.head(10)


sns.barplot(data=importance_df.head(10), x='importance', y='feature',palette="husl");



from sklearn.model_selection import train_test_split
X_train, X_val, train_targets, val_targets = train_test_split(X, targets, test_size=0.1)


model = XGBRegressor(n_jobs=-1, random_state=42, n_estimators=1000, 
                     learning_rate=0.2, max_depth=10, subsample=0.9, 
                     colsample_bytree=0.7)


%%time
model.fit(X, targets)


train_rmspe=rmspe(model.predict(X_train),train_targets)
val_rmspe= rmspe(model.predict(X_val),val_targets)
print('Train RMSE: {}, Validation RMSE: {}'.format(train_rmspe, val_rmspe))


test_preds = model.predict(X_test)
print(test_preds[:10])


submission_df=pd.read_csv("/kaggle/input/rossmann-store-sales/sample_submission.csv")


test_df.Open.isna().sum()


submission_df['Sales'] = test_preds * test_df['Open'].astype('float')
submission_df.fillna(0, inplace=True)
submission_df.to_csv('submission.csv', index=None)



submission_df.head()




