# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import plotly as plt
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected = True)
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from statistics import mean
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Read in the data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", parse_dates=['date'])
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", parse_dates=['date'])


#Establish which columns are numerical and categorical
num_cols = [colname for colname in train_df.columns if train_df[colname].dtype in ['int64','float64']]
cat_cols = [colname for colname in train_df.columns if (train_df[colname].dtype in ['object'])]


#sns.barplot(data = train_df, x = 'date', y = 'num_sold')


#sns.barplot(data = train_df[train_df['country']=='Canada'], x = 'date', y = 'num_sold')


#Analyse dates
print(train_df['date'].min())
print(train_df['date'].max())


for i in cat_cols:
    fig , axes = plt.subplots(1,2, figsize=(10,6))
    sns.countplot(data = train_df, x = i, ax = axes[0])
    sns.boxplot(data= train_df, x = i, y = 'num_sold', ax = axes[1])
    plt.show()





print(f"Number of missing num_sold rows: {train_df['num_sold'].isna().sum()}")
print(f"Number of total rows: {train_df.shape[0]}")


train_df.groupby(['country','product','store'])['num_sold'].count()


#Removing values where num_sold is NAN
train_df_exnan = train_df[train_df['num_sold'].isnull()!=True]


#Read in GDP data
gdp_per_capita = pd.read_csv('/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv')
countries = train_df['country'].unique()
filtered_gdp= gdp_per_capita[gdp_per_capita['Country Name'].isin(countries)]


gdp_years = [str(y) for y in range(2010,2021)]
gdp_ratio = filtered_gdp.set_index("Country Name")[gdp_years].div(
    filtered_gdp.set_index("Country Name")[gdp_years].sum(axis=0),
    axis=1
)


train_df_exnan['gdp'] = train_df_exnan.apply(lambda x: gdp_ratio.loc[x.country, str(x.date.year)], axis=1)
test_df['gdp'] = test_df.apply(lambda x: gdp_ratio.loc[x.country, str(x.date.year)], axis=1)


#Adding date / time variables

train_df_exnan['date'] = pd.to_datetime(train_df_exnan['date'])
train_df_exnan['year'] = pd.to_datetime(train_df_exnan['date']).dt.year
train_df_exnan['day'] = pd.to_datetime(train_df_exnan['date']).dt.day
train_df_exnan['weekday'] = pd.to_datetime(train_df_exnan['date']).dt.weekday 
train_df_exnan['month']= pd.to_datetime(train_df_exnan['date']).dt.month
train_df_exnan['quarter']=pd.to_datetime(train_df_exnan['date']).dt.quarter
train_df_exnan['week']  = pd.to_datetime(train_df_exnan['date']).dt.isocalendar().week


test_df['date'] = pd.to_datetime(test_df['date'])
test_df['year'] = pd.to_datetime(test_df['date']).dt.year
test_df['day'] = pd.to_datetime(test_df['date']).dt.day
test_df['weekday'] = pd.to_datetime(test_df['date']).dt.weekday 
test_df['month']= pd.to_datetime(test_df['date']).dt.month
test_df['quarter']=pd.to_datetime(test_df['date']).dt.quarter
test_df['week']  = pd.to_datetime(test_df['date']).dt.isocalendar().week


#Add time series variables
train_df_exnan['weekday_sin'] = np.sin(2 * np.pi * train_df_exnan['weekday'] / 7)
train_df_exnan['weekday_cos'] = np.cos(2 * np.pi * train_df_exnan['weekday'] / 7)
train_df_exnan['week_sin'] = np.sin(2 * np.pi * train_df_exnan['week'] / 52)
train_df_exnan['week_cos'] = np.cos(2 * np.pi * train_df_exnan['week'] / 52)

test_df['weekday_sin'] = np.sin(2 * np.pi * test_df['weekday'] / 7)
test_df['weekday_cos'] = np.cos(2 * np.pi * test_df['weekday'] / 7)
test_df['week_sin'] = np.sin(2 * np.pi * test_df['week'] / 52)
test_df['week_cos'] = np.cos(2 * np.pi * test_df['week'] / 52)


train_df_exnan.info()


#This ignores week directly
X_cat_cols = ['country', 'store', 'product']
X_num_cols = ['year','gdp','day','month','quarter', 'weekday', 'weekday_sin','weekday_cos','week_sin','week_cos']
X_cols = X_cat_cols + X_num_cols
y_cols = 'num_sold'


X = train_df_exnan[X_cols]
y = train_df_exnan[y_cols]
test = test_df[X_cols]


#One hot encode categorical variables

cols_to_encode = ['country','store','product']
X_OH = pd.get_dummies(X,columns = cols_to_encode)
test_OH = pd.get_dummies(test,columns = cols_to_encode)



#Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_OH,y, test_size=0.10, shuffle=True, random_state=24)


X_train = X_train.to_numpy().astype(int)
X_val = X_val.to_numpy().astype(int)
test  = test_OH.to_numpy().astype(int)


model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_absolute_percentage_error')


model.fit(X_train,y_train, validation_data=(X_val,y_val), epochs = 8)


test_df['num_sold'] = model.predict(test)


submission = test_df[['id', 'num_sold']]
submission.to_csv("submission.csv", index=False)

