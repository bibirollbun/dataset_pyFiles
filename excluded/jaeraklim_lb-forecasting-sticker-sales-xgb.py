import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')

train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


display(train.head(), test.head())


display(train.info(), test.info())


display(train.describe(), test.describe())


display(train.describe(exclude='number'), test.describe(exclude='number'))


train.isnull().sum()[train.isnull().sum()>0]


train.isnull().sum()[train.isnull().sum()>0].values /  train.shape[0] * 100

# There are very few null values only in the target column.


df = train.dropna().drop('id',axis=1)

# drop null values.


df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
#df['day_name'] = df['date'].dt.day_name()
df['weekday'] = df['date'].dt.weekday
df['holiday'] = df['weekday'].apply(lambda x: 1 if x in [0,6] else 0)
df = df.drop('date', axis=1)

test['date'] = pd.to_datetime(test['date'])
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
#test['day_name'] = test['date'].dt.day_name()
test['weekday'] = test['date'].dt.weekday
test['holiday'] = test['weekday'].apply(lambda x: 1 if x in [0,6] else 0)
test = test.drop('date', axis=1)

display(df.head(), test.head())


for col in df.drop('num_sold', axis=1):
    fig,axes = plt.subplots(1,3,figsize=(20,3))
    sns.countplot(x=col,data=df, ax=axes[0])
    sns.barplot(x=col, y='num_sold', data=df, ax=axes[1])
    sns.boxplot(x=col,y='num_sold',data=df, ax=axes[2])
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    axes[0].set_title(f'count plot of {col}')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].set_title(f'bar plot of num_sold by {col}')
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')
    axes[2].set_title(f'count plot of num_sold by {col}')
    if col in ['store','product']:
        axes[0].tick_params(rotation=45)
        axes[1].tick_params(rotation=45)
        axes[2].tick_params(rotation=45)


sns.histplot(df['num_sold'])
plt.show()

# A log transformation is required for the target values.


from sklearn.preprocessing import LabelEncoder

encoder = {}

for col in df.select_dtypes('object'):
    le = LabelEncoder()
    le.fit(pd.concat([df[col],test[col]], axis=0))
    df[col] = le.transform(df[col])
    test[col] = le.transform(test[col])
    encoder[col] = le

df.info()


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
#from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

data = df.drop('num_sold', axis = 1)
target = df['num_sold']

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=123)

rf = RandomForestRegressor()
lr = LinearRegression()
#svr = SVR()
xgb = XGBRegressor()

y_train_log = np.log1p(y_train)

model = [rf,lr,xgb]

for model in model:
    model.fit(X_train, y_train_log)
    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)
    score = mean_absolute_percentage_error(y_test, pred)
    print(f'{model}: {score}')

# The XGBoost algorithm demonstrated the best performance.


target_log = np.log1p(target)

xgb.fit(data, target_log)
pred_log = xgb.predict(test.drop('id',axis=1))
pred = np.expm1(pred_log)
test['num_sold'] = pred
test[['id','num_sold']].to_csv('submission.csv', index=False)

