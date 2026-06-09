import numpy as np
import pandas as pd
import seaborn as sns

import calendar


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
train.head()


train.drop('id', axis =1, inplace = True)


X = train.iloc[:,:-1]
y = train.iloc[:,-1:]
X.shape


X_train, X_test, y_train, y_test = X[:1825],X[1825:], y[:1825], y[1825:]
X_train.shape


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as ras

model = XGBClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

ras(y_test, y_pred)


train.corr()['rainfall'][:-1].abs().sort_values(ascending = False)


sns.heatmap(train.corr())


sns.boxplot(x=train['pressure'])


train.loc[train['pressure'] > 1030, 'pressure'] = train['pressure'].mean()


train['year'] = [1]*365+[2]*365+[3]*365+[4]*365+[5]*365+[6]*365


months = [1]*31+[2]*28+[3]*31+[4]*30+[5]*31+[6]*30+[7]*31+[8]*31+[9]*30+[10]*31+[11]*30+[12]*31
train['month'] = months*6


date = pd.DataFrame({'date_column': pd.date_range(start='2023-01-01', periods=365, freq='D')})
date['date_column'] = pd.to_datetime(date['date_column'])
train['week_month'] = (date['date_column'].apply(lambda x: (x.day - 1) // 7 + 1).to_list())*6


weeks = [j for i in range(1, 53) for j in [i] * 7] + [53]
train['week_year'] = weeks*6


train['day_month'] = [j for i in range(1, 13) for j in [i] * calendar.monthrange(2023, i)[1]]*6


month_rainfall_mean = train[['month','rainfall']].iloc[:1825,:].groupby('month').mean().reset_index()
month_rainfall_mean.rename(columns={'rainfall':'month_rainfall_mean'}, inplace=True)
train = train.merge(month_rainfall_mean, on='month', how='left')


week_month_rainfall_mean = train[['week_month','rainfall']].iloc[:1825,:].groupby('week_month').mean().reset_index()
week_month_rainfall_mean.rename(columns={'rainfall':'week_month_rainfall_mean'}, inplace=True)
#train = train.merge(week_month_rainfall_mean, on='week_month', how='left') decreses accuracy


variables = list(train.columns[1:10].values)
train[variables].describe()


test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test[variables].describe()


STATS = ["mean","count"]
bins = [i for i in range(990,1041,5)]
train['pressure_bin']= np.searchsorted(bins, train['pressure'].values)
pressure_rainfall_mean = train.iloc[:1825,:].groupby('pressure_bin')['rainfall'].agg(STATS).reset_index().fillna(0)
pressure_rainfall_mean.columns = [f'pressure_rainfall_{stat}' if stat not in ['pressure_bin'] else stat for stat in pressure_rainfall_mean.columns]
#pressure_rainfall_mean.rename(columns={'rainfall':'pressure_rainfall_mean'}, inplace=True)
train = train.merge(pressure_rainfall_mean, on = 'pressure_bin', how='left')
train.drop('pressure_bin', axis = 1, inplace=True)


bins = [i for i in range(0,41,5)]
train['maxtemp_bin']= np.searchsorted(bins, train['maxtemp'].values)
maxtemp_rainfall_mean = train[:1825].groupby('maxtemp_bin')['rainfall'].mean().reset_index()
maxtemp_rainfall_mean.rename(columns={'rainfall':'maxtemp_rainfall_mean'}, inplace=True)
#train = train.merge(maxtemp_rainfall_mean, on = 'maxtemp_bin', how='left')
train.drop('maxtemp_bin', axis = 1, inplace=True)


STATS = ["mean","count"]
bins = [i for i in range(0,41,5)]
train['temparature_bin']= np.searchsorted(bins, train['temparature'].values)
temparature_rainfall_mean = train.iloc[:1825,:].groupby('temparature_bin')['rainfall'].agg(STATS).reset_index()
temparature_rainfall_mean.columns = [f'temparature_rainfall_{stat}' if stat not in ['temparature_bin'] else stat for stat in temparature_rainfall_mean.columns]
#temparature_rainfall_mean.rename(columns={'rainfall':'temparature_rainfall_mean'}, inplace=True)
train = train.merge(temparature_rainfall_mean, on = 'temparature_bin', how='left')
train.drop('temparature_bin', axis = 1, inplace=True)


bins = [i for i in range(0,41,5)]
train['mintemp_bin']= np.searchsorted(bins, train['mintemp'].values)
mintemp_rainfall_mean = train[:1825].groupby('mintemp_bin')['rainfall'].mean().reset_index()
mintemp_rainfall_mean.rename(columns={'rainfall':'mintemp_rainfall_mean'}, inplace=True)
#train = train.merge(mintemp_rainfall_mean, on = 'mintemp_bin', how='left')
train.drop('mintemp_bin', axis = 1, inplace=True)


bins = [i for i in range(-10,31,5)]
train['dewpoint_bin']= np.searchsorted(bins, train['dewpoint'].values)
dewpoint_rainfall_mean = train[:1825].groupby('dewpoint_bin')['rainfall'].mean().reset_index()
dewpoint_rainfall_mean.rename(columns={'rainfall':'dewpoint_rainfall_mean'}, inplace=True)
#train = train.merge(dewpoint_rainfall_mean, on = 'dewpoint_bin', how='left')
train.drop('dewpoint_bin', axis = 1, inplace=True)


bins = [i for i in range(30,101,5)]
train['humidity_bin']= np.searchsorted(bins, train['humidity'].values)
humidity_rainfall_mean = train[:1825].groupby('humidity_bin')['rainfall'].mean().reset_index()
humidity_rainfall_mean.rename(columns={'rainfall':'humidity_rainfall_mean'}, inplace=True)
#train = train.merge(humidity_rainfall_mean, on = 'humidity_bin', how='left')
train.drop('humidity_bin', axis = 1, inplace=True)


bins = [i for i in range(0,101,5)]
train['cloud_bin']= np.searchsorted(bins, train['cloud'].values)
cloud_rainfall_mean = train[:1825].groupby('cloud_bin')['rainfall'].mean().reset_index()
cloud_rainfall_mean.rename(columns={'rainfall':'cloud_rainfall_mean'}, inplace=True)
#train = train.merge(cloud_rainfall_mean, on = 'cloud_bin', how='left')
train.drop('cloud_bin', axis = 1, inplace=True)


bins = [i for i in range(0,17,2)]
train['sunshine_bin']= np.searchsorted(bins, train['sunshine'].values)
sunshine_rainfall_mean = train[:1825].groupby('sunshine_bin')['rainfall'].mean().reset_index()
sunshine_rainfall_mean.rename(columns={'rainfall':'sunshine_rainfall_mean'}, inplace=True)
#train = train.merge(sunshine_rainfall_mean, on = 'sunshine_bin', how='left')
train.drop('sunshine_bin', axis = 1, inplace=True)


bins = [i for i in range(0,351,30)]
train['winddirection_bin']= np.searchsorted(bins, train['winddirection'].values)
winddirection_rainfall_mean = train[:1825].groupby('winddirection_bin')['rainfall'].mean().reset_index()
winddirection_rainfall_mean.rename(columns={'rainfall':'winddirection_rainfall_mean'}, inplace=True)
#train = train.merge(winddirection_rainfall_mean, on = 'winddirection_bin', how='left')
train.drop('winddirection_bin', axis = 1, inplace=True)


year_variables_mean = train[['year']+variables].groupby('year').mean().reset_index()
year_variables_mean.columns = [f'{col}_mean' if col not in ['year'] else col for col in year_variables_mean.columns]
#train = train.merge(year_variables_mean, on='year', how='outer') increases error 
train.head()


month_year_variables_mean = train[['year','month']+variables].groupby(['year','month']).mean().reset_index()
month_year_variables_mean.columns = [f'{col}_mean' if col not in ['year','month'] else col for col in month_year_variables_mean.columns]
#train = train.merge(month_year_variables_mean, on=('year','month'), how='outer') decresses accuracy
train.head()


month_variables_mean = train[['month']+variables].groupby(['month']).mean().reset_index()
month_variables_mean.columns = [f'{col}_mean' if col not in ['month'] else col for col in month_variables_mean.columns]
#train = train.merge(month_year_variables_mean, on=('month'), how='outer')
train.head()


features = ['windspeed']
for feature in features:
    largest_num = train[feature].astype(str).max()
    largest_num_len = len(str(largest_num))-1
    num_digits_round = train[feature].astype(int).astype(str).apply(lambda x: len(x)).max()
    num_digits_total = train[feature].astype(str).apply(lambda x: len(x)).max()
    for i in range(1, num_digits_total):
        #train[f'cloud_digit{i}'] = ((train[feature] * 10**(i-num_digits_round)) % 10).fillna(0).astype("int8")
train.head()


train[]


train.columns


#train.drop('year', axis=1, inplace=True)
train.drop('month', axis=1, inplace=True) #commenting as i need the column in testing
#train.drop('week_month', axis=1, inplace=True)
#train.drop('week_year', axis=1, inplace=True)
#train.drop('day_month', axis=1, inplace=True)
#train.drop('day', axis=1, inplace=True)


X = train.drop('rainfall',axis = 1)
y = train.rainfall
X.shape


X_train, X_test, y_train, y_test = X[:1825],X[1825:], y[:1825], y[1825:]
X_train.shape


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as ras

model = XGBClassifier(
        #device="cuda",
        max_depth=6,  
        colsample_bynode=0.3, 
        subsample=0.8,  
        n_estimators=50_000,  
        learning_rate=0.00047,  #
        enable_categorical=True,
        min_child_weight=10,
        #early_stopping_rounds=500,
        random_state = 42
    )

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],  
          verbose=5000,
         )

y_pred = model.predict(X_test)

ras(y_test, y_pred)


0.8165371214151702


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt


model = load_model('/kaggle/working/my_model_backup.keras')


model.summary()


model.save('my_model_backup.keras')
!rm -rf /kaggle/working/my_model.keras


X = train.drop('rainfall',axis = 1)
y = train.rainfall


inputs = Input(shape=(X.shape[1],))
x = Dense(16, activation = 'relu')(inputs)
x = Dense(8, activation = 'relu')(x)
x = Dense(4, activation = 'relu')(x)
x = Dense(2, activation = 'relu')(x)
outputs = Dense(1, activation = 'sigmoid')(x)

model = Model(inputs, outputs)

model.summary()


model.compile(optimizer= 'adam', loss = 'binary_crossentropy',metrics=['AUC'])


history = model.fit(X,y,batch_size =1, epochs=10)
model.save('my_model.keras')
plt.plot(history.history['AUC'])


from sklearn.metrics import roc_auc_score as ras

y_pred = model.predict(X)
ras(y, y_pred)


0.9089539842873177


model.save('my_trained_model.h5')


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_scaled = ss.fit_transform(X)

lr = LogisticRegression()
lr.fit(X_scaled,y)
y_pred = lr.predict_proba(X)

ras(y, y_pred)


y_pred


test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test.head()


test.drop('id', axis =1, inplace = True)


test.fillna(0, inplace = True)


test['year'] = [7]*365+[8]*365


months = [1]*31+[2]*28+[3]*31+[4]*30+[5]*31+[6]*30+[7]*31+[8]*31+[9]*30+[10]*31+[11]*30+[12]*31
test['month'] = months*2
train['month'] = months*6


date = pd.DataFrame({'date_column': pd.date_range(start='2023-01-01', periods=365, freq='D')})
date['date_column'] = pd.to_datetime(date['date_column'])
test['week_month'] = (date['date_column'].apply(lambda x: (x.day - 1) // 7 + 1).to_list())*2


weeks = [j for i in range(1, 53) for j in [i] * 7] + [53]
test['week_year'] = weeks*2


test['day_month'] = [j for i in range(1, 13) for j in [i] * calendar.monthrange(2023, i)[1]]*2


test.columns


month_rainfall_mean = train[['month','rainfall']].groupby('month').mean().reset_index()
month_rainfall_mean.rename(columns={'rainfall':'month_rainfall_mean'}, inplace=True)
test = test.merge(month_rainfall_mean, on='month', how='left')


STATS = ["mean","count"]
bins = [i for i in range(990,1041,5)]
train['pressure_bin']= np.searchsorted(bins, train['pressure'].values)
test['pressure_bin']= np.searchsorted(bins, test['pressure'].values)
pressure_rainfall_mean = train.groupby('pressure_bin')['rainfall'].agg(STATS).reset_index().fillna(0)
pressure_rainfall_mean.columns = [f'pressure_rainfall_{stat}' if stat not in ['pressure_bin'] else stat for stat in pressure_rainfall_mean.columns]
#pressure_rainfall_mean.rename(columns={'rainfall':'pressure_rainfall_mean'}, inplace=True)
test = test.merge(pressure_rainfall_mean, on = 'pressure_bin', how='left')
train.drop('pressure_bin', axis = 1, inplace=True)
test.drop('pressure_bin', axis = 1, inplace=True)


STATS = ["mean","count"]
bins = [i for i in range(0,41,5)]
train['temparature_bin']= np.searchsorted(bins, train['temparature'].values)
test['temparature_bin']= np.searchsorted(bins, test['temparature'].values)
temparature_rainfall_mean = train[:1825].groupby('temparature_bin')['rainfall'].agg(STATS).reset_index()
temparature_rainfall_mean.columns = [f'temparature_rainfall_{stat}' if stat not in ['temparature_bin'] else stat for stat in temparature_rainfall_mean.columns]
#temparature_rainfall_mean.rename(columns={'rainfall':'temparature_rainfall_mean'}, inplace=True)
test = test.merge(temparature_rainfall_mean, on = 'temparature_bin', how='left')
train.drop('temparature_bin', axis = 1, inplace=True)
test.drop('temparature_bin', axis = 1, inplace=True)


test.drop('month', axis=1, inplace=True)


test.shape


!rm -rf /kaggle/working/submission.csv


y_pred = model.predict(test)


sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sub['rainfall'] = y_pred
sub.to_csv('/kaggle/working/submission.csv', index=False)


np.argwhere(np.isnan(test))


test.iloc[517,:]




