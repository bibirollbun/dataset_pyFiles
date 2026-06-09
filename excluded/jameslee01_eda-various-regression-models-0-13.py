!pip install klib


import pandas as pd
import klib
import seaborn as sns

df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
df = klib.data_cleaning(df)


df.isna().sum()/len(df)


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

#df['num_sold'] = imputer.fit_transform(df[['num_sold']])
df = df.dropna()
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df.drop(columns=['id','date'],inplace=True)
categorical = df.select_dtypes(include=['category']).columns.values
numerical = df.select_dtypes(include=['number']).columns.values
print('categorical variables: ' + str(categorical))
print('numerical variables: ' + str(numerical))


df.head()


for i in categorical:
    print(df[i].value_counts())


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()

for i, col in enumerate(numerical):
    sns.histplot(data=df, x=col, bins=10, kde=False, ax=axes[i])
    axes[i].set_title(col)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2,figsize=(18, 15))
axes = axes.flatten()

for i, col in enumerate(categorical):
    sns.violinplot(data=df, x=col, y='num_sold', ax=axes[i])
    axes[i].set_title(col)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3,figsize=(18, 6))
axes = axes.flatten()
for i, col in enumerate(numerical[numerical != 'num_sold']):
  sns.scatterplot(data=df, x=col, y='num_sold', ax=axes[i])
  axes[i].set_title(col)

plt.tight_layout()
plt.show()


import numpy as np
from sklearn.preprocessing import StandardScaler

df['num_sold'] = np.log(df['num_sold'])

#scaler = StandardScaler()
#df[numerical] = scaler.fit_transform(df[numerical])

df = pd.get_dummies(df, columns=categorical, drop_first=False)


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression

X = df.drop(columns=['num_sold'])
y = df['num_sold']

mi_scores = mutual_info_regression(X, y, random_state=42)

mi_df = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': mi_scores
}).sort_values(by='MI_Score', ascending=False)
mi_df


from sklearn.metrics import mean_absolute_percentage_error as mape, r2_score as r2
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

def evaluate_model(model):

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mse = mape(y_train, y_pred_train)
    test_mse = mape(y_test, y_pred_test)
    train_r2 = r2(y_train, y_pred_train)
    test_r2 = r2(y_test, y_pred_test)

    print(f'Train MAPE: {train_mse}')
    print(f'Train R-squared: {train_r2}')
    print(f'Test MAPE: {test_mse}')
    print(f'Test R-squared: {test_r2}')


from sklearn.linear_model import LinearRegression

linear = LinearRegression()
evaluate_model(linear)


from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.001)
evaluate_model(lasso)


from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.001)
evaluate_model(ridge)


from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.001,
                         l1_ratio=0.001
                         )
evaluate_model(elastic_net)


from sklearn.ensemble import RandomForestRegressor
randomforest = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
evaluate_model(randomforest)


from sklearn.ensemble import GradientBoostingRegressor
gradient = GradientBoostingRegressor(n_estimators=200,random_state=42)
evaluate_model(gradient)


from xgboost import XGBRegressor
xgb_model = XGBRegressor(n_estimators=200, random_state=42)
evaluate_model(xgb_model)


from lightgbm import LGBMRegressor
lgbm_model = LGBMRegressor(n_estimators=200, random_state=42)
evaluate_model(lgbm_model)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

simple_nn = Sequential()
simple_nn.add(InputLayer(shape=(X_train.shape[1],)))
simple_nn.add(Dense(32, activation='relu'))
simple_nn.add(Dense(32, activation='relu'))
simple_nn.add(Dense(1,'linear'))

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

opt=Adam(learning_rate=0.001)
cp=ModelCheckpoint('models/simple_nn.keras', save_best_only=True)
simple_nn.compile(optimizer = opt,loss='mape',metrics=['mape'])

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

simple_nn.fit(X_train,y_train,validation_split=0.2,batch_size=16,epochs=10,callbacks=[cp,early_stopping])


X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

y_pred_train = simple_nn.predict(X_train)
y_pred_test = simple_nn.predict(X_test)

train_mse = mape(y_train, y_pred_train)
test_mse = mape(y_test, y_pred_test)
train_r2 = r2(y_train, y_pred_train)
test_r2 = r2(y_test, y_pred_test)

print(f'Train MAPE: {train_mse}')
print(f'Train R-squared: {train_r2}')
print(f'Test MAPE: {test_mse}')
print(f'Test R-squared: {test_r2}')


test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
test = klib.data_cleaning(test)

ids = test['id']
test['date'] = pd.to_datetime(test['date'])
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['day_of_week'] = test['date'].dt.dayofweek
test.drop(columns=['id','date'],inplace=True)
categorical = test.select_dtypes(include=['category']).columns.values
numerical = test.select_dtypes(include=['number']).columns.values


test = pd.get_dummies(test, columns=categorical, drop_first=False)


values = lgbm_model.predict(test)
#values = scaler.inverse_transform(values.reshape(-1, 1)).flatten()
values = np.exp(values)


import numpy as np
import seaborn as sns

real = pd.DataFrame({'id': ids, 'num_sold': np.ravel(values)})
sns.displot(data=real,kind='hist',x='num_sold',bins=20)


real.head()


real.to_csv('/kaggle/working/submission.csv', index=False)




