import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor




import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




train=pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train.columns


train.info
train.isnull().sum()


train.shape


train.head()


cat_cols = ['road_type','lighting','weather','time_of_day']
le=LabelEncoder()

for col in cat_cols:
    train[col] = le.fit_transform(train[col])


train.head(3)


bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in bool_cols:
    train[col]= train[col].astype(int)


train = train.drop(columns=['id'])


x = train.drop(columns=['accident_risk'])
y = train['accident_risk'] 


x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=42)


lr=LinearRegression()
lr.fit(x_train,y_train)


y_pred = lr.predict(x_val)



rmse = np.sqrt(mean_squared_error(y_val, y_pred))
rmse



rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_val)




rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
rmse_rf


n_plot = 100
y_val_plot = y_val.iloc[:n_plot]
y_pred_lr_plot = lr.predict(x_val.iloc[:n_plot])
y_pred_rf_plot = rf.predict(x_val.iloc[:n_plot])

plt.figure(figsize=(15,5))

plt.plot(y_val_plot.values, label='True Value', marker='o')
plt.plot(y_pred_lr_plot, label='Linear Regression Predict', marker='x')
plt.plot(y_pred_rf_plot, label='Random Forest Predict', marker='s')

plt.title('True or Predict Valur (First 100 Lines)')
plt.xlabel('Örnekler')
plt.ylabel('Accident Risk')
plt.legend()
plt.show()











