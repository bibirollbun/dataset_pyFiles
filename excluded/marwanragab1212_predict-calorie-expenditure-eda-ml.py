import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import mean_squared_error,mean_squared_log_error
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error,mean_squared_log_error



train_df=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train_df.head(5)


def info(data):
  print('Shape of data:',data.shape)
  print("----------------------------")
  print('Columns:',data.columns)
  print("----------------------------")
  print('Null values:',data.isnull().sum())
  print("----------------------------")
  print('Data types:',data.dtypes)
  print("----------------------------")
  print('Unique values:',data.nunique())
  print("----------------------------")



info(train_df)


info(test_df)


train_df.describe()


def age_conv(data):
  data['Age']=data['Age'].astype(np.float64)
  return data


age_conv(train_df)
age_conv(test_df)


train_df.dtypes


test_df.dtypes


train_df=pd.get_dummies(train_df,columns=['Sex'],drop_first=True).astype(np.float64)
test_df=pd.get_dummies(test_df,columns=['Sex'],drop_first=True).astype(np.float64)


plt.scatter(train_df['Calories'],train_df['Duration'])
plt.xlabel('Calories')
plt.ylabel('Duration')
plt.title('Calories vs Duration')
plt.show()


plt.scatter(train_df['Calories'],train_df['Heart_Rate'])
plt.xlabel('Calories')
plt.ylabel('Heart_Rate')
plt.title('Calories vs Heart_Rate')
plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Matrix')
plt.show()


def create_BMI(data):
  data['BMI'] = data['Weight'] / (data['Height']/100)**2
  return data


create_BMI(train_df)
create_BMI(test_df)


plt.scatter(train_df['Calories'],train_df['BMI'])
plt.xlabel('Calories')
plt.ylabel('BMI')
plt.title('Calories vs BMI')
plt.show()


plt.scatter(train_df['Calories'],train_df['Age'])
plt.xlabel('Calories')
plt.ylabel('Age')
plt.title('Calories vs Age')
plt.show()


plt.scatter(train_df['Calories'],train_df['Sex_male'])
plt.xlabel('Calories')
plt.ylabel('Sex_male')
plt.title('Calories vs Sex_male')
plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Matrix')
plt.show()


cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
for col in cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=train_df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


train_df.duplicated().sum()


test_df.duplicated().sum()


train_df=train_df.drop("id",axis=1)
test_df=test_df.drop("id",axis=1)


X=train_df.drop("Calories",axis=1)
y=train_df["Calories"]



scaler=StandardScaler()
X=scaler.fit_transform(X)
test_df=scaler.fit_transform(test_df)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


model1 = XGBRegressor(
    n_estimators=500,       
    learning_rate=0.03,      
    max_depth=6,             
    min_child_weight=1,      
    subsample=0.8,          
    colsample_bytree=0.8,   
    gamma=0,               
    reg_alpha=0.1,         
    reg_lambda=1,          
    random_state=42
)

model1.fit(X_train, y_train)

y_pred=model1.predict(X_test)

print(f'MSE: {mean_squared_error(y_test,y_pred)}')
rmsle=np.sqrt(mean_squared_log_error(y_test,y_pred))
print(rmsle)


import lightgbm as lgb

model = lgb.LGBMRegressor(
    objective='regression',
    learning_rate=0.05,
    n_estimators=2000,
    num_leaves=64,
    max_depth=10,
    min_data_in_leaf=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l1=1.0,
    lambda_l2=1.0,
    random_state=42
)

model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print(f'MSE: {mean_squared_error(y_test,y_pred)}')
rmsle=np.sqrt(mean_squared_log_error(y_test,y_pred))
print(rmsle)


ts=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
y_predt=model.predict(test_df)
sumbmission = pd.DataFrame({
    "id": ts["id"],
    "Calories": y_predt
})
sumbmission.to_csv("submission.csv",index=False)
sumbmission.head()

