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


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_2=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submissions=pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train.head()


train_2.head()


print(train_2.shape)
print(train.shape)


plt.hist(x=train_2['Price'])
plt.title('Distribution of price')


train_2.isnull().sum()


plt.hist(x=train['Price'])
plt.title('Distribution of price')


train.info()


train.isnull().sum()


features_with_na=[feature for feature in train.columns if train[feature].isnull().sum()>1]
features_with_na


#identifying impact of features with null values on our dependent variable 
rows=int(np.ceil(len(features_with_na)/3))
col=3
#creating figure and set size
fig, axes=plt.subplots(rows,col,figsize=(15, rows*5))
axes=axes.flatten()

#looping throught features and creating suplots
for idx, feature in enumerate(features_with_na):
    data=train.copy()
    data[feature]=np.where(data[feature].isnull(),1,0)
    #plot on current subplot
    data.groupby(feature)['Price'].median().plot.bar(ax=axes[idx],color=['lightblue','orange'])
    axes[idx].set_title(feature)
    axes[idx].set_xlabel('') 
#hides any unused subplots
for idx in range(len(features_with_na), len(axes)):
    fig.delaxes(axes[idx])
plt.tight_layout()
plt.show()


numerical_features=[feature for feature in train.columns if train[feature].dtype!='O']
print('Numerical Features of data are :',numerical_features)


sns.boxplot(x='Weight Capacity (kg)', y='Brand',data=train_2)
plt.title('Average weights of bagpack based on brands')


train['Weight Capacity (kg)']=train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].median())
train_2['Weight Capacity (kg)']=train_2['Weight Capacity (kg)'].fillna(train_2['Weight Capacity (kg)'].median())


categorical_features=[feature for feature in train.columns if train[feature].dtype=='O']
categorical_features


train_len=len(train)
train_merge=pd.concat([train,train_2],axis=0,ignore_index=False)


train_merge.isnull().sum()


rows = int(np.ceil(len(categorical_features) / 3))  # Ensure it aligns with categorical_features
col = 3

# Creating figure and setting size
fig, axes = plt.subplots(rows, col, figsize=(15, rows * 5))
axes = axes.flatten()

# Looping through features and creating subplots
for idx, feature in enumerate(categorical_features):
    data = train_merge.copy()
    
    # Plot on current subplot
    sns.countplot(x=feature, data=data, ax=axes[idx])
    axes[idx].set_title(feature)
    axes[idx].set_xlabel('')

# Hide any unused subplots
for idx in range(len(categorical_features), len(axes)):  # Aligning with categorical_features
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()


from sklearn.impute import SimpleImputer

# Define the imputer with "constant" strategy, replacing missing values with "Unknown"
imputer = SimpleImputer(strategy="constant", fill_value="Unknown")


train_merge[categorical_features] = imputer.fit_transform(train_merge[categorical_features])

# Check if missing values are handled
print(train_merge.isnull().sum())


# Data preprocessing on test dataset 
test['Weight Capacity (kg)']=test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].median())
test[categorical_features]=imputer.transform(test[categorical_features])
test.isnull().sum()


#mapping size category 
size_mapping={
    'Unknown':0,
    'Small': 1,
    'Medium': 2,
    'Large': 3
}
train_merge['Size']=train_merge['Size'].replace(size_mapping)
test['Size']=test['Size'].replace(size_mapping)



def encode_binary_with_unknown(column):
    return column.map({"Yes": 1, "No": 0, "Unknown": 2}) 

train_merge["Laptop Compartment"] = encode_binary_with_unknown(train_merge["Laptop Compartment"])
train_merge["Waterproof"] = encode_binary_with_unknown(train_merge["Waterproof"])
test["Laptop Compartment"] = encode_binary_with_unknown(test["Laptop Compartment"])
test["Waterproof"] = encode_binary_with_unknown(test["Waterproof"])


train_merge.info()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

cat_cols=['Brand','Material','Style','Color']

for feature in cat_cols:
    train_merge[feature]=le.fit_transform(train_merge[feature])
    test[feature]=le.transform(test[feature])


test.drop(['id'],inplace=True,axis=1)


y=train_merge['Price']
train_merge.drop(['Price','id'], inplace=True, axis=1)


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
feature_scale=[feature for feature in train_merge if feature not in ['id','Price']]


scaler.fit(train_merge[feature_scale])
train_scaled=pd.concat([pd.DataFrame(scaler.transform(train_merge[feature_scale]), columns=feature_scale)],
                    axis=1)
train_scaled.head()


test_scaled=pd.concat([pd.DataFrame(scaler.transform(test[feature_scale]), columns=feature_scale)],
                    axis=1)
test_scaled.head()


from sklearn.model_selection import train_test_split, cross_val_score
from catboost import CatBoostRegressor 
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score


X_train, X_test, y_train, y_test=train_test_split(train_scaled,y,test_size=0.2,random_state=42)


from sklearn.linear_model import LinearRegression

# Initialize and train the model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Predict and evaluate
y_pred_lr = model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f"Linear Regression MSE: {mse_lr}")
rmse = mse_lr ** 0.5
print("RMSE",rmse)


model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
#n_estimators=200, learning_rate=0.2, max_depth=5, random_state=42,subsample=0.8

# Fit the model
model_xgb.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = model_xgb.predict(X_test)
# RMSE calculation
rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))  
print(f'RMSE Score: {rmse:.4f}')


y_pred=model_xgb.predict(test_scaled)


submissions['Price']=y_pred


submissions.to_csv('submission.csv', index=False)
print("✅ Submission file saved successfully!")

