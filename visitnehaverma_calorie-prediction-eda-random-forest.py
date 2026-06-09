# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
sns.set_style('white')
plt.style.use("dark_background")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings
# Specifically ignore Deprecation Warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Ignore future warnings
warnings.filterwarnings('ignore', category=FutureWarning)

warnings.filterwarnings('ignore', category=RuntimeWarning)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
data.head()


print(f"No. of Null Values: {data.isna().sum()}")


data.describe()


data.drop(['id'], axis=1,inplace=True)


data.info()


sns.histplot(data['Calories'])
plt.title("Target Variable before Transformation")
plt.show()


data['Calories_trans'] = np.sqrt(data['Calories'])


sns.histplot(data['Calories_trans'], bins=100)
plt.title("Target Variable after Transformation")
plt.show()


plt.figure(figsize=(15,8))

sns.scatterplot(data, x='Weight', y='Calories_trans', hue='Sex', palette='plasma')
plt.title("Weight Vs. Calories")
plt.ylabel('Calories')
plt.show()


plt.figure(figsize=(15,8))
sns.scatterplot(data, x='Body_Temp', y='Calories_trans', hue='Sex', palette='plasma')
plt.title("Body Temperature Vs. Calories")
plt.xlabel('Body Temperature')
plt.ylabel('Calories')
plt.show()


plt.figure(figsize=(15,8))
sns.scatterplot(data, x='Heart_Rate', y='Calories_trans', hue='Sex',  palette='plasma')
plt.title("Heat Rate Vs. Calories")
plt.xlabel('Heat Rate')
plt.ylabel('Calories')
plt.show()


plt.figure(figsize=(15,8))
sns.scatterplot(data, x='Duration', y='Calories_trans', hue='Sex',  palette='plasma')
plt.title("Duration Vs. Calories")
plt.xlabel('Duration')
plt.ylabel('Calories')
plt.show()


plt.figure(figsize=(15,8))
sns.scatterplot(data, x='Height', y='Calories_trans', hue='Sex',  palette='plasma')
plt.title("Height Vs. Calories")
plt.xlabel('Height')
plt.ylabel('Calories')
plt.show()


plt.figure(figsize=(15,8))
sns.scatterplot(data, x='Age', y='Calories_trans', hue='Sex',  palette='plasma')
plt.title("Age Vs. Calories")
plt.xlabel('Age')
plt.ylabel('Calories')
plt.show()


X = data.drop(['Calories','Calories_trans'],axis=1)
y = data['Calories_trans']
#print(X.shape,y.shape)

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train.shape,X_val.shape)


def cat_dummies(df,cols):
    "Function to convert categorical variables"
    for col in cols:
        dummies = pd.get_dummies(df[col],dtype=int,prefix=col)
        df = pd.concat([df,dummies],axis=1)
        df = df.drop(labels=col, axis=1)
    return df


X_train = cat_dummies(X_train,['Sex'])
X_val = cat_dummies(X_val,['Sex'])


data_copy = pd.concat([X_train,y_train],axis=1)
corr = data_copy.corr()

fig, ax = plt.subplots(figsize=(30, 20))
sns.heatmap(corr, cmap='coolwarm', annot=True, ax=ax)
ax.set_title('Correlation Matrix')
plt.show()


lr = LinearRegression()
model_lr=lr.fit(X_train, y_train)
y_lr_pred = model_lr.predict(X_val)
lr_mse = mean_squared_error(y_val, y_lr_pred)
lr_r2 = r2_score(y_val, y_lr_pred)

print(f"Mean Squared Error: {lr_mse}")
print(f"R2 Score: {lr_r2}")



RF = RandomForestRegressor()
model_rf = RF.fit(X_train,y_train)
y_rf_pred = model_rf.predict(X_val)

rf_mse = mean_squared_error(y_val, y_rf_pred)
rf_r2 = r2_score(y_val, y_rf_pred)
print(f"Mean Squared Error: {rf_mse}")
print(f"R2 Score: {rf_r2}")



plt.scatter(y_val, y_rf_pred,c=abs(y_val - y_rf_pred) ,cmap='coolwarm', label='Error Magnitude',alpha=0.5)
plt.plot(
    [y_val.min(), y_val.max()], 
    [y_val.min(), y_val.max()], 
    'k--', 
    lw=2, 
    label='Perfect Prediction'
)
plt.show()


test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test_df.head()


#Cleaning Test data
X_test = test_df.drop('id',axis=1)
X_test = cat_dummies(X_test,['Sex'])

submission_ids = test_df['id']

#Making Predictions
predictions = model_rf.predict(X_test)



#Create Submission Data
submission = pd.DataFrame({
    'id': submission_ids,
    'Calories': predictions 
})
submission.head()


submission.to_csv('submission.csv', index=False)
print("File Saved!")
print(submission.head())

