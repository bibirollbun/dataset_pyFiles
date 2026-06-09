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


import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
display(train.head())


train.info()


train.describe()


counts = train['Sex'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Ratio')
plt.show()


plt.figure(figsize=(8,6))
sns.barplot(x='Sex', y='Calories', data=train, palette=['skyblue', 'lightpink'])
plt.title('Gender/Calories')
plt.ylabel('AVG')
plt.show()



numeric_columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']

train[numeric_columns].hist(figsize=(14, 10), bins=30, edgecolor='black')
plt.suptitle('Distribution of Numeric Features', fontsize=16)
plt.tight_layout()
plt.show()



grouped_means = train.groupby('Sex')[numeric_columns].mean().T

grouped_means.plot(kind='bar', figsize=(12, 6), colormap='Set2')
plt.title('Average by Gender')
plt.ylabel('Average Value')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()



correlation_matrix = train[numeric_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

train['Sex'] = train['Sex'].str.strip().str.lower()          
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1}) 

print("결측치 개수:", train['Sex'].isnull().sum())      # 0
print("고유값:", train['Sex'].unique())                  # [0. 1.]

features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
target = 'Calories'

X = train[features]
y = train[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"평균 제곱 오차 : {mse:.2f}")
print(f"결정 계수 : {r2:.4f}")


pip install xgboost


import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
target = 'Calories'

X = train[features]
y = train[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=100,              
    learning_rate=0.1,           
    max_depth=6,                  
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))



importance = model.feature_importances_
feature_names = features

sorted_idx = importance.argsort()

plt.figure(figsize=(8,6))
plt.barh([feature_names[i] for i in sorted_idx], importance[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.show()


test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

test['Sex'] = test['Sex'].str.strip().str.lower()
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

test_features = test[features]
test_features_scaled = scaler.transform(test_features)

predictions = model.predict(test_features_scaled)
predictions = np.maximum(predictions, 0)

sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
sample_submission['Calories'] = predictions
sample_submission.to_csv('submission.csv', index=False)





