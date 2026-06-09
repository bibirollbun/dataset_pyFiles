import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score
)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt  
from catboost import CatBoostRegressor
import optuna
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train.head()


train.info()


cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


le =  LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.fit_transform(test['Sex'])


train['BMI'] = train['Weight'] / (train['Height'] / 100) ** 2
train['HR_per_min'] = train['Heart_Rate'] / train['Duration']
train['Temp_per_min'] = train['Body_Temp'] / train['Duration']
train['Effort'] = train['Heart_Rate'] * train['Body_Temp'] * train['Duration']
train['Age_Weight'] = train['Age'] * train['Weight']
train['Weight_per_height'] = train['Weight'] / train['Height']
train['log_Duration'] = np.log1p(train['Duration'])
train['log_HR'] = np.log1p(train['Heart_Rate'])


test['BMI'] = test['Weight'] / (test['Height'] / 100) ** 2
test['HR_per_min'] = test['Heart_Rate'] / test['Duration']
test['Temp_per_min'] = test['Body_Temp'] / test['Duration']
test['Effort'] = test['Heart_Rate'] * test['Body_Temp'] * test['Duration']
test['Age_Weight'] = test['Age'] * test['Weight']
test['Weight_per_height'] = test['Weight'] / test['Height']
test['log_Duration'] = np.log1p(test['Duration'])
test['log_HR'] = np.log1p(test['Heart_Rate'])


correlation = train.corr(numeric_only=True)['Calories'].sort_values(ascending=False)
print(correlation)


plt.figure(figsize=(10, 6))
sns.barplot(x=correlation.values, y=correlation.index, palette='viridis')
plt.title('Feature Correlation with Calories', fontsize=14)
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()


train.head()


X = train.drop(columns='Calories')
y = np.log1p(train["Calories"])  


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)
test_scaled = scaler.fit_transform(test)


model = LinearRegression()
model.fit(X_scaled, y)


test.head()


test_preds_log = model.predict(test_scaled)
test_preds = np.expm1(test_preds_log)  

# Prepare submission
submission['Calories'] = test_preds
submission.to_csv("submission.csv", index=False)

print(submission.head())

