import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv', index_col='id')


# View the first few rows
print(train.head())
print("____________________________________________________")
print("____________________________________________________")
# Basic info (data types, non-null counts)
print(train.info())


print("NaNs in train")
print(train.isnull().sum())
print("________________", end='\n\n')
print("NaNs in test")
print(test.isnull().sum())


# Number of unique values
print(train.nunique())
print("________________", end='\n\n')
print(train['Sex'].value_counts())


train['Height_m'] = train['Height'] / 100
train['BMI'] = train['Weight'] / (train['Height_m'] ** 2)

test['Height_m'] = test['Height'] / 100
test['BMI'] = test['Weight'] / (test['Height_m'] ** 2)


train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})
test['Sex'] = test['Sex'].map({'female': 0, 'male': 1})


# Feature Engineering on Training Set
train['Intensity'] = train['Heart_Rate'] / train['Duration']
train['HR_Temp'] = train['Heart_Rate'] * train['Body_Temp']
train['Dur_Temp'] = train['Duration'] * train['Body_Temp']

# Apply the same to Test Set
test['Intensity'] = test['Heart_Rate'] / test['Duration']
test['HR_Temp'] = test['Heart_Rate'] * test['Body_Temp']
test['Dur_Temp'] = test['Duration'] * test['Body_Temp']

# Use the same feature list
features = ['Duration', 'Heart_Rate', 'Body_Temp', 'Age', 'Sex',
            'BMI', 'Intensity', 'HR_Temp', 'Dur_Temp','Height_m','Weight','Calories']

train = train[features]


test_features = ['Duration', 'Heart_Rate', 'Body_Temp', 'Age', 'Sex',
            'BMI', 'Intensity', 'HR_Temp', 'Dur_Temp','Height_m','Weight']
X_test = test[test_features].copy()


train['Duration_log'] = np.log1p(train['Duration'])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train[features])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train[features])
from sklearn.preprocessing import OneHotEncoder
train['Duration'].fillna(train['Duration'].median(), inplace=True)
X_test['Duration'].fillna(X_test['Duration'].median(), inplace=True)
X_test['Duration_log'] = np.log1p(X_test['Duration'])



print(train)


train['Log_Calories'] = np.log1p(train['Calories'])
X = train.drop(columns=['Calories', 'Log_Calories'])
Y = train['Log_Calories']


corr = train.corr()
# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr[['Calories']].sort_values(by='Calories', ascending=False), annot=True, cmap='coolwarm')
plt.title('Feature Correlation with Calories')
plt.show()


sns.scatterplot(data=train, x='Duration', y='Calories', hue='Sex')
plt.title('Calories vs Duration')
plt.show()


sns.scatterplot(data=train, x='Heart_Rate', y='Calories', hue='Sex')
plt.title('Calories vs Heart Rate by Sex')
plt.show()


sns.scatterplot(data=train, x='Body_Temp', y='Calories', hue='Sex', alpha=0.6)
plt.title('Calories vs Body Temperature by Sex')
plt.show()


num_features = ['Age', 'Height_m', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI', 'Calories']
plt.figure(figsize=(15, 15))
for i, feature in enumerate(num_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(train[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()


sns.boxplot(data=train, x='Sex', y='Calories')
plt.title('Calories Burned by Sex')
plt.show()


X


model = xgb.XGBRegressor(
        max_depth=9,
        colsample_bytree=0.9,
        subsample=0.9,
        n_estimators=3500,
        learning_rate=0.00821,
        random_state=0,
        eval_metric="rmse"
    )
model.fit(X, Y, verbose=10)


y_pred_log = model.predict(X)

# Calculate RMSLE
from sklearn.metrics import mean_squared_log_error
import numpy as np

rmsle = np.sqrt(mean_squared_log_error(np.expm1(Y), np.expm1(y_pred_log)))
print(f'RMSLE: {rmsle:.4f}')


preds = model.predict(X_test)


preds = np.expm1(preds)


sub['Calories'] = preds


sub.to_csv('submission.csv')


sub['Calories'].hist()


sub




