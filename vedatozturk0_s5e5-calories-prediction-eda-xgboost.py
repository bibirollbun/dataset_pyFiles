import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train_df.head()


train_df.info()


train_df.describe().T


num_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']


plt.figure(figsize=(18, 12))
for i, col in enumerate(num_cols):
    plt.subplot(3, 3, i+1)
    sns.histplot(train_df[col], kde=True, bins=40, color="skyblue")
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()


corr = train_df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.show()


sns.barplot(data=train_df, x='Sex', y='Calories', estimator=np.mean, ci='sd', palette='Set2')
plt.title("Average Calories by Gender")
plt.ylabel("Average Calories")
plt.show()


numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']


plt.figure(figsize=(18, 12))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x=train_df[col], color="lightcoral")
    plt.title(f'{col} Boxplot')
plt.tight_layout()
plt.show()


def detect_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    return outliers.shape[0]


for col in numeric_cols:
    n_outliers = detect_outliers(train_df, col)
    print(f"{col}: {n_outliers} outlier")


train_df['Sex'] = train_df['Sex'].map({'female': 0, 'male': 1})
test_df['Sex'] = test_df['Sex'].map({'female': 0, 'male': 1})

train_df['BMI'] = train_df['Weight'] / ((train_df['Height']/100) ** 2)
test_df['BMI'] = test_df['Weight'] / ((test_df['Height']/100) ** 2)

train_df['HR_per_min'] = train_df['Heart_Rate'] / train_df['Duration']
test_df['HR_per_min'] = test_df['Heart_Rate'] / test_df['Duration']

train_df['Temp_Deviation'] = train_df['Body_Temp'] - 37.0
test_df['Temp_Deviation'] = test_df['Body_Temp'] - 37.0

train_df['HRxDuration'] = train_df['Heart_Rate'] * train_df['Duration']
test_df['HRxDuration'] = test_df['Heart_Rate'] * test_df['Duration']



train_df['Metabolic_Estimate'] = train_df['Weight'] * train_df['Duration'] * train_df['Heart_Rate']
test_df['Metabolic_Estimate'] = test_df['Weight'] * test_df['Duration'] * test_df['Heart_Rate']

train_df['Proxy_Calories'] = train_df['Weight'] * train_df['Duration'] * train_df['Body_Temp'] / 1000
test_df['Proxy_Calories'] = test_df['Weight'] * test_df['Duration'] * test_df['Body_Temp'] / 1000

train_df['z_Body_Temp'] = (train_df['Body_Temp'] - train_df['Body_Temp'].mean()) / train_df['Body_Temp'].std()
test_df['z_Body_Temp'] = (test_df['Body_Temp'] - train_df['Body_Temp'].mean()) / train_df['Body_Temp'].std()

train_df['z_Heart_Rate'] = (train_df['Heart_Rate'] - train_df['Heart_Rate'].mean()) / train_df['Heart_Rate'].std()
test_df['z_Heart_Rate'] = (test_df['Heart_Rate'] - train_df['Heart_Rate'].mean()) / train_df['Heart_Rate'].std()


train_df['Duration_per_kg'] = train_df['Duration'] / train_df['Weight']
test_df['Duration_per_kg'] = test_df['Duration'] / test_df['Weight']

train_df['Age_HeartRate'] = train_df['Age'] * train_df['Heart_Rate']
test_df['Age_HeartRate'] = test_df['Age'] * test_df['Heart_Rate']

train_df['Weight_HeartRate'] = train_df['Weight'] * train_df['Heart_Rate']
test_df['Weight_HeartRate'] = test_df['Weight'] * test_df['Heart_Rate']


train_df['HR_per_kg'] = train_df['Heart_Rate'] / train_df['Weight']
test_df['HR_per_kg'] = test_df['Heart_Rate'] / test_df['Weight']

train_df['BMIxHR'] = train_df['BMI'] * train_df['Heart_Rate']
test_df['BMIxHR'] = test_df['BMI'] * test_df['Heart_Rate']

train_df['HR_temp_adj'] = train_df['Heart_Rate'] / (train_df['Body_Temp'] + 0.1)
test_df['HR_temp_adj'] = test_df['Heart_Rate'] / (test_df['Body_Temp'] + 0.1)


train_df['HR_per_age'] = train_df['Heart_Rate'] / (train_df['Age'] + 1)
test_df['HR_per_age'] = test_df['Heart_Rate'] / (test_df['Age'] + 1)




mean_body_temp = train_df['Body_Temp'].mean()
train_df['BodyTemp_Deviation'] = train_df['Body_Temp'] - mean_body_temp
test_df['BodyTemp_Deviation'] = test_df['Body_Temp'] - mean_body_temp

train_df['Max_Heart_Rate'] = 220 - train_df['Age']
test_df['Max_Heart_Rate'] = 220 - test_df['Age']

train_df['HeartRate_Max_HR_Ratio'] = train_df['Heart_Rate'] / train_df['Max_Heart_Rate']
test_df['HeartRate_Max_HR_Ratio'] = test_df['Heart_Rate'] / test_df['Max_Heart_Rate']

train_df['Effort_Score_per_Duration'] = train_df['HRxDuration'] / train_df["Duration"]
test_df['Effort_Score_per_Duration'] = test_df['HRxDuration'] / test_df["Duration"]

train_df['Weight_Age'] = train_df['Weight'] * train_df['Age']
test_df['Weight_Age'] = test_df['Weight'] * test_df['Age']

train_df['Weight_per_BMI'] = train_df['Weight'] / train_df['BMI']
test_df['Weight_per_BMI'] = test_df['Weight'] / test_df['BMI']



train_df['Caloric_Index'] = (train_df['Heart_Rate'] * train_df['Duration']) / (train_df['Weight'] + 1)
test_df['Caloric_Index'] = (test_df['Heart_Rate'] * test_df['Duration']) / (test_df['Weight'] + 1)

train_df['BMI_Adjusted_HR'] = train_df['Heart_Rate'] / (train_df['BMI'] + 0.1)
test_df['BMI_Adjusted_HR'] = test_df['Heart_Rate'] / (test_df['BMI'] + 0.1)

train_df['Age_Weighted_Effort'] = (train_df['Heart_Rate'] * train_df['Duration']) / (train_df['Age'] + 1)
test_df['Age_Weighted_Effort'] = (test_df['Heart_Rate'] * test_df['Duration']) / (test_df['Age'] + 1)

train_df['Temp_Adjusted_Calories'] = train_df['Body_Temp'] * train_df['Heart_Rate'] * train_df['Duration'] / 1000
test_df['Temp_Adjusted_Calories'] = test_df['Body_Temp'] * test_df['Heart_Rate'] * test_df['Duration'] / 1000


train_df["Exercise_Duration_Category"] = pd.cut(train_df["Duration"], bins=[0, 30, 60, 180], labels=["Short", "Medium", "Long"])
test_df["Exercise_Duration_Category"] = pd.cut(test_df["Duration"], bins=[0, 30, 60, 180], labels=["Short", "Medium", "Long"])

train_df["Age_Group"] = pd.cut(train_df["Age"], bins=[0, 30, 50, 100], labels=["Young", "Middle-Aged", "Old"])
test_df["Age_Group"] = pd.cut(test_df["Age"], bins=[0, 30, 50, 100], labels=["Young", "Middle-Aged", "Old"])


scaler_features = [
    'BMI', 'HR_per_min', 'Temp_Deviation', 'HRxDuration', 'Duration_per_kg',
    'Age_HeartRate', 'Weight_HeartRate', 'HR_per_kg',
    'BMIxHR', 'HR_temp_adj', 'HR_per_age','BodyTemp_Deviation',
    'Max_Heart_Rate','HeartRate_Max_HR_Ratio','Effort_Score_per_Duration',
    'Weight_Age','Weight_per_BMI',
    'Age', 'Height', 'Weight', 'Duration',
    'Caloric_Index','BMI_Adjusted_HR','Age_Weighted_Effort','Temp_Adjusted_Calories',
    'Metabolic_Estimate','Proxy_Calories','z_Body_Temp','z_Heart_Rate'
]


scaler = StandardScaler()
train_df[scaler_features] = scaler.fit_transform(train_df[scaler_features])
test_df[scaler_features] = scaler.transform(test_df[scaler_features])


le_duration = LabelEncoder()
train_df["Exercise_Duration_Category_Encoded"] = le_duration.fit_transform(train_df["Exercise_Duration_Category"])
test_df["Exercise_Duration_Category_Encoded"] = le_duration.transform(test_df["Exercise_Duration_Category"])

le_age = LabelEncoder()
train_df["Age_Group_Encoded"] = le_age.fit_transform(train_df["Age_Group"])
test_df["Age_Group_Encoded"] = le_age.transform(test_df["Age_Group"])


train_df = pd.get_dummies(train_df, columns=["Exercise_Duration_Category"], drop_first=True)
test_df = pd.get_dummies(test_df, columns=["Exercise_Duration_Category"], drop_first=True)

train_df = pd.get_dummies(train_df, columns=["Age_Group"], drop_first=True)
test_df = pd.get_dummies(test_df, columns=["Age_Group"], drop_first=True)


bool_cols = train_df.select_dtypes("bool").columns

train_df[bool_cols] = train_df[bool_cols].astype(int)
test_df[bool_cols] = test_df[bool_cols].astype(int)


X = train_df.drop(['Calories', 'Calories_per_min'], axis=1, errors='ignore')
y = train_df['Calories']
X_test = test_df.copy()


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=9,
    subsample=0.9,
    colsample_bytree=0.7,
    eval_metric = 'rmse',
    gamma = 0.01,
    device='cuda',
    random_state=42
)

xgb_model.fit(X_train, y_train)


test_preds_xgb = xgb_model.predict(X_test)


submission = pd.DataFrame({
    "id": test_df["id"],
    "Calories": test_preds_xgb
})

submission.to_csv("submission.csv", index=False)

