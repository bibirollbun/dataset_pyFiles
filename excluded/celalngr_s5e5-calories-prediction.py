import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import make_scorer, mean_squared_log_error
from xgboost import XGBRegressor


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train.head()


train.shape


test.shape


train.describe()


train.isnull().sum().sum()


train.info()


numeric_list = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"]

melted_train = train[numeric_list].melt(var_name="Variable", value_name="Value")

plt.figure(figsize=(12, 6))
sns.boxplot(data=melted_train, x="Variable", y="Value", palette="Set2")
plt.title("Boxplot of Variables", fontsize=14)
plt.ylabel("Value Range")
plt.xlabel("")
plt.grid(axis='y')

plt.tight_layout()
plt.show()


numeric_list = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

melted_test = test[numeric_list].melt(var_name="Variable", value_name="Value")

plt.figure(figsize=(12, 6))
sns.boxplot(data=melted_test, x="Variable", y="Value", palette="Set2")
plt.title("Boxplot of Variables", fontsize=14)
plt.ylabel("Value Range")
plt.xlabel("")
plt.grid(axis='y')

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
corr_matrix = train[numeric_list].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heat Map of Numerical Variables")
plt.tight_layout()
plt.show()


def cap_outliers_iqr(train_df, test_df, columns):
    for col in columns:
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        train_df.loc[train_df[col] > upper_limit, col] = upper_limit
        train_df.loc[train_df[col] < lower_limit, col] = lower_limit

        test_df.loc[test_df[col] > upper_limit, col] = upper_limit
        test_df.loc[test_df[col] < lower_limit, col] = lower_limit

    return train_df, test_df


cols = ["Height", "Weight", "Heart_Rate", "Body_Temp"]
train, test = cap_outliers_iqr(train, test, cols)


Q3 = train["Calories"].quantile(0.75)
Q1 = train["Calories"].quantile(0.25)
IQR = Q3 - Q1

Upper_limit = Q3 + 1.5 * IQR


outlier_upper = train["Calories"] > Upper_limit


train.loc[outlier_upper, "Calories"] = Upper_limit


train["BMI"] = train["Weight"] / (train["Height"] / 100) ** 2
test["BMI"] = test["Weight"] / (test["Height"] / 100) ** 2

train["Effort_Score"] = train["Heart_Rate"] * train["Duration"]
test["Effort_Score"] = test["Heart_Rate"] * test["Duration"]

train["HR_per_min"] = train["Heart_Rate"] / train["Duration"]
test["HR_per_min"] = test["Heart_Rate"] / test["Duration"]

train["HeartRate_per_BodyTemp"] = train["Heart_Rate"] / train["Body_Temp"]
test["HeartRate_per_BodyTemp"] = test["Heart_Rate"] / test["Body_Temp"]

train["HR_age_ratio"] = train["Heart_Rate"] / train["Age"]
test["HR_age_ratio"] = test["Heart_Rate"] / test["Age"]


train["Weight_per_Height"] = train["Weight"] / train["Height"]
test["Weight_per_Height"] = test["Weight"] / test["Height"]

train["Age_BMI"] = train["Age"] * train["BMI"]
test["Age_BMI"] = test["Age"] * test["BMI"]

train["BodyTemp_Duration"] = train["Body_Temp"] * train["Duration"]
test["BodyTemp_Duration"] = test["Body_Temp"] * test["Duration"]

mean_body_temp_train = train["Body_Temp"].mean()
train["BodyTemp_Deviation"] = train["Body_Temp"] - mean_body_temp_train
mean_body_temp_test = test["Body_Temp"].mean()
test["BodyTemp_Deviation"] = test["Body_Temp"] - mean_body_temp_test

train["Max_Heart_Rate"] = 220 - train["Age"]
test["Max_Heart_Rate"] = 220 - test["Age"]
train["HeartRate_Max_HR_Ratio"] = train["Heart_Rate"] / train["Max_Heart_Rate"]
test["HeartRate_Max_HR_Ratio"] = test["Heart_Rate"] / test["Max_Heart_Rate"]

train["Effort_Score_per_Duration"] = train["Effort_Score"] / train["Duration"]
test["Effort_Score_per_Duration"] = test["Effort_Score"] / test["Duration"]

train["Weight_Age"] = train["Weight"] * train["Age"]
test["Weight_Age"] = test["Weight"] * test["Age"]

train["Weight_per_BMI"] = train["Weight"] / train["BMI"]
test["Weight_per_BMI"] = test["Weight"] / test["BMI"]

train["Exercise_Duration_Category"] = pd.cut(train["Duration"], bins=[0, 30, 60, 180], labels=["Short", "Medium", "Long"])
test["Exercise_Duration_Category"] = pd.cut(test["Duration"], bins=[0, 30, 60, 180], labels=["Short", "Medium", "Long"])

train["Age_Group"] = pd.cut(train["Age"], bins=[0, 30, 50, 100], labels=["Young", "Middle-Aged", "Old"])
test["Age_Group"] = pd.cut(test["Age"], bins=[0, 30, 50, 100], labels=["Young", "Middle-Aged", "Old"])


features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI', 'Effort_Score', 
            'HR_per_min', 'HeartRate_per_BodyTemp', 'HR_age_ratio', 'Weight_per_Height', 'Age_BMI', 
            'BodyTemp_Duration', 'BodyTemp_Deviation', 'HeartRate_Max_HR_Ratio', 'Effort_Score_per_Duration', 
            'Weight_Age', 'Weight_per_BMI']


scaler = StandardScaler()

train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])


label_cols = ["Exercise_Duration_Category", "Age_Group"]
le = LabelEncoder()

for col in label_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


train["Sex"] = train["Sex"].map({"male": 1, "female": 0})
test["Sex"] = test["Sex"].map({"male": 1, "female": 0})


train = pd.get_dummies(train, columns=["Sex", "Exercise_Duration_Category", "Age_Group"], drop_first=True)
test = pd.get_dummies(test, columns=["Sex", "Exercise_Duration_Category", "Age_Group"], drop_first=True)


bool_cols = train.select_dtypes("bool").columns

train[bool_cols] = train[bool_cols].astype(int)
test[bool_cols] = test[bool_cols].astype(int)


train.head()


X_train = train.drop(["id", "Calories"], axis=1)
y = train["Calories"]

X_test = test.drop(["id"],axis = 1)


xgb_model = XGBRegressor(subsample = 1.0, reg_lambda = 5, reg_alpha= 0.01, n_estimators= 1000, max_depth= 9,
learning_rate= 0.03, gamma= 0, colsample_bytree= 0.9)


xgb_model.fit(X_train,y)
y_pred = xgb_model.predict(X_test)


submission = pd.DataFrame({
    "id": test["id"],
    "Calories": y_pred 
})

submission.to_csv("submission_xgb_regression.csv", index=False)

