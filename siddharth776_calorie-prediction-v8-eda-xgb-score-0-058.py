import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import make_scorer, mean_squared_log_error
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
from sklearn.linear_model import Ridge, Lasso

from warnings import filterwarnings
filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train.head()


test.head()


print(train.shape)
print(test.shape)


train.describe()


train.isnull().sum()


train.info()


test.isnull().sum()


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


sns.pairplot(train[numeric_list], diag_kind='kde', corner=True, plot_kws={'alpha': 0.6})
plt.suptitle("Pairwise Relationships Between Features", y=1.02)
plt.show()




plt.figure(figsize=(10, 6))
corr = train[numeric_list].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=14)
plt.show()



# Add 'Calories' to numeric_list for correlation
numeric_list_with_target = numeric_list + ["Calories"]

# Calculate correlation with 'Calories'
calorie_corr = train[numeric_list_with_target].corr()['Calories'].drop('Calories').sort_values()

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=calorie_corr.values, y=calorie_corr.index, palette='viridis')
plt.title("Feature Correlation with Calories", fontsize=14)
plt.xlabel("Correlation Coefficient")
plt.grid(axis='x')
plt.tight_layout()
plt.show()




train[numeric_list].corr()



for col in numeric_list[:-1]:  # Skip "Calories" itself
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=train[col], y=train["Calories"], alpha=0.6)
    plt.title(f"Calories vs {col}", fontsize=13)
    plt.xlabel(col)
    plt.ylabel("Calories")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



sns.scatterplot(data=train, x="Duration", y="Calories", hue="Sex", palette="Set1", alpha=0.6)
plt.title("Calories vs Duration by Gender")
plt.grid(True)
plt.show()



sns.scatterplot(data=train, x="Age", y="Calories", hue="Sex", palette="Set1", alpha=0.6)
plt.title("Calories vs Age by Gender")
plt.grid(True)
plt.show()


sns.scatterplot(data=train, x="Weight", y="Calories", hue="Sex", palette="Set1", alpha=0.6)
plt.title("Calories vs Weight by Gender")
plt.grid(True)
plt.show()


sns.scatterplot(data=train, x="Height", y="Calories", hue="Sex", palette="Set1", alpha=0.6)
plt.title("Calories vs Height by Gender")
plt.grid(True)
plt.show()


sns.scatterplot(data=train, x="Heart_Rate", y="Calories", hue="Sex", palette="Set1", alpha=0.6)
plt.title("Calories vs Heart Rate by Gender")
plt.grid(True)
plt.show()


sns.scatterplot(data=train, x="Body_Temp", y="Calories", hue="Sex", palette="Set1", alpha=0.6)
plt.title("Calories vs Body Temperature by Gender")
plt.grid(True)
plt.show()


# --- Basic Health Metrics ---
train["BMI"] = train["Weight"] / (train["Height"] / 100) ** 2
test["BMI"] = test["Weight"] / (test["Height"] / 100) ** 2

train["Weight_per_Height"] = train["Weight"] / train["Height"]
test["Weight_per_Height"] = test["Weight"] / test["Height"]

train["Age_BMI"] = train["Age"] * train["BMI"]
test["Age_BMI"] = test["Age"] * test["BMI"]

# --- Heart-related Metrics ---
train["Max_Heart_Rate"] = 220 - train["Age"]
test["Max_Heart_Rate"] = 220 - test["Age"]

train["HeartRate_Max_HR_Ratio"] = train["Heart_Rate"] / train["Max_Heart_Rate"]
test["HeartRate_Max_HR_Ratio"] = test["Heart_Rate"] / test["Max_Heart_Rate"]

train["HR_per_min"] = train["Heart_Rate"] / train["Duration"]
test["HR_per_min"] = test["Heart_Rate"] / test["Duration"]

train["HeartRate_per_BodyTemp"] = train["Heart_Rate"] / train["Body_Temp"]
test["HeartRate_per_BodyTemp"] = test["Heart_Rate"] / test["Body_Temp"]

train["HR_age_ratio"] = train["Heart_Rate"] / train["Age"]
test["HR_age_ratio"] = test["Heart_Rate"] / test["Age"]

# --- Effort-Based Features ---
train["Effort_Score"] = train["Heart_Rate"] * train["Duration"]
test["Effort_Score"] = test["Heart_Rate"] * test["Duration"]

train["Duration_Heart_BodyTemp"] = train["Duration"] * train["Heart_Rate"] * train["Body_Temp"]
test["Duration_Heart_BodyTemp"] = test["Duration"] * test["Heart_Rate"] * test["Body_Temp"]

train["Weight_Duration_HeartRate"] = train["Weight"] * train["Duration"] * train["Heart_Rate"]
test["Weight_Duration_HeartRate"] = test["Weight"] * test["Duration"] * test["Heart_Rate"]

# --- Temperature-Related Features ---
mean_body_temp_train = train["Body_Temp"].mean()
train["BodyTemp_Deviation"] = train["Body_Temp"] - mean_body_temp_train

mean_body_temp_test = test["Body_Temp"].mean()
test["BodyTemp_Deviation"] = test["Body_Temp"] - mean_body_temp_test

train["BodyTemp_Duration"] = train["Body_Temp"] * train["Duration"]
test["BodyTemp_Duration"] = test["Body_Temp"] * test["Duration"]

# --- Categorical Bucketing ---
train["Exercise_Duration_Category"] = pd.cut(train["Duration"], bins=[0, 30, 60, 180], labels=["Short", "Medium", "Long"])
test["Exercise_Duration_Category"] = pd.cut(test["Duration"], bins=[0, 30, 60, 180], labels=["Short", "Medium", "Long"])

train["Age_Group"] = pd.cut(train["Age"], bins=[0, 30, 50, 100], labels=["Young", "Middle-Aged", "Old"])
test["Age_Group"] = pd.cut(test["Age"], bins=[0, 30, 50, 100], labels=["Young", "Middle-Aged", "Old"])




# --- Label Encoding for Binned Categories ---
from sklearn.preprocessing import LabelEncoder

for col in ["Exercise_Duration_Category", "Age_Group"]:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])



# Total physical exertion combining duration, weight, and HR
train["Total_Workload"] = train["Weight"] * train["Heart_Rate"] * train["Duration"]
test["Total_Workload"] = test["Weight"] * test["Heart_Rate"] * test["Duration"]

# Normalized Effort Score per BMI
train["Effort_per_BMI"] = train["Effort_Score"] / train["BMI"]
test["Effort_per_BMI"] = test["Effort_Score"] / test["BMI"]



# Ensure no duplicate 'Calories' column
features = [col for col in train.columns if train[col].dtype != 'object' and col not in ['id', 'Calories']]
corr_matrix = train[features + ['Calories']].corr()

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix[['Calories']].drop_duplicates().sort_values(by='Calories', ascending=False), 
            annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation of Features with Calories", fontsize=16)
plt.show()



from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define categorical and numerical columns
categorical_cols = ["Sex", "Exercise_Duration_Category", "Age_Group"]
numerical_cols = [col for col in train.columns if col not in ['id', 'Calories'] + categorical_cols]

# Label encode categorical features and scale numerical features
scaler = StandardScaler()
label_encoders = {col: LabelEncoder().fit(train[col]) for col in categorical_cols}

# Apply transformations
train[categorical_cols] = train[categorical_cols].apply(lambda col: label_encoders[col.name].transform(col))
test[categorical_cols] = test[categorical_cols].apply(lambda col: label_encoders[col.name].transform(col))

train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])

# Combine and prepare final dataset
train_scaled = train.copy()
test_scaled = test.copy()

# Preview
print(train_scaled.head())



print("Train Columns:", train.columns.tolist())
print("Test Columns:", test.columns.tolist())



train.head()


import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Step 1: Prepare Data
X_train = train.drop(["id", "Calories"], axis=1)  # Drop 'id' and 'Calories'
y_train = train["Calories"]  # Target variable

X_test = test.drop(["id"], axis=1)  # Drop 'id' from test set

# Step 2: Train Model using XGBoost Regressor
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=7)
xgb_model = XGBRegressor(subsample = 0.8252835138864835, reg_lambda = 3.501605991491236, reg_alpha= 9.019718559862135, n_estimators= 494, max_depth= 12,
learning_rate= 0.025720450946032463, gamma= 1.8903496577328514, colsample_bytree= 0.5789286073361388,min_child_weight =  7)
# Fit model
xgb_model.fit(X_train, y_train)

# Step 3: Make Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Step 4: Create the Submission File
submission = pd.DataFrame({
    "id": test["id"],
    "Calories": y_pred_xgb
})

# Step 5: Save submission to CSV
submission.to_csv("submission.csv", index=False)

# Optionally, evaluate model performance on training set
y_train_pred = xgb_model.predict(X_train)
mae = mean_absolute_error(y_train, y_train_pred)
print(f'Mean Absolute Error on Training Set: {mae}')





