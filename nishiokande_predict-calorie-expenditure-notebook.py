import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error, mean_squared_error


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train_df.head()


test_df.head()


train_df.describe()


train_df["Sex"] = train_df["Sex"].map({"male":0, "female":1})
test_df["Sex"] = test_df["Sex"].map({"male":0, "female":1})


corr_matrix = train_df[["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"]].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=train_df[["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"]])
plt.xticks(rotation=45)
plt.title("Boxplot of Numeric Features")
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=test_df[["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]])
plt.xticks(rotation=45)
plt.title("Boxplot of Numeric Features")
plt.show()


train_df[["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"]].hist(figsize=(15, 10), bins=50, color='skyblue', edgecolor='black')

plt.suptitle('Histograms of Numeric Features', fontsize=16)
plt.tight_layout()
plt.show()


test_df[["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]].hist(figsize=(15, 10), bins=50, color='skyblue', edgecolor='black')

plt.suptitle('Histograms of Numeric Features', fontsize=16)
plt.tight_layout()
plt.show()


# BMI
train_df["BMI"] = train_df["Weight"] / (train_df["Height"] / 100) ** 2
test_df["BMI"] = test_df["Weight"] / (test_df["Height"] / 100) ** 2

# Duration per age
train_df["Duration_per_age"] = train_df["Duration"] / train_df["Age"]
test_df["Duration_per_age"] = test_df["Duration"] / test_df["Age"]

# Heart rate per duration (efficiency)
train_df["HeartRate_per_duration"] = train_df["Heart_Rate"] / train_df["Duration"]
test_df["HeartRate_per_duration"] = test_df["Heart_Rate"] / test_df["Duration"]

train_df['Weight_per_Duration'] = train_df['Weight'] / train_df['Duration']
test_df['Weight_per_Duration'] = test_df['Weight'] / test_df['Duration']

train_df['High_Temp_Flag'] = (train_df['Body_Temp'] > 40).astype(int)
test_df['High_Temp_Flag'] = (test_df['Body_Temp'] > 40).astype(int)

train_df['Body_Temp_Delta'] = train_df['Body_Temp'] - 39.5
test_df['Body_Temp_Delta'] = test_df['Body_Temp'] - 39.5

train_df['Age_Group_Num'] = pd.cut(train_df['Age'], bins=[0, 29, 39, 49, 59, 69, 100], labels=[0, 1, 2, 3, 4, 5]).astype(int)
test_df['Age_Group_Num'] = pd.cut(test_df['Age'], bins=[0, 29, 39, 49, 59, 69, 100], labels=[0, 1, 2, 3, 4, 5]).astype(int)


train_df.head()


train_df["Body_Temp_log"] = np.log1p(train_df["Body_Temp"])
test_df["Body_Temp_log"] = np.log1p(test_df["Body_Temp"])

train_df["HeartRate_per_duration_log"] = np.log1p(train_df["HeartRate_per_duration"])
test_df["HeartRate_per_duration_log"] = np.log1p(test_df["HeartRate_per_duration"])

train_df = train_df.drop(["Body_Temp", "HeartRate_per_duration"], axis=1)
test_df = test_df.drop(["Body_Temp", "HeartRate_per_duration"], axis=1)


train_numeric_cols = train_df.drop(["id", "Sex", "Calories"], axis=1)
train_numeric_cols.hist(figsize=(15, 10), bins=50, color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numeric Features', fontsize=16)
plt.tight_layout()
plt.show()


test_numeric_cols = test_df.drop(["id", "Sex"], axis=1)
test_numeric_cols.hist(figsize=(15, 10), bins=50, color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numeric Features', fontsize=16)
plt.tight_layout()
plt.show()


X = train_df.drop(["id", "Calories"], axis=1)
y = train_df["Calories"]
X_test = test_df.drop(["id"], axis = 1)

# Split data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Create an XGBoost regression model
model = XGBRegressor(
    n_estimators=840,
    max_depth=11,
    learning_rate=0.012509151627202115,
    subsample=0.6390852422092699,
    colsample_bytree=0.7334890825815552,
    gamma=0.14961226042180592,
    min_child_weight=4,
    reg_alpha=0.5841600273634906,
    reg_lambda=0.5838040743264866,
    random_state=42
)

# âœ… log1p transform the target (safe for RMSLE)
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

# Train the model on log-transformed target
model.fit(X_train, y_train_log)

# Predict (log space)
y_pred_log = model.predict(X_val)

# âœ… expm1 to convert back to original space
y_pred = np.expm1(y_pred_log)

# Evaluate the model using RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
print(f"Validation RMSLE: {rmsle:.4f}")


X_test = test_df.drop(["id"], axis = 1)

test_pred_log = model.predict(X_test)
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame({"id": test_df["id"], "Calories": test_pred})

submission.to_csv("submission.csv", index=False)


plt.hist(submission['Calories'],bins=100)
plt.title("Test Preds")
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=submission[["Calories"]])
plt.xticks(rotation=45)
plt.title("Boxplot of TestPred")
plt.show()

