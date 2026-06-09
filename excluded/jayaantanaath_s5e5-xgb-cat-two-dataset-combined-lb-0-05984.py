import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv('../input/playground-series-s5e5/test.csv')


output_df = pd.read_csv('/kaggle/input/fmendesdat263xdemos/calories.csv')
input_df = pd.read_csv('/kaggle/input/fmendesdat263xdemos/exercise.csv')


input_df.tail()


output_df.tail()


calory_df = pd.concat([input_df, output_df], axis=1)


calory_df.drop('User_ID', axis=1, inplace = True)


calory_df['Gender'] = calory_df['Gender'].map({'male':0, 'female':1})

X = calory_df.drop(['Calories'], axis= 1)
y = calory_df['Calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the XGBoost model

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.6, 
    colsample_bytree=0.8, 
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

print('R² score:',xgb_model.score(X_test, y_test))

y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

mse = mean_squared_error(y_train, y_pred_train, squared=False)
print(f"TRain MSE: {mse}")

mse = mean_squared_error(y_test, y_pred_test, squared=False)
print(f"Val MSE: {mse}")

xgb.plot_importance(xgb_model)
plt.show()


print('Shape:',train_df.shape, test_df.shape)
print('train\n',train_df.isna().sum())
print('test\n',test_df.isna().sum())


sex_counts = train_df["Sex"].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribution of Sex")
plt.axis("equal")
plt.show()


train_df.sample()


calory_df.sample()


# Step 1: Drop 'id' from train_df
train_df = train_df.drop(columns=['id'])

# Step 2: Rename calory_df columns to match the updated train_df
calory_df.columns = train_df.columns

# Step 3: Concatenate both DataFrames
combined_df = pd.concat([train_df, calory_df], ignore_index=True)

# Step 4: Shuffle the combined DataFrame
shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)


train_df.shape, calory_df.shape, shuffled_df.shape


shuffled_df.describe()


train_df = shuffled_df.copy()


train_df['BMI'] = train_df['Weight'] / ((train_df['Height'] / 100) ** 2)
test_df['BMI'] = test_df['Weight'] / ((test_df['Height'] / 100) ** 2)

features = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp","Sex","Temp_Binary","HeartRate_binary"]
target = "Calories"

numerical_features = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp","Temp_Binary","HeartRate_binary"]
categorical_features = ["Sex"]

train_df['Cardio_Load'] = train_df['Heart_Rate'] * train_df['Duration']
test_df['Cardio_Load'] = test_df['Heart_Rate'] * test_df['Duration']

train_df["Sex"] = train_df["Sex"].map({"female":0, "male":1})
test_df["Sex"] = test_df["Sex"].map({"female":0, "male":1})

train_df['Temp_Binary'] = np.where(train_df['Body_Temp'] <= 39.5, 0, 1)
test_df['Temp_Binary'] = np.where(test_df['Body_Temp'] <= 39.5, 0, 1)

train_df['HeartRate_binary'] = np.where(train_df['Heart_Rate'] <= 99.5, 0, 1)
test_df['HeartRate_binary'] = np.where(test_df['Heart_Rate'] <= 99.5, 0, 1)


body_temp_mean = train_df["Body_Temp"].mean()
body_temp_mean


# Compute Body_Temp mean from train_df only
body_temp_mean = train_df["Body_Temp"].mean()

# Feature engineering for both train and test datasets
for df in [train_df, test_df]:
    df["Body_Temp_Curvature"] = (df["Body_Temp"] - body_temp_mean) ** 2
    # df["Age_Group"] = pd.cut(df["Age"], bins=[0, 30, 50, 70, 100], labels=["young", "middle", "senior", "elder"])


for df in [train_df, test_df]:
    df["HR_Percentage"] = df["Heart_Rate"] / (220 - df["Age"]) * 100


corr = train_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, linewidth = 0.5, annot= True)


train_df.sample()


X = train_df.drop(['Calories','BMI','Temp_Binary','HeartRate_binary'], axis= 1)
y = train_df['Calories']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# Creating the XGBoost model

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.6, 
    colsample_bytree=0.8, 
    random_state=42,
    tree_method='gpu_hist'
)

# Train the model
xgb_model.fit(X_train, y_train)

print('R² score:',xgb_model.score(X_test, y_test))

y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

mse = mean_squared_error(y_train, y_pred_train, squared=False)
print(f"Train MSE: {mse}")

mse = mean_squared_error(y_test, y_pred_test, squared=False)
print(f"Val MSE: {mse}")

xgb.plot_importance(xgb_model)
plt.show()


from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Creating the CatBoost model
cat_model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.01,
    depth=10,
    subsample=0.6,
    colsample_bylevel=0.8,
    random_seed=42,
    verbose=0  # Set to 100 if you want iteration logs
)

# Train the model
cat_model.fit(X_train, y_train)

print('R² score:', cat_model.score(X_test, y_test))

y_pred_train = cat_model.predict(X_train)
y_pred_test = cat_model.predict(X_test)

mse = mean_squared_error(y_train, y_pred_train, squared=False)
print(f"Train MSE: {mse}")

mse = mean_squared_error(y_test, y_pred_test, squared=False)
print(f"Val MSE: {mse}")


# Plot feature importance
importances = cat_model.get_feature_importance()
plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances)
plt.yticks(range(len(importances)), X_train.columns)
plt.xlabel("Feature Importance")
plt.title("CatBoost Feature Importance")
plt.tight_layout()
plt.show()


X.sample()


test_final = test_df[['Sex','Age','Height','Weight','Duration',	'Heart_Rate','Body_Temp','Cardio_Load','Body_Temp_Curvature','HR_Percentage']]
test_final.shape

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.6, 
    colsample_bytree=0.8, 
    random_state=42,
    tree_method='gpu_hist'
)

# Train the model
xgb_model.fit(X, y)

# Predicting
predictions = xgb_model.predict(test_final)

# Clipping predictions between 1.0 and 314.0
predictions = predictions.clip(1.0, 314.0)

# Assigning to test_df
test_df["Calories_xgb"] = predictions

# Creating the CatBoost model
cat_model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.01,
    depth=10,
    subsample=0.6,
    colsample_bylevel=0.8,
    random_seed=42,
    verbose=0  # Set to 100 if you want iteration logs
)

# Train the model
cat_model.fit(X, y)

# Predicting
predictions = cat_model.predict(test_final)

# Clipping predictions between 1.0 and 314.0
predictions = predictions.clip(1.0, 314.0)

# Assigning to test_df
test_df["Calories_cat"] = predictions


test_df


test_df['Calories'] = test_df[['Calories_xgb', 'Calories_cat']].mean(axis=1)


test_df


submission = test_df[["id", "Calories"]]

submission.to_csv("submission.csv", index=False)

