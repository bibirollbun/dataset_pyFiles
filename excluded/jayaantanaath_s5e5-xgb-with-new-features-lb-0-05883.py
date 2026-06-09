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


print('Shape:',train_df.shape, test_df.shape)
print('train\n',train_df.isna().sum())
print('test\n',test_df.isna().sum())


sex_counts = train_df["Sex"].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribution of Sex")
plt.axis("equal")
plt.show()


train_df['BMI'] = train_df['Weight'] / ((train_df['Height'] / 100) ** 2)
test_df['BMI'] = test_df['Weight'] / ((test_df['Height'] / 100) ** 2)

train_df['Cardio_Load'] = train_df['Heart_Rate'] * train_df['Duration']
test_df['Cardio_Load'] = test_df['Heart_Rate'] * test_df['Duration']

train_df["Sex"] = train_df["Sex"].map({"female":0, "male":1})
test_df["Sex"] = test_df["Sex"].map({"female":0, "male":1})

train_df['Temp_Binary'] = np.where(train_df['Body_Temp'] <= 39.5, 0, 1)
test_df['Temp_Binary'] = np.where(test_df['Body_Temp'] <= 39.5, 0, 1)

train_df['HeartRate_binary'] = np.where(train_df['Heart_Rate'] <= 99.5, 0, 1)
test_df['HeartRate_binary'] = np.where(test_df['Heart_Rate'] <= 99.5, 0, 1)


# Compute Body_Temp mean from train_df only
body_temp_mean = train_df["Body_Temp"].mean()


# Feature engineering for both train and test datasets
for df in [train_df, test_df]:
    df["Body_Temp_Curvature"] = (df["Body_Temp"] - body_temp_mean) ** 2
    df["Age_Group"] = pd.cut(df["Age"], bins=[0, 30, 50, 70, 100], labels=["young", "middle", "senior", "elder"])
    
for df in [train_df, test_df]:
    df["HR_Percentage"] = df["Heart_Rate"] / (220 - df["Age"]) * 100


unique_durations_train = train_df['Duration'].unique()
unique_durations_test = test_df['Duration'].unique()


for duration in unique_durations_train:
    
    heart_rate_col = f'Heart_Rate_Duration_{int(duration)}'
    body_temp_col = f'Body_Temp_Duration_{int(duration)}'
    
    train_df[heart_rate_col] = np.where(train_df['Duration'] == duration, train_df['Heart_Rate'], 0)
    train_df[body_temp_col] = np.where(train_df['Duration'] == duration, train_df['Body_Temp'], 0)

for duration in unique_durations_test:

    heart_rate_col = f'Heart_Rate_Duration_{int(duration)}'
    body_temp_col = f'Body_Temp_Duration_{int(duration)}'
    
    test_df[heart_rate_col] = np.where(test_df['Duration'] == duration, test_df['Heart_Rate'], 0)
    test_df[body_temp_col] = np.where(test_df['Duration'] == duration, test_df['Body_Temp'], 0)


train_dummies = pd.get_dummies(train_df['Age_Group'], drop_first = True)
test_dummies = pd.get_dummies(test_df['Age_Group'], drop_first = True)


train_df = pd.concat([train_df, train_dummies], axis=1)
test_df = pd.concat([test_df, test_dummies], axis=1)

train_df.drop('Age_Group', axis=1, inplace = True)
test_df.drop('Age_Group', axis=1, inplace = True)


train_df.sample()


X = train_df.drop(['id','Calories'], axis= 1)#,'Temp_Binary','HeartRate_binary'
y = train_df['Calories']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def rmsle(y_true, y_pred):
    """
    Calculates the Root Mean Squared Logarithmic Error (RMSLE).

    Args:
        y_true (np.ndarray): The true values (1D array-like).
        y_pred (np.ndarray): The predicted values (1D array-like).

    Returns:
        float: The RMSLE score. Returns NaN if either input contains negative values.
    """
    if np.any(y_true < 0) or np.any(y_pred < 0):
        return np.nan  # Cannot take log of negative values

    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    squared_error = (log_true - log_pred) ** 2
    rmsle_score = np.sqrt(np.mean(squared_error))
    return rmsle_score


# Creating the XGBoost model
xgb_model = xgb.XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=9,
    subsample=0.9,
    colsample_bytree=0.7,
    eval_metric = 'rmse',
    gamma = 0.01,
    tree_method = 'gpu_hist' ,
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

print('R² score:', xgb_model.score(X_test, y_test))

y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
print(f"Train RMSE: {rmse_train}")

rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
print(f"Val RMSE: {rmse_test}")

# Calculate RMSLE for training data
rmsle_train = rmsle(y_train, y_pred_train)
print(f"Train RMSLE: {rmsle_train}")

# Calculate RMSLE for val data
rmsle_test = rmsle(y_test, y_pred_test)
print(f"Val RMSLE: {rmsle_test}")


'''corr = test_final.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, linewidth = 0.5, annot= True)'''


col = [col for col in X.columns]
col
test_final = test_df[col]
test_final.shape


xgb_model = xgb.XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=9,
    subsample=0.9,
    colsample_bytree=0.7,
    eval_metric = 'rmse',
    gamma = 0.01,
    tree_method = 'gpu_hist' ,
    random_state=42
)
xgb_model.fit(X, y)


# Predicting
predictions = xgb_model.predict(test_final)

# Clipping predictions between 1.0 and 314.0
predictions = predictions.clip(1.0, 314.0)

# Assigning to test_df
test_df["Calories"] = predictions
submission = test_df[["id", "Calories"]]


submission.to_csv("submission.csv", index=False)




