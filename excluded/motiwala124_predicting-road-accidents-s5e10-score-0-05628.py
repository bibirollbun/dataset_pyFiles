#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


#loading the datasets
from google.colab import drive
drive.mount('/content/drive')
df_train = pd.read_csv('/content/drive/MyDrive/train.csv')
df_test = pd.read_csv('/content/drive/MyDrive/test.csv')


#display train dataset
df_train.head()


#info
print("Information of the Dataset:\n", df_train.info())

#summary statiistics
print("\nSummary Statistics\n", df_train.describe())

#data types
print("\nData Types\n", df_train.dtypes)

#null values
print("\nNull Values\n", df_train.isnull().sum())

#shape
print("\nShape of the Dataset", df_train.shape)


#convert boolean columns to integers
bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in bool_cols:
  df_train[col] = df_train[col].astype(int)

#check
df_train[bool_cols].head()


#Encoding categorical(object columns)
cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']

#one hot encode (converting into 0/1)
df_train = pd.get_dummies(df_train, columns = cat_cols, drop_first = True)

# Double-check data types
print(df_train.dtypes)

#check
print("Shape after Encoding", df_train.shape)
df_train.head()


# Step: Convert True/False to 0/1 across the whole dataframe
df_train = df_train.replace({True: 1, False: 0})

# Confirm the conversion worked
df_train.head()



#scaling the numeric columns
scaler = StandardScaler()

num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
df_train[num_cols] = scaler.fit_transform(df_train[num_cols])

#check
df_train[num_cols].head()


#seperating features and target columns
X = df_train.drop(['id', 'accident_risk'], axis = 1)
y = df_train['accident_risk']

#Check shapes
print("X shape:", X.shape)
print("y shape:", y.shape)


#train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)



#evaluate model perfomance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 score:", r2)


df_test.head()


#info
print("Information of the Dataset:\n", df_test.info())

#summary statiistics
print("\nSummary Statistics\n", df_test.describe())

#data types
print("\nData Types\n", df_test.dtypes)

#null values
print("\nNull Values\n", df_test.isnull().sum())

#shape
print("\nShape of the Dataset", df_test.shape)


#convert boolean columns to integers
bool_cols2 = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in bool_cols2:
  df_test[col] = df_test[col].astype(int)

#check
df_test[bool_cols].head()


cat_cols2 = ['road_type', 'lighting', 'weather', 'time_of_day']

#one hot encode (converting into 0/1)
df_test = pd.get_dummies(df_test, columns = cat_cols, drop_first = True)

# Double-check data types
print(df_test.dtypes)

#check
print("Shape after Encoding", df_test.shape)
df_test.head()


# Step: Convert True/False to 0/1 across the whole dataframe
df_test = df_test.replace({True: 1, False: 0})

# Confirm the conversion worked
df_test.head()


#scaling the numeric columns
scaler = StandardScaler()

num_cols2 = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
df_test[num_cols2] = scaler.fit_transform(df_test[num_cols2])

#check
df_test[num_cols2].head()


print(df_test.info())
print(df_train.info())

print(df_test.shape)
print(df_train.shape)


df_test.head()


# Predict accident risk for test data
X_test = df_test.drop(columns=['id'])
y_pred = lr.predict(X_test)

# Create submission DataFrame
S5ep10_submission = pd.DataFrame({
    'id': df_test['id'],
    'accident_risk': y_pred
})



S5ep10_submission.head()


#saving file in drive
S5ep10_submission.to_csv('/content/drive/MyDrive/submission.csv', index=False)


S5ep10_submission.tail()


# Pick the top continuous features
num_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

for col in num_features:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df_train[col], y=df_train['accident_risk'], alpha=0.3)
    plt.title(f"{col} vs accident_risk")
    plt.show()



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# (Reconfirm split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_depth=15
)
rf.fit(X_train, y_train)

# Predict on test data
y_pred_rf = rf.predict(X_test)

# Evaluate
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest RMSE:", rmse_rf)
print("Random Forest R2:", r2_rf)



# Predict on the scaled test data
X_test_final = df_test.drop('id', axis=1)   # test data without ID
test_id = df_test['id']                     # keep ID for submission

# Make predictions using the trained Random Forest model
y_pred_test = rf.predict(X_test_final)

# Create submission DataFrame
submission_2 = pd.DataFrame({
    'id': test_id,
    'accident_risk': y_pred_test
})


submission_2.to_csv('/content/drive/MyDrive/submission.csv', index=False)


!pip install xgboost

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# XGBoost Model
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Train
xgb_model.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(X_test)

# Metrics
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost RMSE:", rmse_xgb)
print("XGBoost R2:", r2_xgb)



# Predict on the scaled test data
X_test_final = df_test.drop('id', axis=1)   # test data without ID
test_id = df_test['id']                     # keep ID for submission

# Make predictions using the trained Random Forest model
y_pred_test = rf.predict(X_test_final)

# Create submission DataFrame
submission_2 = pd.DataFrame({
    'id': test_id,
    'accident_risk': y_pred_test
})
submission_2.to_csv('/content/drive/MyDrive/submission.csv', index=False)

