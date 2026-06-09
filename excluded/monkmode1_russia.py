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


car=pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv")


car


car.isnull().sum()


test=pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")


test.isnull().sum()


test


car.date=pd.to_datetime(car.date)


car.info()


car





car['hours']=car.date.dt.hour


car['minutes']=car.date.dt.minute
car['seconds']=car.date.dt.second


car['time']=car.date.dt.time



car['dow']=car.date.dt.day_name()



car['weekend']=np.where(car['dow'].isin(['Sunday','Saturday']),1,0)


car['week'] = car['date'].dt.isocalendar().week



car["month"] = car["date"].dt.month
car["year"] = car["date"].dt.year



car["dom"] = car["date"].dt.day



car["plate_length"] = car["plate"].apply(len)



car["plate_start_letter"] = car["plate"].str[0]






car.info()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.histplot(car['price'], bins=50, kde=True)
plt.title("Distribution of Price")
plt.show()



plt.figure(figsize=(8, 5))
sns.boxplot(x='dow', y='price', data=car, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title("Price Variation Across Days of the Week")
plt.show()



car.groupby('week')['price'].mean().plot(figsize=(10,5), marker='o', title="Average Price Trend Over Weeks")
plt.show()



sns.boxplot(x='hours', y='price', data=car)
plt.title("Price Distribution by Hour of the Day")
plt.show()


corr = car[['price', 'hours', 'minutes', 'seconds', 'week', 'weekend']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()



sns.boxplot(car['price'])
plt.title("Boxplot of Price")
plt.show()



from scipy import stats

# Apply Box-Cox Transformation (only works if price has no zeros or negatives)
car['boxcox_price'], lambda_value = stats.boxcox(car['price'] + 1)  # Adding 1 to handle zeros

# Plot the new distribution
plt.figure(figsize=(8, 5))
sns.histplot(car['boxcox_price'], bins=50, kde=True)
plt.title(f"Box-Cox Transformed Distribution of Price (Î»={lambda_value:.2f})")
plt.xlabel("Box-Cox(price)")
plt.ylabel("Count")
plt.show()

# Check new skewness
print("New Skewness:", car['boxcox_price'].skew())



sns.boxplot(car['boxcox_price'])
plt.title("Boxplot of Price")
plt.show()



def outlier_detection(data_column):
    sorted(data_column)
    Q1,Q3 = np.percentile(data_column,[25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5*IQR)
    upper_range = Q3 + (1.5*IQR)
    return lower_range,upper_range


lower_permissible_limit,upper_permissible_limit = outlier_detection(car["boxcox_price"])


print("Lower limit value: ", lower_permissible_limit)
print("Upper limit value: ", upper_permissible_limit)


price_out=car[(car["boxcox_price"] < lower_permissible_limit)|(car["boxcox_price"] > upper_permissible_limit)]


len(price_out)


lower_permissible_limit1,upper_permissible_limit1 = outlier_detection(car["hours"])


hours_out=car[(car["hours"] < lower_permissible_limit1)|(car["hours"] > upper_permissible_limit1)]


len(hours_out)


car['plate_region'] = car['plate'].str[:2]  # Extract first 2 characters






car['plate_count'] = car.groupby('plate')['plate'].transform('count')



plate_avg_price = car.groupby('plate')['boxcox_price'].mean()
car['plate_encoded'] = car['plate'].map(plate_avg_price)



car['date'] = pd.to_datetime(car['date']).dt.strftime('%Y/%m/%d')
test['date'] = pd.to_datetime(test['date']).dt.strftime('%Y/%m/%d')

car['date'] = pd.to_datetime(car['date'])
test['date'] = pd.to_datetime(test['date'])


from supplemental_english import REGION_CODES as region_codes

reverse_mapping = {code: key for key, values in region_codes.items() for code in values}



def create_features(df):
    df['region_code'] = df['plate'].apply(lambda x:x[6:])
    df['region_name'] = df['region_code'].map(reverse_mapping)
    df['top_code'] =  df['plate'].apply(lambda x:x[:1])
    df['Series'] = df['plate'].apply(lambda x:x[:1] + x[4:6])
    df['Registration_code'] = df['plate'].apply(lambda x:x[1:4])

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    return df

train = create_features(car)
test = create_features(test)


train.columns


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# âœ… Define Features & Target
features = [
    'plate_length', 'plate_start_letter', 'plate_count', 'plate_encoded',
    'hours', 'minutes', 'seconds', 'dow', 'weekend', 'week', 'month', 'year', 'dom', 'day',
    'region_code', 'region_name', 'top_code', 'Series', 'Registration_code'
]
target = 'boxcox_price'

X = train[features].copy()
y = train[target]
# âœ… Encode Categorical Features
categorical_features = ['region_code', 'region_name', 'top_code', 'Series', 'Registration_code']
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
from sklearn.preprocessing import LabelEncoder





for col in categorical_features:
    X[col] = X[col].fillna("Unknown")  # Replace NaN with 'Unknown'
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])



for col in categorical_features:
    X[col] = X[col].astype(str)  # Convert to string
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])



from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = ohe.fit_transform(X[categorical_features])

# Convert to DataFrame and merge with original dataset
X_encoded_df = pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out(categorical_features))
X = X.drop(columns=categorical_features).reset_index(drop=True)
X = pd.concat([X, X_encoded_df], axis=1)



X


print(X.dtypes)
print(X.head())



from sklearn.preprocessing import LabelEncoder

for col in ['plate_start_letter', 'dow']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])



print(X.select_dtypes(include=['object']).columns)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"XGBoost MSE: {mse_xgb}")



from catboost import CatBoostRegressor

cat = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, random_seed=42, verbose=0)
cat.fit(X_train, y_train)
y_pred_cat = cat.predict(X_test)

mse_cat = mean_squared_error(y_test, y_pred_cat)
print(f"CatBoost MSE: {mse_cat}")



from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'iterations': [500, 1000],
    'depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5]
}

# Perform Grid Search
cat_model = CatBoostRegressor(loss_function='RMSE', random_state=42, verbose=0)
grid_search = GridSearchCV(cat_model, param_grid, scoring='neg_mean_squared_error', cv=3, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Train CatBoost with best parameters
best_cat = CatBoostRegressor(**grid_search.best_params_, loss_function='RMSE', random_state=42, verbose=0)
best_cat.fit(X_train, y_train)
y_pred_best_cat = best_cat.predict(X_test)

# Compute new MSE
mse_best_cat = mean_squared_error(y_test, y_pred_best_cat)
print(f"Optimized CatBoost MSE: {mse_best_cat}")



test = test.drop(columns=['price'])



# Convert to datetime
test['date'] = pd.to_datetime(test['date'])

# Extract time-based features
test['hours'] = test['date'].dt.hour
test['minutes'] = test['date'].dt.minute
test['seconds'] = test['date'].dt.second
test['time'] = test['date'].dt.time
test['dow'] = test['date'].dt.day_name()
test['weekend'] = np.where(test['dow'].isin(['Sunday', 'Saturday']), 1, 0)
test['week'] = test['date'].dt.isocalendar().week
test["month"] = test["date"].dt.month
test["year"] = test["date"].dt.year
test["dom"] = test["date"].dt.day
test["plate_length"] = test["plate"].apply(len)
test["plate_start_letter"] = test["plate"].str[0]



test['plate_region'] = test['plate'].str[:2]  # Extract first 2 characters



test['plate_count'] = test.groupby('plate')['plate'].transform('count')



test['plate_encoded'] = test['plate'].map(plate_avg_price)  # Use mapping from train data



test['plate_encoded'].fillna(plate_avg_price.mean(), inplace=True)



kaggles=test['id'].copy()


kaggle=test.copy()


# Drop unnecessary columns (if any)
drop_columns = ['date', 'time', 'plate']  # Keep only relevant features
kaggle = kaggle.drop(columns=drop_columns, errors='ignore')



# Categorical features (must match training set)
categorical_features = ['region_code', 'region_name', 'top_code', 'Series', 'Registration_code']

# Use the same LabelEncoders from training
for col in categorical_features:
    if col in kaggle.columns:
        kaggle[col] = le.fit_transform(kaggle[col])  # Use the same LabelEncoder from training



kaggle[col] = kaggle[col].map(lambda x: le.classes_.index(x) if x in le.classes_ else -1)



# One-hot encode 'plate_start_letter' and 'dow' (if done in training)
kaggle = pd.get_dummies(kaggle, columns=['plate_start_letter', 'dow'], drop_first=True)



# Get missing columns (columns in train but not in test)
missing_cols = set(X_train.columns) - set(kaggle.columns)
for col in missing_cols:
    kaggle[col] = 0  # Add missing columns with default value 0

# Get extra columns (columns in test but not in train)
extra_cols = set(kaggle.columns) - set(X_train.columns)
kaggle = kaggle.drop(columns=extra_cols, errors='ignore')

# Ensure same column order
kaggle = kaggle[X_train.columns]



# Make predictions
test_predictions = best_cat.predict(kaggle)



from scipy.special import inv_boxcox

# Apply inverse Box-Cox transformation (if used in training)
test_predictions = inv_boxcox(test_predictions, lambda_value)  # Replace lambda_value with the one used in training



# Create submission DataFrame
submission = pd.DataFrame({'id': kaggles, 'price': test_predictions})

# Save to CSV for Kaggle submission
submission.to_csv("submission.csv", index=False)

print("âœ… Submission file 'submission.csv' created successfully! ðŸš€")



submission




