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



import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


train_path = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_path = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
train_path.info()


train_path.head(5)



missing_values = train_path.isnull().sum()
missing_values


train_path.keys()


plt.scatter(train_path['country'], train_path['num_sold'])
plt.xlabel('country')
plt.ylabel('num_sold')


plt.figure(figsize=(10, 6))
sns.barplot(data=train_path, x='country', y='num_sold')
plt.title('Price vs Country')
plt.xticks(rotation=45)  
plt.xlabel('Country')
plt.ylabel('Number sold')
plt.tight_layout()
plt.show()


if 'date' in train_path.columns:
    train_path['month'] = pd.to_datetime(train_path['date']).dt.month

plt.figure(figsize=(10, 6))
sns.barplot(x='country', y='num_sold', hue='month', data=train_path)

plt.title('Number of Items Sold by Country and Month')
plt.xlabel('Country')
plt.ylabel('Number Sold')
plt.legend(title='Month', loc='upper right')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


store_sales = train_path.groupby('store')['num_sold'].sum()
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(store_sales.index,store_sales.values)
ax.set_xlabel('Country')
ax.set_ylabel('Total Sales')
ax.set_title('Sales by store')


if 'month' not in train_path.columns:
    train_path['month'] = pd.to_datetime(train_path['date_column']).dt.month
monthly_sales = train_path.groupby('month')['num_sold'].sum()
month_labels = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']
plt.figure(figsize=(8, 8))
plt.pie(monthly_sales, labels=[month_labels[m-1] for m in monthly_sales.index],
        autopct='%1.1f%%', startangle=90)
plt.title('Sales Distribution by Month')
plt.show()


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder 

df = train_path

def data_processing(train_path):
    return train_path

train = data_processing(train_path)
test = data_processing(test_path)
 
def data_processing(df):
   
    categorical_features = ['country', 'store'] 
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, 
                              columns=encoder.get_feature_names_out(categorical_features))

    df = pd.concat([df, encoded_df], axis=1)

    df = df.drop(categorical_features, axis=1)

    return df


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder 

def data_processing(df):
    categorical_features = ['country', 'store', 'product']  
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, 
                              columns=encoder.get_feature_names_out(categorical_features))
    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop(categorical_features, axis=1)
    return df

train_path = '/kaggle/input/playground-series-s5e1/train.csv'
test_path = '/kaggle/input/playground-series-s5e1/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train = data_processing(train_df)
test = data_processing(test_df)

print("Train Data Columns:", train.columns)
print("Test Data Columns:", test.columns)

target_col = 'num_sold'
features = [col for col in train.columns if col != target_col and col != 'date'] 

train['year'] = pd.to_datetime(train['date']).dt.year
train['month'] = pd.to_datetime(train['date']).dt.month
train['day'] = pd.to_datetime(train['date']).dt.day

test['year'] = pd.to_datetime(test['date']).dt.year
test['month'] = pd.to_datetime(test['date']).dt.month
test['day'] = pd.to_datetime(test['date']).dt.day

features.extend(['year', 'month', 'day']) 


X = train[features]
y = train[target_col]
X_test = test[features] 

train = train.dropna(subset=[target_col])  
X = train[features]
y = train[target_col]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_val)
mse_linear = mean_squared_error(y_val, y_pred_linear)
print(f"Linear Regression MSE: {mse_linear}")
test_preds_linear = linear_model.predict(X_test)




sample_submission_path = '/kaggle/input/playground-series-s5e1/sample_submission.csv'
sample_submission = pd.read_csv(sample_submission_path)

submission = sample_submission.copy()  
submission['num_sold'] = test_preds_linear 

submission.to_csv('submission.csv', index=False)
print("Submission saved as submission.csv")


from sklearn.ensemble import RandomForestRegressor 

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)
mse_rf = mean_squared_error(y_val, y_pred_rf)
print(f"Random Forest MSE: {mse_rf}")

test_preds_rf = rf_model.predict(X_test)
submission['num_sold'] = test_preds_rf 

submission.to_csv('submission1.csv', index=False)
print("Submission saved as submission1.csv")


!pip install xgboost

from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)
mse_xgb = mean_squared_error(y_val, y_pred_xgb)
print(f"XGBoost MSE: {mse_xgb}")

test_preds_xgb = xgb_model.predict(X_test)
submission['num_sold'] = test_preds_xgb 

submission.to_csv('submission2.csv', index=False)
print("Submission saved as submission2.csv")


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd  

train_data = pd.read_csv(train_path)

train_data_cleaned = train_data.drop(columns=["id", "date"])

imputer = SimpleImputer(strategy="mean")
train_data_cleaned["num_sold"] = imputer.fit_transform(train_data_cleaned[["num_sold"]])

categorical_cols = ["country", "store", "product"]
label_encoders = {}
for col in categorical_cols:
    encoder = LabelEncoder()
    train_data_cleaned[col] = encoder.fit_transform(train_data_cleaned[col])
    label_encoders[col] = encoder

X = train_data_cleaned.drop(columns=["num_sold"])
y = train_data_cleaned["num_sold"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

xgb.plot_importance(xgb_model, importance_type="weight", title="Feature Importance (Weight)")
plt.show()

