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


# pip install numpy pandas matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import math
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# turning off warnings will prevent you from getting future changes -types of warnings

# machine learning model related libraries
# pip install scikit-learn

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


#load .csv file into variables
train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
extra_train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")



trtrain_df.head(10)ain_df.info()


train_df.head(10)


#makes a list of column names, handy variable for later use
colnames = train_df.columns.tolist()
colnames


train_df.describe()


corr_matrix = train_df.corr(numeric_only=True)

#mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(8, 6))  # Adjust the size as needed
sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt='.2f', linewidths=0.5, vmin=0, vmax=1)
#mask=mask, if you want to add the mask

plt.title('Correlation Matrix of Numerical Features')
plt.show()


# Select only numeric columns
numeric_cols = train_df.select_dtypes(include=['number'])

num_cols = len(numeric_cols.columns) 
grid_size = math.ceil(math.sqrt(num_cols))

fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(grid_size * 3, grid_size * 3))

axes = axes.flatten()

for i, column in enumerate(numeric_cols.columns):
    sns.boxplot(y=train_df[column], ax=axes[i],
                boxprops={"facecolor": "skyblue"},
                medianprops={"color": "black"},
                flierprops={"markerfacecolor": "red", "marker": "o"})
    axes[i].set_title(f"Boxplot of {column}")
    axes[i].set_facecolor("#e6f2ff")
    
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


numeric_columns = ['Price', 'Weight Capacity (kg)', 'Compartments']  
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(train_df[col], kde=True) 
    plt.gca().set_facecolor("#e6f2ff") 
    plt.title(f'Distribution of {col}')
    plt.show()


plt.figure(figsize=(8, 6)) 
sns.countplot(x="Brand", color="skyblue", data=train_df, edgecolor="Black")

plt.gca().set_facecolor("#e6f2ff")

plt.title('Countplot of Brand')
plt.show()


#counting and checking duplicates
duplicates = train_df.duplicated()
sum(duplicates)


duplicates = test_df.duplicated()
sum(duplicates)


duplicates = extra_train_df.duplicated()
sum(duplicates)


train_df.nunique()


for column in train_df.columns:
    print(f"Unique values for '{column}':")
    print(train_df[column].unique())
    print("-" * 40)


test_df.nunique()


for column in test_df.columns:
    print(f"Unique values for '{column}':")
    print(test_df[column].unique())
    print("-" * 40)


#shows how many nulls there is in the data
train_df.isnull().sum()


test_df.isnull().sum()


extra_train_df.isnull().sum()


missing_count = train_df.isnull().sum()
total_rows = len(train_df)

# Calculate the percentage of missing values
missing_percentage = (missing_count / total_rows) * 100

print(missing_percentage)


missing_count_test = test_df.isnull().sum()
total_rows_test = len(test_df)

missing_percentage_test = (missing_count_test / total_rows_test) * 100

print(missing_percentage_test)


train_df['Compartments'].fillna(train_df['Compartments'].mean(), inplace=True)
train_df['Weight Capacity (kg)'].fillna(train_df['Weight Capacity (kg)'].mean(), inplace=True)
train_df['Brand'].fillna(train_df['Brand'].mode()[0], inplace=True)
train_df['Material'].fillna(train_df['Material'].mode()[0], inplace=True)
train_df['Size'].fillna(train_df['Size'].mode()[0], inplace=True)
train_df['Laptop Compartment'].fillna(train_df['Laptop Compartment'].mode()[0], inplace=True)
train_df['Waterproof'].fillna(train_df['Waterproof'].mode()[0], inplace=True)
train_df['Style'].fillna(train_df['Style'].mode()[0], inplace=True)
train_df['Color'].fillna(train_df['Color'].mode()[0], inplace=True)

test_df['Compartments'].fillna(test_df['Compartments'].mean(), inplace=True)
test_df['Weight Capacity (kg)'].fillna(test_df['Weight Capacity (kg)'].mean(), inplace=True)
test_df['Brand'].fillna(test_df['Brand'].mode()[0], inplace=True)
test_df['Material'].fillna(test_df['Material'].mode()[0], inplace=True)
test_df['Size'].fillna(test_df['Size'].mode()[0], inplace=True)
test_df['Laptop Compartment'].fillna(test_df['Laptop Compartment'].mode()[0], inplace=True)
test_df['Waterproof'].fillna(test_df['Waterproof'].mode()[0], inplace=True)
test_df['Style'].fillna(test_df['Style'].mode()[0], inplace=True)
test_df['Color'].fillna(test_df['Color'].mode()[0], inplace=True)

extra_train_df['Compartments'].fillna(test_df['Compartments'].mean(), inplace=True)
extra_train_df['Weight Capacity (kg)'].fillna(test_df['Weight Capacity (kg)'].mean(), inplace=True)
extra_train_df['Brand'].fillna(test_df['Brand'].mode()[0], inplace=True)
extra_train_df['Material'].fillna(test_df['Material'].mode()[0], inplace=True)
extra_train_df['Size'].fillna(test_df['Size'].mode()[0], inplace=True)
extra_train_df['Laptop Compartment'].fillna(test_df['Laptop Compartment'].mode()[0], inplace=True)
extra_train_df['Waterproof'].fillna(test_df['Waterproof'].mode()[0], inplace=True)
extra_train_df['Style'].fillna(test_df['Style'].mode()[0], inplace=True)
extra_train_df['Color'].fillna(test_df['Color'].mode()[0], inplace=True)


missing_count = train_df.isnull().sum()


total_rows = len(train_df)
missing_percentage = (missing_count / total_rows) * 100

print(missing_percentage)


from sklearn.preprocessing import LabelEncoder

# Apply One-Hot Encoding for unordered categories
df_encoded = pd.get_dummies(train_df, columns=['Brand', 'Material', 'Color', 'Style'], drop_first=True)

# Apply Label Encoding for binary columns
le = LabelEncoder()
df_encoded['Laptop Compartment'] = le.fit_transform(df_encoded['Laptop Compartment'])
df_encoded['Waterproof'] = le.fit_transform(df_encoded['Waterproof'])

# Apply Label Encoding for ordinal column (Size)
size_order = {'Small': 0, 'Medium': 1, 'Large': 2}
df_encoded['Size'] = df_encoded['Size'].map(size_order)

print(df_encoded)


#Encoding the test-set
df_encoded_test = pd.get_dummies(test_df, columns=['Brand', 'Material', 'Color', 'Style'], drop_first=True)

le_test = LabelEncoder()
df_encoded_test['Laptop Compartment'] = le_test.fit_transform(df_encoded_test['Laptop Compartment'])
df_encoded_test['Waterproof'] = le_test.fit_transform(df_encoded_test['Waterproof'])

size_order_test = {'Small': 0, 'Medium': 1, 'Large': 2}
df_encoded_test['Size'] = df_encoded_test['Size'].map(size_order_test)

print(df_encoded_test)


#Encoding the extra_train_df
df_encoded_extra = pd.get_dummies(extra_train_df, columns=['Brand', 'Material', 'Color', 'Style'], drop_first=True)

le_extra = LabelEncoder()
df_encoded_extra['Laptop Compartment'] = le_extra.fit_transform(df_encoded_extra['Laptop Compartment'])
df_encoded_extra['Waterproof'] = le_extra.fit_transform(df_encoded_extra['Waterproof'])

size_order_extra = {'Small': 0, 'Medium': 1, 'Large': 2}
df_encoded_extra['Size'] = df_encoded_extra['Size'].map(size_order_extra)

print(df_encoded_extra)


plt.figure(figsize=(8, 6))
heatmap=sns.heatmap(df_encoded.corr(), annot=False, cmap='Blues', fmt=".4f", vmin=0, vmax=1)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=70, fontsize=9)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=9)
plt.title('Correlation Heatmap of Test DataSet')
plt.show()


from sklearn.preprocessing import StandardScaler

numerical_cols = ['Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Weight Capacity (kg)']  # Exclude 'Price'

scaler = StandardScaler()

df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# Keep 'Price' unchanged


#Apply the standardisation for the test set too

numerical_cols_test = ['Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Weight Capacity (kg)']  # Exclude 'Price'

scaler_test = StandardScaler()

df_encoded_test[numerical_cols_test] = scaler_test.fit_transform(df_encoded_test[numerical_cols_test])


#Apply the standardisation for the extra_train_df

numerical_cols_extra = ['Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Weight Capacity (kg)']  # Exclude 'Price'

scaler_extra = StandardScaler()

df_encoded_extra[numerical_cols_extra] = scaler_extra.fit_transform(df_encoded_extra[numerical_cols_extra])


X = df_encoded.drop("Price", axis=1)
y = df_encoded["Price"]

X_extra = df_encoded_extra.drop(columns='Price')
y_extra = df_encoded_extra['Price']

# divide the data to train and test sets for the models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# combine extra training set
X_combined = pd.concat([X_train, X_extra], axis=0)
y_combined = pd.concat([y_train, y_extra], axis=0)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model_LR = LinearRegression()
#model_LR.fit(X_train, y_train)

model_LR.fit(X_combined, y_combined)


# Make predictions on the test data
y_pred_LR = model_LR.predict(X_test)
r2_score_LR = model_LR.score(X_test, y_test)
print(f"Linear Regression R²: {r2_score_LR}")
mse_LR = mean_squared_error(y_test, y_pred_LR)
print(f"Linear Regression MSE: {mse_LR}")
rmse_LR = mean_squared_error(y_test, y_pred_LR, squared=False)
print("Linear Regression RMSE:", rmse_LR)



from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

#creating a variable for the test set
test_df_encoded = df_encoded_test 


for k in [20]:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_train_knn = knn.predict(X_train)
    y_pred_test_knn = knn.predict(X_test)

    #performance metrics
    mse_knn = mean_squared_error(y_test, y_pred_test_knn)
    rmse_knn = np.sqrt(mse_knn)
    r2_knn = r2_score(y_test, y_pred_test_knn)
    print(f"K={k} -> MSE: {mse_knn}, RMSE: {rmse_knn}, R²: {r2_knn}")


import xgboost as xgb

model_xgb = xgb.XGBRegressor(objective='reg:squarederror', 
                             n_estimators=100,
                             max_depth=4,     
                             learning_rate=0.05, 
                             subsample=0.8,   
                             colsample_bytree=0.8)

model_xgb.fit(X_train, y_train)


y_pred_xgb = model_xgb.predict(X_test)


mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost Test MSE: {mse_xgb}")
print(f"XGBoost Test R²: {r2_xgb}")
print(f"RMSE: {rmse_xgb}")

#test_df_encoded = df_encoded_test 
#y_pred_xgb_test = model_xgb.predict(test_df_encoded)


print(f"Linear Regression R²: {r2_score_LR}")
print(f"Linear Regression MSE: {mse_LR}")
print(f"Linear Regression RMSE: {rmse_LR}")

print(f"K={k} -> R²: {r2_knn}")
print(f"K={k} -> MSE: {mse_knn}")
print(f"K={k} -> RMSE: {rmse_knn}")

print(f"XGBoost R²: {r2_xgb}")
print(f"XGBoost MSE: {mse_xgb}")
print(f"XGBoost RMSE: {rmse_xgb}")

