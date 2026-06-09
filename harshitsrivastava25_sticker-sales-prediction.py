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


import pandas as pd 
import numpy as np
import matplotlib.pyplot as pyt


ds = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
ds.dtypes


ds.shape


ds.columns


ds['country'].value_counts()


# total Six countries


ds['store'].value_counts()


# there are total 3 stores


ds['product'].value_counts()


# total Five products 


ds = ds[[ 'date', 'country', 'store', 'product', 'num_sold']]
ds.head()


ds.isnull().sum()


# there are only missing values in the num_sold (the prediction values)
# In this case I think it is better to remove the columns that has this missing values


ds['num_sold'].describe()


# Remove rows with missing values in 'num_sold'
ds = ds.dropna(subset=['num_sold'])

# Verify if missing values are removed
print(ds.isnull().sum())



# Convert 'date' column to datetime format
ds['date'] = pd.to_datetime(ds['date'])
# Verify the conversion
print(ds['date'].dtypes)  # Should show dtype as datetime64[ns]
print(ds.head())          # Display the first few rows


import pandas as pd

# Ensure the 'date' column is in datetime format
ds['date'] = pd.to_datetime(ds['date'], format='%d-%m-%Y')

# Extract the required components and create new columns
ds['Day'] = ds['date'].dt.day_name()  # Extracts the day of the week (e.g., Monday, Tuesday)
ds['Date'] = ds['date'].dt.day       # Extracts the day of the month (e.g., 1, 2, 31)
ds['Month'] = ds['date'].dt.month    # Extracts the month as a number (e.g., 1 for January)
ds['Year'] = ds['date'].dt.year      # Extracts the year (e.g., 2010)

# Drop the original 'date' column if no longer needed (optional)
# ds = ds.drop(columns=['date'])

# Display the first few rows of the updated dataset
print(ds.head())




ds.dtypes



ds.columns 


ds = ds [ ['country', 'store', 'product', 'num_sold', 'Day', 'Date',
       'Month', 'Year']]


lst_cat = [ 'country', 'store', 'product', 'Day']


for item in lst_cat:
    print(ds[item].value_counts())


# for column country 
from sklearn.preprocessing import OneHotEncoder

# Sample DataFrame
df = ds 
# Initialize OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False)

# Fit and transform the 'country' column
one_hot_encoded = one_hot_encoder.fit_transform(df[['country']])

# Create a DataFrame for the encoded data with proper column names
encoded_columns = one_hot_encoder.get_feature_names_out(['country'])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoded_columns)

# Combine the original DataFrame with the encoded columns
df = pd.concat([df, one_hot_df], axis=1)

# Drop the original 'country' column if no longer needed
# df = df.drop('country', axis=1)

print(df)


df.columns


ds = df[['store', 'product', 'num_sold', 'Day', 'Date', 'Month',
       'Year', 'country_Canada', 'country_Finland', 'country_Italy',
       'country_Kenya', 'country_Norway', 'country_Singapore']]


ds.head()


df = ds              
# Fit and transform the 'Day' column
one_hot_encoded = one_hot_encoder.fit_transform(df[['Day']])

# Create a DataFrame for the encoded data with proper column names
encoded_columns = one_hot_encoder.get_feature_names_out(['Day'])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoded_columns)

# Combine the original DataFrame with the encoded columns
df = pd.concat([df, one_hot_df], axis=1)

# Drop the original 'Day' column if no longer needed
# df = df.drop('Day', axis=1)

print(df)


df.drop(columns='Day', inplace=True)
ds=df 
ds.head()


one_hot_encoded = one_hot_encoder.fit_transform(df[['store']])

# Create a DataFrame for the encoded data with proper column names
encoded_columns = one_hot_encoder.get_feature_names_out(['store'])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoded_columns)

# Combine the original DataFrame with the encoded columns
df = pd.concat([df, one_hot_df], axis=1)

# Drop the original 'store' column
df.drop(columns='store', inplace=True)

print(df)


# print(df['store'].apply(lambda x: x == '' or x is None).sum())



# df['store'].value_counts()


# df = df.drop(columns=['store'])


df.head()


df.columns


df["Day_nan"].value_counts()



one_hot_encoder = OneHotEncoder()

# Apply one-hot encoding to the 'product' column
one_hot_encoded = one_hot_encoder.fit_transform(df[['product']])

# Create a DataFrame for the encoded data with proper column names
encoded_columns = one_hot_encoder.get_feature_names_out(['product'])
one_hot_df = pd.DataFrame(one_hot_encoded.toarray(), columns=encoded_columns)

# Combine the original DataFrame with the encoded columns
df = pd.concat([df, one_hot_df], axis=1)

# Drop the original 'product' column
df.drop(columns='product', inplace=True)

print(df)



df['product_nan'].value_counts()


df.dtypes


# Specify the columns to check for the value 1
columns_to_check = ['product_nan', 'store_nan', 'Day_nan']

# Drop rows where any of the specified columns have the value 1
df = df[~df[columns_to_check].eq(1).any(axis=1)]

print(df)



df.shape


# Drop the specified columns
# df = df.drop(columns=['product_nan', 'store_nan', 'Day_nan'], errors='ignore')


df.columns



for column in df.columns: 
    print(df[column].value_counts())


from sklearn.preprocessing import MinMaxScaler

# List of columns to normalize (continuous columns)
columns_to_normalize = [ 'Date', 'Month', 'Year']  # Replace with your actual continuous column names

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Apply MinMax scaling to the selected columns
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

print(df)



# Normalization of the y data is required when We use Neural Networks


# Drop rows where 'num_sold' has missing values
df.dropna(subset=['num_sold'], inplace=True)

# Verify the change
print(df.head())



import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix for the DataFrame
correlation_matrix = df.corr()

# Extract the correlation values with the target variable 'num_sold'
target_corr = correlation_matrix[['num_sold']].sort_values(by='num_sold', ascending=False)

# Plot the heatmap for the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(target_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation with num_sold')
plt.show()






df.columns


len(df.columns)


df.head()


df.drop(columns='Day_nan', inplace=True)
df.drop(columns='product_nan', inplace=True)
df.drop(columns='store_nan', inplace=True)



X = df.iloc[:,1:].values
len(X)


y = df.iloc[:,0].values 
len(y)


# from sklearn.linear_model import LinearRegression, Lasso
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.model_selection import train_test_split
# import pandas as pd

# # Assuming X (input features) and y (target) are already defined
# X = df.drop(columns=['num_sold'])  # Drop the target column from the features
# y = df['num_sold']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 1. Linear Regression (with coefficients)
# linear_model = LinearRegression()
# linear_model.fit(X_train, y_train)
# linear_coeffs = pd.Series(linear_model.coef_, index=X.columns).sort_values(ascending=False)

# # 2. Lasso Regression (with L1 regularization)
# lasso_model = Lasso(alpha=0.1)  # You can adjust alpha for regularization strength
# lasso_model.fit(X_train, y_train)
# lasso_coeffs = pd.Series(lasso_model.coef_, index=X.columns).sort_values(ascending=False)

# # 3. Random Forest Regressor (Feature importance)
# rf_model = RandomForestRegressor(random_state=42)
# rf_model.fit(X_train, y_train)
# rf_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# # 4. Gradient Boosting Regressor (Feature importance)
# gb_model = GradientBoostingRegressor(random_state=42)
# gb_model.fit(X_train, y_train)
# gb_importance = pd.Series(gb_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# # Combine all results into a DataFrame
# importance_df = pd.DataFrame({
#     'Linear Regression': linear_coeffs,
#     'Lasso Regression': lasso_coeffs,
#     'Random Forest': rf_importance,
#     'Gradient Boosting': gb_importance
# })

# # Print the top 5 most important features from each model
# print(importance_df.head(5))



# # Print sorted feature importances
# print("\n")
# print(gb_importance.sort_values(ascending=False))
# print("\n")
# # Print sorted feature importances
# print(rf_importance.sort_values(ascending=False))
# print("\n")
# # Print sorted feature importances
# print(linear_coeffs.sort_values(ascending=False))
# print("\n")
# # Print sorted feature importances
# print(lasso_coeffs.sort_values(ascending=False))



# # from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
# from sklearn.svm import SVR
# from sklearn.tree import DecisionTreeRegressor
# lr = DecisionTreeRegressor()


import pandas as pd

# Assuming feature names are known and stored in a list called feature_names
feature_names = ds.columns
X_df = pd.DataFrame(X, columns=feature_names)
# Assuming X and y are pandas DataFrame/Series
correlations = X_df.corrwith(y)
print(correlations)

# Select features with high correlation (absolute value above a threshold, e.g., 0.5)
relevant_features = correlations[abs(correlations) > 0.5].index
# X_selected = X[relevant_features]
relevant_features



from sklearn.feature_selection import mutual_info_regression


# Assuming feature names are known and stored in a list called feature_names
feature_names = ds.columns
X_df = pd.DataFrame(X, columns=feature_names)

# Calculate mutual information
mi_scores = mutual_info_regression(X_df, y)

# Create a DataFrame for better visualization
mi_df = pd.DataFrame({'Feature': X_df.columns, 'MI Score': mi_scores}).sort_values(by='MI Score', ascending=False)
print(mi_df)

# Select top features based on MI scores
top_features = mi_df[mi_df['MI Score'] > 0.1]['Feature']
X_selected = X_df[top_features]



from sklearn.ensemble import GradientBoostingRegressor

# Fit the model
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X, y)

# Extract feature importance
importances = gb_model.feature_importances_
gb_feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print(gb_feature_importance)



from xgboost import XGBRegressor

# Fit the model
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X, y)

# Extract feature importance
importances = xgb_model.feature_importances_
xgb_feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print(xgb_feature_importance)



from lightgbm import LGBMRegressor

# Fit the model
lgb_model = LGBMRegressor(random_state=42)
lgb_model.fit(X, y)

# Extract feature importance
importances = lgb_model.feature_importances_
lgb_feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print(lgb_feature_importance)



from catboost import CatBoostRegressor

# Fit the model
cat_model = CatBoostRegressor(verbose=0, random_state=42)
cat_model.fit(X, y)

# Extract feature importance
importances = cat_model.feature_importances_
cat_feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print(cat_feature_importance)



from sklearn.linear_model import ElasticNetCV

# Fit the model
elastic_net = ElasticNetCV(cv=5, random_state=42).fit(X, y)

# Extract feature importance
elastic_net_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': elastic_net.coef_}).sort_values(by='Coefficient', ascending=False)
print(elastic_net_importance)



from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Fit RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Extract feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# Select top features (e.g., top 10 or based on a threshold)
top_features = feature_importance_df[feature_importance_df['Importance'] > 0.01]['Feature']
X_selected = X[top_features]



from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

# Initialize RandomForestRegressor
model = RandomForestRegressor(random_state=42)

# Perform RFE
rfe = RFE(estimator=model, n_features_to_select=10)  # Select top 10 features
rfe.fit(X, y)

# Get selected features
selected_features = X.columns[rfe.support_]
X_selected = X[selected_features]
print("Selected Features:", selected_features)



from sklearn.linear_model import LassoCV

# Fit Lasso model
lasso = LassoCV(cv=5, random_state=42).fit(X, y)

# Get features with non-zero coefficients
selected_features = X.columns[lasso.coef_ != 0]
X_selected = X[selected_features]
print("Selected Features:", selected_features)




ds_test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
ds = ds_test


ds = ds[[ 'date', 'country', 'store', 'product']]



# Convert 'date' column to datetime format
ds['date'] = pd.to_datetime(ds['date'])
# Verify the conversion
print(ds['date'].dtypes)  # Should show dtype as datetime64[ns]
print(ds.head())          # Display the first few rows


import pandas as pd

# Ensure the 'date' column is in datetime format
ds['date'] = pd.to_datetime(ds['date'], format='%d-%m-%Y')

# Extract the required components and create new columns
ds['Day'] = ds['date'].dt.day_name()  # Extracts the day of the week (e.g., Monday, Tuesday)
ds['Date'] = ds['date'].dt.day       # Extracts the day of the month (e.g., 1, 2, 31)
ds['Month'] = ds['date'].dt.month    # Extracts the month as a number (e.g., 1 for January)
ds['Year'] = ds['date'].dt.year      # Extracts the year (e.g., 2010)

# Drop the original 'date' column if no longer needed (optional)
# ds = ds.drop(columns=['date'])

# Display the first few rows of the updated dataset
print(ds.head())



ds = ds [ ['country', 'store', 'product', 'Day', 'Date',
       'Month', 'Year']]


# for column country 
from sklearn.preprocessing import OneHotEncoder


# Initialize OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False)

# Fit and transform the 'country' column
one_hot_encoded = one_hot_encoder.fit_transform(ds[['country']])

# Create a DataFrame for the encoded data with proper column names
encoded_columns = one_hot_encoder.get_feature_names_out(['country'])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoded_columns)

# Combine the original DataFrame with the encoded columns
ds = pd.concat([ds, one_hot_df], axis=1)

# Drop the original 'country' column if no longer needed
ds = ds.drop('country', axis=1)

print(ds)


              
# Fit and transform the 'Day' column
one_hot_encoded = one_hot_encoder.fit_transform(ds[['Day']])

# Create a DataFrame for the encoded data with proper column names
encoded_columns = one_hot_encoder.get_feature_names_out(['Day'])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoded_columns)

# Combine the original DataFrame with the encoded columns
ds = pd.concat([ds, one_hot_df], axis=1)

# Drop the original 'Day' column if no longer needed
# ds = ds.drop('Day', inplace=True,axis=1)

print(ds)


ds.columns


one_hot_encoded = one_hot_encoder.fit_transform(ds[['store']])

# Create a DataFrame for the encoded data with proper column names
encoded_columns = one_hot_encoder.get_feature_names_out(['store'])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoded_columns)

# Combine the original DataFrame with the encoded columns
ds = pd.concat([ds, one_hot_df], axis=1)

# Drop the original 'store' column
ds.drop(columns='store', inplace=True,axis=1)

print(ds)



one_hot_encoder = OneHotEncoder()

# Apply one-hot encoding to the 'product' column
one_hot_encoded = one_hot_encoder.fit_transform(ds[['product']])

# Create a DataFrame for the encoded data with proper column names
encoded_columns = one_hot_encoder.get_feature_names_out(['product'])
one_hot_df = pd.DataFrame(one_hot_encoded.toarray(), columns=encoded_columns)

# Combine the original DataFrame with the encoded columns
ds = pd.concat([ds, one_hot_df], axis=1)

# Drop the original 'product' column
ds.drop(columns='product', inplace=True,axis=1)

print(ds)



len(df.columns)


# X_test = ds.loc[:,df.columns].values
ds.columns


ds = ds[[ 'Date', 'Month', 'Year', 'country_Canada', 'country_Finland',
       'country_Italy', 'country_Kenya', 'country_Norway', 'country_Singapore',
       'Day_Friday', 'Day_Monday', 'Day_Saturday', 'Day_Sunday',
       'Day_Thursday', 'Day_Tuesday', 'Day_Wednesday',
       'store_Discount Stickers', 'store_Premium Sticker Mart',
       'store_Stickers for Less', 'product_Holographic Goose',
       'product_Kaggle', 'product_Kaggle Tiers', 'product_Kerneler',
       'product_Kerneler Dark Mode']]


ds.columns


columns_to_normalize = [ 'Date', 'Month', 'Year']  # Replace with your actual continuous column names

# Initialize MinMaxScaler
# scaler = MinMaxScaler()

# Apply MinMax scaling to the selected columns
ds[columns_to_normalize] = scaler.transform(ds[columns_to_normalize])

print(ds)



X_test =ds.iloc[:,:].values


y_pred = best_model.predict(X_test)


len(y_pred)


# Generate the id column starting from 230130 to 328679
id_column = list(range(230130, 328680))

# Create the DataFrame
df = pd.DataFrame({
    'id': id_column,
    'num_sold': y_pred
})

# Save to CSV
df.to_csv('/kaggle/working/Submission5bst.csv', index=False)

# Provide the file path for download
print("CSV file 'Submission2.csv' has been created. You can download it from the Kaggle environment.")

