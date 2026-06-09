# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split 
from xgboost import plot_importance


# Read the data
train = pd.read_csv('../input/sparta-2024-data-science-competition/train.csv', index_col='id')
test = pd.read_csv('../input/sparta-2024-data-science-competition/test.csv', index_col='id')


train.head()


train.columns


null_col_train = train.columns[train.isna().sum() > 0].tolist()

missing_values = train[null_col_train].isna().sum()
missing_percentage = (missing_values / len(train)) * 100


missing_info = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage.round(2).astype(str) + '%'
})

print("Number of missing values and percentage per column")
print(missing_info)



test.head()


test.columns


null_col_test = test.columns[test.isna().sum() > 0].tolist()

missing_values = test[null_col_test].isna().sum()
missing_percentage = (missing_values / len(test)) * 100


missing_info = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage.round(2).astype(str) + '%'
})

print("Number of missing values and percentage per column")
print(missing_info)



train.info()


print("Statistical Summary for Numerical Columns:")
print(train.describe())



# duplicate check
print(f"Number of dupicate rows in train set: {train.duplicated().sum()}")



import matplotlib.pyplot as plt

# Plot histograms for numerical features
train.hist(bins=50, figsize=(20, 15))
plt.suptitle('Distribution of Numerical Features in the Train Set')
plt.show()



# Check the number of duplicates 
print(f"Number of duplicates before removal: {train.duplicated().sum()}")

# Remove duplicates
train.drop_duplicates(inplace=True)

# Check the number of duplicates 
print(f"Number of duplicates after removal: {train.duplicated().sum()}")



# drop columns with more than 50% missing values in train and test sets
train.drop(columns=['neighborhood_overview', 'host_about', 'host_neighbourhood'], inplace=True)
test.drop(columns=['neighborhood_overview', 'host_about', 'host_neighbourhood'], inplace=True)



from sklearn.impute import SimpleImputer

# List of numerical columns with missing values
numerical_columns_with_missing = ['host_listings_count', 'host_total_listings_count', 'bathrooms', 
                                 'bedrooms', 'beds', 'availability_eoy', 'number_of_reviews_ly', 
                                 'estimated_occupancy_l365d', 'estimated_revenue_l365d', 
                                 'review_scores_rating', 'review_scores_accuracy', 
                                 'review_scores_cleanliness', 'review_scores_checkin', 
                                 'review_scores_communication', 'review_scores_location', 
                                 'review_scores_value', 'reviews_per_month']

# apply imputation
imputer = SimpleImputer(strategy='median')

# impute the missing values in the train set
train[numerical_columns_with_missing] = imputer.fit_transform(train[numerical_columns_with_missing])

# impute the missing values in the test set
test[numerical_columns_with_missing] = imputer.transform(test[numerical_columns_with_missing])

print("Missing values in train after imputation:")
print(train[numerical_columns_with_missing].isnull().sum())

print("\nMissing values in test after imputation:")
print(test[numerical_columns_with_missing].isnull().sum())



seed = 42
np.random.seed(seed)


X = train.dropna(axis=0, subset=['price'])

y = X['price']  # Target variable in train set (price)
X.drop(['price'], axis=1, inplace=True)  # Drop 'price' column for features


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y,
                                                                train_size=0.8,
                                                                test_size=0.2,
                                                                random_state=seed)

# Select categorical columns
low_cardinality_cols = [cname for cname in X_train_full.columns 
                        if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns
                if X_train_full[cname].dtype in ['int64', 'float64']]


my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

X_test = test[my_cols].copy()

# One-hot encode the categorical columns
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)


# clean column names
X_train.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for col in X_train.columns]
X_valid.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for col in X_valid.columns]
X_test.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for col in X_test.columns]

# align the dataframes
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)




# model Fitting and Prediction
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=seed)

# Fit the model
xgb.fit(X_train, y_train)

# Get predictions
y_pred = xgb.predict(X_valid)




# Model Evaluation
rmse = np.sqrt(mean_squared_error(y_pred, y_valid))
print("Root Mean Squared Error (RMSE):", rmse)



y_pred = xgb.predict(X_test)

#  submission file
output = pd.DataFrame({'Id': X_test.index,
                       'price': y_pred})


output.to_csv('submission.csv', index=False)
print(output.head())



plt.figure(figsize=(10, 8))
plot_importance(xgb, importance_type='gain', max_num_features=10, color='royalblue')
plt.title('Top 10 Features by Gain', fontsize=16)
plt.show()


