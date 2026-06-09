# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/back-pack-data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv('/kaggle/input/back-pack-data/train.csv')
test_data = pd.read_csv('/kaggle/input/back-pack-data/test.csv')
extra_train_data = pd.read_csv('/kaggle/input/back-pack-data/training_extra.csv')

#lets concat the extra train data to main train data
train_data = pd.concat([train_data,extra_train_data])


train_data.head()


test_data.head()


#check the shape of the train data 
train_data.shape


#lets see the data types and not-null information about the data
train_data.info()


#lets check for the missing values in the data
train_data.isnull().sum()


#lets see about the value counts in each of categorical data
for i in train_data.select_dtypes('object'):
    print(train_data[i].value_counts())
    print('*'*100)


#will see the brand vs price as brand plays some roles in pricing

plt.figure(figsize=(12,6))
x_values = train_data.groupby('Brand')['Price'].mean().sort_values(ascending=False).index
y_values = train_data.groupby('Brand')['Price'].mean().sort_values(ascending=False).values
ax = sns.barplot(x=x_values,y=y_values,palette='rocket')
plt.title('Brand vs Price')
plt.xlabel('Brand')
plt.ylabel('Price')
plt.xticks(rotation=90)
# Iterate over the bars and display actual values
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f")  # Format as needed

plt.show()



plt.figure(figsize=(12,6))

material_sums = train_data.groupby('Material')['Price'].sum().reset_index()

# Create a bar plot with actual values
ax = sns.barplot(x='Material', y='Price', data=material_sums, palette='rainbow')

plt.title('Total Price Distribution by Material')
plt.xlabel('Material')
plt.ylabel('Total Price')

# Iterate over the bars and display actual values
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f")  # Format as needed

plt.xticks(rotation=45)  # Rotate labels if needed
plt.show()



#distribution based on the size of the bags

plt.figure(figsize=(12,6))
ax = sns.countplot(x=train_data.groupby('Size').count().index,palette='rainbow')
plt.title('Size Distribution')
plt.xlabel('Size')
plt.ylabel('Count')
# Iterate over the bars and display actual values
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f")  # Format as needed

plt.show()


#lets see the how the Waterproof will impact the price
plt.figure(figsize=(12,6))
x_values = train_data.groupby('Waterproof')['Price'].mean().sort_values(ascending=False).index
y_values = train_data.groupby('Waterproof')['Price'].mean().sort_values(ascending=False).values
sns.barplot(x=x_values,y=y_values,palette='rocket')
plt.title('Waterproof vs Price')
plt.xlabel('Waterproof')
plt.ylabel('Price')
plt.xticks(rotation=90)
plt.show()



#now lets undersatnd how the weight will play the role in price

plt.figure(figsize=(12,6))
sns.histplot(train_data['Weight Capacity (kg)'],kde=True)
plt.title('Weight vs count')
plt.xlabel('Weight')
plt.ylabel('count')
plt.show()



#lets see the avg price for all the brand's based on the size

brand_size_price = train_data.groupby(['Brand','Size'])['Price'].mean().sort_values(ascending=False)
brand_size_price
plt.figure(figsize=(12,6))
brand_size_price = brand_size_price.reset_index()
ax = sns.barplot(x ='Brand' ,y='Price',hue='Size',data=brand_size_price,palette='rainbow')
plt.title('Brand vs Size vs Price')
plt.xlabel('Brand')
plt.ylabel('Price')
plt.xticks(rotation=90)
# Iterate over the bars and display actual values
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f")  # Format as needed

plt.show()


#brand and style and price
brand_style_price = train_data.groupby(['Brand','Style'])['Price'].mean().sort_values(ascending=False)
brand_style_price = brand_style_price.reset_index()
plt.figure(figsize=(12,6))
ax = sns.barplot(x ='Brand' ,y='Price',hue='Style',data=brand_style_price,palette='rainbow')
plt.title('Brand vs Style vs Price')
plt.xlabel('Brand')
# Iterate over the bars and display actual values
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f")  # Format as needed

plt.show()


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

#saving the Id for the submission
idtest = test_data['id']

'''
drop the id column from the data
'''
train_data.drop(columns=['id'],inplace=True)
test_data.drop(columns=['id'],inplace=True)

# Assume train_data is the training dataset, and test_data is the test dataset
X = train_data.drop(columns=['Price'])  # Features
y = train_data['Price']  # Target variable

# Identify categorical and numerical columns
categorical = X.select_dtypes(include=['object']).columns.tolist()
numerical = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Define transformation pipeline for numerical features (imputation + scaling)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler())  # Standardization
])

# Define transformation pipeline for categorical features (encoding)
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-Hot Encoding with unknown category handling
])

# Combine transformations using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical),
        ('cat', categorical_transformer, categorical)
    ]
)

# Fit-transform on training data
X_processed = preprocessor.fit_transform(X)

# Apply the same transformation to test data
X_test_processed = preprocessor.transform(test_data)

# Convert the processed data back to DataFrames
# Get feature names after transformation
numerical_features = numerical
categorical_features = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical)

# Combine all feature names
all_features = np.concatenate([numerical_features, categorical_features])

# Create DataFrames
X_processed_df = pd.DataFrame(X_processed, columns=all_features)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=all_features)




#lets see how the data looks like now
X_processed_df.head()


X_test_processed_df.head()


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm  as lgb
from catboost import CatBoostRegressor

from sklearn.ensemble import VotingRegressor

Xtrain,Xtest,ytrain,ytest = train_test_split(X_processed_df,y,test_size=0.2,random_state=42)



# Define individual models with best possible hyperparameters
lgb_model = lgb.LGBMRegressor(
    subsample=0.7,
    num_leaves=30,  
    n_estimators=300,
    max_depth=6,  
    learning_rate=0.01,
    feature_fraction=0.8,  # Randomly selects 80% features per tree
    bagging_freq=5,  # Enables bagging
    reg_lambda=2  # L2 regularization to prevent overfitting
)

xgb_model = xgb.XGBRegressor(
    subsample=0.7,
    n_estimators=300,
    max_depth=5,  
    learning_rate=0.01,
    colsample_bytree=0.8,
    gamma=1.5,  # Adds pruning to avoid overfitting
    reg_lambda=2,  # L2 regularization
    reg_alpha=2,  # L1 regularization
    objective='reg:squarederror',
    random_state=42
)


# Fit models individually with early stopping
lgb_model.fit(Xtrain, ytrain, eval_set=[(Xtest, ytest)], eval_metric='rmse', callbacks=[lgb.early_stopping(50)])
xgb_model.fit(Xtrain, ytrain, eval_set=[(Xtest, ytest)], eval_metric='rmse', early_stopping_rounds=50, verbose=False)

# Create Voting Regressor
voting_regressor = VotingRegressor(estimators=[
    ('lightgbm', lgb_model),
    ('xgboost', xgb_model)
])

# Train Voting Regressor
voting_regressor.fit(Xtrain, ytrain)



# Predict on validation set
val_predictions = voting_regressor.predict(Xtest)

# Calculate RMSE on the validation set
rmse = np.sqrt(mean_squared_error(ytest, val_predictions))
print(f"Voting Regressor RMSE: {rmse:.5f}")

# Make predictions on test data
test_predictions = voting_regressor.predict(X_test_processed_df)

# Prepare submission file
submission = pd.DataFrame({'id': idtest, 'Price': test_predictions})
submission.to_csv("submission.csv", index=False)

# Display first few rows
print(submission.head())

