import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Load train and test data
train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


#check train data shape. 11 columns, 300000 rows
train_data.shape


#check test data shape. 10 columns(no price column), 200000 rows
test_data.shape


train_data.head()


test_data.head()


#check for null values in train data
train_data.isnull().sum()


#check for null values in test data
test_data.isnull().sum()


#handle missing values in train data. Replace missing values with mode
train_data['Brand']=train_data['Brand'].fillna(train_data['Brand'].mode()[0])
train_data['Material']=train_data['Material'].fillna(train_data['Material'].mode()[0])
train_data['Size']=train_data['Size'].fillna(train_data['Size'].mode()[0])
train_data['Laptop Compartment']=train_data['Laptop Compartment'].fillna(train_data['Laptop Compartment'].mode()[0])
train_data['Waterproof']=train_data['Waterproof'].fillna(train_data['Waterproof'].mode()[0])
train_data['Style']=train_data['Style'].fillna(train_data['Style'].mode()[0])
train_data['Color']=train_data['Color'].fillna(train_data['Color'].mode()[0])
train_data['Weight Capacity (kg)']=train_data['Weight Capacity (kg)'].fillna(train_data['Weight Capacity (kg)'].mode()[0])


#handle missing values in test data. Replace missing values with mode
test_data['Brand']=test_data['Brand'].fillna(test_data['Brand'].mode()[0])
test_data['Material']=test_data['Material'].fillna(test_data['Material'].mode()[0])
test_data['Size']=test_data['Size'].fillna(test_data['Size'].mode()[0])
test_data['Laptop Compartment']=test_data['Laptop Compartment'].fillna(test_data['Laptop Compartment'].mode()[0])
test_data['Waterproof']=test_data['Waterproof'].fillna(test_data['Waterproof'].mode()[0])
test_data['Style']=test_data['Style'].fillna(test_data['Style'].mode()[0])
test_data['Color']=test_data['Color'].fillna(test_data['Color'].mode()[0])
test_data['Weight Capacity (kg)']=test_data['Weight Capacity (kg)'].fillna(test_data['Weight Capacity (kg)'].mode()[0])


# drop 'id' column in test data
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv").drop("id",axis=1)

from sklearn.model_selection import train_test_split

# Select subset of predictors to use for prediction
cols_to_use = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color', 'Weight Capacity (kg)']

# Convert the columns to type category
train_data[cols_to_use] = train_data[cols_to_use].astype("category")
test_data[cols_to_use] = test_data[cols_to_use].astype("category")

X = train_data[cols_to_use]

# Select target
y = train_data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# Using xgboost for prediction
from xgboost import XGBRegressor

my_model = XGBRegressor(enable_categorical = True,
                        device="cuda",
                        n_estimators=3000,
                        max_depth=9,
                        learning_rate=0.013,
                        n_jobs=4,
                        random_state=42,
                        subsample=0.9,
                        colsample_bytree=0.5,
                        min_child_weight=58,
                        reg_alpha=0.1,
                        reg_lambda=0.5
                        )

my_model.fit(X_train, y_train, 
             early_stopping_rounds=50, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)



from sklearn.metrics import mean_absolute_error

#calculate mean absolute error
predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


sample_submission["Price"] =  my_model.predict(test_data)
sample_submission.to_csv("submission.csv",index=False)
sample_submission

