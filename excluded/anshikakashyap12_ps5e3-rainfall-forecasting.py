# loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# checking if all the libraries imported successfully or not
print("Imported Successfully!")


# Loading the training data
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")

# Loading the testing data
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# checking if datasets have been loaded successfully
print("Datasets Loaded Successfully!")


# printing shapes of datasets
print(train.shape)
print(test.shape)


# going through data's top rows
train.head()


# going through data's last rows
train.tail()


# going through data's top rows
test.head()


# going through data's last rows
test.tail()


# describing dataset's basic information, i.e., info
train.info()


# checking numeric columns' statistics
train.describe()


# examining unique values of dataset
train.nunique()


# examining types of columns

# categorical columns
cat_col = train.select_dtypes(include='object').columns
print('Categorical columns :',cat_col)

# numerical columns
num_col = train.select_dtypes(exclude='object').columns
print('\nNumerical columns :',num_col)


# setting parameters for graphs
plt.figure(figsize=(12, 6))
plt.xticks(rotation=90)

# creating graphs

col_graph = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

for col in col_graph:
    sns.histplot(train[col], kde=True)
    plt.suptitle('Columns Visualization')
    plt.title(f"Distribution of {col}")
    plt.show()

# creating graphs for target column
sns.histplot(train['rainfall'], kde=True)
plt.title("Distribution of Target Column: Rainfall")
plt.show()


train.rename(columns={'temparature': 'temperature'}, inplace=True)
test.rename(columns={'temparature': 'temperature'}, inplace=True)
col_graph[2]='temperature'


# dropping id, day and rainfall columns in both datasets and saving them individually
id_col, day_col, rainfall_col = test['id'], test['day'], train['rainfall']

train.drop(['id', 'day', 'rainfall'], axis=1, inplace=True)
test.drop(['id', 'day'], axis=1, inplace=True)

# checking if columns dropped successfully or not
print('Columns dropped and saved separately!')


# creating function to create boxplots
def plot_boxplots(data, columns):
    for col in columns:
        plt.figure(figsize=(6, 4))
        plt.boxplot(data[col])
        plt.title(f'Boxplot of {col.capitalize()}')
        plt.xlabel(col.capitalize())
        plt.show()

# using the function to create boxplots to find outliers
plot_boxplots(train, col_graph)


# creating function that drops outliers
def outliers(train, col):
    q1 = train[col].quantile(0.03)
    q3 = train[col].quantile(0.97)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    train[col] = np.where(train[col] < lower_bound, lower_bound, train[col])
    train[col] = np.where(train[col] > upper_bound, upper_bound, train[col])
        
# updating num_col variable, as id, day and rainfall columns were dropped
num_col = train.select_dtypes(exclude='object').columns
print('Columns :',num_col)


# using the function to drop outliers
for col in num_col:
    outliers(train, col)
    outliers(test, col)
    
# checking if outliers dropped successfully or not
print('Outliers Dropped Successfully!')


# checking for possible null values in train dataset
train.isnull().sum()


# checking for possible null values in test dataset
test.isnull().sum()


# using SIMPLE IMPUTER to deal with missing values in test dataset
imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(test)
test_imputed = imputer.transform(test)
test_imputed_df = pd.DataFrame(test_imputed, columns=test.columns)


# checking for total of duplicated rows in train dataset
print(train.duplicated().sum())


# checking for total of duplicated rows in test dataset
print(test.duplicated().sum())


# getting column names of train dataset to do feature selection easily
train.columns


# separating data into features and target vairable

# features
X = train.columns

# target variable
Y = rainfall_col


# splitting dataset into 80:20 ratio
X_train, X_test, Y_train, Y_test = train_test_split(train[X], Y, test_size = 0.2, random_state = 42)


# initialising model
rfr = RandomForestRegressor(n_estimators=100, random_state=42, max_depth = 3)

# training model
rfr.fit(X_train, Y_train)

# predicting values using model
rfr_predictions = rfr.predict(X_test)

# saving predictions as whole numbers in a array
rfr_predictions = rfr_predictions.round().astype(int)


# initialising model
xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, max_depth = 3)

# training model
xgb.fit(X_train, Y_train)

# predicting values using model
xgb_predictions = xgb.predict(X_test)

# saving predictions as whole numbers in a array
xgb_predictions = xgb_predictions.round().astype(int)


# initialising model
lr = LinearRegression()

# training model
lr.fit(X_train, Y_train)

# predicting values using model
lr_predictions = lr.predict(X_test)

# saving predictions as whole numbers in a array
lr_predictions = lr_predictions.round().astype(int)


# printing evaluations of model

print("ROC AUC Scores:")

rfr_auc_score = roc_auc_score(Y_test, rfr_predictions)
print(f"Random Forest Regressor: {rfr_auc_score:.2f}")

xgb_auc_score = roc_auc_score(Y_test, xgb_predictions)
print(f"XGBRegressor: {xgb_auc_score:.2f}")

lr_auc_score = roc_auc_score(Y_test, lr_predictions)
print(f"Linear Regression: {lr_auc_score:.2f}")

print()

print("Classification Reports:")

print("Random Forest Regressor:")
print(classification_report(Y_test, rfr_predictions))

print("XGBRegressor:")
print(classification_report(Y_test, xgb_predictions))

print("Linear Regression:")
print(classification_report(Y_test, lr_predictions))


# making predictions
predictions_test = rfr.predict(test_imputed_df[X])

# creating a DataFrame to store the predictions along with IDs
result = pd.DataFrame({'id': id_col, 'rainfall': predictions_test})
print(result)

# saving the results to a CSV file
result.to_csv('submission.csv', index=False)


# # to clear kaggle working
# !rm -rf /kaggle/working/*

