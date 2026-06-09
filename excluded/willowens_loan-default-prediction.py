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
        # print(f"\nTop 5 lines of {filename}:")
        
        # # Open and read the file
        # with open(os.path.join(dirname, filename), 'r') as file:
        #     # Read all lines and take first 5
        #     lines = file.readlines()[:5]
        #     # Print each line
        #     for i, line in enumerate(lines, 1):
        #         print(f"Line {i}: {line.strip()}")

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)

# Try to find the Home Credit Default Risk dataset
home_credit_paths = [path for path in os.listdir('/kaggle/input') if 'home-credit' in path.lower()]
print("\nPossible Home Credit paths:", home_credit_paths)


# Try the standard path first
data_path = '/kaggle/input/home-credit-default-risk/'
application_train = pd.read_csv(f'{data_path}/application_train.csv')
application_test = pd.read_csv(f'{data_path}/application_test.csv')


# Display the first few rows of the dataset and column names
print("application_train")
display(application_train.head())
print(application_train.columns.tolist())

rows, cols = application_train.shape
print(f"Rows: {rows}, Columns: {cols}")

# Total size (rows * columns)
total_size = application_train.size
print(f"Total size (elements): {total_size}")




# Loan Default Prediction - Imports and Setup

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm

# Set random seed for reproducibility
np.random.seed(42)


#Data Cleaning
# Calculate the percentage of NaN values in each column
nan_percentages = application_train.isna().mean() * 100

# Set a threshold for dropping columns (e.g., 50% missing values)
threshold = 50

# Identify columns with NaN percentage above the threshold
columns_to_drop = nan_percentages[nan_percentages > threshold].index

print(len(columns_to_drop))
# Print columns to drop and their NaN percentages
print("Columns to drop (>{}% NaN):".format(threshold))
for col in columns_to_drop:
    print(f"{col}: {nan_percentages[col]:.2f}% NaN")

# Drop the columns 
application_train_dropped = application_train.drop(columns=columns_to_drop)

# Verify the new shape
print(f"New DataFrame shape: {application_train_dropped.shape}")


# Check data types
print("\nData types in the dataset:")
print(application_train_dropped.dtypes.value_counts())

# Check for missing values
missing_values = application_train_dropped.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
missing_percent = (missing_values / len(application_train)) * 100

#We're also going to remove columns where more than 50% of the values are missing
# Identify columns where more than 50% of values are missing
threshold = 30
columns_to_drop = missing_percent[missing_percent > threshold].index
print(len(columns_to_drop))

# Drop columns with >50% missing values
application_train_dropped = application_train.drop(columns=columns_to_drop)
print(application_train_dropped.shape[1])
print("\nTop 40 features with missing values:")
missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percent
})
print(missing_df.head(40))

numerical_features = application_train.select_dtypes(include=['int64', 'float64']).columns.tolist()


print('Training Features shape: ', application_train_dropped.shape)

# one-hot encoding of categorical variables
app_train = pd.get_dummies(application_train_dropped)
numerical_df = app_train[numerical_features]

numerical_df.corr()

correlation_matrix = numerical_df.corr()
target_correlations = correlation_matrix["TARGET"]
best_predictors = target_correlations.abs().drop("TARGET").sort_values(ascending=False)

print(best_predictors)


print(application_train_dropped['TARGET'].value_counts())
application_train_dropped['TARGET'].astype(int).plot.hist();


# Now that we have a more manageable number, Let's take a look at the correlation matrix for the numerical values here
# Select only numerical features (int64 and float64) from application_train_dropped
numerical_features = application_train_dropped.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create a new DataFrame with only numerical features
numerical_df = application_train_dropped[numerical_features]

numerical_df.corr()


correlation_matrix = numerical_df.corr()
target_correlations = correlation_matrix["TARGET"]
best_predictors = target_correlations.abs().drop("TARGET").sort_values(ascending=False)

print(best_predictors)


print(numerical_df['DAYS_EMPLOYED'].describe())
numerical_df['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');


# Create an anomalous flag column
numerical_df['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
numerical_df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

numerical_df['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');

application_test['DAYS_EMPLOYED_ANOM'] = application_test["DAYS_EMPLOYED"] == 365243
application_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

print('There are %d anomalies in the test data out of %d entries' % (application_test["DAYS_EMPLOYED_ANOM"].sum(), len(application_test)))


# Extract the EXT_SOURCE variables and show correlations
ext_data = numerical_df[['TARGET', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
ext_data_corrs


plt.figure(figsize = (8, 6))

# Heatmap of correlations
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');


from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Drop the target from the training data
if 'TARGET' in numerical_df:
    train = numerical_df.drop(columns = ['TARGET'])
else:
    train = numerical_df.copy()
    
# Feature names
features = list(train.columns)
print(features)
# Median imputation of missing values
imputer = SimpleImputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)


# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)

print('Training data shape: ', train.shape)



from sklearn.linear_model import LogisticRegression

# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)

# Train on the training data
log_reg.fit(train, numerical_df["TARGET"])


# Make predictions
# Make sure to select the second column only
# Copy of the testing data

test = application_test.copy()
test = test[features]


test = imputer.transform(test)
test = scaler.transform(test)

print(test.shape)
log_reg_pred = log_reg.predict_proba(test)[:, 1]




# Submission dataframe
submit = application_test[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

submit.head()



# Save the submission to a csv file
submit.to_csv('log_reg_baseline.csv', index = False)


from sklearn.ensemble import RandomForestClassifier

# Make the random forest classifier
random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)


# Train on the training data
train_labels = application_train['TARGET']
random_forest.fit(train, train_labels)

# Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

# Make predictions on the test data
predictions = random_forest.predict_proba(test)[:, 1]


# Make a submission dataframe
submit = application_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

# Save the submission dataframe
submit.to_csv('random_forest_baseline.csv', index = False)


#Make a new dataframe for polynomial features
poly_features = application_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = application_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

# imputer for handling missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')

poly_target = poly_features['TARGET']

poly_features = poly_features.drop(columns = ['TARGET'])

# Need to impute missing values
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures
                                  
# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree = 3)


# Train the polynomial features
poly_transformer.fit(poly_features)

# Transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)


poly_transformer.get_feature_names_out(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]


# Create a dataframe of the features 
poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names_out(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Add in the target
poly_features['TARGET'] = poly_target

# Find the correlations with the target
poly_corrs = poly_features.corr()['TARGET'].sort_values()

# Display most negative and most positive
print(poly_corrs.head(10))
print(poly_corrs.tail(5))


# Put test features into dataframe
poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names_out(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Merge polynomial features into training dataframe
poly_features['SK_ID_CURR'] = application_train['SK_ID_CURR']
app_train_poly = app_train.merge(poly_features, on = 'SK_ID_CURR', how = 'left')

# Merge polnomial features into testing dataframe
poly_features_test['SK_ID_CURR'] =  application_test['SK_ID_CURR']
app_test_poly =  application_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

# Align the dataframes
app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)

# Print out the new shapes
print('Training data with polynomial features shape: ', app_train_poly.shape)
print('Testing data with polynomial features shape:  ', app_test_poly.shape)


poly_features_names = list(app_train_poly.columns)

# Impute the polynomial features
imputer = SimpleImputer(strategy = 'median')

poly_features = imputer.fit_transform(app_train_poly)
poly_features_test = imputer.transform(app_test_poly)

# Scale the polynomial features
scaler = MinMaxScaler(feature_range = (0, 1))

poly_features = scaler.fit_transform(poly_features)
poly_features_test = scaler.transform(poly_features_test)

random_forest_poly = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)


# Train on the training data
random_forest_poly.fit(poly_features, train_labels)

# Make predictions on the test data
predictions = random_forest_poly.predict_proba(poly_features_test)[:, 1]


# Make a submission dataframe
submit = application_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

# Save the submission dataframe
submit.to_csv('random_forest_engineered.csv', index = False)


corr_matrix = numerical_df.corr()


plt.figure(figsize=(100, 80))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# Select the upper triangle of the correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find index of feature columns with correlation greater than 0.9
high_corr = [(column, row, corr_matrix.loc[row, column])
             for column in upper.columns
             for row in upper.index
             if abs(upper.loc[row, column]) > 0.9]
print(high_corr)


corr_matrix = corr_matrix.abs()

# Select the upper triangle of the correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identify pairs of highly correlated variables
threshold = 0.8
to_drop = set()
for column in upper.columns:
    for row in upper.index:
        if upper.loc[row, column] > threshold:
            #  Compare missing value counts
            missing_row = numerical_df[row].isnull().sum()
            missing_col = numerical_df[column].isnull().sum()
            # Step 5: Drop the variable with more missing values
            if missing_row > missing_col:
                to_drop.add(row)
            else:
                to_drop.add(column)



#  Drop the identified columns from the DataFrame
app_train_df_reduced = numerical_df.drop(columns=to_drop)

print(f"Dropped columns due to high correlation and missing values: {to_drop}")


# just make sure we don't have any columns we don't want here
features = list(app_train_df_reduced.columns)
if 'TARGET' in features:
    features.remove('TARGET')
    print("removed target")
test = application_test.copy()
test = test[features]




# Define the features (excluding 'TARGET')
features = list(app_train_df_reduced.columns)
if 'TARGET' in features:
    features.remove('TARGET')

# Extract labels
train_labels = app_train_df_reduced['TARGET']

# Prepare train and test feature sets
train_raw = app_train_df_reduced[features]
test_raw = application_test[features]

# Impute missing values and preserve column names
imputer = SimpleImputer(strategy='median')
train = pd.DataFrame(imputer.fit_transform(train_raw), columns=features)
test = pd.DataFrame(imputer.transform(test_raw), columns=features)

# Train the Random Forest model
random_forest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
random_forest.fit(train, train_labels)

# Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({
    'feature': features,
    'importance': feature_importance_values
})

# Make predictions on the test data
predictions = random_forest.predict_proba(test)[:, 1]

# Optional: sort and view top features
print(feature_importances.sort_values(by='importance', ascending=False).head(10))


# Make a submission dataframe
submit = application_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

# Save the submission dataframe
submit.to_csv('random_forest_Colinear_drop.csv', index = False)




