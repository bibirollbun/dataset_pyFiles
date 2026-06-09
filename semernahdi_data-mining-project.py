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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')



# Load the datasets with low_memory=False
train_data = pd.read_csv("/kaggle/input/microsoft-malware-prediction/train.csv", low_memory=False)
# test_data = pd.read_csv("/kaggle/input/microsoft-malware-prediction/test.csv", low_memory=False)

# Check the first few rows of the training data
# print(train_data.head())



test_data = pd.read_csv("/kaggle/input/microsoft-malware-prediction/test.csv", low_memory=False)


train = train_data
test = test_data


print("Training data shape:", train.shape)
print("Test data shape:", test.shape)


train = train.sample(n=1000000, random_state=42)
test = test.sample(n=1000000, random_state=42)


def downcast_columns(train):
    # Downcast integer columns
    int_cols = train.select_dtypes(include=['int']).columns
    train[int_cols] = train[int_cols].apply(pd.to_numeric, downcast='integer')

    # Downcast float columns
    float_cols = train.select_dtypes(include=['float']).columns
    train[float_cols] = train[float_cols].apply(pd.to_numeric, downcast='float')

    # Downcast non-numerical (object) columns to category
    obj_cols = train.select_dtypes(include=['object']).columns
    train[obj_cols] = train[obj_cols].apply(lambda col: col.astype('category'))

    return train


train_df = downcast_columns(train)
test_df = downcast_columns(test)


train.head()


train.describe().T


test.head()


test.describe().T


# # Plot histograms for numerical columns
# train.hist(bins=30, figsize=(20, 15))
# plt.tight_layout()
# plt.show()

# # KDE plots for more visual analysis of skewness
# for col in train.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns: # Removed 'category', 'object' from include
#     plt.figure()
#     sns.kdeplot(data=train[col], shade=True)
#     plt.title(f'Distribution of {col}')
#     plt.show()


print("Train: ",train.duplicated().any())
print("Test:  ",test.duplicated().any())


# Create a pie chart
counts = train['HasDetections'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=['No Detection (0)', 'Has Detection (1)'], autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue'])

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')
plt.title('Distribution of HasDetections (0 or 1)', fontsize=14)

# Show the plot
plt.show()



# Calculate the correlation matrix only for numerical columns
corr_matrix = train.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']).corr()

# Plot the heatmap
plt.figure(figsize=(30, 30))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')

# Add title
plt.title("Correlation Matrix", fontsize=18)

# Save the plot
plt.savefig('/kaggle/working/correlation_matrix.png', bbox_inches='tight')  # Saving to the working directory
# plt.close()  # Close the plot to avoid it displaying twice in the notebook

# print("Plot saved as /kaggle/working/correlation_matrix.png")



# Calculate the correlation matrix for numerical columns
corr_matrix = train.select_dtypes(include=['number']).corr()

# Set a threshold for high correlation (e.g., > 0.8 or < -0.8)
threshold = 0.8

# Create a mask to filter correlations above the threshold or below the negative threshold
high_corr_pairs = []

# Iterate through the correlation matrix to find pairs with high correlation
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

# Create a DataFrame to display the pairs
high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])

# Display the list of highly correlated feature pairs
print(high_corr_df)



# Calculate skewness and kurtosis for the target variable 'HasDetections'
print("Skewness: %f" % train['HasDetections'].skew())
print("Kurtosis: %f" % train['HasDetections'].kurt())


# Separate numerical and categorical columns
numerical_cols = train.select_dtypes(include=['int64', 'float64', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist()
categorical_cols = train.select_dtypes(include=['category']).columns.tolist()
#For test dataset
numerical_cols_test = test.select_dtypes(include=['int64', 'float64', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist()
categorical_cols_test = test.select_dtypes(include=['category']).columns.tolist()
print(f"Total numerical features: {len(numerical_cols)}")
print("Numerical Columns:", numerical_cols)


print(f"Total Categorical features: {len(categorical_cols)}")
print("Categorical Columns:", categorical_cols)


# Convert to sets to find differences
numerical_cols_set = set(numerical_cols)
numerical_cols_test_set = set(numerical_cols_test)

# Find the difference in both directions
columns_in_train_not_in_test = numerical_cols_set - numerical_cols_test_set
columns_in_test_not_in_train = numerical_cols_test_set - numerical_cols_set

# Print the differences
print(f"Columns in train dataset but not in test dataset: {columns_in_train_not_in_test}")
print(f"Columns in test dataset but not in train dataset: {columns_in_test_not_in_train}")



missing_values = train[numerical_cols].isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
print("\nMissing values in numerical features:")
print(missing_values)


# Calculate the percentage of missing values
missing_percentage = (missing_values / train.shape[0]) * 100

# Create a bar chart for missing data percentages
plt.figure(figsize=(12, 8))
missing_percentage.sort_values(ascending=False).plot(kind='bar', color='skyblue')

# Add titles and labels
plt.title('Percentage of Missing Values in Numerical Features', fontsize=16)
plt.xlabel('Feature', fontsize=14)
plt.ylabel('Percentage of Missing Values (%)', fontsize=14)
plt.xticks(rotation=90)
plt.show()



# Calculate the percentage of missing values for each column
missing_percentage = (missing_values / train.shape[0]) * 100

# Convert the series to a DataFrame for easier viewing
missing_percentage_df = pd.DataFrame({
    'Feature': missing_percentage.index,
    'Missing Percentage': missing_percentage.values
})

# Sort the dataframe by the missing percentage
missing_percentage_df = missing_percentage_df.sort_values(by='Missing Percentage', ascending=False)

# Display the missing percentage in a more readable format
print("Missing Data Percentage per Feature:")
print(missing_percentage_df)



def identify_missing_columns_by_threshold(train, threshold=0.5):
    num_rows = train.shape[0]
    missing_values = train.isnull().sum()
    high_missing_columns = missing_values[missing_values > threshold * num_rows].index.tolist()
    low_missing_columns = missing_values[missing_values <= threshold * num_rows].index.tolist()
    return high_missing_columns, low_missing_columns



# Get the high and low missing columns
high_missing_columns, low_missing_columns = identify_missing_columns_by_threshold(train[numerical_cols])


# Display the results with their lengths
print(f"Columns with more than 50% missing values ({len(high_missing_columns)} columns):")
print(high_missing_columns)

print(f"\nColumns with less than or equal to 50% missing values ({len(low_missing_columns)} columns):")
print(low_missing_columns)


train.drop(columns=['DefaultBrowsersIdentifier', 'Census_IsFlightingInternal', 'Census_ThresholdOptIn', 'Census_IsWIMBootEnabled'], inplace=True)
test.drop(columns=['DefaultBrowsersIdentifier', 'Census_IsFlightingInternal', 'Census_ThresholdOptIn', 'Census_IsWIMBootEnabled'], inplace=True)


# Ensure numerical columns only include those present in both train and test datasets
numerical_cols_train = train.select_dtypes(include=['int64', 'float64', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist()
numerical_cols_test = test.select_dtypes(include=['int64', 'float64', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist()

# Only use common columns between train and test
common_numerical_cols = list(set(numerical_cols_train) & set(numerical_cols_test))

# Check which columns have missing values in the training set
missing_columns = train[common_numerical_cols].isnull().sum()
missing_columns = missing_columns[missing_columns > 0]

# Impute with mean
from sklearn.impute import SimpleImputer

# Create an imputer object to replace missing values with the mean
imputer = SimpleImputer(strategy='mean')

# Apply the imputer to the training set
train[common_numerical_cols] = imputer.fit_transform(train[common_numerical_cols])

# Apply the same imputer to the test set
test[common_numerical_cols] = imputer.transform(test[common_numerical_cols])

# Verify that the missing values have been handled
print(f"Missing values in train set after imputation:\n{train[common_numerical_cols].isnull().sum()}")
print(f"\nMissing values in test set after imputation:\n{test[common_numerical_cols].isnull().sum()}")




# Create a scaler object
scaler = StandardScaler()

# Fit and transform the train data
train[common_numerical_cols] = scaler.fit_transform(train[common_numerical_cols])

# Transform the test data
test[common_numerical_cols] = scaler.transform(test[common_numerical_cols])

# Get descriptive statistics for both train and test data after scaling
train_desc = train[common_numerical_cols].describe().T
test_desc = test[common_numerical_cols].describe().T

# Print the statistics in a more readable format
print("Train Data After Scaling:")
print(train_desc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(2))

print("\nTest Data After Scaling:")
print(test_desc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(2))



# Calculate the correlation matrix for numerical columns
corr_matrix = train.select_dtypes(include=['number']).corr()

# Set a threshold for high correlation (e.g., > 0.8 or < -0.8)
threshold = 0.8
high_corr_matrix = corr_matrix[(corr_matrix > threshold) | (corr_matrix < -threshold)]

# Plot and save the heatmap
plt.figure(figsize=(30, 30))
sns.heatmap(high_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', mask=high_corr_matrix.isna())
plt.title("High Correlation Matrix", fontsize=18)
plt.savefig('high_correlation_matrix.png', bbox_inches='tight')  # Save the heatmap
plt.show()



# Calculate the correlation matrix for numerical columns
corr_matrix = train.select_dtypes(include=['number']).corr()

# Set a threshold for high correlation (e.g., > 0.8 or < -0.8)
threshold = 0.8

# Create a mask to filter correlations above the threshold or below the negative threshold
high_corr_pairs = []

# Iterate through the correlation matrix to find pairs with high correlation
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

# Create a DataFrame to display the pairs
high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])

# Display the list of highly correlated feature pairs
print(high_corr_df)



from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=0.95)  # Keep enough components to explain 95% of the variance
train_pca = pca.fit_transform(train[common_numerical_cols])
test_pca = pca.transform(test[common_numerical_cols])

explained_variance = pca.explained_variance_ratio_

print(f"Explained Variance by each principal component: {explained_variance}")



cumulative_variance = explained_variance.cumsum()
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()


print(f"Number of components selected: {pca.n_components_}")


print(f"Total Categorical features: {len(categorical_cols)}")
print("Categorical Columns:", categorical_cols)


# Get the high and low missing columns
high_missing_columns, low_missing_columns = identify_missing_columns_by_threshold(train[categorical_cols])
# Display the results with their lengths
print(f"Columns with more than 50% missing values ({len(high_missing_columns)} columns):")
print(high_missing_columns)


print(f"\nColumns with less than or equal to 50% missing values ({len(low_missing_columns)} columns):")
print(low_missing_columns)


# Calculate the percentage of missing values for each column
missing_percentage = train[categorical_cols].isnull().mean() * 100  # Percentage of missing values in each column
missing_percentage = missing_percentage.sort_values(ascending=False)  # Sort by descending order

# Display the percentage of missing values
print("Percentage of Missing Values in Each Column:")
print(missing_percentage)


# Visualizing missing values using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(train[categorical_cols].isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Categorical Columns')
plt.show()



train.drop(columns=['PuaMode', 'Census_ProcessorClass', 'Census_InternalBatteryType'], inplace=True)
test.drop(columns=['PuaMode', 'Census_ProcessorClass', 'Census_InternalBatteryType'], inplace=True)


categorical_cols = train.select_dtypes(include=['category']).columns.tolist()
#For test dataset
categorical_cols_test = test.select_dtypes(include=['category']).columns.tolist()



# Display distinct values in each categorical column
for col in categorical_cols:
    print(f"\nDistinct values in column '{col}':")
    print(train[col].value_counts())  # You can use value_counts to display frequency of each category
    print(f"Number of distinct values: {len(train[col].unique())}")



train.drop(columns=['MachineIdentifier'], inplace=True)
test.drop(columns=['MachineIdentifier'], inplace=True)


categorical_cols_test = test.select_dtypes(include=['category']).columns.tolist()
categorical_cols = train.select_dtypes(include=['category']).columns.tolist()
print(f"Total Categorical features: {len(categorical_cols)}")
print("Categorical Columns:", categorical_cols)



# Plotting the distribution of each categorical column
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=train[col], palette='Set2')
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()



missing_values = train[categorical_cols].isnull().sum()

# Print columns with missing values and their count
print(missing_values[missing_values > 0])


# Visualizing missing values using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(train[categorical_cols].isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Categorical Columns')
plt.show()



# Impute missing categorical values with mode
for col in categorical_cols:
    mode_value = train[col].mode()[0]
    train[col].fillna(mode_value, inplace=True)
    test[col].fillna(mode_value, inplace=True)  # Use the same mode from training data


# Verification: Check if there are any missing values remaining in the categorical columns
missing_train = train.isnull().sum()  # Count missing values in train data
missing_test = test.isnull().sum()    # Count missing values in test data

# Display the result
print("Missing values in train data after imputation:")
print(missing_train[missing_train > 0])  # Show columns with missing values in the training set

print("\nMissing values in test data after imputation:")
print(missing_test[missing_test > 0]) 


import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
# Initialize LabelEncoder and OneHotEncoder
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse=False, drop='first')


# Initialize Target Encoder (for target encoding)
target_encoder = ce.TargetEncoder(cols=['EngineVersion', 'AppVersion', 'AvSigVersion', 'OsVer', 'OsBuildLab', 
                                        'Census_OSVersion', 'Census_OSBranch', 'Census_OSSkuName', 
                                        'Census_OSInstallTypeName'])


one_hot_features = ['ProductName', 'Platform', 'Processor', 'SkuEdition', 'SmartScreen', 
                    'Census_MDC2FormFactor', 'Census_DeviceFamily', 'Census_PrimaryDiskTypeName', 
                    'Census_ChassisTypeName', 'Census_PowerPlatformRoleName', 'Census_OSEdition', 
                    'Census_OSSkuName', 'Census_OSInstallTypeName', 'Census_OSWUAutoUpdateOptionsName', 
                    'Census_GenuineStateName', 'Census_ActivationChannel', 'Census_FlightRing']


# Apply One-Hot Encoding
train_encoded = pd.get_dummies(train, columns=one_hot_features, drop_first=True)
test_encoded = pd.get_dummies(test, columns=one_hot_features, drop_first=True)


# 2. **Target Encoding for high cardinality features** (>10 unique categories)
# Apply Target Encoding to columns like 'EngineVersion', 'AppVersion', etc.
train_encoded[['EngineVersion', 'AppVersion', 'AvSigVersion', 'OsVer', 'OsBuildLab', 
               'Census_OSVersion', 'Census_OSBranch', 'Census_OSSkuName', 'Census_OSInstallTypeName']] = target_encoder.fit_transform(
    train[['EngineVersion', 'AppVersion', 'AvSigVersion', 'OsVer', 'OsBuildLab', 'Census_OSVersion', 
           'Census_OSBranch', 'Census_OSSkuName', 'Census_OSInstallTypeName']], train['HasDetections'])  

test_encoded[['EngineVersion', 'AppVersion', 'AvSigVersion', 'OsVer', 'OsBuildLab', 
              'Census_OSVersion', 'Census_OSBranch', 'Census_OSSkuName', 'Census_OSInstallTypeName']] = target_encoder.transform(
    test[['EngineVersion', 'AppVersion', 'AvSigVersion', 'OsVer', 'OsBuildLab', 'Census_OSVersion', 
          'Census_OSBranch', 'Census_OSSkuName', 'Census_OSInstallTypeName']])



# Example: Target Encoding Effect on a Column
plt.figure(figsize=(12, 6))
sns.boxplot(x=train['EngineVersion'], y=train['HasDetections'])
plt.title('Target Encoding for EngineVersion vs HasDetections')
plt.xticks(rotation=90)
plt.show()



print(f"Shape of train data after encoding: {train_encoded.shape}")
print(f"Shape of test data after encoding: {test_encoded.shape}")


# Get the columns in train but not in test
train_columns = set(train_encoded.columns)
test_columns = set(test_encoded.columns)

# Find the difference (columns in train but not in test)
columns_in_train_not_test = list(train_columns - test_columns)

# Display the first 5 columns that exist in train but not in test
print(columns_in_train_not_test[:5])
print(f"Total columns in train but not in test: {len(columns_in_train_not_test)}")



# Get the union of columns between train and test datasets
all_columns = train_encoded.columns.union(test_encoded.columns)

# Align columns for both datasets (adding missing columns with NaN for test)
train_encoded = train_encoded.reindex(columns=all_columns, fill_value=0)
test_encoded = test_encoded.reindex(columns=all_columns, fill_value=0)

print(f"Shape of train data after alignment: {train_encoded.shape}")
print(f"Shape of test data after alignment: {test_encoded.shape}")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib


y = train_encoded['HasDetections']
X = train_encoded.drop(columns=['HasDetections'])


# Split into train and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Display problematic columns
print("Columns with non-numeric data (causing issues with model fitting):")
print(non_numeric_cols)


for col in non_numeric_cols:
    print(f"\nFirst few values in column '{col}':")
    print(X[col].head())


categorical_cols = ['Census_OSArchitecture', 'OsPlatformSubRelease']  # List of categorical columns to encode
X_train_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(test_encoded, columns=categorical_cols, drop_first=True)


all_columns = X_train_encoded.columns.union(X_test_encoded.columns)
X_train_encoded = X_train_encoded.reindex(columns=all_columns, fill_value=0)
X_test_encoded = X_test_encoded.reindex(columns=all_columns, fill_value=0)


print(f"Shape of train data after alignment: {X_train_encoded.shape}")
print(f"Shape of test data after alignment: {X_test_encoded.shape}")


y = train_encoded['HasDetections']
X = train_encoded.drop(columns=['HasDetections'])


from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import randint


X_train, X_val, y_train, y_val = train_test_split(X_train_encoded, y, test_size=0.2, random_state=42)


rf_model = RandomForestClassifier(random_state=42, n_jobs=3) 
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_val)


print("Accuracy on validation set:", accuracy_score(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred))


print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))


# 4. Hyperparameter Tuning using RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt', 'log2']
}


grid_search = RandomizedSearchCV(rf_model, param_distributions=param_grid, n_iter=3, cv=3, n_jobs=-1, random_state=42, verbose=1)
# grid_search.fit(X_train, y_train)


X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, test_size=0.6, random_state=42)
grid_search.fit(X_train_sub, y_train_sub)


rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)


# Best Hyperparameters and Best Model
print("Best Hyperparameters:", grid_search.best_params_)
best_rf_model = grid_search.best_estimator_


# 5. Evaluating the Tuned Model
y_pred_tuned = best_rf_model.predict(X_val)
print("Accuracy on validation set with tuned model:", accuracy_score(y_val, y_pred_tuned))
print("\nClassification Report with tuned model:")
print(classification_report(y_val, y_pred_tuned))


# 6. Cross-Validation on the Training Set
cv_scores = cross_val_score(best_rf_model,X_train_sub, y_train_sub, cv=3, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())


feature_importances = best_rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)


print("Top 10 Important Features:")
print(feature_importance_df.head(10))


# 8. Confusion Matrix and ROC-AUC for Model Evaluation
cm = confusion_matrix(y_val, y_pred_tuned)
print("Confusion Matrix:")
print(cm)


# ROC-AUC Score
auc = roc_auc_score(y_val, best_rf_model.predict_proba(X_val)[:, 1])
print("ROC-AUC Score:", auc)


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, best_rf_model.predict_proba(X_val)[:, 1])
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Confusion matrix values
confusion_matrix = np.array([[63956, 36136],
                             [35083, 64825]])

# Create a heatmap for the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=["Actual 0", "Actual 1"], yticklabels=["Predicted 0", "Predicted 1"])
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Confusion Matrix')
plt.show()


