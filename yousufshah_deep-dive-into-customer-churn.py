# Data manipulation library
import pandas as pd
import numpy as np
import scipy.stats as stats # Maths library

# Visualization library
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Model building & data Preprocessing libraries
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier 
import joblib

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

print('Successfully Upload Neccessary Libraries')


# Load Datasets
df_train= pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv') # Train data
df_test= pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv') # Test data
df_submission= pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv') # Sample_Submission Data

print('Successfully load : Train, Test & Sample_Submission Data')


# Head of train data
df_train.head()


# Head of test data
df_test.head()


# Head of sample_submission data
df_submission.head()


# Shape of datasets
print('Train data, Rows =', df_train.shape[0], 'Columns =', df_train.shape[1], '\n')
print('Test data, Rows =', df_test.shape[0], 'Columns =', df_test.shape[1], '\n')
print('Submission data, Rows =', df_submission.shape[0], 'Columns =', df_submission.shape[1])


# Check for null values in train data
null_train = df_train.isnull().sum().sum()
print(f"Null values in Train Data = {null_train} \n")

# Check for null values in test data
null_test = df_test.isnull().sum().sum()
print(f"Null values in Test Data = {null_test} \n")

# Check for null values in submission data
null_submission = df_submission.isnull().sum().sum()
print(f"Null values in Sample Submission Data = {null_submission}")


# Find duplicate values in train data
duplicates_train = df_train.duplicated().sum()
print(f"Duplicate values in Train Data = {duplicates_train} \n")

# Find duplicate values in test data
duplicates_test = df_test.duplicated().sum()
print(f"Duplicate values in Test Data = {duplicates_test} \n")

# Find duplicate values in submission data
duplicates_submission = df_submission.duplicated().sum()
print(f"Duplicate values in Submission Data = {duplicates_submission}")


# First few rows
df_train.head()


# Shape of train data
print('Train data, Rows =', df_train.shape[0], 'Columns =', df_train.shape[1])


# Column names of train data
df_train.columns


# Info of train data
df_train.info()


# Statistical summary of train data without decimal places
df_train.describe().round(0)


# Drop unnecessary columns
df_train.drop(['id', 'CustomerId', 'Surname'], axis=1, inplace=True)

print("We have dropped 'id', 'CustomerId' & 'Surname'")


# check missing values
print(df_train.isnull().sum())

print('\nTrain data has no missing values')


# Check for duplicate rows 
duplicate_rows = df_train.duplicated().sum()
print(f"Duplicate Rows = {duplicate_rows}")


# Remove duplicate rows
df_train.drop_duplicates(inplace=True)
print("Duplicate rows have been removed")


# Check after removing duplicate rows
duplicate_rows = df_train.duplicated().sum()
print(f"Duplicate Rows = {duplicate_rows} \n")


# Numerical columns to plot
num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']

# Set up the figure and subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2 rows, 2 columns
fig.suptitle('Boxplots of Numerical Columns', fontsize=16, y=1.02, color='white')

# Set background colors
fig.patch.set_facecolor('#2b221f')  # Outer background color
for ax in axes.flat:
    ax.set_facecolor('#4945b5')  # Inner plot area background color

# Loop through numerical columns and plot vertical boxplots
for i, col in enumerate(num_cols):
    row = i // 2  # Row index (0 or 1)
    col_num = i % 2  # Column index (0 or 1)
    sns.boxplot(
        y=df_train[col],  # Vertical boxplot
        ax=axes[row, col_num],
        color='#3ba34e',  # Box color
        width=0.5,   # Adjust box width
        flierprops=dict(
            marker='D',  # Diamond-shaped outliers
            markersize=8,  # Size of outliers
            markerfacecolor='#000000',  # Red fill color
        )
    )
    axes[row, col_num].set_title(col, fontsize=14, color='white')
    axes[row, col_num].set_ylabel('')  # Remove y-axis label for clarity
    axes[row, col_num].tick_params(axis='x', colors='white')
    axes[row, col_num].tick_params(axis='y', colors='white')

# Adjust layout and display
plt.tight_layout()
plt.show()


# Function to find outliers using IQR
def find_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# Function to find outliers using Z-score
def find_outliers_zscore(data, column):
    z_scores = stats.zscore(data[column])
    abs_z_scores = np.abs(z_scores)
    outliers = data[abs_z_scores > 3]
    return outliers

# Columns to check for outliers
columns = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']

# Create a list to store the results
outliers_summary_list = []

# Find outliers for each column and store the results
for col in columns:
    iqr_outliers = find_outliers_iqr(df_train, col)
    zscore_outliers = find_outliers_zscore(df_train, col)
    outliers_summary_list.append({
        'Column': col,
        'IQR Outliers': iqr_outliers.shape[0],
        'Z-score Outliers': zscore_outliers.shape[0],
    })

# Convert the list to a DataFrame
outliers_summary = pd.DataFrame(outliers_summary_list)

# Add a total row
total_row = pd.DataFrame({
    'Column': ['Total'],
    'IQR Outliers': [outliers_summary['IQR Outliers'].sum()],
    'Z-score Outliers': [outliers_summary['Z-score Outliers'].sum()]
})

# Concatenate the total row to the summary DataFrame
outliers_summary = pd.concat([outliers_summary, total_row], ignore_index=True)

print('Outliers Summary Table of numerical columns using IQR & Z-score \n')

# Display the summary table
outliers_summary


# Before type Conversion
df_train.dtypes


# Convert 'Geography' and 'Gender' to category
df_train['Geography'] = df_train['Geography'].astype('category')
df_train['Gender'] = df_train['Gender'].astype('category')

# List of columns to convert to int64
columns_to_convert = ['Age', 'Balance', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Convert specified columns to int64
for col in columns_to_convert:
    df_train[col] = df_train[col].astype('int64')

print("Successfully Complete Type Conversion")


# After type conversion
df_train.dtypes


# After type conversion some duplicate rows are created

# Check for duplicate rows
duplicate_rows = df_train.duplicated().sum()
print(f"Duplicate Rows = {duplicate_rows}\n")

# Remove duplicate rows
df_train.drop_duplicates(inplace=True)
print("Duplicate rows have been removed")


# Product Usage Ratio
df_train['ProductUsageRatio'] = df_train['NumOfProducts'] / 4  

# Credit Score Group
df_train['CreditScoreGroup'] = pd.cut(df_train['CreditScore'], bins=[300, 579, 669, 739, 850], labels=['Poor', 'Fair', 'Good', 'Excellent'])

# Balance to Salary Ratio
df_train['BalanceSalaryRatio'] = (df_train['Balance'] / df_train['EstimatedSalary']).astype(int)

# Tenure to Age Ratio
df_train['TenureToAgeRatio'] = (df_train['Tenure'] / df_train['Age']).round(3)

# Age Group
df_train['AgeGroup'] = pd.cut(df_train['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])

# Balance and Tenure Interaction
df_train['BalanceTenureInteraction'] = (df_train['Balance'] * df_train['Tenure'])

# Tenure group
def bin_tenure(tenure):
    if tenure <= 2:
        return 'New'
    elif tenure <= 5:
        return 'Medium'
    else:
        return 'Long'
# Apply the function to create a new column 'TenureGroup'
df_train['TenureGroup'] = df_train['Tenure'].apply(bin_tenure)


# Financial Stability Index
df_train['FinancialStabilityIndex'] = ((df_train['Balance'] / df_train['Balance'].max()) + \
(df_train['CreditScore'] / df_train['CreditScore'].max()) + \
(df_train['EstimatedSalary'] / df_train['EstimatedSalary'].max())).round(3)

print('We Have Created 8 New Features')


# Select the columns for correlation calculation
columns_for_correlation = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'BalanceSalaryRatio', 'TenureToAgeRatio', 'BalanceTenureInteraction', 'FinancialStabilityIndex']

# Calculate the correlation matrix
correlation_matrix = df_train[columns_for_correlation].corr()

# Display the correlation matrix
correlation_matrix


# Set up the figure and axis
plt.figure(figsize=(12, 8))

# Create a heatmap with customizations
sns.heatmap(
    correlation_matrix,
    annot=True,  # Annotate the cells with correlation values
    fmt=".2f",  # Format the annotation to 2 decimal places
    cmap="coolwarm",  # Color map
    linewidths=0.5,  # Line width between cells
    linecolor='black',  # Line color between cells
    cbar_kws={"shrink": 0.8},  # Color bar customization
    square=True  # Make the cells square-shaped
)

# Set the title and labels
plt.title('Correlation Matrix Heatmap', fontsize=16, color='blue')
plt.xticks(rotation=45, ha='right', fontsize=12, color='black')
plt.yticks(rotation=0, fontsize=12, color='black')

# Show the plot
plt.show()


# Features to plot histograms
features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'BalanceSalaryRatio', 'TenureToAgeRatio', 'BalanceTenureInteraction', 'FinancialStabilityIndex']

# Set up the figure and subplots
fig, axes = plt.subplots(4, 2, figsize=(14, 16))  # 4 rows, 2 columns
fig.suptitle('Histograms of Numeric Features', fontsize=16, y=1.02, color='white')

# Set background colors
fig.patch.set_facecolor('#2b221f')  # Outer background color
for ax in axes.flat:
    ax.set_facecolor('#4945b5')  # Inner plot area background color

# Loop through features and plot histograms
for i, feature in enumerate(features):
    row = i // 2  # Row index (0, 1, 2, 3)
    col_num = i % 2  # Column index (0 or 1)
    sns.histplot(
        df_train[feature], 
        bins=30, 
        kde=True, 
        ax=axes[row, col_num], 
        color='#5c1111'
    )
    axes[row, col_num].set_title(feature, fontsize=14, color='white')
    axes[row, col_num].set_xlabel('')  # Remove x-axis label for clarity
    axes[row, col_num].set_ylabel('')  # Remove y-axis label for clarity
    axes[row, col_num].tick_params(axis='x', colors='white')
    axes[row, col_num].tick_params(axis='y', colors='white')

# Adjust layout and display
plt.tight_layout()
plt.show()


# Columns to calculate skewness and kurtosis
columns_to_analyze = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'BalanceSalaryRatio', 'TenureToAgeRatio', 'BalanceTenureInteraction', 'FinancialStabilityIndex']

print('skewness and kurtosis values close to zero are desirable \n')

# Calculate skewness and kurtosis for each column and log the messages
for col in columns_to_analyze:
    skewness = df_train[col].skew()
    kurtosis = df_train[col].kurtosis()
    print(f"Column: {col}")
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurtosis}\n")


# Categorical features to plot
categorical_features = ['Geography', 'Gender', 'CreditScoreGroup', 'AgeGroup', 'TenureGroup', 'Exited']

# Set up the figure and subplots
fig, axes = plt.subplots(3, 2, figsize=(14, 18))  # 3 rows, 2 columns
fig.suptitle('Countplot of Categorical Features', fontsize=16, y=1.02, color='white')

# Set background colors
fig.patch.set_facecolor('#2b221f')  # Outer background color
for ax in axes.flat:
    ax.set_facecolor('#fff')  # Inner plot area background color

# Loop through categorical features and plot countplots
for i, feature in enumerate(categorical_features):
    row = i // 2  # Row index (0, 1, 2)
    col_num = i % 2  # Column index (0 or 1)
    sns.countplot(
        x=df_train[feature], 
        ax=axes[row, col_num], 
        palette='viridis'
    )
    axes[row, col_num].set_title(feature, fontsize=14, color='white')
    axes[row, col_num].set_xlabel('')  # Remove x-axis label for clarity
    axes[row, col_num].set_ylabel('')  # Remove y-axis label for clarity
    axes[row, col_num].tick_params(axis='x', colors='white')
    axes[row, col_num].tick_params(axis='y', colors='white')

# Remove the last empty subplot
# fig.delaxes(axes[2, 1])

# Adjust layout and display
plt.tight_layout()
plt.show()


# Initialize StandardScaler
scaler = StandardScaler()

# Features to scale
features_to_scale = ['CreditScore', 'Balance', 'Age', 'EstimatedSalary', 'BalanceSalaryRatio', 'TenureToAgeRatio', 'BalanceTenureInteraction', 'FinancialStabilityIndex']

# Apply MinMax scaling
df_train[features_to_scale] = scaler.fit_transform(df_train[features_to_scale])

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Categorical features to encode
categorical_features = ['Geography', 'Gender', 'CreditScoreGroup', 'AgeGroup', 'TenureGroup']

# Apply label encoding
for feature in categorical_features:
    df_train[feature] = label_encoder.fit_transform(df_train[feature])

print("Scaling and encoding completed successfully")


# Separate features (X) and target (y)
X = df_train.drop(columns=['Exited'])
y = df_train['Exited']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE for Oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Original Class Distribution:", y.value_counts())
print("Resampled Class Distribution:", pd.Series(y_resampled).value_counts())

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define Models with Simplified Hyperparameters
models = [
    ('XGBoost', XGBClassifier(random_state=42), {
        'n_estimators': [100, 200],  
        'learning_rate': [0.05, 0.1],  
        'max_depth': [3, 5],  
        'subsample': [0.8, 1.0],  
        'colsample_bytree': [0.8, 1.0]  
    }),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42), {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1], 
        'max_depth': [3, 5],  
        'subsample': [0.8, 1.0]  
    }),
    ('CatBoost', CatBoostClassifier(random_state=42, verbose=0), {
        'iterations': [100, 200],  
        'learning_rate': [0.05, 0.1],  
        'depth': [3, 5]  
    }),
    ('AdaBoost', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), random_state=42), {
        'n_estimators': [50, 100], 
        'learning_rate': [0.05, 0.1]  
    }),
    ('LightGBM', lgb.LGBMClassifier(random_state=42, verbose=0), {
        'n_estimators': [100, 200], 
        'learning_rate': [0.05, 0.1],  
        'max_depth': [3, 5], 
        'subsample': [0.8, 1.0]  
    })
]

# Initialize Results List
results = []

# Loop Through Models and Perform Grid Search
for name, model, params in models:
    print(f"Processing {name}...")
    
    # Grid Search with Stratified Cross-Validation
    grid_search = GridSearchCV(
        model,
        param_grid=params,
        cv=skf,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_resampled, y_resampled)
    
    # Evaluate on Test Data
    y_pred = grid_search.predict(X_test)
    y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    # Store Results
    results.append({
        'Model': name,
        'Best Parameters': grid_search.best_params_,
        'Accuracy': accuracy,
        'AUC-ROC Score': auc_roc
    })
    
    print(f"{name}:")
    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  Accuracy: {accuracy}")
    print(f"  AUC-ROC Score: {auc_roc}")
    print()

# Convert Results to DataFrame and Sort by AUC-ROC
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='AUC-ROC Score', ascending=False)

# Print Results
print("Model Performance (Sorted by AUC-ROC Score):")
print(results_df)

# Print Best Model
best_model_result = results_df.iloc[0]
print("\nBest Model:")
print(f"  Model: {best_model_result['Model']}")
print(f"  Best Parameters: {best_model_result['Best Parameters']}")
print(f"  Accuracy: {best_model_result['Accuracy']}")
print(f"  AUC-ROC Score: {best_model_result['AUC-ROC Score']}")


# Save the best model to a .pkl file 
best_model = grid_search.best_estimator_
joblib.dump(best_model, '/kaggle/working/best_model.pkl')

print("Best model saved to best_model.pkl")


# Load the best model
best_model = joblib.load('/kaggle/working/best_model.pkl')

print("Best model loaded successfully")


# Load the test dataset
df_test = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')

print("Test data loaded successfully")


# Drop unnecessary columns
df_test.drop(['id', 'CustomerId', 'Surname'], axis=1, inplace=True)

# neccessary type conversion that we do with train data 
# Convert 'Geography' and 'Gender' to category
df_test['Geography'] = df_test['Geography'].astype('category')
df_test['Gender'] = df_test['Gender'].astype('category')
# List of columns to convert to int64
columns_to_convert = ['Age', 'Balance', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
# Convert specified columns to int64
for col in columns_to_convert:
    df_test[col] = df_test[col].astype('int64')

print('Data Cleaning Completed')


# Feature Engineering on test data
df_test['ProductUsageRatio'] = df_test['NumOfProducts'] / 4
df_test['CreditScoreGroup'] = pd.cut(df_test['CreditScore'], bins=[300, 579, 669, 739, 850], labels=['Poor', 'Fair', 'Good', 'Excellent'])
df_test['BalanceSalaryRatio'] = (df_test['Balance'] / df_test['EstimatedSalary']).astype(int)
df_test['TenureToAgeRatio'] = (df_test['Tenure'] / df_test['Age']).round(3)
df_test['AgeGroup'] = pd.cut(df_test['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])
df_test['BalanceTenureInteraction'] = (df_test['Balance'] * df_test['Tenure'])

def bin_tenure(tenure):
    if tenure <= 2:
        return 'New'
    elif tenure <= 5:
        return 'Medium'
    else:
        return 'Long'
df_test['TenureGroup'] = df_test['Tenure'].apply(bin_tenure)

df_test['FinancialStabilityIndex'] = ((df_test['Balance'] / df_test['Balance'].max()) + \
(df_test['CreditScore'] / df_test['CreditScore'].max()) + \
(df_test['EstimatedSalary'] / df_test['EstimatedSalary'].max())).round(3)

print('Feature Creation Completed')


# Preprocessing steps
scaler = StandardScaler()
features_to_scale = ['CreditScore', 'Balance', 'Age', 'EstimatedSalary', 'BalanceSalaryRatio', 'TenureToAgeRatio', 'BalanceTenureInteraction', 'FinancialStabilityIndex']
df_test[features_to_scale] = scaler.fit_transform(df_test[features_to_scale])

label_encoder = LabelEncoder()
categorical_features = ['Geography', 'Gender', 'CreditScoreGroup', 'AgeGroup', 'TenureGroup']
for feature in categorical_features:
    df_test[feature] = label_encoder.fit_transform(df_test[feature])

print('Successfully Preprocessed Data')


# Make predictions
predictions = best_model.predict(df_test)

print('Predictions made successfully From test.csv')


# Load the sample submission file
df_submission = pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv')

# Store the predicted values in the 'Exited' column
df_submission['Exited'] = predictions

# Save the updated submission file
df_submission.to_csv('/kaggle/working/my_second_submission.csv', index=False)

print("Predictions saved to my_second_submission.csv")

