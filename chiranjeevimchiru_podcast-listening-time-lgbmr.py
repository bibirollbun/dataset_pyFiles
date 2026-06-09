import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")


# Step 1: Load the datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


print(df_train.info())


# Print general info - - train dataset
#print(df_train.info())

# Identify columns with missing values
missing_cols = df_train.columns[df_train.isnull().any()]
print("\nColumns with missing values:")
print(missing_cols)

# Imputation
for col in missing_cols:
    if df_train[col].dtype in ['float64', 'int64']:  # Numeric types
        mean_value = df_train[col].mean()
        df_train[col].fillna(mean_value, inplace=True)
        print(f"Filled missing values in numeric column '{col}' with mean: {mean_value}")
    else:  # Object or categorical types
        mode_value = df_train[col].mode()[0]
        df_train[col].fillna(mode_value, inplace=True)
        print(f"Filled missing values in categorical column '{col}' with mode: {mode_value}")

# Verify no missing values now
print("\nMissing values after imputation:")
print(df_train.isnull().sum())


# Print general info - test dataset
#print(df_test.info())

# Identify columns with missing values
missing_cols = df_test.columns[df_test.isnull().any()]
print("\nColumns with missing values:")
print(missing_cols)

# Imputation
for col in missing_cols:
    if df_test[col].dtype in ['float64', 'int64']:  # Numeric types
        mean_value = df_test[col].mean()
        df_test[col].fillna(mean_value, inplace=True)
        print(f"Filled missing values in numeric column '{col}' with mean: {mean_value}")
    else:  # Object or categorical types
        mode_value = df_test[col].mode()[0]
        df_test[col].fillna(mode_value, inplace=True)
        print(f"Filled missing values in categorical column '{col}' with mode: {mode_value}")

# Verify no missing values now
print("\nMissing values after imputation:")
print(df_train.isnull().sum())



print(df_train.info())
print(df_test.info())


import matplotlib.pyplot as plt
import seaborn as sns

# List of other columns (excluding Listening_Time_minutes itself)
other_cols = [col for col in df_train.columns if col != 'Listening_Time_minutes']

# Set general plot style
sns.set(style="whitegrid")
plt.figure(figsize=(15, 5 * len(other_cols)))  # adjust figure size

# Loop over variables
for idx, col in enumerate(other_cols):
    plt.subplot(len(other_cols), 1, idx + 1)

    if df_train[col].dtype in ['float64', 'int64']:  # Numeric column
        # Scatter plot
        sns.scatterplot(x=col, y='Listening_Time_minutes', data=df_train)
        plt.title(f'Listening Time vs {col} (Scatter Plot)', fontsize=14)
        plt.xlabel(col)
        plt.ylabel('Listening_Time_minutes')
        
    else:  # Categorical column
        # Boxplot
        sns.boxplot(x=col, y='Listening_Time_minutes', data=df_train)
        plt.title(f'Listening Time by {col} (Box Plot)', fontsize=14)
        plt.xlabel(col)
        plt.ylabel('Listening_Time_minutes')
    
    plt.tight_layout()

plt.show()



# Import necessary libraries

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



# Separate target and features
X = df_train.drop(columns=['Listening_Time_minutes'])
y = df_train['Listening_Time_minutes']


# LightGBM requires numeric data, encode categorical if necessary
X = pd.get_dummies(X)
df_test = pd.get_dummies(df_test)

# Align train and test columns (important if some categories are missing in test set)
X, df_test = X.align(df_test, join='left', axis=1, fill_value=0)

# Split train set for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)




# Prepare datasets for LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid)



from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# Define model
model = LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=1000,
    verbose=-1
)

# Train model with early stopping using callbacks
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='rmse',
    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=100)]
)

# Predict
y_pred_valid = model.predict(X_valid)
y_pred_test = model.predict(df_test)

# Evaluate
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_valid, y_pred_valid, squared=False)
print(f"\nValidation RMSE: {rmse:.4f}")



# Predict the target variable on the df_test dataset
y_pred_test = model.predict(df_test)


# Create a submission dataframe
# Replace 'ID_column' with the actual name of the identifier column in df_test
submission = pd.DataFrame({
    'id': df_test['id'],  # Replace 'ID_column' with actual column name
    'Listening_Time_minutes': y_pred_test
})

# Save the submission dataframe as a CSV file
submission.to_csv('submission.csv', index=False)
submission.head(5)

