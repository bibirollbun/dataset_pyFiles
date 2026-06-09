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





import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



train_df=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')




train_df.head()


train_df.tail()


train_df.shape


train_df.info()


len(train_df)


train_df.columns


train_df.index


train_df.describe()


test_df.head()


test_df.tail()


test_df.info()


test_df.shape


test_df.columns


len(test_df)


test_df.describe()



print("Train Data Null values",train_df.isnull().sum().sum())
print("Test Data Null values",test_df.isnull().sum().sum())




# Enter values in null coulmn

null_count=test_df.isnull().sum()
null_column=null_count[null_count>0]
if not null_count.empty:
    print("Null Column Name", null_column)







test_df['winddirection'].fillna(test_df['winddirection'].mean())
print("Missing values filled", test_df['winddirection'].isnull().sum())




# Replace infinite values with NaN in train and test datasets
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN values with median (to avoid plotting issues)
train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)

# Define numerical features
numerical_features = ['pressure', 'maxtemp', 'temparature', 'mintemp',
                      'dewpoint', 'humidity', 'cloud', 'sunshine', 
                      'winddirection', 'windspeed']

# Plot distributions
for feature in numerical_features:
    plt.figure(figsize=(10, 5))

    # Train Data Distribution
    plt.subplot(1, 2, 1)
    sns.histplot(train_df[feature], kde=True, color='blue', label='Train')
    plt.title(f'Train - Distribution of {feature}')
    
    # Test Data Distribution
    plt.subplot(1, 2, 2)
    sns.histplot(test_df[feature], kde=True, color='green', label='Test')
    plt.title(f'Test - Distribution of {feature}')
    
    plt.legend()
    plt.show()



for feature in numerical_features:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='rainfall', y=feature, data=train_df)
    plt.title(f'Train - {feature} vs. Rainfall')
    plt.tight_layout() 
    plt.show()


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(train_df[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title('Train - Correlation Heatmap')
plt.subplot(1, 2, 2)
sns.heatmap(test_df[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title('Test - Correlation Heatmap')
plt.show()


def create_rolling_features(df, windows, target_col='rainfall'):
    """Creates rolling window features for a DataFrame."""
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
    return df

windows = [3, 7]  # Create rolling windows for 3 and 7 days

train_df = create_rolling_features(train_df, windows)


# Analyze correlation with current rainfall
rolling_cols = [col for col in train_df.columns if 'rolling' in col]
correlation_matrix_rolling = train_df[['rainfall'] + rolling_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_rolling, annot=True, cmap='coolwarm')
plt.title("Correlation of Rolling Rainfall with Current Rainfall")
plt.show()


X = train_df.drop(columns=['rainfall', 'id'])  # Drop target and ID
y = train_df['rainfall']

X_test = test_df.drop(columns=['id'])  # Drop only ID



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Ensure X_test has the same columns as X_train
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
	X_test[col] = np.nan  # Add missing columns with NaN values

# Reorder columns in X_test to match X_train
X_test = X_test[X_train.columns]

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)



from sklearn.impute import SimpleImputer

# Handle missing values in the scaled data
imputer = SimpleImputer(strategy='mean')
X_train_scaled = imputer.fit_transform(X_train_scaled)
X_val_scaled = imputer.transform(X_val_scaled)

log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
log_reg_preds = log_reg.predict_proba(X_val_scaled)[:, 1]
log_reg_auc = roc_auc_score(y_val, log_reg_preds)
print(f"Logistic Regression AUC Score: {log_reg_auc}")



from sklearn.impute import SimpleImputer

# Create an imputer (replace NaN with the median value)
imputer = SimpleImputer(strategy="median")

# Fit and transform X_train and X_val
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)  # Also apply to test data

# Now train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict_proba(X_val)[:, 1]
rf_auc = roc_auc_score(y_val, rf_preds)

print(f"Random Forest AUC Score: {rf_auc}")




xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict_proba(X_val)[:, 1]
xgb_auc = roc_auc_score(y_val, xgb_preds)
print(f"XGBoost AUC Score: {xgb_auc}")



best_model = max(
    [('Logistic Regression', log_reg_auc), 
     ('Random Forest', rf_auc), 
     ('XGBoost', xgb_auc)], 
    key=lambda x: x[1])

print(f"Best Model: {best_model[0]} with AUC: {best_model[1]}")



# Generate final predictions using the best model (Random Forest)
final_preds = rf.predict_proba(X_test)[:, 1]  # Use the probability of the positive class

# Create submission DataFrame
submission = pd.DataFrame({'id': test_df['id'], 'rainfall': final_preds})

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")


