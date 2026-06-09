# Basic Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

# Preprocessing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LassoCV
import lightgbm as lgb

# Evaluation
from sklearn.metrics import mean_squared_error

# Warnings
import warnings
warnings.filterwarnings('ignore')



# Reading the provided train and test datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

train.head()



# Check data types and missing values
train.info()

# Quick Summary Stats
train.describe()



# Missing Values in train
missing_values = train.isnull().sum()
missing_values = missing_values[missing_values > 0]
print(missing_values)



# Missing Values in test
missing_values = test.isnull().sum()
missing_values = missing_values[missing_values > 0]
print(missing_values)



from sklearn.impute import SimpleImputer

# Columns to impute
to_impute = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']

# Create Imputer
imputer = SimpleImputer(strategy='median')

# Fit only on train, apply on train and test
train[to_impute] = imputer.fit_transform(train[to_impute])
test[to_impute] = imputer.transform(test[to_impute])

# Re-check missing values
print("Train missing values after imputation:")
print(train.isnull().sum().sum())

print("Test missing values after imputation:")
print(test.isnull().sum().sum())



plt.figure(figsize=(8,5))
sns.histplot(train['Listening_Time_minutes'], bins=30, kde=True)
plt.title('Distribution of Listening Time (Target)')
plt.xlabel('Listening Time (minutes)')
plt.ylabel('Frequency')
plt.show()



numerical_cols = train.select_dtypes(include=np.number).columns.tolist()
numerical_cols.remove('id')

train[numerical_cols].hist(figsize=(8,8), bins=30)
plt.suptitle('Numerical Feature Distributions', fontsize=20)
plt.show()



categorical_cols = train.select_dtypes(include='object').columns.tolist()

print("Categorical Columns:", categorical_cols)

for col in categorical_cols:
    plt.figure(figsize=(20,6))
    sns.countplot(data=train, x=col, order=train[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()



# Correlation with Target
numeric_only_cols = train.select_dtypes(include=np.number).columns.tolist()
numeric_only_cols.remove('id')

corr = train[numeric_only_cols].corr()
corr_target = corr['Listening_Time_minutes'].sort_values(ascending=False)

print(corr_target)

# Plot
plt.figure(figsize=(6,6))
corr_target.drop('Listening_Time_minutes').plot(kind='bar')
plt.title('Feature Correlation with Listening Time')
plt.show()



# Extract episode number from title
train['Episode_Number'] = train['Episode_Title'].str.extract(r'(\d+)').fillna(0).astype(int)
test['Episode_Number'] = test['Episode_Title'].str.extract(r'(\d+)').fillna(0).astype(int)

# Ads per minute
train['Ads_per_minute'] = train['Number_of_Ads'] / (train['Episode_Length_minutes'] + 1e-5)
test['Ads_per_minute'] = test['Number_of_Ads'] / (test['Episode_Length_minutes'] + 1e-5)



# Drop text columns
train = train.drop(columns=['id', 'Episode_Title'])
test_ids = test['id']
test = test.drop(columns=['id', 'Episode_Title'])

# Combine train + test for consistent encoding
full = pd.concat([train.drop(columns='Listening_Time_minutes'), test], axis=0)
categorical_cols = full.select_dtypes(include='object').columns.tolist()

# One-hot encode
full_encoded = pd.get_dummies(full, columns=categorical_cols)

# Separate back
X_train_full = full_encoded.iloc[:len(train), :].copy()
X_test_full = full_encoded.iloc[len(train):, :].copy()
y_train = train['Listening_Time_minutes'].values



# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test_full)

# Create interactions
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
X_test_poly = poly.transform(X_test_scaled)



from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
# Keep top 300 best features based on correlation with target
selector = SelectKBest(score_func=f_regression, k=300)
X_poly_selected = selector.fit_transform(X_poly, y_train)
X_test_selected = selector.transform(X_test_poly)

print(f"Original Shape: {X_poly.shape}")
print(f"After SelectKBest: {X_poly_selected.shape}")


model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_poly_selected, y_train)


preds = model.predict(X_test_selected)
submission = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': preds})
submission.to_csv('submission.csv', index=False)
submission.head()

