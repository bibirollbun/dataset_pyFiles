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
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.impute import SimpleImputer
import supplemental_english as supplement


# Load data
train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')


train.head()


train.info()


train.describe()


train.isnull().sum()


# Feature engineering function
def extract_plate_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract plate components
    df['prefix'] = df['plate'].str.extract(r'^([A-ZĞ�-Ğ¯]{1,3})')[0]
    df['number'] = df['plate'].str.extract(r'(\d{3})')[0].astype(int)
    df['region_code'] = df['plate'].str.extract(r'(\d{2,3})$')[0].astype(int)
    
    # Date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Special plate indicators
    df['is_gov'] = df['prefix'].isin(supplement.GOVERNMENT_CODES).astype(int)
    df['is_000'] = (df['number'] == 0).astype(int)
    df['is_repdigit'] = (df['number'].astype(str).str[0] == df['number'].astype(str).str[1]) & \
                       (df['number'].astype(str).str[1] == df['number'].astype(str).str[2]).astype(int)
    
    # Popular regions (Moscow, St. Petersburg)
    df['is_moscow'] = (df['region_code'] == 77).astype(int)
    df['is_spb'] = (df['region_code'] == 78).astype(int)
    
    return df


# Apply feature engineering
train = extract_plate_features(train)
test = extract_plate_features(test)


# Define features and target
X = train.drop(['id', 'plate', 'date', 'price'], axis=1)
y = train['price']
X_test = test.drop(['id', 'plate', 'date', 'price'], axis=1)


# Identify categorical and numerical features
categorical_features = ['prefix', 'region_code']
numerical_features = [col for col in X.columns if col not in categorical_features]


# Price distribution
plt.figure(figsize=(12, 6))
sns.histplot(train['price'], bins=50, kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.yscale('log')  # Log scale due to long tail
plt.show()


# Price over time
plt.figure(figsize=(12, 6))
train.groupby('year')['price'].median().plot()
plt.title('Median Price by Year')
plt.ylabel('Price')
plt.show()


# Price by region (top 20)
plt.figure(figsize=(12, 6))
train.groupby('region_code')['price'].median().sort_values(ascending=False).head(20).plot(kind='bar')
plt.title('Median Price by Region (Top 20)')
plt.ylabel('Price')
plt.show()


# Price by prefix
plt.figure(figsize=(12, 6))
train.groupby('prefix')['price'].median().sort_values(ascending=False).head(20).plot(kind='bar')
plt.title('Median Price by Prefix (Top 20)')
plt.ylabel('Price')
plt.show()


# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(train[numerical_features + ['price']].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


# Identify price outliers using IQR
Q1 = train['price'].quantile(0.25)
Q3 = train['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = train[(train['price'] < lower_bound) | (train['price'] > upper_bound)]
print(f"Found {len(outliers)} price outliers ({len(outliers)/len(train):.2%} of data)")


# Visualize outliers
plt.figure(figsize=(12, 6))
sns.boxplot(x=train['price'])
plt.title('Price Boxplot (Showing Outliers)')
plt.show()


# Option 1: Cap outliers
train['price'] = train['price'].clip(lower_bound, upper_bound)


# Option 2: Remove outliers (uncomment to use)
# train = train[(train['price'] >= lower_bound) & (train['price'] <= upper_bound)]


# Define preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])


# Define SMAPE scoring (competition metric)
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

smape_scorer = make_scorer(smape, greater_is_better=False)


# List of models to try
models = {
    #'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42)
}


# Evaluate models with cross-validation
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)])
    
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring=smape_scorer)
    results[name] = -cv_scores.mean()  # Convert back to positive SMAPE
    
    print(f"{name}: Mean SMAPE = {-cv_scores.mean():.2f} (Â±{cv_scores.std():.2f})")




# Plot model comparison
plt.figure(figsize=(10, 6))
pd.Series(results).sort_values().plot(kind='barh')
plt.title('Model Comparison (Lower SMAPE is better)')
plt.xlabel('SMAPE Score')
plt.show()


# Let's tune the best performing model (Random Forest in this case)
param_grid = {
    #'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2]
}

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))])

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=smape_scorer, n_jobs=-1)
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best SMAPE score: {-grid_search.best_score_:.2f}")


# Train final model with best parameters
best_model = grid_search.best_estimator_


# Get feature names after one-hot encoding
feature_names = numerical_features.copy()
for col in categorical_features:
    categories = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].categories_[categorical_features.index(col)]
    feature_names.extend([f"{col}_{cat}" for cat in categories])


# Get feature importances
importances = best_model.named_steps['model'].feature_importances_


# Create feature importance dataframe
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False).head(20)


# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title('Top 20 Feature Importances')
plt.show()


# Make predictions
test_preds = best_model.predict(X_test)


# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'price': test_preds
})


# Ensure no negative prices (though unlikely)
submission['price'] = submission['price'].clip(0)

submission.to_csv('submission.csv', index=False)

