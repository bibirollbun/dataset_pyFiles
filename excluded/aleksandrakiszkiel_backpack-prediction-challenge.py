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


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import mean_squared_error


train_filepath = "/kaggle/input/playground-series-s5e2/train.csv"
test_filepath = "/kaggle/input/playground-series-s5e2/test.csv"

df = pd.read_csv(train_filepath, index_col = 'id')
test_df = pd.read_csv(test_filepath, index_col = 'id')


df.head()


test_df.head()


df.shape


test_df.shape


df.info()


df.describe()


df.isnull().sum()


df.nunique()


test_df.nunique()


print(f"Unique values in train data:")
for column in df.columns:
    print(f"Unique values in column {column}: {df[column].unique()}")


print(f"Unique values in test data:")
for column in test_df.columns:
    print(f"Unique values in column {column}: {test_df[column].unique()}")


continuous_cols = ['Weight Capacity (kg)', 'Price']
discrete_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']


for col in continuous_cols:
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    
    # Histogram
    sns.histplot(df[col], kde=True, discrete=True, edgecolor='black', ax=axes[0])
    axes[0].set_title(f'Histogram of {col}')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Frequency')
    
    # Boxplot
    sns.boxplot(x=df[col], ax=axes[1])
    axes[1].set_title(f'Boxplot of {col}')
    axes[1].set_xlabel(col)
    
    plt.tight_layout()
    plt.show()


for col in discrete_cols:
    print(f"Distribution for {col}:")
    print(df[col].value_counts())
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[col], palette="viridis", order=df[col].value_counts().index)
    plt.title(f'Countplot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


plt.figure(figsize=(10,6))
sns.scatterplot(x = 'Weight Capacity (kg)', y = 'Price', data = df, alpha = 0.5)
plt.title('Weight Capacity vs Price')
plt.xlabel('Weight Capacity (kg)')
plt.ylabel('Price')
plt.show()


X = df.drop('Price', axis=1)
y_train = df['Price']


X_train = X.copy()
X_test = test_df.copy()


num_cols = ['Compartments', 'Weight Capacity (kg)']
cat_cols = ['Brand', 'Material', 'Size', 'Style', 'Color']
binary_cols = ['Laptop Compartment', 'Waterproof']


num_imputer = SimpleImputer(strategy='mean')
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])


def proportional_imputation(series):

    distribution = series.value_counts(normalize=True)
    missing_mask = series.isnull()
    
    if missing_mask.sum() > 0:
        imputed_values = np.random.choice(distribution.index,
                                          size=missing_mask.sum(),
                                          p=distribution.values)
        series.loc[missing_mask] = imputed_values
    return series

X_train[binary_cols] = X_train[binary_cols].apply(proportional_imputation)
X_test[binary_cols] = X_test[binary_cols].apply(proportional_imputation)

X_train.isnull().sum()


binary_cols = ['Laptop Compartment', 'Waterproof']
X_train[binary_cols] = X_train[binary_cols].replace({'Yes': 1, 'No': 0})
X_test[binary_cols] = X_test[binary_cols].replace({'Yes': 1, 'No': 0})

X_train[binary_cols]


oe = OrdinalEncoder(categories=[['Small', 'Medium', 'Large']])

X_train['Size'] = oe.fit_transform(X_train[['Size']])
X_test['Size'] = oe.transform(X_test[['Size']])


X_train['Size']


cat_cols_without_size = ['Brand', 'Material', 'Style', 'Color']

ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(ohe.fit_transform(X_train[cat_cols_without_size]),
                             index=X_train.index, 
                             columns=ohe.get_feature_names_out(cat_cols_without_size))

OH_cols_test = pd.DataFrame(ohe.transform(X_test[cat_cols_without_size]),
                             index=X_test.index, 
                             columns=ohe.get_feature_names_out(cat_cols_without_size))


OH_cols_train


X_train = X_train.drop(columns=cat_cols_without_size)
X_test = X_test.drop(columns=cat_cols_without_size)


X_train = pd.concat([X_train, OH_cols_train], axis=1)
X_test = pd.concat([X_test, OH_cols_test], axis=1)


X_train.info()


model = XGBRegressor(tree_method='hist',device='cuda')

param_grid = {
    'n_estimators': [1000, 2000, 5000],
    'learning_rate': [0.05],
    'max_depth': [4, 5, 6]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs =1)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
best_mse = -grid_search.best_score_ 
print("Best score:", best_mse)
best_rmse = best_mse**0.5
print("Cross-validated RMSE:", best_rmse)
best_model = grid_search.best_estimator_


test_preds = best_model.predict(X_test)

submission = pd.DataFrame({
    'id': test_df.index,         
    'Price': test_preds      
})

# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)


