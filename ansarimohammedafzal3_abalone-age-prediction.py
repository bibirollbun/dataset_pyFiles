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


pip install --upgrade scikit-learn


# data handling
import pandas as pd
import numpy as np

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# preprocessing and model selection
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder

# regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

#evaluation metric
from sklearn.metrics import root_mean_squared_log_error, make_scorer


# ignore unnecessary warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, 
                        message='use_inf_as_na option is deprecated')

warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='pandas.io.formats.format')

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# path for training dataset
train_path = '/kaggle/input/playground-series-s4e4/train.csv'
# path for testing dataset
test_path = '/kaggle/input/playground-series-s4e4/test.csv'

# removing the limit for viewing all rows
pd.set_option('display.max_rows', None)
# removing the limit for viewing all features
pd.set_option('display.max_columns', None)

SEED=42 # seed for reproducibility
SPLIT=0.3 # testing split
FOLDS = 10 # no. of folds for cross-validation


# loading the training dataset
df_train = pd.read_csv(train_path)
#loading the testing dataset
df_test = pd.read_csv(test_path)


df_train.head() # view first 5 rows of training dataset


# drop the id column
df_train = df_train.drop('id', axis=1)

test_ids = df_test['id']
df_test = df_test.drop('id', axis=1)


# vewing the shape of the training dataset
print('='*40)
print(f'Shape of training dataset: {df_train.shape}')
print('='*40)


# checking the training dataset for missing values
missing_values = df_train.isnull().sum()

print('='*50)
print('Features with missing values:')
print(missing_values[missing_values > 0].sort_values(ascending=False))
print('='*50)
print('Percentage (%) of missing values in each feature:')
print(missing_values[missing_values > 0].sort_values(ascending=False) / len(df_train * 100))
print('='*50)


# checking the training dataset for duplicate records
print('='*40)
print(f'Duplicate records in training dataset: {df_train.duplicated().sum()}')
print('='*40)


# viewing the columns in the training dataset
print('='*100)
print(f'Columns in training dataset: {df_train.columns}') # viewing the columns in the training dataset
print('='*100)


# checking the number of datatypes in training dataset
print('='*25)
print(df_train.dtypes.value_counts())
print('='*25)


# separating the numeric and categorical features
train_numeric = df_train.select_dtypes(include='number').columns
train_categorical = df_train.select_dtypes(include='object').columns


# checking summary statistics in training dataset
df_train[train_numeric].describe()


# checking the skewness of numeric features in training dataset
for feature in train_numeric:
    print('='*50)
    print(f'Skewness of {feature}: {df_train[feature].skew()}')
print('='*50)


# helper function for plotting feature distribution
def plot_distribution(df, feature):
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()


# checking distribution of numeric features in training dataset
for feature in train_numeric:
    plot_distribution(df_train, feature)


# checking for outliers using boxplot in numeric features in training dataset
fig, ax = plt.subplots(3, 3, figsize=(15, 12))
ax = ax.flatten()

for i, col in enumerate(train_numeric):
    sns.boxplot(data=df_train, y=col, ax=ax[i])
    ax[i].set_title(f'Boxplot of {col}')

for i in range(len(train_numeric), 9):
    ax[i].set_visible(False)
    
plt.tight_layout()
plt.show()


# classifying features which are normally distributed and not normally distributed
approx_normal_feats = df_train[train_numeric].skew().between(-0.5, 0.5)
not_normal_feats = ~df_train[train_numeric].skew().between(-0.5, 0.5)


# extracting the column names
z_score_cols = approx_normal_feats[approx_normal_feats].index.tolist()
iqr_cols = not_normal_feats[not_normal_feats].index.tolist()


# viewing skewed and approximately normally distributed features
print('='*60)
print('Skewed Features:')
print(iqr_cols)
print('='*60)
print('Approximately Normally Distributed Features:')
print(z_score_cols)
print('='*60) 


# helper function for detecting outliers using z-score method
def z_score_outliers(df, feature):
    if df[feature].std() == 0:
        return 0

    high = df[feature].mean() + 3 * df[feature].std()
    low = df[feature].mean() - 3 * df[feature].std()

    return df[(df[feature] > high) | (df[feature] < low)]


# checking the no. of outliers in normally distributed features
print('='*50)
print('No. of outliers in normally distributed features')
print('='*50)
for feature in z_score_cols:
    print(f'{feature}: {len(z_score_outliers(df_train, feature))}')

# checking the percentage of outliers in normally distributed features
print('='*50)
print('Precentage (%) of outliers in normally distributed features')
print('='*50)

for feature in z_score_cols:
    print(f'{feature}: {len(z_score_outliers(df_train, feature)) / len(df_train[feature]) * 100:.4f}%')

print('='*50)


# helper function for detecting outliers using IQR method
def iqr_outliers(df, feature):

    if df[feature].std() == 0:
        return 0

    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)

    IQR = Q3 - Q1
    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR

    return df[(df[feature] > high) | (df[feature] < low)]


# checking the no. of outliers in skewed features
print('='*50)
print('No. of outliers in skewed features')
print('='*50)

for feature in iqr_cols:
    print(f'{feature}: {len(iqr_outliers(df_train, feature))}')

# checking the percentage of outliers in normally distributed features
print('='*50)
print('Precentage (%) of outliers in skewed features')
print('='*50)

for feature in iqr_cols:
    print(f'{feature}: {len(iqr_outliers(df_train, feature)) / len(df_train[feature]) * 100:.4f}')

print('='*50)


# removing outliers from normally distributed features
outlier_indices = set()
for feature in z_score_cols:
    outlier_df = z_score_outliers(df_train, feature)
    outlier_indices.update(outlier_df.index)

# Remove these indices from the training set
df_train = df_train.drop(index=outlier_indices)


# capping the outliers in skewed features
for feature in iqr_cols:
    if feature != 'Rings': # excluding ring
        Q1 = df_train[feature].quantile(0.25)
        Q3 = df_train[feature].quantile(0.75)
        IQR = Q3 - Q1
        low = Q1 - 1.5 * IQR
        high = Q3 + 1.5 * IQR
        
        # capping training data
        df_train[feature] = np.clip(df_train[feature], low, high)
        # capping test data using SAME bounds
        df_test[feature] = np.clip(df_test[feature], low, high)


# log transform Rings feature
df_train['Rings'] = np.log1p(df_train['Rings'])


# checking distribution of Rings feature after applying log transformation
plot_distribution(df_train, 'Rings')


# checking correlation of numeric features in training dataset
corr = df_train[train_numeric].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.show()


# checking the values in Sex feature
print('='*30)
print(f"Value counts in {df_train['Sex'].value_counts()}")
print('='*30)


# one hot encoding the Sex feature
one_hot_encoder = OneHotEncoder()

sex_encoded = one_hot_encoder.fit_transform(df_train[['Sex']]).toarray()
feature_names = one_hot_encoder.get_feature_names_out(['Sex'])
df_train[feature_names] = sex_encoded

# applying the same encoding on the test dataset
test_encoded = one_hot_encoder.transform(df_test[['Sex']]).toarray()
df_test[feature_names] = test_encoded


# dropping the Sex feature after encoding
df_train = df_train.drop('Sex', axis=1)
df_test = df_test.drop('Sex', axis=1)


df_train.head()


df_test.head()


# data preprocessing
X = df_train.drop('Rings', axis=1)
y = df_train['Rings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT, random_state=SEED)


# scorer for hyper-parameter tuning and cross-validation
rmsle_scorer = make_scorer(root_mean_squared_log_error, greater_is_better=False)


# training DecisionTreeTegressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)

print(root_mean_squared_log_error(y_test, decision_tree.predict(X_test)))


# training RandomForestRegressor
random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)

print(root_mean_squared_log_error(y_test, random_forest.predict(X_test)))


# training XGBRegressor
xgboost_regressor = XGBRegressor()
xgboost_regressor.fit(X_train, y_train)

print(root_mean_squared_log_error(y_test, xgboost_regressor.predict(X_test)))


# training CatBoostRegressor
cat_boost = CatBoostRegressor()
cat_boost.fit(X_train, y_train)

print(root_mean_squared_log_error(y_test, cat_boost.predict(X_test)))


# training LGBMRegressor
light_gbm = LGBMRegressor(verbose=-1)
light_gbm.fit(X_train, y_train)

print(root_mean_squared_log_error(y_test, light_gbm.predict(X_test)))


# parameter grid for Light GBM
param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'num_leaves': [31, 50, 100, 200],
    'max_depth': [5, 10, 15, -1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1],
    'min_child_samples': [5, 10, 20]
}


# tuning and cross-validating LGBMRegressor
random_search = RandomizedSearchCV(light_gbm, param_grid, cv=FOLDS, scoring=rmsle_scorer, n_iter=500, 
                                   random_state=SEED, verbose=1)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
best_score = random_search.best_score_
best_params = random_search.best_params_

print('='*30)
print('Results for Light GBM:')
print(f"Best parameters: {best_params}")
print(f"Best CV score: {-best_score}") 
print('='*30)


# display test predictions
print('='*30)
print(f"Tuned LightGBM RMSLE: {root_mean_squared_log_error(y_test, random_search.predict(X_test))}")
print('='*30)


print('='*30)
print('Light GBM tuning completed.')
print('='*30)


# Light GBM with best hyper-parameters found
final_model = LGBMRegressor(
     subsample=1.0,
    reg_lambda=0.1, 
    reg_alpha=1,
    num_leaves=100,
    n_estimators=500,
    min_child_samples=10,
    max_depth=5,
    learning_rate=0.1,
    colsample_bytree=0.8,
    verbose=-1
)


# training Light GBM on the entire training dataset
final_model.fit(X, y)
predictions = final_model.predict(df_test)
final_predictions = np.expm1(predictions)


# saving test predictions for submissions
submission = pd.DataFrame({
    'id': test_ids,
    'Rings': final_predictions
})
submission.to_csv('submission.csv', index=False)

