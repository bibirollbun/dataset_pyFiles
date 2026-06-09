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


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error


predictions = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
train_df.drop('id', axis=1,inplace=True)
train_df.head(5)


test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test_df.drop('id', axis=1, inplace=True)
test_df.head(5)


print(f'Missing values in Training data:\n{train_df.isnull().sum()}\n')
print(f'Missing values in test data:\n{test_df.isnull().sum()}')


print("Training data info:")
train_df.info()
print('\t')
print('-' * 40)
print('\t')
print("Test data info:")
test_df.info()


fig, ax = plt.subplots(1,2,figsize=(10,5))
sns.boxplot(data=train_df[train_df['Sex']=='male'], x = train_df[train_df['Sex']=='male']['Calories'], palette ='crest', ax=ax[0])
ax[0].set_title('Distribution of Calories in Male')
sns.boxplot(data=train_df[train_df['Sex']=='female'], x = train_df[train_df['Sex']=='female']['Calories'], palette ='flare', ax=ax[1])
ax[1].set_title('Distribution of Calories in Female')
plt.tight_layout()
plt.show()


train_df.describe()


numerical_df_train = train_df.select_dtypes(include=[np.number])
numerical_df_test = test_df.select_dtypes(include=[np.number])


fig, ax = plt.subplots(1,2, figsize = (15,5))
sns.heatmap(data=numerical_df_train.corr(), annot=True, ax=ax[0])
ax[0].set_title(f'Correlation plot of Training data')
sns.heatmap(data=numerical_df_test.select_dtypes(include=[np.number]).corr(), annot=True, ax=ax[1])
ax[1].set_title(f'Correlation plot of Test data')
plt.tight_layout()
plt.show()


cols = ['Duration','Heart_Rate', 'Body_Temp']
for i in cols:
    fig = plt.figure(figsize=(15,5))
    sns.regplot(data=train_df[cols], x=i, y=train_df['Calories'], scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
    plt.title(f'Scatter Plot of {i} and Calories in Training data')
    plt.tight_layout()
    plt.show()


for i in numerical_df_train.columns:
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    sns.boxplot(data = numerical_df_train, x=i, ax=ax[0], palette = 'crest') 
    ax[0].set_title(f'Distribution of {i}')
    sns.histplot(data=numerical_df_train, x=i, kde=True, bins = 10, ax=ax[1], color="green")
    ax[1].set_title(f'Histogram and KDE Dist of {i}')
    plt.tight_layout()
    plt.show()


train_skewness = numerical_df_train.skew().reset_index()
train_skewness.columns = ['Column', 'Train Skewness']
test_skewness = numerical_df_test.skew().reset_index()
test_skewness.columns = ['Column', 'Test Skewness']

skew_df = pd.merge(train_skewness, test_skewness, on='Column', how = 'outer')
skew_df = skew_df.sort_values(by='Train Skewness', ascending=False).reset_index(drop=True)
skew_df


train_df['Calories'] = np.log1p(train_df['Calories'])


train_df['BMI'] = train_df['Weight'] / ((train_df['Height'] / 100) ** 2)
test_df['BMI'] = test_df['Weight'] / ((test_df['Height'] / 100) ** 2)
train_df['Duration by Age'] = train_df['Duration'] / train_df['Age']
test_df['Duration by Age'] = test_df['Duration'] /train_df['Age']
train_df['Duration by Weight'] = train_df['Duration'] / train_df['Weight']
test_df['Duration by Weight'] = test_df['Duration'] / test_df['Weight']


sns.heatmap(train_df.select_dtypes(include=['int','float']).corr())


col_order_train = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Duration by Age', 'Duration by Weight', 'Heart_Rate', 'Body_Temp','BMI' ,'Calories']
train_df = train_df[col_order_train] 
col_order_test = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Duration by Age', 'Duration by Weight', 'Heart_Rate', 'Body_Temp','BMI']
test_df = test_df[col_order_test]


for col in train_df.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.fit_transform(test_df[col])


X_train = train_df.iloc[:,:-1].values
X_test = test_df.iloc[:,:].values
y_train = train_df['Calories'].values


y_binned = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
     'LightGBM': {
        'model': LGBMRegressor(objective='regression', random_state=42),
        'params': {'n_estimators': [100, 200], 'max_depth': [-1, 10], 'learning_rate': [0.1]}
    },
    'XGBoost': {
        'model': XGBRegressor(objective='reg:squaredlogerror', eval_metric='rmsle', random_state=42),
        'params': {'n_estimators': [100, 200], 'max_depth': [5, 10], 'learning_rate': [0.1]}
    },
    'CatBoost': {
        'model': CatBoostRegressor(eval_metric='RMSE', verbose=0, random_state=42),
        'params': {'iterations': [100, 200], 'depth': [4, 6], 'learning_rate': [0.1]}
    }
}

results = []

for model, model_algo in models.items():
    print(f'GridSearchCV for {model}... \n')

    grid = GridSearchCV(estimator=model_algo['model'],
                        param_grid=model_algo['params'],
                        cv=stratified_cv.split(X_train, y_binned),
                        scoring='neg_mean_squared_log_error',
                        n_jobs=-1,
                        verbose=0)
    
    grid.fit(X_train, y_train) 
    
    best_model = grid.best_estimator_
    cv_msle = abs(grid.best_score_)
    cv_rmsle = round(np.sqrt(cv_msle), 4)
    print(cv_rmsle)

    results.append({
        'Model': model ,
        'Best Paramaters': grid.best_params_ ,
        'Cross Validation RMSLE': cv_rmsle
    })

results_df = pd.DataFrame(results).sort_values(by='Cross Validation RMSLE', ascending = True)


results_df


min_rmsle = results_df['Cross Validation RMSLE'].min()
best_models = results_df[results_df['Cross Validation RMSLE'] == min_rmsle]
for _, row in best_models.iterrows():
    print(f"Model: {row['Model']}")
    print(f"Best Params: {row['Best Paramaters']}")
    print(f"CV RMSLE: {row['Cross Validation RMSLE']}")
    print('-' * 40)


xgb_model = XGBRegressor(objective='reg:squarederror',
                                     eval_metric='rmse',
                                     n_estimators=100,
                                     learning_rate= 0.1,
                                     random_state=42,
                                     n_jobs=-1,
                                     colsample_bytree=0.7,
                                     subsample=0.7,
                                     max_depth=10)

xgb_model.fit(X_train, y_train, verbose=False)
y_pred_xgboost = xgb_model.predict(X_test)
feature_imp_df = pd.DataFrame({'Feature':test_df.columns, 'Importance': xgb_model.feature_importances_})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending =False)


plt.figure(figsize=(15,5))
sns.barplot(data=feature_imp_df, y='Feature', x='Importance', palette='flare', orient = 'h')
plt.title('Feature Importance of XGBoost Model')
plt.tight_layout()
plt.show()


y_pred = np.expm1(y_pred_xgboost)
predictions['Calories'] = y_pred
predictions.to_csv("submission.csv", index=False)
predictions.head(10)


print(f'Predictions Mean: {np.mean(y_pred):.3f}')
print(f'Predictions Median: {np.median(y_pred):.3f}')

