# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('../input/playground-series-s5e5/train.csv', index_col = 'id')

print(train.shape)
train.head()


train.info()


train.describe()


num_col = train.drop('Calories', axis = 1).select_dtypes(exclude = 'object').columns
cat_col = train.select_dtypes(include = 'object').columns

print(num_col)
print(cat_col)


# plt.figure(figsize = (12,8))

# for i in range(len(num_col)):
#     plt.subplot(3,2, i+1)
#     sns.kdeplot(train[num_col[i]])

# plt.tight_layout()


# sns.histplot(train['Sex'])


# plt.figure(figsize = (12,12))

# for i in range(len(num_col)):
#     plt.subplot(3,2, i+1)
#     sns.boxplot(data = train, y=str(num_col[i]))

# plt.tight_layout()


# plt.figure(figsize = (12,12))

# for i in range(len(num_col)):
#     plt.subplot(3,2, i+1)
#     sns.regplot(data = train, x=str(num_col[i]), y = 'Calories')

# plt.tight_layout()


# num_corr = train[num_col].corr()
# plt.figure(figsize = (10,8))
# sns.heatmap(data = num_corr, linewidth = 1, annot=True)


corr_to_target = train.corr(numeric_only=True)['Calories'].sort_values(ascending=False)
corr_to_target


high_corr_col = ['Duration', 'Heart_Rate', 'Body_Temp']
low_corr_col = ['Height']


## Simple Feature Engineering
def featureEng(data, high_corr_col, low_corr_col):
    from itertools import combinations
    
    # for i in range(len(col)):
    #     data[f'Feature_Square_{col[i]}'] = data[col[i]] ** 2
    #     data[f'Feature_Cube_{col[i]}'] = data[col[i]] ** 3
    #     data[f'Feature_Log_{col[i]}'] = np.log(data[col[i]])
    #     data[f'Feature_Sqrt_{col[i]}'] = np.sqrt(data[col[i]])

    #Pairing features
    for i, j in combinations(high_corr_col,2):
        data[f'Feature_Pair_{i}_{j}'] = data[i] * data[j]

    #Drop unrelated
    data.drop(low_corr_col, axis=1, inplace=True)
    
    # optimize memory usage
    float_col = data.select_dtypes(include=['float64']).columns
    data[float_col] = data[float_col].astype('float32')

    data['Age'] = data['Age'].astype('int32')

    data['Sex'] = data['Sex'].astype('category')

featureEng(train, high_corr_col, low_corr_col)



train.head()


train.info()


featured_num_col = train.select_dtypes(exclude = ['object', 'category']).drop('Calories', axis = 1).columns 
featured_num_col

# featured_num_corr = train[featured_num_col].corr()
# mask = featured_num_corr.abs() > 0.8
# plt.figure(figsize = (20,15))
# sns.heatmap(data = featured_num_corr, linewidth = 1, annot=True, mask=mask)


# corr_to_target = train.corr(numeric_only=True)['Calories'].sort_values(ascending=False)
# corr_to_target


from sklearn.model_selection import train_test_split


X = train.copy()
y = np.log1p(X.pop('Calories'))

X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size =0.8, random_state =42 ) 
X_train


from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import  cross_val_score
import xgboost as xg
import optuna


ct = ColumnTransformer(
    [
        ('numscale', RobustScaler(), featured_num_col),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_col)
    ]
)

def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators',1,2000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42,
        'enable_categorical':True,
        'tree_method': 'hist',
    }

    pipeline = Pipeline(steps=
        [
            ('preprocessor', ct),
            ('model', xg.XGBRegressor(**xgb_params))
        ]
    )
 
    scores = cross_val_score(pipeline, X_train, y_train, 
                             scoring='neg_mean_squared_log_error', 
                             cv=5, n_jobs=-1)
    
    return -scores.mean()  

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, n_jobs=-1)

print("Best hyperparameters:", study.best_params)


final_model = xg.XGBRegressor(**study.best_params)

final_pipeline = Pipeline(steps=[
    ('preprocessor', ct), 
    ('model', final_model)
])

final_pipeline.fit(X_train, y_train)
y_pred = final_pipeline.predict(X_valid)

print(mean_squared_log_error(y_valid,y_pred))


test = pd.read_csv('../input/playground-series-s5e5/test.csv', index_col = 'id')

featureEng(test,high_corr_col,low_corr_col)

prediction = np.expm1(final_pipeline.predict(test))


submission_df = pd.DataFrame(data={'id': test.index, 'Calories': prediction})


submission_df.to_csv('submission.csv', index=False)

