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


TRAIN_PATH = r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv"
TEST_PATH  = r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv"

import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import optuna
from lightgbm import LGBMRegressor # Using LightGBM

df = pd.read_csv(TRAIN_PATH)
df1 = pd.read_csv(TEST_PATH)


df.info()
df1.info()
df.describe()
df.isnull().mean() * 100



numericalcols = df.select_dtypes(['int64', 'float64'])
categoricalcols = df.select_dtypes(['object'])
plt.figure(figsize=(20, 15))
sns.heatmap(numericalcols.corr(), cmap='YlGnBu', annot=False)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['CORRUCYSTIC_DENSITY'])
plt.title('Box Plot of CORRUCYSTIC_DENSITY')

plt.subplot(1, 2, 2)
sns.kdeplot(df['CORRUCYSTIC_DENSITY'], fill=True)
plt.title('KDE Plot of CORRUCYSTIC_DENSITY')
plt.show()


dfclean = df.dropna(subset=['CORRUCYSTIC_DENSITY'])
x = dfclean.drop('CORRUCYSTIC_DENSITY', axis=1)
y = dfclean['CORRUCYSTIC_DENSITY']


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)





numericfeatures = [col for col in xtrain.columns if xtrain[col].dtype in ['int64', 'float64']]
categoricfeatures = [col for col in xtrain.columns if xtrain[col].dtype == 'object']
numerical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehotencoding', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numericfeatures),
        ('cat', categorical_transformer, categoricfeatures)
    ],
    remainder='passthrough'
)



def objective(trial):
    
    params = {
        'objective': 'regression_l1',  
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100), 
        'subsample': trial.suggest_float('subsample', 0.6, 0.95), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95), 
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), 
        'random_state': 42,
        'n_jobs': -1 
    }

    regressor_obj = LGBMRegressor(**params)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor_obj)
    ])
    model_pipeline.fit(xtrain, ytrain)
    preds = model_pipeline.predict(xtest)
    rmse = np.sqrt(mean_squared_error(ytest, preds))

    return rmse


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("\nOptuna Study Finished.")
print(f"  Best Value (RMSE): {study.best_value}")
print(f"  Best Params: {study.best_params}")


best_params = study.best_trial.params
best_params['objective'] = 'regression_l1'
best_params['metric'] = 'rmse'
best_params['random_state'] = 42
best_params['n_jobs'] = -1


best_regressor = LGBMRegressor(**best_params)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', best_regressor)
])
model_pipeline.fit(xtrain, ytrain)

X_unk = df1
y_unk_pred = model_pipeline.predict(X_unk)



submission = pd.DataFrame({
    'LOCAL_IDENTIFIER': df1['LOCAL_IDENTIFIER'],
    'CORRUCYSTIC_DENSITY': y_unk_pred
})

submission['LOCAL_IDENTIFIER'] = submission['LOCAL_IDENTIFIER'].astype(int)
submission['CORRUCYSTIC_DENSITY'] = submission['CORRUCYSTIC_DENSITY'].astype(float)

submission.to_csv('submission.csv', index=False)
print(submission.head(10))

