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


from scipy.stats import spearmanr, uniform, randint, loguniform
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, plot_importance
from xgboost.callback import EarlyStopping
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_log_error
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import optuna
import seaborn as sns
import math
import time


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col = "id")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv", index_col = "id")


def safe_rmsle(y_true, y_pred): # Avoids rmsle cant be used with negative values' error
    y_pred = np.maximum(y_pred, 0) 
    
    return math.sqrt(mean_squared_log_error(y_true, y_pred))


def safe_rmsle_for_scorer(y_true, y_pred): # Made to be used with cross_val_score
    return - safe_rmsle(y_true, y_pred)

safe_rmsle_scorer = make_scorer(safe_rmsle_for_scorer, greater_is_better=False)


def get_avg_cv_score(pipeline, features, folds):
    X = train[features]
    y = train['Calories']
    
    score = cross_val_score(pipeline, X, y, cv = folds, scoring = safe_rmsle_scorer)


    count = 1
    
    for s in score:
        print(f'fold {count} -> score: {s:.6f}')
        count += 1

    print('#' * 5)

    print(f'avg score: {np.average(score)}')
    print(f'std:       {np.std(score)}')


data = train.copy()
data['Sex'] = data["Sex"].map({'male':0,'female':1})


data.head()


print(f"train set shape: {data.shape}")
print(f"test set shape: {test.shape}")

print('\n')

data.info()

print('\n')
data.describe()


data.hist(bins=50, figsize = (10,10))
plt.show()


print(data['Calories'].skew())
data['Calories'].hist(bins=100)


cal_sqrt = data['Calories'].copy()
cal_sqrt = np.sqrt(cal_sqrt)

print(cal_sqrt.skew())
cal_sqrt.hist(bins=100)


cal_log = data['Calories'].copy()
cal_log = np.log1p(cal_log)

print(cal_log.skew())
cal_log.hist(bins=100)


print(data['Body_Temp'].skew())
data['Body_Temp'].hist(bins=100)


tr = PowerTransformer(standardize=False, copy=False)

tr.fit_transform(data['Body_Temp'].values.reshape(-1,1))

print(data['Body_Temp'].skew())
data['Body_Temp'].hist(bins=100)


corr_matrix = data.corr() # correlation matrix
corr_matrix["Calories"].sort_values(ascending=False)


sns.heatmap(corr_matrix, annot=True)


p_corr_matrix = data.corr(method = 'spearman')

p_corr_matrix["Calories"].sort_values(ascending=False)


sns.heatmap(p_corr_matrix, annot=True)


data["BMI"] = data["Weight"]/((data["Height"]/100) ** 2)
data["BSA"] = ((data["Weight"]* data["Height"])/3600) ** 0.5

# checking correlation
cols_to_drop = ['Age','Duration', 'Body_Temp', 'Heart_Rate', 'Sex']
df = data.drop(cols_to_drop, axis=1)

corr_matrix = df.corr(method='spearman')
corr_matrix["Calories"].sort_values(ascending=False)


strong_corr = ["Duration", "Body_Temp", "Heart_Rate"]
pairs = []


for col1 in strong_corr:
    for col2 in strong_corr:

        cur_pair = [col1, col2]
        cur_pair.sort()
        
        if col1 == col2 or cur_pair in pairs: continue
            
        new_name = col1 + "x" + col2

        data[new_name] = data[col1] * data[col2]

        pairs.append(cur_pair) # Avoids unecessary columns ex: ( AxB e BxA)


# checking correlation
cols_to_drop = ['Age', 'Height', 'Weight', 'Sex', 'BMI', 'BSA']
df = data.drop(cols_to_drop, axis=1)

corr_matrix = df.corr(method='spearman')
corr_matrix["Calories"].sort_values(ascending = False)


data['Heart_Rate/Minute'] = (data['Heart_Rate']/data['Duration']).round(6)
data['DurationxBody_TempxHeart_Rate'] = data['Duration'] * data['Body_Temp'] * data['Heart_Rate']

# checking correlation
cols_to_drop = ['Age', 'Height', 'Weight', 'Sex', 'BMI', 'BSA']
df = data.drop(cols_to_drop, axis=1)

corr_matrix = df.corr(method='spearman')
corr_matrix["Calories"].sort_values(ascending = False)


cols =  [i for i in train.columns if i != "Calories"]

train_duplicates = train.duplicated(subset=cols, keep= False)
test_duplicates = test.duplicated(subset=cols, keep= False)

print(f"duplicates on train set: {train_duplicates.sum()}\nduplicates on test set: {test_duplicates.sum()}")


train = train.groupby(cols, as_index=False, sort=False)["Calories"].mean()

print(f"training set shape after dropping duplicates: {train.shape}")


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        return
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):        
        bmi = X['Weight'] / ((X['Height'] / 100) ** 2)
        duration_x_heart_rate = X['Duration'] * X['Heart_Rate']
        duration_x_body_temp = X['Duration'] * X['Body_Temp']
        body_temp_x_heart_rate = X['Body_Temp'] * X['Heart_Rate']
        heart_rate_per_min = X['Heart_Rate']/X['Duration']

        X_new = X.copy()
        
        # X_new['BMI'] = bmi
        # X_new['DurationxBody_TempxHeart_Rate'] = X['Duration'] * X['Body_Temp'] * X['Heart_Rate']
        X_new['Duration*HeartRate'] = duration_x_heart_rate
        X_new['Duration*BodyTemp'] = duration_x_body_temp
        X_new['BodyTemp*HeartRate'] = body_temp_x_heart_rate
        X_new['Heart_Rate/Min'] = heart_rate_per_min

        return X_new


num_cols = ['Age','Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
cat_cols = ['Sex']


# Defining transformers
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output= False) # Encode categorical attribute 'Sex'
power_tr = PowerTransformer(standardize = True) # Make 'Body_Temp' distribution more normal-like
attr_adder = CombinedAttributesAdder() # Add combined attributes
std_scaler = StandardScaler() # Adjusting parameter scales

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('one_hot_encoder', one_hot_encoder, cat_cols),
    ('power_tr', power_tr, ['Body_Temp']),
    ('attr_adder', attr_adder, num_cols),
    ('scaler', std_scaler, num_cols),
])

# Model
xgb = XGBRegressor(n_estimators =2000, learning_rate= 0.02, random_state = 1)


# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb)
])


# Model
xgb_regressor = XGBRegressor(n_estimators=2000, learning_rate=0.02, random_state = 1)

xgb_tt = TransformedTargetRegressor(
        regressor= xgb_regressor,
        func=np.sqrt,
        inverse_func=np.square
)


# Full pipeline
pipeline_tt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_tt)
])


ftrs = ['Sex', 'Age','Weight', 'Height', 'Duration', 'Heart_Rate', 'Body_Temp']
target = 'Calories'


print('Score WITHOUT target transformation:')
# get_avg_cv_score(pipeline, ftrs, 5)


print('Score WITH target transformation:')
# get_avg_cv_score(pipeline_tt, ftrs, 5)


X, X_test= train[ftrs], test[ftrs]
y = train[target]


# Optuna objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.05, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.1, 0.7),
        'verbosity': 0,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'gpu_id': 0,
    }


    xgb_regressor = XGBRegressor(**params)

    xgb_tt = TransformedTargetRegressor(
        regressor= xgb_regressor,
        func=np.sqrt,
        inverse_func=np.square
    )

    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb_tt)
    ])

    score = cross_val_score(final_pipeline, X, y, cv = 5, scoring = safe_rmsle_scorer)

    return np.mean(score)


#study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=50)


#print("Search Results:")
#print(f"  Score (RMSLE): {study.best_trial.value}")
#print("  Params:")
#for key, value in study.best_trial.params.items():
#    print(f"    {key}: {value}")


X, X_test= train[ftrs], test[ftrs]
y = train[target]


xgb_regressor = XGBRegressor(
    n_estimators=802,
    max_depth=10,
    learning_rate=0.031270710697151614,
    reg_lambda= 9.169512240598808,
    subsample=0.48567391331512755,
    random_state=1
)

xgb_tt = TransformedTargetRegressor(
    regressor= xgb_regressor,
    func=np.sqrt,
    inverse_func=np.square
)

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_tt)
])

final_pipeline.fit(X, y)
preds = final_pipeline.predict(X_test)


# Sometimes XGB predicts negative values
for i in range(len(preds)):
    preds[i] = abs(preds[i])


# Adpating prediction to submission template
sub = pd.DataFrame()
sub["id"] = test.index
sub["Calories"] = preds
sub.to_csv("/kaggle/working/submission.csv", index = False)


sub

