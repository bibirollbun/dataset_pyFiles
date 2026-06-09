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


import os
os.cpu_count()


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')


df.head()


n, m = df.shape 
print(f'The no of data samples in the data set is {n}')
print(f'The no of features in the data set is {m -1}')


list(df.columns)


df.info()


## preparing the column names 

df.columns = df.columns.str.lower().str.replace(' ', '_')
df.head()


## we don't need the id of the peoples as they are not that useful
## Extract the features we are interested in!

features = [
     'sex',
     'age',
     'height',
     'weight',
     'duration',
     'heart_rate',
     'body_temp',
     'calories' ## This is the target variable
 ]

df = df[features] 
df.head()


## Treating NAN data

df.isnull().sum() #no missing data


import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


calories = df.calories.values
sns.histplot(calories, bins = 50) 


log_calories = np.log1p(calories)
sns.histplot(log_calories, bins = 50)


df.calories = log_calories 
df.calories.head()


categorical = ['sex']
numerical = list(set(df.columns) - set(categorical))
print(numerical)


corr_matrix = df[numerical].corr()
sns.heatmap(corr_matrix, annot = True)



sns.histplot(data=df, x='calories', hue='sex', 
             palette={'male': 'blue', 'female': 'red'}, 
             alpha=0.4, bins=30, stat='probability')


df.describe()


df['sex'].value_counts()



from sklearn.metrics import mutual_info_score


scores = mutual_info_score(df.calories, df.weight)
scores


def mutual_info_y_score(series):
    return mutual_info_score(series, df.calories)

mi = df[numerical].apply(mutual_info_y_score).round(5)
mi = mi.sort_values(ascending = False).to_frame(name = "MI")
mi


df.sex = (df.sex == 'male').astype(int) ## { male: 1, female: 0}
df.head()


from sklearn.model_selection import train_test_split


df_fulltrain, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
df_fulltrain.shape, df_test.shape


df_train, df_val = train_test_split(df_fulltrain, test_size = 0.25)
df_train.shape, df_val.shape


assert len(df) == len(df_train) + len(df_val) + len(df_test)


## let's reset the index of the split data

df_fulltrain = df_fulltrain.reset_index(drop = True)
df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)


df_test.head()


## Target variable

y_fulltrain = df_fulltrain.calories.values
y_train = df_train.calories.values
y_val = df_val.calories.values
y_test = df_test.calories.values

## delete the target variables

del df_fulltrain['calories']
del df_train['calories']
del df_val['calories']
del df_test['calories']


df_train.head(7)


y_fulltrain[:7]


X_train = df_train.values
#print(X_train)
ones = np.ones(X_train.shape[0])
#print(ones)
X_train = np.column_stack([ones, X_train])
X_train


X_train.shape, y_train.shape


from sklearn.linear_model import LinearRegression


model = LinearRegression()


model.fit(X_train, y_train)


x_trial = X_train[:2]
y_trial = model.predict(x_trial)
print(y_trial)


w0 = model.intercept_
print(w0)
w1 = model.coef_
print(w1)


df_train.head()


features_weights = {}

for i, c in enumerate(df_train.columns):
    features_weights[c] = w1[i+1]

print(features_weights)




from sklearn.metrics import mean_squared_log_error


y_train_log_preds = model.predict(X_train)
y_train_preds= np.expm1(y_train_log_preds)
RMSE_train = np.sqrt(mean_squared_log_error(np.expm1(y_train), model.predict(X_train)))
RMSE_train


X_val = df_val.values
#print(X_train)
ones = np.ones(X_val.shape[0])
#print(ones)
X_val = np.column_stack([ones, X_val])
X_val


y_val_log_preds = model.predict(X_val)
y_val_preds= np.expm1(y_val_log_preds)
RMSE_val = np.sqrt(mean_squared_log_error(np.expm1(y_val), y_val_preds))
RMSE_val


X_test = df_test.values
ones = np.ones(X_test.shape[0])
X_test = np.column_stack([ones, X_test])
X_test


y_test_log_preds = model.predict(X_test)
y_test_preds= np.expm1(y_test_log_preds)
RMSE_test = np.sqrt(mean_squared_log_error(np.expm1(y_test), y_test_preds))
RMSE_test


## For ease of use, I define a custom function that does this

def get_RMSLE(y_true_logs, X_matrix, model):
    y_pred_logs = model.predict(X_matrix)
    y_preds = np.expm1(y_pred_logs)
    y_trues = np.expm1(y_true_logs)
    return np.sqrt(mean_squared_log_error(y_trues, y_preds))

print(get_RMSLE(y_test, X_test, model))


df_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


df_submission.head()


df_submission.columns = df_submission.columns.str.lower().str.replace(' ', '_')
df_submission.columns


df_submission.sex = (df_submission.sex == 'male').astype(int)
df_submission.head()


sub_features = list(features_weights)
sub_features


## Data Matrix for Submission

X_sub = df_submission[sub_features].values
ones = np.ones(X_sub.shape[0])
X_sub = np.column_stack([ones, X_sub])
X_sub


y_sub_log = model.predict(X_sub)
y_sub_preds = np.expm1(y_sub_log)
y_sub_preds


df_sub = np.column_stack([df_submission.id.values, y_sub_preds])
df_sub


df_sub = pd.DataFrame(df_sub)
df_sub.head()



df_sub = pd.DataFrame(df_sub)
df_sub.columns = ['id', 'Calories']
df_sub.id = df_sub.id.astype(int)
df_sub.head()


# ## let's export the predictions
# df_sub.to_csv('submission.csv', index=False)



c = 0.1
X_train_reg = X_train + c * np.eye(*X_train.shape)
X_train_reg


y_train_reg = y_train.copy()
y_train_reg


model_reg = LinearRegression()
model_reg.fit(X_train_reg, y_train_reg)


w0_reg = model.intercept_
w1_reg = model.coef_
print(w0_reg)
print(w1_reg)


model_reg.score(X_val, y_val)


## Validation
y_val_preds_log = model_reg.predict(X_val)
#print(y_val_preds_log)

y_val_preds = np.expm1(y_val_preds_log)

reg_scores = np.sqrt(mean_squared_log_error(np.expm1(y_val), y_val_preds))
print(reg_scores)


## 
from tqdm.auto import tqdm


y_val


tuning = False
if tuning:
    C = [0.001, 0.01, 0.1, 1, 10, 100]
    scores = []
    
    for c in tqdm(C):
        model_reg = LinearRegression()
        X_train_c = X_train + c * np.eye(*X_train.shape)
        y_train = y_train
        model_reg.fit(X_train_c, y_train)
        ##Validatiion set
        sub_score = get_RMSLE(y_val, X_val, model_reg)
        print(f'C: {c}, score: {sub_score}')
        scores.append(sub_score)
    
    print(scores)


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=10, random_state=0)


X_train = df_train.values
X_val = df_val.values
X_test = df_test.values
X_fulltrain = df_fulltrain.values


rf_model.fit(X_train, y_train)


y_pred_log_rf = rf_model.predict(X_val)
y_pred_log_rf


print(get_RMSLE(y_test, X_test, rf_model))


get_RMSLE?


from tqdm.auto import tqdm


tuning = False

if tuning:
    #hyperparameter Tuning
    #max_depths = [5,10, 15, 20, 50, None]
    N_estimators = np.arange(10, 201, 10)
    scores = []
    n_jobs = -1
    for n_estimator in tqdm(N_estimators):
        rf_model = RandomForestRegressor(n_estimators = n_estimator,
                                         max_depth = None,
                                         n_jobs = -1,
                                         random_state = 1)
        # fitting
        rf_model.fit(X_train, y_train)
    
        # rmse evaluation
        rmse = get_RMSLE(y_val, X_val, rf_model)
    
        scores.append((n_estimator, rmse))
    df_randomforest_scores= pd.DataFrame(scores, columns = ['n_estimators', 'rmse'])
    df_randomforest_scores = df_randomforest_scores.sort_values(by='rmse', ascending = True)
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    plt.plot(df_randomforest_scores.n_estimators, df_randomforest_scores.rmse)


import xgboost as xgb


Dtrain = xgb.DMatrix(X_train, y_train)
Dval = xgb.DMatrix(X_val, y_val)
Dtest = xgb.DMatrix(X_test, y_test)


xgb_params = {
    'eta': 0.1,
    'max_depth': 7,
    'min_child_weight': 2,

    'objective': 'reg:squarederror',
    'nthread':8,

    'seed':1,
    'verbosity': 1,
}


xgb_model = xgb.train(xgb_params, Dtrain, num_boost_round = 150)


get_RMSLE(y_val, Dval, xgb_model)


### Submission Generator Function

def generate_submission(df_sub, model, linear = False, xgb_flag = False):
    X_sub = df_submission[sub_features].values
    if linear:
        ones = np.ones(X_sub.shape[0])
        X_sub = np.column_stack([ones, X_sub])

    if xgb:
        X_sub = xgb.DMatrix(X_sub)

    y_sub_log = model.predict(X_sub)
    y_sub_preds = np.expm1(y_sub_log)
    
    df_sub = np.column_stack([df_submission.id.values, y_sub_preds])
    df_sub = pd.DataFrame(df_sub)
    df_sub.columns = ['id', 'Calories']
    df_sub.id = df_sub.id.astype(int)

    return df_sub



sub__ = generate_submission(df_submission, xgb_model, xgb_flag = True)


sub__.head()


## Export Submssion
sub__.to_csv('submission.csv', index = False)




