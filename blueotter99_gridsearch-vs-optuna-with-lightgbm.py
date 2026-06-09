# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/bank-customer-churn-prediction-challenge/train.csv', index_col = 'id')
df_train.head()


# check for nulls:
df_train.isnull().sum()


# check balance labels
df_train['Exited'].value_counts()/len(df_train['Exited'])


# check for categorical variables:
cat_var = ['Surname','Geography','Gender','HasCrCard','IsActiveMember']
num_var = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']

for var in cat_var:
    print(df_train[var].value_counts())
    print('____'*10)


plt.figure(figsize=(10,10))
fig, axs = plt.subplots(2, 3)

y=[]
for i in range(0,3):
    y = y+[i]*2
x=[i for i in range(0,2)]*3

for i in range(len(num_var)):
    axs[x[i],y[i]].hist(df_train[num_var[i]])
    axs[x[i],y[i]].set_title(num_var[i])

fig.tight_layout()
plt.show()


fig, axs = plt.subplots(2, 2)
axs[0,0].hist(df_train['Age'])
axs[0,0].set_title('Age')
axs[0,1].hist(np.log(df_train['Age']))
axs[0,1].set_title('log_Age')
axs[1,0].hist(df_train['CreditScore'])
axs[1,0].set_title('CreditScore')
axs[1,1].hist(np.power(df_train['CreditScore'],2))
axs[1,1].set_title('sqrt_CreditScore_2')
fig.tight_layout()


# categorical_enc:
def woe_category(column_data, unique=[]):
    if len(unique) == 0:
        unique = column_data.value_counts()/len(column_data)
        unique=unique.apply(lambda x: np.log(x/(1-x)))

    condition = [column_data==unique.index[i] for i in range(len(unique))]
    choicelist = unique.tolist()
    return np.select(condition,choicelist), unique


# encode categorical variables:
cat_encoders = {}
for var in cat_var[1:]:
    df_train[var+'_woe'],cat_encoders[var] = woe_category(df_train[var])


# correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(df_train[num_var+[var+'_woe' for var in cat_encoders.keys()]].corr(),
            annot=df_train[num_var+[var+'_woe' for var in cat_encoders.keys()]].corr())


#feature engineering

# Transformation of Age and CreditScore Variables 
numeric_transformations = {'Age':['log_Age',np.log],'CreditScore':['sqrt_CreditScore_2',np.power,2]}

for var in numeric_transformations.keys():
    if len(numeric_transformations[var])==3:
        df_train[numeric_transformations[var][0]] = df_train[var].apply(lambda x: numeric_transformations[var][1](x,numeric_transformations[var][2]))
    else:
        df_train[numeric_transformations[var][0]] = df_train[var].apply(numeric_transformations[var][1])
    print(var,'=>',numeric_transformations[var][0])
    
cat_var_modelo = [var+'_woe' for var in cat_var[1:]]
num_var_modelo = [var for var in num_var if var not in ['Age','CreditScore']]+['log_Age','sqrt_CreditScore_2']

# listing all variables that will be trained in the models
print('var categoricas: ',cat_var_modelo)
print('var numericas: ', num_var_modelo)


# train test split
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    df_train[cat_var_modelo+num_var_modelo].values, df_train['Exited'].values, test_size=0.33, random_state=42)


# class weights
from sklearn.utils.class_weight import compute_class_weight
class_weight_array = compute_class_weight(class_weight="balanced", 
                     classes=[0,1], 
                     y=df_train['Exited'].values)
class_weight = {i:class_weight_array[i] for i in [0,1]}


# GridsearchCV
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import time

param_grid = {
'n_estimators': [100,500,1000],
"num_leaves": [31, 63, 127],
"max_depth": [-1, 3, 5,7],
"subsample": [0.8, 1.0],
"colsample_bytree": [0.8, 1.0]
}
start_time_grid = time.time()
lgbm = lgb.LGBMClassifier(objective="binary", metric="auc", random_state=42, class_weight=class_weight,verbosity=-1)
grid = GridSearchCV(lgbm, param_grid, cv=5, scoring="roc_auc")
model_grid = grid.fit(X_train, y_train)
finish_time_grid = time.time()


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score,precision_score

print('best parameters: ', grid.best_params_)
print('best_score: ', grid.best_score_)
print('time (minutes):',(finish_time_grid-start_time_grid)/60)

y_pred = model_grid.predict(X_val)
recall_grid = recall_score(y_val, y_pred)
precision_grid = precision_score(y_val,y_pred)
print('recall score validation: ',recall_score(y_val, y_pred))
print('precision score validation: ', precision_score(y_val,y_pred))
cm_grid = confusion_matrix(y_val, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_grid/sum(sum(cm_grid)),
                              display_labels=grid.classes_)
disp.plot()
plt.show()


# optuna
import optuna
from sklearn.metrics import roc_auc_score
# Defining a target function
def objective(trial):
    # Determine hyperparameter values
    params = {'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1),
            'num_leaves': trial.suggest_int("num_leaves", 2, 256),
            'max_depth':trial.suggest_int("max_depth", -1, 50),
            'min_child_samples': trial.suggest_int("min_child_samples", 5, 100),
            'subsample' : trial.suggest_float("subsample", 0.5, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
            'n_estimators': trial.suggest_int("n_estimators", 100, 1000)}
    lgbm = lgb.LGBMClassifier(**params,class_weight=class_weight,verbosity=-1)
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=100)])
    preds = lgbm.predict_proba(X_val)[:, 1]
    auc_roc = roc_auc_score(y_val, preds)
    return auc_roc
# creates optuna run for maximization
study = optuna.create_study(direction="maximize")
start_time_optuna = time.time()
study.optimize(objective, n_trials=100)
finish_time_optuna = time.time()


best_params = study.best_params

lgbm = lgb.LGBMClassifier(**best_params,class_weight=class_weight)
lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=100)])


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score,precision_score

# Print the best set of hyperparameters
print('Best hyperparameters: ', study.best_params)
# Print the corresponding performance
print('Best auc: ', study.best_value)
print('time (minutes):',(finish_time_optuna-start_time_optuna)/60)

y_pred = lgbm.predict(X_val)
recall_optuna = recall_score(y_val, y_pred)
precision_optuna = precision_score(y_val,y_pred)
print('recall score validation: ',recall_score(y_val, y_pred))
print('precision score validation: ', precision_score(y_val,y_pred))
cm = confusion_matrix(y_val, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm/sum(sum(cm)),
                              display_labels=lgbm.classes_)
disp.plot()
plt.show()


# comparison

df_metrics = pd.DataFrame([[grid.best_score_,study.best_value ],
              [(finish_time_grid-start_time_grid)/60,(finish_time_optuna-start_time_optuna)/60],
              [recall_grid,recall_optuna],
              [precision_grid,precision_optuna]], 
            columns = ['GridSearchCV','Optuna'],
            index=np.array(['ROC-AUC','time (minutes)','recall','precision']))

print(df_metrics)
print('___'*10)
fig,axs = plt.subplots(nrows = 1,ncols = 2)
axs[0].set_title("GridSearchCV")
ConfusionMatrixDisplay(confusion_matrix=cm_grid/sum(sum(cm_grid)),
                              display_labels=grid.classes_).plot(include_values=True,  ax=axs[0])

axs[1].set_title("LightGBM")
ConfusionMatrixDisplay(confusion_matrix=cm/sum(sum(cm)),
                              display_labels=lgbm.classes_).plot(include_values=True,  ax=axs[1])
fig.tight_layout()
plt.show()



# read test set

df_test = pd.read_csv('/kaggle/input/bank-customer-churn-prediction-challenge/test.csv')

# feature engineering:
# numeric
print('numeric')
for var in numeric_transformations.keys():
    if len(numeric_transformations[var])==3:
        df_test[numeric_transformations[var][0]] = df_test[var].apply(lambda x: numeric_transformations[var][1](x,numeric_transformations[var][2]))
    else:
        df_test[numeric_transformations[var][0]] = df_test[var].apply(numeric_transformations[var][1])
    print(var,'=>',numeric_transformations[var][0])
    
print('___'*10)
print('categorical')
# categoricals
for var in cat_var[1:]:
    df_test[var+'_woe'],_ = woe_category(df_test[var],cat_encoders[var])
    print(var,'=>',var+'_woe')


# predict classes for test set
X_test = df_test[cat_var_modelo+num_var_modelo].values
y_pred_proba = lgbm.predict_proba(X_test)[:, 1]


df_test['Exited'] = y_pred_proba
df_test[['id','Exited']].to_csv('submission.csv', index = False)




