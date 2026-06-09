# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,roc_curve,roc_auc_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import optuna

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


df_train.head()


test_id = df_test['id']

df_train.drop('id',axis = 1,inplace = True)
df_test.drop('id',axis = 1,inplace = True)


df_train.info()


df_train.isnull().sum()


# Checking for class imbalance
df_train['diagnosed_diabetes'].value_counts(normalize = True)*100


# Seprating numerical and categorical columns

num_cols = df_train.select_dtypes(include =  [int,float]).columns
cat_cols = df_train.select_dtypes(include = [object]).columns


df_train.groupby(['diagnosed_diabetes'],as_index = False)[num_cols].median()


desc = df_train[num_cols].describe().T
desc



# Outlier Detectiom

desc['IQR'] = desc['75%'] - desc['25%']
desc['upper_bound'] = desc['75%'] + 1.5*desc['IQR']
desc['lower_bound'] = desc['25%'] - 1.5*desc['IQR']
desc['outlier_flag'] = ((desc['min']<desc['lower_bound']) | (desc['max']>desc['upper_bound']))

desc


for col in num_cols:
    fig,axes = plt.subplots(1,2,figsize = (10,3))
    sns.boxplot(x = df_train[col],ax  = axes[0])
    axes[0].set_title(f'BoxPlot:{col}')

    sns.histplot(df_train[col],ax = axes[1],kde = True)
    axes[1].set_title(f'Histogram:{col}')

    plt.tight_layout()
    plt.show()
    


plt.figure(figsize = (10,3))
sns.heatmap(df_train[num_cols].corr(),cmap = 'coolwarm',center = 0)
  
plt.show()


for col in cat_cols:
    plt.figure(figsize = (10,3))
    sns.countplot(data = df_train,x = col,hue = df_train['diagnosed_diabetes'])
    plt.title(f'diagnosed_diabetes vs {col}')
    plt.tight_layout()
    plt.show()


cat_cols = cat_cols.drop( ['income_level','education_level'])


# Updating the numerical colums list by removing non-continuous columns
num_cols = num_cols.drop(['family_history_diabetes','hypertension_history','cardiovascular_history','diagnosed_diabetes','alcohol_consumption_per_week'])


edu_level = [['No Formal','Highschool','Graduate','Postgraduate']]
income_level = [['Low','Lower-Middle','Middle','Upper_Middle','High']]

edu_oe = OrdinalEncoder(categories = edu_level,handle_unknown = 'use_encoded_value',unknown_value = -1)
inc_oe = OrdinalEncoder(categories = income_level,handle_unknown = 'use_encoded_value',unknown_value = -1)
ohe = OneHotEncoder(handle_unknown = 'ignore',drop = 'first',sparse = False)

ct = ColumnTransformer(transformers = [
    ('edu_oe',edu_oe,['education_level']),
    ('inc_oe',inc_oe,['income_level']),
    ('ohe',ohe,cat_cols)
],remainder = 'passthrough')


# To handle class imbalance

'''
def objective(trial):
    classifier_name = trial.suggest_categorical('model',['RandomForestClassifier','xgb'])

    if classifier_name == 'RandomForestClassifier':
        n_estimators = trial.suggest_int('n_estimators',50,150)
        max_depth = trial.suggest_int('max_depth',3,20)
        min_samples_split = trial.suggest_int('min_samples_split',5,15)
        min_samples_leaf = trial.suggest_int('min_samples_leaf',3,20)
        bootstrap = trial.suggest_categorical('bootstrap',[True,False])

        model = RandomForestClassifier(
            n_estimators = n_estimators,
            max_depth = max_depth,
            min_samples_leaf = min_samples_leaf,
            min_samples_split = min_samples_split,
            bootstrap = bootstrap,
            class_weight = 'balanced'
        )

    elif classifier_name == 'xgb':
        n_estimators = trial.suggest_int('n_estimators',50,150)
        learninig_rate = trial.suggest_int('learning_rate',0.01,0.3)
        max_depth = trial.suggest_int('max_depth',5,15)
        max_child_weight = trial.suggest_int('max_child_weight',3,10)
        subsample = trial.suggest_int('subsample',0.6,1.0)
        colsample_bytree = trial.suggest_int('colsample_bytree',0.6,1.0)
        reg_alpha = trial.suggest_int('reg_alpha',0.6,1.0)
        reg_lambda = trial.suggest_int('reg_lambda',0.6,1.0)

        model = XGBClassifier(
            n_estimators = n_estimators,
            max_depth = max_depth,
            max_chile_weight = max_child_weight,
            subsample = subsample,
            colsample_bytree = colsample_bytree,
            reg_alpha = reg_alpha,
            reg_lambda = reg_lambda,
            scale_pos_weight = scale_pos_weight,
            n_jobs = -1
        )


    pipeline = Pipeline([
        ('ct',ct),
        ('model',model1)
    ])

    score = cross_val_score(pipeline,x,y,cv = 2,scoring = 'f1').mean()

    return score

    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective,n_trials = 50)
 '''    


x = df_train.drop('diagnosed_diabetes',axis = 1)
y = df_train['diagnosed_diabetes']


x_trans = ct.fit_transform(x)
x_test_trans = ct.transform(df_test)


neg = (y == 0.0).sum()
pos = (y == 1.0).sum()
scale_pos_weight = neg/pos

model = XGBClassifier(
    n_estimators = 5000,
    learning_rate =  0.05,
    max_depth =  3,
    min_child_weight = 9,
    subsample = 0.9324156371717225,
    colsample_bytree = 0.5974622135152583,
    reg_lambda = 3.1929676333640495,
    random_state = 42,
    n_jobs = -1,
)



'''
pipeline = Pipeline([
    ('ct',ct),
    ('model',model)
])
pipeline.fit(x,y)
y_pred = pipeline.predict_proba(df_test)[:,1]
y_pred
'''


model.fit(x_trans,y)


y_pred = model.predict_proba(x_test_trans)[:,1]


submission = pd.DataFrame({'id':test_id,'diagnosed_diabetes':y_pred})
submission.to_csv('submission.csv',index = False)
print('File Sucessfully Submitted')


#score = cross_val_score(pipeline,x,y,cv = 5,scoring = 'f1').mean()


'''
model.fit(
    x_train,
    y_train,
    eval_set = [(x_test,y_test)],
    eval_metric = 'auc',
    early_stopping_rounds = 50,
    verbose = False
)
'''




