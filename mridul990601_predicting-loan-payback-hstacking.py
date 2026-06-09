# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,FunctionTransformer,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import optuna
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sam = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


# Removing id column as it is not contributing in the result
id = df_test['id']
df_train.drop('id',axis = 1,inplace = True)
df_test.drop('id',axis = 1,inplace = True)



df_train.head()


# To check if dataset have any null values
df_train.isnull().sum()


df_train.describe()


# Cheking for class imbalance
df_train['loan_paid_back'].value_counts(normalize = True)


# Seprating numerical and categorical columns
num_cols = df_train.select_dtypes(include= ['int','float']).columns
cat_cols = df_train.select_dtypes(include = ['object']).columns

num_cols.drop('loan_paid_back')


cat_cols


fig,axes = plt.subplots(nrows = 2,ncols = 3,figsize = (18,8))
axes = axes.flatten()
for i,col in enumerate(num_cols):
    sns.histplot(df_train[col],ax = axes[i],kde = True)
    axes[i].set_title(f'{col} Distibution')

plt.tight_layout()
plt.show()


fig,axes = plt.subplots(nrows = 2,ncols = 3,figsize = (20,8))
axes = axes.flatten()

for i,col in enumerate(cat_cols):
    sns.countplot(data = df_train,x = col,ax = axes[i])
    axes[i].set_title(f'{col} CountPlot')

plt.tight_layout()
plt.show()


fig,axes = plt.subplots(nrows = 2,ncols = 3,figsize = (18,8))
axes = axes.flatten()

for i,col in enumerate(num_cols):
    sns.boxplot(data = df_train,x = df_train['loan_paid_back'],y = col,ax = axes[i])
    axes[i].set_title(f'{col} ')

plt.tight_layout()
plt.show()


fig,axes = plt.subplots(nrows = 2,ncols = 3,figsize = (20,8))
axes = axes.flatten()

for i,col in enumerate(cat_cols):
    df_cross = pd.crosstab(df_train[col],df_train['loan_paid_back'],normalize = 'index')
    df_cross.plot(kind = 'bar',ax = axes[i],stacked = True)
    plt.title(f'{col} vs target')


plt.tight_layout()
plt.show()


# Correlation Heatmap

plt.figure(figsize = (8,6))
sns.heatmap(df_train[num_cols].corr(),annot = True,cmap = 'magma')
plt.title('Correlation Heatmap')
plt.show()


# for ordinal columns
edu_ord = [['Other','High School',"Bachelor's","Master's",'PhD']]
edu_oe = OrdinalEncoder(categories = edu_ord,handle_unknown = 'use_encoded_value',unknown_value = -1)

grade_ord = [sorted(df_train['grade_subgrade'].unique())]
grade_oe = OrdinalEncoder(categories = grade_ord,handle_unknown = 'use_encoded_value',unknown_value = -1)

# for nominal columns

le = LabelEncoder()

cat_cols.drop(['education_level','grade_subgrade'])

# for other categorical columns

ohe = OneHotEncoder(handle_unknown = 'ignore',drop = 'first',sparse = False)


# Adding new features

def add(df):

    df['emi_to_incomeRatio'] = (df['loan_amount'] * (df['interest_rate']/100))/df['annual_income']
    df['disposable_income'] = df['annual_income']*(1-df['debt_to_income_ratio']/100)

    bins = [0,560,670,750,800,np.inf]
    labels = [0,1,2,3,4]

    df['cred_risk'] = pd.cut(df['credit_score'],bins = bins,labels = labels)
    df['cred_risk'] = df['cred_risk'].astype(int)

    return df

    


func_trans = FunctionTransformer(func = add,validate = False)

col_trans = ColumnTransformer( transformers = [
    ('edu_oe',edu_oe,['education_level']),
    ('grade_oe',grade_oe,['grade_subgrade']),
    ('ohe',ohe,cat_cols)
],
remainder = 'passthrough')




x = df_train.drop(['loan_paid_back'],axis = 1)
y = df_train['loan_paid_back']


# To handle class imbalance
neg = (y == 0.0).sum()
pos = (y == 1.0).sum()
scale_pos_weight = neg/pos

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
        ('ft',func_trans),
        ('ct',col_trans),
        ('model',model1)
    ])

    score = cross_val_score(pipeline,x,y,cv = 2,scoring = 'f1').mean()

    return score

    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective,n_trials = 50)
  '''      


model1 = XGBClassifier(
    n_estimators = 100,
    learning_rate = 0.01,
    max_depth = 8,
    subsample = 0.9,
    colsample_bystree = 0.9,
    scale_pos_weight = scale_pos_weight,
    reg_alpha = 0.1,
    random_state = 42,
    n_jobs = -1
)
model2 = RandomForestClassifier(
    n_estimators = 100,
    max_depth = 8,
    min_samples_leaf = 6,
    min_samples_split = 8,
    bootstrap = True,
    random_state = 42,
    class_weight = 'balanced'
)

model3 = SVC(probability = True,random_state = 42,max_iter = 100)

base_models = [('xgb',model1),('rf',model2),('svc',model3)]

meta_model = LogisticRegression(random_state = 42)

stack = StackingClassifier(
    estimators = base_models,
    final_estimator = meta_model,
    cv = 3,
    stack_method = 'predict_proba'
)


pipeline = Pipeline([
    ('ft',func_trans),
    ('ct',col_trans),
    ('model',stack)
])

#score = cross_val_score(pipeline,x,y,cv = 5,scoring = 'f1').mean()

pipeline.fit(x,y)



y_pred = pipeline.predict_proba(df_test)[:,1]
y_pred



submission = pd.DataFrame({'id':id,'loan_paid_back':y_pred})
submission.to_csv('submission.csv',index = False)
print('submit file created successfully')










