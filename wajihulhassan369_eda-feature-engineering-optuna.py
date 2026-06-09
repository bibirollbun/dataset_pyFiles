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


# Importing Necessary Libararies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Reading Datasets
train = pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')


train.sample(5)


train.info()


test.info()


train['Age'] = train['Age'].astype(np.int64)
train['IsActiveMember' ] = train['IsActiveMember'].astype(np.int16)
train['HasCrCard'] = train['HasCrCard'].astype(np.int16)

train.drop(columns=['CustomerId','Surname'],inplace = True)

train.rename(columns={'Geography':'Country'},inplace=True)

train.set_index('id',inplace=True)


test['Age'] = test['Age'].astype(np.int64)
test['IsActiveMember' ] = test['IsActiveMember'].astype(np.int16)
test['HasCrCard'] = test['HasCrCard'].astype(np.int16)

test.drop(columns=['CustomerId','Surname'],inplace = True)

test.rename(columns={'Geography':'Country'},inplace=True)

test.set_index('id',inplace=True)


unique_values = train.nunique()

cat_features = unique_values[unique_values < 15].index.tolist()

cont_features = unique_values[unique_values >= 15].index.tolist()

cat_features.remove('Exited')


cat_features


train.describe().T


test.describe().T


train.isnull().sum().sum()


test.isnull().sum().sum()


print("Train Duplicated rows :",train.duplicated().sum())


train_idxs = train[train.duplicated()].index


train.drop(index=train_idxs,inplace=True)


# Setting for better plots
sns.set_style("darkgrid")    
sns.set_palette(["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"])


plt.figure(figsize=(6,5))
ax = sns.countplot(data=train,x='Exited',hue='Exited')
total = train.shape[0]

for bar in ax.patches:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)

    # Add percentage on top of each bar
    height = bar.get_height()
    percentage = f'{100 * height / total:.1f}%'
    ax.text(bar.get_x() + bar.get_width()/2., height + 300,
            percentage,
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.xlabel('Exited',fontsize=13,fontweight='bold',color='#2c3e50')
plt.ylabel('Count',fontsize=13,fontweight='bold',color='#2c3e50')
plt.title('Exited Count Plot',fontsize=15,fontweight='bold',color='#2c3e50')

plt.tick_params(labelsize=10,colors='brown')
plt.xticks(fontweight='500')
plt.yticks(fontweight='500');


sns.histplot(data=train,x='Age',kde=True,hue='Exited',palette='dark',alpha=0.25)
plt.title('Age Kde Plot');


ax = sns.countplot(data=train,x='Country',hue='Exited')

total = train.shape[0]

for bar in ax.patches:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)

    # Add percentage on top of each bar
    height = bar.get_height()
    percentage = f'{100 * height / total:.1f}%'
    ax.text(bar.get_x() + bar.get_width()/2., height + 300,
            percentage,
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.xlabel('Countries',fontsize=13,fontweight='bold',color='#2c3e50')
plt.ylabel('Count',fontsize=13,fontweight='bold',color='#2c3e50')
plt.title('Countrys Count Plot',fontsize=15,fontweight='bold',color='#2c3e50')

plt.tick_params(labelsize=10,colors='brown')
plt.xticks(fontweight='500')
plt.yticks(fontweight='500');


# Returns a dictionary with "key = categories" and "values = percentage difference"
def Percent_Diff(feature):
    categories = train[feature].value_counts().index

    per_diff = []
    for cat in categories:
        per_diff.append(train[(train[feature] == cat) & (train['Exited']==1)].shape[0] / train[(train[feature] == cat)].shape[0] * 100)

    return dict(zip(categories,per_diff))



Percent_Diff('Country')


train[train['Country']=='Spain']['Exited'].value_counts()


# 2 Subplots 
# 1st --> Pointplot
# 2nd --> Count Plot
for col_name in cat_features:
    
    fig,axes = plt.subplots(1,2,figsize=(15,4.5))
    dictionary = Percent_Diff(col_name)
    df= pd.DataFrame(dictionary,index=[0])
    
    # Pointplot 
    sns.pointplot(x=df.columns,y=df.loc[0].values,markers='s',scale=2,ax=axes[0],color='green')
    axes[0].set_xlabel('',fontsize=14,fontweight='bold',color='#2c3e50')
    axes[0].set_ylabel('Percent Differnce (%)',fontsize=14,fontweight='bold',color='#2c3e50')

    axes[0].tick_params(labelsize=12, colors='darkblue')

    # Countplot
    sns.countplot(data=train,y=col_name,ax=axes[1],palette='viridis',hue=col_name)
    axes[1].set_xlabel('Count',fontsize=14,fontweight='bold',color='#2c3e50')
    axes[1].set_ylabel('',fontsize=14,fontweight='bold',color='#2c3e50')
    axes[1].tick_params(labelsize=12, colors='darkblue')
    
    plt.suptitle(f'{col_name}',fontsize=16)
    plt.show()


# Scatter Plot of Age 
plt.figure(figsize=(12,8))
sns.scatterplot(data=train,x = np.random.randint(1,400,size=train.shape[0]),y='Age',hue='Exited')

plt.xlabel('',fontsize=13,fontweight='bold',color='#2c3e50')
plt.ylabel('Age',fontsize=13,fontweight='bold',color='#2c3e50')
plt.title('Scatter Plot of Age',fontsize=15,fontweight='bold',color='#2c3e50')

plt.tick_params(labelsize=10,colors='brown')
plt.xticks(fontweight='500')
plt.yticks(fontweight='500');


plt.figure(figsize=(10,5))
sns.heatmap(train.corr(numeric_only=True),annot=True);


sns.kdeplot(x=train['Balance']+0.01,hue=train['Exited'],log_scale=True);


train['Balance'].describe()


sns.boxplot(data=train,x='Exited',y='Balance');


sns.boxplot(data=train,y='CreditScore',hue='Exited',palette='viridis')





train['Age_group'] =  np.where(train['Age'].between(18,25),'Students',
        np.where(train['Age'].between(26,35),'Early Career',
        np.where(train['Age'].between(36,45),'Adults',
        np.where(train['Age'].between(46,55),'Pre Retirement',
        np.where(train['Age'].between(56,65),'Near Retirement',
        'Retirees')))))

test['Age_group'] =  np.where(test['Age'].between(18,25),'Students',
        np.where(test['Age'].between(26,35),'Early Career',
        np.where(test['Age'].between(36,45),'Adults',
        np.where(test['Age'].between(46,55),'Pre Retirement',
        np.where(test['Age'].between(56,65),'Near Retirement',
        'Retirees')))))


train['Age_group'].value_counts()


train['Balance_greater_50k'] = np.where(train['Balance']>50000,'Above_50k','Below_50k')

test['Balance_greater_50k'] = np.where(test['Balance']>50000,'Above_50k','Below_50k')


train["Different_Classes"] =  np.where(train['EstimatedSalary'].between(0,35000),"Lower Class",
                np.where(train['EstimatedSalary'].between(35001,60000),"Working Class",
                np.where(train['EstimatedSalary'].between(60001,100000),"Middle Class",
                np.where(train['EstimatedSalary'].between(100001,150000),"Upper Middle Class",
                "Upper Class"))))

test["Different_Classes"] =  np.where(test['EstimatedSalary'].between(0,35000),"Lower Class",
                np.where(test['EstimatedSalary'].between(35001,60000),"Working Class",
                np.where(test['EstimatedSalary'].between(60001,100000),"Middle Class",
                np.where(test['EstimatedSalary'].between(100001,150000),"Upper Middle Class",
                "Upper Class"))))


train['credit_score_group'] = np.where(train['CreditScore'].between(300,579),"Poor",
        np.where(train['CreditScore'].between(580,669),"Fair",
                np.where(train['CreditScore'].between(670,739),"Good",
                        np.where(train['CreditScore'].between(740,799),"Very Good",
                                "Excelllent"))))

test['credit_score_group'] = np.where(test['CreditScore'].between(300,579),"Poor",
        np.where(test['CreditScore'].between(580,669),"Fair",
                np.where(test['CreditScore'].between(670,739),"Good",
                        np.where(test['CreditScore'].between(740,799),"Very Good",
                                "Excelllent"))))


train['Engagement_score'] = train['IsActiveMember'] + train['HasCrCard']

test['Engagement_score'] = test['IsActiveMember'] + test['HasCrCard']


train[['IsActiveMember','HasCrCard']]


X = train.drop('Exited',axis=1)
y = train['Exited']


cat_features += ['Age_group','Different_Classes','credit_score_group','Balance_greater_50k']


cat_features


X = pd.get_dummies(X,columns=cat_features,drop_first=True,dtype=np.int8)


test_copy = test.copy()
test_copy = pd.get_dummies(test,columns=cat_features,drop_first=True,dtype=np.int8)


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.33, random_state=42)


y_test


X_train.shape


X_test.shape


from sklearn.preprocessing import PowerTransformer


pt = PowerTransformer() # method = 'yeo-johnson' (default)


train.info()


X_train[cont_features] =  pt.fit_transform(X_train[cont_features])
X_test[cont_features] = pt.transform(X_test[cont_features])


test_copy[cont_features] = pt.transform(test_copy[cont_features])


# After the Transformations 
n_cols = 2
n_rows = 2

plt.figure(figsize=(15,n_rows*5))

for i,col_name in enumerate(cont_features,1):
    plt.subplot(n_rows,n_cols,i)
    
    sns.kdeplot(data=X_train,x=col_name)
    



# Before Transformations
n_cols = 2
n_rows = 2

plt.figure(figsize=(15,n_rows*5))

for i,col_name in enumerate(cont_features,1):
    plt.subplot(n_rows,n_cols,i)
    
    sns.kdeplot(data=train,x=col_name)
    



plt.figure(figsize=(15,12))
sns.heatmap(X_train.corr(numeric_only=True),annot=True);


sns.boxplot(data=train,x='EstimatedSalary');


sns.boxplot(data=train,x='CreditScore');


# Outliers are helping to predict the Target Column
train[train['CreditScore'] < 430]['Exited'].value_counts()


from sklearn.preprocessing import MinMaxScaler


MMScaler = MinMaxScaler()


X_train_transformed = MMScaler.fit_transform(X_train)
X_test_transformed = MMScaler.transform(X_test)


# Coverting to DataFrames
X_train_transformed = pd.DataFrame(X_train_transformed,columns= X_train.columns,index=X_train.index)
X_test_transformed = pd.DataFrame(X_test_transformed,columns= X_test.columns,index=X_test.index)


# for test dataset
test_copy_columns = test_copy.columns
test_copy  = pd.DataFrame(MMScaler.transform(test_copy),columns=test_copy_columns)


test_copy.head()


# Importing Necessary Libraries

from sklearn.model_selection import KFold,StratifiedKFold,cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,confusion_matrix,roc_auc_score

# for Bayesian Optimization
import optuna
from optuna.visualization import plot_optimization_history


# To remove in the final editing of the notebook

X_train_notfull = X_train_transformed[:5000]
X_test_notfull = X_test_transformed[:5000]
y_train_notfull = y_train[:5000]
y_test_notfull = y_test[:5000]


def calculate_roc_auc(clf,X,y):
    y_pred = clf.predict(X)
    return roc_auc_score(y,y_pred)


def Multiple_Ml_Objective(trial):

    classifier = trial.suggest_categorical('classifier',['Logistic Regression','Random Forest'])

    if classifier == 'Logistic Regression':

        C = trial.suggest_float('C', 1e-4, 1e2,log=True)
        lr_class_weight = trial.suggest_categorical('lr_class_weight',['balanced',None])
        penalty = trial.suggest_categorical('penalty',['l1','l2','elasticnet'])
        max_iter = trial.suggest_int('max_iter',20,1000,step=20)
        l1_ratio = None
    
        if penalty == 'l1':
            l1_solver = trial.suggest_categorical('l1_solver',['liblinear','saga'])
            model = LogisticRegression(penalty=penalty,C=C,class_weight=lr_class_weight,solver=l1_solver,max_iter=max_iter,
                                       n_jobs= -1)
        elif penalty == 'l2':
            l2_solver = trial.suggest_categorical('l2_solver',['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga'])
            model = LogisticRegression(penalty=penalty,C=C,class_weight=lr_class_weight,solver=l2_solver,max_iter=max_iter,
                                       n_jobs= -1)
        elif penalty == 'elasticnet':
            l3_solver = 'saga'
            l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
            model = LogisticRegression(penalty=penalty,C=C,class_weight=lr_class_weight,solver=l3_solver,max_iter=max_iter,
                                  l1_ratio=l1_ratio,n_jobs= -1)


    
    elif classifier =='Random Forest':

        n_estimators = trial.suggest_int('n_estimators',100,500,step=100) # changed 

        criterion = trial.suggest_categorical('criterion',['gini','entropy','log_loss'])

        max_depth_option = trial.suggest_categorical('max_depth_option',['auto','fixed'])
        if max_depth_option == 'auto':
            max_depth = None
        else :
            max_depth = trial.suggest_int('max_depth',10,100,step=10)

        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3,0.5,None])

        max_samples = trial.suggest_float('max_samples',0.2,0.6) # changed

        class_weight = trial.suggest_categorical('class_weight',['balanced',None,'balanced_subsample'])

        min_impurity_decrease = trial.suggest_float('min_impurity_decrease',0.0,0.01) 

        model = RandomForestClassifier(
            n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
            max_features=max_features, max_samples=max_samples, class_weight=class_weight,
            min_impurity_decrease=min_impurity_decrease, n_jobs=-1
        )



    cv_results = cross_validate(
        model,
        X_train_transformed,
        y_train,
        scoring=calculate_roc_auc,
        cv= StratifiedKFold(shuffle=True),
        return_train_score=True
    )

    train_score_mean = cv_results['train_score'].mean()
    val_score_mean = cv_results['test_score'].mean()

    trial.set_user_attr('train_score_mean',train_score_mean)
    trial.set_user_attr('train_score_std',cv_results['train_score'].std())
    trial.set_user_attr('test_score_std',cv_results['test_score'].std())
    trial.set_user_attr('overfitting_gap',train_score_mean-val_score_mean)

    return val_score_mean


# Change the n_trails according to your need
multiple_ml_optuna = optuna.create_study(direction='maximize')
multiple_ml_optuna.optimize(Multiple_Ml_Objective,n_jobs=-1,n_trials=5)


df = multiple_ml_optuna.trials_dataframe()


df.sort_values(by=['value','user_attrs_overfitting_gap'],ascending=[False,True])[['value','params_classifier','user_attrs_overfitting_gap','user_attrs_train_score_mean']]


df.params_classifier.value_counts()


def Objective_RF(trial):
    n_estimators = trial.suggest_int('n_estimators',25,500,step=25)
    criterion = trial.suggest_categorical('criterion',['gini', 'entropy', 'log_loss'])
    max_depth = trial.suggest_int('max_depth',10,150,step=10)
    min_samples_split_type = trial.suggest_categorical('min_samples_split_type', ['absolute', 'fraction'])
    
    if min_samples_split_type == 'absolute':
            min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
    else:
        min_samples_split = trial.suggest_float('min_samples_split_frac', 0.005, 0.2)

    min_samples_leaf_type = trial.suggest_categorical('min_samples_leaf_type', ['absolute', 'fraction'])
    if min_samples_leaf_type == 'absolute':
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    else:
        min_samples_leaf = trial.suggest_float('min_samples_leaf_frac', 0.005, 0.08)

    max_features = trial.suggest_categorical('max_features',['sqrt', 'log2',0.1,0.2,0.3,0.4,0.5,0.6])

    min_impurity_decrease = trial.suggest_float('min_impurity_decrease',0.0,0.05)

    max_samples = trial.suggest_float('max_samples',0.1,0.6) 

    class_weight = trial.suggest_categorical('class_weight',['balanced',None,'balanced_subsample'])

    min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.1)


    model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                  max_depth=max_depth, min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf, max_features=max_features,
                                  min_impurity_decrease=min_impurity_decrease, max_samples=max_samples,
                                  class_weight=class_weight, min_weight_fraction_leaf=min_weight_fraction_leaf,
                                  n_jobs=-1)

    cv_results = cross_validate(
        model,
        X_train_transformed,
        y_train,
        scoring=calculate_roc_auc,
        cv=StratifiedKFold(shuffle=True),
        return_train_score=True
    )

    train_score_mean = cv_results['train_score'].mean()
    val_score_mean = cv_results['test_score'].mean()

    trial.set_user_attr('train_score_mean',train_score_mean)
    trial.set_user_attr('train_score_std',cv_results['train_score'].std())
    trial.set_user_attr('test_score_std',cv_results['test_score'].std())
    trial.set_user_attr('overfitting_gap',train_score_mean-val_score_mean)

    return val_score_mean


rf_optuna = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler()) 
good_params_list  = [{'n_estimators': 200, 'criterion': 'log_loss', 'max_depth_option': 'auto', 'max_features': 0.3, 'max_samples': 0.20970533401239885, 'class_weight': 'balanced_subsample', 'min_impurity_decrease': 0.001222420236848637},
                    {'n_estimators': 500, 'criterion': 'entropy', 'max_depth_option': 'fixed', 'max_depth': 20, 'max_features': 0.5, 'max_samples': 0.4350231569217893, 'class_weight': None, 'min_impurity_decrease': 0.0022847064836960785},
                    { 'n_estimators': 100, 'criterion': 'entropy', 'max_depth_option': 'fixed', 'max_depth': 60, 'max_features': 'sqrt', 'max_samples': 0.5585767223218072, 'class_weight': None, 'min_impurity_decrease': 0.005744368126042859}]

for params in good_params_list:
    rf_optuna.enqueue_trial(params)
rf_optuna.optimize(Objective_RF,n_trials=10)


plot_optimization_history(rf_optuna)





def Objective_Log_Reg(trial):
    C = trial.suggest_float('C', 1e-4, 1e2,log=True)
    class_weight = trial.suggest_categorical('class_weight',['balanced',None])
    penalty = trial.suggest_categorical('penalty',['l1','l2','elasticnet'])
    max_iter = trial.suggest_int('max_iter',20,2000,step=20)
    l1_ratio = None
    
    if penalty == 'l1':
        l1_solver = trial.suggest_categorical('l1_solver',['liblinear','saga'])
        model = LogisticRegression(penalty=penalty,C=C,class_weight=class_weight,solver=l1_solver,max_iter=max_iter,
                                       n_jobs= -1)
    elif penalty == 'l2':
        l2_solver = trial.suggest_categorical('l2_solver',['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga'])
        model = LogisticRegression(penalty=penalty,C=C,class_weight=class_weight,solver=l2_solver,max_iter=max_iter,
                                    n_jobs= -1)
    elif penalty == 'elasticnet':
        l3_solver = trial.suggest_categorical('solver',['saga'])
        l1_ratio = trial.suggest_float('l1_ratio', 0.05, 0.9)
        model = LogisticRegression(penalty=penalty,C=C,class_weight=class_weight,solver=l3_solver,max_iter=max_iter,
                                l1_ratio=l1_ratio,n_jobs= -1)


    cv_results = cross_validate(
        model,
        X_train_transformed,
        y_train,
        scoring=calculate_roc_auc,
        cv=StratifiedKFold(shuffle=True),
        return_train_score=True
    )

    train_score_mean = cv_results['train_score'].mean()
    val_score_mean = cv_results['test_score'].mean()

    trial.set_user_attr('train_score_mean',train_score_mean)
    trial.set_user_attr('train_score_std',cv_results['train_score'].std())
    trial.set_user_attr('test_score_std',cv_results['test_score'].std())
    trial.set_user_attr('overfitting_gap',train_score_mean-val_score_mean)

    return val_score_mean


log_reg_optuna = optuna.create_study(direction='maximize')

good_params_list = [{'C': 12.301388165294407, 'class_weight': None, 'penalty': 'elasticnet', 'max_iter': 60, 'l1_ratio': 0.23865384873721693},
{'C': 12.301388165294407, 'class_weight': None, 'penalty': 'elasticnet', 'max_iter': 60, 'l1_ratio': 0.23865384873721693},
{'C': 23.940948108885383, 'class_weight': None, 'penalty': 'l2', 'max_iter': 360, 'l2_solver': 'newton-cholesky'}
]

for params in good_params_list:
    log_reg_optuna.enqueue_trial(params)
log_reg_optuna.optimize(Objective_Log_Reg,n_trials=10)


df = log_reg_optuna.trials_dataframe()


df.sort_values(by=['value','user_attrs_overfitting_gap'],ascending=[False,True])


log_reg_optuna.best_params


if rf_optuna.best_value < log_reg_optuna.best_value:
    log_param = log_reg_optuna.best_params

    log_param = log_reg_optuna.best_params

    try:
        value = log_param.pop('l1_solver')
    except:
        try:
            value = log_param.pop('l2_solver')
        except:
            value = log_param.pop('solver')
    finally:
        log_param.update(
            {'solver':value}
        )
        
    model = LogisticRegression(**log_param,n_jobs=-1)
    
    
else:
    log_param = rf_optuna.best_params

    model = RandomForestClassifier(**log_param,n_jobs=-1)


model


model.fit(X_train_transformed,y_train)
y_pred = model.predict(X_test_transformed)


test_copy.head()


test_predicted = model.predict(test_copy)


test_predicted.shape


submission =  pd.DataFrame({'id':test.index,
             'Exited':test_predicted})


# submission.to_csv('submission.csv')


submission




