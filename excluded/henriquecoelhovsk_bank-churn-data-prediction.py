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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from catboost import CatBoostClassifier


train_data=pd.read_csv(r"/kaggle/input/playground-series-s4e1/train.csv")
test_data=pd.read_csv(r"/kaggle/input/playground-series-s4e1/test.csv")
sample_sub=pd.read_csv(r"/kaggle/input/playground-series-s4e1/sample_submission.csv")
test_data.describe()


print("Formato do train data ", train_data.shape)
print("MOstrando as colunas dos dados", train_data.columns.values)


train_data.head()


test_data.head()


sample_sub.head()


train_data.info()



train_data['Age'] = train_data['Age'].astype(int)
train_data['HasCrCard'] = train_data['HasCrCard'].astype(int)
train_data['IsActiveMember'] = train_data['IsActiveMember'].astype(int)
train_data.info()


train_data.describe()


train_data['NumOfProducts'].unique()


train_data['Geography'].unique()


train_data['Gender'].unique()


train_data['Tenure'].unique()


fig = px.histogram(train_data, x='Exited',text_auto=True, title='Churn Distribution')

fig.show()


fig = px.histogram(train_data, x='Geography',color="Exited",barmode='group',text_auto=True,title="Gráfico de barras agrupadas de rotatividade por região geográfica")
fig.show()


fig = px.histogram(train_data, x='Gender',color="Exited",barmode='group',title="Gráfico de barras agrupadas de rotatividade por genero")
fig.show()


fig = px.histogram(train_data, x='Geography',color="Exited",barnorm="percent",text_auto=True,title="Stacked Bar Chart of Churning for Geography")
fig.show()


fig = px.histogram(train_data, x='Gender',color="Exited",barnorm="percent",text_auto=True,title="Stacked Bar Chart of Churning for Gender")
fig.show()


fig = px.histogram(train_data, x='NumOfProducts',color="Exited",barnorm="percent",text_auto=True,title="Gráfico de barras por NumOfProducts")
fig.show()


fig = px.histogram(train_data, x='NumOfProducts',color="Exited",barmode='group', barnorm="percent",text_auto=True,title="Stacked Bar Chart of Churning for NumOfProducts")
fig.show()


fig = px.histogram(train_data, x='HasCrCard',color="Exited",barnorm="percent",text_auto=True,title="Stacked Bar Chart of Churning for HasCrCard")
fig.show()


fig = px.histogram(train_data, x='HasCrCard',color="Exited",barmode='group', barnorm="percent",text_auto=True,title="Stacked Bar Chart of Churning for HasCrCard")
fig.show()


fig = px.histogram(train_data, x='IsActiveMember',color="Exited",barnorm="percent",text_auto=True,title="Gráfico em barra agrupada para IsActiveMember")
fig.show()


fig = px.histogram(train_data, x='Tenure',color="Exited",text_auto=True,title="Gráfico em barra agrupada para Tenure")
fig.show()


fig = px.histogram(train_data, x='Tenure',color="Exited",barmode='group', barnorm="percent",text_auto=True,title="Stacked Bar Chart of Churning for Tenure")
fig.show()



print("Print columns data ", train_data.columns.values)


df=train_data.drop(['id', 'CustomerId'  ,'HasCrCard', 'Surname'], axis=1)



df_test=test_data.drop(['id', 'CustomerId'  ,'HasCrCard', 'Surname'], axis=1)


df.head()


print("Print new train data shape ",df.shape)
print("Print new train data info ",df.info())


Churn_back = df[df['Exited']==1]['CreditScore'].values
Churn_not = df[df['Exited']==0]['CreditScore'].values


plt.boxplot([Churn_back, Churn_not])



sns.boxplot(x='Exited', y='CreditScore', data=df)
plt.title('Box Plot de CreditScore por Exited')
plt.show()




fig = px.box(df, x="Exited", y="CreditScore")
fig.show()





sns.kdeplot(df['CreditScore'], shade=True)
plt.title('Density Plot of Credit Score')
plt.show()




fig = px.box(df, x="Exited", y="Age")
fig.show()


sns.kdeplot(df['Age'], shade=True)
plt.title('Density Plot of Age')
plt.show()




fig = px.box(df, x="Exited", y="Balance")
fig.show()


sns.kdeplot(df['Balance'], shade=True)
plt.title('Density Plot of Balance')
plt.show()
#Balance, EstimatedSalary



fig = px.box(df, x="Exited", y="EstimatedSalary")
fig.show()


sns.kdeplot(df['EstimatedSalary'], shade=True)
plt.title('Density Plot of EstimatedSalary')
plt.show()
#Balance, EstimatedSalary


sns.scatterplot(x='EstimatedSalary', y='Exited', data=df)
plt.title('Scatter Plot of EstimatedSalary vs. Exited')
plt.show()


sns.scatterplot(x='EstimatedSalary', y='Age', data=df)
plt.title('Scatter Plot of EstimatedSalary vs. Age')
plt.show()



fig = px.line(df, x="CreditScore", y="EstimatedSalary", color='Exited')
fig.show()


correlation_matrix = df[['Age','CreditScore', 'Balance','EstimatedSalary','IsActiveMember','Tenure','NumOfProducts', 'Exited']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


#Now we need to convert categorical data(Gendr and Geography) to numerical data
#Does feature scaling is required for numerical data here?



X_train,X_test,y_train,y_test = train_test_split(df, df['Exited'] , stratify =df['Exited'] , train_size = 0.7)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)





X_train.head()





encoder=ce.TargetEncoder(cols=['Geography'])
X_train['Geography']=encoder.fit_transform(X_train['Geography'],X_train['Exited'])
X_test['Geography']=encoder.transform(X_test['Geography'])

df_test['Geography']=encoder.transform(df_test['Geography'])


encoder=ce.TargetEncoder(cols=['Gender'])
X_train['Gender']=encoder.fit_transform(X_train['Gender'],X_train['Exited'])
X_test['Gender']=encoder.transform(X_test['Gender'])

df_test['Gender']=encoder.transform(df_test['Gender'])


X_train.head()


X_train.head()


X_train=X_train.drop(['Exited'], axis=1)
X_test=X_test.drop(['Exited'], axis=1)


X_train.head()


from sklearn.preprocessing import StandardScaler
cols = list(X_train.columns)
scalar = StandardScaler()
X_train[cols] = scalar.fit_transform(X_train[cols])
X_test[cols] = scalar.transform(X_test[cols])
df_test[cols]= scalar.transform(df_test[cols])


def find_optimal_parameter(X_train,y_train):
    max_depth =[1,5,10,50,100]
    n_estimators=[1,5,10,50,100]
    parameters = {'max_depth':[1,5,10,50,100],'n_estimators':[1,5,10,50,100]}
    model = RandomForestClassifier()
    clf = GridSearchCV(model, parameters, cv=5, scoring='roc_auc',return_train_score=True)
    clf.fit(X_train, y_train)

    train_auc_mean= clf.cv_results_['mean_train_score']
    train_auc_std= clf.cv_results_['std_train_score']
    cv_auc_mean = clf.cv_results_['mean_test_score'] 
    cv_auc_std= clf.cv_results_['std_test_score']
    params=clf.cv_results_['params']
    
    y_train_pred =  clf.predict(X_train)
    print("y_train_pred ",y_train_pred)
    
    results = clf.cv_results_
   # print(results)
    print("Best Parameter: ",clf.best_params_)
    print("best_estimator",clf.best_estimator_)
    print(params)
    params_range = list(range(0,len(max_depth)*len(n_estimators)))
    
    plt.plot(params_range, train_auc_mean, label='Train AUC')
    plt.plot(params_range, cv_auc_mean, label='CV AUC')
    plt.legend()
    plt.xlabel("params")
    plt.ylabel("AUC")
    plt.title("ERROR PLOTS")
    plt.show()
    
    return train_auc_mean,cv_auc_mean,clf.best_params_


# Logistic regression for test and its performance
def model_test_performance(X_train,y_train,X_test,y_test,hyper_params):# instantiate learning model k = optimal_k
    
    # train a classifier and prediction for test
    clf = RandomForestClassifier(max_depth=hyper_params['max_depth'],min_samples_split=hyper_params['n_estimators'])
    clf.fit(X_train, y_train)

    train_fpr, train_tpr, thresholds = roc_curve(y_train, clf.predict_proba(X_train)[:,1])
    test_fpr, test_tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(metrics.auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(metrics.auc(test_fpr, test_tpr)))
    plt.legend()
    plt.xlabel("hyperparameters")
    plt.ylabel("AUC")
    plt.title("ERROR PLOTS")
    plt.show()

    print("="*100)

    print("Train confusion matrix")
    cm_train=confusion_matrix(y_train, clf.predict(X_train))
    print("Train confusion matrix")
    print(cm_train)

    cm=confusion_matrix(y_true=y_test,y_pred=clf.predict(X_test))
    print("Test confusion matrix")
    print(cm)

    labels = ['0','1']

    
    print("-----------Train Confusion Matrics--------")
    df_cm_train = pd.DataFrame(cm_train,  range(2),range(2))    
    sns.heatmap(df_cm_train, annot=True, cmap='Oranges', annot_kws={"size": 20},fmt='g')# font size
    
    
    plt.title("Train Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    
    print("-----------Test Confusion Matrics--------")
    df_cm = pd.DataFrame(cm,  range(2),range(2))    
    sns.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20},fmt='g')# font size

    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    return metrics.auc(test_fpr, test_tpr),clf


def find_optimal_parameter_GBDT(X_train,y_train):
    max_depth =[1,5,10,50,100]
    n_estimators=[1,5,10,50,100]
    parameters = {'max_depth':[1,5,10,50,100],'n_estimators':[1,5,10,50,100]}
    model = xgb.XGBClassifier(booster='gbtree')
    clf = GridSearchCV(model, parameters, cv=5, scoring='roc_auc',return_train_score=True)
    clf.fit(X_train, y_train)

    train_auc_mean= clf.cv_results_['mean_train_score']
    train_auc_std= clf.cv_results_['std_train_score']
    cv_auc_mean = clf.cv_results_['mean_test_score'] 
    cv_auc_std= clf.cv_results_['std_test_score']
    params=clf.cv_results_['params']
    
    y_train_pred =  clf.predict(X_train)
    
    results = clf.cv_results_
    print("Best Parameter: ",clf.best_params_)
    print("best_estimator",clf.best_estimator_)
    print(params)
    params_range = list(range(0,len(max_depth)*len(n_estimators)))
    
    plt.plot(params_range, train_auc_mean, label='Train AUC')
    plt.plot(params_range, cv_auc_mean, label='CV AUC')
    plt.legend()
    plt.xlabel("params")
    plt.ylabel("AUC")
    plt.title("ERROR PLOTS")
    plt.show()
    
       
    return train_auc_mean,cv_auc_mean,clf.best_params_


# XGB for test and its performance
def model_test_performance_GBDT(X_train,y_train,X_test,y_test,hyper_params):# instantiate learning model k = optimal_k
    
    # train a classifier and prediction for test
    clf = xgb.XGBClassifier(booster='gbtree',max_depth=hyper_params['max_depth'],min_samples_split=hyper_params['n_estimators'])
    clf.fit(X_train, y_train)

    train_fpr, train_tpr, thresholds = roc_curve(y_train, clf.predict_proba(X_train)[:,1])
    test_fpr, test_tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(metrics.auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(metrics.auc(test_fpr, test_tpr)))
    plt.legend()
    plt.xlabel("hyperparameters")
    plt.ylabel("AUC")
    plt.title("ERROR PLOTS")
    plt.show()

    print("="*100)

    print("Train confusion matrix")
    cm_train=confusion_matrix(y_train, clf.predict(X_train))
    print("Train confusion matrix")
    print(cm_train)

    cm=confusion_matrix(y_true=y_test,y_pred=clf.predict(X_test))
    print("Test confusion matrix")
    print(cm)

    # Test Confusion Matrix
    labels = ['0','1']

    print("-----------Train Confusion Matrics--------")
    df_cm_train = pd.DataFrame(cm_train,  range(2),range(2))    
    sns.heatmap(df_cm_train, annot=True, cmap='Oranges', annot_kws={"size": 20},fmt='g')# font size
    
    
    plt.title("Train Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    
    print("-----------Test Confusion Matrics--------")
    df_cm = pd.DataFrame(cm,  range(2),range(2))    
    sns.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20},fmt='g')# font size
    
    #skplt.plot_confusion_matrix(y_test ,pred)
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    return metrics.auc(test_fpr, test_tpr),clf



def find_optimal_parameter_CAT(X_train,y_train):
    max_depth =[1,5,10,50,100]
    n_estimators=[5,10,50,100,150]
    parameters = {'max_depth':[1,5,10,50,100],'n_estimators':[5,10,50,100,150]}
    model =  CatBoostClassifier()
    
    clf = GridSearchCV(model, parameters, cv=5, scoring='roc_auc',return_train_score=True)
    clf.fit(X_train, y_train)

    train_auc_mean= clf.cv_results_['mean_train_score']
    train_auc_std= clf.cv_results_['std_train_score']
    cv_auc_mean = clf.cv_results_['mean_test_score'] 
    cv_auc_std= clf.cv_results_['std_test_score']
    params=clf.cv_results_['params']
    
    y_train_pred =  clf.predict(X_train)
    
    results = clf.cv_results_
    print("Best Parameter: ",clf.best_params_)
    print("best_estimator",clf.best_estimator_)
    print(params)
    params_range = list(range(0,len(max_depth)*len(n_estimators)))
    
    plt.plot(params_range, train_auc_mean, label='Train AUC')
    plt.plot(params_range, cv_auc_mean, label='CV AUC')
    plt.legend()
    plt.xlabel("params")
    plt.ylabel("AUC")
    plt.title("ERROR PLOTS")
    plt.show()
    
       
    return train_auc_mean,cv_auc_mean,clf.best_params_


train_auc_mean_CAT,cv_auc_mean_CAT,params_CAT=find_optimal_parameter_CAT(X_train,y_train)


# XGB for test and its performance
def model_test_performance_CAT(X_train,y_train,X_test,y_test,hyper_params):# instantiate learning model k = optimal_k
    
    # train a classifier and prediction for test
    clf = CatBoostClassifier(max_depth=hyper_params['max_depth'],n_estimators=hyper_params['n_estimators']);
    clf.fit(X_train, y_train)

    train_fpr, train_tpr, thresholds = roc_curve(y_train, clf.predict_proba(X_train)[:,1])
    test_fpr, test_tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(metrics.auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(metrics.auc(test_fpr, test_tpr)))
    plt.legend()
    plt.xlabel("hyperparameters")
    plt.ylabel("AUC")
    plt.title("ERROR PLOTS")
    plt.show()

    print("="*100)

    print("Train confusion matrix")
    cm_train=confusion_matrix(y_train, clf.predict(X_train))
    print("Train confusion matrix")
    print(cm_train)

    cm=confusion_matrix(y_true=y_test,y_pred=clf.predict(X_test))
    print("Test confusion matrix")
    print(cm)

    # Test Confusion Matrix
    labels = ['0','1']

    print("-----------Train Confusion Matrics--------")
    df_cm_train = pd.DataFrame(cm_train,  range(2),range(2))    
    sns.heatmap(df_cm_train, annot=True, cmap='Oranges', annot_kws={"size": 20},fmt='g')# font size
    
    
    plt.title("Train Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    
    print("-----------Test Confusion Matrics--------")
    df_cm = pd.DataFrame(cm,  range(2),range(2))    
    sns.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20},fmt='g')# font size

    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    return (metrics.auc(test_fpr, test_tpr),clf)



# train a classifier and prediction for test for BOW
print(params_CAT)
auc_CAT,clf_CAT=model_test_performance_CAT(X_train,y_train,X_test,y_test,params_CAT)



print(auc_CAT)
y_pred_CAT=clf_CAT.predict_proba(df_test)[:, 1]


def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


plot_feature_importance(clf_CAT.feature_importances_,X_train.columns,'CAT BOOST')





# save submission

sample_sub['Exited']=y_pred_CAT
results = sample_sub[['id', 'Exited']]
results.to_csv('/kaggle/working/submission.csv', index=False)


results.head()







