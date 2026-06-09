import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 


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


train = pd.read_csv(r'/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e3/test.csv')

train.head()


train.info()


train.shape


train.describe()


train.isnull().sum()/len(train) * 100


train.shape





# Removing nan values
train  = train[~np.isnan(train['cloud'].values) ]
train = train[~np.isnan(train['humidity'].values)]
train = train[~np.isnan(train['dewpoint'].values)]
train = train[~np.isnan(train['humidity'].values)]
train = train[~np.isnan(train['windspeed'].values)]



print(train.isna().sum())


train.shape


train.head()


def univariate_cont_analysis(df, col):
    fig,ax = plt.subplots(1,2, figsize = (10,5))
    sns.histplot(x=df[col],ax=ax[0])
    sns.boxplot(df[col],ax=ax[1])

def univariate_cat_analysis(df,col):
    fig,ax = plt.subplots(1,2, figsize = (10,5))
    sns.histplot(x=df['col'].value_counts(), ax=ax[0])
    sns.boxplot(df[col].value_counts(),ax=ax[1])

def remove_outliers_using_iqr(df,col,q):

    q1 = df[col].quantile(q)
    q3 =df[col].quantile(1-q)

    iqr = q3 - q1 
    
    lower = q1 - iqr*1.5
    upper = q3 + iqr*1.5
    print(lower)
    print(upper)
    return df[(df[col] > lower) & (df[col] < upper)][col]
    


univariate_cont_analysis(train,'pressure')


univariate_cont_analysis(train,'maxtemp')


train.head()


univariate_cont_analysis(train,'temparature')


train['dewpoint'] = remove_outliers_using_iqr(train,'dewpoint',0.25)
train['humidity'] = remove_outliers_using_iqr(train,'humidity',0.25)
train['cloud'] = remove_outliers_using_iqr(train,'cloud', 0.25)
train['windspeed'] = remove_outliers_using_iqr(train,'windspeed', 0.25)


univariate_cont_analysis(train,'mintemp')
univariate_cont_analysis(train,'dewpoint')
univariate_cont_analysis(train,'humidity')
univariate_cont_analysis(train,'cloud')
univariate_cont_analysis(train,'sunshine')
univariate_cont_analysis(train,'winddirection')
univariate_cont_analysis(train,'windspeed')


train['cloud'].quantile(0.25)


# Still have some outliers in Cloud column 

lower = train['cloud'].quantile(0.05)
upper = train['cloud'].quantile(0.95)

train['cloud'] = train[(train['cloud'] > 70) & (train['cloud'] < upper) ]['cloud']

univariate_cont_analysis(train,'cloud')




train.shape


sns.scatterplot(data=train,x= 'pressure',y = 'temparature')


train.head()





sns.scatterplot(data=train,x= 'temparature',y = 'cloud')


train.pop('id')



plt.figure(figsize=(12,6))
sns.heatmap(train.corr(), annot=True)


train.head()


y_train = train.pop('rainfall')


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(train)


from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train, y_train)


test.head()
id = test.pop('id')
X_test = scaler.transform(test)



y_pred_test = xgb.predict(X_test)


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0]
}
from sklearn.metrics import auc
XGB = XGBClassifier()

xgb_grid = GridSearchCV(n_jobs=-1, 
             estimator=XGB,
             param_grid=param_grid,
             cv=4,
             scoring=auc)


%%time 
xgb_grid.fit(X_train,y_train)


%%time 
y_pred_train = xgb_grid.predict_proba(X_train)[:,1]
y_pred_test = xgb_grid.predict_proba(X_test)[:,1]


# Plotting the AUC-ROC curve of the final model
from sklearn.metrics import roc_auc_score, roc_curve

y_pred_prob = xgb_grid.predict_proba(X_train)[:,1]
auc_score = roc_auc_score(y_train, y_pred_prob)

fpr, tpr, thresholds = roc_curve(y_train, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC-ROC: Final Model")
plt.legend()
plt.show()


sub = pd.DataFrame({'id': id, 'rainfall': y_pred_test })


sub.to_csv('submission.csv', index=0)


#y = xgb.predict(X_test)



#y




