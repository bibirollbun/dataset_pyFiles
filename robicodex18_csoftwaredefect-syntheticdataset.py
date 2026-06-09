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


# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Scikit-learn modules
from sklearn.impute import SimpleImputer,MissingIndicator,KNNImputer  # Corrected name
from sklearn.model_selection import train_test_split,cross_val_score  # Correct
from sklearn.compose import ColumnTransformer  # Correct module
from sklearn.preprocessing import PowerTransformer,OneHotEncoder,OrdinalEncoder,LabelEncoder,StandardScaler,FunctionTransformer,KBinsDiscretizer,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error,r2_score
from sklearn.linear_model import Perceptron,LinearRegression,SGDRegressor,Ridge
from mlxtend.plotting import plot_decision_regions


df=pd.read_csv('/kaggle/input/playground-series-s3e23/train.csv')


dftest=pd.read_csv('/kaggle/input/playground-series-s3e23/test.csv')


df.columns


comb=[df,dftest]


# Loop through each DataFrame in the list
for i in comb:
    i.rename(columns={
        'n': 'noFunctions',
        'v': 'noVertices',
        'l': 'noLoops',
        'd': 'noDecisions',
        'i': 'noInputs',
        'e': 'noExternalDependencies',
        'b': 'noBugs',
        't': 'noTests'
    }, inplace=True)



df.columns


df.info()


df.isna().sum()


dftest.isna().sum()


df.describe()


df.duplicated().sum()


for i in df.columns:
    print(i,df[i].unique())


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True)


xtrain,xtest,ytrain,ytest=train_test_split(df.drop(columns='defects'),df['defects'],test_size=0.2,random_state=42)


comb.extend([xtrain,xtest,ytrain,ytest])


ytrain


for i in comb:
    i.drop(columns='id',inplace=True)


fig,axes=plt.subplots(1,2,figsize=(10,5))
axes=axes.flatten()
sns.boxplot(data=df,x='noBugs',hue='defects',ax=axes[0])
sns.histplot(data=df,x='noBugs',hue='defects',ax=axes[1],kde=True)


# After Transformation
trfTemp=ColumnTransformer([
    ('trfTemp',PowerTransformer(method='yeo-johnson'),['noBugs'])
],remainder='passthrough')
df2=df.copy()
df2=pd.DataFrame(trfTemp.fit_transform(df2),columns=trfTemp.get_feature_names_out())

fig,axes=plt.subplots(1,2,figsize=(10,5))
axes=axes.flatten()
sns.boxplot(data=df2,x='trfTemp__noBugs',hue='remainder__defects',ax=axes[0])
sns.histplot(data=df2,x='trfTemp__noBugs',hue='remainder__defects',ax=axes[1],kde=True)


df.columns


fig,axes=plt.subplots(df.shape[1],2,figsize=(10,70))
axes=axes.flatten()
for ind,attr in enumerate(df.drop(columns='defects').columns):
    print(ind,attr)
    sns.boxplot(data=df,x=attr,hue='defects',ax=axes[2*ind])
    sns.histplot(data=df,x=attr,hue='defects',ax=axes[2*ind+1],kde=True)
    axes[2 * ind].set_title(f'Boxplot of {attr} by Defects')
    axes[2 * ind+1].set_title(f'HistPlot of {attr} by Defects')


# After Transformation
trfTemp=ColumnTransformer([
    ('trfTemp',PowerTransformer(method='yeo-johnson'),df.columns)
],remainder='passthrough')
df2=df.copy()
df2=pd.DataFrame(trfTemp.fit_transform(df2),columns=trfTemp.get_feature_names_out())
print(df2.columns)
fig,axes=plt.subplots(df.shape[1],2,figsize=(10,70))
axes=axes.flatten()
for ind,attr in enumerate(df2.drop(columns='trfTemp__defects').columns):
    print(ind,attr)
    sns.boxplot(data=df2,x=attr,hue='trfTemp__defects',ax=axes[2*ind])
    sns.histplot(data=df2,x=attr,hue='trfTemp__defects',ax=axes[2*ind+1],kde=True)
    axes[2 * ind].set_title(f'Boxplot of {attr} by Defects')
    axes[2 * ind+1].set_title(f'HistPlot of {attr} by Defects')


df.columns


pca=PCA()
pca.fit(xtrain)


pca.explained_variance_ratio_


sum=0
for ind,i in enumerate(pca.explained_variance_ratio_):
    sum+=i
    if(sum>0.99):
        print(ind)
        break


trf=ColumnTransformer([
    ('transform',PowerTransformer(method='yeo-johnson'),slice(0,df.shape[1]-1)), #avoided Target Column in Cross Validation
    ('ss',StandardScaler(),slice(0,df.shape[1]-1)) #avoided Target Column in Cross Validation
],remainder='passthrough')


df2=df.copy()
pd.DataFrame(trf.fit_transform(df2),columns=trf.get_feature_names_out()).columns


df2=df.copy()
df2=pd.DataFrame(trf.fit_transform(df2),columns=trf.get_feature_names_out())
df2.columns


# Core models
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# Ensemble models
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)

# Optional external models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



models = {
    "LogisticRegression": LogisticRegression(),
    "RidgeClassifier": RidgeClassifier(),
    "SGDClassifier": SGDClassifier(),
    "PassiveAggressive": PassiveAggressiveClassifier(),
    
    "KNeighbors": KNeighborsClassifier(),
    
    "GaussianNB": GaussianNB(),
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
    "CategoricalNB": CategoricalNB(),

    "DecisionTree": DecisionTreeClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Bagging": BaggingClassifier(),
    "HistGradientBoosting": HistGradientBoostingClassifier(),

    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(),
}



for name,model in models.items():
    try:
        scores=cross_val_score(model,df2.drop(columns=['remainder__defects']),df2['remainder__defects'],cv=5,scoring='accuracy')
        scores2=cross_val_score(model,df2.drop(columns=['remainder__defects']),df2['remainder__defects'],cv=5,scoring='f1_weighted')
        print(name,"Accuracy: ",scores.mean(),"F1 Weighted: ",scores2.mean())
    except Exception as e:
        print(f"{name} failed:")


pipe=Pipeline([
    ('trf',trf),
    ('model',GradientBoostingClassifier())
])


# pipe.fit(xtrain,ytrain)


# yPred=pipe.predict(xtest)


# from sklearn.metrics import f1_score


# accuracy_score(yPred,ytest)


pipe.fit(df.drop(columns=['defects']),df['defects'])


yPred=pipe.predict(dftest)


submission=pd.DataFrame({'defects':yPred})


submission.to_csv('/kaggle/working/submission.csv', index=False)

