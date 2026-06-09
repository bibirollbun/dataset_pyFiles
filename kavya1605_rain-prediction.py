import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
train_df


train_df.describe()


for i in train_df.columns:
    plt.scatter(train_df[i], train_df['rainfall'])
    plt.xlabel(i)
    plt.ylabel('Rainfall')
    plt.show()


train_df.corr()


sns.heatmap(train_df.corr(), annot=True)


train_df.isnull().sum()


train_df.info()


train_df.describe()


sns.distplot(train_df['day'])


train_df[train_df['day']> train_df['day'].mean() + 3* (train_df['day'].std())]


sns.distplot(train_df['pressure'])


train_df[train_df['pressure']> train_df['pressure'].mean() + 3* (train_df['pressure'].std())]


df2= train_df[train_df['pressure']<= train_df['pressure'].mean() + 3* (train_df['pressure'].std())]
df2.describe()


sns.displot(df2['pressure'])


sns.distplot(train_df['maxtemp'])


train_df[train_df['maxtemp']> train_df['maxtemp'].mean() + 3* (train_df['maxtemp'].std())]


sns.distplot(train_df['temparature'])


train_df[train_df['temparature']> train_df['temparature'].mean() + 3* (train_df['temparature'].std())]


sns.distplot(train_df['mintemp'])


train_df[train_df['mintemp']> train_df['mintemp'].mean() + 3* (train_df['mintemp'].std())]


sns.distplot(train_df['dewpoint'])


train_df[train_df['dewpoint']> train_df['dewpoint'].mean() + 3* (train_df['dewpoint'].std())]


sns.distplot(train_df['humidity'])


train_df[train_df['humidity']> train_df['humidity'].mean() + 3* (train_df['humidity'].std())]


sns.distplot(train_df['cloud'])


train_df[train_df['cloud']> train_df['cloud'].mean() + 3* (train_df['cloud'].std())]


sns.distplot(train_df['sunshine'])


train_df[train_df['sunshine']> train_df['sunshine'].mean() + 3* (train_df['sunshine'].std())]


sns.distplot(train_df['winddirection'])


train_df[train_df['winddirection']> train_df['winddirection'].mean() + 3* (train_df['winddirection'].std())]


sns.distplot(train_df['windspeed'])


train_df[train_df['windspeed']> train_df['windspeed'].mean() + 3* (train_df['windspeed'].std())]


df2= train_df[train_df['windspeed']<= train_df['windspeed'].mean() + 3*(train_df['windspeed'].std())]


df2.describe()


sns.distplot(df2['windspeed'])


df2= df2.drop(['id'], axis= 'columns')


df2


x= df2.drop(['rainfall'], axis= 'columns')
x


y= df2['rainfall']


y


from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
x_scaled= scaler.fit_transform(x)


x_scaled


from sklearn.decomposition import PCA

pca=PCA(0.95)
x_pca= pca.fit_transform(x_scaled)
x_pca.shape


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split( x_pca, y, test_size= 0.2)


from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()
lr.fit(x_train, y_train)
lr.score(x_test, y_test)


svm= SVC(C=1,kernel='linear')
svm.fit(x_train, y_train)
svm.score(x_test, y_test)


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


model_params={
    'logistic_regression':{
        'model':LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'C':[1,5,10]
        }
    },

    'svm':{
    'model': SVC(gamma='auto'),
    'params': {
        'C':[1,5,10,20],
        'kernel':['rbf', 'linear']
        }
    },

    'decision_tree':{
        'model': DecisionTreeClassifier(),
        'params':{
            'criterion': ['gini', 'entropy']
        }
    },

    'random_forest':{
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators': [1,5,10]
        }
    },

    'guassian_nb':{
        'model': GaussianNB(),
        'params':{}
    },


}


from sklearn.model_selection import GridSearchCV


score=[]

for model_name, mp in model_params.items():
    clf= GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(x_train,y_train)
    score.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_param': clf.best_params_
    })

score_df= pd.DataFrame(score, columns=['model', 'best_score', 'best_param'])
score_df


test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test_df


test_df.describe()


test_df.isnull().sum()


test_df['winddirection'] = test_df['winddirection'].fillna(test_df['windspeed'].mean())



test_df.isnull().sum()


test_df2=test_df.drop(['id'],axis='columns')



test_df3 = scaler.transform(test_df2)


test_df4= pca.transform(test_df3)


pred=svm.predict(test_df4)


#pred=lr.predict(test_df3)


id= test_df['id']


submission= pd.DataFrame({
    'id': id,
    'price': pred
})

submission.to_csv('submission.csv', index=False)

