import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns#数据可视化库，主要用于绘制统计图表

pd.options.display.max_columns = None

#sklearn：机器学习库
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import * #导入全部函数和类
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import *
from sklearn.ensemble import RandomForestClassifier
from xgboost import *
from sklearn.model_selection import * #用于数据划分；交叉验证（cross-validation）；超参数搜索



df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')
df.sample(frac=0.2)



df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')

X = df.drop(['output'], axis = 1)
y = df['output'].values

#对特征数值范围敏感的模型 e.x.基于梯度下降优化的模型需要标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)#标准正态分布转化

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_hat = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_hat))
print('Matrice de confusion:')#混淆矩阵：用来展示预测结果和真实标签之间的对比。[[TN  FP],[FN  TP]]
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))
RocCurveDisplay.from_estimator(model, X_test, y_test)#AUC小于0.5没有二分类任务区分能力


model.predict_proba(X_test)#模型对每个样本属于各类别的概率


from sklearn.model_selection import * #从 scikit-learn 的 model_selection 模块 中导入 全部函数和类。
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')

X = df.drop(['output'], axis = 1)
y = df['output'].values

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

model=DecisionTreeClassifier()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_hat))
print('Matrice de confusion:')#混淆矩阵：用来展示预测结果和真实标签之间的对比。[[TN  FP],[FN  TP]]
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model,X_test,y_test)


from sklearn.model_selection import *

df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')

X = df.drop(['output'], axis = 1)
y = df['output'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')

X = df.drop(['output'], axis = 1)
y = df['output'].values
#Arbre de décision
print('Arbre de Décision')
model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

#Régression Logistique
print('Régression Logistique')
model = LogisticRegression()
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')

X = df.drop(['output'], axis = 1)
y = df['output'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

#Arbre de décision
print('Arbre de Décision niveau 3')
model = DecisionTreeClassifier(max_depth = 3)
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

print('Arbre de Décision niveau 5')
model = DecisionTreeClassifier(max_depth = 5)
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

print('Arbre de Décision niveau 10')
model = DecisionTreeClassifier(max_depth = 10)
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

print('Arbre de Décision niveau 100')
model = DecisionTreeClassifier(max_depth = 100)
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())



df = pd.read_csv('/kaggle/input/basic-datasets/cancer.csv')
df.head()


df = df.drop(['id', 'Unnamed: 32'], axis = 1)
df['diagnosis'] = df['diagnosis'].map({'B':0, 'M':1})#用来对列的每个元素进行转换
df.head()


X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


df = pd.read_csv('/kaggle/input/basic-datasets/penguins.csv')
df.head()


sns.pairplot(df[df.island == 'Biscoe'], hue = 'species')


sns.pairplot(df,hue='sex')


df.info()


df = df.dropna()#用以删除缺失值NaN


df['sex']=df['sex'].map({'male':0, 'female':1})


df=pd.get_dummies(df, columns=['island','species'])#多类别分类
df.head()


#Train / Test
X = df.drop('sex', axis=1)
y = df['sex']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Modèle Arbre de décision
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

#Métriques d'évaluation 类别用准确率
print("Accuracy:", accuracy_score(y_test, y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)

#Modèle LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

#Métriques d'évaluation
print("Accuracy:", accuracy_score(y_test, y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


def mse(y, u):
    # Mean Squared Error, erreur quadratique
    return np.mean((y - u)**2)

def mae(y, u):
    # Mean Absoluted Error
    return np.mean(abs(y - u))

def mape(y, u):
    # Mean Absoluted Percentage Error
    return np.mean(abs(y - u) / y)

def score_r2(y, u):
    # 
    return 1 - np.sum((y - u)**2) / np.sum((y - y.mean())**2)


#Train / Test
X = df.drop('flipper_length_mm', axis=1)
y = df['flipper_length_mm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Modèle LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

#Métriques d'évaluation 针对预测数值
print('RMSE : ', np.sqrt(mean_squared_error(y_test, y_hat)))
print('MAE : ', mean_absolute_error(y_test, y_hat))
print('MAPE : ', mean_absolute_percentage_error(y_test, y_hat))
print('Score R2 : ', r2_score(y_test, y_hat))

plt.scatter(y_test, y_hat)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c = 'red')


from sklearn.tree import *

#Train / Test
X = df.drop('flipper_length_mm', axis=1)
y = df['flipper_length_mm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Modèle Arbre de décission 'regressor'
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

#Métriques d'évaluation
print('RMSE : ', np.sqrt(mean_squared_error(y_test, y_hat)))
print('MAE : ', mean_absolute_error(y_test, y_hat))
print('MAPE : ', mean_absolute_percentage_error(y_test, y_hat))
print('Score R2 : ', r2_score(y_test, y_hat))

plt.scatter(y_test, y_hat)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c = 'red')


#Dataset
df = pd.read_csv('/kaggle/input/basic-datasets/penguins.csv')

#Préparation des données
df = df.dropna() #Supprime les espaces avec 'NaN'
df['sex'] = df['sex'].map({'male':0, 'female':1})
df = pd.get_dummies(df, columns = ['island', 'species'])

#Train / Test
X = df.drop('sex', axis=1)
y = df['sex']

#Modèle Arbre de décision
model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne:', scores.mean())
print('Ecart type :', scores.std())

#Forêt aléatoire
print('Forêt aléatoire')
model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne:', scores.mean())
print('Ecart type :', scores.std())

#Etreme Gradient Boosting
print('Extreme Gradient Boosting')
model = XGBClassifier()
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne:', scores.mean())
print('Ecart type :', scores.std())

#Régression Logistique
print('Régression Logistique')
model = LogisticRegression()
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


#Dataset
df = pd.read_csv('/kaggle/input/basic-datasets/penguins.csv')

#Préparation des données
df = df.dropna() #Supprime les espaces avec 'NaN'
df['sex'] = df['sex'].map({'male':0, 'female':1})
df = pd.get_dummies(df, columns = ['island', 'species'])

#Train / Test
X = df.drop('flipper_length_mm', axis=1)
y = df['flipper_length_mm']

#Modèle Arbre de décision
print('Arbre de décision')
model = DecisionTreeRegressor()
scores = cross_val_score(model, X, y, cv = 20, scoring = 'r2')
print('Accuracy moyenne:', scores.mean())
print('Ecart type :', scores.std())

#Forêt aléatoire
print('Forêt aléatoire')
model = RandomForestRegressor()
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne:', scores.mean())
print('Ecart type :', scores.std())

#Etreme Gradient Boosting
print('Extreme Gradient Boosting')
model = XGBRegressor()
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne:', scores.mean())
print('Ecart type :', scores.std())

#Régression Logistique
print('Régression Logistique')
model = LinearRegression()
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


df = pd.read_csv('/kaggle/input/basic-datasets/titanic.csv')
df.head(15)


df.info()


df.fillna(0)#填缺失值为0


df['Age']=df['Age'].fillna(df['Age'].mean())


df['Embarked'].value_counts()


#Préparation des données
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df = df.drop(['PassengerId', 'Name', 'Ticket','Cabin', 'Embarked'], axis = 1)

#Train / Test
X = df.drop('Survived', axis=1)
y = df['Survived']

#Modèle Arbre de décision
model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne:', scores.mean())
print('Ecart type :', scores.std())

#Forêt aléatoire
print('Forêt aléatoire')
model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne:', scores.mean())
print('Ecart type :', scores.std())

#Etreme Gradient Boosting
print('Extreme Gradient Boosting')
model = XGBClassifier()
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne:', scores.mean())
print('Ecart type :', scores.std())

#Régression Logistique
print('Régression Logistique')
model = LogisticRegression()
scores = cross_val_score(model, X, y, cv = 100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


df = pd.read_csv('/kaggle/input/basic-datasets/churn-small.csv')
df.head()


df = pd.read_csv('/kaggle/input/basic-datasets/churn-big.csv')

#Préparation des données
df = pd.get_dummies(df, columns = ['State'])
df['International plan'] = df['International plan'].map({'No':0, 'Yes':1})
df['Voice mail plan'] = df['Voice mail plan'].map({'No':0, 'Yes':1})

#Train / Test
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

scaler = StandardScaler()
X = scaler.fit_transform(X)

#Modèle Arbre de décision
model = LogisticRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

#Métriques d'évaluation
print("Accuracy:", accuracy_score(y_test, y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


df = pd.read_csv('/kaggle/input/credit-card-fraud-prediction/train.csv')

#Préparation des données
df = df.drop(['id', 'Time'], axis = 1)

#Train / Test
X = df.drop('IsFraud', axis=1)
y = df['IsFraud'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Modèle Arbre de décision
model = LogisticRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

#Métriques d'évaluation
print("Accuracy:", accuracy_score(y_test, y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


df.shape


df['IsFraud'].value_counts()


#!pip install -qU scikit-learn imbalanced-learn


#from imblearn.under_sampling import RandomUnderSampler
#sampler=RandomUnderSampler()
#X,y=sampler.fit_resample(X,y)


