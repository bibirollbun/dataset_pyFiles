import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns

pd.options.display.max_columns=None

from sklearn.linear_model import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

from sklearn.model_selection import train_test_split

from sklearn.ensemble import *

from xgboost import *


from sklearn.linear_model import LogisticRegression

df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')
X = df.drop(['output'],axis=1)
y = df['output'].values

scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y)


model = LogisticRegression()
model.fit(X_train,y_train)
y_hat = np.round(model.predict(X_test))

print('Accuracy :',accuracy_score(y_test,y_hat))

## On regarde si l'examen est fiable

print('Matrice de confusion :')
print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test,y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


model.predict_proba(X_test)


df.sample(frac=0.2)


from sklearn.model_selection import *

df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')
X = df.drop(['output'],axis=1)
y = df['output'].values

scaler=StandardScaler()
X=scaler.fit_transform(X)

model = LogisticRegression()
scores = cross_val_score(model,X,y, cv=100)

print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())



## Comparaison du neurone LogisticRegression avec DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')
X = df.drop(['output'],axis=1)
y = df['output'].values


X_train,X_test,y_train,y_test = train_test_split(X,y)


model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_hat = np.round(model.predict(X_test))

print('Accuracy :',accuracy_score(y_test,y_hat))

## On regarde si l'examen est fiable

print('Matrice de confusion :')
print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test,y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')
X = df.drop(['output'],axis=1)
y = df['output'].values


print('Arbre de decision :')
model = DecisionTreeClassifier()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

print('LogisticRegression :')
model = LogisticRegression()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')
X = df.drop(['output'],axis=1)
y = df['output'].values


print('Arbre de decision depth=3 :')
model = DecisionTreeClassifier(max_depth=3)
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

print('Arbre de decision depth=5 :')
model = DecisionTreeClassifier(max_depth=5)
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

print('Arbre de decision depth=10 :')
model = DecisionTreeClassifier(max_depth=10)
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

print('Arbre de decision depth=100 :')
model = DecisionTreeClassifier(max_depth=100)
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


df = pd.read_csv('/kaggle/input/basic-datasets/cancer.csv')
df.head()


df.columns


df = df.drop(['Unnamed: 32' ,'id'], axis=1)


df['diagnosis'] = df['diagnosis'].map({'B':0,'M':1})
df.head()


X = df.drop(['diagnosis'],axis=1)
y = df['diagnosis'].values

scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y)


model = LogisticRegression()
model.fit(X_train,y_train)
y_hat = np.round(model.predict(X_test))

print('Accuracy :',accuracy_score(y_test,y_hat))

## On regarde si l'examen est fiable

print('Matrice de confusion :')
print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test,y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


df = pd.read_csv('/kaggle/input/basic-datasets/penguins.csv')




######    Traitement du df penguins   #######



## Enlève les lignes avec un NaN
df = df.dropna()

##On passe le sexe en binaire
df['sex'] = df['sex'].map({'male':0,'female':1})

## Encodage binaire
df = pd.get_dummies(df,columns=['island','species'])


df.info()



df.head()


sns.pairplot(df[df.island=='Biscoe'], hue='species')


## Prediction sexe

X = df.drop(['sex'],axis=1)
y = df['sex'].values

scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y)


model = LogisticRegression()
model.fit(X_train,y_train)
y_hat = np.round(model.predict(X_test))

print('Accuracy :',accuracy_score(y_test,y_hat))

## On regarde si l'examen est fiable

print('Matrice de confusion :')
print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test,y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


##Prediction taille nageoire

X = df.drop(['flipper_length_mm'],axis=1)
y = df['flipper_length_mm'].values

scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y)


model = LogisticRegression()
model.fit(X_train,y_train)
y_hat = np.round(model.predict(X_test))

print('Accuracy :',accuracy_score(y_test,y_hat))

## On regarde si l'examen est fiable

print('Matrice de confusion :')
print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test,y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


# Trop peu de representants par classes (1,2,...) donc regression linéaire


X = df.drop(['flipper_length_mm'],axis=1)
y = df['flipper_length_mm'].values

scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y)


model = LinearRegression()
model.fit(X_train,y_train)
y_hat = np.round(model.predict(X_test))


## On regarde si l'examen est fiable

print('RMSE : ', np.sqrt(mean_squared_error(y_test, y_hat)))
print('MAE : ', mean_absolute_error(y_test, y_hat))
print('MAPE : ', mean_absolute_percentage_error(y_test, y_hat))
print('Score R2 : ', r2_score(y_test, y_hat))

plt.scatter(y_test,y_hat)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()], c='red')


from sklearn.tree import *

X = df.drop(['flipper_length_mm'],axis=1)
y = df['flipper_length_mm'].values

scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y)


model = DecisionTreeRegressor()
model.fit(X_train,y_train)
y_hat = np.round(model.predict(X_test))


## On regarde si l'examen est fiable

print('RMSE : ', np.sqrt(mean_squared_error(y_test, y_hat)))
print('MAE : ', mean_absolute_error(y_test, y_hat))
print('MAPE : ', mean_absolute_percentage_error(y_test, y_hat))
print('Score R2 : ', r2_score(y_test, y_hat))

plt.scatter(y_test,y_hat)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()], c='red')


df = pd.read_csv('/kaggle/input/basic-datasets/penguins.csv')

## Enlève les lignes avec un NaN
df = df.dropna()

##On passe le sexe en binaire
df['sex'] = df['sex'].map({'male':0,'female':1})

## Encodage binaire
df = pd.get_dummies(df,columns=['island','species'])


X = df.drop(['sex'],axis=1)
y = df['sex'].values


## Arbre de Décision

print('Arbre de decision :')
model = DecisionTreeClassifier()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


## Forêt Aléatoire

print('Forêt aléatoire :')
model = RandomForestClassifier()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


## Extreme Gradient Boosting

print('Extreme Gradient Boosting :')
model = XGBClassifier()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


## Regression Logistique

print('Regression Logistique :')
model = LogisticRegression()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


df = pd.read_csv('/kaggle/input/basic-datasets/penguins.csv')

## Enlève les lignes avec un NaN
df = df.dropna()

##On passe le sexe en binaire
df['sex'] = df['sex'].map({'male':0,'female':1})

## Encodage binaire
df = pd.get_dummies(df,columns=['island','species'])


X = df.drop(['flipper_length_mm'],axis=1)
y = df['flipper_length_mm'].values


## Arbre de Décision

print('Arbre de decision :')
model = DecisionTreeRegressor()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


## Forêt Aléatoire

print('Forêt aléatoire :')
model = RandomForestRegressor()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


## Extreme Gradient Boosting

print('Extreme Gradient Boosting :')
model = XGBRegressor()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


## Regression Logistique

print('Regression Logistique :')
model = LogisticRegression()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


df = pd.read_csv('/kaggle/input/basic-datasets/titanic.csv')
df.head()


df['Age'] = df['Age'].fillna(df['Age'].mean())
df = df.dropna()
df = df.drop(['Name','Ticket'],axis=1)
df['Sex'] = df['Sex'].map({'female':1, 'male':0})
df['Embarked'] = df['Embarked'].map({'C':1, 'S':0, 'Q':2})


df.head()


df['Cabin'].value_counts()


X = df.drop(['Survived'],axis=1)
y = df['Survived'].values


## Arbre de Décision

print('Arbre de decision :')
model = DecisionTreeClassifier()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())



df = pd.read_csv('/kaggle/input/basic-datasets/churn-big.csv')
df['International plan'] = df['International plan'].map({'Yes':1, 'No':0})
df['Voice mail plan'] = df['Voice mail plan'].map({'Yes':1, 'No':0})
df = df.drop(['State'], axis=1)
# ou alors get_dummies (mais étrangement, les résultats sont meilleurs sans le get_dummies : 0.95% VS 0.93% au XGBC )
df.head()


df.shape


pd.read_csv('/kaggle/input/basic-datasets/churn-big.csv').head()


X = df.drop(['Churn'],axis=1)
y = df['Churn'].values



## Arbre de Décision

print('Arbre de decision :')
model = DecisionTreeClassifier()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


## Forêt Aléatoire

print('Forêt aléatoire :')
model = RandomForestClassifier()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


## Extreme Gradient Boosting

print('Extreme Gradient Boosting :')
model = XGBClassifier()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


## Regression Logistique

print('Regression Logistique :')
model = LogisticRegression()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


df = pd.read_csv('/kaggle/input/credit-card-fraud-prediction/train.csv')
df = df.drop(['id','Time'], axis = 1)
df.head()


##Sans drop id et time

X = df.drop(['IsFraud'],axis=1)
y = df['IsFraud'].values


## Regression Logistique

print('Regression Logistique :')
model = LogisticRegression()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test,y_hat))


##En dropant id et time

X = df.drop(['IsFraud'],axis=1)
y = df['IsFraud'].values


## Regression Logistique

print('Regression Logistique :')
model = LogisticRegression()
scores = cross_val_score(model,X,y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


## On force l'installation de imb (mal implantée dans kaggle)
!pip install -qU scikit-learn imbalanced-learn


?

sous echantilloner train
sur echantilloner test

oversampler duplique les lignes...
smote créer des nouvelles données inspirées de celles présentes

