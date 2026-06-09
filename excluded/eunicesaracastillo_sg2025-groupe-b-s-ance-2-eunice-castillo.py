import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns

pd.options.display.max_columns = None

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *

from xgboost import *

from sklearn.metrics import * 


df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')

X = df.drop(['output'], axis = 1)
y = df['output'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


model.predict_proba(X_test)


from sklearn.model_selection import *
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')

X = df.drop(['output'], axis = 1)
y = df['output'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


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


from sklearn.model_selection import *
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')

X = df.drop(['output'], axis = 1)
y = df['output'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

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
df['diagnosis'] = df['diagnosis'].map({'B':0, 'M':1})
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


sns.pairplot(df, hue = 'sex')


df.info()


df = df.dropna() #Supprime les espaces avec 'NaN'
df.info()


df['sex'] = df['sex'].map({'male':0, 'female':1})
df.head()


df = pd.get_dummies(df, columns = ['island', 'species'])
df.head()


#Train / Test
X = df.drop('sex', axis=1)
y = df['sex']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Modèle Arbre de décision
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

#Métriques d'évaluation
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


#Train / Test
X = df.drop('flipper_length_mm', axis=1)
y = df['flipper_length_mm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Modèle LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

#Métriques d'évaluation
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


df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'].value_counts()
df['Embarked'] = df['Embarked'].fillna('S')
df.head()


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





df = pd.read_csv('/kaggle/input/basic-datasets/churn-big.csv')
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
df.head()


from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv('/kaggle/input/credit-card-fraud-prediction/train.csv')

#Préparation des données
df = df.drop(['id', 'Time'], axis = 1)

#Train / Test
X = df.drop('IsFraud', axis=1)
y = df['IsFraud']

scaler = StandardScaler()
X = scaler.fit_transform(X)

sampler = RandomUnderSampler()
X,y = sampler.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Modèle Arbre de décision
model = RandomForestClassifier()
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


!pip install -qU scikit-learn imbalanced-learn


from imblearn.under_sampling import RandomUnderSampler

sampler = RandomUnderSampler()

X,y = sampler.fit_resample(X, y)


y.value_counts()


from imblearn.over_sampling import RandomOverSampler

sampler = RandomOverSampler()

X,y = sampler.fit_resample(X, y)


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv('/kaggle/input/credit-card-fraud-prediction/train.csv')

#Préparation des données
df = df.drop(['id', 'Time'], axis = 1)

#Train / Test
X = df.drop('IsFraud', axis=1)
y = df['IsFraud']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

sampler = RandomOverSampler()
X_train,y_train = sampler.fit_resample(X_train, y_train)

sampler = RandomUnderSampler()
X_test, y_test = sampler.fit_resample(X_test, y_test)

#Modèle Arbre de décision
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

#Métriques d'évaluation
print("Accuracy:", accuracy_score(y_test, y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

df = pd.read_csv('/kaggle/input/credit-card-fraud-prediction/train.csv')

#Préparation des données
df = df.drop(['id', 'Time'], axis = 1)

#Train / Test
X = df.drop('IsFraud', axis=1)
y = df['IsFraud']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

sampler = SMOTE()
X_train,y_train = sampler.fit_resample(X_train, y_train)

sampler = RandomUnderSampler()
X_test, y_test = sampler.fit_resample(X_test, y_test)

#Modèle Arbre de décision
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

#Métriques d'évaluation
print("Accuracy:", accuracy_score(y_test, y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)

