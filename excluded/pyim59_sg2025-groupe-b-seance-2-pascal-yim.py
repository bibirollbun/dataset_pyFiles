import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns

pd.options.display.max_columns=None

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import *
from sklearn.ensemble import *

from xgboost import *

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import *
from sklearn.metrics import *





from sklearn.model_selection import train_test_split

df = pd.read_csv("/kaggle/input/basic-datasets/heart.csv")

X = df.drop(["output"], axis=1)
y = df["output"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train,y_train)
y_hat = model.predict(X_test)

print('Accuracy :', accuracy_score(y_test,y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


model.predict_proba(X_test)


from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("/kaggle/input/basic-datasets/heart.csv")

X = df.drop(["output"], axis=1)
y = df["output"].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_hat = model.predict(X_test)

print('Accuracy :', accuracy_score(y_test,y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


from sklearn.model_selection import *

df = pd.read_csv("/kaggle/input/basic-datasets/heart.csv")

X = df.drop(["output"], axis=1)
y = df["output"].values

# Arbre de décision
print('Arbre de décision')

model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())

# Régression Logisitique
print('Regression logistique')

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())



df = pd.read_csv("/kaggle/input/basic-datasets/heart.csv")

X = df.drop(["output"], axis=1)
y = df["output"].values

# Arbre de décision
print('Arbre de décision niveau 3')

model = DecisionTreeClassifier(max_depth=3)
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())

# Arbre de décision
print('Arbre de décision niveau 5')

model = DecisionTreeClassifier(max_depth=5)
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())

# Arbre de décision
print('Arbre de décision niveau 10')

model = DecisionTreeClassifier(max_depth=10)
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())

# Arbre de décision
print('Arbre de décision niveau 100')

model = DecisionTreeClassifier(max_depth=100)
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())



df = pd.read_csv("/kaggle/input/basic-datasets/cancer.csv")
df.head()


df = pd.read_csv("/kaggle/input/basic-datasets/cancer.csv")

# Préparation des données
df = df.drop(['id', 'Unnamed: 32'], axis=1)
df['diagnosis'] = df['diagnosis'].map({'B':0, 'M':1})

X = df.drop(["diagnosis"], axis=1)
y = df["diagnosis"].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_hat = model.predict(X_test)

print('Accuracy :', accuracy_score(y_test,y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


df = pd.read_csv("/kaggle/input/basic-datasets/penguins.csv")
df.head()


sns.pairplot(df, hue='sex')


df.info()


df = df.dropna()


df['sex'] = df['sex'].map({'male':0, 'female':1})


df = pd.get_dummies(df, columns=['island', 'species'])


df.head()


df = pd.read_csv("/kaggle/input/basic-datasets/penguins.csv")

# Préparation des donnéeds
df = df.dropna()
df['sex'] = df['sex'].map({'male':0, 'female':1})
df = pd.get_dummies(df, columns=['island', 'species'])

# Train/Test
X = df.drop(["sex"], axis=1)
y = df["sex"].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# Modèle Arbre de décision
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_hat = model.predict(X_test)

# Métriques d'évaluation
print('Accuracy :', accuracy_score(y_test,y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


df = pd.read_csv("/kaggle/input/basic-datasets/penguins.csv")

# Préparation des donnéeds
df = df.dropna()
df['sex'] = df['sex'].map({'male':0, 'female':1})
df = pd.get_dummies(df, columns=['island', 'species'])

# Train/Test
X = df.drop(["sex"], axis=1)
y = df["sex"].values

# Arbre de décision
print('Arbre de décision')

model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())

# Forêt aléatoire
print('Forêt aléatoire')

model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())

# Extreme Gradient Boosting
print('Extreme Gradient Boosting')

model = XGBClassifier()
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())

# Régression Logisitique
print('Regression logistique')

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())



from sklearn.tree import *

df = pd.read_csv("/kaggle/input/basic-datasets/penguins.csv")

# Préparation des donnéeds
df = df.dropna()
df['sex'] = df['sex'].map({'male':0, 'female':1})
df = pd.get_dummies(df, columns=['island', 'species'])

# Train/Test
X = df.drop(["flipper_length_mm"], axis=1)
y = df["flipper_length_mm"].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# Modèle Arbre de décision
model = DecisionTreeRegressor()
model.fit(X_train,y_train)
y_hat = model.predict(X_test)

# Métriques d'évaluation
print('RMSE : ', np.sqrt(mean_squared_error(y_test, y_hat)))
print('MAE : ', mean_absolute_error(y_test, y_hat))
print('MAPE : ', mean_absolute_percentage_error(y_test, y_hat))
print('Score R2 : ', r2_score(y_test, y_hat))

plt.scatter(y_test, y_hat)
plt.plot([y_test.min(),y_test.max()], [y_test.min(),y_test.max()], c='red')


df = pd.read_csv("/kaggle/input/basic-datasets/penguins.csv")

# Préparation des donnéeds
df = df.dropna()
df['sex'] = df['sex'].map({'male':0, 'female':1})
df = pd.get_dummies(df, columns=['island', 'species'])

# Train/Test
X = df.drop(["flipper_length_mm"], axis=1)
y = df["flipper_length_mm"].values

# Arbre de décision
print('Arbre de décision')

model = DecisionTreeRegressor()
scores = cross_val_score(model, X, y, cv=20,scoring='r2')
print('Accuracy moyenne : ', -scores.mean())
print('Ecart type : ', scores.std())

# Forêt aléatoire
print('Forêt aléatoire')

model = RandomForestRegressor()
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())

# Extreme Gradient Boosting
print('Extreme Gradient Boosting')

model = XGBRegressor()
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())

# Régression Logisitique
print('Regression linéaire')


model = LinearRegression()
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())



df = pd.read_csv("/kaggle/input/basic-datasets/titanic.csv")
df.head()


df.info()


df['Age'] = df['Age'].fillna(df['Age'].mean())


df['Embarked'].value_counts()


df.columns


df = pd.read_csv("/kaggle/input/basic-datasets/titanic.csv")

# Préparation des donnéeds
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df = pd.get_dummies(df, columns=['Embarked'])
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1)

# Train/Test
X = df.drop(["Survived"], axis=1)
y = df["Survived"].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# Modèle Arbre de décision
model = XGBClassifier()
model.fit(X_train,y_train)
y_hat = model.predict(X_test)

# Métriques d'évaluation
print('Accuracy :', accuracy_score(y_test,y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


df = pd.read_csv("/kaggle/input/basic-datasets/churn-big.csv")
df.head()


df.shape


df = pd.read_csv("/kaggle/input/basic-datasets/churn-small.csv")

# Préparation des donnéeds

df['International plan'] = df['International plan'].map({'No':0, 'Yes':1})
df['Voice mail plan'] = df['Voice mail plan'].map({'No':0, 'Yes':1})
df = pd.get_dummies(df, columns=['State'])


# Train/Test
X = df.drop(["Churn"], axis=1)
y = df["Churn"].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Modèle Arbre de décision
model = LogisticRegression()
model.fit(X_train,y_train)
y_hat = model.predict(X_test)

# Métriques d'évaluation
print('Accuracy :', accuracy_score(y_test,y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)





df = pd.read_csv("/kaggle/input/credit-card-fraud-prediction/train.csv")


df.head()


df.shape


df['IsFraud'].value_counts()


df = pd.read_csv("/kaggle/input/credit-card-fraud-prediction/train.csv")

# Préparation des données
df = df.drop(['id','Time'], axis=1)

# Train/Test
X = df.drop(["IsFraud"], axis=1)
y = df["IsFraud"].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Modèle Arbre de décision
model = LogisticRegression()
model.fit(X_train,y_train)
y_hat = model.predict(X_test)

# Métriques d'évaluation
print('Accuracy :', accuracy_score(y_test,y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


!pip install -qU scikit-learn imbalanced-learn




