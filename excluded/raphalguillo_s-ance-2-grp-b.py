import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

pd.options.display.max_columns=None

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import *
from sklearn.ensemble import *

from xgboost import *


df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')
x = df.drop(['output'], axis=1)
y = df['output'].values

scaler = StandardScaler() # Mise à l'échelle pour la descente de gradient
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3) # On sépare la base de donnée en 2 : une partie entraînement et une partie test

model = LogisticRegression()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)

print('Accuracy :', accuracy_score(y_test,y_hat)) # Attention : accuracy sur y_test


# On peut séparer en 2 la base de données, entraîner le model sur 50% des données et le tester sur les 50% restants.
# Ou on peut tester sur une nouvelle base de donnée.


# On obtient des résultats différents car le set de données test change à chaque appel.
# Pour obtenir un grand nombre de valeurs on peut utiliser cross_val_score :


scores = cross_val_score(model, x, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())


# Le résultat est toujours le même : c'est la valeur qu'on peut donner au client
# On peut augmenter le nombre de tirages avec cv=10 ou 100 mais attention l'écart-type est plus élevé


print('Matrice de confusion :')
print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test,y_hat))


model.predict_proba(x_test) # Pour être sûr de ce que le model dit : 0,99 à gauche pour 0,01 à droite >> très sûr


# Courbe ROC : p.16 diapo 3 (NB. AUC = Area Under Curve)
RocCurveDisplay.from_estimator(model, x_test, y_test)


from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')
x = df.drop(['output'], axis=1)
y = df['output'].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3) # On sépare la base de donnée en 2 : une partie entraînement et une partie test

model = DecisionTreeClassifier()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)

print('Accuracy :', accuracy_score(y_test,y_hat)) # Attention : accuracy sur y_test


scores = cross_val_score(model, x, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())
print('Matrice de confusion :')
print(confusion_matrix(y_test,y_hat))


from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')
x = df.drop(['output'], axis=1)
y = df['output'].values

#Avec DecisionTreeClassifier
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)
scores = cross_val_score(model, x, y, cv=100)
print('Accuracy moyenne dectree: ', scores.mean())
print('Ecart type : ', scores.std())

#Ou avec LogistiqueRegression
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
model = LogisticRegression()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)
scores = cross_val_score(model, x, y, cv=100)
print('Accuracy moyenne reglog : ', scores.mean())
print('Ecart type : ', scores.std())


from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')
x = df.drop(['output'], axis=1)
y = df['output'].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3) # On sépare la base de donnée en 2 : une partie entraînement et une partie test

model = DecisionTreeClassifier(max_depth=500)
scores = cross_val_score(model, x, y, cv=100)
print('Accuracy moyenne arbre de décision niveau 500 : ', scores.mean())
print('Ecart type : ', scores.std())


df = pd.read_csv('/kaggle/input/basic-datasets/cancer.csv') # Importe le dataset cancer
# Préparation des données
df = df.drop(['id','Unnamed: 32'], axis=1) # Retire la colonne inconnue
df['diagnosis'] = df['diagnosis'].map({'B':0, 'M':1}) # Change les lettres en chiffre
df.columns


x = df.drop(['diagnosis'], axis=1) # Prend toutes les colonnes autres que diagnosis
y = df['diagnosis'].values # Prend la colonne diagnosis

scaler = StandardScaler() # Mise à l'échelle pour la descente de gradient
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
model = LogisticRegression()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)

# print('Accuracy :', accuracy_score(y_test,y_hat))
scores = cross_val_score(model, x, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())
# Matrice de confusion
print('Matrice de confusion :')
print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test,y_hat))
# Courbe ROC
RocCurveDisplay.from_estimator(model, x_test, y_test)


df = pd.read_csv('/kaggle/input/basic-datasets/penguins.csv') # Importe le dataset pingouins
df.head()
df.columns
df.info


x = df.drop(['species'], axis=1) # Prend toutes les colonnes autres que species
y = df['species'].values # Prend la colonne species

scaler = StandardScaler() # Mise à l'échelle pour la descente de gradient
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
model = LogisticRegression()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)

# print('Accuracy :', accuracy_score(y_test,y_hat))
scores = cross_val_score(model, x, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())
# Matrice de confusion
print('Matrice de confusion :')
print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test,y_hat))
# Courbe ROC
RocCurveDisplay.from_estimator(model, x_test, y_test)


sns.pairplot(df, hue='species')


sns.pairplot(data=df,hue='island')


# On filtre pour ne faire apparaître qu'une espèce :
sns.pairplot(df[df.species=='Gentoo'],hue='island')


# Les Gentoo ne sont que sur l'île Biscoe
# On regarde quelle espèce est sur une des îles
sns.pairplot(df[df.island=='Dream'],hue='species')


df.info()
# On voit qu'il manque des valeurs dans sex notamment


df = df.dropna() # Supprime les lignes où il y a un Nan
df.head()


df['sex'] = df['sex'].map({'male':0,'female':1}) # Modifie les mots par des chiffres
df = pd.get_dummies(df, columns=['island','species']) # Crée une colonne par île : Vrai si l'île est Biscoe (p.ex)
df.head()


x = df.drop(['sex'], axis=1) # Prend toutes les colonnes autres que sex
y = df['sex'].values # Prend la colonne sex

scaler = StandardScaler() # Mise à l'échelle pour la descente de gradient
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
model = LogisticRegression()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)


scores = cross_val_score(model, x, y, cv=100)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())
# Matrice de confusion
print('Matrice de confusion :')
print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test,y_hat))
# Courbe ROC
RocCurveDisplay.from_estimator(model, x_test, y_test)


x = df.drop(['flipper_length_mm'], axis=1) # Prend toutes les colonnes autres que sex
y = df['flipper_length_mm'].values # Prend la colonne sex

scaler = StandardScaler() # Mise à l'échelle pour la descente de gradient
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
model = LogisticRegression()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)

scores = cross_val_score(model, x, y, cv=10)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())
# Matrice de confusion
print('Matrice de confusion :')
print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test,y_hat))
# Courbe ROC
RocCurveDisplay.from_estimator(model, x_test, y_test)


model = LinearRegression() #On crée une sorte de régression
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
model.fit(x_train,y_train)
y_hat = model.predict(x_test)

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

print('RMSE : ', np.sqrt(mean_squared_error(y_test, y_hat)))
print('MAE : ', mean_absolute_error(y_test, y_hat))
print('MAPE : ', mean_absolute_percentage_error(y_test, y_hat))
print('Score R2 : ', r2_score(y_test, y_hat))

plt.scatter(y_test,y_hat)
plt.plot([y_test.min(),y_test.max()], [y_test.min(),y_test.max()],c='red')


from sklearn.tree import *

model = DecisionTreeRegressor() #On crée une sorte de régression
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
model.fit(x_train,y_train)
y_hat = model.predict(x_test)


print('RMSE : ', np.sqrt(mean_squared_error(y_test, y_hat)))
print('MAE : ', mean_absolute_error(y_test, y_hat))
print('MAPE : ', mean_absolute_percentage_error(y_test, y_hat))
print('Score R2 : ', r2_score(y_test, y_hat))

plt.scatter(y_test,y_hat)
plt.plot([y_test.min(),y_test.max()], [y_test.min(),y_test.max()],c='red')


df = pd.read_csv('/kaggle/input/basic-datasets/penguins.csv') # Importe le dataset pingouins
df = df.dropna() # Supprime les lignes où il y a un Nan
df['sex'] = df['sex'].map({'male':0,'female':1}) # Modifie les mots par des chiffres
df = pd.get_dummies(df, columns=['island','species']) # Crée une colonne par île : Vrai si l'île est Biscoe (p.ex)

x = df.drop(['sex'], axis=1) # Prend toutes les colonnes autres que sex
y = df['sex'].values # Prend la colonne sex

scaler = StandardScaler() # Mise à l'échelle pour la descente de gradient
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

print('LogisticRegression')
model = LogisticRegression()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)
scores = cross_val_score(model, x, y, cv=100)
print('Accuracy moyenne LogReg : ', scores.mean())
print('Ecart type : ', scores.std())

print('Random Forest Classifier')
model = RandomForestClassifier()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)
scores = cross_val_score(model, x, y, cv=100)
print('Accuracy moyenne RandForClass : ', scores.mean())
print('Ecart type : ', scores.std())

# Extreme Gradient Boosting
print('Extreme Gradient Boosting')

model = XGBClassifier()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)
scores = cross_val_score(model, x, y, cv=100)
print('Accuracy moyenne XGB : ', scores.mean())
print('Ecart type : ', scores.std())


df = pd.read_csv('/kaggle/input/basic-datasets/penguins.csv') # Importe le dataset pingouins
df = df.dropna() # Supprime les lignes où il y a un Nan
df['sex'] = df['sex'].map({'male':0,'female':1}) # Modifie les mots par des chiffres
df = pd.get_dummies(df, columns=['island','species']) # Crée une colonne par île : Vrai si l'île est Biscoe (p.ex)

x = df.drop(['flipper_length_mm'], axis=1) # Prend toutes les colonnes autres que sex
y = df['flipper_length_mm'].values # Prend la colonne sex

scaler = StandardScaler() # Mise à l'échelle pour la descente de gradient
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

print('LinearRegression')
model = LinearRegression()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)
scores = cross_val_score(model, x, y, cv=100)
print('Accuracy moyenne LogReg : ', scores.mean())
print('Ecart type : ', scores.std())

print('DecisionTreeRegressor')
model = DecisionTreeRegressor()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)
scores = cross_val_score(model, x, y, cv=20,scoring='r2')
print('Accuracy moyenne LogReg : ', scores.mean())
print('Ecart type : ', scores.std())

print('Random Forest Regressor')
model = RandomForestRegressor()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)
scores = cross_val_score(model, x, y, cv=20)
print('Accuracy moyenne RandForClass : ', scores.mean())
print('Ecart type : ', scores.std())

# Extreme Gradient Boosting
print('Extreme Gradient Boosting')
model = XGBRegressor()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)
scores = cross_val_score(model, x, y, cv=100)
print('Accuracy moyenne XGB : ', scores.mean())
print('Ecart type : ', scores.std())


df = pd.read_csv('/kaggle/input/basic-datasets/titanic.csv')
df.head()


sns.pairplot(df)


df.info()


df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].map({'male':0,'female':1})
df = pd.get_dummies(df, columns=['Embarked'])
df = df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)


# Ou ça
df['Embarked'].value_counts()


x = df.drop(['Survived'], axis=1) # Prend toutes les colonnes autres que sex
y = df['Survived'].values # Prend la colonne sex

scaler = StandardScaler() # Mise à l'échelle pour la descente de gradient
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
model = XGBClassifier()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)
scores = cross_val_score(model, x, y, cv=100)
print('Accuracy moyenne XGB : ', scores.mean())
print('Ecart type : ', scores.std())


import pandas as pd
from sklearn.model_selection import *
from xgboost import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

df = pd.read_csv('/kaggle/input/basic-datasets/churn-big.csv')
#df.head()
#df.info()
#df.shape

#Préparation des données
df = pd.get_dummies(df, columns=['State'])
df['International plan'] = df['International plan'].map({'No':0,'Yes':1})
df['Voice mail plan'] = df['Voice mail plan'].map({'No':0,'Yes':1})

#Train / Test
x = df.drop(['Churn'], axis=1)
y = df['Churn'].values

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.3)

#Modèle arbre de décision
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)

#Métriques d'évaluation
scores = cross_val_score(model, x, y, cv=10)
print('Accuracy moyenne : ', scores.mean())
print('Ecart type : ', scores.std())
# Matrice de confusion
print('Matrice de confusion :')
print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test,y_hat))


import pandas as pd
df = pd.read_csv('/kaggle/input/credit-card-fraud-prediction/train.csv')
#AVEC RANDOM FOREST
#Préparation des données
df = df.drop(['id','Time'], axis=1)
x = df.drop(['IsFraud'], axis=1)
y = df['IsFraud'].values

scaler = StandardScaler() # Mise à l'échelle pour la descente de gradient
x = scaler.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#Modèle random forest
model = RandomForestClassifier()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)

#Métriques d'évaluation
print('Accuracy :', accuracy_score(y_test,y_hat))
print('Matrice de confusion :')
print(confusion_matrix_test,y_hat)
print(classification_report(y_test,y_hat))


!pip install -qU scikit-learn imbalanced-learn


from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import *


df = pd.read_csv('/kaggle/input/credit-card-fraud-prediction/train.csv')
#AVEC RANDOM FOREST
#Préparation des données
df = df.drop(['id','Time'], axis=1)
x = df.drop(['IsFraud'], axis=1)
y = df['IsFraud']

scaler = StandardScaler() # Mise à l'échelle pour la descente de gradient
x = scaler.fit_transform(x)

sampler = RandomUnderSampler()
x,y = sampler.fit_resample(x,y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#Modèle Arbre de décision
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
y_hat = model.predict(x_test)

#Métriques d'évaluation
print('Accuracy :', accuracy_score(y_test,y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test,y_hat))
RocCurveDisplay.from_estimator(model, x_test, y_test)




