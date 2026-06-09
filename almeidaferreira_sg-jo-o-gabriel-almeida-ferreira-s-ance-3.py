# Pour installer les bibliothÃ¨ques scikit-learn et imbalanced-learn
!pip install -qU scikit-learn imbalanced-learn


# Cartouche d'importation des modules

import matplotlib.pyplot as plt # Pour tracer des graphiques
import seaborn as sns           # Pour crÃ©er des graphiques esthÃ©tiques et simples
import pandas as pd             # Pour manipuler des tableaux de donnÃ©es
import numpy as np              # Pour effectuer des calculs numÃ©riques
import warnings                 # Pour gÃ©rer les avertissements

from sklearn.linear_model import LogisticRegression # Pour crÃ©er et entraÃ®ner un modÃ¨le de rÃ©gression logistique
from sklearn.linear_model import LinearRegression   # Pour crÃ©er et entraÃ®ner un modÃ¨le de rÃ©gression linÃ©aire
from sklearn.preprocessing import StandardScaler    # Pour normaliser les donnÃ©es
from sklearn.metrics import *                       # Pour Ã©valuer les performances du modÃ¨le

from sklearn.model_selection import *  # Import des fonctions pour train/test et validation croisÃ©e
from sklearn.tree import *             # Import des classes pour arbres de dÃ©cision
from sklearn.ensemble import *         # Pour utiliser des mÃ©thodes dâ€™ensemble 
from sklearn.metrics import *          # Pour calculer dâ€™autres mÃ©triques dâ€™Ã©valuation
from xgboost import *                  # Pour utiliser la librairie XGBoost 


# Import des techniques de rÃ©Ã©chantillonnage pour traiter le dÃ©sÃ©quilibre des classes
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

# Importer toutes les fonctions de Keras
from tensorflow.keras import *


# Cartouche de configuration

warnings.filterwarnings("ignore")     # Ignorer les avertissements
pd.options.display.max_columns = None # Afficher toutes les colonnes dâ€™un DataFrame


# Chargement du dataset
df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')

# DÃ©finition des variables
X = df.drop(['output'], axis=1)
y = df['output'].values

# Normalisation des donnÃ©es
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Division des donnÃ©es en ensembles d'entraÃ®nement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CrÃ©er et entraÃ®ner le modÃ¨le de rÃ©gression logistique
model = LogisticRegression()
model.fit(X_train, y_train)

# PrÃ©diction 
y_hat = model.predict(X_test)

# Validation croisÃ©e pour Ã©valuer la performance du modÃ¨le
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


# Affichage des mÃ©triques d'Ã©valuation du modÃ¨le
print('Accuracy :', accuracy_score(y_test, y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))


# PrÃ©diction des probabilitÃ©s sur l'ensemble de test
model.predict_proba(X_test)


# Affichage de la courbe ROC pour Ã©valuer la performance du modÃ¨le
RocCurveDisplay.from_estimator(model, X_test, y_test)


# Chargement du dataset
df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')

# DÃ©finition des variables (X : variables indÃ©pendantes, y : variable cible)
X = df.drop(['output'], axis=1)
y = df['output'].values

# Normalisation des donnÃ©es (pas encore nÃ©cessaire car certaines mÃ©thodes n'en ont pas besoin,
# mais utile pour des modÃ¨les sensibles aux Ã©chelles, comme la rÃ©gression logistique ou SVM)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# SÃ©paration en donnÃ©es d'entraÃ®nement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CrÃ©ation des modÃ¨les Ã  comparer
# 1. RÃ©gression logistique
model_1 = LogisticRegression()

# 2. Arbre de dÃ©cision
model = DecisionTreeClassifier()

# EntraÃ®nement du modÃ¨le choisi (ici : arbre de dÃ©cision)
model.fit(X_train, y_train)

# PrÃ©diction sur les donnÃ©es de test
y_hat = model.predict(X_test)

# Ã‰valuation du modÃ¨le
print("Accuracy:", accuracy_score(y_test, y_hat))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

# Affichage de la courbe ROC
RocCurveDisplay.from_estimator(model, X_test, y_test)


# Chargement du dataset
df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')

# DÃ©finition des variables
X = df.drop(['output'], axis=1)
y = df['output'].values

# Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# MÃ©thode 1 : Arbre de dÃ©cision
print("MÃ©thode 1 : Arbre de DÃ©cision")
model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv=100)
print("Accuracy moyenne :", scores.mean())
print("Ã‰cart type :", scores.std(), "\n")

#  MÃ©thode 2 : RÃ©gression logistique
print("MÃ©thode 2 : RÃ©gression Logistique")
model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=100)
print("Accuracy moyenne :", scores.mean())
print("Ã‰cart type :", scores.std())


# Chargement du dataset
df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')

# DÃ©finition des variables
X = df.drop(['output'], axis=1)
y = df['output'].values

# Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Arbre de dÃ©cision niveau 3
print("1. Arbre de DÃ©cision niveau 3")
model = DecisionTreeClassifier(max_depth=3)
scores = cross_val_score(model, X, y, cv=100)
print("Accuracy moyenne :", scores.mean())
print("Ã‰cart type :", scores.std(), "\n")

# Arbre de dÃ©cision niveau 5
print("2. Arbre de DÃ©cision niveau 5")
model = DecisionTreeClassifier(max_depth=5)
scores = cross_val_score(model, X, y, cv=100)
print("Accuracy moyenne :", scores.mean())
print("Ã‰cart type :", scores.std(), "\n")

# Arbre de dÃ©cision niveau 10
print("3. Arbre de DÃ©cision niveau 10")
model = DecisionTreeClassifier(max_depth=10)
scores = cross_val_score(model, X, y, cv=100)
print("Accuracy moyenne :", scores.mean())
print("Ã‰cart type :", scores.std(), "\n")

# Arbre de dÃ©cision niveau 100
print("4. Arbre de DÃ©cision niveau 100")
model = DecisionTreeClassifier(max_depth=100)
scores = cross_val_score(model, X, y, cv=100)
print("Accuracy moyenne :", scores.mean())
print("Ã‰cart type :", scores.std())


# Lire un fichier CSV et le mettre dans un DataFrame
df = pd.read_csv('/kaggle/input/basic-datasets/cancer.csv')


# Afficher les premiÃ¨res lignes du DataFrame
df.head()


# Suppression des colonnes inutiles
df = df.drop(['id', 'Unnamed: 32'], axis=1)

# Encodage de la variable cible
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})


# Afficher les premiÃ¨res lignes du DataFrame
df.head()


# DÃ©finition des variables
X = df.drop(["diagnosis"], axis=1)
y = df["diagnosis"].values

# Division des donnÃ©es en ensembles d'entraÃ®nement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CrÃ©er et entraÃ®ner le modÃ¨le d'arbre de dÃ©cision
model = LogisticRegression()
model.fit(X_train, y_train)

# Ã‰valuation du modÃ¨le
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne ;',scores.mean())
print('Ecart type ;',scores.std())

# Affichage de la courbe ROC
RocCurveDisplay.from_estimator(model, X_test, y_test)


# DÃ©finition des variables
X = df.drop(["diagnosis"], axis=1)
y = df["diagnosis"].values

# Division des donnÃ©es en ensembles d'entraÃ®nement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CrÃ©er et entraÃ®ner le modÃ¨le d'arbre de dÃ©cision
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

# Ã‰valuation du modÃ¨le
print("Accuracy:", accuracy_score(y_test, y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

# Affichage de la courbe ROC
RocCurveDisplay.from_estimator(model, X_test, y_test)


# Lire un fichier CSV et le mettre dans un DataFrame
df = pd.read_csv('/kaggle/input/basic-datasets/penguins.csv')


# Afficher les premiÃ¨res lignes du DataFrame
df.head()


# CrÃ©e une matrice de graphiques en nuage de points
sns.pairplot(df, hue='species')


# CrÃ©e une matrice de graphiques en nuage de points
sns.pairplot(df, hue='island')


# CrÃ©e une matrice de graphiques en nuage de points
sns.pairplot(df[df.species == 'Gentoo'], hue='island')


# CrÃ©e une matrice de graphiques en nuage de points
sns.pairplot(df[df.island == 'Biscoe'], hue='species')


# Affiche un rÃ©sumÃ© des colonnes, des types de donnÃ©es et du nombre de valeurs non nulles
df.info()


# PrÃ©-traitement des donnÃ©es
df = df.dropna()                                     # Supprime toutes les lignes contenant au moins un NaN
df['sex'] = df['sex'].map({'male': 0, 'female': 1})  # Encodage de la variable 'sex'


# Afficher les premiÃ¨res lignes du DataFrame
df.head()


# Encodage des variables catÃ©gorielles
df = pd.get_dummies(df, columns=['island', 'species'])


# Afficher les premiÃ¨res lignes du DataFrame
df.head()


# PrÃ©dire le sex
# DÃ©finition des variables
X = df.drop('sex', axis=1)
y = df['sex']

# Division des donnÃ©es en ensembles d'entraÃ®nement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ModÃ¨le Arbre de dÃ©cision
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

# Ã‰valuation du modÃ¨le Arbre de dÃ©cision
print("Accuracy:", accuracy_score(y_test, y_hat))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))
RocCurveDisplay.from_estimator(model, X_test, y_test)

# ModÃ¨le RÃ©gression logistique
model = LogisticRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

# Ã‰valuation du modÃ¨le RÃ©gression logistique
print("Accuracy:", accuracy_score(y_test, y_hat))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

# Affichage de la courbe ROC
RocCurveDisplay.from_estimator(model, X_test, y_test)


# PrÃ©dire le 'flipper_lenght_mm' avec rÃ©gression linÃ©aire
# DÃ©finition des variables
X = df.drop('flipper_length_mm', axis=1)
y = df['flipper_length_mm']

# Division des donnÃ©es en ensembles d'entraÃ®nement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ModÃ¨le RÃ©gression linÃ©aire
model = LinearRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

# Ã‰valuation du modÃ¨le
print('RMSE :', np.sqrt(mean_squared_error(y_test, y_hat)))
print('MAE :', mean_absolute_error(y_test, y_hat))
print('MAPE :', mean_absolute_percentage_error(y_test, y_hat))
print('Score R2 :', r2_score(y_test, y_hat))

# Graphique des valeurs rÃ©elles vs prÃ©dites
plt.scatter(y_test, y_hat)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c='red')


# PrÃ©dire le 'flipper_lenght_mm' avec arbre de dÃ©cision regressor
# DÃ©finition des variables
X = df.drop('flipper_length_mm', axis=1)
y = df['flipper_length_mm']

# Division des donnÃ©es en ensembles d'entraÃ®nement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ModÃ¨le Arbre de dÃ©cision pour rÃ©gression
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

# Ã‰valuation du modÃ¨le
print('RMSE :', np.sqrt(mean_squared_error(y_test, y_hat)))
print('MAE :', mean_absolute_error(y_test, y_hat))
print('MAPE :', mean_absolute_percentage_error(y_test, y_hat))
print('Score R2 :', r2_score(y_test, y_hat))

# Graphique des valeurs rÃ©elles vs prÃ©dites
plt.scatter(y_test, y_hat)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c='red')


# Sex / Classifiers
# Chargement du dataset
df = pd.read_csv('/kaggle/input/basic-datasets/penguins.csv')

# PrÃ©paration des donnÃ©es
df = df.dropna()  # Supprime les lignes contenant NaN
df['sex'] = df['sex'].map({'male': 0, 'female': 1})  # Encodage de la variable cible
df = pd.get_dummies(df, columns=['island', 'species'])  # Encodage des variables catÃ©gorielles

# DÃ©finition des variables
X = df.drop('sex', axis=1)
y = df['sex']

# ModÃ¨le ForÃªt alÃ©atoire
model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=100)
print('\nForÃªt alÃ©atoire')
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

# ModÃ¨le Extreme Gradient Boosting
model = XGBClassifier()
scores = cross_val_score(model, X, y, cv=100)
print('\nExtreme Gradient Boosting')
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

# ModÃ¨le RÃ©gression Logistique
model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=100)
print('\nRÃ©gression Logistique')
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

# ModÃ¨le Arbre de dÃ©cision
model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv=100)
print('\nArbre de dÃ©cision')
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

# Graphique des valeurs rÃ©elles vs prÃ©dites
plt.scatter(y_test, y_hat)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c='red')


# Flipper_length_mm / Regressors
# Chargement du dataset
df = pd.read_csv('/kaggle/input/basic-datasets/penguins.csv')

# PrÃ©paration des donnÃ©es
df = df.dropna()  # Supprime les lignes contenant NaN
df['sex'] = df['sex'].map({'male': 0, 'female': 1})  # Encodage de la variable 'sex'
df = pd.get_dummies(df, columns=['island', 'species'])  # Encodage des variables catÃ©gorielles

# DÃ©finition des variables
X = df.drop('flipper_length_mm', axis=1)
y = df['flipper_length_mm']

# ModÃ¨le Arbre de dÃ©cision
print('\nArbre de dÃ©cision')
model = DecisionTreeRegressor()
scores = cross_val_score(model, X, y, cv=20, scoring='r2')
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

# ModÃ¨le ForÃªt alÃ©atoire
print('\nForÃªt alÃ©atoire')
model = RandomForestRegressor()
scores = cross_val_score(model, X, y, cv=100, scoring='r2')
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

# ModÃ¨le Extreme Gradient Boosting
print('\nExtreme Gradient Boosting')
model = XGBRegressor()
scores = cross_val_score(model, X, y, cv=100, scoring='r2')
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

# ModÃ¨le RÃ©gression linÃ©aire
print('\nRÃ©gression linÃ©aire')
model = LinearRegression()
scores = cross_val_score(model, X, y, cv=100, scoring='r2')
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


# Lire un fichier CSV et le mettre dans un DataFrame
df = pd.read_csv('/kaggle/input/basic-datasets/titanic.csv')


# Afficher les premiÃ¨res lignes du DataFrame
df.head(20)


# Affiche un rÃ©sumÃ© des colonnes, des types de donnÃ©es et du nombre de valeurs non nulles
df.info()


# Traitement des donnÃ©es manquantes
df['Age'] = df['Age'].fillna(df['Age'].mean()) # Remplir les valeurs manquantes de 'Age' par la moyenne
df['Embarked'] = df['Embarked'].fillna('S')    # Remplir les valeurs manquantes de 'Embarked' par 'S'

# VÃ©rification de la distribution de 'Embarked'
df['Embarked'].value_counts()

# Affichage des premiÃ¨res lignes
df.head()


# PrÃ©paration des donnÃ©es
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# DÃ©finition des variables
X = df.drop('Survived', axis=1)
y = df['Survived']
# ModÃ¨le Arbre de dÃ©cision
model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv=100)
print('\nArbre de dÃ©cision')
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

# ModÃ¨le ForÃªt alÃ©atoire
model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=100)
print('\nForÃªt alÃ©atoire')
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

# ModÃ¨le Extreme Gradient Boosting
model = XGBClassifier()
scores = cross_val_score(model, X, y, cv=100)
print('\nExtreme Gradient Boosting')
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())

# ModÃ¨le RÃ©gression Logistique
model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=100)
print('\nRÃ©gression Logistique')
print('Accuracy moyenne :', scores.mean())
print('Ecart type :', scores.std())


# Lire un fichier CSV et le mettre dans un DataFrame
df = pd.read_csv('/kaggle/input/basic-datasets/churn-small.csv')


# Afficher les premiÃ¨res lignes du DataFrame
df.head()


# Taille du dataset
df.shape


# Chargement du dataset
df = pd.read_csv('/kaggle/input/basic-datasets/churn-big.csv')

# PrÃ©paration des donnÃ©es
df = pd.get_dummies(df, columns=['State'])
df['International plan'] = df['International plan'].map({'No': 0, 'Yes': 1})
df['Voice mail plan'] = df['Voice mail plan'].map({'No': 0, 'Yes': 1})

# DÃ©finition des variables
X = df.drop('Churn', axis=1)
y = df['Churn']

# Division des donnÃ©es en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalisation des donnÃ©es
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ModÃ¨le RÃ©gression Logistique
model = LogisticRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

# MÃ©triques d'Ã©valuation
print('\nAccuracy :', accuracy_score(y_test, y_hat))
print('\nMatrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print('\nRapport de classification :')
print(classification_report(y_test, y_hat))

# Courbe ROC
RocCurveDisplay.from_estimator(model, X_test, y_test)


# Chargement du dataset
df = pd.read_csv('/kaggle/input/credit-card-fraud-prediction/train.csv')

# PrÃ©paration des donnÃ©es
df = df.drop(['id', 'Time'], axis=1)

# DÃ©finition des variables
X = df.drop('IsFraud', axis=1)
y = df['IsFraud'].values

# Division des donnÃ©es en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ModÃ¨le RÃ©gression Logistique
model = LogisticRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

# MÃ©triques d'Ã©valuation
print('\nAccuracy :', accuracy_score(y_test, y_hat))
print('\nMatrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print('\nRapport de classification :')
print(classification_report(y_test, y_hat))

# Courbe ROC
RocCurveDisplay.from_estimator(model, X_test, y_test)


# Taille du dataset
df.shape


# Distribution de la variable cible
df['IsFraud'].value_counts()


# Chargement du dataset
df = pd.read_csv("/kaggle/input/credit-card-fraud-prediction/train.csv")

# PrÃ©paration des donnÃ©es
cols_to_drop = [c for c in ['id', 'Time'] if c in df.columns]
df = df.drop(columns=cols_to_drop)

# DÃ©finition des variables
X = df.drop(columns=["IsFraud"])
y = df["IsFraud"].values

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ModÃ¨le ForÃªt alÃ©atoire
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

# MÃ©triques d'Ã©valuation
print('\nAccuracy :', accuracy_score(y_test, y_hat))
print('\nMatrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print('\nRapport de classification :')
print(classification_report(y_test, y_hat))

# Courbe ROC
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show()


# Chargement du dataset
df = pd.read_csv("/kaggle/input/credit-card-fraud-prediction/train.csv")

# PrÃ©paration des donnÃ©es
df = df.drop(['id', 'Time'], axis=1)

# DÃ©finition des variables
X = df.drop(["IsFraud"], axis=1)
y = df["IsFraud"].values

# RÃ©Ã©chantillonnage pour Ã©quilibrer les classes
sampler = RandomOverSampler()
X, y = sampler.fit_resample(X, y)

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ModÃ¨le ForÃªt alÃ©atoire
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

# MÃ©triques d'Ã©valuation
print('\nAccuracy :', accuracy_score(y_test, y_hat))
print('\nMatrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print('\nRapport de classification :')
print(classification_report(y_test, y_hat))

# Courbe ROC
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show()


# Import des modules pour crÃ©er un rÃ©seau de neurones
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Chargement du dataset
df = pd.read_csv('/kaggle/input/basic-datasets/heart.csv')

# DÃ©finition des variables
X = df.drop(['output'], axis=1)
y = df['output'].values

# Normalisation des donnÃ©es
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y)

# ParamÃ¨tres d'entraÃ®nement
epochs = 100

# CrÃ©ation du modÃ¨le de rÃ©seau de neurones
model = Sequential()
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])  # sgd ou adam

# EntraÃ®nement du modÃ¨le
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0)
y_hat = model.predict(X_test)
y_hat = np.round(y_hat.flatten())

# MÃ©triques d'Ã©valuation
print('\nAccuracy :', accuracy_score(y_test, y_hat))
print('\nMatrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print('\nRapport de classification :')
print(classification_report(y_test, y_hat))

# Courbe de loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], c='red')
plt.show()

# Courbe d'accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], c='red')
plt.show()



# Arrondir les prÃ©dictions pour obtenir des classes binaires
y_hat = np.round(y_hat.flatten())
print(y_hat)


# Historique de l'accuracy pendant l'entraÃ®nement
history.history['accuracy']


# Chargement du dataset
df = pd.read_csv('/kaggle/input/basic-datasets/iris.csv')


# AperÃ§u des premiÃ¨res lignes du dataset
df.head()


# Visualisation des relations entre variables colorÃ©e par 'species'
sns.pairplot(df, hue='species')


# Encodage one-hot de la colonne 'species' 
df = pd.get_dummies(df, 'species')
df.head()


# Chargement du dataset
df = pd.read_csv('/kaggle/input/basic-datasets/iris.csv')

# PrÃ©paration des donnÃ©es
df = pd.get_dummies(df, columns=['species'])

# DÃ©finition des variables
X = df.drop(['species_setosa', 'species_versicolor', 'species_virginica'], axis=1)
y = df[['species_setosa', 'species_versicolor', 'species_virginica']].values

# Normalisation des donnÃ©es
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y)

# ParamÃ¨tres d'entraÃ®nement
epochs = 500

# CrÃ©ation du modÃ¨le de rÃ©seau de neurones
model = Sequential()
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])  # sgd ou adam

# EntraÃ®nement du modÃ¨le
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0)
y_hat = model.predict(X_test)
y_hat = y_hat.argmax(axis=1)
y_test = y_test.argmax(axis=1)

# Courbes d'entraÃ®nement
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], c='red')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], c='red')
plt.show()

# MÃ©triques d'Ã©valuation
print('\nAccuracy :', accuracy_score(y_test, y_hat))
print('\nMatrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print('\nRapport de classification :')
print(classification_report(y_test, y_hat))



# Chargement du dataset
df = pd.read_csv('/kaggle/input/basic-datasets/iris.csv')

# Encodage de la variable cible
df['species'] = df['species'].map({'setosa': 0, 'virginica': 1, 'versicolor': 2})

# DÃ©finition des variables
X = df.drop(['species'], axis=1)
y = df['species'].values

# Normalisation des donnÃ©es
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ParamÃ¨tres d'entraÃ®nement
epochs = 500

# CrÃ©ation du modÃ¨le de rÃ©seau de neurones
model = Sequential()
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])  # sgd ou adam

# EntraÃ®nement du modÃ¨le
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0)
y_hat = model.predict(X_test)
y_hat = y_hat.argmax(axis=1)

# Courbes d'entraÃ®nement
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], c='red')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], c='red')
plt.show()

# MÃ©triques d'Ã©valuation
print('\nAccuracy :', accuracy_score(y_test, y_hat))
print('\nMatrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print('\nRapport de classification :')
print(classification_report(y_test, y_hat))


# Chargement du dataset
df = pd.read_csv('/kaggle/input/basic-datasets/iris.csv')

# Encodage de la variable cible
df['species'] = df['species'].map({'setosa':0, 'virginica':1, 'versicolor':2})

# DÃ©finition des variables
X = df.drop(['species'], axis=1)
y = df['species'].values

# Normalisation des donnÃ©es
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ParamÃ¨tres d'entraÃ®nement
epochs = 500

# CrÃ©ation du modÃ¨le de rÃ©seau de neurones
model = Sequential()
# Couches cachÃ©es avec ReLU
model.add(Dense(24, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
# Couche de sortie avec softmax pour classification multi-classes
model.add(Dense(3, activation='softmax'))

# Compilation du modÃ¨le
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])  # sgd ou adam

# EntraÃ®nement du modÃ¨le
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0)
y_hat = model.predict(X_test)
y_hat = y_hat.argmax(axis=1)

# Courbes d'entraÃ®nement
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], c='red')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], c='red')
plt.show()

# MÃ©triques d'Ã©valuation
print('\nAccuracy :', accuracy_score(y_test, y_hat))
print('\nMatrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print('\nRapport de classification :')
print(classification_report(y_test, y_hat))


# Chargement du dataset
df = pd.read_csv('/kaggle/input/images-in-csv-datasets/sign_mnist_small.csv')


# DÃ©finition des variables
X = df.drop(['label'], axis=1).values
y = df['label'].values

# Valeurs uniques dans la variable cible
np.unique(y)


# Code ASCII du caractÃ¨re 'A'
print(ord('A'))

# CaractÃ¨re correspondant au code ASCII 65+2
print(chr(65 + 2))


# Reshape des donnÃ©es pour afficher les images
images = X.reshape(len(X), 28, 28)

for i in range(10):
    plt.imshow(images[i], cmap='Greys')
    plt.title(chr(65 + y[i])) 
    plt.show()


# Normalisation des donnÃ©es
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ParamÃ¨tres d'entraÃ®nement
epochs = 10

# CrÃ©ation du modÃ¨le de rÃ©seau de neurones
model = Sequential()
# Couches cachÃ©es avec ReLU
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
# Couche de sortie avec softmax
model.add(Dense(25, activation='softmax'))

# Compilation du modÃ¨le
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# EntraÃ®nement du modÃ¨le
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)
y_hat = model.predict(X_test)
y_hat = y_hat.argmax(axis=1)

# Courbes d'entraÃ®nement
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], c='red')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], c='red')
plt.show()

# MÃ©triques d'Ã©valuation
print('\nAccuracy :', accuracy_score(y_test, y_hat))
print('\nMatrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print('\nRapport de classification :')
print(classification_report(y_test, y_hat))


# Chargement du dataset
df = pd.read_csv('/kaggle/input/images-in-csv-datasets/cifar10_small.csv')


# Afficher les premiÃ¨res lignes du DataFrame
df.head()


# Taille du dataset
df.shape


# DÃ©finition des variables
X = df.drop(['label'], axis=1).values
y = df['label'].values

# Valeurs uniques dans la variable cible
print(np.unique(y))

# Reshape pour affichage des images
images = X.reshape(len(X), 32, 32, 3)

# Affichage des 10 premiÃ¨res images avec leur label
for i in range(10):
    plt.imshow(images[i])
    plt.title(y[i])
    plt.show()


from tensorflow.keras.models import Sequential, load_model # CrÃ©ation et chargement de modÃ¨les Keras
from tensorflow.keras.layers import *                      # Toutes les couches Keras (Dense, Conv2D, MaxPooling2D, Flatten, etc.)


# Chargement des donnÃ©es
X = df.drop(['label'], axis=1).values
y = df['label'].values

# Reshape pour CNN et normalisation
X = X.reshape(len(X), 32, 32, 3)
X = X / 255

# SÃ©paration en ensembles d'entraÃ®nement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ParamÃ¨tres d'entraÃ®nement
epochs = 10
batch_size = 256

# CrÃ©ation du modÃ¨le CNN
model = Sequential()
model.add(InputLayer(input_shape=(32, 32, 3)))          # Couche d'entrÃ©e
model.add(Conv2D(16, (5,5), padding='same', activation='relu'))  # Convolution 1
model.add(Conv2D(16, (5,5), padding='same', activation='relu'))  # Convolution 2
model.add(MaxPooling2D(pool_size=(2,2)))                         # Pooling 1

model.add(Conv2D(20, (5,5), padding='same', activation='relu'))  # Convolution 3
model.add(Conv2D(20, (5,5), padding='same', activation='relu'))  # Convolution 4
model.add(MaxPooling2D(pool_size=(2,2)))                         # Pooling 2 

model.add(Conv2D(20, (5,5), padding='same', activation='relu'))  # Convolution 5
model.add(Conv2D(20, (5,5), padding='same', activation='relu'))  # Convolution 6
model.add(MaxPooling2D(pool_size=(2,2)))                         # Pooling 3

model.add(Flatten())                       
model.add(Dense(10, activation='softmax'))

# Compilation du modÃ¨le
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# EntraÃ®nement
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1, batch_size=batch_size)

# PrÃ©dictions
y_hat = model.predict(X_test)
y_hat = y_hat.argmax(axis=1)  # Convertir les probabilitÃ©s en labels

# Courbes d'entraÃ®nement
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], c='red')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], c='red')
plt.show()

# Ã‰valuation
print('Accuracy :', accuracy_score(y_test, y_hat))
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))


# Sauvegarde du modÃ¨le entraÃ®nÃ©
model.save('cifar.h5')


# RÃ©sumÃ© de l'architecture du modÃ¨le
model.summary()


# load_model('cifar.h5').predict(X_test)
load_model('cifar.h5').predict(X_test)


# Pour charger un dataset dâ€™images depuis un dossier
from tensorflow.keras.preprocessing import image_dataset_from_directory


# Charger les images dâ€™entraÃ®nement depuis le dossier
train = image_dataset_from_directory('/kaggle/input/cat-and-dog/training_set', image_size=(200,200))


# Charger les images de test depuis le dossier
test = image_dataset_from_directory('/kaggle/input/cat-and-dog/test_set/test_set', image_size=(200,200))


# Charger les images dâ€™entraÃ®nement
train = image_dataset_from_directory("/kaggle/input/cat-and-dog/training_set/training_set", image_size=(200,200))

# Charger les images de test
test = image_dataset_from_directory("/kaggle/input/cat-and-dog/test_set/test_set", image_size=(200,200))

# Nombre dâ€™Ã©poques
epochs = 10

# ModÃ¨le CNN
model = Sequential()
model.add(InputLayer(input_shape=(200,200,3)))  
model.add(BatchNormalization())             

model.add(Conv2D(16, (5,5),padding='same',activation='relu'))
model.add(Conv2D(16, (5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))      
model.add(BatchNormalization())

model.add(Conv2D(20, (5,5),padding='same',activation='relu'))
model.add(Conv2D(20, (5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(20, (5,5),padding='same',activation='relu'))
model.add(Conv2D(20, (5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Flatten())                            

# Sortie
model.add(Dense(2,activation='softmax'))

# Compiler le modÃ¨le
model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

# EntraÃ®ner le modÃ¨le
history = model.fit(train, epochs=epochs, validation_data=(test), verbose=1,batch_size=256)

# Courbe de perte
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], c='red')
plt.show()

# Courbe de prÃ©cision
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], c='red')
plt.show()


# Importer le modÃ¨le VGG16 prÃ©-entraÃ®nÃ©
from tensorflow.keras.applications import *


# Charger le modÃ¨le VGG16 prÃ©-entraÃ®nÃ© sans la couche finale
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(200,200,3))


# Afficher le rÃ©sumÃ© du modÃ¨le VGG16
vgg.summary()


# Installer la librairie ultralytics
!pip install ultralytics


# Importer le modÃ¨le YOLO et OpenCV
from ultralytics import YOLO
import cv2

# Charger le modÃ¨le YOLO prÃ©-entraÃ®nÃ©
model = YOLO('yolov8n.pt')

# EntraÃ®ner le modÃ¨le sur le dataset de voitures
model.train(data='/kaggle/input/cardetection/car/data.yaml', epochs=5, device=0)

# PrÃ©dire sur une image de test
result = model.predict('/kaggle/input/d/eirinigeorgiadou/cardetection/IMG_7500.jpeg')

# Convertir l'image pour affichage avec matplotlib
image = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)

# Afficher l'image avec les dÃ©tections
plt.imshow(image)
plt.show()

