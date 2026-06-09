### This notebook is meant for academic purposes only


"""
On importe les diverses librairies :
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Librairie Scikit Learn : modules pour le pré-traitement et métriques : 
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold

# Librairie Scikit Learn : Regresseurs : 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

# Librairies lightgbm et catboost, pour les algorithmes de la famille gradient boosting : 
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# On charge les données X et y dans des DataFrames pandas.
# Les DataFrame sont des structures sous forme de tableau, avec des lignes et colonnes.
# X a deux dimensions; y a une seule dimension (c'est un vecteur).

X = pd.read_csv("/kaggle/input/house-prices-preprocessed/X.csv")
y = pd.read_csv("/kaggle/input/house-prices-preprocessed/Y.csv")

DO=3


"""
On affiche de 5 premières lignes du tableau X.
Les lignes ont des index uniques qui commencent à 0.
Les colonnes ont des noms sous forme de chaîne de caractères.
"""

X.head(3)


"""
On affiche le y. Il correspond dans notre cas au prix
"""

y


X_temporary, X_test, y_temporary, y_test = train_test_split(X, y, test_size=100, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temporary, y_temporary, test_size=100, random_state=42)


# On initialise le modèle choisi: 
model = CatBoostRegressor(iterations=100) 
#model = RandomForestClassifier(n_estimators=100) 

if DO == 1:
    model.fit(X_train, y_train)
    prediction_test = model.predict(X_test)
    score_test = mean_squared_error(y_test, prediction_test)

    print("Score test", score_test)
elif DO == 2:
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    prediction_test = model.predict(X_test)
    score_test = mean_squared_error(y_test, prediction_test)

    prediction_val = model.predict(X_val)
    score_val = mean_squared_error(y_val, prediction_val)

    print("Score val", score_val, "Score test", score_test)
elif DO == 3:
    # Entraînement du modèle, avec la méthode "fit"
    # On donne en entrée l'ensemble d'entraînement (X_train, y_train)
    # On donne aussi l'ensemble de validation (X_val, y_val) 
    # A chaque itération, on a un nouveau modèle, et on fait une prédiction sur X_train et X_val, 
    # et on calcule le score entre la prédiction et y_train / y_val respectivement
    # Attention : on entraîne que sur l'enselble d'entraînement (X_train, y_train), pas sur X_val, y_val !!!
    # use_best_model=True : on enregistre, et on garde pour la prédiction par la suite,
    # le modèle de l'itération où on a le meilleur score sur l'ensemble de validation
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], use_best_model=True)

    # On fait une prédiction sur X_test; X_test est utilisé APRES l'entraînement
    prediction_test = model.predict(X_test)    

    # On calcule le score entre la prediction et la valeur réelle y_test
    score_test = mean_squared_error(y_test, prediction_test, squared=False)

    # On fait une prédiction sur X_val ... 
    prediction_val = model.predict(X_val)

    # ... et on calcule le score entre cette prédiction et la valeur réelle y_val
    score_val = mean_squared_error(y_val, prediction_val, squared=False)

    # On imprime les deux scores : 
    print("Score val", score_val, "Score test", score_test)



prediction_val

