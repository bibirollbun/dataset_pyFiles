import pickle
import os
# ^^^ pyforest auto-imports - don't write above this line
import pandas as pd
import numpy as np
from termcolor import colored

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder


#modeles :
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from xgboost import XGBClassifier


#import metrics
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import cross_val_score


import time
import pickle


from termcolor import colored
import os
import warnings
warnings.filterwarnings('ignore')


#chargement fichier : 
path  = "/kaggle/input/preprocesser-fertilizer/"
with open(f"{path}/StandardScaler.pkl", 'rb') as file:
    std = pickle.load(file)
with open(f"{path}/feature_encoder.pkl", 'rb') as file:
    feature_encoder = pickle.load(file)
with open(f"{path}/targetencoder.pkl", 'rb') as file:
    targetencoder = pickle.load(file)
    
# print(f"Données chargées depuis {colored('scaler.pkl','blue')} : {colored(scaler,'green',attrs=['bold'])}")


train_preprocessed = pd.read_csv("/kaggle/input/df-preprocessed-fertilizer/train_preprocessed.csv",index_col=0)
df = train_preprocessed.copy()
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")



df.head()


test.head()


df = df.drop("id", axis = 1)
id_test = test["id"]
test = test.drop("id",axis = 1)


var_qual_feat = [i for i in feature_encoder.keys()]
var_cont = [i for i in df.select_dtypes(include=float)]
target = [i for i in df if i not in (var_cont + var_qual_feat )]


feature_encoder


#Parcours du dictionnaire contenant les LabelEncoder de chaque feature :
for i in var_qual_feat:
    test[i] = feature_encoder[i].transform(test[i]) #Encodage par feature


test.head()
#Données correctement encodés





#### Sample df : 
# - Pour tester modelisation rapidement :

sample = df.sample(1000)
y = sample['Fertilizer Name'] 
X = sample.drop(['Fertilizer Name'],axis=1)
X_train, X_test, y_train,y_test = train_test_split(X, y,test_size = 0.2, random_state =42,stratify=y)


y = df['Fertilizer Name'] 
X = df.drop(['Fertilizer Name'],axis=1)
X_train, X_test, y_train,y_test = train_test_split(X, y,test_size = 0.2, random_state =42,stratify=y)



#instanciation modèle :
model = XGBClassifier(
    objective='multi:softprob', #softprob pour classification multiclass
    num_class=len(np.unique(y_train)),#indique explicitement nombre de classe
#     n_estimators=3200,
#     learning_rate=0.045,         
#     max_depth=7,                
#     colsample_bytree=0.6,       
#     colsample_bylevel=0.8,      
#     subsample=0.8,
)

#Entrainement : 
model.fit(X_train, y_train)


#Récupération des probabilités de prédictions : 
y_pred_probs = model.predict_proba(X_test)
#récupération prédiction :
y_pred = model.predict(X_test)


print("shape : ", y_pred_probs.shape) #(observation, classes distinctes de la target)
print()
first_obs = y_pred_probs[0]
print("Probabilité de prediction : \n", first_obs)
print()
print("Classes :\n", targetencoder.classes_)


print(f"Pour la première prédiction, nous avons obtenu les % suivants :")
pourcent_firstobs = [f"{np.around(i*100,2)} %"  for i in first_obs]
print(pourcent_firstobs)
print()
print(f"La prédiction la plus élevée est la classe : {np.argmax(first_obs)}")
classe_firstobs = np.argmax(first_obs)
print(f"Cette classe correspond à la valeur original : {targetencoder.classes_[classe_firstobs]}")
print()
print(f"Si nous regardons non pas le predict_proba mais le predict :")
print(f"Nous obtenons bien la prédiction : {targetencoder.inverse_transform(y_pred[[0]])}")


print("Prédiction pour la 1ere observation :")
print(pourcent_firstobs)
print()
print("Ordre dans lequel les indices devraient être pour les avoir")
print("dans l'ordre croissant :")
print()
print(np.argsort(first_obs))


np.argsort(y_pred_probs, axis=1) # permet d'avoir le classement des prédictions
#axis = 1 pour que le classement dans les colonnes de chaque liste
#et non pas entre les liste


#Récupération du top 3 prédiction dans l'ordre décroissant  : 
top_3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]  
#Affichons 5 premières prédictions :
top_3_preds[:5] 


#transformation de y_test en liste : 
actual = [[label] for label in y_test]


actual[:5]


top_3_preds[:5]


print(f"Pour y_test (actual), la vraie target de la première prédiction est : {actual[0]}")
print()
print("On regarde si cette valeur vraie se trouve dans l'une des 3 prédictions de la 1ere observation")
print("Si elle l'est, on regarde sa position, plus elle est élevée et mieux c'est ")
print()
print(f"La valeur {actual[0]} est dans l'une des 3 prédictions de la première observation ?")
print(f"{top_3_preds[0]}")
print("Si non => mauvaise prédiction")
print("Si oui, en position 1, meilleure prédiction (=score 1) si position 3, plus faible score (1/3 = .33)")
#et si en position 2 alors 1/2 = .5 de score


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


map3_score = mapk(actual, top_3_preds)
print(f"✅ MAP@3 Score: {map3_score:.5f}")


test_probs = model.predict_proba(test)
top_3_preds = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
top_3_labels = targetencoder.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
submission = pd.DataFrame({
    'id': id_test,
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved as 'submission.csv'")



submission.head()

