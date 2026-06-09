from sklearn.preprocessing import StandardScaler
# ^^^ pyforest auto-imports - don't write above this line
import numpy as np
import pandas as pd 
import os
from termcolor import colored


import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


import time

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import HuberRegressor

import shap

import lime
import lime.lime_tabular


import pickle

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
df = train.copy()


print(df.shape)
df.head()


X,y = df.drop(["id","BeatsPerMinute"],axis=1), df["BeatsPerMinute"]
X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=42)


var_cont = X.columns.tolist()


std = StandardScaler()
#Fit uniquement sur le train
std.fit(X_train[var_cont])

# Transform train et test avec les mêmes paramètres
X_train[var_cont] = std.transform(X_train[var_cont])
X_test[var_cont]  = std.transform(X_test[var_cont])
X[var_cont] = std.transform(X[var_cont])  


y


print("X train :", X_train.shape)
print("X test :", X_test.shape)
print("y train :", y_train.shape)
print("y test :", y_test.shape)
print()
print("X :", X.shape)
print("y :", y.shape)





def evaluer_modele(models):
    """L'utilisateur entre un ou une liste de modèle qui sera entrainée, on affichera un graphique du cross_val_score
    sur 5 splits et un tuple de 3 valeurs contenant les prédictions, le datafram"""    
    #On verifie si l'utilisateur à rentrer une liste de modèle ou un modèle unique
    if type(models) != list: #on transforme en liste si un seul modèle
        models = [models]     
    #Création des dictionnaires qui contiendrons les prédictions et les différentes metriques
    metrics = {'MSE': [], 'RMSE': [], 'MAE': [], 'R2': [], 'Exe_time':[]}
    predictions = {}    
    #dictionnaire pour stocker les modèles entraînés
    trained_models = {}      
    
    for model in models:
        model_name = model.__class__.__name__
        scores = cross_val_score(estimator=model, X=X_train.values, y=y_train,  scoring="neg_mean_squared_error")#RMSE
        #rappel du score : negative MSE, plus la valeur retournée est élevée (+ proche de 0) et mieux c'est
        #par ex -1500 < -5, -5 est plus élevé, donc meilleur.
        
        #Debut entrainement
        start_time = time.time()
        # Entraîner le modèle sur les données d'entraînement
        model.fit(X_train, y_train)
        #Fin entrainement :
        end_time = time.time()
        # durée totale de l'entrainement 
        training_time = round(end_time - start_time,2)
        # Durée en minute :
        training_time_min = round(training_time/60,2)        
        
        #prédictions sur X_test
        y_pred = model.predict(X_test)
        # Calcul des métriques sur X_test
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Ajoute au dictionnaire
        metrics['MSE'].append(mse),
        metrics['RMSE'].append(rmse)
        metrics['MAE'].append(mae)
        metrics['R2'].append(r2)
        metrics["Exe_time"].append(training_time_min)        
        #Ajout des prédictions dans le dictionnaire 
        predictions[model_name] = y_pred
        #Stockage du modèle entrainé
        trained_models[model_name] = model  

        # lineplot pour le cross val score :
        plt.figure(figsize=(8,3))
        sns.lineplot(x = [i+1 for i in range(len(scores))], y = scores)
        plt.xlabel('n')
        plt.ylabel('neg_MSE')
        plt.title(f"{model_name}")
        plt.show()
        #CV moyen sur 5 splits :
        print("Score moyen sur 5 splits : ", round(scores.mean(),2))
        print("-"*30)       
    # Convertir metrics en DataFrame avec les noms des modèles comme index
    metrics_df = pd.DataFrame(metrics, index=[model.__class__.__name__ for model in models])
    
    return trained_models, predictions, metrics_df


#Entrainement 
models = [
    DummyRegressor(strategy="mean"),
    LinearRegression(),
    Ridge(),
    Lasso(),
    KNeighborsRegressor(),
#     RandomForestRegressor(),
#     SVR(kernel="rbf"),
#     AdaBoostRegressor(),
    XGBRegressor(),
]


model_trained, prediction, metric_df = evaluer_modele(models)


metric_df.drop("R2",axis=1, inplace=True)


plt.figure(figsize=(12,10))
for i,col in enumerate(metric_df,1):
    plt.subplot(3,2,i)
    plt.title(f"{col}")
    ax = sns.barplot(x = metric_df.index, y = col , data = metric_df)
    lab = ax.get_xticklabels()
    ax.set_xticklabels(labels =lab,rotation=45)
    ax.set_ylabel(None)
    plt.tight_layout()
plt.show()


def visualisation_pred_real(list_model,xsize=14,ysize=14):
    """
    Fonction qui récupère les prédictions d'un modèle et affiche un graphique en comparant avec les valeurs réelles
    les valeurs réelles sont sur l'axe des x et les valeurs prédites sont sur l'axe des y.
    La ligne rouge en pointillés représente une correspondance parfaite entre les valeurs réelles et prédites. 
    Dans un modèle parfait, tous les points se situeraient le long de cette ligne.
    return : scatterplot
    """
    
    plt.figure(figsize=(xsize,ysize))
    plt.suptitle('Scatter Plot des Valeurs Réelles vs. Prédites')

    for i, model in enumerate(list_model,1):
        plt.subplot(4,2,i)    
        y_pred = list_model[model].predict(X_test)
        plt.scatter(y_test, y_pred, color='blue', alpha=0.5) #scatterplot avec les vraies valeurs en x et les valeurs prédites en y
        plt.title(f"{model}")
        # Ajouter une ligne diagonale pour indiquer une correspondance parfaite entre les valeurs réelles et prédites
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
                 color='red', linestyle='--', label="Ideal (y = ŷ)")

        # Label et  titre
        plt.xlabel('Valeurs Réelles')
        plt.ylabel('Valeurs Prédites')
        plt.legend()
    plt.tight_layout()
    plt.show()


visualisation_pred_real(model_trained,xsize=14,ysize=14)


model1 = model_trained['LinearRegression']


        # Création de l'explainer SHAP pour le modèle 
explainer = shap.Explainer(model1, X_train)
        #Calcul des valeurs SHAP sur les données de test
shap_values = explainer(X_test)
        #Titre graphique
plt.suptitle(model1.__class__.__name__)

        # Graphique global : importance moyenne des features + direction
shap.summary_plot(shap_values, X_test)
        
        # Graphique local : explication détaillée pour la première observation
shap.plots.waterfall(shap_values[0])


model2 = model_trained['XGBRegressor']
# SHAP 
explainer = shap.TreeExplainer(model2)
shap_values = explainer(X_test)

#Titre graph:
plt.suptitle(model2.__class__.__name__)

# Summary plot :
shap.summary_plot(shap_values, X_test)

# prédiction individuelle:
shap.plots.waterfall(shap_values[0])


def new_prediction(model, data):
    #Récupération des mêmes features que celles utilisés pour l'entrainement des modèles
    columns = model.feature_names_in_
    #Récupération de l'id du jeu de données
    id_data = data.index
    #Préparation du dataframe à tester :
    X = data[columns]
    #Récupération des prédictions
    numeric_prediction = model.predict(X)
    
    #Conversion des prédiction en données d'origine (textuelles et non numérique)
    # class_predicted = encoder_label.inverse_transform(numeric_prediction)   
    
    #Transformation des prédictions en dataframe avec l'id en index
    prediction_df = pd.DataFrame(numeric_prediction, columns = [y.name], index = id_data)
#     prediction_df = pd.DataFrame(class_predicted, columns = ["class"], index = id_data)
    return prediction_df



test[var_cont]  = std.transform(test[var_cont])


best_model = XGBRegressor().fit(X,y)


submission_pred = new_prediction(best_model,test)
#Récupération dans un fichier .csv pour soumission à kaggle
submission_pred.to_csv("XGBRegressor_1.csv")

