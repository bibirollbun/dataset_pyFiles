from sklearn.preprocessing import LabelEncoder
# ^^^ pyforest auto-imports - don't write above this line
import numpy as np
import pandas as pd 
import os
from termcolor import colored


import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import time

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,OrdinalEncoder


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

import pickle
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
df = train.copy()


target = df.accident_risk


# Séparation des données en fonction du dtypes
var_bool = df.select_dtypes("bool").columns.tolist()
var_qual = df.select_dtypes("object").columns.tolist()
var_cont = [i for i in df.select_dtypes("float") if i != target.name]
var_dis  = [i for i in df.select_dtypes("int") if i != "id"]


#Echantillon pour tester la modélisation :
# df = df.sample(1000, random_state=42)


X,y = df.drop(["id",target.name],axis=1), df[target.name]
X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=42)


X_train.head(1)


print("X train :", X_train.shape)
print("X test :", X_test.shape)
print("y train :", y_train.shape)
print("y test :", y_test.shape)
print()
print("X :", X.shape)
print("y :", y.shape)


std = StandardScaler()
#Fit uniquement sur le train
std.fit(X_train[var_cont])

# Transform pour le train et test :
X_train[var_cont] = std.transform(X_train[var_cont])
X_test[var_cont]  = std.transform(X_test[var_cont])
X[var_cont] = std.transform(X[var_cont])  


encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
encoder.fit(X_train[var_qual])

# Transformations
X_train_encoded = pd.DataFrame(
    encoder.transform(X_train[var_qual]),
    columns=encoder.get_feature_names_out(var_qual),
    index=X_train.index
)
X_test_encoded = pd.DataFrame(
    encoder.transform(X_test[var_qual]),
    columns=encoder.get_feature_names_out(var_qual),
    index=X_test.index
)

# On retire les anciennes colonnes et on concatène les nouvelles
X_train = pd.concat([X_train.drop(columns=var_qual), X_train_encoded], axis=1)
X_test  = pd.concat([X_test.drop(columns=var_qual), X_test_encoded], axis=1)


X_encoded = pd.DataFrame(
    encoder.transform(X[var_qual]),
    columns=encoder.get_feature_names_out(var_qual),
    index=X.index)
X = pd.concat([X.drop(columns=var_qual), X_encoded], axis=1)


X_train.head(3)


ordinalencoder = OrdinalEncoder()

# Fit sur le train
ordinalencoder.fit(X_train[var_bool])

# Transform sur train et test
X_train[var_bool] = ordinalencoder.transform(X_train[var_bool])
X_test[var_bool]  = ordinalencoder.transform(X_test[var_bool])
X[var_bool] = ordinalencoder.transform(X[var_bool])  


X_train.head(3)


X_test.head(3)


def eval_cross_val(models, X_train, y_train, cv=5, scoring='neg_mean_squared_error'):
    """Retourne les scores de cross-validation pour une liste de modèles."""
    if type(models) != list:
        models = [models]
        
    cv_scores = {}
    for model in models:
        model_name = model.__class__.__name__
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        cv_scores[model_name] = scores
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


eval_cross_val(models, X_train, y_train, cv=5, scoring='neg_mean_squared_error')


def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    """Entraîne les modèles et retourne prédictions, métriques et modèles entraînés"""
    if type(models) != list:
        models = [models]
    
    metrics = {'MSE': [], 'RMSE': [], 'MAE': [], 'R2': [], 'Exe_time': []}
    predictions = {}
    trained_models = {}
    
    for model in models:
        model_name = model.__class__.__name__
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = round(time.time() - start_time, 2)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics['MSE'].append(mse)
        metrics['RMSE'].append(rmse)
        metrics['MAE'].append(mae)
        metrics['R2'].append(r2)
        metrics['Exe_time'].append(round(training_time/60,2))
        
        predictions[model_name] = y_pred
        trained_models[model_name] = model
    
    metrics_df = pd.DataFrame(metrics, index=[model.__class__.__name__ for model in models])
    return trained_models, predictions, metrics_df



model_trained, prediction, metric_df = train_and_evaluate(models, X_train, y_train, X_test, y_test)


metric_df.drop("R2",axis=1, inplace=True)
plt.figure(figsize=(12,10))
for i,col in enumerate(metric_df,1):
    plt.subplot(3,2,i)
    plt.grid()
    plt.title(f"{col}")
    ax = sns.barplot(x = metric_df.index, y = col , data = metric_df, palette="tab10")
    lab = ax.get_xticklabels()
    ax.set_xticklabels(labels =lab,rotation=45)
    ax.set_ylabel(None)
    plt.tight_layout()
plt.show()


def visualisation_pred_real(list_model, X_train=X_train, X_test=X_test, xsize=14,ysize=14):
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


visualisation_pred_real(model_trained, X_train=X_train, X_test=X_test, xsize=14,ysize=14)


### Suppression des variables booléennes :
X_train_red = X_train.drop(columns=var_bool)
X_test_red = X_test.drop(columns=var_bool)


eval_cross_val(models, X_train_red, y_train, cv=5, scoring='neg_mean_squared_error')


model_trained_red, prediction_red, metric_df_red =  train_and_evaluate(models, X_train_red, y_train, X_test_red, y_test)


metric_df_red.drop("R2",axis=1, inplace=True)
plt.figure(figsize=(12,10))
for i,col in enumerate(metric_df_red,1):
    plt.subplot(3,2,i)
    plt.grid()
    plt.title(f"{col}")
    ax = sns.barplot(x = metric_df_red.index, y = col , data = metric_df_red, palette="tab10")
    lab = ax.get_xticklabels()
    ax.set_xticklabels(labels =lab,rotation=45)
    ax.set_ylabel(None)
    plt.tight_layout()
plt.show()



visualisation_pred_real(model_trained_red, X_train_red, X_test_red, xsize=14,ysize=14)


final_model = model_trained["XGBRegressor"]
final_model_red = model_trained_red["XGBRegressor"]


        # Création de l'explainer SHAP pour le modèle 
explainer = shap.Explainer(final_model, X_train)
        #Calcul des valeurs SHAP sur les données de test
shap_values = explainer(X_test)
        #Titre graphique
plt.suptitle(final_model.__class__.__name__)

        # Graphique global : importance moyenne des features + direction
shap.summary_plot(shap_values, X_test)
        
        # Graphique local : explication détaillée pour la première observation
shap.plots.waterfall(shap_values[0])


        # Création de l'explainer SHAP pour le modèle 
explainer = shap.Explainer(final_model_red, X_train_red)
        #Calcul des valeurs SHAP sur les données de test
shap_values = explainer(X_test_red)
        #Titre graphique
plt.suptitle(final_model_red.__class__.__name__)

        # Graphique global : importance moyenne des features + direction
shap.summary_plot(shap_values, X_test_red)
        
        # Graphique local : explication détaillée pour la première observation
shap.plots.waterfall(shap_values[0])





def new_prediction(model, data):
    #Récupération des mêmes features que celles utilisés pour l'entrainement des modèles
    columns = model.feature_names_in_
    #Récupération de l'id du jeu de données
    id_data = data.id
    #Préparation du dataframe à tester :
    X = data[columns]
    #Récupération des prédictions
    numeric_prediction = model.predict(X)
     
    #Transformation des prédictions en dataframe avec l'id en index
    prediction_df = pd.DataFrame(numeric_prediction, columns=[y.name], index=id_data).rename_axis("id")
    return prediction_df


#Standardisation :
test[var_cont] = std.transform(test[var_cont])
# OneHotEncoder sur variables qualitatives
test_encoded = pd.DataFrame(
    encoder.transform(test[var_qual]),
    columns=encoder.get_feature_names_out(var_qual),
    index=test.index
)
test = pd.concat([test.drop(columns=var_qual), test_encoded], axis=1)

#OrdinalEncoder sur variables booléennes :
test[var_bool] = ordinalencoder.transform(test[var_bool])  



best_model = XGBRegressor().fit(X,y)
submission_pred = new_prediction(best_model,test)
#Récupération dans un fichier .csv pour soumission à kaggle
submission_pred.to_csv("XGBRegressor_1.csv")


X_red = X.drop(columns=var_bool)


best_model_red = XGBRegressor().fit(X_red,y)
submission_pred = new_prediction(best_model_red,test)
#Récupération dans un fichier .csv pour soumission à kaggle
submission_pred.to_csv("XGBRegressor_1_red.csv")


submission_pred = new_prediction(best_model_red,test)
#Récupération dans un fichier .csv pour soumission à kaggle
submission_pred.to_csv("XGBRegressor_1_red.csv")




