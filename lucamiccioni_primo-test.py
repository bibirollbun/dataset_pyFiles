from supplemental_english import *
import pandas as pd

TRAIN_DATA = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
GOV_CODES = GOVERNMENT_CODES


import matplotlib.pyplot as plt
import seaborn as sns

# Carica il dataset
data = TRAIN_DATA
print("Prime righe del dataset:")
print(TRAIN_DATA.head())
print("\n")

# Info di base
print("Info sui dati:")
print(data.shape)
print(data.dtypes)
print("\n")

# Statistiche descrittive
print("Statistiche descrittive:")
print(data.describe(include='all'))
print("\n")

# Missing values
print("Valori mancanti per colonna:")
print(data.isnull().sum())
print("\n")

# Distribuzione della colonna target
plt.figure(figsize=(8, 4))
sns.histplot(data["price"], kde=True, bins=40)
plt.title("Distribuzione del prezzo")
plt.xlabel("Prezzo")
plt.ylabel("Frequenza")
plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder

def prepare_data(knowledge):
    
    X = TRAIN_DATA
    encoder = LabelEncoder()
    
    # Suddivisione targa in parti
    X["letters"] = X["plate"].str[0] + X["plate"].str[4] + X["plate"].str[5] # Caratteri alfabetici
    X["nums"] = X["plate"].str[1:4] # Caratteri numerici
    X["region"] = X["plate"].str[6:] # Caratteri relativi alla regione

    # Suddivisione data in giorno, mese e anno
    X["date"] = pd.to_datetime(X["date"])
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day

    if(knowledge): # Aggiunta feature nei dati di training per modelli con conoscenza
        # Feature: lettere della targa uguale
        X["has_equal_letters"] = X["letters"].apply(lambda s: all(c == str(s)[0] for c in str(s)))

        #Feature: numeri della targa uguali
        X["has_equal_nums"] = X["nums"].apply(lambda s: all(c == str(s)[0] for c in str(s)))

        # Feature: numeri in sequenza
        X["has_nums_in_sequence"] = find_sequence_nums(X)
        
        # Feature: prima parte della targa palindroma (5 caratteri iniziali)
        X["is_palindrome"] = X["plate"].str[0:5].apply(lambda s: True if s == s[::-1] else False)

        # Feature: targa proibita alla vendita, priorità su strada e importanza
        # X["fobidden_to_buy", "advantage_on_road", "significance"] = find_special_plate(X)

    # Codifica dati in formato numerico
    X["letters"] = encoder.fit_transform(X["letters"])
    X["nums"] = encoder.fit_transform(X["nums"])
    X["region"] = encoder.fit_transform(X["region"])

    # Rimozione colonne non utili al training
    X = X.drop(columns=["id", "price", "date", "plate"])
    y = TRAIN_DATA["price"]

    analyze_data(X, knowledge)

    return (X, y)


# Restituisce i dettagli di una targa speciale, se la trova cercando nel dict delle targhe governative
def find_special_plate(plate):
    for (letters, num_range, region), details in GOV_CODES.items():
        if(plate["letters"] == letters and plate["region"] == region and num_range[0] <= plate["nums"] <= num_range[1]):
            return details[1], details[2], details[3] # In posizione 0 c'è la descrizione del tipo di targa speciale
    return 0, 0,0

# Controlla se i numeri di targa sono in sequenza crescente (es. 123) o descrescente (es. 987)
def find_sequence_nums(plate):
    seq_up = ["012", "123", "234", "345", "456", "567", "678", "789"]
    seq_down = ["987", "876", "765", "654", "543", "432", "321", "210"]
    return any(seq in plate["nums"] for seq in seq_up) or any(seq in plate["nums"] for seq in seq_down)

# Funzione per il calcolo delle prestazioni di un modello secondo le metriche del problema
def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100


# qui va il codice che analizza i dati, come il codice sotto 'Analisi dei dati'

def analyze_data(data, knowledge):
    # Prime righe dei dati
    print("Prime righe dei dati:")
    print(data.head())
    print("\n")
    
    # Info di base
    print("Info sui dati:")
    print(data.shape)
    print(data.dtypes)
    print("\n")
    
    # Statistiche descrittive
    print("Statistiche descrittive:")
    print(data.describe(include='all'))
    print("\n")
    
    # Missing values
    print("Valori mancanti per colonna:")
    print(data.isnull().sum())
    print("\n")

    # Correlazione dei dati
    plt.figure(figsize=(10,8))
    sns.heatmap(data.corr(), cmap='coolwarm', annot=True, fmt='.2f')
    plt.title("Matrice di Correlazione")
    plt.show()


import sys
import random
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

class Model(Enum):
    MLP = 1,
    RND_FOREST = 2,
    GBOOST = 3,
    XGBOOST = 4,
    CAT_BOOST = 5
    
def train_and_test_model(model, knowledge, seed=None):
    # Inizializzazione variabili
    seed = random.randint(1,100) if seed is None else seed
    smape_error = 0

    # Preparazione dei dati
    X, y = prepare_data(knowledge)
    y = np.array(y).reshape(-1, 1).ravel()

    # Suddivisione dei dati per train e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    if model == Model.MLP:
        smape = MultiLayerPerceptron(X_train, X_test, y_train, y_test, seed)
    elif model == Model.RND_FOREST:
        smape = RandomForest(X_train, X_test, y_train, y_test, seed)
    elif model == Model.GBOOST:
        smape = GradientBoost(X_train, X_test, y_train, y_test, seed)
    elif model == Model.XGBOOST:
        smape = XGBoost(X_train, X_test, y_train, y_test, seed)
    elif model == Model.CAT_BOOST:
        smape = CatBoosting(X_train, X_test, y_train, y_test, seed)
    else:
        print("Errore: nessun modello selezionato", file=sys.stderr)
        return
    
    print("Seed: ", seed)
    print(f'Symmetric Mean Absolute Percentage Error: {smape:.2f}')
    
    return smape


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

def MultiLayerPerceptron(X_train, X_test, y_train, y_test, seed):
    # Inizializzazione variabili
    scaler = StandardScaler()
    
    # Normalizzazione dei dati
    X_train_copy = X_train
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Creazione del modello e training
    model = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', max_iter=1000, random_state=seed, verbose=False)
    model.fit(X_train, y_train)

    # Predizione dati e valutazione del modello
    y_pred = model.predict(X_test)
    smape_score = smape(y_test, y_pred)

    # Plot della rilevanza delle feature
    plt.figure(figsize=(10, 6))
    plt.barh(X_train_copy.columns, permutation_importance(model, X_test, y_test, n_repeats=10, random_state=seed).importances_mean)
    plt.title("Importanza feature - MLP")
    plt.xlabel("Importanza feature")
    plt.show()
    
    return smape_score


from sklearn.ensemble import RandomForestRegressor

def RandomForest(X_train, X_test, y_train, y_test, seed):
    # Creazione del modello
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed)
    
    # Training del modello
    model.fit(X_train, y_train)
    
    # Predizione dati e valutazione del modello
    y_pred = model.predict(X_test)
    smape_score = smape(y_test, y_pred)

    # Plot della rilevanza delle feature
    plt.figure(figsize=(10, 6))
    plt.barh(X_train.columns, model.feature_importances_)
    plt.title("Importanza feature - Random Forest")
    plt.xlabel("Importanza")
    plt.show()
    
    return smape_score


from sklearn.ensemble import GradientBoostingRegressor

def GradientBoost(X_train, X_test, y_train, y_test, seed):
    # Creazione del modello
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=seed)
    
    # Training del modello
    model.fit(X_train, y_train)
    
    # Predizione dati e valutazione del modello
    y_pred = model.predict(X_test)
    smape_score = smape(y_test, y_pred)

    # Plot della rilevanza delle feature
    plt.figure(figsize=(10, 6))
    plt.barh(X_train.columns, model.feature_importances_)
    plt.title("Importanza feature - Gradient Boost")
    plt.xlabel("Importanza")
    plt.show()

    return smape_score


!pip install xgboost

import xgboost as xgb

def XGBoost(X_train, X_test, y_train, y_test, seed):
    # Creazione del modello
    model = xgb.XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, objective='reg:squarederror', random_state=seed)
    
    # Training e predizione dei dati
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Valutazione del modello
    smape_score = smape(y_test, y_pred)

    # Plot della rilevanza delle feature
    importance = model.get_booster().get_score(importance_type='weight')
    feature_names = list(importance.keys())
    feature_importance = list(importance.values())

    # Plot della rilevanza delle feature
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance)
    plt.title("Importanza feature - XGBoost")
    plt.xlabel("Importanza feature")
    plt.tight_layout()
    plt.show()

    return smape_score


!pip install catboost

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

def CatBoosting(X_train, X_test, y_train, y_test, seed):
    # Crazione del modello
    model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, loss_function='MAE', random_seed=seed, verbose=False)
    
    # Training del modello
    model.fit(X_train, y_train)
    
    # Predizione dati e valutazione del modello
    y_pred = model.predict(X_test)
    smape_score = smape(y_test, y_pred)

    # Plot della rilevanza delle feature
    plt.figure(figsize=(10, 6))
    plt.barh(X_train.columns, model.get_feature_importance())
    plt.title("Importanza feature - CatBoost")
    plt.xlabel("Importanza feature")
    plt.show()

    return smape_score


seed = 1
smape_mlp = train_and_test_model(Model.MLP, False, seed)


smape_mlp_k = train_and_test_model(Model.MLP, True, seed)


smape_rnd_forest = train_and_test_model(Model.RND_FOREST, True, seed)


smape_gboost = train_and_test_model(Model.GBOOST, True, seed)


smape_xgboost = train_and_test_model(Model.XGBOOST, True, seed)


smape_cat_boost = train_and_test_model(Model.CAT_BOOST, True, seed)


cols = ["MLP", "MLP_K", "RANDOM FOREST", "GBOOST", "XGBOOST", "CAT BOOST"]
rows = [smape_mlp, smape_mlp_k, smape_rnd_forest, smape_gboost, smape_xgboost, smape_cat_boost]
plt.figure(figsize=(10, 6))
plt.barh(cols, rows)
plt.title("Confronto modelli")
plt.xlabel("SMAPE score")
plt.ylabel("Modello")
plt.show()

