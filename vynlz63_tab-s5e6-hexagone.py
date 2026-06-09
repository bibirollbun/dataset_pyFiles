# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import MinMaxScaler, RobustScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier

from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import xgboost as xgb

import category_encoders as ce

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


# Dans X, on va prendre toutes les colonnes sauf "id" qui est unique
FEATURES = [c for c in df_test.columns if c != "id"] 

CAT_COLS = ["Soil Type", "Crop Type"] # Données catégorielles
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS] # Données numériques
LABEL = "Fertilizer Name" # Le label ou le Y

# Nombre de classes
CLASSES = 7


enc = OrdinalEncoder()
enc.fit(df_train[CAT_COLS])
df_train[CAT_COLS] = enc.transform(df_train[CAT_COLS])
df_test[CAT_COLS] = enc.transform(df_test[CAT_COLS]) 


X = df_train[FEATURES]
y = df_train[LABEL]
X_test = df_test[FEATURES]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


model = AdaBoostClassifier()
model.fit(X_train, y_train)


pred = model.predict(X_test)


# Calculer le score à partir de la métrique, en comparant pred et y_test. 
# N'oubliez pas d'importer la métrique ! 
from sklearn.metrics import accuracy_score  # ou une autre métrique
pred_val = model.predict(X_val)
score_val = accuracy_score(y_val,pred_val)
print(score_val)


# Intégration de `apk`, `mapk`, et génération des top-3 prédictions dans le même script

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb

# ─────────── Fonctions apk et mapk ───────────
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k for a single instance.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k over multiple instances.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# ─────────── Chargement des données ───────────
base_path = "/kaggle/input/playground-series-s5e6"
train_path = os.path.join(base_path, "train.csv")
test_path  = os.path.join(base_path, "test.csv")
sub_path   = os.path.join(base_path, "sample_submission.csv")

if not os.path.exists(train_path) or not os.path.exists(test_path) or not os.path.exists(sub_path):
    print("⚠️ Les fichiers n'ont pas été trouvés dans /kaggle/input/playground-series-s5e6.")
    print("   Vérifie que tu es bien dans l’environnement Kaggle ou que les noms de fichiers sont corrects.")
else:
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    sub      = pd.read_csv(sub_path)

    # ─────────── Préparation des colonnes ───────────
    FEATURES = [c for c in df_test.columns if c != "id"]
    CAT_COLS = ["Soil Type", "Crop Type"]
    LABEL    = "Fertilizer Name"

    # Encodage ordinal des variables catégorielles
    enc = OrdinalEncoder()
    enc.fit(df_train[CAT_COLS])
    df_train[CAT_COLS] = enc.transform(df_train[CAT_COLS])
    df_test[CAT_COLS]  = enc.transform(df_test[CAT_COLS])

    X      = df_train[FEATURES]
    y      = df_train[LABEL]
    X_test = df_test[FEATURES]

    # ─────────── Split train/validation ───────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # ─────────── Définition des modèles ───────────
    models = {
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(
            use_label_encoder=False, eval_metric="mlogloss", random_state=42
        ),
    }

    results = []
    all_top3_val = []  # pour stocker les top-3 preds sur la validation
    actual_val = [[label] for label in y_val]  # actual sous forme [[label], …]

    for name, model in models.items():
        # Entraînement
        model.fit(X_train, y_train)
        # Prédiction sur validation
        pred_val = model.predict(X_val)
        acc = accuracy_score(y_val, pred_val)

        # Calcul du MAP@3 sur la validation
        # 1. Probabilités sur X_val
        probs_val = model.predict_proba(X_val)
        # 2. Extraction des indices des trois classes les plus probables
        top3_idx_val = np.argsort(probs_val, axis=1)[:, -3:][:, ::-1]
        # 3. Conversion en noms de classes
        class_labels = model.classes_
        top3_labels_val = np.vectorize(lambda x: class_labels[x])(top3_idx_val)
        # 4. Constituer la liste des listes de prédictions val
        pred_list_val = [list(row) for row in top3_labels_val]
        # 5. Calcul MAP@3
        map3 = mapk(actual_val, pred_list_val, k=3)

        # Stockage et affichage
        results.append({
            "Model": name,
            "Validation Accuracy": acc,
            "MAP@3 (val)": map3
        })
        print(f"--- {name} ---")
        print(f"Accuracy on validation : {acc:.4f}")
        print(f"MAP@3 on validation   : {map3:.4f}")
        print(classification_report(y_val, pred_val, zero_division=0))
        print()

        # On garde les top-3 de l'unique (à titre d'exemple, pour le meilleur modèle)
        # Ici on met à jour all_top3_val pour le dernier modèle itéré,
        # mais on pourra réutiliser uniquement pour le modèle final retenu.
        all_top3_val = pred_list_val

    # ─────────── Tableau comparatif ───────────
    results_df = (
        pd.DataFrame(results)
        .sort_values(by="Validation Accuracy", ascending=False)
        .reset_index(drop=True)
    )
    print("=== Tableau comparatif des modèles (Accuracy et MAP@3) ===")
    print(results_df, "\n")

    # ─────────── Choix du meilleur modèle et prédiction sur test ───────────
    best_model_name = results_df.loc[0, "Model"]
    best_model      = models[best_model_name]
    print(f"Meilleur modèle sélectionné : {best_model_name}\n")

    # Réentraîne ce meilleur modèle sur l'ensemble train+validation
    best_model.fit(X, y)

    # ─── Génération des top-3 prédictions sur le test ───
    probs_test = best_model.predict_proba(X_test)
    top3_idx_test = np.argsort(probs_test, axis=1)[:, -3:][:, ::-1]
    class_labels_test = best_model.classes_
    top3_labels_test = np.vectorize(lambda x: class_labels_test[x])(top3_idx_test)
    # Chaque ligne → liste des 3 classes
    top3_list_test = [list(row) for row in top3_labels_test]

    # On veut écrire ces 3 classes dans une seule colonne, séparées par espace
    sub[LABEL] = [" ".join(row) for row in top3_list_test]

    # Affichage des premières lignes et enregistrement
    print("Premières lignes du fichier de soumission (top-3 séparé par espace) :")
    print(sub.head(), "\n")
    sub.to_csv("submission.csv", index=False)
    print("✔️ Fichier 'submission.csv' généré dans le répertoire courant.")



sub[LABEL] = model.predict(X_test)
sub.to_csv("submission.csv", index=False)

