import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


dir_path='/kaggle/input/phems-hackathon-early-sepsis-prediction/'
train_path = '/kaggle/input/phems-hackathon-early-sepsis-prediction/'
test_path = '/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/'

def calculate_age_in_months(day, birth_datetime):

    current_day = pd.to_datetime(day)
    birth_date = pd.to_datetime(birth_datetime)

    # Calcul de l'âge en mois
    age_in_months = (current_day.year - birth_date.year) * 12 + (current_day.month - birth_date.month)

    return age_in_months
    
def find_last_drug_usage(sepsis_group, drugs_group):
    # Tri par datetime
    sepsis_group = sepsis_group.sort_values(by="measurement_datetime").copy()
    drugs_group = drugs_group.sort_values(by="drug_datetime_hourly").copy()
    
    pointer = 0
    n_drugs = len(drugs_group)

    last_drug_ids = []
    last_route_ids = []

    for i, row in sepsis_group.iterrows():
        current_time = row["measurement_datetime"]
        
        # Avancer le pointeur tant que la prise suivante est <= current_time
        while pointer < (n_drugs - 1) and drugs_group.iloc[pointer + 1]["drug_datetime_hourly"] <= current_time:
            pointer += 1
        
        # Vérifier si la prise pointée est avant ou égale à la mesure
        if n_drugs > 0 and drugs_group.iloc[pointer]["drug_datetime_hourly"] <= current_time:
            last_drug_ids.append(drugs_group.iloc[pointer]["drug_concept_id"])
            last_route_ids.append(drugs_group.iloc[pointer]["route_concept_id"])
        else:
            last_drug_ids.append(None)
            last_route_ids.append(None)

    sepsis_group["last_drug_concept_id"] = last_drug_ids
    sepsis_group["last_route_concept_id"] = last_route_ids

    return sepsis_group



def create_dataset(path, is_train=False, encoders=None):

    # 1) --------------------------------------------------- DF_SEPSIS
    df_sepsis = pd.read_csv(f"{dir_path}{path}ing_data/SepsisLabel_{path}.csv").drop_duplicates()
    print("Size "+path+" : ")
    print(df_sepsis.shape[0])
    df_sepsis['day'] = df_sepsis['measurement_datetime'].apply(lambda x: x[:10] if pd.notna(x) else None)
    df_sepsis["measurement_datetime"] = pd.to_datetime(df_sepsis["measurement_datetime"], errors="coerce")
    df_sepsis = df_sepsis.sort_values(by=["person_id", "measurement_datetime"])
    df_sepsis["temps_ecoule"] = (
        df_sepsis.groupby("person_id")["measurement_datetime"]
                 .diff()
                 .dt.total_seconds()
        / 3600
    )
    df_sepsis["temps_ecoule"] = df_sepsis["temps_ecoule"].fillna(0)
    # Extraire une colonne "day" (date sans heure)
    

    # 2) --------------------------------------------------- DF_DEMOGRAPHIC
    df_demographic = pd.read_csv(f"{dir_path}{path}ing_data/person_demographics_episode_{path}.csv")
    df_demographic = df_demographic.sort_values(by="visit_start_date").drop_duplicates(subset=["person_id"], keep="last")

    df_merged = pd.merge(df_sepsis, df_demographic, on=['person_id'], how='left')
    df_merged = df_merged.dropna(subset=["day", "birth_datetime"])
    df_merged["birth_datetime"] = pd.to_datetime(df_merged["birth_datetime"], errors="coerce")

    # Calcul de l'âge
    df_merged["age_in_months"] = df_merged.apply(
        lambda row: calculate_age_in_months(row["day"], row["birth_datetime"]), axis=1
    )
    # Supprimer des colonnes inutiles
    df_merged.drop(["visit_occurrence_id", "visit_start_date", "birth_datetime"], axis=1, inplace=True, errors='ignore')

    # 3) --------------------------------------------------- DF_DRUGS : Merge exact + “last usage”
    df_drugs = pd.read_csv(f"{dir_path}{path}ing_data/drugsexposure_{path}.csv")
    df_drugs["drug_datetime_hourly"] = pd.to_datetime(df_drugs["drug_datetime_hourly"], errors="coerce")

    #
    # 3A) Merge "exact" => current_drug_concept_id / current_route_concept_id
    #
    # Possibilité : si plusieurs prises de médicaments dans la même heure, on les concatène dans une seule cellule.
    df_drugs_agg = (
        df_drugs.groupby(["person_id", "drug_datetime_hourly"])
                .agg({
                    "drug_concept_id": lambda x: " ".join(sorted(map(str, x))),
                    "route_concept_id": lambda x: " ".join(sorted(map(str, x)))
                })
                .reset_index()
                .rename(columns={
                    "drug_concept_id": "current_drug_concept_id",
                    "route_concept_id": "current_route_concept_id"
                })
    )

    # On fait un merge (left) avec df_merged sur la (person_id, measurement_datetime == drug_datetime_hourly)
    df_merged = pd.merge(
        df_merged,
        df_drugs_agg,
        how="left",
        left_on=["person_id", "measurement_datetime"],
        right_on=["person_id", "drug_datetime_hourly"]
    )

    # On n'a plus besoin de la colonne 'drug_datetime_hourly' post-merge
    df_merged.drop(["drug_datetime_hourly"], axis=1, inplace=True, errors='ignore')

    # 3B) Ajouter la "last usage" => last_drug_concept_id / last_route_concept_id
    # On regroupe par person_id et on applique find_last_drug_usage
    groups_sepsis = []
    for pid, grp_sepsis in df_merged.groupby("person_id"):
        # Filtrer le DF de toutes les prises pour ce patient
        grp_drugs = df_drugs[df_drugs["person_id"] == pid]
        updated_grp = find_last_drug_usage(grp_sepsis, grp_drugs)
        groups_sepsis.append(updated_grp)

    df_merged = pd.concat(groups_sepsis, axis=0).reset_index(drop=True)

    # 4) --------------------------------------------------- DF_OBS
    df_obs = pd.read_csv(f"{dir_path}{path}ing_data/measurement_meds_{path}.csv")
    df_obs["day"] = df_obs['measurement_datetime'].apply(lambda x: x[:10] if pd.notna(x) else None)
    df_obs["measurement_datetime"] = pd.to_datetime(df_obs["measurement_datetime"], errors="coerce")

    # Filtrage valeurs aberrantes
    df_obs = df_obs[df_obs["Heart rate"].between(0, 200, inclusive="both")]
    df_obs = df_obs[df_obs["Respiratory rate"].between(0, 40, inclusive="both")]

    # Agrégation par jour
    df_obs_agg = df_obs.groupby(["person_id", "day"]).agg({
        "Body temperature": "max",
        "Respiratory rate": "max",
        "Heart rate": "max",
        "Measurement of oxygen saturation at periphery": "mean"
    }).reset_index()

    df_merged = df_merged.merge(df_obs_agg, on=["person_id", "day"], how="left")

    # Remplir NaN par la médiane
    for col in ["Body temperature", "Respiratory rate", "Heart rate", "Measurement of oxygen saturation at periphery"]:
        median_val = df_merged[col].median(skipna=True)
        df_merged[col] = df_merged[col].fillna(median_val)

    # 5) --------------------------------------------------- DF_LAB
    df_lab = pd.read_csv(f"{dir_path}{path}ing_data/measurement_lab_{path}.csv")
    df_lab.columns = df_lab.columns.str.replace('[', '(', regex=False).str.replace(']', ')', regex=False)
    df_lab["day"] = df_lab['measurement_datetime'].apply(lambda x: x[:10] if pd.notna(x) else None)
    df_lab["measurement_datetime"] = pd.to_datetime(df_lab["measurement_datetime"], errors="coerce")

    columns_to_exclude = ["person_id", "day", "visit_occurence_id", "measurement_datetime"]
    df_lab_agg = (
        df_lab.groupby(["person_id", "day"])
              .agg({col: "mean" for col in df_lab.columns if col not in columns_to_exclude})
              .reset_index()
    )
    df_merged = df_merged.merge(df_lab_agg, on=["person_id", "day"], how="left")

    # Remplir NaN
    numeric_cols_lab = [c for c in df_lab.columns if c not in columns_to_exclude]
    for col in numeric_cols_lab:
        median_val = df_merged[col].median(skipna=True)
        df_merged[col] = df_merged[col].fillna(median_val)

    # 6) --------------------------------------------------- Nettoyage final
    cols_to_drop = ["day", "visit_occurrence_id_x", "visit_occurrence_id_y", 
                    "visit_occurence_id", "Ionised calcium measurement"]
    df_merged.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # 7) --------------------------------------------------- Encodage
    # On encode 4 colonnes : gender + current_drug_concept_id + current_route_concept_id + last_drug_concept_id + last_route_concept_id
    cat_cols = [
        "gender",
        "current_drug_concept_id", "current_route_concept_id",
        "last_drug_concept_id", "last_route_concept_id"
    ]

    if is_train:
        for col in cat_cols:
            df_merged[col] = df_merged[col].astype(str)
            le = LabelEncoder()
            df_merged[col] = le.fit_transform(df_merged[col])
            encoders[col] = le
    else:
        for col in cat_cols:
            df_merged[col] = df_merged[col].astype(str)
            le = encoders[col]
            # Remplace éventuelles nouvelles classes inconnues par la première classe connue
            df_merged[col] = df_merged[col].where(df_merged[col].isin(le.classes_), le.classes_[0])
            df_merged[col] = le.transform(df_merged[col])

    return df_merged



encoders = {}  # On stocke les labelencoders ici

# Créer le dataset TRAIN
df_train = create_dataset("train", is_train=True, encoders=encoders)

# Créer le dataset TEST
df_test = create_dataset("test", is_train=False, encoders=encoders)


counts = df_train["SepsisLabel"].value_counts()
print(counts)
#On remarque que la répartition des classes est désequilibrée


df_train.head()


# Features et cible
X = df_train.drop(["person_id","SepsisLabel", "measurement_datetime","visit_occurrence_id"], axis=1)  # Features
y = df_train["SepsisLabel"]  # Cible binaire
groups = df_train["person_id"]  # Groupes pour s'assurer que les mêmes personnes ne sont pas divisées

# Création d'un GroupShuffleSplit pour éviter qu'un même person_id se retrouves dans le dataset de validation et de train
gss = GroupShuffleSplit(test_size=0.2, random_state=42)  # 20% des données pour la validation
train_idx, val_idx = next(gss.split(X, y, groups))

# Division des données
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# Application de SMOTE sur le jeu d'entraînement pour sur-échantilloner la classe minoritaire
smote = SMOTE(random_state=5)
X_train_np = X_train.values  # Convertir en tableau NumPy pour SMOTE
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_np, y_train)

# Conversion des données réséchantillonnées en DataFrame et Series
X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
y_train_resampled = pd.Series(y_train_resampled)


# Instanciation du modèle
xgb_model = xgb.XGBClassifier(
    n_estimators=200,       # Nombre maximal d'arbres
    learning_rate=0.05,     # Taux d'apprentissage
    max_depth=8,            # Profondeur des arbres
    subsample=0.8,          # Échantillonnage des lignes
    colsample_bytree=0.8,   # Échantillonnage des features
    random_state=42,
    reg_alpha=0.1,          # Régularisation L1 (mettre à zéro les poids inutiles)
    reg_lambda=1.0          # Régularisation L2 (réduction des poids inutiles)
)

# Entraînement avec early stopping
xgb_model.fit(
    X_train_resampled, y_train_resampled,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",             # Calcul d'AUC à chaque itération
    early_stopping_rounds=10,
    verbose=True
)




# Prédiction de la probabilité (classe 1)
y_pred_proba = xgb_model.predict_proba(X_val)[:,1]

# Calcul de l'AUC
auc_val = roc_auc_score(y_val, y_pred_proba)
print("AUC sur l'ensemble de validation :", auc_val)



# Prédictions
y_pred_class = xgb_model.predict(X_val)

# Calcul de l'accuracy
acc_val = accuracy_score(y_val, y_pred_class)

# Calcul des F1 scores pour chaque classe
f1_val_per_class = f1_score(y_val, y_pred_class, average=None)

# Affichage de l'accuracy
print("Accuracy :", acc_val)

# Affichage des F1 scores pour chaque classe
for label, f1 in enumerate(f1_val_per_class):
    print(f"F1 Score for class {label}:", f1)

# Ou vous pouvez afficher un rapport détaillé avec classification_report
print("\nClassification Report:")
print(classification_report(y_val, y_pred_class))



xgb.plot_importance(xgb_model, max_num_features=10, height=0.4)
plt.show()


lgbm_model = lgb.LGBMClassifier(
    boosting_type="gbdt",         # Gradient Boosting classique
    objective="binary",           # Pour un problème de classification binaire
    n_estimators=300,             # Nombre total d'arbres
    learning_rate=0.1,            # Taux d'apprentissage initial
    num_leaves=31,                # Nombre maximal de feuilles dans un arbre
    feature_fraction=0.8,         # Échantillonnage des features (colonnes)
    bagging_fraction=0.8,         # Échantillonnage des lignes (bagging)
    random_state=42,               # Reproductibilité
    is_unbalance=True
)

#Entrainement

lgbm_model.fit(X_train_resampled, y_train_resampled)

# Prédiction sur le jeu de validation
y_val_pred_proba = lgbm_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_val_pred_proba)
print(f"AUC pour LightGBM : {auc:.4f}")


y_pred_class = lgbm_model.predict(X_val)
acc_val = accuracy_score(y_val, y_pred_class)
f1_val  = f1_score(y_val, y_pred_class)

print("Accuracy :", acc_val)
print("F1 Score :", f1_val)



def prepare_lstm_data(df, sequence_length=10):
    X, y = [], []
    grouped = df.groupby("person_id")  # Grouper par identifiant de personne
    for _, group in grouped:
        group = group.sort_values("measurement_datetime")  # Trier par ordre temporel
        features = group.drop(["SepsisLabel", "measurement_datetime", "person_id"], axis=1, errors="ignore").values
        labels = group["SepsisLabel"].values
        # Générer les séquences
        for i in range(len(group) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(labels[i + sequence_length - 1])
    return np.array(X), np.array(y)

# Préparation des données pour LSTM
X_lstm, y_lstm = prepare_lstm_data(df_train)

# Diviser les données en ensemble d'entraînement et de validation
X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(
    X_lstm, y_lstm, 
    test_size=0.2,        # 20% des données pour la validation
    random_state=42,      # Reproductibilité
    stratify=y_lstm            # Maintenir la proportion des classes
)


# Étape 1 : Aplatir les données LSTM pour SMOTE
X_train_flat_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], -1)  # (n_samples, sequence_length * n_features)

# Étape 2 : Appliquer SMOTE
smote = SMOTE(random_state=41)
X_train_resampled_lstm, y_train_resampled_lstm = smote.fit_resample(X_train_flat_lstm, y_train_lstm)

# Étape 3 : Reformater les données resampled pour LSTM
sequence_length = X_train_lstm.shape[1]  # Garder la longueur des séquences initiales
n_features = X_train_lstm.shape[2]      # Garder le nombre de features initiales
X_train_resampled_lstm = X_train_resampled_lstm.reshape(-1, sequence_length, n_features)



# Définir le modèle LSTM amélioré
lstm_model = Sequential([
    LSTM(128, input_shape=(X_train_resampled_lstm.shape[1], X_train_resampled_lstm.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compiler avec un taux d'apprentissage personnalisé
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Utiliser EarlyStopping pour éviter l'overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entraîner le modèle
history = lstm_model.fit(
    X_train_resampled_lstm, y_train_resampled_lstm,
    validation_data=(X_val_lstm, y_val_lstm),
    epochs=5,          # Plus d'époques pour un apprentissage approfondi
    batch_size=64,      # Taille des batchs ajustée
    verbose=1
)


y_pred = lstm_model.predict(X_val_lstm)
y_pred_labels = (y_pred > 0.5).astype(int)  # Convertir les probabilités en labels binaires

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_val_lstm, y_pred_labels))
print(confusion_matrix(y_val_lstm, y_pred_labels))



# 1) Créer un identifiant "person_id_datetime"
df_test["person_id_datetime"] = df_test["person_id"].astype(str) + "_" + df_test["measurement_datetime"].astype(str)

# 2) Préparer le DataFrame de features pour la prédiction
#    => enlever les colonnes qui ne sont pas des features (person_id, datetime, etc.)
#    => on suppose que df_test contient déjà les mêmes colonnes que X_train
X_test = df_test.drop(["person_id","measurement_datetime", "person_id_datetime","visit_occurrence_id"], axis=1, errors="ignore")

# 3) Prédire la classe SepsisLabel (0 ou 1)
y_test_pred_label = xgb_model.predict(X_test) #CHANGER LE NOM DU MODELE EN FONCTION DU RESULTAT SOUHAITE : lgbm_model OU xgb_model, pour LSTM voir en dessous

# 4) Créer la table de soumission
submission = pd.DataFrame({
    "person_id_datetime": df_test["person_id_datetime"],
    "SepsisLabel": y_test_pred_label
})

# 5) Sauvegarder en CSV
submission.to_csv("submission.csv", index=False)

print("Fichier de soumission créé : SepsisLabel_submission.csv")
print(submission.head())


from IPython.display import FileLink
FileLink('submission.csv')
#Lien pour télécharger le fichier


import numpy as np
import pandas as pd

# 1) Créer un identifiant "person_id_datetime"
df_test["person_id_datetime"] = df_test["person_id"].astype(str) + "_" + df_test["measurement_datetime"].astype(str)

# 2) Préparer les données sous forme de séquences pour le modèle LSTM
def prepare_lstm_data_test(df, sequence_length=10):
    """
    Prépare les données de test pour le modèle LSTM en créant des séquences temporelles.
    Cette fonction garantit que toutes les lignes de df_test sont utilisées, 
    même si elles ne sont pas assez nombreuses pour former une séquence complète.
    """
    X = []
    ids = []
    grouped = df.groupby("person_id")
    for person_id, group in grouped:
        group = group.sort_values("measurement_datetime")
        features = group.drop(["person_id", "measurement_datetime", "person_id_datetime"], axis=1, errors="ignore").values
        person_ids = group["person_id_datetime"].values
        
        # Si le groupe contient moins de lignes que la longueur de séquence
        if len(group) < sequence_length:
            # Compléter avec des zéros pour atteindre `sequence_length`
            padded_features = np.zeros((sequence_length, features.shape[1]))
            padded_features[-len(features):] = features  # Ajouter les valeurs existantes à la fin
            X.append(padded_features)
            ids.append(person_ids[-1])  # Dernier identifiant correspondant
        else:
            for i in range(len(group) - sequence_length + 1):
                X.append(features[i:i + sequence_length])
                ids.append(person_ids[i + sequence_length - 1])  # ID correspondant à la dernière étape de la séquence
    return np.array(X), ids

# Préparer les séquences pour le test
sequence_length = X_train_lstm.shape[1]  # La longueur des séquences doit être cohérente avec l'entraînement
X_test_lstm, test_ids = prepare_lstm_data_test(df_test, sequence_length=sequence_length)

# 3) Prédire la classe SepsisLabel (0 ou 1)
y_test_pred_proba = lstm_model.predict(X_test_lstm)  # Probabilités
y_test_pred_label = (y_test_pred_proba > 0.5).astype(int).flatten()  # Seuil 0.5 pour les labels

# 4) Associer les prédictions à `df_test`
# Créer une table avec toutes les lignes originales de `df_test` pour conserver la cohérence
submission = df_test[["person_id_datetime"]].copy()
submission = submission.merge(
    pd.DataFrame({"person_id_datetime": test_ids, "SepsisLabel": y_test_pred_label}),
    on="person_id_datetime",
    how="left"
)

# Remplir les lignes sans prédiction par une valeur par défaut (par exemple, 0)
submission["SepsisLabel"] = submission["SepsisLabel"].fillna(0).astype(int)

# 5) Sauvegarder en CSV
submission.to_csv("SepsisLabel_submission_LSTM.csv", index=False)

print("Fichier de soumission créé : SepsisLabel_submission_LSTM.csv")
print(submission.head())


