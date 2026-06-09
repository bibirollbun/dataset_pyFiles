# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os

# Lister les fichiers disponibles dans le dataset
dataset_path = "/kaggle/input/home-credit-default-risk"
print(os.listdir(dataset_path))



import pandas as pd

# Charger les datasets principaux
df_train = pd.read_csv(f"{dataset_path}/application_train.csv")
df_test = pd.read_csv(f"{dataset_path}/application_test.csv")
df_bureau = pd.read_csv(f"{dataset_path}/bureau.csv")
df_previous = pd.read_csv(f"{dataset_path}/previous_application.csv")

# Afficher les premiÃ¨res lignes du dataset train
df_train.head()



missing_values = df_train.isnull().sum() / len(df_train) * 100
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

print("Colonnes avec des valeurs manquantes :")
print(missing_values)



print(df_train.describe())  # Statistiques des variables numÃ©riques
print(df_train["TARGET"].value_counts(normalize=True) * 100)  # RÃ©partition de la cible



import emoji
print(emoji.emojize("ğŸ“ˆ Voici notre graphique ! ğŸš€"))



print("\U0001F4C8 Voici notre graphique ! \U0001F680")



import matplotlib.pyplot as plt
import seaborn as sns

# Tracer un heatmap des valeurs manquantes
plt.figure(figsize=(10, 6))
sns.heatmap(df_train.isnull(), cbar=False, cmap="viridis")
plt.title("AperÃ§u des valeurs manquantes")
plt.show()



import pandas as pd

# DÃ©finir le chemin du dataset (assurez-vous qu'il est correct)
dataset_path = "/kaggle/input/home-credit-default-risk"

# Charger les donnÃ©es
df_train = pd.read_csv(f"{dataset_path}/application_train.csv")

# VÃ©rifier si le dataset est bien chargÃ©
print(df_train.shape)  # Devrait afficher (307511, 122)



import matplotlib.pyplot as plt
import seaborn as sns

# VÃ©rifier que df_train est bien dÃ©fini
if 'df_train' in locals():
    # Tracer un heatmap des valeurs manquantes
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_train.isnull(), cbar=False, cmap="viridis")
    plt.title("AperÃ§u des valeurs manquantes")
    plt.show()
else:
    print("âš ï¸� Le dataset df_train n'est pas dÃ©fini. VÃ©rifiez le chargement des donnÃ©es.")



# VÃ©rifier que df_train est bien dÃ©fini
if 'df_train' in locals():
    # Tracer un heatmap des valeurs manquantes
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_train.isnull(), cbar=False, cmap="viridis")
    plt.title("AperÃ§u des valeurs manquantes") # Sans emoji
    plt.show()
else:
    print("âš ï¸� Le dataset df_train n'est pas dÃ©fini. VÃ©rifiez le chargement des donnÃ©es.")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les donnÃ©es (assurez-vous que df_train est bien dÃ©fini avant d'exÃ©cuter le code)
# df_train = pd.read_csv(f"{dataset_path}/application_train.csv")  # Si les donnÃ©es ne sont pas encore chargÃ©es

# Configurer Matplotlib pour utiliser une police compatible avec les emojis
plt.rcParams['font.family'] = 'DejaVu Sans'

# Tracer un heatmap des valeurs manquantes
plt.figure(figsize=(12, 6))
sns.heatmap(df_train.isnull(), cbar=False, cmap="viridis")

# Ajouter un titre avec un emoji
plt.title("AperÃ§u des valeurs manquantes")

# Afficher la figure
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les donnÃ©es (assurez-vous que df_train est bien dÃ©fini avant d'exÃ©cuter le code)
# df_train = pd.read_csv(f"{dataset_path}/application_train.csv")  # Si besoin

# DÃ©finir une police compatible avec les emojis
plt.rcParams['font.family'] = 'Arial Unicode MS'

# Tracer un heatmap des valeurs manquantes
plt.figure(figsize=(12, 6))
sns.heatmap(df_train.isnull(), cbar=False, cmap="viridis")

# Ajouter un titre avec un emoji
plt.title("AperÃ§u des valeurs manquantes")

# Afficher la figure
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1ï¸�âƒ£ Calculer le nombre de valeurs manquantes par ligne
df_train["nb_missing"] = df_train.isnull().sum(axis=1)

# 2ï¸�âƒ£ CrÃ©er des catÃ©gories de valeurs manquantes
df_train["missing_category"] = pd.qcut(df_train["nb_missing"], q=5, labels=["TrÃ¨s peu", "Peu", "Moyen", "Beaucoup", "Ã‰normÃ©ment"])

# 3ï¸�âƒ£ Calculer la proportion de dÃ©faut (TARGET=1) pour chaque catÃ©gorie
missing_impact = df_train.groupby("missing_category")["TARGET"].mean()

# 4ï¸�âƒ£ Tracer le graphique
plt.figure(figsize=(10, 5))
sns.barplot(x=missing_impact.index, y=missing_impact.values, palette="magma")

# Ajouter des labels
plt.xlabel("CatÃ©gorie de valeurs manquantes")
plt.ylabel("Proportion de dÃ©faut de paiement")
plt.title("ğŸ“Š Impact des valeurs manquantes sur le risque de dÃ©faut")
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# SÃ©lection des variables Ã  visualiser
variables = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

plt.figure(figsize=(15, 10))

# Tracer les boxplots pour chaque variable
for i, var in enumerate(variables, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df_train["TARGET"], y=df_train[var])
    plt.title(f"Impact de {var} sur le dÃ©faut de paiement")
    plt.xlabel("DÃ©faut de paiement (0 = Non, 1 = Oui)")
    plt.ylabel(var)

plt.tight_layout()
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# SÃ©lection des variables d'intÃ©rÃªt
variables_interet = ["TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

# Calcul de la matrice de corrÃ©lation
correlation_matrix = df_train[variables_interet].corr()

# Affichage avec un heatmap
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("CorrÃ©lation entre les variables et le risque de dÃ©faut")
plt.show()



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# SÃ©lection des variables pertinentes
features = ["DAYS_BIRTH", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
target = "TARGET"

# CrÃ©ation du DataFrame avec uniquement ces variables
df_model = df_train[features + [target]].copy()

# Remplacement des valeurs manquantes par la mÃ©diane
imputer = SimpleImputer(strategy="median")
df_model[features] = imputer.fit_transform(df_model[features])

# SÃ©paration Train / Test
X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalisation des variables (important pour certains modÃ¨les)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("âœ… DonnÃ©es prÃ©parÃ©es avec succÃ¨s !")



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# CrÃ©ation et entraÃ®nement du modÃ¨le
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# PrÃ©dictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # ProbabilitÃ© d'appartenance Ã  la classe "1" (dÃ©faut)

# Ã‰valuation du modÃ¨le
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"ğŸ�¯ Accuracy : {accuracy:.4f}")
print(f"ğŸ”¥ AUC-ROC Score : {roc_auc:.4f}")
print("\nğŸ“Š Rapport de classification :\n", classification_report(y_test, y_pred))



import matplotlib.pyplot as plt
import seaborn as sns

# Extraire l'importance des variables
importances = model.feature_importances_
feature_names = X.columns

# CrÃ©er un DataFrame pour afficher les rÃ©sultats
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Affichage sous forme de graphique
plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
plt.title("ğŸ”� Importance des variables dans le modÃ¨le")
plt.xlabel("Importance")
plt.ylabel("Variables")
plt.show()



import plotly.express as px
import pandas as pd

# Extraire l'importance des variables
importances = model.feature_importances_
feature_names = X.columns

# CrÃ©er un DataFrame
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=True)  # Trier pour un affichage clair

# CrÃ©er un graphique interactif
fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="ğŸ”� Importance des variables dans le modÃ¨le",
    color="Importance",
    color_continuous_scale="blues"
)

# Afficher le graphique
fig.show()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# DÃ©finir les features et la cible
features = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
X = df_train[features]
y = df_train["TARGET"]

# GÃ©rer les valeurs manquantes (remplacer par la mÃ©diane)
X = X.fillna(X.median())

# SÃ©parer en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation des donnÃ©es
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Initialiser les modÃ¨les
logreg = LogisticRegression()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")

# EntraÃ®ner et Ã©valuer chaque modÃ¨le
models = {"Logistic Regression": logreg, "Random Forest": rf, "XGBoost": xgb}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print(f"ğŸ“Œ {name} - Accuracy: {acc:.4f} | AUC-ROC: {auc:.4f}")



# CrÃ©ation des nouvelles variables
df_train["CREDIT_INCOME_RATIO"] = df_train["AMT_CREDIT"] / df_train["AMT_INCOME_TOTAL"]
df_train["ANNUITY_CREDIT_RATIO"] = df_train["AMT_ANNUITY"] / df_train["AMT_CREDIT"]
df_train["AGE_YEARS"] = df_train["DAYS_BIRTH"] / -365
df_train["EMPLOYMENT_YEARS"] = df_train["DAYS_EMPLOYED"] / -365
df_train["EXT_SOURCE_MEAN"] = df_train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)

# Transformer les variables binaires
df_train["FLAG_OWN_CAR"] = df_train["FLAG_OWN_CAR"].map({"Y": 1, "N": 0})
df_train["FLAG_OWN_REALTY"] = df_train["FLAG_OWN_REALTY"].map({"Y": 1, "N": 0})

# CatÃ©goriser l'Ã¢ge
df_train["AGE_CATEGORY"] = pd.cut(df_train["AGE_YEARS"],
                                  bins=[20, 30, 50, 100], 
                                  labels=["Jeune", "Adulte", "Senior"])

# VÃ©rification
df_train[["CREDIT_INCOME_RATIO", "ANNUITY_CREDIT_RATIO", "AGE_YEARS", "EMPLOYMENT_YEARS", "EXT_SOURCE_MEAN", "AGE_CATEGORY"]].head()



import seaborn as sns
import matplotlib.pyplot as plt

# SÃ©lection des variables d'intÃ©rÃªt
new_features = ["CREDIT_INCOME_RATIO", "ANNUITY_CREDIT_RATIO", "AGE_YEARS", 
                "EMPLOYMENT_YEARS", "EXT_SOURCE_MEAN"]

# Calcul de la matrice de corrÃ©lation
corr_matrix = df_train[new_features + ["TARGET"]].corr()

# Affichage des coefficients de corrÃ©lation
print(corr_matrix["TARGET"].sort_values(ascending=False))



# Tracer une heatmap de corrÃ©lation
plt.figure(figsize=(8, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("CorrÃ©lation entre les nouvelles variables et le risque de dÃ©faut")
plt.show()



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# SÃ©lection des variables pertinentes
features = ["EXT_SOURCE_MEAN", "AGE_YEARS"]
X = df_train[features]
y = df_train["TARGET"]

# SÃ©parer en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Remplacer les valeurs manquantes par la moyenne
imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Initialiser les modÃ¨les
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# EntraÃ®ner et Ã©valuer les modÃ¨les
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    print(f"ğŸ“Œ {name} - Accuracy: {accuracy:.4f} | AUC-ROC: {auc_roc:.4f}")



# Optimisation pour XGBoost :

from sklearn.model_selection import GridSearchCV
import xgboost as xgb

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 500, 1000],
    'scale_pos_weight': [5, 10, 15]  # Pour gÃ©rer le dÃ©sÃ©quilibre
}

xgb_model = xgb.XGBClassifier()
grid_search = GridSearchCV(xgb_model, param_grid, scoring='roc_auc', cv=3, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"Meilleurs paramÃ¨tres : {grid_search.best_params_}")
print(f"Meilleur AUC-ROC : {grid_search.best_score_:.4f}")



param_grid_v2 = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05],
    'n_estimators': [100, 300, 500],
    'scale_pos_weight': [5]  # FixÃ© pour Ã©viter trop de tests
}

xgb_model_v2 = xgb.XGBClassifier()
grid_search_v2 = GridSearchCV(xgb_model_v2, param_grid_v2, scoring='roc_auc', cv=3, n_jobs=-1)
grid_search_v2.fit(X_train_scaled, y_train)

print(f"Meilleurs paramÃ¨tres V2 : {grid_search_v2.best_params_}")
print(f"Meilleur AUC-ROC V2 : {grid_search_v2.best_score_:.4f}")



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# RÃ©cupÃ©rer les noms des features depuis X_train original
feature_names = X_train.columns  # âœ… Utiliser X_train et non X_train_scaled

# Extraire l'importance des features
feature_importances = grid_search.best_estimator_.feature_importances_

# CrÃ©er un DataFrame avec les noms des features
feat_importances = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Trier par importance dÃ©croissante
feat_importances = feat_importances.sort_values(by="Importance", ascending=False)

# Visualisation
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_importances.head(15))
plt.title("Top 15 Features les plus importantes")
plt.show()



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # X_train devient un NumPy array
X_test_scaled = scaler.transform(X_test)

# Correction en utilisant X_train.columns
feature_names = X_train.columns



# VÃ©rifions toutes les importances sans les trier
feat_importances_all = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Affichons toutes les variables triÃ©es par importance dÃ©croissante
print(feat_importances_all.sort_values(by="Importance", ascending=False))



from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Imputation des NaN avec la mÃ©diane
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)

# EntraÃ®ner un Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_imputed, y_train)

# Extraire l'importance des features
rf_feature_importances = rf_model.feature_importances_

# CrÃ©er un DataFrame avec les noms des variables
rf_feat_importances = pd.DataFrame({'Feature': feature_names, 'Importance': rf_feature_importances})
rf_feat_importances = rf_feat_importances.sort_values(by="Importance", ascending=False)

# Visualisation
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=rf_feat_importances.head(15))
plt.title("Top 15 Features les plus importantes (Random Forest)")
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Extraire l'importance des features pour XGBoost
feature_importances_xgb = grid_search.best_estimator_.feature_importances_

# CrÃ©er un DataFrame
feat_importances_xgb = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances_xgb})
feat_importances_xgb = feat_importances_xgb.sort_values(by="Importance", ascending=False)

# Visualisation
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_importances_xgb.head(15), palette="Blues_r")
plt.title("Top 15 Features les plus importantes (XGBoost)")
plt.show()



from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

# PrÃ©dictions de probabilitÃ©s
y_pred_proba = grid_search.best_estimator_.predict_proba(X_test_scaled)[:, 1]

# Calcul de la courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Calcul de la courbe Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

# Tracer la courbe ROC
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'AUC-ROC: {roc_auc:.4f}', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Courbe ROC")
plt.legend()

# Tracer la courbe Precision-Recall
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'AUC PR: {pr_auc:.4f}', color='red')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Courbe Precision-Recall")
plt.legend()

plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# ProbabilitÃ©s prÃ©dites
y_pred_proba = grid_search.best_estimator_.predict_proba(X_test_scaled)[:, 1]

# Tester diffÃ©rents seuils
thresholds = np.arange(0.1, 0.9, 0.05)
precisions, recalls, f1_scores = [], [], []

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

# Visualiser les rÃ©sultats
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions, label='Precision', marker='o')
plt.plot(thresholds, recalls, label='Recall', marker='o')
plt.plot(thresholds, f1_scores, label='F1-Score', marker='o')
plt.xlabel('Seuil de classification')
plt.ylabel('Score')
plt.title('Influence du seuil de classification sur les mÃ©triques')
plt.legend()
plt.show()


precision_score(y_test, y_pred, zero_division=0)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# ProbabilitÃ©s prÃ©dites
y_pred_proba = grid_search.best_estimator_.predict_proba(X_test_scaled)[:, 1]

# Tester diffÃ©rents seuils
thresholds = np.arange(0.1, 0.9, 0.05)  # Seuils de 0.1 Ã  0.9 avec un pas de 0.05
precisions, recalls, f1_scores = [], [], []

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
    precisions.append(precision_score(y_test, y_pred, zero_division=0))  # zero_division=0 pour Ã©viter l'erreur
    recalls.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

# Visualiser les rÃ©sultats
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions, label='Precision', marker='o', color='blue')
plt.plot(thresholds, recalls, label='Recall', marker='o', color='orange')
plt.plot(thresholds, f1_scores, label='F1-Score', marker='o', color='green')
plt.xlabel('Seuil de classification')
plt.ylabel('Score')
plt.title('Influence du seuil de classification sur les mÃ©triques')
plt.legend()
plt.grid(True)
plt.show()



from sklearn.metrics import confusion_matrix, classification_report

# GÃ©nÃ©rer les prÃ©dictions finales avec le seuil optimal (0.5)
y_pred_final = (y_pred_proba >= 0.5).astype(int)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred_final)

# Afficher le rapport de classification
print("ğŸ“‹ Rapport de classification final :")
print(classification_report(y_test, y_pred_final))

# Afficher la matrice de confusion
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("ğŸ“Š Matrice de confusion finale")
plt.xlabel("PrÃ©dictions")
plt.ylabel("Vraies valeurs")
plt.show()



import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Charger le modÃ¨le et le scaler
model = joblib.load('final_xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

def preprocess_data(df):
    """PrÃ©traiter les nouvelles donnÃ©es pour le scoring."""
    # CrÃ©er les nouvelles variables
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["AGE_YEARS"] = df["DAYS_BIRTH"] / -365
    df["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED"] / -365
    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)

    # Supprimer les colonnes inutiles
    features = ["CREDIT_INCOME_RATIO", "ANNUITY_CREDIT_RATIO", "AGE_YEARS", "EMPLOYMENT_YEARS", "EXT_SOURCE_MEAN"]
    df = df[features]

    # GÃ©rer les valeurs manquantes
    df.fillna(df.mean(), inplace=True)

    # Standardiser les donnÃ©es
    df_scaled = scaler.transform(df)
    return df_scaled

def predict_risk(df, threshold=0.5):
    """PrÃ©dire le risque de dÃ©faut avec un seuil personnalisÃ©."""
    df_scaled = preprocess_data(df)
    proba = model.predict_proba(df_scaled)[:, 1]
    predictions = (proba >= threshold).astype(int)
    return predictions, proba



import os
print(os.getcwd())  # Affiche le rÃ©pertoire de travail actuel



print(os.listdir())  # Liste les fichiers du rÃ©pertoire courant



import joblib
joblib.dump(model, '/kaggle/working/final_xgboost_model.pkl')



import joblib

# Charger le modÃ¨le
model = joblib.load('/kaggle/working/final_xgboost_model.pkl')
print("ModÃ¨le chargÃ© avec succÃ¨s !")



import joblib

# Charger le modÃ¨le XGBoost final
model = joblib.load('/kaggle/working/final_xgboost_model.pkl')
print(" ModÃ¨le XGBoost chargÃ© avec succÃ¨s !")



import pandas as pd

# Exemple de nouvelles donnÃ©es
new_data = pd.DataFrame({
    'EXT_SOURCE_MEAN': [0.5, 0.3],
    'AGE_YEARS': [45, 32]
})

# PrÃ©diction
predictions = model.predict(new_data)
print("PrÃ©dictions :", predictions)

# PrÃ©dictions de probabilitÃ©s
pred_probs = model.predict_proba(new_data)[:, 1]
print("ProbabilitÃ©s de dÃ©faut :", pred_probs)



import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Charger le modÃ¨le et le scaler
model = joblib.load('/kaggle/working/final_xgboost_model.pkl')
scaler = joblib.load('/kaggle/working/scaler.pkl')

# CrÃ©er le pipeline
pipeline = Pipeline([
    ('scaler', scaler),
    ('model', model)
])

# Fonction de scoring
def predict_risk(df_input):
    """
    df_input : DataFrame avec les caractÃ©ristiques des clients
    Retourne : PrÃ©dictions et probabilitÃ©s de dÃ©faut
    """
    predictions = pipeline.predict(df_input)
    probabilities = pipeline.predict_proba(df_input)[:, 1]  # ProbabilitÃ© de dÃ©faut
    return predictions, probabilities



import os
print(os.listdir('/kaggle/working'))



import joblib

# Sauvegarder le scaler
joblib.dump(scaler, '/kaggle/working/scaler.pkl')
print("âœ… Scaler sauvegardÃ© avec succÃ¨s !")



import joblib

# Charger le modÃ¨le et le scaler
model = joblib.load('/kaggle/working/final_xgboost_model.pkl')
scaler = joblib.load('/kaggle/working/scaler.pkl')

print("âœ… ModÃ¨le et scaler chargÃ©s avec succÃ¨s !")



import pandas as pd
import numpy as np
import joblib

# Charger le modÃ¨le et le scaler
model = joblib.load('/kaggle/working/final_xgboost_model.pkl')
scaler = joblib.load('/kaggle/working/scaler.pkl')

def predict_default(input_data):
    """
    Pipeline de scoring pour prÃ©dire le risque de dÃ©faut de paiement.
    - Normalise les donnÃ©es
    - Retourne la probabilitÃ© de dÃ©faut
    """
    # Convertir les donnÃ©es en DataFrame
    df = pd.DataFrame(input_data, index=[0])
    
    # Normaliser les donnÃ©es
    df_scaled = scaler.transform(df)
    
    # PrÃ©dire la probabilitÃ© de dÃ©faut
    proba = model.predict_proba(df_scaled)[:, 1]
    
    return proba[0]



# DonnÃ©es dâ€™exemple
client_1 = {'EXT_SOURCE_MEAN': 0.6, 'AGE_YEARS': 35}
client_2 = {'EXT_SOURCE_MEAN': 0.3, 'AGE_YEARS': 50}

# PrÃ©dictions
print(f"ğŸ’¡ Client 1 - ProbabilitÃ© de dÃ©faut : {predict_default(client_1):.4f}")
print(f"ğŸ’¡ Client 2 - ProbabilitÃ© de dÃ©faut : {predict_default(client_2):.4f}")








