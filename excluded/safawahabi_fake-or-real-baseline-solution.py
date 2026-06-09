import sys
import os
from pathlib import Path
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import re
import unicodedata
from nltk.corpus import words
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


def build_dataset(base_path="/kaggle/input/fake-or-real-the-impostor-hunt/data/train"):
    base = Path(base_path)
    rows = []

    # parcourir les sous-dossiers
    for folder in sorted(base.iterdir()):
        if folder.is_dir() and folder.name.startswith("article_"):
            # extraire id numérique
            article_id = int(folder.name.split("_")[-1])

            file1 = folder / "file_1.txt"
            file2 = folder / "file_2.txt"

            # lire contenu
            file1_content = file1.read_text(encoding="utf-8").strip() if file1.exists() else None
            file2_content = file2.read_text(encoding="utf-8").strip() if file2.exists() else None

            rows.append({
                "id": article_id,
                "file_1": file1_content,
                "file_2": file2_content
            })

    df = pd.DataFrame(rows)
    return df

df = build_dataset("/kaggle/input/fake-or-real-the-impostor-hunt/data/train")
print(df.head())


train_df = pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv")
print(train_df.head())


df_merged = df.merge(train_df, on='id')
print(df_merged.head())


y = df_merged["real_text_id"] - 1
X_train, X_test, y_train, y_test = train_test_split(df_merged, y, test_size=0.2, random_state=42)


def is_empty_text(text):
    """
    Retourne 1 si le texte est vide ou ne contient que des espaces, 0 sinon.
    """
    if not isinstance(text, str):
        return 1 
        
    if not text.strip():
        return 1
    else:
        return 0


english_words = set(words.words())

def clean_and_filter_english_v2(text):
    """
    Fonction de nettoyage plus robuste pour ne garder que les mots anglais.
    """
    if not isinstance(text, str):
        return ""
    
    # Étape 1 : Nettoyage agressif des caractères
    # On convertit en minuscules et on remplace tout ce qui n'est pas une lettre latine par un espace
    cleaned_text = re.sub(r'[^a-z\s]', ' ', text.lower())
    
    # Étape 2 : Filtration des mots non-anglais
    # On divise le texte en mots et on garde ceux qui sont dans l'ensemble 'english_words'
    filtered_words = [word for word in cleaned_text.split() if word in english_words]
    
    # Étape 3 : Reconstruction du texte nettoyé
    return ' '.join(filtered_words)


real_texts = []
for index, row in df_merged.iterrows():
    real_text_id = row["real_text_id"]
    real_texts.append(row[f'file_{real_text_id}'])

# Convertissez la liste en Series Pandas pour le vectorizer
real_texts_series = pd.Series(real_texts)

vectorizer = TfidfVectorizer(max_df=0.8, min_df=1, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(real_texts_series)

feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A1

sorted_indices = np.argsort(tfidf_scores)[::-1]
top_n = 100

astronomy_words = [feature_names[i] for i in sorted_indices[:top_n]]

print("Liste des mots d'astronomie générée automatiquement :")
print(astronomy_words)


def calculate_topicality_score(text):
    """
    Calcule le score de pertinence thématique pour un texte donné.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    # Nettoyage et tokenisation du texte (basique)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    words = re.sub(r'[^a-z\s]', ' ', text.lower()).split()
    
    if not words:
        return 0.0
    
    # Compte les mots qui sont dans notre vocabulaire thématique
    on_topic_word_count = sum(1 for word in words if word in astronomy_words)
    
    # Retourne la proportion
    return on_topic_word_count / len(words)


def get_repetition_score(text, threshold=0.3):
    """
    Calcule le ratio de répétition et retourne 1 si le texte dépasse le seuil, 0 sinon.
    """
    if not isinstance(text, str):
        return 0
    
    mots = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    if not mots:
        return 0
    
    total = len(mots)
    freqs = Counter(mots)
    
    # Trouver le nombre d'occurrences du mot le plus fréquent
    count = freqs.most_common(1)[0][1]
    
    # Calculer le ratio de la répétition
    ratio = count / total
    
    # Appliquer le seuil
    return 1 if ratio > threshold else 0


get_file_1 = FunctionTransformer(lambda x: x['file_1'], validate=False)
get_file_2 = FunctionTransformer(lambda x: x['file_2'], validate=False)

text_cleaner = FunctionTransformer(lambda x: x.apply(clean_and_filter_english_v2), validate=False)
vectorizer_1 = TfidfVectorizer(max_features=100)
vectorizer_2 = TfidfVectorizer(max_features=100)

text_pipeline_1 = Pipeline([('selector', get_file_1), ('cleaner', text_cleaner), ('vectorizer', vectorizer_1)])
text_pipeline_2 = Pipeline([('selector', get_file_2), ('cleaner', text_cleaner), ('vectorizer', vectorizer_2)])
empty_pipeline_1 = Pipeline([('selector', get_file_1), ('is_empty', FunctionTransformer( lambda x: x.apply(is_empty_text).values.reshape(-1, 1), validate=False))])
empty_pipeline_2 = Pipeline([('selector', get_file_2), ('is_empty', FunctionTransformer( lambda x: x.apply(is_empty_text).values.reshape(-1, 1), validate=False))])
topicality_pipeline_1 = Pipeline([('selector', get_file_1), ('topicality_score', FunctionTransformer(lambda x: x.apply(calculate_topicality_score).values.reshape(-1, 1), validate=False))])
topicality_pipeline_2 = Pipeline([('selector', get_file_2), ('topicality_score', FunctionTransformer(lambda x: x.apply(calculate_topicality_score).values.reshape(-1, 1), validate=False))])
repetition_pipeline_1 = Pipeline([('selector', get_file_1), ('repetition_score', FunctionTransformer(lambda x: x.apply(get_repetition_score).values.reshape(-1, 1), validate=False))])
repetition_pipeline_2 = Pipeline([('selector', get_file_2), ('repetition_score', FunctionTransformer(lambda x: x.apply(get_repetition_score).values.reshape(-1, 1), validate=False))])

# Combinez toutes les caractéristiques
features = FeatureUnion([
    ('tfidf_features_1', text_pipeline_1),
    ('tfidf_features_2', text_pipeline_2),
    ('empty_text_1', empty_pipeline_1),
    ('empty_text_2', empty_pipeline_2),
    ('topicality_score_1', topicality_pipeline_1),
    ('topicality_score_2', topicality_pipeline_2),
    ('repetition_score_1', repetition_pipeline_1),
    ('repetition_score_2', repetition_pipeline_2),
])

# Pipeline avec la Régression Logistique
pipeline_lr = Pipeline([('features', features), ('classifier', LogisticRegression(max_iter=1000))])

# Pipeline avec XGBoost
pipeline_xgb = Pipeline([('features', features), ('classifier', XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss'))])


# Entraînement et prédiction avec la Régression Logistique
print("Entraînement avec la Régression Logistique...")
pipeline_lr.fit(X_train, y_train)
y_pred_lr = pipeline_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Précision avec la Régression Logistique : {accuracy_lr:.4f}")


incorrect_predictions = (y_pred_lr != y_test)
num_errors = incorrect_predictions.sum()
print(f"Le modèle a fait {num_errors} erreurs sur {len(y_test)} exemples.")


X_test_with_preds = X_test.copy()
X_test_with_preds['y_true'] = y_test
X_test_with_preds['y_pred'] = y_pred_lr
misclassified_df = X_test_with_preds[incorrect_predictions]

# Sauvegarde en CSV
misclassified_df.to_csv("baseline-misclassified_text.csv", index=False, sep=";")


param_grid_lr = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__max_iter': [1000]
}

grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)


print("Démarrage de la recherche des meilleurs hyperparamètres...")
grid_search_lr.fit(X_train, y_train)
print("Recherche terminée.")


print("Meilleurs paramètres pour la Régression Logistique :", grid_search_lr.best_params_)
print("Meilleur score de validation :", grid_search_lr.best_score_)


best_pipeline_lr = grid_search_lr.best_estimator_
print("Le meilleur pipeline a été trouvé et peut être utilisé pour la prédiction finale.")


print("Entraînement avec XGBoost...")
pipeline_xgb.fit(X_train, y_train)
y_pred_xgb = pipeline_xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"Précision avec XGBoost : {accuracy_xgb:.4f}")


param_grid_xgb = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1]
}

grid_search_xgb = GridSearchCV(pipeline_xgb, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)

grid_search_xgb.fit(X_train, y_train)

print("Meilleurs paramètres pour XGBoost :", grid_search_xgb.best_params_)
print("Meilleur score de validation :", grid_search_xgb.best_score_)


best_pipeline_xgb = grid_search_xgb.best_estimator_

y_pred_final = best_pipeline_xgb.predict(X_test)

final_accuracy = accuracy_score(y_test, y_pred_final)

print(f"Meilleure configuration trouvée (CV score): {grid_search_xgb.best_score_:.4f}")
print(f"Score de Précision Final sur X_test : {final_accuracy:.4f}")


df_test = build_dataset("/kaggle/input/fake-or-real-the-impostor-hunt/data/test")
print(df_test.head())


y_pred_test = best_pipeline_xgb.predict(df_test)
print(y_pred_test)


df_submission = pd.DataFrame({
    "id": df_test["id"],
    "real_text_id": y_pred_test
})
print(df_submission.head())

# Sauvegarde en CSV
df_submission.to_csv("submission.csv", sep=";", index=False)

