"""
Baseline para competição Kaggle: scc-5949-IA-2025-classification
Tarefa: classificar "Engagement" (high/low) maximizando F1-Score Macro

Estratégia:
- Limpeza leve do texto
- TF-IDF (1-2 grams) + features numéricas padronizadas (reactions, comments, score)
- Classificador LinearSVC com class_weight="balanced"
- Avaliação com StratifiedKFold (5 folds)
- Treino final em todo o train e geração de submission.csv
- Comando de submissão via Kaggle API no final
"""

# ==========================
# 0. IMPORTS & SETUP
# ==========================
!pip install emoji unidecode --quiet

import os
import re
import emoji
import joblib
import numpy as np
import pandas as pd
from unidecode import unidecode

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
print(os.listdir("/kaggle/input/"))


COMP_NAME = "scc-5949-ia-2025-classification"  

# Ajustando
TRAIN_PATH = "/kaggle/input/scc-5949-ia-2025-classification/df_social_data_train.csv"  
TEST_PATH  = "/kaggle/input/scc-5949-ia-2025-classification/df_social_data_test.csv"  
ID_COL     = "ID"
SUBMISSION_PATH = "submission.csv"
MODEL_PATH = "modelo_final.pkl"


# 1. Carregando dados
train_df = pd.read_csv(TRAIN_PATH)
print("Train shape:", train_df.shape)
print(train_df.head())

# Alvo
TARGET_COL = "engagement"


# 2. Limpeza básica de texto
def basic_clean(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)       # remove URLs
    text = emoji.replace_emoji(text, replace=' ')            # remove emojis
    text = re.sub(r"[^\w\s]", " ", text)                 # pontuação → espaço
    text = unidecode(text)                                   # remove acentos
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_df["content_clean"] = train_df["content"].apply(basic_clean)

# Features
TEXT_COL = "content_clean"
NUM_COLS = ["reactions", "comments", "score"]

X = train_df[[TEXT_COL] + NUM_COLS]
y = train_df[TARGET_COL]


# 3. Split (hold-out rápido ou CV) (opcional p/ inspecionar)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)


# 4. PIPELINE: TF-IDF + NUM + LinearSVC

preprocess = ColumnTransformer(
    transformers=[
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            min_df=3,
            max_df=0.9,
            strip_accents='unicode',
            sublinear_tf=True
        ), TEXT_COL),
        ("num", StandardScaler(with_mean=False), NUM_COLS),
    ]
)

clf = LinearSVC(class_weight="balanced", random_state=42)

pipe = Pipeline([
    ("prep", preprocess),
    ("clf", clf)
])

# Fit no hold-out
pipe.fit(X_tr, y_tr)
val_preds = pipe.predict(X_val)

print("F1-Macro (val):", f1_score(y_val, val_preds, average="macro"))
print(classification_report(y_val, val_preds))

# Matriz de Confusão (opcional)
print("Confusion Matrix:\n", confusion_matrix(y_val, val_preds))



# 5. CROSS-VALIDATION
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
print("CV F1-macro:", cv_scores.mean(), "+/-", cv_scores.std())



# 6. TREINA MODELO FINAL E SALVAR
pipe.fit(X, y)
joblib.dump(pipe, MODEL_PATH)
print(f"Modelo salvo em {MODEL_PATH}")

