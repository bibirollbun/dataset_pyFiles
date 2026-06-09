
# CELDA 1: Importar Librerías

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack
import re
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Librerías importadas correctamente")




# CELDA 2: Cargar Datos

train = pd.read_csv('/kaggle/input/traveler-insights-utb/train.csv')
test_real = pd.read_csv('/kaggle/input/traveler-insights-utb/submission.csv')  

print("="*80)
print("CARGA DE DATOS COMPLETADA")
print("="*80)

print(f"Tamaño train: {train.shape}")
print(f"Tamaño test_real: {test_real.shape}")
print(f"Columnas train: {train.columns.tolist()}")
print(f"Columnas test_real: {test_real.columns.tolist()}")

display(train.head())
display(test_real.head())




# CELDA 3: Preprocesamiento de Texto

def limpiar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto)
    texto = re.sub(r'@\w+|#\w+', '', texto)
    texto = re.sub(r'[^a-záéíóúñü\s.,!?]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

train['Comentario_limpio'] = train['Comentario'].apply(limpiar_texto)
test_real['Comentario_limpio'] = test_real['Comentario'].apply(limpiar_texto)

train['longitud_comentario'] = train['Comentario'].fillna('').apply(len)
test_real['longitud_comentario'] = test_real['Comentario'].fillna('').apply(len)

train['Valoración_num'] = train['Valoración_num'].fillna(train['Valoración_num'].median())
test_real['Valoración_num'] = test_real['Valoración_num'].fillna(test_real['Valoración_num'].median())

print("Preprocesamiento completado")




# CELDA 4: Extracción de Características

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95,
    strip_accents='unicode'
)

X_tfidf_train = tfidf.fit_transform(train['Comentario_limpio'])
features_adicionales_train = train[['Valoración_num', 'longitud_comentario']].values
X_train_combined = hstack([X_tfidf_train, features_adicionales_train])
y_train = train['Sentimiento']

print(f"Características combinadas train: {X_train_combined.shape}")




# CELDA 5: División Train/Validación

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_combined, y_train, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train
)

print(f"Train: {X_tr.shape}, Val: {X_val.shape}")




# CELDA 6: Entrenar Modelo 

from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_tr, y_tr)

y_pred_val = nb_model.predict(X_val)

print(f"Accuracy validación: {accuracy_score(y_val, y_pred_val):.4f}")
print(classification_report(y_val, y_pred_val, target_names=['Negativo', 'Neutral', 'Positivo']))

cm = confusion_matrix(y_val, y_pred_val)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negativo', 'Neutral', 'Positivo'],
            yticklabels=['Negativo', 'Neutral', 'Positivo'])
plt.show()

best_model = nb_model



# CELDA 7: Reentrenar con Todos los Datos
best_model.fit(X_train_combined, y_train)
print("Modelo reentrenado con todos los datos correctamente")



# CELDA 8: Generar Características Test


submission_template = pd.read_csv('/kaggle/input/traveler-insights-utb/submission.csv')
ids_kaggle = submission_template['ID'].tolist()
test_129 = test_real[test_real['ID'].isin(ids_kaggle)].copy()

X_tfidf_test = tfidf.transform(test_129['Comentario_limpio'])
features_test = test_129[['Valoración_num', 'longitud_comentario']].values
X_test_combined = hstack([X_tfidf_test, features_test])

print(f"Características combinadas test (129 filas): {X_test_combined.shape}")



# CELDA 9: Predecir y Crear Submission (versión corregida)
predictions_final = best_model.predict(X_test_combined)


predictions_final = [int(p) if isinstance(p, np.integer) else 
                     {'negativo':0, 'neutral':1, 'positivo':2}[p] if isinstance(p, str) else p
                     for p in predictions_final]


submission_final = pd.DataFrame({
    'ID': submission_template['ID'],
    'Sentimiento': [predictions_final[test_129['ID'].tolist().index(i)] for i in submission_template['ID']]
})

submission_final['Sentimiento'] = submission_final['Sentimiento'].astype(int)

print(f"Filas en submission: {len(submission_final)}")
display(submission_final.head())

submission_final.to_csv('submission_final.csv', index=False)
print("Archivo 'submission_final.csv' generado correctamente")


