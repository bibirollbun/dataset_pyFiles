# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

def download_nltk_resources():
    """Descarga los recursos necesarios de NLTK."""
    try:
        stopwords.words('spanish')
    except LookupError:
        print("Descargando stopwords de NLTK...")
        nltk.download('stopwords')
    try:
       
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Descargando Punkt de NLTK...")
        nltk.download('punkt')


def clean_text(text):
    """Limpia el texto: minúsculas, elimina puntuación, números y caracteres especiales."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'\[.*?\]', '', text) # Eliminar texto entre corchetes
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Eliminar URLs
    text = re.sub(r'<.*?>+', '', text) # Eliminar etiquetas HTML
    text = re.sub(r'[^a-záéíóúñ\s]', '', text) # Mantener solo letras y espacios
    text = re.sub(r'\n', '', text) # Eliminar saltos de línea
    text = re.sub(r'\w*\d\w*', '', text) # Eliminar palabras que contienen números
    return text

def preprocess_text(text):
    """Preprocesa el texto: tokenización, eliminación de stopwords y stemming."""
    if not isinstance(text, str):
        return ""
   
    tokens = nltk.word_tokenize(text)


    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stop_words]

  
    stemmer = SnowballStemmer('spanish')
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)

def main():
    """Función principal para ejecutar el pipeline de análisis de sentimientos."""

    TRAIN_FILE_PATH = os.path.join('train.csv')
    TEST_FILE_PATH = os.path.join('submission.csv')
    SUBMISSION_FILE_PATH = 'final_submission.csv'


    download_nltk_resources()

  
    print("Cargando datos...")
    try:
     
        train_df = pd.read_csv(TRAIN_FILE_PATH)
        test_df = pd.read_csv(TEST_FILE_PATH)
        print("Datos cargados exitosamente.")
        print(f"Forma del conjunto de entrenamiento: {train_df.shape}")
        print(f"Forma del conjunto de prueba: {test_df.shape}")
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo {e.filename}.")
        print("Asegúrate de que los archivos 'test.csv' y 'submission.csv' estén en una carpeta llamada 'data'.")
        return

    # 4. Limpieza de Datos
    print("\nIniciando limpieza de datos...")

    train_df.dropna(subset=['Comentario', 'Sentimiento'], inplace=True)

   
    train_df.drop_duplicates(subset=['Comentario'], inplace=True, keep='first')

    train_df['Comentario'] = train_df['Comentario'].fillna('')
    test_df['Comentario'] = test_df['Comentario'].fillna('')
    print(f"Forma del conjunto de entrenamiento después de la limpieza: {train_df.shape}")
    print("Limpieza de datos completada.")

    print("\nIniciando preprocesamiento de texto (esto puede tardar unos minutos)...")
    train_df['Processed_Comment'] = train_df['Comentario'].apply(lambda x: preprocess_text(clean_text(x)))
    test_df['Processed_Comment'] = test_df['Comentario'].apply(lambda x: preprocess_text(clean_text(x)))
    print("Preprocesamiento de texto completado.")

    # 6. Feature Engineering (TF-IDF)
    print("\nCreando características con TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf_vectorizer.fit_transform(train_df['Processed_Comment'])
    X_test_final = tfidf_vectorizer.transform(test_df['Processed_Comment'])

    # Mapeo de la variable objetivo a números (0: negativo, 1: neutral, 2: positivo)
    sentiment_map = {'negativo': 0, 'neutral': 1, 'positivo': 2}
    train_df['Sentiment_Num'] = train_df['Sentimiento'].map(sentiment_map)
    y = train_df['Sentiment_Num']
    print("Vectorización TF-IDF completada.")

    # 7. División de Datos para Evaluación y Optimización de Hiperparámetros
    print("\nDividiendo datos para evaluación y buscando los mejores hiperparámetros con GridSearchCV...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42, class_weight='balanced')
    param_grid = {'C': [0.1, 1, 10]} 

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model_for_eval = grid_search.best_estimator_
    print(f"\nMejores hiperparámetros encontrados: {grid_search.best_params_}")

 
    print("\n--- Reporte de Evaluación del Modelo (sobre datos de validación) ---")
    y_pred_val = best_model_for_eval.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Accuracy en el conjunto de validación: {accuracy:.4f}")
    print("\nReporte de Clasificación:")
   
    target_names = [k for k, v in sorted(sentiment_map.items(), key=lambda item: item[1])]
    print(classification_report(y_val, y_pred_val, target_names=target_names))
    print("-----------------------------------------------------------------")

    # 9. Re-entrenamiento con todos los datos y Predicción Final
    print("\nRe-entrenando el modelo con TODOS los datos de entrenamiento...")
    final_model = LogisticRegression(**best_model_for_eval.get_params())
    final_model.fit(X, y) 
    print("Modelo final entrenado.")

    print("\nGenerando predicciones para el conjunto de prueba final...")
    test_predictions = final_model.predict(X_test_final)

    
    submission_df = pd.DataFrame({'ID': test_df['ID'], 'Sentimiento': test_predictions})
    submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)

    print(f"\n¡Proceso completado! El archivo '{SUBMISSION_FILE_PATH}' ha sido creado con {len(submission_df)} predicciones.")

if __name__ == '__main__':
    main()



