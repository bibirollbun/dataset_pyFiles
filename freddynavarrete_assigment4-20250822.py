!pip install langdetect


!wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin


!pip install fasttext


!pip install langid



import os
import pandas as pd
import numpy as np
import re
import string
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import fasttext
import langid
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns


# Semilla para reproducibilidad
DetectorFactory.seed = 42
np.random.seed(42)


# =============================================================================
# 1. CARGAR DATOS DE ENTRENAMIENTO
# =============================================================================
train_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
test_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"


# Cargar datos de entrenamiento
train_data = []
for folder_name in sorted(os.listdir(train_path)):
    folder_path = os.path.join(train_path, folder_name)
    if os.path.isdir(folder_path):
        try:
            with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8', errors='ignore') as f1:
                text1 = f1.read().strip()
            with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8', errors='ignore') as f2:
                text2 = f2.read().strip()
            index = int(folder_name[-4:])
            train_data.append((index, text1, text2))
        except Exception as e:
            print(f"Error reading directory {folder_name}: {e}")

df_train = pd.DataFrame(train_data, columns=['id', 'file_1', 'file_2']).set_index('id')


# Cargar datos de prueba
test_data = []
for folder_name in sorted(os.listdir(test_path)):
    folder_path = os.path.join(test_path, folder_name)
    if os.path.isdir(folder_path):
        try:
            with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8', errors='ignore') as f1:
                text1 = f1.read().strip()
            with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8', errors='ignore') as f2:
                text2 = f2.read().strip()
            index = int(folder_name[-4:])
            test_data.append((index, text1, text2))
        except Exception as e:
            print(f"Error reading directory {folder_name}: {e}")

df_test = pd.DataFrame(test_data, columns=['id', 'file_1', 'file_2']).set_index('id')



# Cargar ground truth
df_train_gt = pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv")
y_train = df_train_gt['real_text_id'].values

# Cargar modelo FastText
fasttext_model = fasttext.load_model("lid.176.bin")

print("Datos cargados correctamente")
print(f"Entrenamiento: {df_train.shape[0]} muestras")
print(f"Prueba: {df_test.shape[0]} muestras")



# Intentar cargar fastText
try:
    import fasttext
    fasttext_model = fasttext.load_model("lid.176.bin")  # modelo de detección de idiomas
    fasttext_available = True
except Exception as e:
    print("FastText no disponible:", e)
    fasttext_available = False



# =============================================================================
# 2. EXTRACCIÓN DE CARACTERÍSTICAS AVANZADAS (CORREGIDO)
# =============================================================================
print("Extrayendo características de entrenamiento...")

# Listas para almacenar características
train_features_list = []

# Procesar cada par de textos de entrenamiento
for idx, row in df_train.iterrows():
    text1 = row['file_1']
    text2 = row['file_2']
    
    features = {}
    
    # Procesar ambos textos
    for text_idx, text in enumerate([text1, text2], 1):
        # Preprocesamiento básico
        text_clean = text.lower()
        text_clean = re.sub(r'\s+', ' ', text_clean)
        text_clean = re.sub(r'[^\w\s]', '', text_clean)
        
        words = text_clean.split()
        
        # Características de longitud
        features[f'text{text_idx}_char_count'] = len(text_clean)
        features[f'text{text_idx}_word_count'] = len(words)
        features[f'text{text_idx}_avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Detección de lenguaje (múltiples métodos)
        try:
            features[f'text{text_idx}_langdetect_en'] = 1 if detect(text) == 'en' else 0
        except:
            features[f'text{text_idx}_langdetect_en'] = 0
        
        try:
            lang, confidence = langid.classify(text)
            features[f'text{text_idx}_langid_en'] = 1 if lang == 'en' else 0
            features[f'text{text_idx}_langid_confidence'] = confidence
        except:
            features[f'text{text_idx}_langid_en'] = 0
            features[f'text{text_idx}_langid_confidence'] = 0
        
        # FastText language detection (con corrección del error)
        if fasttext_available:
            try:
                # Solución al error de NumPy - manejamos la predicción de forma segura
                prediction_result = fasttext_model.predict(text.replace('\n', ' '), k=1)
                label = prediction_result[0][0] if prediction_result[0] else ''
                prob = prediction_result[1][0] if prediction_result[1] else 0.0
                
                features[f'text{text_idx}_fasttext_en'] = 1 if label == '__label__en' else 0
                features[f'text{text_idx}_fasttext_confidence'] = float(prob)  # Convertir a float explícitamente
            except Exception as e:
                features[f'text{text_idx}_fasttext_en'] = 0
                features[f'text{text_idx}_fasttext_confidence'] = 0.0
        else:
            features[f'text{text_idx}_fasttext_en'] = 0
            features[f'text{text_idx}_fasttext_confidence'] = 0.0
        
        # Características basadas en caracteres
        latin_chars = 0
        total_alpha_chars = 0
        for c in text_clean:
            try:
                if c.isalpha():
                    total_alpha_chars += 1
                    if 'LATIN' in unicodedata.name(c):
                        latin_chars += 1
            except:
                pass
        
        features[f'text{text_idx}_latin_ratio'] = latin_chars / total_alpha_chars if total_alpha_chars > 0 else 0
        
        # Características de complejidad
        features[f'text{text_idx}_unique_words_ratio'] = len(set(words)) / len(words) if words else 0
        features[f'text{text_idx}_digit_ratio'] = sum(1 for c in text_clean if c.isdigit()) / len(text_clean) if text_clean else 0
        features[f'text{text_idx}_uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    # Características comparativas
    for feature_type in ['char_count', 'word_count', 'avg_word_length', 'langdetect_en', 
                        'langid_en', 'langid_confidence', 'fasttext_en', 'fasttext_confidence',
                        'latin_ratio', 'unique_words_ratio', 'digit_ratio', 'uppercase_ratio']:
        val1 = features.get(f'text1_{feature_type}', 0)
        val2 = features.get(f'text2_{feature_type}', 0)
        features[f'{feature_type}_diff'] = val1 - val2
        if val2 != 0:
            features[f'{feature_type}_ratio'] = val1 / val2
        else:
            features[f'{feature_type}_ratio'] = 0 if val1 == 0 else 999999  # Valor grande para evitar división por cero
    
    train_features_list.append(features)

# Convertir a DataFrame
X_train = pd.DataFrame(train_features_list)


print("Extrayendo características de prueba...")

# Procesar datos de prueba
test_features_list = []

for idx, row in df_test.iterrows():
    text1 = row['file_1']
    text2 = row['file_2']
    
    features = {}
    
    # Procesar ambos textos
    for text_idx, text in enumerate([text1, text2], 1):
        # Preprocesamiento básico
        text_clean = text.lower()
        text_clean = re.sub(r'\s+', ' ', text_clean)
        text_clean = re.sub(r'[^\w\s]', '', text_clean)
        
        words = text_clean.split()
        
        # Características de longitud
        features[f'text{text_idx}_char_count'] = len(text_clean)
        features[f'text{text_idx}_word_count'] = len(words)
        features[f'text{text_idx}_avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Detección de lenguaje (múltiples métodos)
        try:
            features[f'text{text_idx}_langdetect_en'] = 1 if detect(text) == 'en' else 0
        except:
            features[f'text{text_idx}_langdetect_en'] = 0
        
        try:
            lang, confidence = langid.classify(text)
            features[f'text{text_idx}_langid_en'] = 1 if lang == 'en' else 0
            features[f'text{text_idx}_langid_confidence'] = confidence
        except:
            features[f'text{text_idx}_langid_en'] = 0
            features[f'text{text_idx}_langid_confidence'] = 0
        
        # FastText language detection
        if fasttext_available:
            try:
                prediction_result = fasttext_model.predict(text.replace('\n', ' '), k=1)
                label = prediction_result[0][0] if prediction_result[0] else ''
                prob = prediction_result[1][0] if prediction_result[1] else 0.0
                
                features[f'text{text_idx}_fasttext_en'] = 1 if label == '__label__en' else 0
                features[f'text{text_idx}_fasttext_confidence'] = float(prob)
            except Exception as e:
                features[f'text{text_idx}_fasttext_en'] = 0
                features[f'text{text_idx}_fasttext_confidence'] = 0.0
        else:
            features[f'text{text_idx}_fasttext_en'] = 0
            features[f'text{text_idx}_fasttext_confidence'] = 0.0
        
        # Características basadas en caracteres
        latin_chars = 0
        total_alpha_chars = 0
        for c in text_clean:
            try:
                if c.isalpha():
                    total_alpha_chars += 1
                    if 'LATIN' in unicodedata.name(c):
                        latin_chars += 1
            except:
                pass
        
        features[f'text{text_idx}_latin_ratio'] = latin_chars / total_alpha_chars if total_alpha_chars > 0 else 0
        
        # Características de complejidad
        features[f'text{text_idx}_unique_words_ratio'] = len(set(words)) / len(words) if words else 0
        features[f'text{text_idx}_digit_ratio'] = sum(1 for c in text_clean if c.isdigit()) / len(text_clean) if text_clean else 0
        features[f'text{text_idx}_uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    # Características comparativas
    for feature_type in ['char_count', 'word_count', 'avg_word_length', 'langdetect_en', 
                        'langid_en', 'langid_confidence', 'fasttext_en', 'fasttext_confidence',
                        'latin_ratio', 'unique_words_ratio', 'digit_ratio', 'uppercase_ratio']:
        val1 = features.get(f'text1_{feature_type}', 0)
        val2 = features.get(f'text2_{feature_type}', 0)
        features[f'{feature_type}_diff'] = val1 - val2
        if val2 != 0:
            features[f'{feature_type}_ratio'] = val1 / val2
        else:
            features[f'{feature_type}_ratio'] = 0 if val1 == 0 else 999999
    
    test_features_list.append(features)

X_test = pd.DataFrame(test_features_list)


print("Características extraídas correctamente")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# Reemplazar valores infinitos y NaN
X_train = X_train.replace([np.inf, -np.inf], 999999)
X_test = X_test.replace([np.inf, -np.inf], 999999)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


# =============================================================================
# 3. ENTRENAMIENTO DEL MODELO
# =============================================================================
print("Entrenando modelo...")

# Entrenar Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Precisión en validación cruzada: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

# Precisión en entrenamiento completo
train_preds = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_preds)
print(f"Precisión en entrenamiento completo: {train_accuracy:.4f}")



# =============================================================================
# 4. PREDICCIÓN EN DATOS DE PRUEBA
# =============================================================================
print("Prediciendo en datos de prueba...")
test_predictions = model.predict(X_test)

# Crear submission
submission = pd.DataFrame({
    'id': range(len(test_predictions)),
    'real_text_id': test_predictions
})

submission_path = '/kaggle/working/improved_submission.csv'
submission.to_csv(submission_path, index=False)
print(f"Submission guardado en: {submission_path}")


# Verificar que el archivo se creó correctamente
if os.path.exists(submission_path):
    print("✓ Archivo de submission creado exitosamente")
    submission_sample = pd.read_csv(submission_path)
    print("\nPrimeras 5 filas del archivo de submission:")
    print(submission_sample.head())
else:
    print("✗ Error: No se pudo crear el archivo de submission")


# =============================================================================
# 5. ANÁLISIS DE RESULTADOS
# =============================================================================
# Características importantes
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 características más importantes:")
print(feature_importances.head(10))

# Gráfico de características importantes
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances.head(15))
plt.title('Top 15 Características Más Importantes')
plt.tight_layout()
plt.show()

# Resultados finales
print("\n" + "="*50)
print("RESULTADOS FINALES")
print("="*50)
print(f"Muestras de entrenamiento: {len(y_train)}")
print(f"Precisión validación cruzada: {np.mean(cv_scores):.4f}")
print(f"Precisión entrenamiento completo: {train_accuracy:.4f}")
print(f"Muestras de prueba predichas: {len(test_predictions)}")
print(f"Archivo de submission creado: {submission_path}")

