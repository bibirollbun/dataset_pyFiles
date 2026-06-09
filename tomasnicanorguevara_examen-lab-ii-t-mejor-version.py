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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración de paths
train_path = '/kaggle/input/petfinder-adoption-prediction/train/train.csv'
test_path = '/kaggle/input/petfinder-adoption-prediction/test/test.csv'
breed_labels_path = '/kaggle/input/petfinder-adoption-prediction/breed_labels.csv'
color_labels_path = '/kaggle/input/petfinder-adoption-prediction/color_labels.csv'
state_labels_path = '/kaggle/input/petfinder-adoption-prediction/state_labels.csv'

# Cargar datos
print("Cargando datos...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
breed_labels = pd.read_csv(breed_labels_path)
color_labels = pd.read_csv(color_labels_path)
state_labels = pd.read_csv(state_labels_path)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Feature Engineering
def extract_sentiment_features(df, mode='train'):
    """Extrae características de sentiment análisis"""
    sentiment_path = f'/kaggle/input/petfinder-adoption-prediction/{mode}_sentiment/'
    
    sentiment_features = []
    
    for pet_id in df['PetID']:
        try:
            with open(f'{sentiment_path}{pet_id}.json', 'r') as f:
                sentiment_data = json.load(f)
                
            if 'documentSentiment' in sentiment_data:
                magnitude = sentiment_data['documentSentiment'].get('magnitude', 0)
                score = sentiment_data['documentSentiment'].get('score', 0)
            else:
                magnitude = 0
                score = 0
                
            # Extraer información de lenguaje
            language = sentiment_data.get('language', 'unknown')
            
            sentiment_features.append({
                'PetID': pet_id,
                'sentiment_magnitude': magnitude,
                'sentiment_score': score,
                'sentiment_language': language
            })
        except:
            sentiment_features.append({
                'PetID': pet_id,
                'sentiment_magnitude': 0,
                'sentiment_score': 0,
                'sentiment_language': 'unknown'
            })
    
    return pd.DataFrame(sentiment_features)

def extract_metadata_features(df, mode='train'):
    """Extrae características de metadatos de imágenes"""
    metadata_path = f'/kaggle/input/petfinder-adoption-prediction/{mode}_metadata/'
    
    metadata_features = []
    
    for pet_id in df['PetID']:
        vertex_x_all = []
        vertex_y_all = []
        bounding_confidence_all = []
        bounding_importance_all = []
        dominant_colors = []
        label_scores = []
        
        try:
            # Buscar todos los archivos de metadata para este PetID
            files = [f for f in os.listdir(metadata_path) if f.startswith(pet_id)]
            
            for file in files:
                with open(os.path.join(metadata_path, file), 'r') as f:
                    metadata = json.load(f)
                
                # Extraer características de rostros/objetos
                if 'faceAnnotations' in metadata:
                    for face in metadata['faceAnnotations']:
                        if 'boundingPoly' in face:
                            for vertex in face['boundingPoly'].get('vertices', []):
                                vertex_x_all.append(vertex.get('x', 0))
                                vertex_y_all.append(vertex.get('y', 0))
                        bounding_confidence_all.append(face.get('detectionConfidence', 0))
                
                # Extraer colores dominantes
                if 'imagePropertiesAnnotation' in metadata:
                    colors = metadata['imagePropertiesAnnotation'].get('dominantColors', {}).get('colors', [])
                    for color in colors:
                        dominant_colors.append(color.get('score', 0))
                        
                # Extraer labels
                if 'labelAnnotations' in metadata:
                    for label in metadata['labelAnnotations']:
                        label_scores.append(label.get('score', 0))
        except:
            pass
        
        metadata_features.append({
            'PetID': pet_id,
            'vertex_x_mean': np.mean(vertex_x_all) if vertex_x_all else 0,
            'vertex_y_mean': np.mean(vertex_y_all) if vertex_y_all else 0,
            'bounding_confidence_mean': np.mean(bounding_confidence_all) if bounding_confidence_all else 0,
            'dominant_color_mean': np.mean(dominant_colors) if dominant_colors else 0,
            'label_score_mean': np.mean(label_scores) if label_scores else 0,
            'num_metadata_files': len([f for f in os.listdir(metadata_path) if f.startswith(pet_id)])
        })
    
    return pd.DataFrame(metadata_features)

# Crear características básicas
def create_features(df):
    """Crea características adicionales"""
    df = df.copy()
    
    # Features de texto
    df['Description_length'] = df['Description'].fillna('').apply(len)
    df['Description_words'] = df['Description'].fillna('').apply(lambda x: len(x.split()))
    df['Name_length'] = df['Name'].fillna('').apply(len)
    df['has_name'] = (df['Name'].fillna('') != '').astype(int)
    
    # Features de combinaciones
    df['Age_MaturitySize'] = df['Age'] * df['MaturitySize']
    df['Age_FurLength'] = df['Age'] * df['FurLength']
    df['Fee_Health'] = df['Fee'] * df['Health']
    df['Fee_Vaccinated'] = df['Fee'] * df['Vaccinated']
    df['Fee_Dewormed'] = df['Fee'] * df['Dewormed']
    df['Fee_Sterilized'] = df['Fee'] * df['Sterilized']
    
    # Features de agrupación
    df['Health_Total'] = df['Vaccinated'] + df['Dewormed'] + df['Sterilized']
    
    # Features de color
    df['Color_count'] = (df['Color1'] != 0).astype(int) + \
                        (df['Color2'] != 0).astype(int) + \
                        (df['Color3'] != 0).astype(int)
    
    # Features de video/foto
    df['Total_media'] = df['PhotoAmt'] + df['VideoAmt']
    df['has_video'] = (df['VideoAmt'] > 0).astype(int)
    
    return df

print("\nExtrayendo características de sentiment...")
train_sentiment = extract_sentiment_features(train_df, 'train')
test_sentiment = extract_sentiment_features(test_df, 'test')

print("Extrayendo características de metadata...")
train_metadata = extract_metadata_features(train_df, 'train')
test_metadata = extract_metadata_features(test_df, 'test')

print("Creando características adicionales...")
train_df = create_features(train_df)
test_df = create_features(test_df)

# Merge features
train_df = train_df.merge(train_sentiment, on='PetID', how='left')
train_df = train_df.merge(train_metadata, on='PetID', how='left')
test_df = test_df.merge(test_sentiment, on='PetID', how='left')
test_df = test_df.merge(test_metadata, on='PetID', how='left')

# Preparar datos para el modelo
features_to_drop = ['PetID', 'Name', 'Description', 'RescuerID', 'AdoptionSpeed', 'sentiment_language']
X = train_df.drop([col for col in features_to_drop if col in train_df.columns], axis=1)
y = train_df['AdoptionSpeed']
X_test = test_df.drop([col for col in features_to_drop if col in test_df.columns], axis=1)

print(f"\nFeatures utilizadas: {X.shape[1]}")
print(f"Distribución de clases:\n{y.value_counts().sort_index()}")

# Configuración del modelo LightGBM
params = {
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Cross-validation
n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_predictions = np.zeros((len(X), 5))
test_predictions = np.zeros((len(X_test), 5))
kappa_scores = []

print("\nEntrenando modelo con Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Predicciones OOF
    oof_predictions[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    
    # Predicciones de test
    test_predictions += model.predict(X_test, num_iteration=model.best_iteration) / n_folds
    
    # Calcular Cohen's Kappa
    oof_pred_classes = np.argmax(oof_predictions[val_idx], axis=1)
    kappa = cohen_kappa_score(y_val, oof_pred_classes, weights='quadratic')
    kappa_scores.append(kappa)
    print(f"Fold {fold + 1} Kappa Score: {kappa:.4f}")

# Kappa score promedio
oof_pred_classes_all = np.argmax(oof_predictions, axis=1)
overall_kappa = cohen_kappa_score(y, oof_pred_classes_all, weights='quadratic')
print(f"\nOverall OOF Kappa Score: {overall_kappa:.4f}")
print(f"Mean CV Kappa Score: {np.mean(kappa_scores):.4f} (+/- {np.std(kappa_scores):.4f})")

# Crear submission
submission = pd.DataFrame({
    'PetID': test_df['PetID'],
    'AdoptionSpeed': np.argmax(test_predictions, axis=1)
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")
print(f"Submission shape: {submission.shape}")
print("\nDistribución de predicciones:")
print(submission['AdoptionSpeed'].value_counts().sort_index())

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importance(importance_type='gain')
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nTop 20 características más importantes:")
print(feature_importance.head(20))

