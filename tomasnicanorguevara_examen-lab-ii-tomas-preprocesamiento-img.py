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
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("NOTEBOOK 2: PREPROCESAMIENTO DE IMÁGENES Y METADATA (MEJORADO)")
print("="*60)

# Paths
train_path = '/kaggle/input/petfinder-adoption-prediction/train/train.csv'
test_path = '/kaggle/input/petfinder-adoption-prediction/test/test.csv'

# Cargar datos
print("\nCargando datos...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

def extract_metadata_features(df, mode='train'):
    """Extrae características avanzadas de metadatos de imágenes"""
    metadata_path = f'/kaggle/input/petfinder-adoption-prediction/{mode}_metadata/'
    
    metadata_features = []
    
    print(f"\nExtrayendo metadata features de {mode}...")
    for pet_id in tqdm(df['PetID']):
        # Inicializar listas para cada tipo de feature
        vertex_x_all = []
        vertex_y_all = []
        bounding_confidence_all = []
        bounding_importance_all = []
        
        # Color features
        dominant_colors_rgb = {'red': [], 'green': [], 'blue': []}
        dominant_pixel_fractions = []
        dominant_scores = []
        
        # Label features
        label_scores = []
        label_descriptions = []
        
        # Crop hints
        crop_confidences = []
        crop_importance_fractions = []
        
        # Image properties
        image_widths = []
        image_heights = []
        
        # Face detection features
        face_joy_likelihood = []
        face_sorrow_likelihood = []
        face_anger_likelihood = []
        face_surprise_likelihood = []
        face_blur_likelihood = []
        face_headwear_likelihood = []
        
        # Object/Logo detection
        num_logos = 0
        num_texts = 0
        num_landmarks = 0
        
        try:
            # Buscar todos los archivos de metadata para este PetID
            files = [f for f in os.listdir(metadata_path) if f.startswith(pet_id)]
            num_files = len(files)
            
            for file in files:
                try:
                    with open(os.path.join(metadata_path, file), 'r') as f:
                        metadata = json.load(f)
                    
                    # Face Annotations (más detalladas)
                    if 'faceAnnotations' in metadata:
                        for face in metadata['faceAnnotations']:
                            if 'boundingPoly' in face:
                                for vertex in face['boundingPoly'].get('vertices', []):
                                    vertex_x_all.append(vertex.get('x', 0))
                                    vertex_y_all.append(vertex.get('y', 0))
                            bounding_confidence_all.append(face.get('detectionConfidence', 0))
                            
                            # Likelihood features (convertir a valores numéricos)
                            likelihood_map = {'VERY_UNLIKELY': 0, 'UNLIKELY': 1, 'POSSIBLE': 2, 
                                            'LIKELY': 3, 'VERY_LIKELY': 4, 'UNKNOWN': 2}
                            face_joy_likelihood.append(likelihood_map.get(face.get('joyLikelihood', 'UNKNOWN'), 2))
                            face_sorrow_likelihood.append(likelihood_map.get(face.get('sorrowLikelihood', 'UNKNOWN'), 2))
                            face_anger_likelihood.append(likelihood_map.get(face.get('angerLikelihood', 'UNKNOWN'), 2))
                            face_surprise_likelihood.append(likelihood_map.get(face.get('surpriseLikelihood', 'UNKNOWN'), 2))
                            face_blur_likelihood.append(likelihood_map.get(face.get('blurredLikelihood', 'UNKNOWN'), 2))
                            face_headwear_likelihood.append(likelihood_map.get(face.get('headwearLikelihood', 'UNKNOWN'), 2))
                    
                    # Image Properties - Colores dominantes
                    if 'imagePropertiesAnnotation' in metadata:
                        colors = metadata['imagePropertiesAnnotation'].get('dominantColors', {}).get('colors', [])
                        for color_info in colors:
                            dominant_scores.append(color_info.get('score', 0))
                            dominant_pixel_fractions.append(color_info.get('pixelFraction', 0))
                            if 'color' in color_info:
                                rgb = color_info['color']
                                dominant_colors_rgb['red'].append(rgb.get('red', 0))
                                dominant_colors_rgb['green'].append(rgb.get('green', 0))
                                dominant_colors_rgb['blue'].append(rgb.get('blue', 0))
                    
                    # Label Annotations
                    if 'labelAnnotations' in metadata:
                        for label in metadata['labelAnnotations']:
                            label_scores.append(label.get('score', 0))
                            label_descriptions.append(label.get('description', '').lower())
                    
                    # Crop Hints
                    if 'cropHintsAnnotation' in metadata:
                        hints = metadata['cropHintsAnnotation'].get('cropHints', [])
                        for hint in hints:
                            crop_confidences.append(hint.get('confidence', 0))
                            crop_importance_fractions.append(hint.get('importanceFraction', 0))
                    
                    # Logo Detection
                    if 'logoAnnotations' in metadata:
                        num_logos += len(metadata['logoAnnotations'])
                    
                    # Text Detection
                    if 'textAnnotations' in metadata:
                        num_texts += len(metadata['textAnnotations'])
                    
                    # Landmark Detection
                    if 'landmarkAnnotations' in metadata:
                        num_landmarks += len(metadata['landmarkAnnotations'])
                    
                except:
                    continue
                    
        except:
            num_files = 0
        
        # Categorías comunes de labels (ajustar según análisis exploratorio)
        animal_labels = ['dog', 'cat', 'puppy', 'kitten', 'pet', 'animal', 'mammal']
        outdoor_labels = ['outdoor', 'grass', 'garden', 'park', 'nature']
        indoor_labels = ['indoor', 'floor', 'room', 'furniture', 'carpet']
        
        has_animal_label = any(label in label_descriptions for label in animal_labels)
        has_outdoor_label = any(label in label_descriptions for label in outdoor_labels)
        has_indoor_label = any(label in label_descriptions for label in indoor_labels)
        
        # Calcular features agregadas
        metadata_features.append({
            'PetID': pet_id,
            
            # General metadata
            'num_metadata_files': num_files,
            'has_metadata': int(num_files > 0),
            
            # Face/Object detection
            'vertex_x_mean': np.mean(vertex_x_all) if vertex_x_all else 0,
            'vertex_x_std': np.std(vertex_x_all) if vertex_x_all else 0,
            'vertex_y_mean': np.mean(vertex_y_all) if vertex_y_all else 0,
            'vertex_y_std': np.std(vertex_y_all) if vertex_y_all else 0,
            'bounding_confidence_mean': np.mean(bounding_confidence_all) if bounding_confidence_all else 0,
            'bounding_confidence_max': np.max(bounding_confidence_all) if bounding_confidence_all else 0,
            'bounding_confidence_std': np.std(bounding_confidence_all) if bounding_confidence_all else 0,
            'num_faces_detected': len(bounding_confidence_all),
            'has_face': int(len(bounding_confidence_all) > 0),
            
            # Face emotion features
            'face_joy_mean': np.mean(face_joy_likelihood) if face_joy_likelihood else 0,
            'face_sorrow_mean': np.mean(face_sorrow_likelihood) if face_sorrow_likelihood else 0,
            'face_anger_mean': np.mean(face_anger_likelihood) if face_anger_likelihood else 0,
            'face_surprise_mean': np.mean(face_surprise_likelihood) if face_surprise_likelihood else 0,
            'face_blur_mean': np.mean(face_blur_likelihood) if face_blur_likelihood else 0,
            'face_headwear_mean': np.mean(face_headwear_likelihood) if face_headwear_likelihood else 0,
            
            # Color features
            'dominant_red_mean': np.mean(dominant_colors_rgb['red']) if dominant_colors_rgb['red'] else 0,
            'dominant_green_mean': np.mean(dominant_colors_rgb['green']) if dominant_colors_rgb['green'] else 0,
            'dominant_blue_mean': np.mean(dominant_colors_rgb['blue']) if dominant_colors_rgb['blue'] else 0,
            'dominant_red_std': np.std(dominant_colors_rgb['red']) if dominant_colors_rgb['red'] else 0,
            'dominant_green_std': np.std(dominant_colors_rgb['green']) if dominant_colors_rgb['green'] else 0,
            'dominant_blue_std': np.std(dominant_colors_rgb['blue']) if dominant_colors_rgb['blue'] else 0,
            'color_brightness': np.mean([np.mean(dominant_colors_rgb['red']), 
                                        np.mean(dominant_colors_rgb['green']), 
                                        np.mean(dominant_colors_rgb['blue'])]) if dominant_colors_rgb['red'] else 0,
            'dominant_score_mean': np.mean(dominant_scores) if dominant_scores else 0,
            'dominant_score_max': np.max(dominant_scores) if dominant_scores else 0,
            'dominant_score_std': np.std(dominant_scores) if dominant_scores else 0,
            'dominant_pixel_fraction_mean': np.mean(dominant_pixel_fractions) if dominant_pixel_fractions else 0,
            'dominant_pixel_fraction_max': np.max(dominant_pixel_fractions) if dominant_pixel_fractions else 0,
            'num_dominant_colors': len(dominant_scores),
            'color_diversity': np.std(dominant_scores) if len(dominant_scores) > 1 else 0,
            
            # Label features
            'label_score_mean': np.mean(label_scores) if label_scores else 0,
            'label_score_std': np.std(label_scores) if label_scores else 0,
            'label_score_max': np.max(label_scores) if label_scores else 0,
            'label_score_min': np.min(label_scores) if label_scores else 0,
            'num_labels': len(label_scores),
            'has_animal_label': int(has_animal_label),
            'has_outdoor_label': int(has_outdoor_label),
            'has_indoor_label': int(has_indoor_label),
            
            # Crop hints
            'crop_confidence_mean': np.mean(crop_confidences) if crop_confidences else 0,
            'crop_confidence_max': np.max(crop_confidences) if crop_confidences else 0,
            'crop_importance_mean': np.mean(crop_importance_fractions) if crop_importance_fractions else 0,
            'num_crop_hints': len(crop_confidences),
            
            # Other detections
            'num_logos': num_logos,
            'num_texts': num_texts,
            'num_landmarks': num_landmarks,
            'has_text': int(num_texts > 0),
            
            # Composite features
            'image_quality_score': (
                np.mean(label_scores) if label_scores else 0 * 0.3 +
                (1 - np.mean(face_blur_likelihood) / 4 if face_blur_likelihood else 0) * 0.3 +
                np.mean(crop_confidences) if crop_confidences else 0 * 0.4
            ),
            'metadata_richness': num_files + len(label_scores) + len(dominant_scores) + len(crop_confidences),
        })
    
    return pd.DataFrame(metadata_features)

# Extraer metadata features
train_metadata = extract_metadata_features(train_df, 'train')
test_metadata = extract_metadata_features(test_df, 'test')

# Guardar outputs
print("\nGuardando archivos de salida...")
train_metadata.to_csv('train_image_features.csv', index=False)
test_metadata.to_csv('test_image_features.csv', index=False)

print("\n✓ Archivos creados:")
print("  - train_image_features.csv")
print("  - test_image_features.csv")

print(f"\nShape train: {train_metadata.shape}")
print(f"Shape test: {test_metadata.shape}")
print(f"\nNúmero de features generadas: {train_metadata.shape[1] - 1}")

# Estadísticas de cobertura
print("\n" + "="*60)
print("ESTADÍSTICAS DE METADATA")
print("="*60)
print(f"Train - Porcentaje con metadata: {train_metadata['has_metadata'].mean()*100:.2f}%")
print(f"Test - Porcentaje con metadata: {test_metadata['has_metadata'].mean()*100:.2f}%")
print(f"Train - Promedio archivos por PetID: {train_metadata['num_metadata_files'].mean():.2f}")
print(f"Test - Promedio archivos por PetID: {test_metadata['num_metadata_files'].mean():.2f}")
print(f"Train - Porcentaje con rostros detectados: {train_metadata['has_face'].mean()*100:.2f}%")
print(f"Train - Promedio labels por imagen: {train_metadata['num_labels'].mean():.2f}")
print(f"Train - Promedio colores dominantes: {train_metadata['num_dominant_colors'].mean():.2f}")

print("\n" + "="*60)
print("PREPROCESAMIENTO DE IMÁGENES COMPLETADO")
print("="*60)

