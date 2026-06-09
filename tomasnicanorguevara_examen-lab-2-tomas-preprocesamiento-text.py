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
print("NOTEBOOK 1: PREPROCESAMIENTO DE TEXTO Y SENTIMENT (MEJORADO)")
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

def extract_sentiment_features(df, mode='train'):
    """Extrae características avanzadas de sentiment análisis"""
    sentiment_path = f'/kaggle/input/petfinder-adoption-prediction/{mode}_sentiment/'
    
    sentiment_features = []
    
    print(f"\nExtrayendo sentiment features de {mode}...")
    for pet_id in tqdm(df['PetID']):
        try:
            with open(f'{sentiment_path}{pet_id}.json', 'r') as f:
                sentiment_data = json.load(f)
                
            # Document sentiment
            if 'documentSentiment' in sentiment_data:
                magnitude = sentiment_data['documentSentiment'].get('magnitude', 0)
                score = sentiment_data['documentSentiment'].get('score', 0)
            else:
                magnitude = 0
                score = 0
                
            # Language
            language = sentiment_data.get('language', 'unknown')
            language_is_english = int(language == 'en')
            
            # Sentences statistics
            num_sentences = len(sentiment_data.get('sentences', []))
            sentence_magnitudes = []
            sentence_scores = []
            
            for sentence in sentiment_data.get('sentences', []):
                if 'sentiment' in sentence:
                    sentence_magnitudes.append(sentence['sentiment'].get('magnitude', 0))
                    sentence_scores.append(sentence['sentiment'].get('score', 0))
            
            # Entities statistics
            entities = sentiment_data.get('entities', [])
            num_entities = len(entities)
            entity_types = []
            entity_saliences = []
            entity_sentiments = []
            
            for entity in entities:
                entity_types.append(entity.get('type', 'UNKNOWN'))
                entity_saliences.append(entity.get('salience', 0))
                if 'sentiment' in entity:
                    entity_sentiments.append(entity['sentiment'].get('score', 0))
            
            # Entity type counts
            num_person = entity_types.count('PERSON')
            num_location = entity_types.count('LOCATION')
            num_organization = entity_types.count('ORGANIZATION')
            num_event = entity_types.count('EVENT')
            num_work_of_art = entity_types.count('WORK_OF_ART')
            num_consumer_good = entity_types.count('CONSUMER_GOOD')
            num_other = entity_types.count('OTHER')
            
            # Sentiment complexity
            sentiment_variance = np.var(sentence_scores) if len(sentence_scores) > 1 else 0
            sentiment_range = (max(sentence_scores) - min(sentence_scores)) if sentence_scores else 0
            
            # Positive/negative sentences ratio
            positive_sentences = sum(1 for s in sentence_scores if s > 0.1)
            negative_sentences = sum(1 for s in sentence_scores if s < -0.1)
            neutral_sentences = num_sentences - positive_sentences - negative_sentences
            
            sentiment_features.append({
                'PetID': pet_id,
                # Document level
                'sentiment_magnitude': magnitude,
                'sentiment_score': score,
                'sentiment_language': language,
                'language_is_english': language_is_english,
                
                # Sentences
                'num_sentences': num_sentences,
                'sentence_magnitude_mean': np.mean(sentence_magnitudes) if sentence_magnitudes else 0,
                'sentence_magnitude_std': np.std(sentence_magnitudes) if sentence_magnitudes else 0,
                'sentence_magnitude_max': np.max(sentence_magnitudes) if sentence_magnitudes else 0,
                'sentence_magnitude_min': np.min(sentence_magnitudes) if sentence_magnitudes else 0,
                'sentence_score_mean': np.mean(sentence_scores) if sentence_scores else 0,
                'sentence_score_std': np.std(sentence_scores) if sentence_scores else 0,
                'sentence_score_max': np.max(sentence_scores) if sentence_scores else 0,
                'sentence_score_min': np.min(sentence_scores) if sentence_scores else 0,
                'sentiment_variance': sentiment_variance,
                'sentiment_range': sentiment_range,
                'positive_sentences': positive_sentences,
                'negative_sentences': negative_sentences,
                'neutral_sentences': neutral_sentences,
                'positive_ratio': positive_sentences / num_sentences if num_sentences > 0 else 0,
                'negative_ratio': negative_sentences / num_sentences if num_sentences > 0 else 0,
                
                # Entities
                'num_entities': num_entities,
                'entity_salience_mean': np.mean(entity_saliences) if entity_saliences else 0,
                'entity_salience_max': np.max(entity_saliences) if entity_saliences else 0,
                'entity_salience_sum': np.sum(entity_saliences) if entity_saliences else 0,
                'entity_sentiment_mean': np.mean(entity_sentiments) if entity_sentiments else 0,
                'num_person': num_person,
                'num_location': num_location,
                'num_organization': num_organization,
                'num_event': num_event,
                'num_work_of_art': num_work_of_art,
                'num_consumer_good': num_consumer_good,
                'num_other_entity': num_other,
                
                # Ratios
                'entities_per_sentence': num_entities / num_sentences if num_sentences > 0 else 0,
                'magnitude_per_sentence': magnitude / num_sentences if num_sentences > 0 else 0,
            })
        except Exception as e:
            sentiment_features.append({
                'PetID': pet_id,
                'sentiment_magnitude': 0, 'sentiment_score': 0, 'sentiment_language': 'unknown',
                'language_is_english': 0, 'num_sentences': 0, 'sentence_magnitude_mean': 0,
                'sentence_magnitude_std': 0, 'sentence_magnitude_max': 0, 'sentence_magnitude_min': 0,
                'sentence_score_mean': 0, 'sentence_score_std': 0, 'sentence_score_max': 0,
                'sentence_score_min': 0, 'sentiment_variance': 0, 'sentiment_range': 0,
                'positive_sentences': 0, 'negative_sentences': 0, 'neutral_sentences': 0,
                'positive_ratio': 0, 'negative_ratio': 0, 'num_entities': 0,
                'entity_salience_mean': 0, 'entity_salience_max': 0, 'entity_salience_sum': 0,
                'entity_sentiment_mean': 0, 'num_person': 0, 'num_location': 0,
                'num_organization': 0, 'num_event': 0, 'num_work_of_art': 0,
                'num_consumer_good': 0, 'num_other_entity': 0, 'entities_per_sentence': 0,
                'magnitude_per_sentence': 0
            })
    
    return pd.DataFrame(sentiment_features)

def create_text_features(df):
    """Crea características avanzadas basadas en texto"""
    df = df.copy()
    
    # Features de descripción
    df['Description'] = df['Description'].fillna('')
    df['Name'] = df['Name'].fillna('')
    
    # Longitud y palabras
    df['Description_length'] = df['Description'].apply(len)
    df['Description_words'] = df['Description'].apply(lambda x: len(x.split()))
    df['Description_unique_words'] = df['Description'].apply(
        lambda x: len(set(x.lower().split())) if x else 0
    )
    
    # Promedio longitud de palabra
    df['Description_avg_word_length'] = df['Description'].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if x else 0
    )
    
    # Conteo de caracteres especiales
    df['Description_exclamation'] = df['Description'].apply(lambda x: x.count('!'))
    df['Description_question'] = df['Description'].apply(lambda x: x.count('?'))
    df['Description_dots'] = df['Description'].apply(lambda x: x.count('.'))
    df['Description_commas'] = df['Description'].apply(lambda x: x.count(','))
    df['Description_uppercase'] = df['Description'].apply(lambda x: sum(1 for c in x if c.isupper()))
    df['Description_digits'] = df['Description'].apply(lambda x: sum(1 for c in x if c.isdigit()))
    
    # Ratios
    df['Description_uppercase_ratio'] = df['Description_uppercase'] / (df['Description_length'] + 1)
    df['Description_digit_ratio'] = df['Description_digits'] / (df['Description_length'] + 1)
    df['words_per_sentence'] = df['Description_words'] / np.maximum(df['Description_dots'] + 1, 1)
    df['unique_word_ratio'] = df['Description_unique_words'] / (df['Description_words'] + 1)
    
    # Features de nombre
    df['Name_length'] = df['Name'].apply(len)
    df['has_name'] = (df['Name'] != '').astype(int)
    df['Name_words'] = df['Name'].apply(lambda x: len(x.split()))
    df['Name_uppercase'] = df['Name'].apply(lambda x: sum(1 for c in x if c.isupper()))
    df['Name_is_capitalized'] = df['Name'].apply(lambda x: x[0].isupper() if x else 0).astype(int)
    
    # Palabras clave comunes en descripciones (ajusta según tu dataset)
    keywords = ['loving', 'friendly', 'playful', 'cute', 'adorable', 'sweet', 'good', 
                'healthy', 'active', 'calm', 'trained', 'vaccinated', 'adopt', 'home']
    
    for keyword in keywords:
        df[f'keyword_{keyword}'] = df['Description'].str.lower().str.contains(keyword).astype(int)
    
    df['keyword_count'] = sum(df[f'keyword_{keyword}'] for keyword in keywords)
    
    # Complexity score
    df['text_complexity'] = (
        df['Description_words'] * 0.3 +
        df['Description_unique_words'] * 0.3 +
        df['Description_avg_word_length'] * 10 +
        df['Description_dots'] * 2
    )
    
    # Engagement score (características que llaman la atención)
    df['engagement_score'] = (
        df['Description_exclamation'] * 2 +
        df['Description_question'] * 1.5 +
        df['keyword_count'] * 3 +
        df['has_name'] * 5
    )
    
    # Drop intermediate columns
    cols_to_keep = ['PetID', 'Description_length', 'Description_words', 'Description_unique_words',
                    'Description_avg_word_length', 'Description_exclamation', 'Description_question',
                    'Description_dots', 'Description_commas', 'Description_uppercase', 
                    'Description_digits', 'Description_uppercase_ratio', 'Description_digit_ratio',
                    'words_per_sentence', 'unique_word_ratio', 'Name_length', 'has_name',
                    'Name_words', 'Name_uppercase', 'Name_is_capitalized', 'keyword_count',
                    'text_complexity', 'engagement_score'] + [f'keyword_{k}' for k in keywords]
    
    return df[cols_to_keep]

# Extraer sentiment features
train_sentiment = extract_sentiment_features(train_df, 'train')
test_sentiment = extract_sentiment_features(test_df, 'test')

# Crear text features
print("\nCreando características de texto avanzadas...")
train_text = create_text_features(train_df)
test_text = create_text_features(test_df)

# Merge features
train_text_final = train_sentiment.merge(train_text, on='PetID', how='left')
test_text_final = test_sentiment.merge(test_text, on='PetID', how='left')

# Guardar outputs
print("\nGuardando archivos de salida...")
train_text_final.to_csv('train_text_features.csv', index=False)
test_text_final.to_csv('test_text_features.csv', index=False)

print("\n✓ Archivos creados:")
print("  - train_text_features.csv")
print("  - test_text_features.csv")

print(f"\nShape train: {train_text_final.shape}")
print(f"Shape test: {test_text_final.shape}")
print(f"\nNúmero de features generadas: {train_text_final.shape[1] - 1}")

# Estadísticas
print("\n" + "="*60)
print("ESTADÍSTICAS DE SENTIMENT")
print("="*60)
print(f"Train - Promedio sentiment score: {train_text_final['sentiment_score'].mean():.4f}")
print(f"Train - Promedio sentiment magnitude: {train_text_final['sentiment_magnitude'].mean():.4f}")
print(f"Train - Descripciones en inglés: {train_text_final['language_is_english'].mean()*100:.1f}%")
print(f"Train - Promedio palabras por descripción: {train_text_final['Description_words'].mean():.1f}")

print("\n" + "="*60)
print("PREPROCESAMIENTO DE TEXTO COMPLETADO")
print("="*60)

