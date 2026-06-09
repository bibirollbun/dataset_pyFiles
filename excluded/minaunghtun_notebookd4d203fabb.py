"""
KKBOX Music Recommendation Challenge - Two Tower Architecture
Competition: Predict if a user will listen to a song repetitively after first listen
Evaluation: ROC AUC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# For neural network model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)

# ============================================================================
# 1. DATA LOADING & EXTRACTION
# ============================================================================
print("\n" + "="*80)
print("STEP 1: DATA LOADING & EXTRACTION")
print("="*80)

# Path to compressed files in Kaggle input
input_path = '/kaggle/input/kkbox-music-recommendation-challenge/'
files_to_extract = ['train.csv', 'test.csv', 'songs.csv', 'members.csv']

# Extract .7z files into the current working directory (/kaggle/working)
for file in files_to_extract:
    if not os.path.exists(file):
        print(f"Extracting {file}.7z...")
        # Using the system 7z utility available in Kaggle kernels
        os.system(f'7z x {input_path}{file}.7z')

# Load the extracted datasets from the working directory
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
songs_df = pd.read_csv('songs.csv')
members_df = pd.read_csv('members.csv')

print(f"\nTrain shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Songs shape: {songs_df.shape}")
print(f"Members shape: {members_df.shape}")

print("\nTrain data sample:")
print(train_df.head(3))

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Summary statistics
print("\nTrain dataset summary:")
print(train_df.describe())

print("\nTarget distribution:")
print(train_df['target'].value_counts())
print(f"Target mean (positive rate): {train_df['target'].mean():.4f}")

# Missing values check
print("\nMissing values in train:")
print(train_df.isnull().sum())

print("\nMissing values in songs:")
print(songs_df.isnull().sum())

print("\nMissing values in members:")
print(members_df.isnull().sum())

# Data distribution visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Target distribution
axes[0, 0].bar(['Not Repeated (0)', 'Repeated (1)'], 
               train_df['target'].value_counts().values)
axes[0, 0].set_title('Target Distribution')
axes[0, 0].set_ylabel('Count')

# Source system tab distribution
train_df['source_system_tab'].value_counts().head(10).plot(kind='barh', ax=axes[0, 1])
axes[0, 1].set_title('Top 10 Source System Tabs')
axes[0, 1].set_xlabel('Count')

# Source type distribution
train_df['source_type'].value_counts().head(10).plot(kind='barh', ax=axes[1, 0])
axes[1, 0].set_title('Top 10 Source Types')
axes[1, 0].set_xlabel('Count')

# Age distribution (after cleaning outliers)
members_clean = members_df[(members_df['bd'] > 0) & (members_df['bd'] < 100)]
axes[1, 1].hist(members_clean['bd'], bins=30, edgecolor='black')
axes[1, 1].set_title('Age Distribution (Cleaned)')
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nEDA visualizations saved.")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("STEP 3: FEATURE ENGINEERING")
print("="*80)

# Merge train/test with songs and members data
print("\nMerging datasets...")
train_merged = train_df.merge(songs_df, on='song_id', how='left')
train_merged = train_merged.merge(members_df, on='msno', how='left')

test_merged = test_df.merge(songs_df, on='song_id', how='left')
test_merged = test_merged.merge(members_df, on='msno', how='left')

print(f"Train merged shape: {train_merged.shape}")
print(f"Test merged shape: {test_merged.shape}")

# ---- ID Encoding (LabelEncoder for user/song IDs) ----
print("\nEncoding user and song IDs...")

# User ID encoding
user_encoder = LabelEncoder()
all_users = pd.concat([train_merged['msno'], test_merged['msno']]).unique()
user_encoder.fit(all_users)

train_merged['user_id_encoded'] = user_encoder.transform(train_merged['msno'])
test_merged['user_id_encoded'] = user_encoder.transform(test_merged['msno'])

# Song ID encoding
song_encoder = LabelEncoder()
all_songs = pd.concat([train_merged['song_id'], test_merged['song_id']]).unique()
song_encoder.fit(all_songs)

train_merged['song_id_encoded'] = song_encoder.transform(train_merged['song_id'])
test_merged['song_id_encoded'] = song_encoder.transform(test_merged['song_id'])

n_users = len(user_encoder.classes_)
n_songs = len(song_encoder.classes_)
print(f"Number of unique users: {n_users}")
print(f"Number of unique songs: {n_songs}")

# ---- Metadata Features ----
print("\nProcessing metadata features...")

# Genre: take first genre if multiple exist
train_merged['genre_first'] = train_merged['genre_ids'].fillna('unknown').astype(str).apply(
    lambda x: x.split('|')[0] if x != 'nan' else 'unknown'
)
test_merged['genre_first'] = test_merged['genre_ids'].fillna('unknown').astype(str).apply(
    lambda x: x.split('|')[0] if x != 'nan' else 'unknown'
)

# Artist encoding
artist_encoder = LabelEncoder()
all_artists = pd.concat([
    train_merged['artist_name'].fillna('unknown'),
    test_merged['artist_name'].fillna('unknown')
]).unique()
artist_encoder.fit(all_artists)

train_merged['artist_encoded'] = artist_encoder.transform(train_merged['artist_name'].fillna('unknown'))
test_merged['artist_encoded'] = artist_encoder.transform(test_merged['artist_name'].fillna('unknown'))

# Language encoding
language_encoder = LabelEncoder()
all_languages = pd.concat([
    train_merged['language'].fillna(-1).astype(int).astype(str),
    test_merged['language'].fillna(-1).astype(int).astype(str)
]).unique()
language_encoder.fit(all_languages)

train_merged['language_encoded'] = language_encoder.transform(train_merged['language'].fillna(-1).astype(int).astype(str))
test_merged['language_encoded'] = language_encoder.transform(test_merged['language'].fillna(-1).astype(int).astype(str))

# ---- Temporal Features ----
print("\nExtracting temporal features...")

# Registration date features
train_merged['registration_init_time'] = train_merged['registration_init_time'].fillna(0).astype(int)
test_merged['registration_init_time'] = test_merged['registration_init_time'].fillna(0).astype(int)

train_merged['reg_year'] = train_merged['registration_init_time'].apply(
    lambda x: int(str(x)[:4]) if x > 0 else 2000
)
train_merged['reg_month'] = train_merged['registration_init_time'].apply(
    lambda x: int(str(x)[4:6]) if x > 0 and len(str(x)) >= 6 else 1
)

test_merged['reg_year'] = test_merged['registration_init_time'].apply(
    lambda x: int(str(x)[:4]) if x > 0 else 2000
)
test_merged['reg_month'] = test_merged['registration_init_time'].apply(
    lambda x: int(str(x)[4:6]) if x > 0 and len(str(x)) >= 6 else 1
)

# ---- Interaction Features ----
print("\nCreating interaction features...")

# Song length normalized (in seconds)
train_merged['song_length_sec'] = train_merged['song_length'].fillna(0) / 1000
test_merged['song_length_sec'] = test_merged['song_length'].fillna(0) / 1000

# Age cleaning and normalization
train_merged['age_clean'] = train_merged['bd'].apply(
    lambda x: x if (x > 0 and x < 100) else 25  # Default to median age
)
test_merged['age_clean'] = test_merged['bd'].apply(
    lambda x: x if (x > 0 and x < 100) else 25
)

# Gender encoding (fill missing with 'unknown')
train_merged['gender'] = train_merged['gender'].fillna('unknown')
test_merged['gender'] = test_merged['gender'].fillna('unknown')

gender_encoder = LabelEncoder()
gender_encoder.fit(['male', 'female', 'unknown'])
train_merged['gender_encoded'] = gender_encoder.transform(train_merged['gender'])
test_merged['gender_encoded'] = gender_encoder.transform(test_merged['gender'])

# City encoding
city_encoder = LabelEncoder()
all_cities = pd.concat([
    train_merged['city'].fillna(0).astype(int).astype(str),
    test_merged['city'].fillna(0).astype(int).astype(str)
]).unique()
city_encoder.fit(all_cities)

train_merged['city_encoded'] = city_encoder.transform(train_merged['city'].fillna(0).astype(int).astype(str))
test_merged['city_encoded'] = city_encoder.transform(test_merged['city'].fillna(0).astype(int).astype(str))

# Source features encoding
source_tab_encoder = LabelEncoder()
all_tabs = pd.concat([train_merged['source_system_tab'], test_merged['source_system_tab']]).fillna('unknown').unique()
source_tab_encoder.fit(all_tabs)

train_merged['source_tab_encoded'] = source_tab_encoder.transform(train_merged['source_system_tab'].fillna('unknown'))
test_merged['source_tab_encoded'] = source_tab_encoder.transform(test_merged['source_system_tab'].fillna('unknown'))

print("\nFeature engineering completed.")
print(f"Final train shape: {train_merged.shape}")
print(f"Final test shape: {test_merged.shape}")

# ============================================================================
# 4. MODEL ARCHITECTURE - TWO TOWER DESIGN
# ============================================================================
print("\n" + "="*80)
print("STEP 4: MODEL ARCHITECTURE - TWO TOWER DESIGN")
print("="*80)

def build_two_tower_model(n_users, n_songs, embedding_dim=32):
    # ---- USER TOWER ----
    user_id_input = layers.Input(shape=(1,), name='user_id')
    user_age_input = layers.Input(shape=(1,), name='user_age')
    user_gender_input = layers.Input(shape=(1,), name='user_gender')
    user_city_input = layers.Input(shape=(1,), name='user_city')
    user_reg_year_input = layers.Input(shape=(1,), name='user_reg_year')
    
    user_embedding = layers.Embedding(n_users, embedding_dim, name='user_embedding')(user_id_input)
    user_embedding = layers.Flatten()(user_embedding)
    
    user_features = layers.Concatenate()([
        layers.Flatten()(user_age_input),
        layers.Flatten()(user_gender_input),
        layers.Flatten()(user_city_input),
        layers.Flatten()(user_reg_year_input)
    ])
    
    user_tower = layers.Concatenate()([user_embedding, user_features])
    user_tower = layers.Dense(64, activation='relu', name='user_dense1')(user_tower)
    user_tower = layers.BatchNormalization()(user_tower)
    user_tower = layers.Dropout(0.3)(user_tower)
    user_tower = layers.Dense(32, activation='relu', name='user_dense2')(user_tower)
    
    # ---- ITEM (SONG) TOWER ----
    song_id_input = layers.Input(shape=(1,), name='song_id')
    song_length_input = layers.Input(shape=(1,), name='song_length')
    song_language_input = layers.Input(shape=(1,), name='song_language')
    song_artist_input = layers.Input(shape=(1,), name='song_artist')
    source_tab_input = layers.Input(shape=(1,), name='source_tab')
    
    song_embedding = layers.Embedding(n_songs, embedding_dim, name='song_embedding')(song_id_input)
    song_embedding = layers.Flatten()(song_embedding)
    
    song_features = layers.Concatenate()([
        layers.Flatten()(song_length_input),
        layers.Flatten()(song_language_input),
        layers.Flatten()(song_artist_input),
        layers.Flatten()(source_tab_input)
    ])
    
    item_tower = layers.Concatenate()([song_embedding, song_features])
    item_tower = layers.Dense(64, activation='relu', name='item_dense1')(item_tower)
    item_tower = layers.BatchNormalization()(item_tower)
    item_tower = layers.Dropout(0.3)(item_tower)
    item_tower = layers.Dense(32, activation='relu', name='item_dense2')(item_tower)
    
    # ---- INTERACTION MECHANISM ----
    interaction = layers.Dot(axes=1, normalize=False)([user_tower, item_tower])
    
    combined = layers.Concatenate()([user_tower, item_tower, interaction])
    combined = layers.Dense(32, activation='relu')(combined)
    combined = layers.Dropout(0.2)(combined)
    output = layers.Dense(1, activation='sigmoid', name='output')(combined)
    
    model = Model(
        inputs=[
            user_id_input, user_age_input, user_gender_input, user_city_input, user_reg_year_input,
            song_id_input, song_length_input, song_language_input, song_artist_input, source_tab_input
        ],
        outputs=output
    )
    return model

model = build_two_tower_model(n_users, n_songs, embedding_dim=32)
model.summary()

# ============================================================================
# 5. TRAINING
# ============================================================================
print("\n" + "="*80)
print("STEP 5: TRAINING")
print("="*80)

feature_columns = [
    'user_id_encoded', 'age_clean', 'gender_encoded', 'city_encoded', 'reg_year',
    'song_id_encoded', 'song_length_sec', 'language_encoded', 'artist_encoded', 'source_tab_encoded'
]

X_train_full = train_merged[feature_columns].values
y_train_full = train_merged['target'].values

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

def prepare_model_inputs(X):
    return {
        'user_id': X[:, 0], 'user_age': X[:, 1], 'user_gender': X[:, 2], 'user_city': X[:, 3],
        'user_reg_year': X[:, 4], 'song_id': X[:, 5], 'song_length': X[:, 6],
        'song_language': X[:, 7], 'song_artist': X[:, 8], 'source_tab': X[:, 9]
    }

train_inputs = prepare_model_inputs(X_train)
val_inputs = prepare_model_inputs(X_val)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[keras.metrics.AUC(name='auc')]
)

history = model.fit(
    train_inputs, y_train,
    validation_data=(val_inputs, y_val),
    epochs=10, batch_size=1024,
    callbacks=[EarlyStopping(monitor='val_auc', patience=3, restore_best_weights=True, mode='max'),
               ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, mode='max')],
    verbose=1
)

# Plotting History
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'], label='Train'); axes[0].plot(history.history['val_loss'], label='Val'); axes[0].set_title('Loss')
axes[1].plot(history.history['auc'], label='Train'); axes[1].plot(history.history['val_auc'], label='Val'); axes[1].set_title('AUC')
plt.show()

# ============================================================================
# 6. INFERENCE & SUBMISSION
# ============================================================================
print("\n" + "="*80)
print("STEP 6: INFERENCE & SUBMISSION")
print("="*80)

X_test = test_merged[feature_columns].values
test_inputs = prepare_model_inputs(X_test)

print("\nGenerating predictions on test set...")
predictions = model.predict(test_inputs, batch_size=2048, verbose=1).flatten()

submission = pd.DataFrame({'id': test_df['id'], 'target': predictions})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file saved as 'submission.csv'")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

