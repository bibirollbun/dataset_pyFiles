import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

# Chargement des données
train_df = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test_df = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')

train_df_clean = train_df.dropna(subset=['sii']).copy()

train_columns = set(train_df_clean.columns)
test_columns = set(test_df.columns)
columns_to_drop = list(train_columns - test_columns - {'sii'})
train_df_clean.drop(columns=columns_to_drop, inplace=True)
test_df.drop(columns=list(test_columns - train_columns), errors='ignore', inplace=True)

# Encodage des variables catégorielles
label_encoder = LabelEncoder()
categorical_columns = train_df_clean.select_dtypes(include=['object']).columns

for col in categorical_columns:
    train_df_clean[col] = train_df_clean[col].astype(str)
    test_df[col] = test_df[col].astype(str)
    train_df_clean[col] = label_encoder.fit_transform(train_df_clean[col])
    test_df[col] = test_df[col].map(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1)

# Remplissage des valeurs manquantes pour les variables numériques
numeric_cols = train_df_clean.select_dtypes(include=['float', 'int']).columns
numeric_cols = [col for col in numeric_cols if col != 'sii']

for col in numeric_cols:
    median_value = train_df_clean[col].median()
    train_df_clean[col] = train_df_clean[col].fillna(median_value)
    test_df[col] = test_df[col].fillna(median_value)

print(f"Taille des données après encodage : {train_df_clean.shape}")

# Séparation des features et de la target
X = train_df_clean.drop(columns=['id', 'sii'])
y = train_df_clean['sii']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Mise à l'échelle des features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_df.drop(columns=['id']))

num_classes = len(y.unique())
y_train_encoded = to_categorical(y_train, num_classes=num_classes)
y_val_encoded = to_categorical(y_val, num_classes=num_classes)

# Pour utiliser un CNN, on reshape nos données en ajoutant une dimension de canal
X_train_model = np.expand_dims(X_train_scaled, axis=2)  # shape: (n_samples, n_features, 1)
X_val_model   = np.expand_dims(X_val_scaled, axis=2)
X_test_model  = np.expand_dims(X_test_scaled, axis=2)

def create_cnn(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Première bloc de convolution
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # Deuxième bloc de convolution
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # Troisième bloc de convolution
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # Passage aux couches denses
    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

cnn_model = create_cnn(input_shape=X_train_model.shape[1:], num_classes=num_classes)
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Entraînement du modèle
history = cnn_model.fit(
    X_train_model, y_train_encoded,
    validation_data=(X_val_model, y_val_encoded),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Prédiction sur le jeu de test
test_pred_probs = cnn_model.predict(X_test_model)
test_pred = np.argmax(test_pred_probs, axis=1)

submission = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv")
submission['sii'] = test_pred
submission.to_csv('submission.csv', index=False)

print("✅ Fichier de soumission généré avec succès !")
print(submission)


