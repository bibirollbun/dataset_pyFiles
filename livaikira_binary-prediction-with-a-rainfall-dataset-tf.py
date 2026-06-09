# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

import os
import warnings
warnings.filterwarnings('ignore')

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")


# Load data
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

# Display sample data
print(f"Train shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")
train_data.head()


# Check for missing values
print("Missing values in train data:\n", train_data.isnull().sum())
print("\nMissing values in test data:\n", test_data.isnull().sum())


# Check target distribution
print("Target distribution:")
print(train_data['rainfall'].value_counts())
print(f"Percentage of positive cases: {train_data['rainfall'].mean() * 100:.2f}%")

# Visualize target distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='rainfall', data=train_data)
plt.title('Target Distribution')
plt.show()


# Separate features and target
X = train_data.drop(['id', 'rainfall'], axis=1)
y = train_data['rainfall']
test_features = test_data.drop(['id'], axis=1)

# Feature names for later use
feature_names = X.columns.tolist()

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test_features)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Validation set shape: {X_val_scaled.shape}")
print(f"Test set shape: {test_scaled.shape}")


# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Build a simple dense neural network
def build_basic_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(512, activation='relu'),
        # layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model

# Create callbacks
early_stopping = EarlyStopping(monitor='val_auc', patience=30, mode='max', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=0.00001, verbose=1)

# Build and train the model
basic_model = build_basic_model((X_train_scaled.shape[1],))
basic_model.summary()

history = basic_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate on validation set
val_preds = basic_model.predict(X_val_scaled)
val_auc = roc_auc_score(y_val, val_preds)
print(f"Validation AUC: {val_auc:.4f}")


# Build a residual block
def residual_block(x, units, dropout_rate=0.2):
    residual = x
    
    y = layers.Dense(units)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Dropout(dropout_rate)(y)
    
    y = layers.Dense(units)(y)
    y = layers.BatchNormalization()(y)
    
    # If input shape doesn't match output shape
    if residual.shape[-1] != units:
        residual = layers.Dense(units)(residual)
    
    output = layers.Add()([y, residual])
    output = layers.Activation('relu')(output)
    
    return output

# Build a more complex residual network
def build_resnet_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Dense(512)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Add residual blocks
    x = residual_block(x, 512)
    x = residual_block(x, 256)
    x = residual_block(x, 128)
    x = residual_block(x, 64)
    x = residual_block(x, 32)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model

# Build and train the model
resnet_model = build_resnet_model((X_train_scaled.shape[1],))
resnet_model.summary()

# Create a model checkpoint callback
checkpoint = ModelCheckpoint(
    'best_resnet_model.keras',
    monitor='val_auc',
    mode='max',
    save_best_only=True,
    verbose=1
)

res_history = resnet_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)


# Load the best model
best_resnet_model = keras.models.load_model('best_resnet_model.keras')

# Evaluate on validation set
val_resnet_preds = best_resnet_model.predict(X_val_scaled)
val_resnet_auc = roc_auc_score(y_val, val_resnet_preds)
print(f"ResNet Validation AUC: {val_resnet_auc:.4f}")

# Compare with basic model
print(f"Basic Model Validation AUC: {val_auc:.4f}")


# Implement SELU activation for self-normalization
def build_snn_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(512, activation='selu', kernel_initializer='lecun_normal'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='selu', kernel_initializer='lecun_normal'),
        # layers.Dropout(0.2),
        layers.Dense(128, activation='selu', kernel_initializer='lecun_normal'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='selu', kernel_initializer='lecun_normal'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='selu', kernel_initializer='lecun_normal'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model

# Build and train the SNN model
snn_model = build_snn_model((X_train_scaled.shape[1],))
snn_model.summary()

# Create a model checkpoint callback
snn_checkpoint = ModelCheckpoint(
    'best_snn_model.keras',
    monitor='val_auc',
    mode='max',
    save_best_only=True,
    verbose=1
)

snn_history = snn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr, snn_checkpoint],
    verbose=1
)


# Load the best SNN model
best_snn_model = keras.models.load_model('best_snn_model.keras')

# Evaluate on validation set
val_snn_preds = best_snn_model.predict(X_val_scaled)
val_snn_auc = roc_auc_score(y_val, val_snn_preds)
print(f"SNN Validation AUC: {val_snn_auc:.4f}")


# Create an ensemble of models
# We'll use weighted averaging since models may have different performance levels

# Determine weights based on validation AUC
total_auc = val_auc + val_resnet_auc #+ val_snn_auc
basic_weight = val_auc / total_auc
resnet_weight = val_resnet_auc / total_auc
snn_weight = val_snn_auc / total_auc

print(f"Ensemble weights:")
print(f"- Basic model: {basic_weight:.4f}")
print(f"- ResNet model: {resnet_weight:.4f}")
print(f"- SNN model: {snn_weight:.4f}")

# Make validation predictions with all models
ensemble_val_preds = (
    basic_weight * val_preds + 
    resnet_weight * val_resnet_preds + 
    snn_weight * val_snn_preds
)

# Evaluate ensemble on validation set
ensemble_val_auc = roc_auc_score(y_val, ensemble_val_preds)
print(f"Ensemble Validation AUC: {ensemble_val_auc:.4f}")

# Print model comparison
print("\nModel performance comparison:")
print(f"- Basic model AUC: {val_auc:.4f}")
print(f"- ResNet model AUC: {val_resnet_auc:.4f}")
print(f"- SNN model AUC: {val_snn_auc:.4f}")
print(f"- Ensemble model AUC: {ensemble_val_auc:.4f}")


# Generate predictions from each model
basic_preds = basic_model.predict(test_scaled)
resnet_preds = best_resnet_model.predict(test_scaled)
snn_preds = best_snn_model.predict(test_scaled)

# Create ensemble predictions
ensemble_preds = (
    basic_weight * basic_preds + 
    resnet_weight * resnet_preds + 
    snn_weight * snn_preds
)

ensemble_preds = (ensemble_preds >= 0.5).astype(int)

# Prepare submission
submission = sample_submission.copy()
submission['rainfall'] = ensemble_preds.flatten()

# Save submission file
submission.to_csv('submission.csv', index=False)
submission.head()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin

# Define an ensemble model wrapper
class EnsembleModel(BaseEstimator, RegressorMixin):
    def __init__(self, basic_model, resnet_model, snn_model, basic_weight, resnet_weight, snn_weight):
        self.basic_model = basic_model
        self.resnet_model = resnet_model
        self.snn_model = snn_model
        self.basic_weight = basic_weight
        self.resnet_weight = resnet_weight
        self.snn_weight = snn_weight

    def fit(self, X, y):
        pass  # No training needed, just a wrapper for predictions

    def predict(self, X):
        basic_pred = self.basic_model.predict(X).flatten()
        resnet_pred = self.resnet_model.predict(X).flatten()
        snn_pred = self.snn_model.predict(X).flatten()
        return (
            self.basic_weight * basic_pred +
            self.resnet_weight * resnet_pred +
            self.snn_weight * snn_pred
        )

# Instantiate the ensemble model
ensemble_model = EnsembleModel(
    basic_model=basic_model,
    resnet_model=best_resnet_model,
    snn_model=best_snn_model,
    basic_weight=basic_weight,
    resnet_weight=resnet_weight,
    snn_weight=snn_weight
)

# Calculate permutation importance
result = permutation_importance(
    ensemble_model, X_val_scaled, y_val,
    n_repeats=10,
    random_state=42,
    scoring='roc_auc'
)

# Create a DataFrame with feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': result.importances_mean,
    'Std': result.importances_std
})

# Sort features by importance
importance_df = importance_df.sort_values('Importance', ascending=False)

# Plot top features
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), xerr=importance_df['Std'].head(15))
plt.title('Top 15 Features by Importance')
plt.tight_layout()
plt.show()

importance_df.head(15)



# Load a pre-trained model (e.g., a smaller version of ResNet or a VGG model)
# For demonstration, let's assume we adapt the basic model

# Load pre-trained weights (replace 'path_to_pretrained_weights' with the actual path)
# basic_model.load_weights('path_to_pretrained_weights')

# Option 1: Freeze some layers and fine-tune the rest
# for layer in basic_model.layers[:-3]:  # Freeze all layers except the last 3
#     layer.trainable = False

# Option 2: Train all layers with a very small learning rate
# new_learning_rate = 0.00001
# basic_model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=new_learning_rate),
#     loss='binary_crossentropy',
#     metrics=['accuracy', keras.metrics.AUC(name='auc')]
# )

# Fine-tune the model
# fine_tune_history = basic_model.fit(
#     X_train_scaled, y_train,
#     validation_data=(X_val_scaled, y_val),
#     epochs=20,  # Shorter training
#     batch_size=32,
#     callbacks=[early_stopping, reduce_lr],
#     verbose=1
# )

# Evaluate the fine-tuned model
# val_fine_tune_preds = basic_model.predict(X_val_scaled)
# val_fine_tune_auc = roc_auc_score(y_val, val_fine_tune_preds)
# print(f"Fine-tuned Validation AUC: {val_fine_tune_auc:.4f}")

# Note: The above code is commented out as it requires a pre-trained model and its weights.
# Uncomment and adapt the code when you have a pre-trained model available.

