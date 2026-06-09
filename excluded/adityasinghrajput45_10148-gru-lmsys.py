!pip install --upgrade protobuf==3.20.3

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')


# Display TensorFlow and GPU information
print(f"TensorFlow Version: {tf.__version__}")
available_gpus = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {available_gpus}")


# Load training data with folds
df_train = pd.read_csv('/kaggle/input/10148-nfold-lmsys/train_folds.csv')
print(f"Dataset Shape: {df_train.shape}")
print(f"Available Columns: {df_train.columns.tolist()}")


# Combine text features with separator tokens
df_train['text_combined'] = (df_train['prompt'] + ' [SEP] ' + 
                               df_train['response_a'] + ' [SEP] ' + 
                               df_train['response_b'])


# Configure tokenization parameters
VOCAB_SIZE = 20000
SEQUENCE_LENGTH = 512


# Initialize and fit tokenizer
text_tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
text_tokenizer.fit_on_texts(df_train['text_combined'])


# Save tokenizer for later use
tokenizer_save_path = '/kaggle/working/tokenizer_gru.pkl'
with open(tokenizer_save_path, 'wb') as file:
    pickle.dump(text_tokenizer, file)

print(f"Total Vocabulary Size: {len(text_tokenizer.word_index)}")
print(f"Using Top Words: {VOCAB_SIZE}")


# Define BiGRU model architecture
def build_bigru_model(vocab_size, embed_dim=128, seq_length=512):
    """
    Builds a Bidirectional GRU model for text classification
    """
    input_layer = layers.Input(shape=(seq_length,))
    
    # Embedding layer with masking
    embedding = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(input_layer)
    
    # First bidirectional GRU layer
    bigru_1 = layers.Bidirectional(layers.GRU(128, return_sequences=True))(embedding)
    dropout_1 = layers.Dropout(0.3)(bigru_1)
    
    # Second bidirectional GRU layer
    bigru_2 = layers.Bidirectional(layers.GRU(64))(dropout_1)
    dropout_2 = layers.Dropout(0.3)(bigru_2)
    
    # Dense layers
    dense_1 = layers.Dense(64, activation='relu')(dropout_2)
    dropout_3 = layers.Dropout(0.2)(dense_1)
    
    # Output layer for 3-class classification
    output_layer = layers.Dense(3, activation='softmax')(dropout_3)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model



# Training configuration
NUM_EPOCHS = 1
BATCH_SIZE = 32
N_FOLDS = 5



# Storage for results
validation_scores = []
trained_models = []


# Cross-validation training loop
for fold_idx in range(N_FOLDS):
    print("\n" + "=" * 70)
    print(f"FOLD {fold_idx} TRAINING")
    print("=" * 70)
    
    # Split data into train and validation
    train_mask = df_train['fold'] != fold_idx
    val_mask = df_train['fold'] == fold_idx
    
    df_train_fold = df_train[train_mask].copy()
    df_val_fold = df_train[val_mask].copy()
    
    print(f"Training samples: {len(df_train_fold)} | Validation samples: {len(df_val_fold)}")
    
    # Tokenize and pad sequences
    train_sequences = text_tokenizer.texts_to_sequences(df_train_fold['text_combined'])
    val_sequences = text_tokenizer.texts_to_sequences(df_val_fold['text_combined'])
    
    X_train_padded = pad_sequences(train_sequences, maxlen=SEQUENCE_LENGTH, 
                                   padding='post', truncating='post')
    X_val_padded = pad_sequences(val_sequences, maxlen=SEQUENCE_LENGTH, 
                                 padding='post', truncating='post')
    
    # Prepare target labels
    y_train_labels = df_train_fold[['winner_model_a', 'winner_model_b', 'winner_tie']].values
    y_val_labels = df_val_fold[['winner_model_a', 'winner_model_b', 'winner_tie']].values
    
    print(f"Input shape: {X_train_padded.shape}")
    print(f"Target shape: {y_train_labels.shape}")
    
    # Build model
    model = build_bigru_model(vocab_size=VOCAB_SIZE, embed_dim=128, seq_length=SEQUENCE_LENGTH)
    
    # Compile with optimizer and loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Define callbacks
    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    training_history = model.fit(
        X_train_padded, y_train_labels,
        validation_data=(X_val_padded, y_val_labels),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop_callback, lr_scheduler],
        verbose=1
    )
    
    # Generate predictions and calculate validation score
    predictions_val = model.predict(X_val_padded, batch_size=BATCH_SIZE, verbose=0)
    logloss_score = log_loss(y_val_labels, predictions_val)
    
    print(f"Fold {fold_idx} | Validation Log Loss: {logloss_score}")
    validation_scores.append(logloss_score)
    
    # Save model
    model_save_path = f'/kaggle/working/gru_model_fold_{fold_idx}.h5'
    model.save(model_save_path)
    print(f"Saved model: {model_save_path}")
    
    trained_models.append(model)
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(training_history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(training_history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title(f'Fold {fold_idx} - Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(training_history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax2.plot(training_history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_title(f'Fold {fold_idx} - Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'/kaggle/working/gru_fold_{fold_idx}_history.png', dpi=100, bbox_inches='tight')
    plt.show()


# Display cross-validation summary
print("\n" + "=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)

for idx, score in enumerate(validation_scores):
    print(f"Fold {idx}: {score:.6f}")

avg_score = np.mean(validation_scores)
std_score = np.std(validation_scores)

print(f"\nAverage CV Log Loss: {avg_score:.6f}")
print(f"Standard Deviation: {std_score:.6f}")


# Save results to CSV
cv_results = pd.DataFrame({
    'fold': list(range(N_FOLDS)),
    'log_loss': validation_scores
})
cv_results.to_csv('/kaggle/working/gru_cv_results.csv', index=False)

print("\n" + "=" * 70)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)

