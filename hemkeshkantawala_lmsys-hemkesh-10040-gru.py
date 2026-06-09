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


print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Variable 'train' renamed to 'df_train_folds'
df_train_folds = pd.read_csv('/kaggle/input/lmsys-hemkesh-10040-n-folds/train_folds.csv')

print("Train Shape:", df_train_folds.shape)
print("Columns:", df_train_folds.columns.tolist())


df_train_folds['text_concatenated'] = df_train_folds['prompt'] + ' [SEP] ' + df_train_folds['response_a'] + ' [SEP] ' + df_train_folds['response_b']


VOCAB_SIZE_LIMIT = 20000
SEQUENCE_LENGTH = 512

text_tokenizer = Tokenizer(num_words=VOCAB_SIZE_LIMIT, oov_token='<OOV>')

# Using previously renamed variables 'df_train_folds' and 'text_concatenated'
text_tokenizer.fit_on_texts(df_train_folds['text_concatenated'])

with open('/kaggle/working/tokenizer_gru.pkl', 'wb') as file_handler:
    pickle.dump(text_tokenizer, file_handler)

print("Vocabulary Size:", len(text_tokenizer.word_index))
print("Using top words:", VOCAB_SIZE_LIMIT)


def build_gru_architecture(total_vocabulary_size, embed_dimension=128, sequence_limit=512):
    input_layer = layers.Input(shape=(sequence_limit,))
    
    # Passing inputs through Embedding layer
    hidden_layer = layers.Embedding(total_vocabulary_size, embed_dimension, mask_zero=True)(input_layer)
    
    # First Bidirectional GRU Layer
    hidden_layer = layers.Bidirectional(layers.GRU(128, return_sequences=True))(hidden_layer)
    hidden_layer = layers.Dropout(0.3)(hidden_layer)
    
    # Second Bidirectional GRU Layer
    hidden_layer = layers.Bidirectional(layers.GRU(64))(hidden_layer)
    hidden_layer = layers.Dropout(0.3)(hidden_layer)
    
    # Dense Layers
    hidden_layer = layers.Dense(64, activation='relu')(hidden_layer)
    hidden_layer = layers.Dropout(0.2)(hidden_layer)
    
    output_layer = layers.Dense(3, activation='softmax')(hidden_layer)
    
    return keras.Model(inputs=input_layer, outputs=output_layer)


TRAINING_EPOCHS = 3
TRAIN_BATCH_SIZE = 32
k_fold_count = 5
fold_performance_metrics = []
ensemble_models = []


for current_fold_index in range(k_fold_count):
    print("\n" + "="*80)
    print(f"TRAINING FOLD {current_fold_index}")
    print("="*80)
    
    # Splitting data based on 'fold' column using the renamed dataframe
    fold_train_subset = df_train_folds[df_train_folds['fold'] != current_fold_index].copy()
    fold_validation_subset = df_train_folds[df_train_folds['fold'] == current_fold_index].copy()
    
    print("Train Size:", len(fold_train_subset), "Val Size:", len(fold_validation_subset))
    
    # Tokenization using renamed tokenizer and text column
    train_sequences = text_tokenizer.texts_to_sequences(fold_train_subset['text_concatenated'])
    validation_sequences = text_tokenizer.texts_to_sequences(fold_validation_subset['text_concatenated'])
    
    # Padding using renamed sequence length variable
    train_padded_inputs = pad_sequences(train_sequences, maxlen=SEQUENCE_LENGTH, padding='post', truncating='post')
    validation_padded_inputs = pad_sequences(validation_sequences, maxlen=SEQUENCE_LENGTH, padding='post', truncating='post')
    
    # Target extraction
    train_targets = fold_train_subset[['winner_model_a', 'winner_model_b', 'winner_tie']].values
    validation_targets = fold_validation_subset[['winner_model_a', 'winner_model_b', 'winner_tie']].values
    
    print("X_train shape:", train_padded_inputs.shape)
    print("y_train shape:", train_targets.shape)
    
    # Model instantiation using renamed function and constants
    current_gru_model = build_gru_architecture(total_vocabulary_size=VOCAB_SIZE_LIMIT, embed_dimension=128, sequence_limit=SEQUENCE_LENGTH)
    
    current_gru_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    current_gru_model.summary()
    
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
    
    training_history = current_gru_model.fit(
        train_padded_inputs, train_targets,
        validation_data=(validation_padded_inputs, validation_targets),
        epochs=TRAINING_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        callbacks=[early_stop_callback, lr_scheduler],
        verbose=1
    )
    
    validation_predictions = current_gru_model.predict(validation_padded_inputs, batch_size=TRAIN_BATCH_SIZE, verbose=0)
    validation_log_loss = log_loss(validation_targets, validation_predictions)
    
    print("Fold", current_fold_index, "Validation Log Loss:", validation_log_loss)
    fold_performance_metrics.append(validation_log_loss)
    
    saved_model_filepath = f'/kaggle/working/gru_model_fold_{current_fold_index}.h5'
    current_gru_model.save(saved_model_filepath)
    print("Model saved:", saved_model_filepath)
    
    ensemble_models.append(current_gru_model)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(training_history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(training_history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title(f'Fold {current_fold_index} - Loss')
    axes[0].legend()
    
    axes[1].plot(training_history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(training_history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_title(f'Fold {current_fold_index} - Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'/kaggle/working/gru_fold_{current_fold_index}_history.png', dpi=100)
    plt.show()


print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)

for fold_idx, individual_loss in enumerate(fold_performance_metrics):
    print(f"Fold {fold_idx}: {individual_loss}")

average_log_loss = np.mean(fold_performance_metrics)
log_loss_standard_deviation = np.std(fold_performance_metrics)

print("\nMean CV Score:", average_log_loss)
print("Std CV Score:", log_loss_standard_deviation)

cv_results_dataframe = pd.DataFrame({
    'fold': range(k_fold_count),
    'log_loss': fold_performance_metrics
})

cv_results_dataframe.to_csv('/kaggle/working/gru_cv_results.csv', index=False)

print("\n" + "="*80)
print("GRU TRAINING COMPLETED")
print("="*80)

