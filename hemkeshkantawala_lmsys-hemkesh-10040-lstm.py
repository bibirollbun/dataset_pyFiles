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

train_df = pd.read_csv('/kaggle/input/lmsys-hemkesh-10040-n-folds/train_folds.csv')

print("Train Shape:", train_df.shape)
print("Columns:", train_df.columns.tolist())



train_df['combined_text'] = (
    train_df['prompt'] + ' [SEP] ' +
    train_df['response_a'] + ' [SEP] ' +
    train_df['response_b']
)


MAX_WORDS = 20000
MAX_LEN = 512

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['combined_text'])

with open('/kaggle/working/tokenizer_lstm.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Vocabulary Size:", len(tokenizer.word_index))
print("Using top", MAX_WORDS)


def create_lstm_model(vocab_size, embedding_dim=128, max_len=512):
    inputs = layers.Input(shape=(max_len,))
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(3, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs)


EPOCHS = 5
BATCH_SIZE = 32
n_folds = 5

fold_scores = []
models = []


for fold in range(n_folds):
    print("\n" + "="*80)
    print(f"TRAINING FOLD {fold}")
    print("="*80)
    
    train_data = train_df[train_df['fold'] != fold].copy()
    val_data = train_df[train_df['fold'] == fold].copy()
    
    print("Train Size:", len(train_data), "Val Size:", len(val_data))
    
    X_train_seq = tokenizer.texts_to_sequences(train_data['combined_text'])
    X_val_seq = tokenizer.texts_to_sequences(val_data['combined_text'])
    
    X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_val = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    y_train = train_data[['winner_model_a', 'winner_model_b', 'winner_tie']].values
    y_val = val_data[['winner_model_a', 'winner_model_b', 'winner_tie']].values
    
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    
    model = create_lstm_model(vocab_size=MAX_WORDS, embedding_dim=128, max_len=MAX_LEN)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    val_preds = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    val_loss = log_loss(y_val, val_preds)
    
    print("Fold", fold, "Validation Log Loss:", val_loss)
    fold_scores.append(val_loss)
    
    model_path = f'/kaggle/working/lstm_model_fold_{fold}.h5'
    model.save(model_path)
    print("Model saved:", model_path)
    
    models.append(model)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['loss'], label='Train Loss', color='#2ecc71', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='#e74c3c', linewidth=2)
    axes[0].set_title(f'Fold {fold} - Loss')
    axes[0].legend()
    
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', color='#2ecc71', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', color='#e74c3c', linewidth=2)
    axes[1].set_title(f'Fold {fold} - Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'/kaggle/working/lstm_fold_{fold}_history.png', dpi=100, bbox_inches='tight')
    plt.show()


print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)

for i, score in enumerate(fold_scores):
    print(f"Fold {i}: {score:.6f}")

mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)

print("Mean CV Score:", mean_score)
print("Std CV Score:", std_score)

results_df = pd.DataFrame({
    'fold': range(n_folds),
    'log_loss': fold_scores
})

results_df.to_csv('/kaggle/working/lstm_cv_results.csv', index=False)

print("\n" + "="*80)
print("LSTM TRAINING COMPLETED")
print("="*80)


