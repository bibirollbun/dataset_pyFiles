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
%matplotlib inline



print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

train = pd.read_csv('/kaggle/input/lmsys-n-folds/train_folds.csv')
print("\nTrain Shape:", train.shape)
print("Columns:", train.columns.tolist())

display(train.head())



train['combined_text'] = train['prompt'].fillna('') + ' [SEP] ' + \
                         train['response_a'].fillna('') + ' [SEP] ' + \
                         train['response_b'].fillna('')

print("Sample combined_text:")
print(train['combined_text'].iloc[0][:400])



MAX_WORDS = 20000
MAX_LEN = 512

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(train['combined_text'].astype(str).values)

with open('/kaggle/working/tokenizer_rnn.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"Vocabulary Size (unique tokens): {len(tokenizer.word_index)}")
print(f"Using top {MAX_WORDS} words and max sequence length {MAX_LEN}")



def create_rnn_model(vocab_size, embedding_dim=128, max_len=512):
    inputs = layers.Input(shape=(max_len,))
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs)
    x = layers.SimpleRNN(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.SimpleRNN(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(3, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model



EPOCHS = 10
BATCH_SIZE = 32
n_folds = 5

fold_scores = []
models = []
histories = []  



def prepare_sequences(tokenizer, texts, max_len=MAX_LEN):
    seqs = tokenizer.texts_to_sequences(texts.astype(str).values)
    return pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')


for fold in range(n_folds):
    print("\n" + "="*80)
    print(f"TRAINING FOLD {fold}")
    print("="*80)
    
    train_data = train[train['fold'] != fold].copy()
    val_data = train[train['fold'] == fold].copy()
    
    print(f"Train Size: {len(train_data)}, Val Size: {len(val_data)}")
    
    X_train = prepare_sequences(tokenizer, train_data['combined_text'], max_len=MAX_LEN)
    X_val = prepare_sequences(tokenizer, val_data['combined_text'], max_len=MAX_LEN)
    
    y_train = train_data[['winner_model_a', 'winner_model_b', 'winner_tie']].values
    y_val = val_data[['winner_model_a', 'winner_model_b', 'winner_tie']].values
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    model = create_rnn_model(vocab_size=MAX_WORDS, embedding_dim=128, max_len=MAX_LEN)
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
    print(f"\nFold {fold} Validation Log Loss: {val_loss:.6f}")
    fold_scores.append(val_loss)
    histories.append(history)
    
    model_path = f'/kaggle/working/rnn_model_fold_{fold}.h5'
    model.save(model_path)
    print(f"Model saved: {model_path}")
    models.append(model)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title(f'Fold {fold} - Loss', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_title(f'Fold {fold} - Accuracy', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'/kaggle/working/rnn_fold_{fold}_history.png', dpi=100, bbox_inches='tight')
    plt.show()



print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)

for i, score in enumerate(fold_scores):
    print(f"Fold {i}: {score:.6f}")

mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)
print(f"\nMean CV Score: {mean_score:.6f}")
print(f"Std CV Score: {std_score:.6f}")

results_df = pd.DataFrame({'fold': range(len(fold_scores)), 'log_loss': fold_scores})
results_df.to_csv('/kaggle/working/rnn_cv_results.csv', index=False)

print("\n" + "="*80)
print("RNN TRAINING COMPLETED")
print("="*80)


