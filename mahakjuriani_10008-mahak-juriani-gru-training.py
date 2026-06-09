import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import torch.optim as optim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import log_loss
import pickle
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
train = pd.read_csv('/kaggle/input/creating-folds-lmsys-10008-mahakjuriani/train_5folds.csv')
train.head()


train['text'] = 'User prompt: ' + train['prompt'] +  '\n\nModel A :\n' + train['response_a'] +'\n\n--------\n\nModel B:\n'  + train['response_b']
print(train['text'][4])


print(len(train))
train.head()


kfold = 5
test_texts = train[train['kfold']==kfold]['text'].values
train_texts = train[train['kfold']!=kfold]['text'].values

len(test_texts)+len(train_texts)


test_label = train[train['kfold']==kfold]['label'].values
train_label = train[train['kfold']!=kfold]['label'].values

len(test_label)+len(train_label)


MAX_WORDS = 20000
MAX_LEN = 512

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(train['text'])

with open('/kaggle/working/tokenizer_gru.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Vocabulary Size:", len(tokenizer.word_index))
print("Using top words:", MAX_WORDS)


from tensorflow import keras
from tensorflow.keras import layers

def create_gru_model(vocab_size=MAX_WORDS, embedding_dim=128, max_len=MAX_LEN):

    inputs = layers.Input(shape=(max_len,))
    
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs)
    
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Bidirectional(layers.GRU(64))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(3, activation='softmax')(x)  # 3 classes fixed

    return keras.Model(inputs, outputs)


from tqdm.keras import TqdmCallback

n_folds = 5
BATCH_SIZE = 32
EPOCHS = 2
fold_scores = []
models = []

for fold in range(n_folds):
    print("\n" + "="*80)
    print(f"TRAINING FOLD {fold}")
    print("="*80)
    
    train_data = train[train['kfold'] != fold].copy()
    val_data = train[train['kfold'] == fold].copy()

    X_train_seq = tokenizer.texts_to_sequences(train_data['text'])
    X_val_seq = tokenizer.texts_to_sequences(val_data['text'])

    X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post')
    X_val = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding='post')

    y_train = train_data[['winner_model_a', 'winner_model_b', 'winner_tie']].values
    y_val = val_data[['winner_model_a', 'winner_model_b', 'winner_tie']].values

    model = create_gru_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[TqdmCallback(verbose=1)],
        verbose=1
    )

    # Evaluate
    val_preds = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    val_loss = log_loss(y_val, val_preds)
    print(f"Fold {fold} Validation Log Loss: {val_loss:.4f}")
    fold_scores.append(val_loss)

    model_path = f'/kaggle/working/gru_model_fold_{fold}.keras'
    model.save(model_path)
    print("Model saved:", model_path)

    from tensorflow.keras import backend as K
    K.clear_session()

print("\nAll folds completed.")
print("Validation Log Loss per fold:", fold_scores)
print("Mean Log Loss:", np.mean(fold_scores))

