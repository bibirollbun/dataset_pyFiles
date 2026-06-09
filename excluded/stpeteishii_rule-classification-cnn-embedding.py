


# 1. Import
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Conv1D, GlobalMaxPooling1D,
                                     Concatenate, Dense, Dropout)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix



# 2. Load train.csv
# 2. Load data
train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
print(len(train),len(test))


# 3. Combine text columns into a single input text
def combine_text(row):
    return (
        f"Subreddit: {row['subreddit']} [SEP] "
        f"Rule: {row['rule']} [SEP] "
        f"Positive Example 1: {row['positive_example_1']} [SEP] "
        f"Positive Example 2: {row['positive_example_2']} [SEP] "
        f"Negative Example 1: {row['negative_example_1']} [SEP] "
        f"Negative Example 2: {row['negative_example_2']} [SEP] "
        f"Post: {row['body']}"
    )

train['text'] = train.apply(combine_text, axis=1)
test['text'] = test.apply(combine_text, axis=1)


# 3. Tokenization and padding
MAX_VOCAB_SIZE = 10000
MAX_SEQ_LEN = 150

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train['text'])
sequences = tokenizer.texts_to_sequences(train['text'])
padded = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')
test_sequences = tokenizer.texts_to_sequences(test['text'])
test_padded = pad_sequences(test_sequences, maxlen=MAX_SEQ_LEN, padding='post')


# 4. Split data
X_train, X_val, y_train, y_val = train_test_split(
    padded, train['rule_violation'].values,
    test_size=0.2, random_state=42, stratify=train['rule_violation']
)


# 5. Build CNN + Embedding model
def build_cnn_model(vocab_size, embedding_dim=128, max_len=150, filter_sizes=[3,4,5], num_filters=64):
    inputs = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(inputs)
    
    conv_blocks = []
    for size in filter_sizes:
        conv = Conv1D(filters=num_filters, kernel_size=size, activation='relu')(x)
        pool = GlobalMaxPooling1D()(conv)
        conv_blocks.append(pool)

    x = Concatenate()(conv_blocks)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_cnn_model(vocab_size=MAX_VOCAB_SIZE)
model.summary()



# 6. Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=2
)


# 7. Evaluate
y_pred_prob = model.predict(X_val)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=["not_violation", "violation"]))


# 9. Predict on test set
test_pred_prob = model.predict(test_padded)
test['rule_violation'] = (test_pred_prob > 0.5).astype(int).flatten()

# Save submission
test[['row_id', 'rule_violation']].to_csv("submission.csv", index=False)
print("submission.csv saved!")




