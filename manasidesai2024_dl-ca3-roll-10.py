import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


train_df = pd.read_csv("/content/train-reviews-gmaps.csv")
test_df = pd.read_csv("/content/test-review-gmaps-new.csv")


print(train_df.head())
X = train_df['reviews']
y = train_df['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


#Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(test_df['reviews'])


#Padding
max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')


model = Sequential([
    Embedding(10000, 64, input_shape=(max_len,)),
    LSTM(64, dropout=0.4, recurrent_dropout=0.4),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


y_train_numeric = y_train.apply(lambda x: 1 if x == 'Positive' else 0)
y_val_numeric = y_val.apply(lambda x: 1 if x == 'Positive' else 0)


model.fit(X_train_pad, y_train_numeric, epochs=5, validation_data=(X_val_pad, y_val_numeric), batch_size=64)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

loss, accuracy = model.evaluate(X_val_pad, y_val_numeric, verbose=0)
print(f"Validation Accuracy: {accuracy:.4f}")
y_val_pred = (model.predict(X_val_pad) > 0.5).astype(int)
precision = precision_score(y_val_numeric, y_val_pred)
recall = recall_score(y_val_numeric, y_val_pred)
f1 = f1_score(y_val_numeric, y_val_pred)

print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1-score: {f1:.4f}")


predictions = (model.predict(X_test_pad) > 0.5).astype(int)
test_df['predicted_label'] = predictions


test_df.to_csv("submission.csv", index=False)

