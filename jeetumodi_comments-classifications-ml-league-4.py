import re
import string

def clean_text(text: str):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+\.(jpg|jpeg|png|gif|bmp|svg)', '', text, flags=re.IGNORECASE)
    text=re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"-]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


import pandas as pd
from tqdm import tqdm
import seaborn as sns
tqdm.pandas()

df=pd.read_csv("/kaggle/input/comments-classification/Dataset/train.csv")
df["clean_text"]=df['comment_text'].progress_apply(clean_text)

plot = sns.countplot(x=df['psychotic_depression'], data=df)
print(df['psychotic_depression'].value_counts())


df.head()


from sklearn.model_selection import train_test_split

X=df['clean_text']
y=df['psychotic_depression']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalAveragePooling1D, Attention


vocab_size = 12000
max_len = 200
embedding_dim = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')



inputs = Input(shape=(max_len,))

# Embedding Layer
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(inputs)

# BiLSTM stack
x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)
x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)

# Attention
attention_out = Attention()([x, x])

# Pooling
x = GlobalAveragePooling1D()(attention_out)

# Dense + Dropout
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)

# Output layer
outputs = Dense(1, activation="sigmoid")(x)

# Build model
model = Model(inputs, outputs)




# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()



import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

# Define early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True )

# Training
history = model.fit(
    X_train_pad, np.array(y_train),
    validation_data=(X_test_pad, np.array(y_test)),
    epochs=15,
    batch_size=256,
    verbose=1,
    callbacks=[early_stop]
)



import numpy as np
from sklearn.metrics import classification_report, f1_score

def evaluate_best_threshold(model, X_test_pad, y_test, thresholds=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.65]):
    # Predict probabilities once
    y_pred_probs = model.predict(X_test_pad, verbose=0)

    best_threshold = None
    best_f1 = -1
    best_report = None
    best_preds = None

    for t in thresholds:
        y_pred = (y_pred_probs > t).astype("int32")
        f1 = f1_score(y_test, y_pred, pos_label=1)  # F1 only for class 1

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_report = classification_report(y_test, y_pred, digits=4)
            best_preds = y_pred

    print(f"\nBest Threshold: {best_threshold}")
    print(f"Best F1 Score (class 1): {best_f1:.4f}")
    print("Classification Report:\n", best_report)

    return best_threshold, best_preds



# Predictions
y_pred_probs = model.predict(X_test_pad)
best_t, best_preds = evaluate_best_threshold(model, X_test_pad, y_test)


!pip install joblib

import joblib

# Save model
model.save("lstm_sentiment_model.h5")
print("âœ… Model saved as lstm_sentiment_model.h5")

# Save tokenizer
joblib.dump(tokenizer, "tokenizer.joblib")
print("âœ… Tokenizer saved as tokenizer.joblib")





test_df=pd.read_csv("/kaggle/input/comments-classification/Dataset/test.csv")


import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np

# Load tokenizer and model
tokenizer = joblib.load("/kaggle/working/tokenizer.joblib")
model = load_model("/kaggle/working/lstm_sentiment_model.h5")

# New text to predict
new_texts = test_df['comment_text']

# Tokenize and pad
sequences = tokenizer.texts_to_sequences(new_texts)
padded = pad_sequences(sequences, maxlen=max_len, padding='post')

# Predict
predictions = model.predict(padded)

# labeling
pred_labels = (predictions > best_t).astype(int)

pred_labels=pred_labels.flatten()

df=pd.DataFrame(
    {
        "ID":[i for i in range(1,len(new_texts)+1)],
        "psychotic_depression":pred_labels
    }
)

df.to_csv("submission.csv",index=False)


df.head()




