import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')


def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'http\S+', ' URL ', text)
    text = re.sub(r'@\w+', ' USER ', text)
    text = re.sub(r'[^a-z0-9\s\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


if 'body' in train.columns:
    text_col = 'body'
elif 'input' in train.columns:
    text_col = 'input'
else:
    possible = [c for c in train.columns if train[c].dtype == 'object']
    if not possible:
        raise ValueError("Couldn't find text column in train CSV. Add 'body' or 'input' column.")
    text_col = possible[0]
print("Detected text column:", text_col)


if 'rule_violation' in train.columns:
    label_col = 'rule_violation'
elif "label" in df.columns:
    label_col = "label"
else:
    raise ValueError("Couldn't find label column in train CSV. Expected 'rule_violation' or 'label'.")


train[text_col] = train[text_col].astype(str).map(basic_clean)
train = train[[text_col, label_col]].dropna()
train[label_col] = train[label_col].astype(int)


train, val, train_labels, val_labels = train_test_split(
    train[text_col], train[label_col], test_size=0.15, stratify=train[label_col], random_state=42
)


print(f'train size: {train.size}')
print(f'validation size: {val.size}')
print(f'train label size: {train_labels.size}')
print(f'validation label size: {val_labels.size}')


MAX_VOCAB = 25000
MAX_LEN = 256
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<unk>')
tokenizer.fit_on_texts(train)


def encode_and_pad(texts):
    return pad_sequences(
        tokenizer.texts_to_sequences(texts),
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )


X_train = encode_and_pad(train)
X_val   = encode_and_pad(val)
y_train = np.array(train_labels)
y_val = np.array(val_labels)


print("Shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)


EMB_DIM = 128
HIDDEN_DIM = 128
dropout_rate = 0.3


def build_model(vocab_size=MAX_VOCAB, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, max_len=MAX_LEN):
    inp = layers.Input(shape=(max_len,), dtype="int32")
    emb = layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=max_len)(inp)
    x = layers.Bidirectional(layers.LSTM(hidden_dim, return_sequences=False))(emb)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=inp, outputs=out)
    return model


model = build_model()
model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
)
model.summary()


classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {int(c): w for c, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_rule_violation.h5", monitor="val_loss", save_best_only=True)
]


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=12,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=callbacks
)


val_preds_prob = model.predict(X_val, batch_size=128).ravel()
val_preds = (val_preds_prob > 0.5).astype(int)
print("\nValidation metrics")
print("Accuracy:", accuracy_score(y_val, val_preds))
print(classification_report(y_val, val_preds, digits=4))


if 'body' in test.columns:
    test_text_col = 'body'
elif 'input' in test.columns:
    test_text_col = input
else:
    possible = [c for c in test.columns if test[c].dtype == object]
    if possible:
        test_test_col = possible[0]
    else:
        raise ValueError("Couldn't find text column in test CSV. Add 'body' or 'input' column.")
    print("Detected test text column:", test_text_col)
    test_df[test_text_col] = test_df[test_text_col].astype(str).map(basic_clean)


full_texts = pd.concat([train, val], ignore_index=True)
full_labels = np.concatenate([y_train, y_val], axis=0)
tokenizer.fit_on_texts(full_texts)  # re-fit tokenizer on full text (optional)
X_full = pad_sequences(tokenizer.texts_to_sequences(full_texts), maxlen=MAX_LEN, padding="post", truncating="post")


model_full = build_model()
model_full.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"]
)
model_full.fit(
    X_full, full_labels,
    epochs=10,
    batch_size=64,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)]
)


X_test = encode_and_pad(test[test_text_col].astype(str))
probs = model_full.predict(X_test, batch_size=128).ravel()
preds = (probs > 0.5).astype(int)


if "id" in test.columns:
    out_id = test["id"].values
elif "row_id" in test.columns:
    out_id = test["row_id"].values
else:
    out_id = np.arange(len(test))


submission = pd.DataFrame({
    "row_id": out_id,
    "rule_violation": preds,
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Saved submission.csv (row_id, rule_violation)")


model_full.save("rule_violation_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("Saved model + tokenizer")




