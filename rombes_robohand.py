import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Bidirectional, LSTM, TimeDistributed, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import f1_score, classification_report


def find_file(filename):
    for root, dirs, files in os.walk('/kaggle/input'):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(filename)

X_train = np.load(find_file("X_train.npy"))
X_test = np.load(find_file("X_test.npy"))
y_df = pd.read_csv(find_file("y_train.csv"))
sample_sub = pd.read_csv(find_file("sample_submission.csv"))

N, S, T = X_train.shape
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train: {y_df.shape}")


print(y_df['class'].value_counts())
sns.countplot(x='class', data=y_df)
plt.title("Распределение классов")
plt.show()


y_matrix = np.full((N, T), -1, dtype=int)
for _, row in y_df.iterrows():
    s, t = map(int, row["sample-timestep"].split("-"))
    y_matrix[s, t] = row["class"]

# Заполняем пропуски модой
mode_val = Counter(y_matrix[y_matrix != -1]).most_common(1)[0][0]
y_matrix[y_matrix == -1] = mode_val
print(f"Пропусков заполнено модой: {mode_val}")


plt.figure(figsize=(12,6))
for i in range(5):
    plt.plot(X_train[i,0,:], label=f"sample {i} sensor 0")
plt.title("Примеры сигналов с первого сенсора")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.legend()
plt.show()


X_train_nts = np.transpose(X_train, (0, 2, 1))
X_test_nts = np.transpose(X_test, (0, 2, 1))

scaler = StandardScaler()
X_train_flat = X_train_nts.reshape(-1, S)
X_test_flat = X_test_nts.reshape(-1, S)
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(N, T, S)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test_nts.shape[0], T, S)

num_classes = int(y_matrix.max()) + 1
y_cat = tf.keras.utils.to_categorical(y_matrix, num_classes=num_classes)


sample_mode = np.array([Counter(y_matrix[i]).most_common(1)[0][0] for i in range(N)])
train_idx, val_idx = train_test_split(np.arange(N), test_size=0.2, random_state=42, stratify=sample_mode)

X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
y_tr, y_val = y_cat[train_idx], y_cat[val_idx]
y_val_true = y_matrix[val_idx]
print("Тренировочные данные:", X_tr.shape, y_tr.shape)
print("Проверочные данные:", X_val.shape, y_val.shape)


flat_all = y_matrix.flatten()
cw = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=flat_all)
cw_dict = dict(enumerate(cw))

def seq_weights(y_seq):
    w = np.zeros_like(y_seq, dtype=float)
    for c, wv in cw_dict.items():
        w[y_seq == c] = wv
    return w

sw_tr = seq_weights(y_matrix[train_idx])
sw_val = seq_weights(y_matrix[val_idx])


def build_model(T, S, C):
    inp = Input(shape=(T, S))
    x = Conv1D(64, 5, padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    out = TimeDistributed(Dense(C, activation="softmax"))(x)
    return Model(inp, out)

model = build_model(T, S, num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4),
    ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True)
]


history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val, sw_val),
    epochs=35,
    batch_size=16,
    sample_weight=sw_tr,
    callbacks=callbacks,
    verbose=1
)


plt.figure(figsize=(10,4))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Loss по эпохам")
plt.legend()
plt.show()


pred_val = model.predict(X_val)
pred_val_labels = np.argmax(pred_val, axis=-1).reshape(-1)
y_true_flat = y_val_true.reshape(-1)

print("VAL Macro F1 =", f1_score(y_true_flat, pred_val_labels, average='macro'))


pred_val = model.predict(X_val)  # предсказания на валидации
pred_val_labels = np.argmax(pred_val, axis=-1).reshape(-1)

y_val_flat = y_val_true.reshape(-1)  # истинные метки валидации

df_compare = pd.DataFrame({
    "pred_class": pred_val_labels,
    "true_class": y_val_flat
})

print(df_compare)



for i in range(10):
    
    plt.figure(figsize=(14,5))
    
    plt.plot(pred_val_labels[i*T:(i+1)*T], label="Predicted")
    print(i*T, (i+1)*T)
    plt.plot(y_val_true[i], label="True", alpha=0.7)
    
    plt.title(f"Сравнение предсказаний и истинных классов (пример {i})")
    plt.xlabel("Time step")
    plt.ylabel("Class")
    plt.legend()
    plt.show()



pred_test = model.predict(X_test_scaled)
pred_labels_flat = np.argmax(pred_test, axis=-1).flatten()
pred_labels_flat = pred_labels_flat[:len(sample_sub)]

submission = sample_sub.copy()
submission['class'] = pred_labels_flat.astype(int)
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv:", submission.shape)

