import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, average_precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_recall_curve, auc


df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df.head()


df.info()


print("Soil Type:")
print(df['Soil Type'].unique())
print("\nCrop Type:")
print(df['Crop Type'].unique())
print("\nFertilizer Name:")
print(df['Fertilizer Name'].unique())


df_encoded = pd.get_dummies(df, columns=['Soil Type', 'Crop Type'], drop_first=True)
df_encoded.head()


df_encoded.columns


df_encoded.info()


X = df_encoded.drop(['id','Fertilizer Name'], axis=1).astype(int)
y = df['Fertilizer Name']
print("\nShape data fitur (X):", X.shape)
print("Shape data target (y):", y.shape)
X.head()


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Ubah ke one-hot encoding untuk output layer softmax
y_categorical = to_categorical(y_encoded)


# Rasio umum: 70% training, 15% validation, 15% testing
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_categorical, test_size=0.15, random_state=42, stratify=y_encoded)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=(0.15/0.85), random_state=42, stratify=np.argmax(y_train_val, axis=1)) # 0.15/0.85 agar menjadi 15% dari total


# Identifikasi kolom numerik (selain yang boolean/one-hot encoded)
numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


# Membangun Model Deep Learning (Keras TensorFlow) ---
input_shape = X_train.shape[1] # Jumlah fitur
output_classes = y_categorical.shape[1] # Jumlah kelas target

model = Sequential([
    Dense(128, activation='relu', input_shape=(input_shape,)),
    Dropout(0.3), # Mencegah overfitting
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(output_classes, activation='softmax') # Softmax untuk klasifikasi multi-kelas
])

# Kompilasi model
model.compile(optimizer='adam',
              loss='categorical_crossentropy', # Cocok untuk one-hot encoded target
              metrics=['accuracy'])

model.summary()


print("\n--- Training Start ---")
history = model.fit(X_train, y_train,
                    epochs = 100, # Jumlah epoch, bisa disesuaikan
                    batch_size=16, # Ukuran batch
                    validation_data=(X_val, y_val))

print("\n--- Training Completed ---")


# Prediksi pada data test
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)


# Prediksi pada data test
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Mengkonversi y_test kembali ke label numerik asli untuk classification_report
y_test_labels = np.argmax(y_test, axis=1)

# a. Accuracy
accuracy = accuracy_score(y_test_labels, y_pred)
print(f"\nAccuracy pada Data Test: {accuracy:.4f}")

# b. Classification Report
target_names = label_encoder.classes_
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred, target_names=target_names))


# c. mAP@3 (Mean Average Precision at 3)
y_test_binary = y_test

average_precisions = []
for i, class_name in enumerate(target_names):
    # Dapatkan probabilitas untuk kelas saat ini
    y_true_class = y_test_binary[:, i]
    y_score_class = y_pred_proba[:, i]

    # Handle kasus di mana tidak ada instance positif untuk kelas ini di test set
    if np.sum(y_true_class) == 0:
        print(f"Peringatan: Tidak ada instance positif untuk kelas '{class_name}' di set pengujian. AP untuk kelas ini akan dilewati.")
        continue

    precision, recall, _ = precision_recall_curve(y_true_class, y_score_class)
    ap = auc(recall, precision)
    average_precisions.append(ap)
    print(f"Average Precision (AP) untuk kelas '{class_name}': {ap:.4f}")

if average_precisions:
    mean_average_precision = np.mean(average_precisions)
    print(f"\nMean Average Precision (mAP) Seluruh Kelas: {mean_average_precision:.4f}")
else:
    print("\nUnable to calculate mAP because there is no valid Average Precision.")


def calculate_top_k_accuracy(y_true_one_hot, y_pred_proba, k=3):
    num_samples = y_true_one_hot.shape[0]
    top_k_correct = 0
    for i in range(num_samples):
        # Dapatkan indeks kelas asli
        true_class_idx = np.argmax(y_true_one_hot[i])
        # Dapatkan indeks k prediksi teratas berdasarkan probabilitas
        top_k_pred_indices = np.argsort(y_pred_proba[i])[-k:][::-1] # Ambil k tertinggi, descending

        if true_class_idx in top_k_pred_indices:
            top_k_correct += 1
    return top_k_correct / num_samples

print(f"\nTop-3 Accuracy (jika ini yang dimaksud dengan mAP@3): {calculate_top_k_accuracy(y_test, y_pred_proba, k=3):.4f}")


df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
df_test.head()


df_test.info()


df_test_encoded = pd.get_dummies(df_test, columns=['Soil Type', 'Crop Type'], drop_first=True)
df_test_encoded.head()


df_test_encoded.columns


test_ids = df_test_encoded['id']
X_test_predict = df_test_encoded.drop('id', axis=1)

X_test_predict = X_test_predict[X_train.columns]

X_test_predict[numerical_cols] = scaler.transform(X_test_predict[numerical_cols])


# --- Melakukan Prediksi Probabilitas ---
predictions_proba = model.predict(X_test_predict)

# --- Mengambil 3 Prediksi Teratas ---
top_3_indices = np.argsort(predictions_proba, axis=1)[:, -3:][::-1]

# Mengubah indeks menjadi nama pupuk yang sebenarnya
top_3_fertilizers = label_encoder.inverse_transform(top_3_indices.reshape(-1)).reshape(top_3_indices.shape)


# Menggabungkan nama-nama pupuk dengan spasi sebagai pemisah
results_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': [' '.join(row) for row in top_3_fertilizers]
})

print("\nHasil Prediksi 3 Optimal Fertilizer Teratas (digabung dalam 1 kolom):")
results_df.head(10)


results_df.to_csv('Optimal Fertilizer with Keras.csv', index=False)
print("\nThe prediction results have been saved")

