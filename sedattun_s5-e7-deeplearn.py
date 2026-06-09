import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
subm = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


print(train.head())
print(test.head())


print(train.info())
print(train.info())


print(train.isnull().sum())
print(test.isnull().sum())


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


print("Eğitim Veri Seti Boyutu:", train.shape)
print("Test Veri Seti Boyutu:", test.shape)


# Test setindeki User_id'leri daha sonra gönderim dosyası için sakla
test_user_ids = test['id']

# Model için gereksiz olan User_id sütununu kaldır
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# Hedef değişken (y) ve özellikler (X) olarak ayır
X = train.drop("Personality", axis=1)
y = train["Personality"]
X_test = test.copy()


numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("\nSayısal Sütunlar:", numerical_cols)
print("Kategorik Sütunlar:", categorical_cols)


num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

print("\nEksik veriler dolduruldu.")


X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Eğitim ve test setlerinin sütunlarını eşitleme (olası farklı kategorilerden dolayı)
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Hedef Değişkeni (Personality) Label Encoding ile dönüştürme
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"\nHedef değişken dönüştürüldü. Sınıflar: {label_encoder.classes_} -> {np.unique(y_encoded)}")
print(f"Toplam Sınıf Sayısı: {num_classes}")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

print(f"\nVeri ön işleme tamamlandı. Eğitim için özellik sayısı: {X_scaled.shape[1]}")


model = tf.keras.models.Sequential([
    # Giriş Katmanı
    tf.keras.layers.Input(shape=(X_scaled.shape[1],)),
    
    # 1. Gizli Katman
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    # 2. Gizli Katman
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    # 3. Gizli Katman
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    
    # Çıkış Katmanı
    tf.keras.layers.Dense(num_classes, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model mimarisini göster
model.summary()


# 4. MODELİ EĞİTME
# -------------------------------------------

# Erken Durdurma (Early Stopping) callback'i
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',      # İzlenecek metrik
    patience=15,             # Gelişme olmadan beklenecek epoch sayısı
    restore_best_weights=True # En iyi ağırlıkları geri yükle
)

print("\nModel eğitimi başlıyor...")

history = model.fit(
    X_scaled,
    y_encoded,
    epochs=100,  # Maksimum epoch sayısı (erken durdurma sayesinde daha önce bitebilir)
    batch_size=64,
    validation_split=0.2, # Verinin %20'sini doğrulama için ayır
    callbacks=[early_stopping],
    verbose=1 # Eğitim sürecini göster
)

print("\nModel eğitimi tamamlandı.")


# Test seti üzerinde tahmin yapma
predictions_proba = model.predict(X_test_scaled)
predicted_classes_encoded = np.argmax(predictions_proba, axis=1)

# Tahmin edilen sayısal etiketleri orijinal metin etiketlerine geri dönüştürme
predicted_personality = label_encoder.inverse_transform(predicted_classes_encoded)

# Gönderim için DataFrame oluşturma
submission_df = pd.DataFrame({
    'User_id': test_user_ids,
    'Personality': predicted_personality
})



# Gönderim dosyasını kaydetme
submission_df.to_csv('submission.csv', index=False)

print("\n'submission.csv' dosyası başarıyla oluşturuldu.")
print(submission_df.head())




