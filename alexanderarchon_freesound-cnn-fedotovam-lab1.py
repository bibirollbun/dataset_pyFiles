import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import warnings
warnings.filterwarnings('ignore')


# Правильные пути к данным
train_csv_path = "../input/freesound-audio-tagging/train.csv"
train_audio_path = "../input/freesound-audio-tagging/audio_train/"
test_audio_path = "../input/freesound-audio-tagging/audio_test/"
sample_submission_path = "../input/freesound-audio-tagging/sample_submission.csv"


# Загрузка данных
train_df = pd.read_csv(train_csv_path)
sample_submission = pd.read_csv(sample_submission_path)

print("Размер тренировочных данных:", train_df.shape)
print("Уникальные классы:", train_df['label'].nunique())
print("\nПервые 5 записей:")
print(train_df.head())


# Анализ данных
print("\nРаспределение классов:")
class_distribution = train_df['label'].value_counts()
print(class_distribution)

print(f"\nПроверка ручной верификации:")
print(train_df['manually_verified'].value_counts())


# Визуализация распределения классов
plt.figure(figsize=(15, 8))
class_distribution.plot(kind='bar')
plt.title('Распределение аудио классов')
plt.xlabel('Классы')
plt.ylabel('Количество')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Функция для извлечения Mel-спектрограмм
def extract_mel_spectrogram(file_path, duration=3, sr=22050, n_mels=128):
    try:
        # Загружаем аудиофайл
        audio, sample_rate = librosa.load(file_path, duration=duration, sr=sr)
        
        # Если аудио короче duration, дополняем нулями
        if len(audio) < sr * duration:
            audio = np.pad(audio, (0, max(0, sr * duration - len(audio))), mode='constant')
        
        # Создаем Mel-спектрограмму
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, fmax=8000
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Нормализуем спектрограмму
        mel_spectrogram_db = (mel_spectrogram_db - mel_spectrogram_db.mean()) / (mel_spectrogram_db.std() + 1e-8)
        
        return mel_spectrogram_db
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {str(e)}")
        return None


# Функция для извлечения расширенного набора признаков
def extract_comprehensive_features(file_path, duration=3, sr=22050):
    try:
        audio, sample_rate = librosa.load(file_path, duration=duration, sr=sr)
        
        # Дополняем нулями если нужно
        if len(audio) < sr * duration:
            audio = np.pad(audio, (0, max(0, sr * duration - len(audio))), mode='constant')
        
        # Mel-спектрограмма
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
        
        # Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        
        # Статистические признаки
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        
        statistical_features = np.array([
            spectral_centroid.mean(), spectral_centroid.std(),
            spectral_rolloff.mean(), spectral_rolloff.std(),
            zero_crossing_rate.mean(), zero_crossing_rate.std(),
            np.mean(audio), np.std(audio), np.max(audio), np.min(audio)
        ])
        
        # Объединяем все признаки
        all_features = np.concatenate([
            mfccs_mean, chroma_mean, spectral_contrast_mean, 
            tonnetz_mean, statistical_features
        ])
        
        return {
            'mel_spectrogram': mel_spec_db,
            'features': all_features
        }
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {str(e)}")
        return None


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available and configured")
else:
    print("GPU not available")


# Извлекаем Mel-спектрограммы для тренировочных данных
print("Извлечение Mel-спектрограмм из тренировочных данных...")
spectrograms = []
features_list = []
labels = []

for index, row in train_df.iterrows():
    if index % 100 == 0:
        print(f"Обработано {index}/{len(train_df)} файлов")
    
    file_path = os.path.join(train_audio_path, row['fname'])
    result = extract_comprehensive_features(file_path)
    
    if result is not None:
        # Добавляем размерность канала для CNN
        spectrogram = np.expand_dims(result['mel_spectrogram'], axis=-1)
        spectrograms.append(spectrogram)
        features_list.append(result['features'])
        labels.append(row['label'])


# Преобразуем в numpy массивы
X_spectrograms = np.array(spectrograms)
X_features = np.array(features_list)
y = np.array(labels)

print(f"Форма спектрограмм: {X_spectrograms.shape}")
print(f"Форма дополнительных признаков: {X_features.shape}")
print(f"Форма меток: {y.shape}")


# Кодируем метки
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Закодированные классы: {label_encoder.classes_}")


# Разделяем данные на тренировочную и валидационную выборки
X_train_spec, X_val_spec, X_train_feat, X_val_feat, y_train, y_val = train_test_split(
    X_spectrograms, X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Тренировочные спектрограммы: {X_train_spec.shape}")
print(f"Валидационные спектрограммы: {X_val_spec.shape}")


# Улучшенная CNN архитектура с двумя входами
def create_advanced_cnn_model(spectrogram_shape, feature_shape, num_classes):
    # Вход для спектрограмм
    spec_input = layers.Input(shape=spectrogram_shape, name='spectrogram_input')
    
    # CNN ветка для спектрограмм
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(spec_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    # Вход для дополнительных признаков
    feat_input = layers.Input(shape=(feature_shape,), name='feature_input')
    y = layers.Dense(128, activation='relu')(feat_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)
    
    # Объединяем обе ветки
    combined = layers.concatenate([x, y])
    
    # Полносвязные слои
    z = layers.Dense(512, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001))(combined)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.5)(z)
    
    z = layers.Dense(256, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001))(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.5)(z)
    
    outputs = layers.Dense(num_classes, activation='softmax')(z)
    
    model = models.Model(inputs=[spec_input, feat_input], outputs=outputs)
    return model


# Создаем модель
spectrogram_shape = X_train_spec.shape[1:]
feature_shape = X_train_feat.shape[1]
num_classes = len(label_encoder.classes_)

model = create_advanced_cnn_model(spectrogram_shape, feature_shape, num_classes)


# Компилируем модель
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()


# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
)


# Обучаем модель
print("Начало обучения модели...")
history = model.fit(
    [X_train_spec, X_train_feat], y_train,
    epochs=100,
    batch_size=32,
    validation_data=([X_val_spec, X_val_feat], y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


# Визуализируем процесс обучения
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Точность модели')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Функция потерь')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()

plt.tight_layout()
plt.show()


# Оценка модели
val_loss, val_accuracy = model.evaluate([X_val_spec, X_val_feat], y_val, verbose=0)
print(f"Валидационная точность: {val_accuracy:.4f}")
print(f"Валидационные потери: {val_loss:.4f}")


# Предсказания и матрица ошибок
y_pred = model.predict([X_val_spec, X_val_feat])
y_pred_classes = np.argmax(y_pred, axis=1)

plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_val, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Матрица ошибок CNN модели')
plt.xlabel('Предсказанные метки')
plt.ylabel('Истинные метки')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Функция для предсказания на тестовых данных
def predict_test_data(model, test_audio_path, sample_submission_df, label_encoder):
    print("Обработка тестовых данных...")
    test_spectrograms = []
    test_features = []
    test_filenames = []
    
    for fname in sample_submission_df['fname']:
        file_path = os.path.join(test_audio_path, fname)
        result = extract_comprehensive_features(file_path)
        
        if result is not None:
            spectrogram = np.expand_dims(result['mel_spectrogram'], axis=-1)
            test_spectrograms.append(spectrogram)
            test_features.append(result['features'])
            test_filenames.append(fname)
    
    X_test_spec = np.array(test_spectrograms)
    X_test_feat = np.array(test_features)
    
    # Предсказания
    test_predictions = model.predict([X_test_spec, X_test_feat])
    test_pred_classes = np.argmax(test_predictions, axis=1)
    test_pred_labels = label_encoder.inverse_transform(test_pred_classes)
    
    # Создаем submission файл
    submission_df = pd.DataFrame({
        'fname': test_filenames,
        'label': test_pred_labels
    })
    
    return submission_df


# Предсказываем на тестовых данных
final_submission = predict_test_data(model, test_audio_path, sample_submission, label_encoder)


# Сохраняем результаты
final_submission.to_csv('submission.csv', index=False)
print("Submission файл сохранен как 'submission.csv'")

