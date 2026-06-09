import os

import numpy as np
import pandas as pd
import librosa
import librosa.display

import matplotlib.pyplot as plt

from tqdm import tqdm
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau



# Пути к данным
TRAIN_CSV = '/kaggle/input/freesound-audio-tagging/train.csv'
TRAIN_DIR = '/kaggle/input/freesound-audio-tagging/audio_train/'

SAMPLE_SUBMISSION_CSV = '/kaggle/input/freesound-audio-tagging/sample_submission.csv'
TEST_DIR = '/kaggle/input/freesound-audio-tagging/audio_test/'

# Параметры аудио
# Частота дискретизации
SAMPLE_RATE = 44100
 # Длительность в секундах
DURATION = 3
# Общее количество точек
SAMPLES = SAMPLE_RATE * DURATION 

# Параметры изображений
# Размер изображения
IMG_SIZE = 128



# Функция для приведения аудиосигнала к фиксированной длине
def ensure_sample_length(y, target_length):
    if len(y) > target_length:
        return y[:target_length]
    else:
        padding = target_length - len(y)
        return np.pad(y, (0, padding), 'constant')

# Функция для извлечения признаков из аудиофайла
def get_features(file_path):
    try:
        # Загружаем аудио
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Приводим аудио к нужной длине
        y = ensure_sample_length(y, SAMPLES)

        # Извлекаем мел-спектрограмму
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        melspec_db = librosa.power_to_db(melspec, ref=np.max)

        # Извлекаем скорость измениний мэл-спектрограммы (первая производная)
        delta = librosa.feature.delta(melspec_db)

        # Извлекаем ускорение изменений мэл-спектрограммы (вторая производная)
        delta2 = librosa.feature.delta(melspec_db, order=2)

        # Собираем 3 признака в одно изображение
        img = np.stack([melspec_db, delta, delta2], axis=-1)
        
        # Min-Max нормализуем значения от 0 до 1
        min_val = img.min()
        max_val = img.max()
        if (max_val - min_val) > 0:
            img = (img - min_val) / (max_val - min_val)
        
        # Изменяем размер до квадратного
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        return img_resized

    except Exception as e:
        print(f"Error: {e}")
        return np.zeros((IMG_SIZE, IMG_SIZE, 3))

# Демонстрация одного примера
print("Демонстрация одного примера:")
df = pd.read_csv(TRAIN_CSV)
sample_file = os.path.join(TRAIN_DIR, df.iloc[0]['fname'])
sample_img = get_features(sample_file)

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1); plt.title("Mel-Spec"); plt.imshow(sample_img[:,:,0])
plt.subplot(1, 3, 2); plt.title("Delta"); plt.imshow(sample_img[:,:,1])
plt.subplot(1, 3, 3); plt.title("Delta-Delta"); plt.imshow(sample_img[:,:,2])
plt.show()



# Считываем тренировочный датасет
df = pd.read_csv(TRAIN_CSV)

# Отделяем только верифицированные данные
df_clean = df[df['manually_verified'] == 1].reset_index(drop=True)
print(f"Используем: {len(df_clean)} верифицированных из {len(df)}")

# Создаем пустые списки для картинок и меток классов
X = []
y = []

# Запускаем цикл обработки файлов с прогресс-баром
print("Предобработка данных:")
for i in tqdm(range(len(df_clean))):
    # Получаем путь к файлу
    row = df_clean.iloc[i]
    file_path = os.path.join(TRAIN_DIR, row['fname'])
    
    # Извлекаем признаки
    img = get_features(file_path)
    
    # Добавляем в массив данных, если картинка не пустая
    if np.sum(img) != 0:
        X.append(img)
        y.append(row['label'])

# Превращаем списки в NumPy массивы
X = np.array(X)

# Кодируем метки в формате One-Hot Encoding
lb = LabelBinarizer()
y_onehot = lb.fit_transform(y)

# Разделение на Train и Validation
X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot)



def build_model(num_classes):
    # Загружаем ResNet50 без верхнего слоя, предобученный на ImageNet
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Разрешаем обучение весов базовой модели
    base_model.trainable = True

    # Собираем итоговую архитектуру
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),                # Сворачиваем карты признаков
        Dropout(0.5),                            # Выключаем половину нейронов для защиты от переобучения
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

model = build_model(y_onehot.shape[1])

# Компилируем модель с метриками и функцией потерь
# Используем низкий learning rate, так как модель уже предобучена
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Выводим структуру модели в консоль
model.summary()



# Добавляем чекпоинты для выбора лучшей модели и уменьшаем шаг обучения, если вышли на плато
checkpoint = ModelCheckpoint("best_audio_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Параметры запуска обучения
BATCH_SIZE = 32
EPOCHS = 20

# Тренируем модель
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, reduce_lr],
    verbose=1
)


# Загружаем лучшие веса
model.load_weights("best_audio_model.keras")

# Делаем тестовый прогон на валидации
y_pred_prob = model.predict(X_val)
y_pred_indices = np.argmax(y_pred_prob, axis=1)
y_val_indices = np.argmax(y_val, axis=1)

acc = accuracy_score(y_val_indices, y_pred_indices)
print(f"Итоговая точность: {acc:.2f}")

# Выводим графики обучения
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.show()



# Загружаем шаблон результирующего файла
test_df = pd.read_csv(SAMPLE_SUBMISSION_CSV)

X_test = []
valid_indices = []

print("Предобработка тестовых данных:")
for i in tqdm(range(len(test_df))):
    fname = test_df.iloc[i]['fname']
    file_path = os.path.join(TEST_DIR, fname)
    
    img = get_features(file_path)
    
    # Собираем батч
    X_test.append(img)
    valid_indices.append(i)

X_test = np.array(X_test)

print("Предсказание...")
# Получаем вероятности для каждого класса
preds = model.predict(X_test, batch_size=32, verbose=1)

top3_indices = np.argsort(preds, axis=1)[:, :-4:-1]

# Собираем результаты в строку через пробел
predicted_labels = []
for i in range(len(top3_indices)):
    indices = top3_indices[i]
    # Преобразуем индексы обратно в названия классов
    labels = [lb.classes_[idx] for idx in indices]
    predicted_labels.append(" ".join(labels))

# Записываем в DataFrame
test_df.loc[valid_indices, 'label'] = predicted_labels

# Сохраняем результат
submission_file = 'submission.csv'
test_df.to_csv(submission_file, index=False)

