# Проверка наличия и пути к файлу .json
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# /kaggle/input/key-json3/kaggle (3).json


import os
import shutil

# Создать директорию /root/.kaggle
os.makedirs('/root/.kaggle', exist_ok=True)

# Путь к  файлу 
source_path = '/kaggle/input/key-json3/kaggle (3).json'  

# Переместить и переименовать
shutil.copy(source_path, '/root/.kaggle/kaggle.json')  # Используем copy, так как input только для чтения
os.chmod('/root/.kaggle/kaggle.json', 0o600)

# Проверить наличие
print("Файл kaggle.json:", os.path.exists('/root/.kaggle/kaggle.json'))

# Проверить содержимое
import json
with open('/root/.kaggle/kaggle.json', 'r') as f:
    print("Содержимое kaggle.json:", json.load(f))


from kaggle.api.kaggle_api_extended import KaggleApi

# Аутентификация API
api = KaggleApi()
api.authenticate()
print("Kaggle API аутентифицирован")

# Проверка соревнования
print("Проверка соревнования skillbox-computer-vision-project...")
competitions = api.competitions_list(search='skillbox-computer-vision-project')
if competitions:
    for comp in competitions:
        print(f"Найдено соревнование: {comp.id} ({comp.title})")
else:
    print("Соревнование с ID 'skillbox-computer-vision-project' не найдено. Проверьте ID.")


# Исправить путь конфигурации
# Сообщение об ошибке упоминало /root/.config/kaggle.
# Сбросьте переменную окружения:
os.environ['KAGGLE_CONFIG_DIR'] = '/root/.kaggle'
print("KAGGLE_CONFIG_DIR:", os.environ.get('KAGGLE_CONFIG_DIR'))


# Импорт библиотек
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

from PIL import Image
import gdown
import zipfile
from sklearn.preprocessing import LabelEncoder
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint

# import os
# import shutil
# from kaggle.api.kaggle_api_extended import KaggleApi


# Отключаем mixed precision (использование float16/float32) в пользу 
# чистого float32 для стабильности
mixed_precision.set_global_policy('float32')


# Проверка всех соревнований
print("Список доступных соревнований:")
competitions = api.competitions_list()
for comp in competitions:
    print(f"ID: {comp.id}, Title: {comp.title}")


# 1. Проверка подключения GPU и вывод информации о видеокарте
print("Доступные устройства:", tf.config.list_physical_devices('GPU'))
!nvidia-smi


# 3. Загрузка и распаковка тренировочных изображений
# Загрузка тренировочных данных с Google Drive
train_url = "https://drive.google.com/uc?id=1TG9P5B2k3eTbC4XDxDmEc07dyAORPC16"
print("Загрузка тренировочных данных...")
gdown.download(train_url, "/kaggle/working/train_data.zip", quiet=False)

# Проверка файла
if not os.path.exists("/kaggle/working/train_data.zip"):
    raise FileNotFoundError("train_data.zip не загружен")

print("Распаковка тренировочных данных...")
with zipfile.ZipFile("/kaggle/working/train_data.zip", 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/train_data")

# Проверка распаковки
if not os.path.exists("/kaggle/working/train_data/train/"):
    raise FileNotFoundError("Директория /kaggle/working/train_data/train/ не создана")

# Установка путей
train_image_dir = "/kaggle/working/train_data/train/"
train_csv_path = "/kaggle/input/skillbox-computer-vision-project/train.csv"
test_image_dir = "/kaggle/input/my-dataset3/test_kaggle/"
submission_csv_path = "/kaggle/input/skillbox-computer-vision-project/sample_submission.csv"

# Проверка файлов и директорий
if not os.path.exists(train_csv_path) or not os.path.exists(submission_csv_path):
    raise FileNotFoundError("train.csv or sample_submission.csv not found")
if not os.path.exists(train_image_dir) or not os.path.exists(test_image_dir):
    raise FileNotFoundError("Image directories not found")
print(f"Директории {train_image_dir} и {test_image_dir} найдены.")

# Диагностика
print("Содержимое /kaggle/input/skillbox-computer-vision-project/:")
!ls -lh /kaggle/input/skillbox-computer-vision-project/
print("Первые 5 файлов в test_kaggle:")
!ls -lh /kaggle/input/my-dataset3/test_kaggle/ | head -n 5

# Загрузка CSV
df_train = pd.read_csv(train_csv_path)
df_sample_submission = pd.read_csv(submission_csv_path)
print("Train shape:", df_train.shape)
print("Sample submission shape:", df_sample_submission.shape)

# Проверка путей
print("Sample submission image paths (before correction):")
print(df_sample_submission['image_path'].head())

# Корректировка путей к изображениям для соответствия структуре каталогов Kaggle
df_train['image_path'] = df_train['image_path'].apply(
    lambda x: os.path.join(train_image_dir, x.replace('./train/', ''))
)
df_sample_submission['image_path'] = df_sample_submission['image_path'].apply(
    lambda x: os.path.join(test_image_dir, x)
)

# Проверка путей
print("Sample submission image paths (after correction):")
print(df_sample_submission['image_path'].head())

# Проверка файлов
for path in df_sample_submission['image_path'].head():
    if not os.path.exists(path):
        print(f"File not found: {path}")



# 4. Подготовка данных
# Кодирование меток эмоций в числовой формат
label_encoder = LabelEncoder()
df_train = df_train.dropna(subset=['emotion'])
df_train['emotion_encoded'] = label_encoder.fit_transform(df_train['emotion'])
num_classes = len(label_encoder.classes_)
print("Classes:", label_encoder.classes_)


# Разделение данных с учетом стратификации
df_train_split, df_val_split = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['emotion'])



# Аугментация и предобработка
# Используются стандартные преобразования: повороты, сдвиги, отражения
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,  # Добавьте vertical flip
    fill_mode='nearest',
    zoom_range=0.3,
    shear_range=0.2,
    brightness_range=[0.8, 1.2]
)


# Создание генераторов данных 
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

img_size = (256, 256)
batch_size = 32 #8

# Создание генератора для тренировочных данных с загрузкой из DataFrame
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train_split,
    x_col='image_path',
    y_col='emotion',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=df_val_split,
    x_col='image_path',
    y_col='emotion',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_sample_submission,
    x_col='image_path',
    y_col=None,
    target_size=(256, 256),
    batch_size=8,
    class_mode=None,
    shuffle=False
)


# Балансировка классов
class_indices = train_generator.class_indices  # Словарь {emotion: index}
classes = np.unique(df_train['emotion'])
class_weights = compute_class_weight('balanced', classes=classes, y=df_train['emotion'])
class_weight_dict = dict(zip([class_indices[cls] for cls in classes], class_weights))
print("Class weights:", class_weight_dict)


# 6. Создание модели с одним слоем
# Использование предобученной EfficientNetB2 (без верхних слоев)
# Замораживание весов базовой модели для первого этапа обучения
base_model = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base_model.trainable = False

# Добавление собственных слоев поверх базовой модели
# Используются BatchNormalization и Dropout для регуляризации
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
    Dropout(0.5),
    Dense(9, activation='softmax')  # 9 классов
])

optimizer = Adam(learning_rate=2e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# # 6. Создание модели с двумя слоями
# # Использование предобученной EfficientNetB2 (без верхних слоев)
# # Замораживание весов базовой модели для первого этапа обучения
# base_model = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
# base_model.trainable = False

# # Добавление собственных слоев поверх базовой модели
# # Используются BatchNormalization и Dropout для регуляризации
# model = Sequential([
#     base_model,
#     GlobalAveragePooling2D(),
#     BatchNormalization(),
#     Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
#     Dropout(0.6),
#     BatchNormalization(),
#     Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
#     Dropout(0.4),
#     Dense(num_classes, activation='softmax')
# ])

# optimizer = Adam(learning_rate=1e-4) #1e-6
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()



print("Starting transfer learning...")
history_transfer = model.fit(
    train_generator,
    epochs= 5, #10,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1, mode='max'),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=1e-8, verbose=1, mode='max'),
        ModelCheckpoint('/kaggle/working/best_model_transfer.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ],
    verbose=1
)



# 7. Fine-Tuning
# Размораживание последних 10 слоев базовой модели для тонкой настройки
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=3e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели с callback'ами:
# EarlyStopping для остановки при ухудшении качества
# ReduceLROnPlateau для динамического изменения learning rate
# NaNLogger для отслеживания NaN
print("Starting fine-tuning...")
history_finetune = model.fit(
    train_generator,
    epochs= 10, #25,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1, mode='max'),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=1e-8, verbose=1, mode='max'),
        ModelCheckpoint('/kaggle/working/best_model_finetune.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ],
    verbose=1
)


# 8. Графики. Визуализация процесса обучения (loss и accuracy)
# Объединяем данные из history_transfer и history_finetune
train_loss = history_transfer.history['loss'] + history_finetune.history['loss']
val_loss = history_transfer.history['val_loss'] + history_finetune.history['val_loss']
train_accuracy = history_transfer.history['accuracy'] + history_finetune.history['accuracy']
val_accuracy = history_transfer.history['val_accuracy'] + history_finetune.history['val_accuracy']

# График loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Train Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='orange')
plt.axvline(x=5, color='black', linestyle='--', label='Fine-tuning start')  # Для 5+10
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('/kaggle/working/training_loss.png')
plt.show()
plt.close()

# График accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracy, label='Train Accuracy', color='green')
plt.plot(val_accuracy, label='Validation Accuracy', color='red')
plt.axvline(x=5, color='black', linestyle='--', label='Fine-tuning start')  # Для 5+10
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('/kaggle/working/training_accuracy.png')
plt.show()
plt.close()


# 8. Предсказания
print("Making predictions on test set...")
test_generator.reset()  # Сброс генератора
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

# Маппинг индексов классов обратно в имена эмоций
index_to_emotion = {v: k for k, v in class_indices.items()}
predicted_emotions = [index_to_emotion[idx] for idx in predicted_classes]

# Извлечение имени файла из полного пути
submission_image_paths = [os.path.basename(path) for path in df_sample_submission['image_path']]

# Создание submission.csv по образцу
submission = pd.DataFrame({
    'image_path': submission_image_paths,
    'emotion': predicted_emotions
})

# Проверка формата
print("Проверка submission.csv...")
print("Строк в submission:", len(submission))
print("Строк в df_sample_submission:", len(df_sample_submission))
print("Уникальные эмоции:", set(predicted_emotions))
print("Первые 5 строк submission:")
print(submission.head())
print("Колонки submission:", list(submission.columns))

# # Проверка соответствия ожидаемым эмоциям
# if not set(predicted_emotions).issubset(expected_emotions):
#     print("Предупреждение: найдены недопустимые эмоции:", set(predicted_emotions) - expected_emotions)

# # Проверка sample_submission.csv
# try:
#     sample_submission = pd.read_csv('/kaggle/input/<competition-name>/sample_submission.csv')  # Уточните путь
#     print("Sample submission первые 5 строк:")
#     print(sample_submission.head())
#     print("Колонки sample_submission:", list(sample_submission.columns))
#     if list(submission.columns) != list(sample_submission.columns):
#         print("Ошибка: колонки submission не совпадают с sample_submission!")
#     if len(submission) != len(sample_submission):
#         print("Ошибка: количество строк в submission не совпадает с sample_submission!")
# except FileNotFoundError:
#     print("sample_submission.csv не найден. Уточните путь, например, /kaggle/input/<competition-name>/sample_submission.csv")

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file saved to /kaggle/working/submission.csv")


# Визуализация тестовых изображений
print("Visualizing test images with predicted emotions...")
test_generator.reset()
test_images = next(test_generator)  # Для предсказаний
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
predicted_emotions = [index_to_emotion[idx] for idx in predicted_classes]

# Загружаем исходные изображения напрямую из df_sample_submission
test_paths = df_sample_submission['image_path'].iloc[test_generator.index_array[:5]].values

plt.figure(figsize=(15, 5))
for i, img_path in enumerate(test_paths[:5]):
    img = plt.imread(img_path)  # Читаем изображение напрямую
    if img.shape[-1] != 3:  # Если grayscale, преобразуем в RGB
        img = np.stack([img] * 3, axis=-1)
    if img.dtype == np.float32 or img.max() <= 1.0:  # Если нормализовано в [0, 1]
        img = (img * 255).astype(np.uint8)
    
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(f"Pred: {predicted_emotions[i]}")
    plt.axis('off')
plt.savefig('/kaggle/working/test_predictions.png')
plt.show()
plt.close()


# # Отправка результатов на Kaggle
# # Очистка временных файлов

# # 11. Удаление весов
# !rm -f /kaggle/working/*.h5
# print("Removed .h5 files")

# # 12. Отправка
# print("Submitting to Kaggle...")
# !kaggle competitions submit -c skillbox-computer-vision-project -f /kaggle/working/submission.csv -m "EfficientNetB2, fine-tuning"

# # 13. Проверка результатов
# print("Checking submission results...")
# !kaggle competitions submissions -c skillbox-computer-vision-project

# # 14. Проверка файлов перед очисткой
# print("Файлы в /kaggle/working/ перед очисткой:")
# !ls -lh /kaggle/working/

# # 15. Очистка
# !rm -rf /kaggle/working/train_data
# print("Cleaned temporary files")

