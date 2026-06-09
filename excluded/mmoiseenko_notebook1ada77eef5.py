import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Импорт всех необходимых библиотек
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import zipfile
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Проверяем доступность GPU
print("Доступные устройства:", tf.config.list_physical_devices())
print("Используется GPU:", "Да" if tf.test.is_gpu_available() else "Нет")


# Разархивируем train.zip
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

# Создаем директории для структурированного хранения данных
base_dir = 'cats_vs_dogs'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

os.makedirs(base_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Создаем поддиректории для каждого класса
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

for dir_path in [train_cats_dir, train_dogs_dir, validation_cats_dir, 
                 validation_dogs_dir, test_cats_dir, test_dogs_dir]:
    os.makedirs(dir_path, exist_ok=True)


# Функция для разделения исходных данных на train/validation/test
def organize_dataset(source_dir, train_size=0.7, val_size=0.15, test_size=0.15):
    """
    Организует данные в структурированные папки для обучения, валидации и тестирования.
    Исходные данные должны быть в формате: 'cat.0.jpg', 'dog.0.jpg' и т.д.
    """
    cat_images = [f for f in os.listdir(source_dir) if f.startswith('cat')]
    dog_images = [f for f in os.listdir(source_dir) if f.startswith('dog')]
    
    # Разделяем данные для кошек
    cat_train, cat_temp = train_test_split(cat_images, train_size=train_size, random_state=42)
    cat_val, cat_test = train_test_split(cat_temp, train_size=val_size/(val_size+test_size), random_state=42)
    
    # Разделяем данные для собак
    dog_train, dog_temp = train_test_split(dog_images, train_size=train_size, random_state=42)
    dog_val, dog_test = train_test_split(dog_temp, train_size=val_size/(val_size+test_size), random_state=42)
    
    # Копируем файлы в соответствующие директории
    for image in cat_train:
        os.rename(os.path.join(source_dir, image), os.path.join(train_cats_dir, image))
    for image in dog_train:
        os.rename(os.path.join(source_dir, image), os.path.join(train_dogs_dir, image))
    for image in cat_val:
        os.rename(os.path.join(source_dir, image), os.path.join(validation_cats_dir, image))
    for image in dog_val:
        os.rename(os.path.join(source_dir, image), os.path.join(validation_dogs_dir, image))
    for image in cat_test:
        os.rename(os.path.join(source_dir, image), os.path.join(test_cats_dir, image))
    for image in dog_test:
        os.rename(os.path.join(source_dir, image), os.path.join(test_dogs_dir, image))

# Вызываем функцию организации данных (выполнить один раз)
organize_dataset('train')


# Параметры модели
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

# Создаем генераторы данных с аугментацией
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Создаем генераторы
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'  # Это создаст метки с shape (batch_size,)
)

validation_generator = val_test_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("Классы:", train_generator.class_indices)


def create_model_fixed(base_model_type='mobilenet_v2'):
    """
    Создает модель с исправленной архитектурой для совместимости с binary_crossentropy
    """
    
    # Выбираем базовую модель
    if base_model_type == 'mobilenet_v2':
        base_model = applications.MobileNetV2(
            input_shape=IMG_SIZE + (3,),
            include_top=False,
            weights='imagenet'
        )
    elif base_model_type == 'vgg16':
        base_model = applications.VGG16(
            input_shape=IMG_SIZE + (3,),
            include_top=False,
            weights='imagenet'
        )
    else:
        raise ValueError("Поддерживаемые модели: 'mobilenet_v2' или 'vgg16'")
    
    # Замораживаем веса базовой модели
    base_model.trainable = False
    
    # Создаем полную модель с исправлением размерности выхода
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Выходной слой - используем 1 нейрон с sigmoid для бинарной классификации
    # Но теперь мы явно задаем правильную размерность
    outputs = layers.Dense(1, activation='sigmoid', name='output_layer')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Компилируем модель
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',  # Автоматически работает с (batch_size,) и (batch_size, 1)
        metrics=['accuracy']
    )
    
    return model

# Создаем исправленную модель
model = create_model_fixed('mobilenet_v2')
model.summary()


# Callbacks для улучшения обучения
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_cats_vs_dogs_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# Проверяем, что генераторы работают правильно
print("Проверка генераторов:")
for images_batch, labels_batch in train_generator:
    print(f"Размер батча изображений: {images_batch.shape}")
    print(f"Размер батча меток: {labels_batch.shape}")
    print(f"Диапазон значений изображений: [{images_batch.min():.3f}, {images_batch.max():.3f}]")
    print(f"Примеры меток: {labels_batch[:5]}")
    break

# Первый этап: обучение только новых слоев
print("\n=== ПЕРВЫЙ ЭТАП: Обучение только новых слоев ===")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=2,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=callbacks,
    verbose=1
)


def fine_tune_model_fixed(model, base_model_type='mobilenet_v2'):
    """
    Размораживает часть слоев базовой модели для тонкой настройки
    """
    
    base_model = model.layers[1]  # Базовая модель - второй слой (после Input)
    base_model.trainable = True
    
    # Размораживаем только верхние слои
    if base_model_type == 'mobilenet_v2':
        fine_tune_at = 100  # Размораживаем последние слои
    else:
        fine_tune_at = len(base_model.layers) // 2
    
    # Замораживаем первые слои
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Перекомпилируем с меньшим learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Разморожено слоев: {sum([l.trainable for l in base_model.layers])}")
    return model

# Применяем fine-tuning
print("\n=== ВТОРОЙ ЭТАП: Fine-tuning ===")
model = fine_tune_model_fixed(model, 'mobilenet_v2')

# Продолжаем обучение с fine-tuning
history_fine = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=2,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=callbacks,
    verbose=1
)


# Визуализация и оценка
def plot_training_history(history, fine_history=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    if fine_history is not None:
        # Объединяем историю
        acc = history.history['accuracy'] + fine_history.history['accuracy']
        val_acc = history.history['val_accuracy'] + fine_history.history['val_accuracy']
        loss = history.history['loss'] + fine_history.history['loss']
        val_loss = history.history['val_loss'] + fine_history.history['val_loss']
        epochs_range = range(1, len(acc) + 1)
    else:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(1, len(acc) + 1)
    
    # График точности
    ax1.plot(epochs_range, acc, 'b-', label='Training Accuracy')
    ax1.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # График потерь
    ax2.plot(epochs_range, loss, 'b-', label='Training Loss')
    ax2.plot(epochs_range, val_loss, 'r-', label='Validation Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_training_history(history, history_fine)

# Оценка модели на тестовых данных
print("\n=== ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ ===")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Вычисляем Log Loss
test_predictions = model.predict(test_generator)
test_true_labels = test_generator.classes

# Исправляем размерности
test_predictions_flat = test_predictions.flatten()  # Преобразуем (n, 1) в (n,)

# Вычисляем binary crossentropy (log loss) с правильными размерностями
log_loss = keras.losses.binary_crossentropy(
    tf.convert_to_tensor(test_true_labels, dtype=tf.float32),
    tf.convert_to_tensor(test_predictions_flat, dtype=tf.float32)
)
log_loss = tf.reduce_mean(log_loss).numpy()

print(f"Log Loss: {log_loss:.4f}")

if log_loss < 0.3:
    print("УСПЕХ: Log Loss < 0.3 достигнут!")
else:
    print(f"Log Loss = {log_loss:.4f} > 0.3")
    print("Необходимо улучшение!")


# Предсказания и матрица ошибок
test_predictions = model.predict(test_generator)
predicted_classes = (test_predictions > 0.5).astype("int32").flatten()
true_classes = test_generator.classes

# Матрица ошибок
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Cats', 'Dogs'], 
            yticklabels=['Cats', 'Dogs'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Отчет по классификации
print("\nОтчет по классификации:")
print(classification_report(true_classes, predicted_classes, 
                          target_names=['Cats', 'Dogs']))

# Визуализируем примеры предсказаний
def plot_sample_predictions(num_samples=12):
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    axes = axes.ravel()
    
    test_files = test_generator.filenames
    
    for i in range(num_samples):
        # Берем случайные примеры
        idx = np.random.randint(0, len(test_files))
        
        # Загружаем изображение
        img_path = os.path.join(test_dir, test_files[idx])
        img = plt.imread(img_path)
        
        # Предсказание
        actual_label = 'cat' if true_classes[idx] == 0 else 'dog'
        predicted_label = 'cat' if predicted_classes[idx] == 0 else 'dog'
        confidence = test_predictions[idx][0] if predicted_label == 'dog' else 1 - test_predictions[idx][0]
        
        # Отображаем
        axes[i].imshow(img)
        color = 'green' if actual_label == predicted_label else 'red'
        axes[i].set_title(f'True: {actual_label}\nPred: {predicted_label}\nConf: {confidence:.3f}', 
                         color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

plot_sample_predictions()


# Разархивируем test.zip 
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

# Проверяем содержимое test директории
extracted_test_dir = 'test'  # Папка, созданная после распаковки test.zip
test_files = os.listdir(extracted_test_dir)
print(f"Найдено {len(test_files)} тестовых изображений")
print("Примеры файлов:", test_files[:5])

# Создаем генератор для тестовых данных конкурса
competition_test_datagen = ImageDataGenerator(rescale=1./255)

competition_test_generator = competition_test_datagen.flow_from_directory(
    directory='./',  # Указываем корневую директорию
    classes=[extracted_test_dir],  # Указываем поддиректорию с изображениями
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,  # Важно: нет меток классов
    shuffle=False     # Важно: сохраняем порядок файлов для соответствия с sample_submission
)

# Проверяем генератор
print(f"Всего тестовых изображений для конкурса: {competition_test_generator.samples}")

# Делаем предсказания на тестовых данных конкурса
print("\n=== ПРЕДСКАЗАНИЯ ДЛЯ KAGGLE ===")
competition_predictions = model.predict(competition_test_generator, 
                                      verbose=1, 
                                      steps=len(competition_test_generator))

# competition_predictions содержит вероятности класса 1 (собаки)
# Для Kaggle нам нужно именно это - вероятность что изображение собаки
kaggle_predictions = competition_predictions.flatten()

print(f"Форма предсказаний: {kaggle_predictions.shape}")
print(f"Примеры предсказаний: {kaggle_predictions[:10]}")
print(f"Диапазон предсказаний: [{kaggle_predictions.min():.3f}, {kaggle_predictions.max():.3f}]")

# Получаем имена файлов в том же порядке, что и генератор
test_filenames = competition_test_generator.filenames

print(f"\nПримеры имен файлов: {test_filenames[:5]}")

# Извлекаем только числовые ID из имен файлов (убираем расширение .jpg)
# Имена файлов выглядят как 'test/1234.jpg', нам нужно '1234'
test_ids = [os.path.splitext(os.path.basename(f))[0] for f in test_filenames]

print(f"Примеры ID: {test_ids[:5]}")

# Создаем DataFrame для submission
submission_df = pd.DataFrame({
    'id': test_ids,
    'label': kaggle_predictions
})

print("\nПервые 10 строк submission файла:")
print(submission_df.head(10))

# Проверяем статистику предсказаний
print(f"\nСтатистика предсказаний:")
print(f"Средняя вероятность: {submission_df['label'].mean():.4f}")
print(f"Медианная вероятность: {submission_df['label'].median():.4f}")
print(f"Стандартное отклонение: {submission_df['label'].std():.4f}")

# Визуализируем распределение предсказаний
plt.figure(figsize=(10, 6))
plt.hist(submission_df['label'], bins=50, alpha=0.7, edgecolor='black')
plt.title('Распределение предсказанных вероятностей для тестового набора')
plt.xlabel('Вероятность (собака)')
plt.ylabel('Количество изображений')
plt.grid(True, alpha=0.3)
plt.show()

# Сохраняем submission файл
submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"\n✅ Submission файл сохранен как: {submission_filename}")
print(f"Размер файла: {os.path.getsize(submission_filename)} байт")

# Проверяем содержимое сохраненного файла
saved_submission = pd.read_csv(submission_filename)
print(f"\nПроверка сохраненного файла:")
print(f"Количество строк: {len(saved_submission)}")
print(f"Первые 5 строк:")
print(saved_submission.head())

# Дополнительная проверка: сравниваем с sample_submission
sample_submission_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv'
if os.path.exists(sample_submission_path):
    sample_submission = pd.read_csv(sample_submission_path)
    print(f"\n=== СРАВНЕНИЕ С SAMPLE SUBMISSION ===")
    print(f"Sample submission размер: {len(sample_submission)} строк")
    print("Sample submission первые 5 строк:")
    print(sample_submission.head())
    
    # Проверяем, что мои ID совпадают с sample (в том же порядке)
    if len(saved_submission) == len(sample_submission):
        print("Количество строк совпадает с sample submission")
    else:
        print("Количество строк НЕ совпадает с sample submission")
        
    # Проверяем диапазон предсказаний
    print(f"Наш диапазон предсказаний: [{saved_submission['label'].min():.3f}, {saved_submission['label'].max():.3f}]")
    print(f"Sample диапазон: [{sample_submission['label'].min():.3f}, {sample_submission['label'].max():.3f}]")

# Создаем файл с метаинформацией о модели
with open('model_info.txt', 'w') as f:
    f.write(f"Модель: Cats vs Dogs Classifier\n")
    f.write(f"Базовая архитектура: MobileNetV2\n")
    f.write(f"Размер изображения: {IMG_SIZE}\n")
    f.write(f"Final Train Accuracy: {history.history['accuracy'][-1]:.4f}\n")
    f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Log Loss: {log_loss:.4f}\n")
    f.write(f"Количество тестовых изображений: {len(submission_df)}\n")
    f.write(f"Дата создания: {pd.Timestamp.now()}\n")

print("\nВСЕ ГОТОВО ДЛЯ ОТПРАВКИ НА KAGGLE!")

