# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


! pip install -U albumentations tensorflow opencv-python


# Ячейка 1: Импорт библиотек и настройка окружения
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras import layers, models
from tensorflow.keras import callbacks as kcallbacks
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# настройка GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f" Обнаружено GPU: {len(gpus)} устройств")
    except RuntimeError as e:
        print(f" Ошибка настройки GPU: {e}")
else:
    print(" GPU не обнаружены, будет использоваться CPU")


# загрузка и анализ данных
try:
    train_df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')
    train_df['id_code'] = train_df['id_code'] + '.png'
    print(f" Данные успешно загружены")
    print(f"  - Размер датасета: {train_df.shape}")
    print(f"  - Колонки: {list(train_df.columns)}")
except FileNotFoundError:
    print("- Файл train.csv не найден!")
    # показать доступные файлы
    print("\nДоступные файлы в input:")
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(f"  {os.path.join(dirname, filename)}")
    raise

# анализ баланса классов
class_distribution = train_df['diagnosis'].value_counts().sort_index()
for class_id, count in class_distribution.items():
    percentage = (count / len(train_df)) * 100
    print(f"  Класс {class_id}: {count:4d} изображений ({percentage:.1f}%)")

# визуализация распределения
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
bars = plt.bar(class_distribution.index, class_distribution.values)
plt.title('Распределение классов', fontsize=14, fontweight='bold')
plt.xlabel('Класс (степень ретинопатии)', fontsize=12)
plt.ylabel('Количество изображений', fontsize=12)
plt.xticks(range(5))
plt.grid(True, alpha=0.3)

# добавление значений на столбцы
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.subplot(1, 2, 2)
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
plt.pie(class_distribution.values, labels=class_distribution.index, 
        autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Процентное распределение', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()


# разделение данных на train/val

# стратифицированное разделение для сохранения пропорций классов
train_df, val_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=42,
    stratify=train_df['diagnosis']
)

print(f" Данные успешно разделены:")
print(f"  - Тренировочный набор: {len(train_df):5d} изображений ({len(train_df)/len(train_df)*100:.1f}%)")
print(f"  - Валидационный набор: {len(val_df):5d} изображений ({len(val_df)/len(train_df)*100:.1f}%)")

# проверка распределения классов в наборах
print("Тренировочный набор:")
train_class_dist = train_df['diagnosis'].value_counts().sort_index()
for class_id in range(5):
    count = train_class_dist.get(class_id, 0)
    percentage = (count / len(train_df)) * 100
    print(f"  Класс {class_id}: {count:4d} ({percentage:.1f}%)")

print("\nВалидационный набор:")
val_class_dist = val_df['diagnosis'].value_counts().sort_index()
for class_id in range(5):
    count = val_class_dist.get(class_id, 0)
    percentage = (count / len(val_df)) * 100
    print(f"  Класс {class_id}: {count:4d} ({percentage:.1f}%)")

#  визуализация
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(train_class_dist.index, train_class_dist.values, color='steelblue', alpha=0.7)
axes[0].set_title('Распределение в тренировочном наборе', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Класс')
axes[0].set_ylabel('Количество')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(range(5))

axes[1].bar(val_class_dist.index, val_class_dist.values, color='lightcoral', alpha=0.7)
axes[1].set_title('Распределение в валидационном наборе', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Класс')
axes[1].set_ylabel('Количество')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(range(5))

plt.tight_layout()
plt.show()


# функции предобработки изображений

def preprocess_retina_image(image_path, size=256, augment=False):
    """
    Улучшенная предобработка изображений глазного дна
    Включает удаление черных границ, CLAHE, изменение размера и нормализацию
    """
    # проверка существования файла
    if not os.path.exists(image_path):
        print(f"  ⚠ Файл не найден: {os.path.basename(image_path)}")
        return np.zeros((size, size, 3), dtype=np.float32)
    
    try:
        # загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ⚠ Не удалось загрузить: {os.path.basename(image_path)}")
            return np.zeros((size, size, 3), dtype=np.float32)
        
        # конвертация в RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # удаление черных границ
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # добавляем небольшой отступ
            pad = 5
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(image.shape[1] - x, w + 2*pad)
            h = min(image.shape[0] - y, h + 2*pad)
            
            if w > 10 and h > 10:  # проверка минимального размера
                image = image[y:y+h, x:x+w]
        
        # улучшение контраста с помощью CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # изменение размера с сохранением пропорций
        h, w = image.shape[:2]
        scale = size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        image = cv2.resize(image, (new_w, new_h))
        
        # добавление паддинга для квадрата
        pad_h = (size - new_h) // 2
        pad_w = (size - new_w) // 2
        
        image = cv2.copyMakeBorder(
            image, 
            pad_h, size - new_h - pad_h,
            pad_w, size - new_w - pad_w,
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )
        
        # нормализация
        image = image.astype(np.float32) / 255.0
        
        # аугментация (только для тренировочных данных)
        if augment:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, 
                    contrast_limit=0.1, 
                    p=0.5
                ),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                    A.MotionBlur(blur_limit=(3, 7), p=0.3),
                ], p=0.2),
            ])
            image = transform(image=image)['image']
        
        return image
        
    except Exception as e:
        print(f"  Ошибка обработки {os.path.basename(image_path)}: {str(e)[:50]}...")
        return np.zeros((size, size, 3), dtype=np.float32)

print(" Функции предобработки определены")
print("  - preprocess_retina_image - основная функция обработки")
print("  - Поддерживает аугментацию и CLAHE контраст")

# тестирование функции на примере
# проверяем путь к изображениям
train_path = '/kaggle/input/aptos2019-blindness-detection/train_images'
test_image_path = os.path.join(train_path, train_df.iloc[0]['id_code'])

print(f"  Путь к изображениям: {train_path}")
print(f"  Пример изображения: {test_image_path}")
print(f"  Файл существует: {os.path.exists(test_image_path)}")

if os.path.exists(test_image_path):
    # обработка без аугментации
    image_no_aug = preprocess_retina_image(test_image_path, augment=False)
    
    # обработка с аугментацией
    image_aug = preprocess_retina_image(test_image_path, augment=True)
    
    print(f" Тест успешен")
    print(f"    Размер изображения: {image_no_aug.shape}")
    print(f"    Диапазон значений: [{image_no_aug.min():.3f}, {image_no_aug.max():.3f}]")
    print(f"    Тип данных: {image_no_aug.dtype}")
    
    # визуализация результатов обработки
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].imshow(image_no_aug)
    axes[0].set_title('Без аугментации', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(image_aug)
    axes[1].set_title('С аугментацией', fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle('Результаты предобработки изображения', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print(" Тестовый файл не найден!")


# создание генератора данных
class RetinaDataGenerator(tf.keras.utils.Sequence):
    """
    Генератор данных для изображений глазного дна
    Поддерживает кэширование, аугментацию и балансировку классов
    """
    
    def __init__(self, df, base_path, batch_size=16, augment=False, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.base_path = base_path
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.image_cache = {}  # кэш для ускорения загрузки
        self.on_epoch_end()
        
        print(f"  Создан генератор с параметрами:")
        print(f"    • Изображений: {len(self.df)}")
        print(f"    • Batch size: {batch_size}")
        print(f"    • Аугментация: {'Да' if augment else 'Нет'}")
        print(f"    • Перемешивание: {'Да' if shuffle else 'Нет'}")
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        
        images = []
        labels = []
        
        for _, row in batch_df.iterrows():
            img_key = row['id_code']
            
            # используем кэш для неаугментированных изображений
            if img_key in self.image_cache and not self.augment:
                image = self.image_cache[img_key]
            else:
                img_path = os.path.join(self.base_path, img_key)
                image = preprocess_retina_image(img_path, augment=self.augment)
                if not self.augment:  # кэшируем только оригинальные
                    self.image_cache[img_key] = image
            
            images.append(image)
            labels.append(row['diagnosis'])
        
        # преобразуем в массивы
        images_array = np.array(images, dtype=np.float32)
        labels_array = tf.keras.utils.to_categorical(labels, num_classes=5)
        
        return images_array, labels_array
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def show_batch_info(self, batch_idx=0):
        """Показать информацию о батче"""
        images, labels = self[batch_idx]
        
        print(f"\n ИНФОРМАЦИЯ О БАТЧЕ {batch_idx}:")
        print(f"  Размер батча: {len(images)} изображений")
        print(f"  Размер изображения: {images[0].shape}")
        print(f"  Диапазон значений пикселей: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Распределение классов в батче:")
        
        class_counts = np.bincount(np.argmax(labels, axis=1))
        for class_id, count in enumerate(class_counts):
            if count > 0:
                print(f"    Класс {class_id}: {count} изображений")

print(" Класс RetinaDataGenerator определен")

# тестирование генератора

# создаем тестовые генераторы
train_gen = RetinaDataGenerator(
    train_df.iloc[:100],  # берем только 100 для теста
    train_path,
    batch_size=8,
    augment=False,
    shuffle=True
)

# получаем первый батч
test_batch_idx = 0
images, labels = train_gen[test_batch_idx]

# выводим информацию
train_gen.show_batch_info(test_batch_idx)

# визуализация батча
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for i in range(min(8, len(images))):
    ax = axes[i]
    ax.imshow(images[i])
    true_class = np.argmax(labels[i])
    ax.set_title(f'Класс: {true_class}', fontweight='bold')
    ax.axis('off')

plt.suptitle('Примеры изображений из батча (без аугментации)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# создание и компиляция модели
def build_retinopathy_model(input_shape=(256, 256, 3)):
    """
    Создание модели для классификации диабетической ретинопатии
    на основе EfficientNetB3 с кастомными слоями
    """
    
    print(" Строим архитектуру модели...")
    
    # загрузка предобученной базовой модели
    base_model = tf.keras.applications.EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        pooling=None
    )
    
    # замораживаем веса базовой модели (на первых эпохах)
    base_model.trainable = False
    print(f"  • Базовая модель: EfficientNetB3")
    print(f"  • Заморожено слоев: {len(base_model.layers)}")
    
    # создаем кастомные слои
    inputs = tf.keras.Input(shape=input_shape)
    
    # добавляем аугментацию как часть модели
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomContrast(0.1)(x)
    
    # предобработка для EfficientNet
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    
    # пропускаем через базовую модель
    x = base_model(x)
    
    # глобальный средний пулинг
    x = layers.GlobalAveragePooling2D()(x)
    
    # регуляризация
    x = layers.Dropout(0.5)(x)
    
    # полносвязные слои
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # выходной слой
    outputs = layers.Dense(5, activation='softmax')(x)
    
    # создаем модель
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Retinopathy_Classifier")
    
    print(f"  • Всего слоев в модели: {len(model.layers)}")
    print(f"  • Параметры модели: {model.count_params():,}")
    
    return model

# создаем модель
print("\n СОЗДАНИЕ МОДЕЛИ:")
model = build_retinopathy_model()

# выводим архитектуру
print("\n АРХИТЕКТУРА МОДЕЛИ:")
model.summary()

# компиляция модели
print("\n КОМПИЛЯЦИЯ МОДЕЛИ:")

# Learning rate schedule
#lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
#    initial_learning_rate=1e-4,
#    first_decay_steps=500,
#    t_mul=2.0,
#    m_mul=0.5,
#    alpha=1e-6
#)

# оптимизатор
optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

# метрики
metrics = [
    'accuracy',
    tf.keras.metrics.AUC(name='auc', multi_label=True),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
]

# компиляция
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=metrics
)

print(" Модель успешно скомпилирована")
print(f"  • Оптимизатор: Adam с CosineDecayRestarts")
print(f"  • Начальный LR: {1e-4}")
print(f"  • Функция потерь: categorical_crossentropy")
print(f"  • Метрики: {len(metrics)} метрик")

# проверка возможности предсказания
print("\n ТЕСТ ПРЕДСКАЗАНИЯ МОДЕЛИ:")
test_input = np.random.randn(2, 256, 256, 3).astype(np.float32)
test_output = model.predict(test_input, verbose=0)

print(f"  • Вход: {test_input.shape}")
print(f"  • Выход: {test_output.shape}")
print(f"  • Сумма вероятностей по классам: {test_output.sum(axis=1)}")
print(" Модель работает корректно")


# расчет весов классов и создание генераторов

# расчет весов классов для балансировки

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['diagnosis']),
    y=train_df['diagnosis']
)

# преобразуем в словарь
class_weights_dict = {i: float(w) for i, w in enumerate(class_weights)}

# выводим информацию
for class_id, weight in class_weights_dict.items():
    original_count = train_df[train_df['diagnosis'] == class_id].shape[0]
    print(f"  Класс {class_id}: {original_count:4d} изображений → вес {weight:.2f}")

# визуализация весов
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(class_weights_dict.keys(), class_weights_dict.values(), color='skyblue')
ax.set_xlabel('Класс', fontsize=12)
ax.set_ylabel('Вес класса', fontsize=12)
ax.set_title('Веса классов для балансировки', fontsize=14, fontweight='bold')
ax.set_xticks(range(5))
ax.grid(True, alpha=0.3, axis='y')

# добавляем значения на столбцы
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# создание генераторов данных
train_gen = RetinaDataGenerator(
    df=train_df,
    base_path=train_path,
    batch_size=16,  # уменьшенный размер батча для стабильности
    augment=True,   # аугментация для тренировки
    shuffle=True
)

val_gen = RetinaDataGenerator(
    df=val_df,
    base_path=train_path,
    batch_size=16,
    augment=False,  # без аугментации для валидации
    shuffle=False   # без перемешивания для стабильной оценки
)

print(" Генераторы созданы:")
print(f"  • Тренировочный генератор: {len(train_gen)} батчей")
print(f"  • Валидационный генератор: {len(val_gen)} батчей")
print(f"  • Размер батча: 16 изображений")


# тестовый батч из тренировочного генератора
train_images, train_labels = train_gen[0]
val_images, val_labels = val_gen[0]

print("Тренировочный генератор:")
print(f"  • Размер батча: {train_images.shape}")
print(f"  • Диапазон значений: [{train_images.min():.3f}, {train_images.max():.3f}]")

print("\nВалидационный генератор:")
print(f"  • Размер батча: {val_images.shape}")
print(f"  • Диапазон значений: [{val_images.min():.3f}, {val_images.max():.3f}]")

# сравнение распределения классов
train_classes = np.argmax(train_labels, axis=1)
val_classes = np.argmax(val_labels, axis=1)

print("\nРаспределение классов в первом батче:")
for class_id in range(5):
    train_count = np.sum(train_classes == class_id)
    val_count = np.sum(val_classes == class_id)
    print(f"  Класс {class_id}: train={train_count}, val={val_count}")


# обучение модели

callbacks = [
    # сохранение лучшей модели
    kcallbacks.ModelCheckpoint(
        filepath='best_model_retinopathy.keras',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    
    # ранняя остановка
    kcallbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    ),
    
    # динамическое изменение learning rate
    kcallbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),

    kcallbacks.LearningRateScheduler(
        lambda epoch: 1e-4 * (0.5 ** (epoch // 10)),  # уменьшаем LR каждые 10 эпох
        verbose=1
    ),
    
    # логирование в CSV
    kcallbacks.CSVLogger(
        'training_history.csv',
        separator=',',
        append=False
    ),
    
    # TensorBoard (опционально)
    # kcallbacks.TensorBoard(
    #     log_dir='./logs',
    #     histogram_freq=1,
    #     write_graph=True,
    #     write_images=True
    # )
]

print(" Callback'ы настроены:")
for i, callback in enumerate(callbacks, 1):
    print(f"  {i}. {callback.__class__.__name__}")

# обучение модели
print("\n ЗАПУСК ОБУЧЕНИЯ:")
print(f"  • Эпох: 30")
print(f"  • Batch size: 16")
print(f"  • Размер тренировочного набора: {len(train_df)}")
print(f"  • Размер валидационного набора: {len(val_df)}")
print(f"  • Шагов за эпоху: {len(train_gen)}")

history = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)

print("\n ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print(f"  • Фактическое количество эпох: {len(history.history['loss'])}")
print(f"  • Лучшая модель сохранена как: 'best_model_retinopathy.keras'")

# Сохранение полной модели
print("\n СОХРАНЕНИЕ МОДЕЛИ:")
model.save('retinopathy_final_model.keras')
print(" Модель сохранена как 'retinopathy_final_model.keras'")


# визуализация результатов обучения

# загрузка истории обучения если нужно
if 'history' not in locals():
    try:
        # Пытаемся загрузить из CSV
        history_df = pd.read_csv('training_history.csv')
        history = type('History', (), {'history': history_df.to_dict('list')})()
        print(" История обучения загружена из CSV файла")
    except:
        print(" История обучения не найдена")
        history = None

if history:
    print(f"\n ИСТОРИЯ ОБУЧЕНИЯ ({len(history.history['loss'])} эпох):")
    
    # находим лучшую эпоху
    val_auc_values = history.history.get('val_auc', [])
    if val_auc_values:
        best_epoch = np.argmax(val_auc_values)
        print(f"  • Лучшая эпоха: {best_epoch + 1}")
        print(f"  • Лучший val_auc: {val_auc_values[best_epoch]:.4f}")
    
    # создаем графики
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Функция потерь', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_title('Точность', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. AUC
    if 'auc' in history.history:
        axes[0, 2].plot(history.history['auc'], label='Train AUC', linewidth=2)
        axes[0, 2].plot(history.history['val_auc'], label='Val AUC', linewidth=2)
        axes[0, 2].set_title('AUC (площадь под кривой)', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Эпоха')
        axes[0, 2].set_ylabel('AUC')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
        axes[1, 0].set_title('Precision (точность)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Эпоха')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
        axes[1, 1].set_title('Recall (полнота)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Эпоха')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Learning Rate
    if 'lr' in history.history:
        axes[1, 2].plot(history.history['lr'], linewidth=2, color='purple')
        axes[1, 2].set_title('Learning Rate', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Эпоха')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # таблица с лучшими значениями метрик
    print("\n ЛУЧШИЕ МЕТРИКИ МОДЕЛИ:")
    
    metrics_data = []
    for metric in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
        if metric in history.history:
            values = history.history[metric]
            if 'val_' in metric:
                best_idx = np.argmin(values) if 'loss' in metric else np.argmax(values)
            else:
                best_idx = len(values) - 1  # последняя эпоха для train
                
            best_value = values[best_idx]
            metrics_data.append([metric, f"{best_value:.4f}", best_idx + 1])
    
    # добавляем AUC если есть
    for metric in ['auc', 'val_auc']:
        if metric in history.history:
            values = history.history[metric]
            best_idx = np.argmax(values)
            best_value = values[best_idx]
            metrics_data.append([metric, f"{best_value:.4f}", best_idx + 1])
    
    # выводим таблицу
    print(f"{'Метрика':<15} {'Значение':<10} {'Эпоха':<6}")
    print("-"*35)
    for row in metrics_data:
        print(f"{row[0]:<15} {row[1]:<10} {row[2]:<6}")


# оценка модели и предсказания

# загрузка лучшей модели если нужно
try:
    best_model = tf.keras.models.load_model('best_model_retinopathy.keras')
    print(" Загружена лучшая модель из 'best_model_retinopathy.keras'")
except:
    print(" Лучшая модель не найдена, используем текущую модель")
    best_model = model

# Оценка модели
print("\n ОЦЕНКА МОДЕЛИ НА ВАЛИДАЦИОННОМ НАБОРЕ:")

results = best_model.evaluate(val_gen, verbose=0)

print("Метрики модели:")

metrics_names = best_model.metrics_names
for name, value in zip(metrics_names, results):
    print(f"{name:20}: {value:.4f}")

# сбор предсказаний для confusion matrix
print("\n АНАЛИЗ ПРЕДСКАЗАНИЙ:")
y_true_all = []
y_pred_all = []
y_pred_probs_all = []

print("Сбор предсказаний...")
for i in range(len(val_gen)):
    images, labels = val_gen[i]
    predictions = best_model.predict(images, verbose=0)
    
    y_true_all.extend(np.argmax(labels, axis=1))
    y_pred_all.extend(np.argmax(predictions, axis=1))
    y_pred_probs_all.extend(predictions)
    
    if (i + 1) % 5 == 0:
        print(f"  Обработано {i + 1}/{len(val_gen)} батчей")

print(f" Собрано {len(y_true_all)} предсказаний")

# Confusion Matrix
print("\n CONFUSION MATRIX:")
cm = confusion_matrix(y_true_all, y_pred_all)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f'Class {i}' for i in range(5)],
            yticklabels=[f'Class {i}' for i in range(5)])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Истинный класс', fontsize=12)
plt.xlabel('Предсказанный класс', fontsize=12)
plt.tight_layout()
plt.show()

# Classification Report
print("\n CLASSIFICATION REPORT:")

report = classification_report(y_true_all, y_pred_all, 
                               target_names=[f'Class {i}' for i in range(5)],
                               digits=4)
print(report)

# визуализация примеров предсказаний
print("\n ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ:")

# берем примеры из валидационного набора
sample_images, sample_labels = val_gen[0]
sample_predictions = best_model.predict(sample_images, verbose=0)

# выбираем 6 примеров для визуализации
num_samples = min(6, len(sample_images))
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i in range(num_samples):
    ax = axes[i]
    
    # показываем изображение
    ax.imshow(sample_images[i])
    
    # получаем истинный и предсказанный класс
    true_class = np.argmax(sample_labels[i])
    pred_class = np.argmax(sample_predictions[i])
    pred_prob = np.max(sample_predictions[i])
    
    # определяем цвет текста (зеленый если правильно, красный если нет)
    color = 'green' if true_class == pred_class else 'red'
    
    # добавляем информацию
    ax.set_title(f'Истинный: {true_class} | Предсказанный: {pred_class}\nВероятность: {pred_prob:.2%}', 
                 color=color, fontweight='bold')
    ax.axis('off')
    
    # добавляем рамку соответствующего цвета
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)

# скрываем неиспользованные оси
for i in range(num_samples, len(axes)):
    axes[i].axis('off')

plt.suptitle('Примеры предсказаний модели', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# статистика предсказаний
print("\n СТАТИСТИКА ПРЕДСКАЗАНИЙ:")

accuracy = np.mean(np.array(y_true_all) == np.array(y_pred_all))
print(f"Общая точность: {accuracy:.2%}")

# точность по классам
print("\nТочность по классам:")
for class_id in range(5):
    class_indices = np.where(np.array(y_true_all) == class_id)[0]
    if len(class_indices) > 0:
        class_accuracy = np.mean(np.array(y_pred_all)[class_indices] == class_id)
        print(f"  Класс {class_id}: {class_accuracy:.2%} ({len(class_indices)} примеров)")


# заключение и рекомендации
print("\n РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
print("-"*40)

if 'history' in locals():
    final_accuracy = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0
    final_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else 0
    
    print(f" Финальная точность на валидации: {final_accuracy:.2%}")
    print(f" Финальные потери на валидации: {final_loss:.4f}")
    
    if final_accuracy > 0.85:
        print(" Отличный результат! Модель хорошо обучилась.")
    elif final_accuracy > 0.70:
        print(" Хороший результат. Есть возможности для улучшения.")
    else:
        print(" Результат можно улучшить")


