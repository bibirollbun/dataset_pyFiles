import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# --- НАЛАШТУВАННЯ MULTI-GPU ---
try:
    # Створюємо стратегію розподілення
    strategy = tf.distribute.MirroredStrategy()
    print(f'Кількість знайдених пристроїв: {strategy.num_replicas_in_sync}')
    
    # Автоматичне налаштування буфера даних
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Збільшуємо розмір батчу, оскільки у нас 2 GPU
    # Якщо для 1 GPU було 64, то для 2 ставимо 128 (64 * 2)
    BATCH_SIZE_PER_REPLICA = 64
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    
except Exception as e:
    print(f"Помилка ініціалізації TPU/GPU: {e}")
    # Fallback на звичайний режим, якщо GPU немає
    strategy = tf.distribute.get_strategy() 
    GLOBAL_BATCH_SIZE = 64

print(f"Глобальний розмір батчу: {GLOBAL_BATCH_SIZE}")


# --- ПРОЕКТ 1: CIFAR-10 Multi-GPU ---
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization

# 1. Дані
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

# 2. Створення моделі в контексті Multi-GPU
with strategy.scope():
    model = Sequential([
        Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)),
        BatchNormalization(),
        Conv2D(32, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2,2)),
        Dropout(0.3),

        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2,2)),
        Dropout(0.4),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])

# 3. Навчання (з використанням глобального батчу)
history = model.fit(x_train, y_cat_train, 
                    epochs=8, 
                    batch_size=GLOBAL_BATCH_SIZE,
                    validation_data=(x_test, y_cat_test))

# --- ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ ---
def visualize_results(model, x_test, y_test, num_samples=15):
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Випадковий вибір
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    x_sample = x_test[indices]
    y_true = y_test[indices]
    
    # Передбачення
    predictions = model.predict(x_sample, batch_size=GLOBAL_BATCH_SIZE)
    pred_labels = np.argmax(predictions, axis=1)
    
    # Побудова сітки
    plt.figure(figsize=(15, 8))
    for i in range(num_samples):
        plt.subplot(3, 5, i+1)
        plt.imshow(x_sample[i])
        
        true_label = class_names[y_true[i][0]]
        pred_label = class_names[pred_labels[i]]
        
        # Колір тексту: Зелений якщо вірно, Червоний якщо помилка
        col = 'green' if true_label == pred_label else 'red'
        
        plt.title(f"T: {true_label}\nP: {pred_label}", color=col, fontsize=12, fontweight='bold')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("\n--- Результати розпізнавання ---")
visualize_results(model, x_test, y_test)


# --- ПРОЕКТ 2: Multi-GPU + TF.Data Pipeline ---
import zipfile
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# 1. Швидке розпакування (якщо ще не зроблено)
if not os.path.exists('/kaggle/working/train'):
    with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as zip_ref:
        zip_ref.extractall('/kaggle/working/')

# 2. Підготовка DataFrame
base_dir = '/kaggle/working/train'
filenames = os.listdir(base_dir)
filepaths = [os.path.join(base_dir, f) for f in filenames]
labels = [1 if 'dog' in f else 0 for f in filenames] # 1=Dog, 0=Cat

df = pd.DataFrame({'filepath': filepaths, 'label': labels})
df['label'] = df['label'].astype('float32') # Важливо для TF
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. Створення Efficient Pipeline
IMG_SIZE = (128, 128)

def process_img(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img, label

def get_dataset(dataframe, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((dataframe['filepath'].values, dataframe['label'].values))
    ds = ds.map(process_img, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=2000)
    ds = ds.batch(GLOBAL_BATCH_SIZE, drop_remainder=False) 
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = get_dataset(train_df, shuffle=True)
val_ds = get_dataset(val_df)

# 4. Модель Multi-GPU
with strategy.scope():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        # GPU Augmentation layers
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# 5. Навчання
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# --- ВІЗУАЛІЗАЦІЯ ПЕРЕДБАЧЕНЬ ---
def show_predictions(dataset, model, num=10):
    # Беремо 1 батч з датасету
    imgs, labels = next(iter(dataset))
    preds = model.predict(imgs)
    
    plt.figure(figsize=(15, 6))
    for i in range(num):
        if i >= len(imgs): break
        ax = plt.subplot(2, 5, i + 1)
        
        # Картинка (повертаємо з тензора в numpy)
        img_np = imgs[i].numpy().astype("uint8")
        plt.imshow(img_np)
        
        # Логіка
        is_dog_pred = preds[i][0] > 0.5
        is_dog_true = labels[i] == 1.0
        
        conf = preds[i][0] if is_dog_pred else 1 - preds[i][0]
        pred_text = "Dog" if is_dog_pred else "Cat"
        true_text = "Dog" if is_dog_true else "Cat"
        
        col = 'green' if is_dog_pred == is_dog_true else 'red'
        
        plt.title(f"True: {true_text}\nPred: {pred_text} ({conf:.1%})", color=col)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

print("Візуалізація роботи моделі:")
show_predictions(val_ds, model)


# --- ПРОЕКТ 3: RNN Multi-GPU (True Forecasting - Fixed) ---
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# Очищення пам'яті перед запуском
tf.keras.backend.clear_session()

# --- 0. НАЛАШТУВАННЯ (Страхування від помилок) ---
try:
    strategy = tf.distribute.MirroredStrategy()
    print(f'Кількість GPU: {strategy.num_replicas_in_sync}')
    GLOBAL_BATCH_SIZE = 128 
except:
    strategy = tf.distribute.get_strategy()
    GLOBAL_BATCH_SIZE = 64

# --- 1. ГЕНЕРАЦІЯ ДАНИХ ---
x = np.linspace(0, 100, 2000) 
y = np.sin(x)

split_index = 1600 
train_data = y[:split_index]
test_data = y[split_index:] 

# --- 2. ПІДГОТОВКА DATASET ---
length = 50 

def create_dataset(dataset, look_back=50):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

x_train, y_train = create_dataset(train_data, length)

# Решейп [Samples, Time Steps, Features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Створення tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(1000).batch(GLOBAL_BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

print(f"Форма тренувальних даних: {x_train.shape}")

# --- 3. НАВЧАННЯ (Multi-GPU Model) ---
print("Створення тренувальної моделі...")
with strategy.scope():
    model_train = Sequential([
        Input(shape=(length, 1)), # Keras 3 стиль
        LSTM(64, return_sequences=False),
        Dense(1)
    ])
    model_train.compile(optimizer='adam', loss='mse')

print("Починаємо навчання на GPU...")
model_train.fit(train_ds, epochs=10, verbose=1)

# --- 4. СТВОРЕННЯ МОДЕЛІ ДЛЯ ПРОГНОЗУ (Inference Model) ---

print("Копіювання ваг у модель для прогнозування...")
inference_model = Sequential([
    Input(shape=(length, 1)),
    LSTM(64, return_sequences=False),
    Dense(1)
])

inference_model.set_weights(model_train.get_weights())

forecast_steps = 400 
current_batch = train_data[-length:] 
current_batch = current_batch.reshape((1, length, 1)) 

forecast_predictions = []

print(f"Генеруємо прогноз на {forecast_steps} кроків...")

for i in range(forecast_steps):
    current_pred = inference_model.predict(current_batch, verbose=0)[0]
    
    forecast_predictions.append(current_pred)
    
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# --- 6. ВІЗУАЛІЗАЦІЯ ---
plt.figure(figsize=(14, 7))

# Кінець тренувальних даних
plt.plot(np.arange(1400, split_index), train_data[1400:], label='Training Data (End)', color='blue')

# Реальне майбутнє (Ground Truth)
true_future_idx = np.arange(split_index, split_index + len(test_data))
plt.plot(true_future_idx, test_data, label='True Future', color='green', alpha=0.5)

# Наш прогноз
forecast_idx = np.arange(split_index, split_index + forecast_steps)
plt.plot(forecast_idx, forecast_predictions, label='RNN Auto-regressive Forecast', color='red', linestyle='--', linewidth=2)

plt.title("RNN True Forecasting (Multi-GPU Training -> Single Inference)")
plt.xlabel("Time Steps")
plt.legend()
plt.grid(True)
plt.show()


import tensorflow as tf
import numpy as np
import os
import time

# --- 1. АВТОМАТИЧНИЙ ПОШУК ТА ЗАВАНТАЖЕННЯ ФАЙЛУ ---
directory = '/kaggle/input/kobzar'
text_file = None

# Шукаємо файл у папці
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".txt"):
            text_file = os.path.join(root, file)
            break

if text_file:
    print(f"Знайдено файл: {text_file}")
    # Читаємо файл (обов'язково utf-8 для кирилиці)
    text = open(text_file, 'rb').read().decode(encoding='utf-8')
    print(f"Довжина тексту: {len(text)} символів")
else:
    print("Помилка: Текстовий файл (.txt) не знайдено у папці!")
    # Якщо файл не знайдено, зупиняємо виконання
    raise FileNotFoundError("Перевірте структуру датасету")

# Подивимось на перші рядки
print("\n--- ПРИКЛАД ТЕКСТУ ---")
print(text[:200])

# --- 2. ПІДГОТОВКА ДАНИХ ---
# Створюємо словник унікальних символів (а, б, в, г, ґ...)
vocab = sorted(set(text))
print(f'\nУнікальних символів: {len(vocab)}')

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# Нарізка на шматки (seq_length)
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
dataset = sequences.map(split_input_target)

# Batching
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# --- 3. МОДЕЛЬ НАВЧАННЯ ---
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Input

model_nlp = Sequential([
    Embedding(vocab_size, embedding_dim),
    GRU(rnn_units, return_sequences=True, stateful=False, recurrent_initializer='glorot_uniform'),
    Dense(vocab_size)
])

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model_nlp.compile(optimizer='adam', loss=loss)

# --- 4. НАВЧАННЯ ---
EPOCHS = 30
print(f"Починаємо навчання на {EPOCHS} епох...")
history = model_nlp.fit(dataset, epochs=EPOCHS)


# --- 5. ГЕНЕРАЦІЙНА МОДЕЛЬ ---
model_gen = Sequential([
    Input(batch_shape=(1, None)), 
    Embedding(vocab_size, embedding_dim),
    GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
    Dense(vocab_size)
])

# Копіюємо навчені ваги
model_gen.set_weights(model_nlp.get_weights())
print("Модель готова до творчості.")

# --- 6. ФУНКЦІЯ ГЕНЕРАЦІЇ ---
def generate_text_safe(model, start_string):
    # Скидаємо пам'ять GRU шару
    for layer in model.layers:
        if hasattr(layer, 'reset_states'):
            layer.reset_states()

    num_generate = 300 # Кількість літер
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    temperature = 0.45

    if len(start_string) > 1:
        for i in range(len(start_string) - 1):
            x = input_eval[:, i:i+1]
            model(x)

    next_input = input_eval[:, -1:]

    for i in range(num_generate):
        predictions = model(next_input)
        predictions = tf.squeeze(predictions, 0)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        
        predicted_id = tf.random.categorical(tf.expand_dims(predictions, 0), num_samples=1)[-1,0].numpy()

        next_input = tf.expand_dims([predicted_id], 0)
        next_input = tf.expand_dims(next_input, 0)
        
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

# --- 7. РЕЗУЛЬТАТ ---
print("\n--- ШТУЧНИЙ КОБЗАР ---")
try:
    generated_verse = generate_text_safe(model_gen, u"Штучний інтелект")
    print(generated_verse)
except Exception as e:
    print(f"Помилка: {e}")


# --- ПРОЕКТ 5: Autoencoder Multi-GPU ---
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D

# 1. Дані
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train.astype('float32') / 255., -1)
x_test = np.expand_dims(x_test.astype('float32') / 255., -1)

noise = 0.5
x_train_noisy = np.clip(x_train + noise * np.random.normal(size=x_train.shape), 0., 1.)
x_test_noisy = np.clip(x_test + noise * np.random.normal(size=x_test.shape), 0., 1.)

# 2. Модель Multi-GPU
with strategy.scope():
    input_img = Input(shape=(28, 28, 1))
    
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPool2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPool2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = tf.keras.models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 3. Навчання
autoencoder.fit(x_train_noisy, x_train, 
                epochs=5, 
                batch_size=GLOBAL_BATCH_SIZE, # 128
                validation_data=(x_test_noisy, x_test))

# 4. Візуалізація підтвердження
decoded_imgs = autoencoder.predict(x_test_noisy[:10])

n = 10
plt.figure(figsize=(20, 6))
plt.suptitle("Зверху: Зашумлене зображення | Знизу: Відновлене AI", fontsize=16)
for i in range(n):
    # Noisy
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Завантаження моделі
print("Завантаження моделі Style Transfer...")
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Функції обробки
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor

# 2. Вибір зображень
try:
    content_dir = '/kaggle/working/train'
    content_filename = os.listdir(content_dir)[0]
    content_path = os.path.join(content_dir, content_filename)
    print(f"Використовуємо фото: {content_filename}")
except:
    content_path = tf.keras.utils.get_file('labrador.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

content_image = load_img(content_path)

# СТИЛЬ (Style) - Ван Гог "Зоряна ніч"
style_url = 'https://storage.googleapis.com/khanhlvg-public.appspot.com/arbitrary-style-transfer/style23.jpg'
path_to_style = tf.keras.utils.get_file('starry_night.jpg', style_url)
style_image = load_img(path_to_style)

# 3. Стилізація
print("Застосування стилю...")
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# 4. Результат
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.title("Оригінал")
plt.imshow(tensor_to_image(content_image))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Стиль (Ван Гог)")
plt.imshow(tensor_to_image(style_image))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Результат AI")
plt.imshow(tensor_to_image(stylized_image))
plt.axis('off')

plt.show()


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
import time

# --- 1. ПІДГОТОВКА ДАНИХ ---
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 

BUFFER_SIZE = 60000
BATCH_SIZE = 128

# Створюємо dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# --- 2. ГЕНЕРАТОР (Художник) ---
def make_generator_model():
    model = tf.keras.Sequential()
    # Починаємо з "зерна" (seed)
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) 

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

# --- 3. ДИСКРИМІНАТОР (Критик) ---
def make_discriminator_model():
    model = tf.keras.Sequential()
    
    # Згортка 1
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Згортка 2
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1)) # Видає число: позитивне = справжнє, негативне = фейк

    return model

generator = make_generator_model()
discriminator = make_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# --- 5. ЦИКЛ НАВЧАННЯ ---
EPOCHS = 30 
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Функція для малювання
def generate_and_save_images(model, epoch, test_input):
  predictions = model(test_input, training=False)

  plt.figure(figsize=(4, 4))
  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      img = predictions[i, :, :, 0] * 127.5 + 127.5
      plt.imshow(img.numpy().astype('uint8'), cmap='gray')
      plt.axis('off')
  
  plt.suptitle(f"Epoch: {epoch}")
  plt.show()

print("Починаємо навчання GAN...")
for epoch in range(EPOCHS):
  start = time.time()

  for image_batch in train_dataset:
    train_step(image_batch)

  print(f'Time for epoch {epoch + 1} is {time.time()-start:.1f} sec')
  generate_and_save_images(generator, epoch + 1, seed)

print("Навчання завершено.")

