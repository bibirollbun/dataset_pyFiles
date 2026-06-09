# Hücre 1: Importlar, Sabitler ve Veri Hazırlık

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import random
# Uyarıları kapatma kodları (isteğe bağlı)
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Sabitler
DATA_DIR = '/kaggle/input/alaska2-image-steganalysis/' # Kaggle veri seti yolu
# Sınıflar - Alaska2 için J-EMB sınıfı olmalı, UERD değil
CLASSES = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
CLASS_LABELS = {cls: i for i, cls in enumerate(CLASSES)}

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = len(CLASSES)
AUTOTUNE = tf.data.AUTOTUNE

# Transformer Ayarları (Model mimarisi için, burada tanımlı kalabilir veya Hücre 2'ye taşınabilir)
NUM_HEADS = 8
FF_DIM = 2048
NUM_TRANSFORMER_BLOCKS = 1
MLP_HIDDEN_UNITS = [512]
DROPOUT_RATE = 0.1


# --- Veri Yükleme ve Hazırlık ---
def list_files_and_labels(data_dir):
    all_image_paths = []
    all_image_labels = []
    for cls_name in CLASSES:
        cls_dir = os.path.join(data_dir, cls_name)
        if os.path.exists(cls_dir):
            image_files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith('.jpg')]
            all_image_paths.extend(image_files)
            all_image_labels.extend([CLASS_LABELS[cls_name]] * len(image_files))
        else:
            print(f"Uyarı: Klasör bulunamadı: {cls_dir}")
    return all_image_paths, all_image_labels

all_paths, all_labels = list_files_and_labels(DATA_DIR)

combined = list(zip(all_paths, all_labels))
random.shuffle(combined) # Veriyi karıştır
all_paths, all_labels = zip(*combined)


# !!! BURASI DEĞİŞTİRİLDİ: Veri setini %80 Eğitim / %10 Doğrulama / %10 Test olarak ayırıyoruz !!!
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1 # TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT = 1.0 olduğundan emin olun

total_size = len(all_paths)
train_size = int(total_size * TRAIN_SPLIT)
val_size = int(total_size * VAL_SPLIT)
# Test boyutu kalan kısım
test_size = total_size - train_size - val_size


# Veriyi ayırma için indexleri hesapla
train_end_index = train_size
val_end_index = train_size + val_size # Validasyonun bittiği yer, testin başladığı yer


# Veriyi dilimleme
train_paths = all_paths[:train_end_index]
train_labels = all_labels[:train_end_index]

val_paths = all_paths[train_end_index:val_end_index]
val_labels = all_labels[train_end_index:val_end_index]

test_paths = all_paths[val_end_index:]
test_labels = all_labels[val_end_index:]


# !!! BURASI GÜNCELLENDİ: Üç yeni veri set boyutunu gösteriyoruz !!!
print(f"Toplam veri: {total_size}")
print(f"Eğitim verisi ({TRAIN_SPLIT*100}%): {len(train_paths)}")
print(f"Doğrulama verisi ({VAL_SPLIT*100}%): {len(val_paths)}")
print(f"Test verisi ({TEST_SPLIT*100}%): {len(test_paths)}")


# EfficientNet preprocessing kullanılıyor (Aynı)
def preprocess_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    # EfficientNet'in kendi ön işlemcisini kullan
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    label = tf.one_hot(label, NUM_CLASSES)
    return img, label


# tf.data datasetlerini oluşturma (Üç dataset olacak)
train_dataset = tf.data.Dataset.from_tensor_slices((list(train_paths), list(train_labels)))
train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((list(val_paths), list(val_labels)))
val_dataset = val_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# !!! BURASI EKLENDİ: Test dataseti oluşturma !!!
test_dataset = tf.data.Dataset.from_tensor_slices((list(test_paths), list(test_labels)))
test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
# Test seti için shuffle ve prefetch gerekli olmayabilir, sadece batch yeterli
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)


print("\nHücre 1 Tamamlandı: Veri 3 parçaya ayrıldı ve hazırlandı.")


# Hücre 2: Model Mimarisi Tanımları

# EfficientNet Özellik Çıkarıcıyı ayrı bir fonksiyonla oluşturalım
def get_efficientnet_backbone(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
     base_model = tf.keras.applications.EfficientNetB0(
         include_top=False,
         weights='imagenet',
         input_shape=input_shape
     )
     return base_model

# Transformer Encoder Bloğu
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Özellik haritasını token dizisine dönüştürme katmanı
class RearrangeToTokens(layers.Layer):
    def call(self, x):
        batch_size, h, w, c = tf.unstack(tf.shape(x))
        x = tf.reshape(x, (batch_size, h * w, c))
        return x

# Konumsal Embedding katmanı
class PositionEmbedding(layers.Layer):
    def __init__(self, sequence_length, embed_dim):
        super().__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.position_embedding = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        embedded_positions = self.position_embedding(positions)
        return inputs + embedded_positions

# Hibrit Modeli Oluşturan Ana Fonksiyon
# Bu fonksiyon sadece mimariyi kurar, dondurma/çözme dışarıda yapılacak
def build_hybrid_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=input_shape)

    # EfficientNet Backbone'u burada oluşturuyoruz
    effnet_backbone = get_efficientnet_backbone(input_shape)
    # Backbone'dan çıkan özellikleri alıyoruz
    effnet_features = effnet_backbone(inputs)

    # Geri kalan katmanlar (Transformer ve Sınıflandırma başlığı)
    tokens = RearrangeToTokens()(effnet_features)
    embed_dim = tokens.shape[-1]
    sequence_length = tokens.shape[1]
    position_embedding_layer = PositionEmbedding(sequence_length, embed_dim)
    x = position_embedding_layer(tokens)
    x = layers.Dropout(DROPOUT_RATE)(x)

    for _ in range(NUM_TRANSFORMER_BLOCKS):
        x = TransformerBlock(embed_dim, NUM_HEADS, FF_DIM, DROPOUT_RATE)(x)

    representation = layers.GlobalAveragePooling1D()(x)

    for units in MLP_HIDDEN_UNITS:
        representation = layers.Dense(units, activation="relu")(representation)
        representation = layers.Dropout(DROPOUT_RATE)(representation)

    outputs = layers.Dense(num_classes, activation="softmax")(representation)

    # Modeli döndürürken backbone'u da döndürelim ki dışarıdan erişip dondurabilelim
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model, effnet_backbone # Modeli ve backbone'u döndür

print("Hücre 2 Tamamlandı: Model mimarisi tanımlandı.")


# Hücre 3: Model Oluşturma ve Aşama 1 Eğitim (Backbone Dondurulmuş)

# Modeli ve backbone referansını oluştur
# Bu fonksiyonu sadece bir kez çalıştırmanız gerekir!
model, effnet_backbone = build_hybrid_model()


print("\n--- Aşama 1: Backbone Donduruluyor ---")
# Backbone'u dondur
effnet_backbone.trainable = False

# Modeli Aşama 1 için derle
# Sadece yeni katmanları eğiteceğimiz için biraz daha yüksek bir LR kullanabiliriz
optimizer_phase1 = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer_phase1,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Aşama 1 Model Özeti (Trainable params sadece ek katmanlar olmalı):")
model.summary()

EPOCHS_PHASE1 = 5 # Örn: Backbone dondurulmuşken 5 epoch eğit

print(f"\nAşama 1 Eğitime başlıyor ({EPOCHS_PHASE1} epoch)...")
history_phase1 = model.fit(
    train_dataset,
    epochs=EPOCHS_PHASE1,
    validation_data=val_dataset
)

print("\nHücre 3 Tamamlandı: Aşama 1 Eğitim Bitti.")


# Hücre 4: Aşama 2 Eğitim (Backbone Çözülmüş - İnce Ayar)

print("\n--- Aşama 2: Backbone Çözülüyor ve İnce Ayar ---")
# Backbone'u çöz
effnet_backbone.trainable = True

# Modeli tekrar derle (Önemli! Trainability değiştiğinde tekrar derlemek gerekir)
# İnce ayar için çok daha düşük bir LR kullan
optimizer_phase2 = tf.keras.optimizers.Adam(learning_rate=1e-5) # Genellikle 1e-5 veya daha düşük

# Aşama 2 için Learning Rate Scheduler (İsteğe bağlı)
reduce_lr_phase2 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-8,
    verbose=1
)

model.compile(optimizer=optimizer_phase2,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Aşama 2 Model Özeti (Tüm katmanlar trainable olmalı):")
model.summary() # Trainable params sayısının arttığını göreceksiniz

EPOCHS_PHASE2 = 15 # Örn: İnce ayar için 15 epoch daha eğit
TOTAL_EPOCHS = EPOCHS_PHASE1 + EPOCHS_PHASE2 # Toplam epoch sayısı Hücre 3'teki EPOCHS_PHASE1'den alınır


print(f"\nAşama 2 Eğitime başlıyor ({EPOCHS_PHASE2} epoch, toplam {TOTAL_EPOCHS} epoch)...")
history_phase2 = model.fit(
    train_dataset,
    epochs=TOTAL_EPOCHS,         # Toplam epoch sayısı
    initial_epoch=EPOCHS_PHASE1,  # Aşama 1'in bittiği epoch'tan başla
    validation_data=val_dataset,
    callbacks=[reduce_lr_phase2] # Learning Rate Scheduler'ı ekle
)

print("\nHücre 4 Tamamlandı: İki Aşamalı Eğitim Bitti.")


