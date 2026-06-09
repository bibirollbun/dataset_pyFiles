import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# GPU bellek optimizasyonu
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU aktif!")
else:
    print("GPU bulunamadı, CPU kullanılacak.")

# Hiperparametreler
image_size = (224, 224)
batch_size = 128
epochs_gan = 1000  # Zamanın yetip yetmediğini kontrol et, gerekirse 500’e düşürebilirsin
epochs_model_efficientnet = 50
epochs_model_resnet = 50
latent_dim = 200
num_gan_images = 1916  # GAN ile 1,916 sentetik malignant görüntü
num_aug_images = 1000  # Augmentasyon ile 1,000 ek malignant
gan_batch_size = 100
learning_rate_gan = 0.00005  # GAN için düşük öğrenme oranı
custom_threshold = 0.45  # FP’yi düşürmek için hafif artırılmış eşik
post_process_threshold = 0.35  # Post-processing için eşik (FN’yi kontrol altında tutmak için)

# Veri yolları
train_csv_path = '/kaggle/input/siim-isic-melanoma-classification/train.csv'
train_images_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
gan_images_path = '/kaggle/working/gan_generated_malignant/'
aug_images_path = '/kaggle/working/augmented_malignant/'
new_train_csv_path = '/kaggle/working/new_train.csv'
test_csv_path = '/kaggle/input/siim-isic-melanoma-classification/test.csv'
test_images_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

# Klasörleri oluştur
os.makedirs(gan_images_path, exist_ok=True)
os.makedirs(aug_images_path, exist_ok=True)

# Orijinal train.csv’yi yükle
train_df = pd.read_csv(train_csv_path)

# Eksik değerleri doldur
train_df['sex'] = train_df['sex'].fillna('unknown')
train_df['age_approx'] = train_df['age_approx'].fillna(train_df['age_approx'].mean())
train_df['anatom_site_general_challenge'] = train_df['anatom_site_general_challenge'].fillna('unknown')

# Hasta metadata’sını işleme
# Kategorik değişkenleri encode etme
label_encoder_sex = LabelEncoder()
label_encoder_site = LabelEncoder()

train_df['sex_encoded'] = label_encoder_sex.fit_transform(train_df['sex'])
train_df['anatom_site_encoded'] = label_encoder_site.fit_transform(train_df['anatom_site_general_challenge'])

# Sayısal değişkeni (age_approx) normalize etme
train_df['age_approx'] = (train_df['age_approx'] - train_df['age_approx'].mean()) / train_df['age_approx'].std()

# Benign ve Malignant örnekleri ayır
benign_df = train_df[train_df['target'] == 0]
malignant_df = train_df[train_df['target'] == 1]
print(f"Orijinal benign örnek sayısı: {len(benign_df)}")
print(f"Orijinal malignant örnek sayısı: {len(malignant_df)}")

# Benign veriyi 5,000’e düşür (dengeli bir veri seti için)
train_benign_df = benign_df.sample(n=5000, random_state=42)
print(f"Eğitim için seçilen benign örnek sayısı: {len(train_benign_df)}")

# Malignant görüntüleri yükleme ve ön işleme (GAN için)
def load_and_preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 127.5  # [-1, 1] aralığına normalize et
    return img

malignant_image_paths = [os.path.join(train_images_path, img_name + '.jpg') for img_name in malignant_df['image_name']]
malignant_dataset = tf.data.Dataset.from_tensor_slices(malignant_image_paths)
malignant_dataset = malignant_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
malignant_dataset = malignant_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Generator modeli - Dropout ve Noise ile çeşitlilik
def build_generator():
    inputs = Input(shape=(latent_dim,))
    noise = layers.GaussianNoise(0.1)(inputs)  # Gürültü ekleyerek çeşitlilik
    x = layers.Dense(7 * 7 * 256, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((7, 7, 256))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2DTranspose(3, (5, 5), strides=(4, 4), padding='same', use_bias=False, activation='tanh')(x)
    return models.Model(inputs, x)

# Discriminator modeli - Dropout ile regularization
def build_discriminator():
    inputs = Input(shape=(224, 224, 3))
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs, outputs)

# Modelleri oluştur
generator = build_generator()
discriminator = build_discriminator()

# Optimizer - Düşük öğrenme oranı
generator_optimizer = Adam(learning_rate=learning_rate_gan, beta_1=0.5)
discriminator_optimizer = Adam(learning_rate=learning_rate_gan, beta_1=0.5)

# Kayıp fonksiyonları - Label Smoothing ile iyileştirme
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_loss(real_output, fake_output):
    real_labels = tf.random.uniform(shape=real_output.shape, minval=0.9, maxval=1.0)  # Label smoothing
    fake_labels = tf.random.uniform(shape=fake_output.shape, minval=0.0, maxval=0.1)
    real_loss = cross_entropy(real_labels, real_output)
    fake_loss = cross_entropy(fake_labels, fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    target_labels = tf.random.uniform(shape=fake_output.shape, minval=0.9, maxval=1.0)
    return cross_entropy(target_labels, fake_output)

# Eğitim adımı
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, latent_dim], stddev=1.5)  # Noise’a çeşitlilik
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    return gen_loss, disc_loss

# GAN tarafından üretilen görselleri görselleştirme fonksiyonu
def plot_generated_images(generator, epoch, num_images=5):
    noise = tf.random.normal([num_images, latent_dim], stddev=1.5)
    generated_images = generator(noise, training=False)
    generated_images = (generated_images * 127.5 + 127.5).numpy()  # [0, 255] aralığına geri dön

    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(generated_images[i].astype(np.uint8))
        plt.axis('off')
        plt.title(f"Image {i+1}")
    plt.suptitle(f"Generated Images at Epoch {epoch + 1}")
    plt.show()

# GAN eğitimi
def train_gan(dataset, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for image_batch in dataset:
            if image_batch.shape[0] == batch_size:  # Batch boyutunu kontrol et
                gen_loss, disc_loss = train_step(image_batch)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1} - Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")
        if (epoch + 1) % 100 == 0:  # Her 100 epoch’ta bir görselleri göster
            plot_generated_images(generator, epoch)

train_gan(malignant_dataset, epochs_gan)

# Sentetik görüntü üretimi ve diske kaydetme
def generate_and_save_gan_images(generator, total_images, batch_size, save_path):
    num_batches = total_images // batch_size
    for batch_idx in range(num_batches):
        noise = tf.random.normal([batch_size, latent_dim], stddev=1.5)
        generated_images = generator(noise, training=False)
        generated_images = (generated_images * 127.5 + 127.5).numpy()  # [0, 255] aralığına geri dön
        for i in range(batch_size):
            img_name = f"gan_malignant_{batch_idx * batch_size + i}.jpg"
            img_path = os.path.join(save_path, img_name)
            plt.imsave(img_path, generated_images[i].astype(np.uint8))
    remaining_images = total_images % batch_size
    if remaining_images > 0:
        noise = tf.random.normal([remaining_images, latent_dim], stddev=1.5)
        generated_images = generator(noise, training=False)
        generated_images = (generated_images * 127.5 + 127.5).numpy()
        for i in range(remaining_images):
            img_name = f"gan_malignant_{num_batches * batch_size + i}.jpg"
            img_path = os.path.join(save_path, img_name)
            plt.imsave(img_path, generated_images[i].astype(np.uint8))

print(f"{num_gan_images} adet sentetik malignant görüntü diske kaydediliyor...")
generate_and_save_gan_images(generator, num_gan_images, gan_batch_size, gan_images_path)

# Augmentasyonla 1,000 malignant üretme
def augment_image(image_path, is_malignant=True):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    if is_malignant:
        # Malignant için daha agresif augmentasyon
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.3)
        img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
        img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
        img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    else:
        # Benign için daha hafif augmentasyon (modelin benign’i daha iyi öğrenmesi için)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.cast(img, tf.uint8)
    return img.numpy()

# Malignant için augmentasyon
aug_rows = []
for i in range(num_aug_images):
    random_row = malignant_df.sample(1).iloc[0]
    img_path = os.path.join(train_images_path, random_row['image_name'] + '.jpg')
    aug_img = augment_image(img_path, is_malignant=True)
    aug_img_name = f"augmented_malignant_{i}.jpg"
    aug_img_path = os.path.join(aug_images_path, aug_img_name)
    plt.imsave(aug_img_path, aug_img)
    new_row = random_row.copy()
    new_row['image_name'] = aug_img_name
    new_row['target'] = 1
    aug_rows.append(new_row)

aug_df = pd.DataFrame(aug_rows)

# GAN verilerini CSV’ye ekle
gan_rows = []
for i in range(num_gan_images):
    img_name = f"gan_malignant_{i}.jpg"
    random_row = malignant_df.sample(1).iloc[0]
    new_row = random_row.copy()
    new_row['image_name'] = img_name
    new_row['target'] = 1
    gan_rows.append(new_row)

gan_df = pd.DataFrame(gan_rows)

# Tüm veriyi birleştir
train_df_final = pd.concat([train_benign_df, malignant_df, gan_df, aug_df], ignore_index=True)
print(f"GAN ve augmentasyon sonrası veri seti boyutu: {train_df_final.shape}")

# Malignant sınıfının 3,500 olduğunu kontrol et
assert len(train_df_final[train_df_final['target'] == 1]) == 3500, "Malignant sınıfı 3,500’e ulaşmadı!"

# Yeni train.csv’yi kaydet
train_df_final.to_csv(new_train_csv_path, index=False)
print(f"Yeni train.csv kaydedildi: {new_train_csv_path}")

# Train, val ve test setlerini böl (%80 train, %10 val, %10 test)
def get_image_path(image_name):
    if 'gan_malignant_' in image_name:
        return os.path.join(gan_images_path, image_name)
    elif 'augmented_malignant_' in image_name:
        return os.path.join(aug_images_path, image_name)
    else:
        return os.path.join(train_images_path, image_name + '.jpg')

train_df_final['image_path'] = train_df_final['image_name'].apply(get_image_path)
train_df_final['target'] = train_df_final['target'].astype(float)

# Hasta metadata’sını tekrar encode et (birleştirme sonrası)
train_df_final['sex_encoded'] = label_encoder_sex.transform(train_df_final['sex'])
train_df_final['anatom_site_encoded'] = label_encoder_site.transform(train_df_final['anatom_site_general_challenge'])
train_df_final['age_approx'] = (train_df_final['age_approx'] - train_df_final['age_approx'].mean()) / train_df_final['age_approx'].std()

X = train_df_final[['image_path', 'sex_encoded', 'age_approx', 'anatom_site_encoded']]
y = train_df_final['target']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Eğitim seti: {len(X_train)} örnek (%80)")
print(f"Doğrulama seti: {len(X_val)} örnek (%10)")
print(f"Test seti: {len(X_test)} örnek (%10)")

# Veri pipeline’ı (Görüntü ve metadata’yı birleştirme)
def load_and_preprocess_image(image_path, sex, age, site, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    if 'train' in str(image_path):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.3)
        img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
        img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
        img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    label = tf.cast(label, tf.float32)
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return (img, metadata), label

def create_dataset(df, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values,
        df['target'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site, label: load_and_preprocess_image(img_path, sex, age, site, label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).repeat()
    return dataset

def create_test_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values,
        df['target'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site, label: load_and_preprocess_image(img_path, sex, age, site, label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(pd.concat([X_train, y_train], axis=1), batch_size, shuffle=True)
val_dataset = create_dataset(pd.concat([X_val, y_val], axis=1), batch_size, shuffle=False)
test_dataset = create_test_dataset(pd.concat([X_test, y_test], axis=1), batch_size)

# Modeli oluştur (Görüntü ve metadata’yı birleştirme)
def build_model_with_metadata(base_model):
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    # Görüntü girişi
    image_input = Input(shape=(224, 224, 3), name='image_input')
    x = base_model(image_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Metadata girişi
    metadata_input = Input(shape=(3,), name='metadata_input')
    m = layers.Dense(16, activation='relu')(metadata_input)
    m = layers.Dense(8, activation='relu')(m)

    # Görüntü ve metadata’yı birleştirme
    combined = layers.concatenate([x, m])
    combined = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
    combined = layers.Dropout(0.5)(combined)
    output = layers.Dense(1, activation='sigmoid')(combined)

    model = models.Model(inputs=[image_input, metadata_input], outputs=output)
    return model

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # 10 epoch boyunca val_loss iyileşmezse durur
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
checkpoint_efficientnet = ModelCheckpoint('/kaggle/working/efficientnet_best_model.keras',
                                         monitor='val_loss', save_best_only=True, mode='min')
checkpoint_resnet = ModelCheckpoint('/kaggle/working/resnet_best_model.keras',
                                    monitor='val_loss', save_best_only=True, mode='min')

# Class weight - FP’yi düşürmek için ağırlığı biraz azalttık
class_weight = {0: 1.0, 1: 8.0}  # 10.0’dan 8.0’a düşürüldü

# Focal Loss - Benign sınıfına daha fazla odaklanmak için alpha artırıldı
def focal_loss(gamma=2.0, alpha=0.5):  # alpha 0.25’ten 0.5’e çıkarıldı
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1) - (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0)
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# 1. EfficientNetB0 Modeli
print("EfficientNetB0 modeli eğitiliyor...")
base_model_efficientnet = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_efficientnet = build_model_with_metadata(base_model_efficientnet)

model_efficientnet.compile(optimizer=Adam(learning_rate=0.00001), loss=focal_loss(), metrics=['accuracy'])

model_efficientnet.fit(train_dataset, 
                       epochs=epochs_model_efficientnet, 
                       validation_data=val_dataset, 
                       steps_per_epoch=len(X_train) // batch_size, 
                       validation_steps=len(X_val) // batch_size,
                       class_weight=class_weight, 
                       callbacks=[early_stopping, reduce_lr, checkpoint_efficientnet])

# 2. ResNet50V2 Modeli
print("ResNet50V2 modeli eğitiliyor...")
base_model_resnet = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_resnet = build_model_with_metadata(base_model_resnet)

model_resnet.compile(optimizer=Adam(learning_rate=0.00001), loss=focal_loss(), metrics=['accuracy'])

model_resnet.fit(train_dataset, 
                 epochs=epochs_model_resnet, 
                 validation_data=val_dataset, 
                 steps_per_epoch=len(X_train) // batch_size, 
                 validation_steps=len(X_val) // batch_size,
                 class_weight=class_weight, 
                 callbacks=[early_stopping, reduce_lr, checkpoint_resnet])

# Validation setinde modellerin performansını değerlendirme
print("Validation setinde modellerin performansını değerlendiriliyor...")
val_predictions_efficientnet = model_efficientnet.predict(val_dataset)
val_predictions_resnet = model_resnet.predict(val_dataset)

y_val_true = y_val.values.astype(int)

# Varsayılan threshold (0.5) ile tahminler
val_pred_efficientnet = (val_predictions_efficientnet > 0.5).astype(int).flatten()
val_pred_resnet = (val_predictions_resnet > 0.5).astype(int).flatten()

# Precision, Recall ve F1-score hesaplama
precision_efficientnet = precision_score(y_val_true, val_pred_efficientnet)
recall_efficientnet = recall_score(y_val_true, val_pred_efficientnet)
f1_efficientnet = f1_score(y_val_true, val_pred_efficientnet)

precision_resnet = precision_score(y_val_true, val_pred_resnet)
recall_resnet = recall_score(y_val_true, val_pred_resnet)
f1_resnet = f1_score(y_val_true, val_pred_resnet)

print(f"EfficientNetB0 - Validation Precision: {precision_efficientnet:.4f}, Recall: {recall_efficientnet:.4f}, F1: {f1_efficientnet:.4f}")
print(f"ResNet50V2 - Validation Precision: {precision_resnet:.4f}, Recall: {recall_resnet:.4f}, F1: {f1_resnet:.4f}")

# Ensemble ağırlıklarını hesaplama
# FP’yi düşürmek için precision’a biraz daha fazla önem veriyoruz
weight_efficientnet = (0.6 * precision_efficientnet + 0.4 * f1_efficientnet) / (0.6 * (precision_efficientnet + precision_resnet) + 0.4 * (f1_efficientnet + f1_resnet))
weight_resnet = (0.6 * precision_resnet + 0.4 * f1_resnet) / (0.6 * (precision_efficientnet + precision_resnet) + 0.4 * (f1_efficientnet + f1_resnet))

# Ağırlıkları normalize etme
total_weight = weight_efficientnet + weight_resnet
ensemble_weight_efficientnet = weight_efficientnet / total_weight
ensemble_weight_resnet = weight_resnet / total_weight

print(f"Optimize edilmiş ensemble ağırlıkları - EfficientNetB0: {ensemble_weight_efficientnet:.4f}, ResNet50V2: {ensemble_weight_resnet:.4f}")

# Test seti üzerinde tahmin
print("Test seti üzerinde tahmin yapılıyor...")
predictions_efficientnet = model_efficientnet.predict(test_dataset)
predictions_resnet = model_resnet.predict(test_dataset)

# Ensemble: Tahminleri birleştirme
print("Ensemble tahminleri birleştiriliyor...")
predictions_ensemble = (ensemble_weight_efficientnet * predictions_efficientnet + 
                        ensemble_weight_resnet * predictions_resnet)

y_true = y_test.values.astype(int)

# Varsayılan threshold (0.5) ile değerlendirme
y_pred_ensemble = (predictions_ensemble > 0.5).astype(int).flatten()

print("Ensemble - Classification Report (Default Threshold 0.5):")
print(classification_report(y_true, y_pred_ensemble))
print("Ensemble - Confusion Matrix (Default Threshold 0.5):")
print(confusion_matrix(y_true, y_pred_ensemble))

# Custom threshold ile değerlendirme
y_pred_custom = (predictions_ensemble > custom_threshold).astype(int).flatten()

print(f"Ensemble - Classification Report (Custom Threshold {custom_threshold}):")
print(classification_report(y_true, y_pred_custom))
print(f"Ensemble - Confusion Matrix (Custom Threshold {custom_threshold}):")
print(confusion_matrix(y_true, y_pred_custom))

# Post-processing ile FN azaltma
post_processed_pred = np.where(predictions_ensemble > post_process_threshold, 1, 0).flatten()

print(f"Post-Processed - Classification Report (Threshold {post_process_threshold}):")
print(classification_report(y_true, post_processed_pred))
print(f"Post-Processed - Confusion Matrix (Threshold {post_process_threshold}):")
print(confusion_matrix(y_true, post_processed_pred))

# Hatalı tahmin analizi
print("\nKanserli sınıfta (target=1) hatalı tahmin edilen resimler (Custom Threshold):")
misclassified_malignant = []
original_malignant_count = 0
synthetic_malignant_count = 0

X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

for i in range(len(y_true)):
    if y_true[i] == 1 and y_pred_custom[i] != y_true[i]:
        image_name = X_test_reset.iloc[i]['image_path'].split('/')[-1]
        misclassified_malignant.append(image_name)
        print(f"Hatalı tahmin edilen resim: {image_name}")
        if 'gan_malignant_' in image_name or 'augmented_malignant_' in image_name:
            synthetic_malignant_count += 1
        else:
            original_malignant_count += 1

print(f"\nToplam hatalı tahmin edilen kanserli resim sayısı: {len(misclassified_malignant)}")
print(f"Orijinal malignantlardan hatalı tahmin edilen: {original_malignant_count}")
print(f"GAN veya augmentasyon ile üretilen sentetik malignantlardan hatalı tahmin edilen: {synthetic_malignant_count}")

# Submission dosyası
print("\nAsıl test verisi ile submission.csv oluşturuluyor...")
test_df = pd.read_csv(test_csv_path)
test_df['image_path'] = test_df['image_name'].apply(lambda x: os.path.join(test_images_path, x + '.jpg'))

# Test verisi için metadata’yı işleme
test_df['sex'] = test_df['sex'].fillna('unknown')
test_df['age_approx'] = test_df['age_approx'].fillna(test_df['age_approx'].mean())
test_df['anatom_site_general_challenge'] = test_df['anatom_site_general_challenge'].fillna('unknown')

test_df['sex_encoded'] = label_encoder_sex.transform(test_df['sex'])
test_df['anatom_site_encoded'] = label_encoder_site.transform(test_df['anatom_site_general_challenge'])
test_df['age_approx'] = (test_df['age_approx'] - test_df['age_approx'].mean()) / test_df['age_approx'].std()

def load_and_preprocess_test_image(image_path, sex, age, site):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return img, metadata

def create_submission_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site: load_and_preprocess_test_image(img_path, sex, age, site),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

test_dataset_submission = create_submission_dataset(test_df, batch_size)

# Her iki modelle submission tahmini
predictions_submission_efficientnet = model_efficientnet.predict(test_dataset_submission)
predictions_submission_resnet = model_resnet.predict(test_dataset_submission)

# Ensemble tahmini
predictions_submission_ensemble = (ensemble_weight_efficientnet * predictions_submission_efficientnet + 
                                   ensemble_weight_resnet * predictions_submission_resnet)

# Custom threshold ile tahmin
test_df['target'] = (predictions_submission_ensemble > custom_threshold).astype(int).flatten()

submission = test_df[['image_name', 'target']]
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("submission.csv oluşturuldu: /kaggle/working/submission.csv")


import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

# Hiperparametreler (önceki kodundan)
image_size = (224, 224)
batch_size = 128
custom_threshold = 0.45
post_process_threshold = 0.35

# Focal Loss fonksiyonunu tanımla
def focal_loss(gamma=2.0, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1) - (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0)
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# Modelleri yükle
model_efficientnet = tf.keras.models.load_model('/kaggle/working/efficientnet_best_model.keras', custom_objects={'focal_loss_fixed': focal_loss()})
model_resnet = tf.keras.models.load_model('/kaggle/working/resnet_best_model.keras', custom_objects={'focal_loss_fixed': focal_loss()})
print("Modeller yüklendi!")

# Test seti için veri setini oluştur (önceki kodundan)
def load_and_preprocess_image(image_path, sex, age, site, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    if 'train' in str(image_path):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.3)
        img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
        img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
        img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    label = tf.cast(label, tf.float32)
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return (img, metadata), label

def create_test_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values,
        df['target'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site, label: load_and_preprocess_image(img_path, sex, age, site, label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Test setini oluştur
test_dataset = create_test_dataset(pd.concat([X_test, y_test], axis=1), batch_size)

# Validation setinde ağırlıkları hesapla
def create_dataset(df, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values,
        df['target'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site, label: load_and_preprocess_image(img_path, sex, age, site, label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

val_dataset = create_dataset(pd.concat([X_val, y_val], axis=1), batch_size, shuffle=False)
val_predictions_efficientnet = model_efficientnet.predict(val_dataset, steps=len(X_val) // batch_size + 1)
val_predictions_resnet = model_resnet.predict(val_dataset, steps=len(X_val) // batch_size + 1)

y_val_true = y_val.values.astype(int)

# Varsayılan threshold (0.5) ile tahminler
val_pred_efficientnet = (val_predictions_efficientnet > 0.5).astype(int).flatten()
val_pred_resnet = (val_predictions_resnet > 0.5).astype(int).flatten()

# Precision, Recall ve F1-score hesaplama
precision_efficientnet = precision_score(y_val_true, val_pred_efficientnet)
recall_efficientnet = recall_score(y_val_true, val_pred_efficientnet)
f1_efficientnet = f1_score(y_val_true, val_pred_efficientnet)

precision_resnet = precision_score(y_val_true, val_pred_resnet)
recall_resnet = recall_score(y_val_true, val_pred_resnet)
f1_resnet = f1_score(y_val_true, val_pred_resnet)

print(f"EfficientNetB0 - Validation Precision: {precision_efficientnet:.4f}, Recall: {recall_efficientnet:.4f}, F1: {f1_efficientnet:.4f}")
print(f"ResNet50V2 - Validation Precision: {precision_resnet:.4f}, Recall: {recall_resnet:.4f}, F1: {f1_resnet:.4f}")

# Ensemble ağırlıklarını hesaplama
weight_efficientnet = (0.6 * precision_efficientnet + 0.4 * f1_efficientnet) / (0.6 * (precision_efficientnet + precision_resnet) + 0.4 * (f1_efficientnet + f1_resnet))
weight_resnet = (0.6 * precision_resnet + 0.4 * f1_resnet) / (0.6 * (precision_efficientnet + precision_resnet) + 0.4 * (f1_efficientnet + f1_resnet))

# Ağırlıkları normalize etme
total_weight = weight_efficientnet + weight_resnet
ensemble_weight_efficientnet = weight_efficientnet / total_weight
ensemble_weight_resnet = weight_resnet / total_weight

print(f"Optimize edilmiş ensemble ağırlıkları - EfficientNetB0: {ensemble_weight_efficientnet:.4f}, ResNet50V2: {ensemble_weight_resnet:.4f}")

# Test seti üzerinde tahmin
print("Test seti üzerinde tahmin yapılıyor...")
predictions_efficientnet = model_efficientnet.predict(test_dataset, steps=len(X_test) // batch_size + 1)
predictions_resnet = model_resnet.predict(test_dataset, steps=len(X_test) // batch_size + 1)

# Ensemble tahminleri
predictions_ensemble = (ensemble_weight_efficientnet * predictions_efficientnet + 
                        ensemble_weight_resnet * predictions_resnet)

y_true = y_test.values.astype(int)

# Varsayılan threshold (0.5) ile değerlendirme
y_pred_ensemble = (predictions_ensemble > 0.5).astype(int).flatten()

print("Ensemble - Classification Report (Default Threshold 0.5):")
print(classification_report(y_true, y_pred_ensemble))
print("Ensemble - Confusion Matrix (Default Threshold 0.5):")
print(confusion_matrix(y_true, y_pred_ensemble))

# Custom threshold ile değerlendirme
y_pred_custom = (predictions_ensemble > custom_threshold).astype(int).flatten()

print(f"Ensemble - Classification Report (Custom Threshold {custom_threshold}):")
print(classification_report(y_true, y_pred_custom))
print(f"Ensemble - Confusion Matrix (Custom Threshold {custom_threshold}):")
print(confusion_matrix(y_true, y_pred_custom))

# Post-processing ile FN azaltma
post_processed_pred = np.where(predictions_ensemble > post_process_threshold, 1, 0).flatten()

print(f"Post-Processed - Classification Report (Threshold {post_process_threshold}):")
print(classification_report(y_true, post_processed_pred))
print(f"Post-Processed - Confusion Matrix (Threshold {post_process_threshold}):")
print(confusion_matrix(y_true, post_processed_pred))

# Hatalı tahmin analizi
print("\nKanserli sınıfta (target=1) hatalı tahmin edilen resimler (Custom Threshold):")
misclassified_malignant = []
original_malignant_count = 0
synthetic_malignant_count = 0

X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

for i in range(len(y_true)):
    if y_true[i] == 1 and y_pred_custom[i] != y_true[i]:
        image_name = X_test_reset.iloc[i]['image_path'].split('/')[-1]
        misclassified_malignant.append(image_name)
        print(f"Hatalı tahmin edilen resim: {image_name}")
        if 'gan_malignant_' in image_name or 'augmented_malignant_' in image_name:
            synthetic_malignant_count += 1
        else:
            original_malignant_count += 1

print(f"\nToplam hatalı tahmin edilen kanserli resim sayısı: {len(misclassified_malignant)}")
print(f"Orijinal malignantlardan hatalı tahmin edilen: {original_malignant_count}")
print(f"GAN veya augmentasyon ile üretilen sentetik malignantlardan hatalı tahmin edilen: {synthetic_malignant_count}")


import pandas as pd
import tensorflow as tf
import os

# Test verisi yolları
test_csv_path = '/kaggle/input/siim-isic-melanoma-classification/test.csv'
test_images_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

# Hiperparametreler
image_size = (224, 224)
batch_size = 128
custom_threshold = 0.45  # Custom Threshold
ensemble_weight_efficientnet = 0.5286  # Validation setinden hesaplanan ağırlık
ensemble_weight_resnet = 0.4714  # Validation setinden hesaplanan ağırlık

# Submission dosyası
print("\nAsıl test verisi ile submission.csv oluşturuluyor...")
test_df = pd.read_csv(test_csv_path)
test_df['image_path'] = test_df['image_name'].apply(lambda x: os.path.join(test_images_path, x + '.jpg'))

# Test verisi için metadata’yı işleme
test_df['sex'] = test_df['sex'].fillna('unknown')
test_df['age_approx'] = test_df['age_approx'].fillna(test_df['age_approx'].mean())
test_df['anatom_site_general_challenge'] = test_df['anatom_site_general_challenge'].fillna('unknown')

test_df['sex_encoded'] = label_encoder_sex.transform(test_df['sex'])
test_df['anatom_site_encoded'] = label_encoder_site.transform(test_df['anatom_site_general_challenge'])
test_df['age_approx'] = (test_df['age_approx'] - test_df['age_approx'].mean()) / test_df['age_approx'].std()

# Test dataset oluşturma fonksiyonu
def load_and_preprocess_test_image(image_path, sex, age, site):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return img, metadata

def create_submission_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site: load_and_preprocess_test_image(img_path, sex, age, site),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

test_dataset_submission = create_submission_dataset(test_df, batch_size)

# Dataset’ten görüntü ve metadata’yı ayırarak predict için hazırlık
def prepare_inputs(dataset):
    images = []
    metadatas = []
    for img, metadata in dataset:
        images.append(img)
        metadatas.append(metadata)
    return tf.concat(images, axis=0), tf.concat(metadatas, axis=0)

# Görüntü ve metadata’yı ayır
print("Görüntü ve metadata ayrılıyor...")
images, metadatas = prepare_inputs(test_dataset_submission)

# Her iki modelle submission tahmini
print("EfficientNetB0 ile tahmin yapılıyor...")
predictions_submission_efficientnet = model_efficientnet.predict([images, metadatas], batch_size=batch_size)

print("ResNet50V2 ile tahmin yapılıyor...")
predictions_submission_resnet = model_resnet.predict([images, metadatas], batch_size=batch_size)

# Ensemble tahmini
print("Ensemble tahmini oluşturuluyor...")
predictions_submission_ensemble = (ensemble_weight_efficientnet * predictions_submission_efficientnet + 
                                   ensemble_weight_resnet * predictions_submission_resnet)

# Custom threshold ile tahmin
print(f"Custom Threshold ({custom_threshold}) ile sınıflandırma yapılıyor...")
test_df['target'] = (predictions_submission_ensemble > custom_threshold).astype(int).flatten()

# Submission.csv dosyasını oluştur
submission = test_df[['image_name', 'target']]
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("submission.csv oluşturuldu: /kaggle/working/submission.csv")


import pandas as pd
import tensorflow as tf
import os
import numpy as np

# Test verisi yolları
test_csv_path = '/kaggle/input/siim-isic-melanoma-classification/test.csv'
test_images_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

# Hiperparametreler
image_size = (224, 224)
batch_size = 32  # Batch size zaten düşürülmüştü
ensemble_weight_efficientnet = 0.5286  # Validation setinden hesaplanan ağırlık
ensemble_weight_resnet = 0.4714  # Validation setinden hesaplanan ağırlık

# Submission dosyası
print("\nAsıl test verisi ile submission.csv oluşturuluyor...")
test_df = pd.read_csv(test_csv_path)
test_df['image_path'] = test_df['image_name'].apply(lambda x: os.path.join(test_images_path, x + '.jpg'))

# Test verisi için metadata’yı işleme
test_df['sex'] = test_df['sex'].fillna('unknown')
test_df['age_approx'] = test_df['age_approx'].fillna(test_df['age_approx'].mean())
test_df['anatom_site_general_challenge'] = test_df['anatom_site_general_challenge'].fillna('unknown')

# Label encoder’ların mevcut olduğundan emin ol
try:
    test_df['sex_encoded'] = label_encoder_sex.transform(test_df['sex'])
    test_df['anatom_site_encoded'] = label_encoder_site.transform(test_df['anatom_site_general_challenge'])
except NameError:
    from sklearn.preprocessing import LabelEncoder
    print("Label encoder’lar bulunamadı, yeniden oluşturuluyor...")
    label_encoder_sex = LabelEncoder()
    label_encoder_site = LabelEncoder()
    # Eğitim verisinden label encoder’ları oluştur (eğer mevcut değilse)
    train_df = pd.read_csv('/kaggle/working/new_train.csv')
    label_encoder_sex.fit(train_df['sex'])
    label_encoder_site.fit(train_df['anatom_site_general_challenge'])
    test_df['sex_encoded'] = label_encoder_sex.transform(test_df['sex'])
    test_df['anatom_site_encoded'] = label_encoder_site.transform(test_df['anatom_site_general_challenge'])

test_df['age_approx'] = (test_df['age_approx'] - test_df['age_approx'].mean()) / test_df['age_approx'].std()

# Test dataset oluşturma fonksiyonu
def load_and_preprocess_test_image(image_path, sex, age, site):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return img, metadata

def create_submission_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site: load_and_preprocess_test_image(img_path, sex, age, site),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

test_dataset_submission = create_submission_dataset(test_df, batch_size)

# Batch’ler halinde tahmin yapma
print("EfficientNetB0 ile tahmin yapılıyor...")
predictions_efficientnet = []
for batch_images, batch_metadatas in test_dataset_submission:
    batch_predictions = model_efficientnet.predict([batch_images, batch_metadatas], batch_size=batch_size, verbose=0)
    predictions_efficientnet.append(batch_predictions)

# Tüm batch’lerin tahminlerini birleştir
predictions_submission_efficientnet = np.concatenate(predictions_efficientnet, axis=0)

print("ResNet50V2 ile tahmin yapılıyor...")
predictions_resnet = []
for batch_images, batch_metadatas in test_dataset_submission:
    batch_predictions = model_resnet.predict([batch_images, batch_metadatas], batch_size=batch_size, verbose=0)
    predictions_resnet.append(batch_predictions)

# Tüm batch’lerin tahminlerini birleştir
predictions_submission_resnet = np.concatenate(predictions_resnet, axis=0)

# Ensemble tahmini
print("Ensemble tahmini oluşturuluyor...")
predictions_submission_ensemble = (ensemble_weight_efficientnet * predictions_submission_efficientnet + 
                                   ensemble_weight_resnet * predictions_submission_resnet)

# Ham olasılık değerlerini kullan (threshold uygulamadan)
print("Ham olasılık değerleri submission için hazırlanıyor...")
test_df['target'] = predictions_submission_ensemble.flatten()  # Ham olasılık değerleri (0-1 arası)

# Submission.csv dosyasını oluştur
submission = test_df[['image_name', 'target']]
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("submission.csv oluşturuldu: /kaggle/working/submission23.csv")


import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import DenseNet201

# GPU bellek optimizasyonu
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU aktif!")
else:
    print("GPU bulunamadı, CPU kullanılacak.")

# Hiperparametreler
image_size = (224, 224)
batch_size = 64  # DenseNet201 daha büyük bir model, batch_size'ı düşürdüm
epochs_model_densenet = 60  # 70 epoch
custom_threshold = 0.45
post_process_threshold = 0.35

# Veri yolları
new_train_csv_path = '/kaggle/working/new_train.csv'
train_images_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
gan_images_path = '/kaggle/working/gan_generated_malignant/'
aug_images_path = '/kaggle/working/augmented_malignant/'
test_csv_path = '/kaggle/input/siim-isic-melanoma-classification/test.csv'
test_images_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

# Yeni train.csv’yi yükle (GAN ve augmentasyon sonrası veri)
print("Yeni train.csv yükleniyor...")
train_df_final = pd.read_csv(new_train_csv_path)
print(f"GAN ve augmentasyon sonrası veri seti boyutu: {train_df_final.shape}")

# Malignant sınıfının 3,500 olduğunu kontrol et
assert len(train_df_final[train_df_final['target'] == 1]) == 3500, "Malignant sınıfı 3,500’e ulaşmadı!"

# Görüntü yollarını güncelle
def get_image_path(image_name):
    if 'gan_malignant_' in image_name:
        return os.path.join(gan_images_path, image_name)
    elif 'augmented_malignant_' in image_name:
        return os.path.join(aug_images_path, image_name)
    else:
        return os.path.join(train_images_path, image_name + '.jpg')

# Hasta metadata’sını işleme
label_encoder_sex = LabelEncoder()
label_encoder_site = LabelEncoder()
train_df_final['sex'] = train_df_final['sex'].fillna('unknown')
train_df_final['age_approx'] = train_df_final['age_approx'].fillna(train_df_final['age_approx'].mean())
train_df_final['anatom_site_general_challenge'] = train_df_final['anatom_site_general_challenge'].fillna('unknown')
train_df_final['sex_encoded'] = label_encoder_sex.fit_transform(train_df_final['sex'])
train_df_final['anatom_site_encoded'] = label_encoder_site.fit_transform(train_df_final['anatom_site_general_challenge'])
train_df_final['age_approx'] = (train_df_final['age_approx'] - train_df_final['age_approx'].mean()) / train_df_final['age_approx'].std()

train_df_final['image_path'] = train_df_final['image_name'].apply(get_image_path)
train_df_final['target'] = train_df_final['target'].astype(float)

# Train, val ve test setlerini böl (%80 train, %10 val, %10 test)
X = train_df_final[['image_path', 'sex_encoded', 'age_approx', 'anatom_site_encoded']]
y = train_df_final['target']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Eğitim seti: {len(X_train)} örnek (%80)")
print(f"Doğrulama seti: {len(X_val)} örnek (%10)")
print(f"Test seti: {len(X_test)} örnek (%10)")

# Veri pipeline’ı (Görüntü ve metadata’yı birleştirme)
def load_and_preprocess_image(image_path, sex, age, site, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    if 'train' in str(image_path):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.3)
        img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
        img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
        img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    img = tf.keras.applications.densenet.preprocess_input(img)  # DenseNet için preprocess_input
    label = tf.cast(label, tf.float32)
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return (img, metadata), label

def create_dataset(df, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values,
        df['target'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site, label: load_and_preprocess_image(img_path, sex, age, site, label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def create_test_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values,
        df['target'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site, label: load_and_preprocess_image(img_path, sex, age, site, label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(pd.concat([X_train, y_train], axis=1), batch_size, shuffle=True)
val_dataset = create_dataset(pd.concat([X_val, y_val], axis=1), batch_size, shuffle=False)
test_dataset = create_test_dataset(pd.concat([X_test, y_test], axis=1), batch_size)

# Steps per epoch ve validation steps
steps_per_epoch = len(X_train) // batch_size
validation_steps = len(X_val) // batch_size

# Modeli oluştur (Görüntü ve metadata’yı birleştirme)
def build_model_with_metadata(base_model):
    base_model.trainable = True
    for layer in base_model.layers[:-20]:  # Daha fazla katmanı donduruyoruz
        layer.trainable = False

    # Görüntü girişi
    image_input = Input(shape=(224, 224, 3), name='image_input')
    x = base_model(image_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Metadata girişi
    metadata_input = Input(shape=(3,), name='metadata_input')
    m = layers.Dense(16, activation='relu')(metadata_input)
    m = layers.Dense(8, activation='relu')(m)

    # Görüntü ve metadata’yı birleştirme
    combined = layers.concatenate([x, m])
    combined = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
    combined = layers.Dropout(0.5)(combined)
    output = layers.Dense(1, activation='sigmoid')(combined)

    model = models.Model(inputs=[image_input, metadata_input], outputs=output)
    return model

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # Daha uzun sabır
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
checkpoint_densenet = ModelCheckpoint('/kaggle/working/densenet_best_model.keras',
                                     monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Class weight
class_weight = {0: 1.0, 1: 8.0}

# Focal Loss
def focal_loss(gamma=2.0, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1) - (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0)
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# DenseNet201 Modeli
print("DenseNet201 modeli eğitiliyor...")
base_model_densenet = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_densenet = build_model_with_metadata(base_model_densenet)

# Learning rate scheduler (cosine annealing)
initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, steps_per_epoch * epochs_model_densenet
)

model_densenet.compile(optimizer=Adam(learning_rate=lr_schedule), loss=focal_loss(), metrics=['accuracy'])

# Model eğitimi
model_densenet.fit(train_dataset, 
                   epochs=epochs_model_densenet, 
                   validation_data=val_dataset, 
                   steps_per_epoch=steps_per_epoch, 
                   validation_steps=validation_steps,
                   class_weight=class_weight, 
                   callbacks=[early_stopping, reduce_lr, checkpoint_densenet])

# Test seti üzerinde tahmin
print("Test seti üzerinde tahmin yapılıyor...")
predictions_densenet = model_densenet.predict(test_dataset)

y_true = y_test.values.astype(int)

# Varsayılan threshold (0.5) ile değerlendirme
y_pred_densenet = (predictions_densenet > 0.5).astype(int).flatten()

print("DenseNet201 - Classification Report (Default Threshold 0.5):")
print(classification_report(y_true, y_pred_densenet))
print("DenseNet201 - Confusion Matrix (Default Threshold 0.5):")
print(confusion_matrix(y_true, y_pred_densenet))

# Custom threshold ile değerlendirme
y_pred_custom = (predictions_densenet > custom_threshold).astype(int).flatten()

print(f"DenseNet201 - Classification Report (Custom Threshold {custom_threshold}):")
print(classification_report(y_true, y_pred_custom))
print(f"DenseNet201 - Confusion Matrix (Custom Threshold {custom_threshold}):")
print(confusion_matrix(y_true, y_pred_custom))

# Post-processing ile FN azaltma
post_processed_pred = np.where(predictions_densenet > post_process_threshold, 1, 0).flatten()

print(f"DenseNet201 Post-Processed - Classification Report (Threshold {post_process_threshold}):")
print(classification_report(y_true, post_processed_pred))
print(f"DenseNet201 Post-Processed - Confusion Matrix (Threshold {post_process_threshold}):")
print(confusion_matrix(y_true, post_processed_pred))

# Hatalı tahmin analizi
print("\nKanserli sınıfta (target=1) hatalı tahmin edilen resimler (Custom Threshold):")
misclassified_malignant = []
original_malignant_count = 0
synthetic_malignant_count = 0

X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

for i in range(len(y_true)):
    if y_true[i] == 1 and y_pred_custom[i] != y_true[i]:
        image_name = X_test_reset.iloc[i]['image_path'].split('/')[-1]
        misclassified_malignant.append(image_name)
        print(f"Hatalı tahmin edilen resim: {image_name}")
        if 'gan_malignant_' in image_name or 'augmented_malignant_' in image_name:
            synthetic_malignant_count += 1
        else:
            original_malignant_count += 1

print(f"\nToplam hatalı tahmin edilen kanserli resim sayısı: {len(misclassified_malignant)}")
print(f"Orijinal malignantlardan hatalı tahmin edilen: {original_malignant_count}")
print(f"GAN veya augmentasyon ile üretilen sentetik malignantlardan hatalı tahmin edilen: {synthetic_malignant_count}")

# Submission dosyası
print("\nAsıl test verisi ile submission.csv oluşturuluyor...")
test_df = pd.read_csv(test_csv_path)
test_df['image_path'] = test_df['image_name'].apply(lambda x: os.path.join(test_images_path, x + '.jpg'))

# Test verisi için metadata’yı işleme
test_df['sex'] = test_df['sex'].fillna('unknown')
test_df['age_approx'] = test_df['age_approx'].fillna(test_df['age_approx'].mean())
test_df['anatom_site_general_challenge'] = test_df['anatom_site_general_challenge'].fillna('unknown')

test_df['sex_encoded'] = label_encoder_sex.transform(test_df['sex'])
test_df['anatom_site_encoded'] = label_encoder_site.transform(test_df['anatom_site_general_challenge'])
test_df['age_approx'] = (test_df['age_approx'] - test_df['age_approx'].mean()) / test_df['age_approx'].std()

def load_and_preprocess_test_image(image_path, sex, age, site):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.keras.applications.densenet.preprocess_input(img)  # DenseNet için preprocess_input
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return img, metadata

def create_submission_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site: load_and_preprocess_test_image(img_path, sex, age, site),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

test_dataset_submission = create_submission_dataset(test_df, batch_size)

# DenseNet201 ile submission tahmini
print("DenseNet201 ile tahmin yapılıyor...")
predictions_submission_densenet = model_densenet.predict(test_dataset_submission)

# Custom threshold ile tahmin
test_df['target'] = (predictions_submission_densenet > custom_threshold).astype(int).flatten()

submission = test_df[['image_name', 'target']]
submission.to_csv('/kaggle/working/submission_densenet.csv', index=False)
print("submission_densenet.csv oluşturuldu: /kaggle/working/submission_densenet.csv")


import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers
from tensorflow.keras.applications import DenseNet201

# GPU bellek optimizasyonu
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU aktif!")
else:
    print("GPU bulunamadı, CPU kullanılacak.")

# Hiperparametreler
image_size = (224, 224)
batch_size = 32  # Test için batch_size'ı 32 olarak tutuyorum
custom_threshold = 0.45
post_process_threshold = 0.35

# Veri yolları
new_train_csv_path = '/kaggle/working/new_train.csv'
train_images_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
gan_images_path = '/kaggle/working/gan_generated_malignant/'
aug_images_path = '/kaggle/working/augmented_malignant/'
test_csv_path = '/kaggle/input/siim-isic-melanoma-classification/test.csv'
test_images_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

# Yeni train.csv’yi yükle (GAN ve augmentasyon sonrası veri)
print("Yeni train.csv yükleniyor...")
train_df_final = pd.read_csv(new_train_csv_path)
print(f"GAN ve augmentasyon sonrası veri seti boyutu: {train_df_final.shape}")

# Malignant sınıfının 3,500 olduğunu kontrol et
assert len(train_df_final[train_df_final['target'] == 1]) == 3500, "Malignant sınıfı 3,500’e ulaşmadı!"

# Görüntü yollarını güncelle
def get_image_path(image_name):
    if 'gan_malignant_' in image_name:
        return os.path.join(gan_images_path, image_name)
    elif 'augmented_malignant_' in image_name:
        return os.path.join(aug_images_path, image_name)
    else:
        return os.path.join(train_images_path, image_name + '.jpg')

# Hasta metadata’sını işleme
label_encoder_sex = LabelEncoder()
label_encoder_site = LabelEncoder()
train_df_final['sex'] = train_df_final['sex'].fillna('unknown')
train_df_final['age_approx'] = train_df_final['age_approx'].fillna(train_df_final['age_approx'].mean())
train_df_final['anatom_site_general_challenge'] = train_df_final['anatom_site_general_challenge'].fillna('unknown')
train_df_final['sex_encoded'] = label_encoder_sex.fit_transform(train_df_final['sex'])
train_df_final['anatom_site_encoded'] = label_encoder_site.fit_transform(train_df_final['anatom_site_general_challenge'])
train_df_final['age_approx'] = (train_df_final['age_approx'] - train_df_final['age_approx'].mean()) / train_df_final['age_approx'].std()

train_df_final['image_path'] = train_df_final['image_name'].apply(get_image_path)
train_df_final['target'] = train_df_final['target'].astype(float)

# Train, val ve test setlerini böl (%80 train, %10 val, %10 test)
X = train_df_final[['image_path', 'sex_encoded', 'age_approx', 'anatom_site_encoded']]
y = train_df_final['target']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Eğitim seti: {len(X_train)} örnek (%80)")
print(f"Doğrulama seti: {len(X_val)} örnek (%10)")
print(f"Test seti: {len(X_test)} örnek (%10)")

# Veri pipeline’ı (Görüntü ve metadata’yı birleştirme)
def load_and_preprocess_image(image_path, sex, age, site, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.keras.applications.densenet.preprocess_input(img)  # DenseNet için preprocess_input
    label = tf.cast(label, tf.float32)
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return (img, metadata), label

def create_test_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values,
        df['target'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site, label: load_and_preprocess_image(img_path, sex, age, site, label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()  # Veri setinin tükenmesini önlemek için
    return dataset

test_dataset = create_test_dataset(pd.concat([X_test, y_test], axis=1), batch_size)

# Test seti için steps
test_steps = (len(X_test) + batch_size - 1) // batch_size
print(f"Test steps: {test_steps}")

# Modeli oluştur (Görüntü ve metadata’yı birleştirme)
def build_model_with_metadata(base_model):
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Görüntü girişi
    image_input = Input(shape=(224, 224, 3), name='image_input')
    x = base_model(image_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Metadata girişi
    metadata_input = Input(shape=(3,), name='metadata_input')
    m = layers.Dense(16, activation='relu')(metadata_input)
    m = layers.Dense(8, activation='relu')(m)

    # Görüntü ve metadata’yı birleştirme
    combined = layers.concatenate([x, m])
    combined = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
    combined = layers.Dropout(0.5)(combined)
    output = layers.Dense(1, activation='sigmoid')(combined)

    model = models.Model(inputs=[image_input, metadata_input], outputs=output)
    return model

# Focal Loss
def focal_loss(gamma=2.0, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1) - (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0)
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# DenseNet201 Modeli
print("DenseNet201 modeli oluşturuluyor...")
base_model_densenet = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_densenet = build_model_with_metadata(base_model_densenet)

# Modeli derleme
model_densenet.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])

# Kaydedilmiş ağırlıkları yükleme
print("Kaydedilmiş ağırlıklar yükleniyor...")
model_densenet.load_weights('/kaggle/working/densenet_best_model.keras')

# Test seti üzerinde tahmin
print("Test seti üzerinde tahmin yapılıyor...")
predictions_densenet = model_densenet.predict(test_dataset, steps=test_steps)

y_true = y_test.values.astype(int)

# Varsayılan threshold (0.5) ile değerlendirme
y_pred_densenet = (predictions_densenet > 0.5).astype(int).flatten()

print("DenseNet201 - Classification Report (Default Threshold 0.5):")
print(classification_report(y_true, y_pred_densenet))
print("DenseNet201 - Confusion Matrix (Default Threshold 0.5):")
print(confusion_matrix(y_true, y_pred_densenet))

# Custom threshold ile değerlendirme
y_pred_custom = (predictions_densenet > custom_threshold).astype(int).flatten()

print(f"DenseNet201 - Classification Report (Custom Threshold {custom_threshold}):")
print(classification_report(y_true, y_pred_custom))
print(f"DenseNet201 - Confusion Matrix (Custom Threshold {custom_threshold}):")
print(confusion_matrix(y_true, y_pred_custom))

# Post-processing ile FN azaltma
post_processed_pred = np.where(predictions_densenet > post_process_threshold, 1, 0).flatten()

print(f"DenseNet201 Post-Processed - Classification Report (Threshold {post_process_threshold}):")
print(classification_report(y_true, post_processed_pred))
print(f"DenseNet201 Post-Processed - Confusion Matrix (Threshold {post_process_threshold}):")
print(confusion_matrix(y_true, post_processed_pred))

# Hatalı tahmin analizi
print("\nKanserli sınıfta (target=1) hatalı tahmin edilen resimler (Custom Threshold):")
misclassified_malignant = []
original_malignant_count = 0
synthetic_malignant_count = 0

X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

for i in range(len(y_true)):
    if y_true[i] == 1 and y_pred_custom[i] != y_true[i]:
        image_name = X_test_reset.iloc[i]['image_path'].split('/')[-1]
        misclassified_malignant.append(image_name)
        print(f"Hatalı tahmin edilen resim: {image_name}")
        if 'gan_malignant_' in image_name or 'augmented_malignant_' in image_name:
            synthetic_malignant_count += 1
        else:
            original_malignant_count += 1

print(f"\nToplam hatalı tahmin edilen kanserli resim sayısı: {len(misclassified_malignant)}")
print(f"Orijinal malignantlardan hatalı tahmin edilen: {original_malignant_count}")
print(f"GAN veya augmentasyon ile üretilen sentetik malignantlardan hatalı tahmin edilen: {synthetic_malignant_count}")

# Submission dosyası
print("\nAsıl test verisi ile submission.csv oluşturuluyor...")
test_df = pd.read_csv(test_csv_path)
test_df['image_path'] = test_df['image_name'].apply(lambda x: os.path.join(test_images_path, x + '.jpg'))

# Test verisi için metadata’yı işleme
test_df['sex'] = test_df['sex'].fillna('unknown')
test_df['age_approx'] = test_df['age_approx'].fillna(test_df['age_approx'].mean())
test_df['anatom_site_general_challenge'] = test_df['anatom_site_general_challenge'].fillna('unknown')

test_df['sex_encoded'] = label_encoder_sex.transform(test_df['sex'])
test_df['anatom_site_encoded'] = label_encoder_site.transform(test_df['anatom_site_general_challenge'])
test_df['age_approx'] = (test_df['age_approx'] - test_df['age_approx'].mean()) / test_df['age_approx'].std()

def load_and_preprocess_test_image(image_path, sex, age, site):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.keras.applications.densenet.preprocess_input(img)  # DenseNet için preprocess_input
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return img, metadata

def create_submission_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site: load_and_preprocess_test_image(img_path, sex, age, site),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()  # Veri setinin tükenmesini önlemek için
    return dataset

test_dataset_submission = create_submission_dataset(test_df, batch_size)
submission_steps = (len(test_df) + batch_size - 1) // batch_size

# DenseNet201 ile submission tahmini
print("DenseNet201 ile tahmin yapılıyor...")
predictions_submission_densenet = model_densenet.predict(test_dataset_submission, steps=submission_steps)

# Custom threshold ile tahmin
test_df['target'] = (predictions_submission_densenet > custom_threshold).astype(int).flatten()

submission = test_df[['image_name', 'target']]
submission.to_csv('/kaggle/working/submission_densenet_epoch12.csv', index=False)
print("submission_densenet_epoch12.csv oluşturuldu: /kaggle/working/submission_densenet_epoch12.csv")


import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers
from tensorflow.keras.applications import DenseNet201

# GPU bellek optimizasyonu
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU aktif!")
else:
    print("GPU bulunamadı, CPU kullanılacak.")

# Hiperparametreler
image_size = (224, 224)
batch_size = 32  # Test için batch_size'ı 32 olarak tutuyorum
custom_threshold = 0.45
post_process_threshold = 0.35

# Veri yolları
new_train_csv_path = '/kaggle/working/new_train.csv'
train_images_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
gan_images_path = '/kaggle/working/gan_generated_malignant/'
aug_images_path = '/kaggle/working/augmented_malignant/'
test_csv_path = '/kaggle/input/siim-isic-melanoma-classification/test.csv'
test_images_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

# Yeni train.csv’yi yükle (GAN ve augmentasyon sonrası veri)
print("Yeni train.csv yükleniyor...")
train_df_final = pd.read_csv(new_train_csv_path)
print(f"GAN ve augmentasyon sonrası veri seti boyutu: {train_df_final.shape}")

# Malignant sınıfının 3,500 olduğunu kontrol et
assert len(train_df_final[train_df_final['target'] == 1]) == 3500, "Malignant sınıfı 3,500’e ulaşmadı!"

# Görüntü yollarını güncelle
def get_image_path(image_name):
    if 'gan_malignant_' in image_name:
        return os.path.join(gan_images_path, image_name)
    elif 'augmented_malignant_' in image_name:
        return os.path.join(aug_images_path, image_name)
    else:
        return os.path.join(train_images_path, image_name + '.jpg')

# Hasta metadata’sını işleme
label_encoder_sex = LabelEncoder()
label_encoder_site = LabelEncoder()
train_df_final['sex'] = train_df_final['sex'].fillna('unknown')
train_df_final['age_approx'] = train_df_final['age_approx'].fillna(train_df_final['age_approx'].mean())
train_df_final['anatom_site_general_challenge'] = train_df_final['anatom_site_general_challenge'].fillna('unknown')
train_df_final['sex_encoded'] = label_encoder_sex.fit_transform(train_df_final['sex'])
train_df_final['anatom_site_encoded'] = label_encoder_site.fit_transform(train_df_final['anatom_site_general_challenge'])
train_df_final['age_approx'] = (train_df_final['age_approx'] - train_df_final['age_approx'].mean()) / train_df_final['age_approx'].std()

train_df_final['image_path'] = train_df_final['image_name'].apply(get_image_path)
train_df_final['target'] = train_df_final['target'].astype(float)

# Train, val ve test setlerini böl (%80 train, %10 val, %10 test)
X = train_df_final[['image_path', 'sex_encoded', 'age_approx', 'anatom_site_encoded']]
y = train_df_final['target']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Eğitim seti: {len(X_train)} örnek (%80)")
print(f"Doğrulama seti: {len(X_val)} örnek (%10)")
print(f"Test seti: {len(X_test)} örnek (%10)")

# Veri pipeline’ı (Görüntü ve metadata’yı birleştirme)
def load_and_preprocess_image(image_path, sex, age, site, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.keras.applications.densenet.preprocess_input(img)  # DenseNet için preprocess_input
    label = tf.cast(label, tf.float32)
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return (img, metadata), label

def create_test_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values,
        df['target'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site, label: load_and_preprocess_image(img_path, sex, age, site, label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()  # Veri setinin tükenmesini önlemek için
    return dataset

test_dataset = create_test_dataset(pd.concat([X_test, y_test], axis=1), batch_size)

# Test seti için steps
test_steps = (len(X_test) + batch_size - 1) // batch_size
print(f"Test steps: {test_steps}")

# Modeli oluştur (Görüntü ve metadata’yı birleştirme)
def build_model_with_metadata(base_model):
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Görüntü girişi
    image_input = Input(shape=(224, 224, 3), name='image_input')
    x = base_model(image_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Metadata girişi
    metadata_input = Input(shape=(3,), name='metadata_input')
    m = layers.Dense(16, activation='relu')(metadata_input)
    m = layers.Dense(8, activation='relu')(m)

    # Görüntü ve metadata’yı birleştirme
    combined = layers.concatenate([x, m])
    combined = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
    combined = layers.Dropout(0.5)(combined)
    output = layers.Dense(1, activation='sigmoid')(combined)

    model = models.Model(inputs=[image_input, metadata_input], outputs=output)
    return model

# Focal Loss
def focal_loss(gamma=2.0, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1) - (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0)
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# DenseNet201 Modeli
print("DenseNet201 modeli oluşturuluyor...")
base_model_densenet = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_densenet = build_model_with_metadata(base_model_densenet)

# Modeli derleme
model_densenet.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])

# Kaydedilmiş ağırlıkları yükleme
print("Kaydedilmiş ağırlıklar yükleniyor...")
model_densenet.load_weights('/kaggle/working/densenet_best_model.keras')

# Test seti üzerinde tahmin
print("Test seti üzerinde tahmin yapılıyor...")
predictions_densenet = model_densenet.predict(test_dataset, steps=test_steps)

y_true = y_test.values.astype(int)

# Varsayılan threshold (0.5) ile değerlendirme
y_pred_densenet = (predictions_densenet > 0.5).astype(int).flatten()

print("DenseNet201 - Classification Report (Default Threshold 0.5):")
print(classification_report(y_true, y_pred_densenet))
print("DenseNet201 - Confusion Matrix (Default Threshold 0.5):")
print(confusion_matrix(y_true, y_pred_densenet))

# Custom threshold ile değerlendirme
y_pred_custom = (predictions_densenet > custom_threshold).astype(int).flatten()

print(f"DenseNet201 - Classification Report (Custom Threshold {custom_threshold}):")
print(classification_report(y_true, y_pred_custom))
print(f"DenseNet201 - Confusion Matrix (Custom Threshold {custom_threshold}):")
print(confusion_matrix(y_true, y_pred_custom))

# Post-processing ile FN azaltma
post_processed_pred = np.where(predictions_densenet > post_process_threshold, 1, 0).flatten()

print(f"DenseNet201 Post-Processed - Classification Report (Threshold {post_process_threshold}):")
print(classification_report(y_true, post_processed_pred))
print(f"DenseNet201 Post-Processed - Confusion Matrix (Threshold {post_process_threshold}):")
print(confusion_matrix(y_true, post_processed_pred))

# Hatalı tahmin analizi
print("\nKanserli sınıfta (target=1) hatalı tahmin edilen resimler (Custom Threshold):")
misclassified_malignant = []
original_malignant_count = 0
synthetic_malignant_count = 0

X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

for i in range(len(y_true)):
    if y_true[i] == 1 and y_pred_custom[i] != y_true[i]:
        image_name = X_test_reset.iloc[i]['image_path'].split('/')[-1]
        misclassified_malignant.append(image_name)
        print(f"Hatalı tahmin edilen resim: {image_name}")
        if 'gan_malignant_' in image_name or 'augmented_malignant_' in image_name:
            synthetic_malignant_count += 1
        else:
            original_malignant_count += 1

print(f"\nToplam hatalı tahmin edilen kanserli resim sayısı: {len(misclassified_malignant)}")
print(f"Orijinal malignantlardan hatalı tahmin edilen: {original_malignant_count}")
print(f"GAN veya augmentasyon ile üretilen sentetik malignantlardan hatalı tahmin edilen: {synthetic_malignant_count}")

# Submission dosyası
print("\nAsıl test verisi ile submission.csv oluşturuluyor...")
test_df = pd.read_csv(test_csv_path)
test_df['image_path'] = test_df['image_name'].apply(lambda x: os.path.join(test_images_path, x + '.jpg'))

# Test verisi için metadata’yı işleme
test_df['sex'] = test_df['sex'].fillna('unknown')
test_df['age_approx'] = test_df['age_approx'].fillna(test_df['age_approx'].mean())
test_df['anatom_site_general_challenge'] = test_df['anatom_site_general_challenge'].fillna('unknown')

test_df['sex_encoded'] = label_encoder_sex.transform(test_df['sex'])
test_df['anatom_site_encoded'] = label_encoder_site.transform(test_df['anatom_site_general_challenge'])
test_df['age_approx'] = (test_df['age_approx'] - test_df['age_approx'].mean()) / test_df['age_approx'].std()

def load_and_preprocess_test_image(image_path, sex, age, site):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.keras.applications.densenet.preprocess_input(img)  # DenseNet için preprocess_input
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return img, metadata  # İki ayrı çıkış döndürüyoruz

def create_submission_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site: load_and_preprocess_test_image(img_path, sex, age, site),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()  # Veri setinin tükenmesini önlemek için
    return dataset

test_dataset_submission = create_submission_dataset(test_df, batch_size)
submission_steps = (len(test_df) + batch_size - 1) // batch_size

# DenseNet201 ile submission tahmini
print("DenseNet201 ile tahmin yapılıyor...")

# Veri setini iki giriş olarak ayırmak için bir yardımcı fonksiyon
def split_inputs(dataset, steps):
    images = []
    metadata = []
    iterator = iter(dataset)
    for _ in range(steps):
        batch = next(iterator)
        images.append(batch[0])  # İlk eleman: img
        metadata.append(batch[1])  # İkinci eleman: metadata
    images = tf.concat(images, axis=0)
    metadata = tf.concat(metadata, axis=0)
    return [images, metadata]

# Veri setini ayır ve tahmin yap
inputs = split_inputs(test_dataset_submission, submission_steps)
predictions_submission_densenet = model_densenet.predict(inputs, batch_size=batch_size)

# Custom threshold ile tahmin
test_df['target'] = (predictions_submission_densenet > custom_threshold).astype(int).flatten()

submission = test_df[['image_name', 'target']]
submission.to_csv('/kaggle/working/submission_densenet_epoch12.csv', index=False)
print("submission_densenet_epoch12.csv oluşturuldu: /kaggle/working/submission_densenet_epoch12.csv")


import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers
from tensorflow.keras.applications import DenseNet201

# GPU bellek optimizasyonu
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU aktif!")
else:
    print("GPU bulunamadı, CPU kullanılacak.")

# Hiperparametreler
image_size = (224, 224)
batch_size = 32  # Test için batch_size'ı 32 olarak tutuyorum
custom_threshold = 0.45
post_process_threshold = 0.35

# Veri yolları
new_train_csv_path = '/kaggle/working/new_train.csv'
train_images_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
gan_images_path = '/kaggle/working/gan_generated_malignant/'
aug_images_path = '/kaggle/working/augmented_malignant/'
test_csv_path = '/kaggle/input/siim-isic-melanoma-classification/test.csv'
test_images_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

# Yeni train.csv’yi yükle (GAN ve augmentasyon sonrası veri)
print("Yeni train.csv yükleniyor...")
train_df_final = pd.read_csv(new_train_csv_path)
print(f"GAN ve augmentasyon sonrası veri seti boyutu: {train_df_final.shape}")

# Malignant sınıfının 3,500 olduğunu kontrol et
assert len(train_df_final[train_df_final['target'] == 1]) == 3500, "Malignant sınıfı 3,500’e ulaşmadı!"

# Görüntü yollarını güncelle
def get_image_path(image_name):
    if 'gan_malignant_' in image_name:
        return os.path.join(gan_images_path, image_name)
    elif 'augmented_malignant_' in image_name:
        return os.path.join(aug_images_path, image_name)
    else:
        return os.path.join(train_images_path, image_name + '.jpg')

# Hasta metadata’sını işleme
label_encoder_sex = LabelEncoder()
label_encoder_site = LabelEncoder()
train_df_final['sex'] = train_df_final['sex'].fillna('unknown')
train_df_final['age_approx'] = train_df_final['age_approx'].fillna(train_df_final['age_approx'].mean())
train_df_final['anatom_site_general_challenge'] = train_df_final['anatom_site_general_challenge'].fillna('unknown')
train_df_final['sex_encoded'] = label_encoder_sex.fit_transform(train_df_final['sex'])
train_df_final['anatom_site_encoded'] = label_encoder_site.fit_transform(train_df_final['anatom_site_general_challenge'])
train_df_final['age_approx'] = (train_df_final['age_approx'] - train_df_final['age_approx'].mean()) / train_df_final['age_approx'].std()

train_df_final['image_path'] = train_df_final['image_name'].apply(get_image_path)
train_df_final['target'] = train_df_final['target'].astype(float)

# Train, val ve test setlerini böl (%80 train, %10 val, %10 test)
X = train_df_final[['image_path', 'sex_encoded', 'age_approx', 'anatom_site_encoded']]
y = train_df_final['target']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Eğitim seti: {len(X_train)} örnek (%80)")
print(f"Doğrulama seti: {len(X_val)} örnek (%10)")
print(f"Test seti: {len(X_test)} örnek (%10)")

# Veri pipeline’ı (Görüntü ve metadata’yı birleştirme)
def load_and_preprocess_image(image_path, sex, age, site, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.keras.applications.densenet.preprocess_input(img)  # DenseNet için preprocess_input
    label = tf.cast(label, tf.float32)
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return (img, metadata), label

def create_test_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values,
        df['target'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site, label: load_and_preprocess_image(img_path, sex, age, site, label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()  # Veri setinin tükenmesini önlemek için
    return dataset

test_dataset = create_test_dataset(pd.concat([X_test, y_test], axis=1), batch_size)

# Test seti için steps
test_steps = (len(X_test) + batch_size - 1) // batch_size
print(f"Test steps: {test_steps}")

# Modeli oluştur (Görüntü ve metadata’yı birleştirme)
def build_model_with_metadata(base_model):
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Görüntü girişi
    image_input = Input(shape=(224, 224, 3), name='image_input')
    x = base_model(image_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Metadata girişi
    metadata_input = Input(shape=(3,), name='metadata_input')
    m = layers.Dense(16, activation='relu')(metadata_input)
    m = layers.Dense(8, activation='relu')(m)

    # Görüntü ve metadata’yı birleştirme
    combined = layers.concatenate([x, m])
    combined = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
    combined = layers.Dropout(0.5)(combined)
    output = layers.Dense(1, activation='sigmoid')(combined)

    model = models.Model(inputs=[image_input, metadata_input], outputs=output)
    return model

# Focal Loss
def focal_loss(gamma=2.0, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1) - (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0)
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# DenseNet201 Modeli
print("DenseNet201 modeli oluşturuluyor...")
base_model_densenet = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_densenet = build_model_with_metadata(base_model_densenet)

# Modeli derleme
model_densenet.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])

# Kaydedilmiş ağırlıkları yükleme
print("Kaydedilmiş ağırlıklar yükleniyor...")
model_densenet.load_weights('/kaggle/working/densenet_best_model.keras')

# Test seti üzerinde tahmin
print("Test seti üzerinde tahmin yapılıyor...")
predictions_densenet = model_densenet.predict(test_dataset, steps=test_steps)

y_true = y_test.values.astype(int)

# Varsayılan threshold (0.5) ile değerlendirme
y_pred_densenet = (predictions_densenet > 0.5).astype(int).flatten()

print("DenseNet201 - Classification Report (Default Threshold 0.5):")
print(classification_report(y_true, y_pred_densenet))
print("DenseNet201 - Confusion Matrix (Default Threshold 0.5):")
print(confusion_matrix(y_true, y_pred_densenet))

# Custom threshold ile değerlendirme
y_pred_custom = (predictions_densenet > custom_threshold).astype(int).flatten()

print(f"DenseNet201 - Classification Report (Custom Threshold {custom_threshold}):")
print(classification_report(y_true, y_pred_custom))
print(f"DenseNet201 - Confusion Matrix (Custom Threshold {custom_threshold}):")
print(confusion_matrix(y_true, y_pred_custom))

# Post-processing ile FN azaltma
post_processed_pred = np.where(predictions_densenet > post_process_threshold, 1, 0).flatten()

print(f"DenseNet201 Post-Processed - Classification Report (Threshold {post_process_threshold}):")
print(classification_report(y_true, post_processed_pred))
print(f"DenseNet201 Post-Processed - Confusion Matrix (Threshold {post_process_threshold}):")
print(confusion_matrix(y_true, post_processed_pred))

# Hatalı tahmin analizi
print("\nKanserli sınıfta (target=1) hatalı tahmin edilen resimler (Custom Threshold):")
misclassified_malignant = []
original_malignant_count = 0
synthetic_malignant_count = 0

X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

for i in range(len(y_true)):
    if y_true[i] == 1 and y_pred_custom[i] != y_true[i]:
        image_name = X_test_reset.iloc[i]['image_path'].split('/')[-1]
        misclassified_malignant.append(image_name)
        print(f"Hatalı tahmin edilen resim: {image_name}")
        if 'gan_malignant_' in image_name or 'augmented_malignant_' in image_name:
            synthetic_malignant_count += 1
        else:
            original_malignant_count += 1

print(f"\nToplam hatalı tahmin edilen kanserli resim sayısı: {len(misclassified_malignant)}")
print(f"Orijinal malignantlardan hatalı tahmin edilen: {original_malignant_count}")
print(f"GAN veya augmentasyon ile üretilen sentetik malignantlardan hatalı tahmin edilen: {synthetic_malignant_count}")

# Submission dosyası
print("\nAsıl test verisi ile submission.csv oluşturuluyor...")
test_df = pd.read_csv(test_csv_path)
test_df['image_path'] = test_df['image_name'].apply(lambda x: os.path.join(test_images_path, x + '.jpg'))

# Test verisi için metadata’yı işleme
test_df['sex'] = test_df['sex'].fillna('unknown')
test_df['age_approx'] = test_df['age_approx'].fillna(test_df['age_approx'].mean())
test_df['anatom_site_general_challenge'] = test_df['anatom_site_general_challenge'].fillna('unknown')

test_df['sex_encoded'] = label_encoder_sex.transform(test_df['sex'])
test_df['anatom_site_encoded'] = label_encoder_site.transform(test_df['anatom_site_general_challenge'])
test_df['age_approx'] = (test_df['age_approx'] - test_df['age_approx'].mean()) / test_df['age_approx'].std()

def load_and_preprocess_test_image(image_path, sex, age, site):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.keras.applications.densenet.preprocess_input(img)  # DenseNet için preprocess_input
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return img, metadata  # İki ayrı çıkış döndürüyoruz

def create_submission_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site: load_and_preprocess_test_image(img_path, sex, age, site),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()  # Veri setinin tükenmesini önlemek için
    return dataset

test_dataset_submission = create_submission_dataset(test_df, batch_size)
submission_steps = (len(test_df) + batch_size - 1) // batch_size

# DenseNet201 ile submission tahmini
print("DenseNet201 ile tahmin yapılıyor...")
# Doğrudan test_dataset_submission’ı batch’ler halinde predict’e veriyoruz
predictions_submission_densenet = model_densenet.predict(test_dataset_submission, steps=submission_steps)

# Custom threshold ile tahmin
test_df['target'] = (predictions_submission_densenet > custom_threshold).astype(int).flatten()

submission = test_df[['image_name', 'target']]
submission.to_csv('/kaggle/working/submission_densenet_epoch12.csv', index=False)
print("submission_densenet_epoch12.csv oluşturuldu: /kaggle/working/submission_densenet_epoch12.csv")


import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers
from tensorflow.keras.applications import DenseNet201

# GPU bellek optimizasyonu
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU aktif!")
else:
    print("GPU bulunamadı, CPU kullanılacak.")

# Hiperparametreler
image_size = (224, 224)
batch_size = 32  # Test için batch_size'ı 32 olarak tutuyorum
custom_threshold = 0.45
post_process_threshold = 0.35

# Veri yolları
new_train_csv_path = '/kaggle/working/new_train.csv'
train_images_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
gan_images_path = '/kaggle/working/gan_generated_malignant/'
aug_images_path = '/kaggle/working/augmented_malignant/'
test_csv_path = '/kaggle/input/siim-isic-melanoma-classification/test.csv'
test_images_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

# Yeni train.csv’yi yükle (GAN ve augmentasyon sonrası veri)
print("Yeni train.csv yükleniyor...")
train_df_final = pd.read_csv(new_train_csv_path)
print(f"GAN ve augmentasyon sonrası veri seti boyutu: {train_df_final.shape}")

# Malignant sınıfının 3,500 olduğunu kontrol et
assert len(train_df_final[train_df_final['target'] == 1]) == 3500, "Malignant sınıfı 3,500’e ulaşmadı!"

# Görüntü yollarını güncelle
def get_image_path(image_name):
    if 'gan_malignant_' in image_name:
        return os.path.join(gan_images_path, image_name)
    elif 'augmented_malignant_' in image_name:
        return os.path.join(aug_images_path, image_name)
    else:
        return os.path.join(train_images_path, image_name + '.jpg')

# Hasta metadata’sını işleme
label_encoder_sex = LabelEncoder()
label_encoder_site = LabelEncoder()
train_df_final['sex'] = train_df_final['sex'].fillna('unknown')
train_df_final['age_approx'] = train_df_final['age_approx'].fillna(train_df_final['age_approx'].mean())
train_df_final['anatom_site_general_challenge'] = train_df_final['anatom_site_general_challenge'].fillna('unknown')
train_df_final['sex_encoded'] = label_encoder_sex.fit_transform(train_df_final['sex'])
train_df_final['anatom_site_encoded'] = label_encoder_site.fit_transform(train_df_final['anatom_site_general_challenge'])
train_df_final['age_approx'] = (train_df_final['age_approx'] - train_df_final['age_approx'].mean()) / train_df_final['age_approx'].std()

train_df_final['image_path'] = train_df_final['image_name'].apply(get_image_path)
train_df_final['target'] = train_df_final['target'].astype(float)

# Train, val ve test setlerini böl (%80 train, %10 val, %10 test)
X = train_df_final[['image_path', 'sex_encoded', 'age_approx', 'anatom_site_encoded']]
y = train_df_final['target']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Eğitim seti: {len(X_train)} örnek (%80)")
print(f"Doğrulama seti: {len(X_val)} örnek (%10)")
print(f"Test seti: {len(X_test)} örnek (%10)")

# Veri pipeline’ı (Görüntü ve metadata’yı birleştirme)
def load_and_preprocess_image(image_path, sex, age, site, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.keras.applications.densenet.preprocess_input(img)  # DenseNet için preprocess_input
    label = tf.cast(label, tf.float32)
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return (img, metadata), label

def create_test_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values,
        df['target'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site, label: load_and_preprocess_image(img_path, sex, age, site, label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()  # Veri setinin tükenmesini önlemek için
    return dataset

test_dataset = create_test_dataset(pd.concat([X_test, y_test], axis=1), batch_size)

# Test seti için steps
test_steps = (len(X_test) + batch_size - 1) // batch_size
print(f"Test steps: {test_steps}")

# Modeli oluştur (Görüntü ve metadata’yı birleştirme)
def build_model_with_metadata(base_model):
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Görüntü girişi
    image_input = Input(shape=(224, 224, 3), name='image_input')
    x = base_model(image_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Metadata girişi
    metadata_input = Input(shape=(3,), name='metadata_input')
    m = layers.Dense(16, activation='relu')(metadata_input)
    m = layers.Dense(8, activation='relu')(m)

    # Görüntü ve metadata’yı birleştirme
    combined = layers.concatenate([x, m])
    combined = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
    combined = layers.Dropout(0.5)(combined)
    output = layers.Dense(1, activation='sigmoid')(combined)

    model = models.Model(inputs=[image_input, metadata_input], outputs=output)
    return model

# Focal Loss
def focal_loss(gamma=2.0, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1) - (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0)
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# DenseNet201 Modeli
print("DenseNet201 modeli oluşturuluyor...")
base_model_densenet = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_densenet = build_model_with_metadata(base_model_densenet)

# Modeli derleme
model_densenet.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])

# Kaydedilmiş ağırlıkları yükleme
print("Kaydedilmiş ağırlıklar yükleniyor...")
model_densenet.load_weights('/kaggle/working/densenet_best_model.keras')

# Test seti üzerinde tahmin
print("Test seti üzerinde tahmin yapılıyor...")
predictions_densenet = model_densenet.predict(test_dataset, steps=test_steps)

y_true = y_test.values.astype(int)

# Varsayılan threshold (0.5) ile değerlendirme
y_pred_densenet = (predictions_densenet > 0.5).astype(int).flatten()

print("DenseNet201 - Classification Report (Default Threshold 0.5):")
print(classification_report(y_true, y_pred_densenet))
print("DenseNet201 - Confusion Matrix (Default Threshold 0.5):")
print(confusion_matrix(y_true, y_pred_densenet))

# Custom threshold ile değerlendirme
y_pred_custom = (predictions_densenet > custom_threshold).astype(int).flatten()

print(f"DenseNet201 - Classification Report (Custom Threshold {custom_threshold}):")
print(classification_report(y_true, y_pred_custom))
print(f"DenseNet201 - Confusion Matrix (Custom Threshold {custom_threshold}):")
print(confusion_matrix(y_true, y_pred_custom))

# Post-processing ile FN azaltma
post_processed_pred = np.where(predictions_densenet > post_process_threshold, 1, 0).flatten()

print(f"DenseNet201 Post-Processed - Classification Report (Threshold {post_process_threshold}):")
print(classification_report(y_true, post_processed_pred))
print(f"DenseNet201 Post-Processed - Confusion Matrix (Threshold {post_process_threshold}):")
print(confusion_matrix(y_true, post_processed_pred))

# Hatalı tahmin analizi
print("\nKanserli sınıfta (target=1) hatalı tahmin edilen resimler (Custom Threshold):")
misclassified_malignant = []
original_malignant_count = 0
synthetic_malignant_count = 0

X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

for i in range(len(y_true)):
    if y_true[i] == 1 and y_pred_custom[i] != y_true[i]:
        image_name = X_test_reset.iloc[i]['image_path'].split('/')[-1]
        misclassified_malignant.append(image_name)
        print(f"Hatalı tahmin edilen resim: {image_name}")
        if 'gan_malignant_' in image_name or 'augmented_malignant_' in image_name:
            synthetic_malignant_count += 1
        else:
            original_malignant_count += 1

print(f"\nToplam hatalı tahmin edilen kanserli resim sayısı: {len(misclassified_malignant)}")
print(f"Orijinal malignantlardan hatalı tahmin edilen: {original_malignant_count}")
print(f"GAN veya augmentasyon ile üretilen sentetik malignantlardan hatalı tahmin edilen: {synthetic_malignant_count}")

# Submission dosyası
print("\nAsıl test verisi ile submission.csv oluşturuluyor...")
test_df = pd.read_csv(test_csv_path)
test_df['image_path'] = test_df['image_name'].apply(lambda x: os.path.join(test_images_path, x + '.jpg'))

# Test verisi için metadata’yı işleme
test_df['sex'] = test_df['sex'].fillna('unknown')
test_df['age_approx'] = test_df['age_approx'].fillna(test_df['age_approx'].mean())
test_df['anatom_site_general_challenge'] = test_df['anatom_site_general_challenge'].fillna('unknown')

test_df['sex_encoded'] = label_encoder_sex.transform(test_df['sex'])
test_df['anatom_site_encoded'] = label_encoder_site.transform(test_df['anatom_site_general_challenge'])
test_df['age_approx'] = (test_df['age_approx'] - test_df['age_approx'].mean()) / test_df['age_approx'].std()

def load_and_preprocess_test_image(image_path, sex, age, site):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.keras.applications.densenet.preprocess_input(img)  # DenseNet için preprocess_input
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return img, metadata  # İki ayrı tensor döndürüyoruz

def create_submission_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site: load_and_preprocess_test_image(img_path, sex, age, site),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Veri setini oluştur
test_dataset_submission = create_submission_dataset(test_df, batch_size)
submission_steps = (len(test_df) + batch_size - 1) // batch_size

# DenseNet201 ile tahmin (batch’ler halinde)
print("DenseNet201 ile tahmin yapılıyor...")
predictions_list = []
iterator = iter(test_dataset_submission)
for _ in range(submission_steps):
    img, metadata = next(iterator)
    # Her batch için tahmin yap
    batch_predictions = model_densenet.predict([img, metadata], batch_size=batch_size, verbose=0)
    predictions_list.append(batch_predictions)

# Tüm tahminleri birleştir
predictions_submission_densenet = np.concatenate(predictions_list, axis=0)

# Custom threshold ile tahmin
test_df['target'] = (predictions_submission_densenet > custom_threshold).astype(int).flatten()

# Submission dosyasını kaydet
submission = test_df[['image_name', 'target']]
submission.to_csv('/kaggle/working/submission_densenet_epoch12.csv', index=False)
print("submission_densenet_epoch12.csv oluşturuldu: /kaggle/working/submission_densenet_epoch12.csv")


import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers
from tensorflow.keras.applications import DenseNet201, EfficientNetB0, ResNet50V2

# GPU bellek optimizasyonu
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU aktif!")
else:
    print("GPU bulunamadı, CPU kullanılacak.")

# Hiperparametreler
image_size = (224, 224)
batch_size = 32
custom_threshold = 0.45
post_process_threshold = 0.35

# Veri yolları
new_train_csv_path = '/kaggle/working/new_train.csv'
train_images_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
gan_images_path = '/kaggle/working/gan_generated_malignant/'
aug_images_path = '/kaggle/working/augmented_malignant/'
test_csv_path = '/kaggle/input/siim-isic-melanoma-classification/test.csv'
test_images_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

# Yeni train.csv’yi yükle (GAN ve augmentasyon sonrası veri)
print("Yeni train.csv yükleniyor...")
train_df_final = pd.read_csv(new_train_csv_path)
print(f"GAN ve augmentasyon sonrası veri seti boyutu: {train_df_final.shape}")

# Malignant sınıfının 3,500 olduğunu kontrol et
assert len(train_df_final[train_df_final['target'] == 1]) == 3500, "Malignant sınıfı 3,500’e ulaşmadı!"

# Görüntü yollarını güncelle
def get_image_path(image_name):
    if 'gan_malignant_' in image_name:
        return os.path.join(gan_images_path, image_name)
    elif 'augmented_malignant_' in image_name:
        return os.path.join(aug_images_path, image_name)
    else:
        return os.path.join(train_images_path, image_name + '.jpg')

# Hasta metadata’sını işleme
label_encoder_sex = LabelEncoder()
label_encoder_site = LabelEncoder()
train_df_final['sex'] = train_df_final['sex'].fillna('unknown')
train_df_final['age_approx'] = train_df_final['age_approx'].fillna(train_df_final['age_approx'].mean())
train_df_final['anatom_site_general_challenge'] = train_df_final['anatom_site_general_challenge'].fillna('unknown')
train_df_final['sex_encoded'] = label_encoder_sex.fit_transform(train_df_final['sex'])
train_df_final['anatom_site_encoded'] = label_encoder_site.fit_transform(train_df_final['anatom_site_general_challenge'])
train_df_final['age_approx'] = (train_df_final['age_approx'] - train_df_final['age_approx'].mean()) / train_df_final['age_approx'].std()

train_df_final['image_path'] = train_df_final['image_name'].apply(get_image_path)
train_df_final['target'] = train_df_final['target'].astype(float)

# Train, val ve test setlerini böl (%80 train, %10 val, %10 test)
X = train_df_final[['image_path', 'sex_encoded', 'age_approx', 'anatom_site_encoded']]
y = train_df_final['target']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Eğitim seti: {len(X_train)} örnek (%80)")
print(f"Doğrulama seti: {len(X_val)} örnek (%10)")
print(f"Test seti: {len(X_test)} örnek (%10)")

# Veri pipeline’ı (Görüntü ve metadata’yı birleştirme)
def load_and_preprocess_image(image_path, sex, age, site, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.keras.applications.densenet.preprocess_input(img)  # DenseNet için preprocess_input
    label = tf.cast(label, tf.float32)
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return (img, metadata), label

def create_test_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values,
        df['target'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site, label: load_and_preprocess_image(img_path, sex, age, site, label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()  # Veri setinin tükenmesini önlemek için
    return dataset

test_dataset = create_test_dataset(pd.concat([X_test, y_test], axis=1), batch_size)

# Test seti için steps
test_steps = (len(X_test) + batch_size - 1) // batch_size
print(f"Test steps: {test_steps}")

# Modeli oluştur (Görüntü ve metadata’yı birleştirme)
def build_model_with_metadata(base_model):
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Görüntü girişi
    image_input = Input(shape=(224, 224, 3), name='image_input')
    x = base_model(image_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Metadata girişi
    metadata_input = Input(shape=(3,), name='metadata_input')
    m = layers.Dense(16, activation='relu')(metadata_input)
    m = layers.Dense(8, activation='relu')(m)

    # Görüntü ve metadata’yı birleştirme
    combined = layers.concatenate([x, m])
    combined = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
    combined = layers.Dropout(0.5)(combined)
    output = layers.Dense(1, activation='sigmoid')(combined)

    model = models.Model(inputs=[image_input, metadata_input], outputs=output)
    return model

# Focal Loss
def focal_loss(gamma=2.0, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1) - (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0)
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# DenseNet201 Modeli
print("DenseNet201 modeli oluşturuluyor...")
base_model_densenet = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_densenet = build_model_with_metadata(base_model_densenet)
model_densenet.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])
print("DenseNet201 ağırlıkları yükleniyor...")
model_densenet.load_weights('/kaggle/working/densenet_best_model.keras')

# EfficientNetB0 Modeli
print("EfficientNetB0 modeli oluşturuluyor...")
base_model_efficientnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_efficientnet = build_model_with_metadata(base_model_efficientnet)
model_efficientnet.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])
print("EfficientNetB0 ağırlıkları yükleniyor...")
model_efficientnet.load_weights('/kaggle/working/efficientnet_best_model.keras')

# ResNet50V2 Modeli
print("ResNet50V2 modeli oluşturuluyor...")
base_model_resnet = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_resnet = build_model_with_metadata(base_model_resnet)
model_resnet.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])
print("ResNet50V2 ağırlıkları yükleniyor...")
model_resnet.load_weights('/kaggle/working/resnet_best_model.keras')

# Test seti üzerinde tahmin
print("Test seti üzerinde tahmin yapılıyor...")

# Test seti için doğrudan batch’ler halinde tahmin yapıyoruz
print("DenseNet201 ile test seti tahmini yapılıyor...")
predictions_densenet = model_densenet.predict(test_dataset, steps=test_steps)

print("EfficientNetB0 ile test seti tahmini yapılıyor...")
predictions_efficientnet_test = model_efficientnet.predict(test_dataset, steps=test_steps)

print("ResNet50V2 ile test seti tahmini yapılıyor...")
predictions_resnet_test = model_resnet.predict(test_dataset, steps=test_steps)

y_true = y_test.values.astype(int)

# DenseNet201 Metrikleri
print("\n=== DenseNet201 Test Seti Metrikleri ===")
y_pred_densenet = (predictions_densenet > custom_threshold).astype(int).flatten()
print(f"DenseNet201 - Classification Report (Custom Threshold {custom_threshold}):")
print(classification_report(y_true, y_pred_densenet))
print(f"DenseNet201 - Confusion Matrix (Custom Threshold {custom_threshold}):")
print(confusion_matrix(y_true, y_pred_densenet))

# EfficientNetB0 Metrikleri
print("\n=== EfficientNetB0 Test Seti Metrikleri ===")
y_pred_efficientnet = (predictions_efficientnet_test > custom_threshold).astype(int).flatten()
print(f"EfficientNetB0 - Classification Report (Custom Threshold {custom_threshold}):")
print(classification_report(y_true, y_pred_efficientnet))
print(f"EfficientNetB0 - Confusion Matrix (Custom Threshold {custom_threshold}):")
print(confusion_matrix(y_true, y_pred_efficientnet))

# ResNet50V2 Metrikleri
print("\n=== ResNet50V2 Test Seti Metrikleri ===")
y_pred_resnet = (predictions_resnet_test > custom_threshold).astype(int).flatten()
print(f"ResNet50V2 - Classification Report (Custom Threshold {custom_threshold}):")
print(classification_report(y_true, y_pred_resnet))
print(f"ResNet50V2 - Confusion Matrix (Custom Threshold {custom_threshold}):")
print(confusion_matrix(y_true, y_pred_resnet))

# Hatalı tahmin analizi (DenseNet201 için, diğer modeller için de eklenebilir)
print("\nKanserli sınıfta (target=1) hatalı tahmin edilen resimler (DenseNet201, Custom Threshold):")
misclassified_malignant = []
original_malignant_count = 0
synthetic_malignant_count = 0

X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

for i in range(len(y_true)):
    if y_true[i] == 1 and y_pred_densenet[i] != y_true[i]:
        image_name = X_test_reset.iloc[i]['image_path'].split('/')[-1]
        misclassified_malignant.append(image_name)
        print(f"Hatalı tahmin edilen resim: {image_name}")
        if 'gan_malignant_' in image_name or 'augmented_malignant_' in image_name:
            synthetic_malignant_count += 1
        else:
            original_malignant_count += 1

print(f"\nToplam hatalı tahmin edilen kanserli resim sayısı: {len(misclassified_malignant)}")
print(f"Orijinal malignantlardan hatalı tahmin edilen: {original_malignant_count}")
print(f"GAN veya augmentasyon ile üretilen sentetik malignantlardan hatalı tahmin edilen: {synthetic_malignant_count}")

# Submission dosyası için Kaggle test verisi
print("\nAsıl test verisi ile submission.csv oluşturuluyor...")
test_df = pd.read_csv(test_csv_path)
test_df['image_path'] = test_df['image_name'].apply(lambda x: os.path.join(test_images_path, x + '.jpg'))

# Test verisi için metadata’yı işleme
test_df['sex'] = test_df['sex'].fillna('unknown')
test_df['age_approx'] = test_df['age_approx'].fillna(test_df['age_approx'].mean())
test_df['anatom_site_general_challenge'] = test_df['anatom_site_general_challenge'].fillna('unknown')

test_df['sex_encoded'] = label_encoder_sex.transform(test_df['sex'])
test_df['anatom_site_encoded'] = label_encoder_site.transform(test_df['anatom_site_general_challenge'])
test_df['age_approx'] = (test_df['age_approx'] - test_df['age_approx'].mean()) / test_df['age_approx'].std()

def load_and_preprocess_test_image(image_path, sex, age, site):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.keras.applications.densenet.preprocess_input(img)  # DenseNet için preprocess_input
    metadata = tf.stack([tf.cast(sex, tf.float32), tf.cast(age, tf.float32), tf.cast(site, tf.float32)])
    return img, metadata

def create_submission_dataset(df, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['sex_encoded'].values,
        df['age_approx'].values,
        df['anatom_site_encoded'].values
    ))
    dataset = dataset.map(lambda img_path, sex, age, site: load_and_preprocess_test_image(img_path, sex, age, site),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()  # Veri setinin tükenmesini önlemek için
    return dataset

test_dataset_submission = create_submission_dataset(test_df, batch_size)
submission_steps = (len(test_df) + batch_size - 1) // batch_size

# Kaggle test verisi için doğrudan batch’ler halinde tahmin yapıyoruz
print("DenseNet201 ile submission tahmini yapılıyor...")
predictions_submission_densenet = model_densenet.predict(test_dataset_submission, steps=submission_steps)

print("EfficientNetB0 ile submission tahmini yapılıyor...")
predictions_submission_efficientnet = model_efficientnet.predict(test_dataset_submission, steps=submission_steps)

print("ResNet50V2 ile submission tahmini yapılıyor...")
predictions_submission_resnet = model_resnet.predict(test_dataset_submission, steps=submission_steps)

# Ensemble ağırlıkları
ensemble_weight_efficientnet = 0.30
ensemble_weight_resnet = 0.20
ensemble_weight_densenet = 0.50

# Ensemble tahmin
print("Ensemble tahmin yapılıyor...")
predictions_ensemble = (ensemble_weight_efficientnet * predictions_submission_efficientnet +
                       ensemble_weight_resnet * predictions_submission_resnet +
                       ensemble_weight_densenet * predictions_submission_densenet)

# Custom threshold ile ensemble tahmin
test_df['target'] = (predictions_ensemble > custom_threshold).astype(int).flatten()

# Ensemble submission dosyası oluşturma
submission = test_df[['image_name', 'target']]
submission.to_csv('/kaggle/working/submission_ensemble.csv', index=False)
print("submission_ensemble.csv oluşturuldu: /kaggle/working/submission_ensemble.csv")

# Karşılaştırma için her modelin submission dosyasını da oluştur
# DenseNet201
test_df['target'] = (predictions_submission_densenet > custom_threshold).astype(int).flatten()
submission_densenet = test_df[['image_name', 'target']]
submission_densenet.to_csv('/kaggle/working/submission_densenet.csv', index=False)
print("submission_densenet.csv oluşturuldu: /kaggle/working/submission_densenet.csv")

# EfficientNetB0
test_df['target'] = (predictions_submission_efficientnet > custom_threshold).astype(int).flatten()
submission_efficientnet = test_df[['image_name', 'target']]
submission_efficientnet.to_csv('/kaggle/working/submission_efficientnet.csv', index=False)
print("submission_efficientnet.csv oluşturuldu: /kaggle/working/submission_efficientnet.csv")

# ResNet50V2
test_df['target'] = (predictions_submission_resnet > custom_threshold).astype(int).flatten()
submission_resnet = test_df[['image_name', 'target']]
submission_resnet.to_csv('/kaggle/working/submission_resnet.csv', index=False)
print("submission_resnet.csv oluşturuldu: /kaggle/working/submission_resnet.csv")

