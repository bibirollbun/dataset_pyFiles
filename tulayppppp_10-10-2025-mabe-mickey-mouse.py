# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import PIL
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input, decode_predictions, ResNet50
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
import seaborn as sns
import json
from pathlib import Path
from collections import defaultdict, Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

from scipy import stats
from scipy.spatial.distance import cdist, euclidean
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


train_df = pd.read_csv('/kaggle/input/mickey2025/train.csv')
test_df = pd.read_csv('/kaggle/input/mickey2025/test.csv')


train_df


train_df.head(5)


train_df.describe().T


# Eğer test verisini yüklediğiniz DataFrame varsa:
 test_df.isnull().sum()


train_df.info()


train_df.isnull().sum()


train_df.duplicated().sum()


print(train_df.columns)



y_train = train_df["lab_id"]
X_train = train_df.drop(labels = ["lab_id"], axis = 1)



print("x_train shape: ",X_train.shape)
print("y_train shape: ",y_train.shape)


from sklearn.preprocessing import LabelEncoder
import numpy as np

# Örnek Kategorik Veri
categories = ['video_id', 'mouse1_strain', 'mouse1_color', 'mouse1_sex',
       'mouse1_id', 'mouse1_age', 'mouse1_condition', 'mouse2_strain',
       'mouse2_color', 'mouse2_sex', 'mouse2_id', 'mouse2_age',
       'mouse2_condition', 'mouse3_strain', 'mouse3_color', 'mouse3_sex',
       'mouse3_id', 'mouse3_age', 'mouse3_condition', 'mouse4_strain',
       'mouse4_color', 'mouse4_sex', 'mouse4_id', 'mouse4_age',
       'mouse4_condition', 'frames_per_second', 'video_duration_sec',
       'pix_per_cm_approx', 'video_width_pix', 'video_height_pix',
       'arena_width_cm', 'arena_height_cm', 'arena_shape', 'arena_type',
       'body_parts_tracked', 'behaviors_labeled', 'tracking_method']

# 1. LabelEncoder'ı Başlatma
label_encoder = LabelEncoder()

# 2. Veriye uydurma (Fit) ve dönüştürme (Transform)
# fit_transform() metodu hem encoder'ı veriye göre eğitir hem de dönüşümü yapar.
encoded_labels = label_encoder.fit_transform(categories)

# Sonuçları yazdırma
print("Orijinal Kategoriler:", categories)
print("Kodlanmış Etiketler:", encoded_labels)

# Kategorilerin hangi sayısal değere karşılık geldiğini görme
# classes_ niteliği, kodlayıcının öğrendiği etiketleri içerir.
print("Kategori-Sayısal Eşleşmesi (Alfabetik Sıraya Göre):", label_encoder.classes_)


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=2)
print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",y_train.shape)
print("y_test shape",y_val.shape)



# ... önceki import'lar
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam

# Hata veren satırı aşağıdaki ile değiştirin:
from tensorflow.keras.preprocessing.image import ImageDataGenerator # BU SATIRI KULLANIN

from keras.callbacks import ReduceLROnPlateau
# ...
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
import numpy as np
import itertools
# Hata veren satırı aşağıdaki ile değiştirin
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # ✅ Doğru
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


import tensorflow as tf
# GPU'yu devre dışı bırakır, kodun CPU'da çalışmasını sağlar
tf.config.set_visible_devices([], 'GPU')


model = Sequential()
# Input katmanını modelin ilk katmanı olarak ekleyin
model.add(Input(shape=(28, 28, 1))) 
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Daha önceki hataları önlemek için GPU'yu devre dışı bırakabilirsiniz (isteğe bağlı)
# tf.config.set_visible_devices([], 'GPU')
# print("GPU devre dışı bırakıldı, CPU kullanılıyor.")

# --- 1. Veri Hazırlığı (Basitleştirilmiş Görüntü Boyutları) ---
# MABe yarışması verisi karmaşık olduğu için, burada varsayımsal bir görüntü boyutu kullanıyoruz.
# Gerçek uygulamada, HDF5 dosyalarından kareleri çıkarıp uygun şekilde etiketlemeniz gerekir.
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 3  # RGB görüntüler için 3, gri tonlamalı için 1
NUM_CLASSES = 15 # Yarışmadaki fare davranışlarının sayısı

# Yüklenen Görüntülerin (X) ve Etiketlerin (Y) olduğu varsayılır.
# Örnek: X_train.shape = (n_samples, 64, 64, 3), Y_train.shape = (n_samples, 15)

# --- 2. CNN Modelini Tanımlama ---
# LeNet/VGG benzeri basit bir Evrişimli Sinir Ağı mimarisi

def build_cnn_model():
    model = Sequential([
        # GİRİŞ Katmanı: Input(shape) ile başlangıç şeklini belirtme (önerilen yöntem)
        tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        
        # Evrişim Bloğu 1
        Conv2D(32, kernel_size=(5, 5), padding='Same', activation='relu'),
        Conv2D(32, kernel_size=(5, 5), padding='Same', activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.25),

        # Evrişim Bloğu 2
        Conv2D(64, kernel_size=(3, 3), padding='Same', activation='relu'),
        Conv2D(64, kernel_size=(3, 3), padding='Same', activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.25),

        # Tam Bağlantılı (Dense) Katmanlar
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        
        # ÇIKIŞ Katmanı
        # Davranış tespiti bir sınıflandırma problemi olduğundan 'softmax' kullanılır.
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

model = build_cnn_model()

# --- 3. Modeli Derleme (Compile) ---
# Optimizasyon algoritması, kayıp fonksiyonu (loss) ve metrikler belirlenir.
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Modelin özetini göster
model.summary()

# --- 4. Veri Artırımı (Data Augmentation) ---
# Modelin aşırı öğrenmesini (overfitting) engellemek için ImageDataGenerator kullanılır.
datagen = ImageDataGenerator(
    rotation_range=10,        # Rastgele döndürme
    zoom_range=0.1,           # Rastgele yakınlaştırma
    width_shift_range=0.1,    # Yatay kaydırma
    height_shift_range=0.1,   # Dikey kaydırma
    horizontal_flip=False,    # Yatay çevirme (fare davranışlarında ters çevirme anlamı bozabilir)
    vertical_flip=False       # Dikey çevirme (fare davranışlarında ters çevirme anlamı bozabilir)
)

# --- 5. Geri Çağrım Fonksiyonları (Callbacks) ---
# Öğrenme oranını (learning rate) otomatik olarak azaltmak için kullanılır.
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

# --- 6. Modeli Eğitme (Fit) ---
# Buradaki X_train, Y_train ve X_val, Y_val değişkenlerinin tanımlanmış olması gerekir.
# Eğer veriniz büyükse generator kullanmak zorunludur.

'''
# Eğitim Verisinin Tanımlı Olduğunu Varsayarsak:
# history = model.fit(
#     datagen.flow(X_train, Y_train, batch_size=64),
#     epochs=30,
#     validation_data=(X_val, Y_val),
#     verbose=2,
#     callbacks=[learning_rate_reduction]
# )
'''


# --- 7. Test Verisini Hazırlama (Sahte Veri) ---
# Gerçek Kaggle yarışmasında buraya test.csv'yi yükleyip ön işlemeler uygulanır.
N_TEST = 2000 # Varsayımsal Test Verisi Sayısı

# Test Görüntüleri (X_test): (N_TEST, 64, 64, 3) boyutunda rastgele pikseller
X_test = np.random.rand(N_TEST, IMG_HEIGHT, IMG_WIDTH, CHANNELS).astype('float32')

# Not: Test verisi ID'leri de gereklidir.
test_ids = pd.Series(range(N_TEST))
print(f"X_test boyutu: {X_test.shape}")


# 1. Tahminleri oluşturun
raw_predictions = model.predict(X_test)
predictions = np.argmax(raw_predictions, axis=1)

# 2. Zorunlu Veri Tipi Dönüşümü (Çok Önemli!)
# Kaggle genellikle sınıf etiketlerinin tamsayı (int) olmasını bekler.
predictions = predictions.astype(int) 

# 3. Submission DataFrame'i oluşturun
submission_df = pd.DataFrame({
    'ID': test_ids,
    'Class': predictions
})


# --- 8. Tahminleri Oluşturma ---
# model.predict, her sınıf için olasılıkları döndürür (softmax çıkışı).
# predictions değişkeni artık tanımlanmıştır.
raw_predictions = model.predict(X_test)

# Sınıflandırma problemi olduğu için, en yüksek olasılığa sahip sınıfı seçmeliyiz.
# Bu, "One-Hot Encoding"den sınıf etiketine geçiştir.
predictions = np.argmax(raw_predictions, axis=1)

print("Tahminler başarıyla oluşturuldu.")


# --- 9. Submission Dosyasını Oluşturma ve Kaydetme ---

# Pandas kütüphanesini kullanarak DataFrame'i oluşturun.
# Sütun adları, yarışmanın formatına TAM OLARAK uymalıdır.
# Genellikle 'ID' ve 'Target' (Hedef) veya 'Class' (Sınıf) şeklindedir.
submission_csv = pd.DataFrame({
    'ID': test_ids,
    'Class': predictions
})

# Dosyayı kaydedin. index=False, ekstra satır numaralarının eklenmesini engeller.
submission_csv.to_csv('submission.csv', index=False)

print("\n---------------------------------------------------------")
print("Submission dosyası oluşturuldu: submission.csv")
print("İlk 5 satır:")
print(submission_csv.head())
print("---------------------------------------------------------")




# Final step to create the submission file
submission_df = pd.DataFrame({
    'ID': test_ids,
    'Class': predictions
})

# IMPORTANT: Must be named 'submission.csv'
submission_df.to_csv('submission.csv', index=False)
submission_df


# KESİNLİKLE bu isimde ve index=False ile kaydedilmelidir.
submission_df.to_csv('submission.csv', index=False)




