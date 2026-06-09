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





import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(os.path.join(dirname))
    for filename in filenames:
        print("   →", filename)


import pandas as pd

train_df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
train_df.head()


import cv2
import matplotlib.pyplot as plt

img_name = train_df['id_code'][0] + ".png"
img_path = "/kaggle/input/aptos2019-blindness-detection/train_images/" + img_name

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis('off')
plt.title("Example Image")


import cv2
import numpy as np
from tqdm import tqdm # لإظهار شريط التحميل

# 1. تعريف دالة المعالجة الاحترافية
def preprocess_retina_image(image, img_size=380):
    # تحويل لرمادي لإيجاد القناع
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # عمل قناع لعزل السواد (أي بكسل قيمته أكبر من 10 يعتبر جزء من العين)
    mask = gray > 10
    # قص الأجزاء السوداء الزائدة
    if np.any(mask):
        image = image[np.ix_(mask.any(1), mask.any(0))]
    # تغيير الحجم لـ 380 (المثالي لـ EfficientNetB4)
    image = cv2.resize(image, (img_size, img_size))
    # تقنية تحسين التباين (Ben Graham's Processing) لإبراز العروق
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), img_size/10), -4, 128)
    return image

# 2. تطبيق المعالجة على مصفوفة الصور X كاملة
print("جاري معالجة وتحسين الصور... قد يستغرق ذلك دقيقة")
X_processed = []
for img in tqdm(X):
    # نستخدم الدالة التي عرفناها فوق
    processed_img = preprocess_retina_image(img)
    X_processed.append(processed_img)

# تحويل القائمة إلى مصفوفة numpy وتحديث X الأصلية
X = np.array(X_processed)

print("تم تحسين جميع الصور بنجاح. الأبعاد الجديدة:", X.shape)


print("X shape:", X.shape)
print("y shape:", y.shape)


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 1. تعريف طبقة توليد البيانات (Data Augmentation)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2), 
    layers.RandomZoom(0.2),      
    layers.RandomContrast(0.2),  
])

# 2. تعريف المدخلات
inputs = layers.Input(shape=(224, 224, 3))

# 3. استدعاء طبقة الـ Augmentation هنا (أول خطوة بعد المدخلات)
x = data_augmentation(inputs)

# 4. إضافة طبقة المعالجة المدمجة
x = Lambda(preprocess_input)(x)

# 5. تحميل الموديل الأساسي (Fine-tuning)
base_model = EfficientNetB4(weights='imagenet', include_top=False)
base_model.trainable = True
for layer in base_model.layers[:-60]: # تجميد أغلب الطبقات وفتح آخر 60
    layer.trainable = False

# 6. ربط الطبقات ببعضها
x = base_model(x)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x) # رفعنا الـ Dropout قليلاً لزيادة الدقة
output = Dense(5, activation='softmax')(x)

# 7. إنشاء الموديل النهائي
model = Model(inputs, output)

# 8. الـ Compile (تأكد من استخدام Learning Rate صغير)
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("تم دمج طبقة الـ Data Augmentation بنجاح!")


from sklearn.utils import class_weight
import numpy as np

# حساب الأوزان بناءً على توزيع الصور في بياناتك
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(weights))

print("تم حساب أوزان الأصناف لموازنة البيانات:")
print(class_weights_dict)


# 1. فتح الموديل بالكامل للتدريب
base_model.trainable = True 

# 2. استخدام Learning Rate صغير جداً (للحفاظ على استقرار الموديل)
# لاحظ غيرناه إلى 1e-5 ليكون أقوى قليلاً من الـ 1e-7 الذي توقف عنده الموديل
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("تم فتح جميع الطبقات للتدريب العميق.")


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 1. مراقب لإيقاف التدريب إذا توقفت الدقة عن التحسن (صبر لمدة 15 دورة)
early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)

# 2. مراقب لتقليل سرعة التعلم إذا ثبتت الدقة (للدخول في التفاصيل الدقيقة جداً)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-8, verbose=1)

# 3. حفظ أفضل نسخة من الموديل تلقائياً
checkpoint = ModelCheckpoint('best_model_retinopathy.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

print("تم تجهيز المراقبين للوصول للدقة القصوى.")


# ملاحظة: تأكد أن X_train و y_train تم تقسيمهم بعد عملية الـ Crop والتحسين التي قمت بها
print("بدء مرحلة الـ Fine-Tuning العميق...")

history_final = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,             # عدد دورات كافٍ للوصول للـ 98%
    batch_size=8,          # حجم صغير للدقة العالية وتجنب امتلاء الذاكرة
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights_dict  # الأوزان التي حسبتها في صورتك الأخيرة
)


# 1. تعريف طبقة توليد البيانات (Data Augmentation)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2), 
    layers.RandomZoom(0.2),      
    layers.RandomContrast(0.2),  
])

# 2. تعريف المدخلات
inputs = layers.Input(shape=(224, 224, 3))

# 3. استدعاء طبقة الـ Augmentation هنا (أول خطوة بعد المدخلات)
x = data_augmentation(inputs)

# 4. إضافة طبقة المعالجة المدمجة
x = Lambda(preprocess_input)(x)

# 5. تحميل الموديل الأساسي (Fine-tuning)
base_model = EfficientNetB4(weights='imagenet', include_top=False)
base_model.trainable = True
for layer in base_model.layers[:-60]: # تجميد أغلب الطبقات وفتح آخر 60
    layer.trainable = False

# 6. ربط الطبقات ببعضها
x = base_model(x)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x) # رفعنا الـ Dropout قليلاً لزيادة الدقة
output = Dense(5, activation='softmax')(x)

# 7. إنشاء الموديل النهائي
model = Model(inputs, output)

# 8. الـ Compile (تأكد من استخدام Learning Rate صغير)
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("تم دمج طبقة الـ Data Augmentation بنجاح!")


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights_dict # ضروري جداً لموازنة الـ 5 أصناف
)

