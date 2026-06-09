import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


labels = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
print(labels.head())


plt.figure(figsize=(15,5))
labels['breed'].value_counts().plot(kind='bar')
plt.title('Number of Images by Breed')
plt.show()


sample = labels.sample(1).iloc[0]
img_path = f"/kaggle/input/dog-breed-identification/train/{sample['id']}.jpg"
img = Image.open(img_path)
plt.imshow(img)
plt.title(sample['breed'])
plt.axis('off')
plt.show()


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
labels['breed_idx'] = le.fit_transform(labels['breed'])

# 학습/검증 셋 분리 (10%는 검증용)
train_df, valid_df = train_test_split(labels, test_size=0.1, stratify=labels['breed_idx'], random_state=42)
print(f"Train: {len(train_df)}, Valid: {len(valid_df)}")

#  'filename' 컬럼 각각 추가
train_df = train_df.copy()    # 혹시 warning 방지
valid_df = valid_df.copy()
train_df['filename'] = train_df['id'] + '.jpg'
valid_df['filename'] = valid_df['id'] + '.jpg'


import tensorflow as tf

img_size = 128
batch_size = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.1
)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    directory='/kaggle/input/dog-breed-identification/train/',
    x_col='filename',
    y_col='breed',
    target_size=(img_size, img_size),
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True
)

valid_gen = valid_datagen.flow_from_dataframe(
    valid_df,
    directory='/kaggle/input/dog-breed-identification/train/',
    x_col='filename',
    y_col='breed',
    target_size=(img_size, img_size),
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)



import tensorflow as tf
from tensorflow.keras import layers, models

# MobileNetV2: 빠르고 성능 좋은 사전학습 이미지 분류 모델
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_size, img_size, 3),
    include_top=False,      # 마지막 분류 레이어 빼고 feature만 사용
    weights='imagenet'      # ImageNet 데이터로 사전학습된 가중치 사용
)
base_model.trainable = False   # 처음에는 base_model을 고정(학습하지 않음)

# 우리 데이터에 맞는 분류 레이어 쌓기
model = models.Sequential([
    base_model,                             # feature 추출
    layers.GlobalAveragePooling2D(),        # feature map을 1차원으로 만듦
    layers.Dense(256, activation='relu'),   # 중간 레이어(자유롭게 수정 가능)
    layers.Dropout(0.3),                    # 과적합 방지
    layers.Dense(120, activation='softmax') # 120개 품종 분류 (클래스 수 꼭 맞추기!)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()  # 모델 구조 한눈에 보기



# 모델 학습 (fit)
history = model.fit(
    train_gen,
    epochs=5,  # 빠른 실습은 5, 더 길게 하고 싶으면 10~20도 OK
    validation_data=valid_gen
)


import matplotlib.pyplot as plt

# 학습 기록(history)에 저장된 정확도/손실 정보 시각화
plt.figure(figsize=(12, 5))

# 정확도
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 손실(Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()



import numpy as np
from PIL import Image

# 1. 샘플 이미지 선택 (검증셋에서 한 장)
sample = valid_df.sample(1).iloc[0]
sample_img_path = f"/kaggle/input/dog-breed-identification/train/{sample['filename']}"

# 2. 이미지 불러와서 전처리
img = Image.open(sample_img_path).convert('RGB').resize((img_size, img_size))
img_array = np.array(img) / 255.0  # 정규화
img_tensor = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

# 3. 예측
preds = model.predict(img_tensor)
pred_idx = np.argmax(preds, axis=1)[0]
pred_breed = le.inverse_transform([pred_idx])[0]

# 4. 실제 품종과 예측 품종 비교
plt.imshow(img)
plt.axis('off')
plt.title(f"Actual: {sample['breed']}\nPredicted: {pred_breed}")
plt.show()

print(f"실제 품종: {sample['breed']}")
print(f"예측 품종: {pred_breed}")


