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
import zipfile
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import log_loss, confusion_matrix, classification_report


base_dir = '/kaggle/working'
train_zip = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
test_zip = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'


# 再実行時の対策：workingディレクトリをクリーンアップ
!rm -rf /kaggle/working/*
os.makedirs(base_dir, exist_ok=True)


with zipfile.ZipFile(train_zip, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(base_dir, 'train_raw'))

with zipfile.ZipFile(test_zip, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(base_dir, 'test'))


# データ格納用フォルダ作成
folders = ['images/train/dog', 'images/train/cat', 'images/val/dog', 'images/val/cat']
for folder in folders:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)


# 3. train_rawからtrain/valにデータを分割
train_path = os.path.join(base_dir, 'train_raw/train')
all_images = os.listdir(train_path)
random.shuffle(all_images)

split_ratio = 0.8  # 80%:20%
split_idx = int(len(all_images) * split_ratio)

train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# 4. 移動関数
def move_images(image_list, subset):
    for img in image_list:
        label = 'dog' if 'dog' in img else 'cat'
        dest_dir = os.path.join(base_dir, f'images/{subset}/{label}')
        src = os.path.join(train_path, img)
        dst = os.path.join(dest_dir, img)
        os.replace(src, dst)

# 5. 実際に移動
move_images(train_images, 'train')
move_images(val_images, 'val')

# 6. 分割結果を確認
print(f"Train total: {len(train_images)}")
print(f"Val total: {len(val_images)}")


train_path = os.path.join(base_dir, 'train_raw/train')
all_images = os.listdir(train_path)
random.shuffle(all_images)

split_ratio = 0.8
split_idx = int(len(all_images) * split_ratio)

train_images = all_images[:split_idx]
val_images = all_images[split_idx:]


def move_images(image_list, subset):
    for img in image_list:
        label = 'dog' if 'dog' in img else 'cat'
        dest_dir = os.path.join(base_dir, f'images/{subset}/{label}')
        src = os.path.join(train_path, img)
        dst = os.path.join(dest_dir, img)
        os.replace(src, dst)

move_images(train_images, 'train')
move_images(val_images, 'val')


img_size = (224, 224)
batch_size = 32
classes = ['cat', 'dog']

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'images/train'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    classes=classes
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(base_dir, 'images/val'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    classes=classes,
    shuffle=False
)


dog_count = len(os.listdir(os.path.join(base_dir, 'images/train/dog')))
cat_count = len(os.listdir(os.path.join(base_dir, 'images/train/cat')))
plt.bar(['Dogs', 'Cats'], [dog_count, cat_count], color=['blue', 'orange'])
plt.title('Training Data Distribution')
plt.show()


# ランダムサンプル表示
dog_samples = random.sample(os.listdir(os.path.join(base_dir, 'images/train/dog')), 3)
cat_samples = random.sample(os.listdir(os.path.join(base_dir, 'images/train/cat')), 3)

fig, axes = plt.subplots(2, 3, figsize=(8, 6))
for i, img_name in enumerate(dog_samples):
    img = Image.open(os.path.join(base_dir, 'images/train/dog', img_name))
    axes[0, i].imshow(img)
    axes[0, i].axis('off')
    axes[0, i].set_title('Dog')
for i, img_name in enumerate(cat_samples):
    img = Image.open(os.path.join(base_dir, 'images/train/cat', img_name))
    axes[1, i].imshow(img)
    axes[1, i].axis('off')
    axes[1, i].set_title('Cat')
plt.show()


print("Class indices:", train_generator.class_indices)
print("Val Generator classes shape:", val_generator.classes.shape)
print("First 10 true labels:", val_generator.classes[:10])


input_tensor = Input(shape=(224,224,3))
base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)


# Fine-tuning：最後の10層のみ学習
for layer in base_model.layers[:-10]:
    layer.trainable = False

# ===============================
# コンパイル設定（ラベルスムージング追加）
# ===============================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy'])


# 分割後のディレクトリ構造を確認
import os

print("Train folders:", os.listdir(os.path.join(base_dir, 'images/train')))
print("Validation folders:", os.listdir(os.path.join(base_dir, 'images/val')))

# 各クラスの枚数を確認
for subset in ['train', 'val']:
    for label in ['dog', 'cat']:
        path = os.path.join(base_dir, f'images/{subset}/{label}')
        print(f"{subset}/{label}: {len(os.listdir(path))}")



# ランダムに画像を表示して分割を確認
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os

# 新しいディレクトリから画像を取得
train_dir = os.path.join(base_dir, 'images/train')
classes = ['dog', 'cat']

# クラスをランダムに選んで、画像を1枚選択
chosen_class = random.choice(classes)
class_dir = os.path.join(train_dir, chosen_class)
sample_img = random.choice(os.listdir(class_dir))

# 画像を表示
img_path = os.path.join(class_dir, sample_img)
img = mpimg.imread(img_path)
plt.imshow(img)
plt.title(f"Sample from train set: {chosen_class}/{sample_img}")
plt.axis('off')
plt.show()


epochs = 5
steps_per_epoch = train_generator.samples // batch_size
validation_steps = val_generator.samples // batch_size

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=[early_stop, reduce_lr]
)


# 現状のモデルでValデータを評価
val_loss, val_acc = model.evaluate(val_generator, verbose=1)
print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")

# クラスインデックス確認
print("Class indices:", train_generator.class_indices)
print("Val Generator classes shape:", val_generator.classes.shape)
print("First 10 true labels:", val_generator.classes[:10])


from sklearn.metrics import classification_report, log_loss

# 予測値を取得
val_generator.reset()
pred_probs = model.predict(val_generator, verbose=1)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = val_generator.classes

# Log Loss
val_log_loss = log_loss(true_classes, pred_probs)
print(f"Validation Log Loss: {val_log_loss:.6f}")

# Classification Report
target_names = list(train_generator.class_indices.keys())
print("\nClassification Report:")
print(classification_report(true_classes, pred_classes, target_names=target_names))



plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.show()


val_generator.reset()
val_preds = model.predict(val_generator)
val_labels = val_generator.classes
print("Validation Log Loss:", log_loss(val_labels, val_preds))


#Test Generator
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    directory='/kaggle/working/test',
    target_size=img_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)

#予測
predictions = model.predict(test_generator)

#提出用DataFrame作成
submission = pd.DataFrame({
    'id': [int(os.path.basename(path).split('.')[0]) for path in test_generator.filenames],
    'label': predictions[:, 1]  # 犬クラスの確率
})

#ID順に並び替え
submission = submission.sort_values('id').reset_index(drop=True)

#CSV出力
submission.to_csv('submission.csv', index=False)
print("Kaggle提出用 submission.csv 完了")

#確認
print(submission.head())
print(submission.tail())



unique, counts = np.unique(val_generator.classes, return_counts=True)
print(dict(zip(unique, counts)))


print("Train total:", len(train_images))
print("Val total:", len(val_images))
print("重複画像数:", len(set(train_images) & set(val_images)))


