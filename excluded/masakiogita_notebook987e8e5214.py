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


import zipfile

# zipファイルのパス
zip_train_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
zip_test_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'
# 展開先のパス
extract_train_path = '/kaggle/working'
extract_test_path = '/kaggle/working'
# 展開処理
with zipfile.ZipFile(zip_train_path, 'r') as zip_ref:
    zip_ref.extractall(extract_train_path)
with zipfile.ZipFile(zip_test_path, 'r') as zip_ref:
    zip_ref.extractall(extract_test_path)
print("ファイル展開完了")


train_dir = '/kaggle/working/train'  # 展開された画像の場所

filenames = os.listdir(train_dir)
labels = ['dog' if 'dog' in fname else 'cat' for fname in filenames]
print("ラベル付け完了")


# trainファイルのフルパス
train_filepaths = [os.path.join(train_dir, fname) for fname in filenames]

# train DataFrame作成
train_df = pd.DataFrame({
    'filename': filenames,
    'filepath': train_filepaths,
    'label': labels
})
print("trainDF作成完了")

# testファイルのパス
test_dir = '/kaggle/working/test'
test_filenames = os.listdir(test_dir)
test_filepaths = [os.path.join(test_dir, fname) for fname in test_filenames]

# test DataFrame作成
test_df = pd.DataFrame({
    'filename': test_filenames,
    'filepath': test_filepaths
})
print("testDF作成完了")

# 確認（ラベルはtrainのみ）
print(train_df.head())
print(test_df.head())


from PIL import Image

# 保存先ディレクトリ作成
train_resized_dir = '/kaggle/working/train_128'
test_resized_dir = '/kaggle/working/test_128'
os.makedirs(train_resized_dir, exist_ok=True)
os.makedirs(test_resized_dir, exist_ok=True)

# ----------- train画像のリサイズ -----------
for fname in train_df['filename']:
    src_path = os.path.join(train_dir, fname)
    dst_path = os.path.join(train_resized_dir, fname)
    with Image.open(src_path) as img:
        img_resized = img.resize((128, 128))
        img_resized.save(dst_path)

# filepathを更新
train_df['filepath'] = train_df['filename'].apply(lambda x: os.path.join(train_resized_dir, x))
print("trainのリサイズ完了")

# ----------- test画像のリサイズ -----------
for fname in test_df['filename']:
    src_path = os.path.join(test_dir, fname)
    dst_path = os.path.join(test_resized_dir, fname)
    with Image.open(src_path) as img:
        img_resized = img.resize((128, 128))
        img_resized.save(dst_path)

# filepathを更新
test_df['filepath'] = test_df['filename'].apply(lambda x: os.path.join(test_resized_dir, x))
print("testのリサイズ完了")


# ----------- train_128のサイズ確認 -----------
correct_train_size = 0
for path in train_df['filepath']:
    with Image.open(path) as img:
        if img.size == (128, 128):
            correct_train_size += 1

print(f"128x128に正しくリサイズされたtrain画像：{correct_train_size} / {len(train_df)}")

# ----------- test_128のサイズ確認 -----------
correct_test_size = 0
for path in test_df['filepath']:
    with Image.open(path) as img:
        if img.size == (128, 128):
            correct_test_size += 1

print(f"128x128に正しくリサイズされたtest画像：{correct_test_size} / {len(test_df)}")


from PIL import Image
from sklearn.model_selection import train_test_split

# データとラベルの準備
X = []
y = []

for _, row in train_df.iterrows():
    img = Image.open(row['filepath']).convert('L')  # グレースケール
    img_array = np.array(img) / 255.0               # 正規化（0〜1）
    X.append(img_array)
    y.append(0 if row['label'] == 'cat' else 1)

X = np.array(X).reshape(-1, 128, 128, 1).astype(np.float32)  # チャンネル追加＆型変換
y = np.array(y).astype(np.int32)
print("X shape:", X.shape, "| y shape:", y.shape)


train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.1, random_state=42)


import tensorflow as tf

train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
valid_ds = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))

BATCH_SIZE = 32

train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # 2値分類
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


#history = model.fit(train_ds, validation_data=valid_ds, epochs=10)
#model.save('/kaggle/working/my_model.h5')  # 保存パスとファイル名は自由
#print("✅ モデルを保存しました")


test_images = []
image_ids = []

for _, row in test_df.iterrows():
    img = Image.open(row['filepath']).convert('L')  # グレースケール
    img_array = np.array(img) / 255.0               # 正規化
    img_array = img_array.reshape(128, 128, 1)
    test_images.append(img_array)

    # id（ファイル名の「.jpg」を取り除いた整数）を保存
    image_ids.append(int(row['filename'].split('.')[0]))

# numpy配列に変換
test_images = np.array(test_images).astype(np.float32)


# 予測（出力は0.0〜1.0）
preds = model.predict(test_images)
preds = preds.flatten()  # 1次元に


import pandas as pd

submission = pd.DataFrame({
    'id': image_ids,
    'label': preds  # 確率のままでもOK（Kaggleで受け入れられる）
})

submission = submission.sort_values('id')  # id順にソート（Kaggle提出形式）
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv を保存しました")


import os
print(os.listdir('/kaggle/working'))

