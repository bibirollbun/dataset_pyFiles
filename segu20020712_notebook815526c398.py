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

# 解凍先のディレクトリ
extract_dir = '/kaggle/working'

# train.zip の解凍
with zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/train.zip', 'r') as zip_ref:
    zip_ref.extractall(os.path.join(extract_dir, 'train'))

# test.zip の解凍
with zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/test.zip', 'r') as zip_ref:
    zip_ref.extractall(os.path.join(extract_dir, 'test'))    


for dirname, _, _ in os.walk('/kaggle/working'):
    print(dirname)


!pip install -U efficientnet


# トレーニング画像が保存されているディレクトリのパスを変数 train_dir に代入
train_dir = '/kaggle/working/train/train'

# テスト画像が保存されているディレクトリのパスを変数 test_dir に代入
test_dir = '/kaggle/working/test'

# train.csvを読み込んで train_df に代入(ファイル名(id)とラベル(has_cactus)が記述されている)
train_df = pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv')

# train_df の先頭20行を表示
train_df.head(20)


# ディレクトリ内のファイル数を数える関数
def count_files(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

# テスト画像が保存されているディレクトリのパスを変数 test_dir2 に代入
test_dir2 = '/kaggle/working/test/test'

# トレーニングとテストディレクトリのファイル数を取得
train_count = count_files(train_dir)
test_count = count_files(test_dir2)

# 結果を表示
print(f'Train images: {train_count}')
print(f'Test images: {test_count}')


# クラスの割合（パーセンテージ）を表示
class_ratio = train_df['has_cactus'].value_counts(normalize=True) * 100
print(class_ratio)


import matplotlib.pyplot as plt

# 件数をカウント
counts = train_df['has_cactus'].value_counts()

# ラベル設定（1: サボテンあり, 0: サボテンなし）
labels = ['Has Cactus (1)', 'No Cactus (0)']
colors = ['lightgreen', 'lightcoral']

# 円グラフを作成
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Distribution of Cactus Presence (has_cactus)')
plt.axis('equal')  # 円を真円にする
plt.show()


from sklearn.utils.class_weight import compute_class_weight

# クラスの重みを自動計算（0と1の出現比に応じて）
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['has_cactus']),
    y=train_df['has_cactus']
)
class_weights_dict = dict(enumerate(class_weights))
print(class_weights_dict)


import cv2
cactus = []
# cactus images
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][0]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][1]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][2]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][8]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][9]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][12]))
# no cactus images
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][6]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][7]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][11]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][14]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][16]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][17]))


labels = ['cactus','cactus','cactus','cactus','cactus','cactus',
          'no cactus','no cactus',' no cactus','no cactus','no cactus',' no cactus']

import matplotlib.pyplot as plt

plt.figure(figsize=[10,10])
for x in range(0,12):
    plt.subplot(4, 3,x+1)
    plt.imshow(cactus[x])
    plt.title(labels[x])
    x += 1

plt.tight_layout()
plt.show()


import cv2
cactus = []
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][0]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][1]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][2]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][3]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][4]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][5]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][6]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][7]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][8]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][9]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][10]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][11]))

import matplotlib.pyplot as plt

# 画像を表示（例：最初の12枚）
plt.figure(figsize=(10, 10))
for x in range(0,12):
    plt.subplot(4, 3, x + 1)
    plt.imshow(cactus[x])
    x += 1

plt.tight_layout()
plt.show()


from keras import applications
# from efficientnet import EfficientNetB3
from tensorflow.keras.applications import EfficientNetB3
from keras import callbacks
from keras.models import Sequential


# train_df の 'has_cactus' 列のデータ型を文字列型（str）に変換している処理
train_df['has_cactus'] = train_df['has_cactus'].astype('str')


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

# ランダムに90度ずつ回転・左右反転・上下反転・明るさ変更する関数
def custom_preprocessing(image):
    # ランダムに0,1,2,3回回転（= 0°, 90°, 180°, 270°）
    k = random.randint(0, 3)
    image = np.rot90(image, k)
    
    # ランダムに左右反転
    if random.random() > 0.5:
        image = np.fliplr(image)

    # ランダムに上下反転
    if random.random() > 0.5:
        image = np.flipud(image)

    # 明るさをランダムに調整（0.8〜1.2倍）
    factor = random.uniform(0.8, 1.2)
    image = np.clip(image * factor, 0, 255).astype(np.uint32) / 255.0 # 画像値を0〜255から0〜1に正規化
        
    return image
    
# 学習データに対してデータ拡張・前処理（学習データの画像にいろんな変換処理を加える）
train_datagen = ImageDataGenerator(
    validation_split=0.10,      # データの10%を検証用に分ける
    # rotation_range=40,          # 画像をランダムに最大40度回転させる
    # width_shift_range=0.2,      # 横方向に最大20%移動
    # height_shift_range=0.2,     # 縦方向に最大20%移動
    # shear_range=0.2,            # せん断変換をかける
    # zoom_range=0.2,             # 20%まで拡大縮小
    # horizontal_flip=True,       # 左右反転（ランダム）
    # vertical_flip=True,          # 上下反転（ランダム）
    # brightness_range=[0.8, 1.2] # 明るさの変化
    # fill_mode='nearest'       # 変換後にできる空白部分を最も近い画素で埋める
    
    preprocessing_function=custom_preprocessing,  # カスタム処理を適用
    #rescale=1/255              # 画像値を0〜255から0〜1に正規化
)

# 訓練データ用の画像ジェネレーターを作成
train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory = train_dir,
    x_col="id",
    y_col="has_cactus",
    target_size=(32,32),
    subset="training",
    batch_size=1024,      # バッチサイズ：512→1024に変更
    shuffle=True,
    class_mode="binary"
)

# 検証用の画像ジェネレーターを作成
val_generator = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory = train_dir,
    x_col="id",
    y_col="has_cactus",
    target_size=(32,32),
    subset="validation",
    batch_size=512,      # バッチサイズ：256→512に変更
    shuffle=True,
    class_mode="binary"
)


import matplotlib.pyplot as plt

# 最初の12枚の画像を読み込み
cactus = []
for i in range(12):
    img = cv2.imread(train_dir + '/' + train_df['id'][i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCVはBGRなのでRGBに変換
    cactus.append(img)

# 前処理後の画像を保存
cactus_augmented = [custom_preprocessing(img) for img in cactus]

# 変換後の画像を表示
plt.figure(figsize=(10, 10))
for i in range(12):
    plt.subplot(4, 3, i + 1)
    plt.imshow(cactus_augmented[i])
    plt.title(f"Image {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()


# テストデータに前処理
test_datagen = ImageDataGenerator(
    rescale=1/255        # 画像値を0〜255から0〜1に正規化
)

#テスト用の画像ジェネレーターを作成
test_generator = test_datagen.flow_from_directory(
    directory = test_dir,
    target_size=(32,32),
    batch_size=1,
    shuffle=False,
    class_mode=None     # ラベル（クラス）がない前提で読み込み（予測専用）
)


from keras.layers import Dense
from keras.optimizers import Adam

# EfficientNetB3 の読み込み（特徴抽出器としてのみ使う）
efficient_net = EfficientNetB3(
    weights='imagenet',      # ImageNet で学習された EfficientNetB3 の重みを 再利用
    input_shape=(32,32,3),
    include_top=False,       # 上位層（分類層）は使わない
    pooling='max'            # 出力を1ベクトルにする
)

# 新たな分類用モデルを構築
model = Sequential()        # KerasのSequentialモデルを初期化
model.add(efficient_net)    #  EfficientNetB3モデルを追加
model.add(Dense(units = 120, activation='relu'))   # 全結合層（Dense層）を追加
model.add(Dense(units = 120, activation = 'relu')) # もう一層全結合層（Dense層）を追加
model.add(Dense(units = 1, activation='sigmoid'))  # 最終出力層を追加（出力は0～1）
model.summary()  # モデル全体の構成・パラメータ数を表示


model.compile(
    optimizer=Adam(learning_rate=0.0001),  # 最適化手法をAdamに設定、学習率は0.0001
    loss='binary_crossentropy',            # 損失関数をバイナリクロスエントロピーに設定
    metrics=['accuracy']                   # 学習中や評価時に「正解率」を計測するよう指定
)


history = model.fit(
    train_generator,
    epochs = 100,
    # 1エポック内で何回学習ステップを行うか（何バッチ処理するか）を指定
    steps_per_epoch = 15,   # 30→15に変更。通常、訓練画像枚数 / バッチサイズ でok
    validation_data = val_generator,
    # 1エポックごとに、検証データで何ステップ分の評価を行うかを指定
    validation_steps = 3,    # 7→3に変更。通常は、検証画像枚数 / バッチサイズ でok
    class_weight=class_weights_dict  # クラスウェイトの適用
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc) + 1)

plt.plot(epochs,acc,'bo',label = 'Training Accuracy')
plt.plot(epochs,val_acc,'b',label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()


# preds = model.predict_generator
preds = model.predict(
    test_generator,
    steps=len(test_generator.filenames),
    verbose=1    # 進捗バーも表示される（追記）
)


image_ids = [name.split('/')[-1] for name in test_generator.filenames]
predictions = preds.flatten()
data = {'id': image_ids, 'has_cactus':predictions} 
submission = pd.DataFrame(data)
print(submission.head(20))

# # 予測IDの取り出し
# image_ids = [os.path.basename(path) for path in test_generator.filenames]

# # 予測結果を1次元に
# predictions = preds.flatten()

# # 念のため長さを確認
# assert len(image_ids) == len(predictions), "IDと予測の数が一致しません！"

# # 提出用データフレーム
# submission = pd.DataFrame({
#     'id': image_ids,
#     'has_cactus': predictions
# })

# # 確認
# print(submission.head())


submission.to_csv("/kaggle/working/submission.csv", index=False)


print(os.listdir("/kaggle/working"))

