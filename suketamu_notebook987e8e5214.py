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


# ----------- train画像のリサイズとグレースケール化確認-----------
for fname in train_df['filepath']:
    src_path = os.path.join(train_dir, fname)
    dst_path = os.path.join(train_resized_dir, fname)
    with Image.open(src_path) as img:
        # リサイズ後にグレースケールへ変換し、保存する
        img_converted = img.resize((128, 128)).convert('L')
        img_converted.save(dst_path)

print("train画像のグレースケール化完了")
# ----------- test画像のグレースケール化確認 -----------
for fname in test_df['filepath']:
    src_path = os.path.join(test_dir, fname)
    dst_path = os.path.join(test_resized_dir, fname)
    with Image.open(src_path) as img:
        # リサイズ後にグレースケールへ変換し、保存する
        img_converted = img.resize((128, 128)).convert('L')
        img_converted.save(dst_path)

print("test画像のグレースケール化完了")



from PIL import Image
import os

# 確認したい画像が保存されているフォルダ
# test_128 フォルダも同様に確認できます
target_dir = '/kaggle/working/train_128'

# フォルダ内のファイル名リストを取得
try:
    file_list = os.listdir(target_dir)

    # 最初の5枚だけをサンプルとして確認する
    for fname in file_list[:5]:
        image_path = os.path.join(target_dir, fname)
        with Image.open(image_path) as img:
            print(f"ファイル名: {fname}, 画像モード: {img.mode}")
            if img.mode == 'L':
                print("✅ 正しくグレースケール化されています。")
            else:
                print(f"❌ グレースケールではありません。(モード: {img.mode})")
            print("-" * 20) # 区切り線

except FileNotFoundError:
    print(f"エラー: {target_dir} が見つかりません。")
except IndexError:
    print(f"エラー: {target_dir} にファイルがありません。")


from PIL import Image
import os
import random # randomライブラリをインポート

# 確認したい画像が保存されているフォルダ
target_dir = '/kaggle/working/train_128'

try:
    file_list = os.listdir(target_dir)

    # リストの中からランダムに5つのファイルを選ぶ
    # ファイル数が5未満の場合は、存在するすべてのファイルを選ぶ
    num_samples = min(5, len(file_list))
    if num_samples > 0:
        sample_files = random.sample(file_list, num_samples)
    else:
        sample_files = [] # ファイルがない場合は空のリスト

    if not sample_files:
        print(f"フォルダ {target_dir} にファイルがありません。")

    # サンプル画像をチェック
    for fname in sample_files:
        image_path = os.path.join(target_dir, fname)
        with Image.open(image_path) as img:
            print(f"ファイル名: {fname}, 画像モード: {img.mode}")
            if img.mode == 'L':
                print("✅ 正しくグレースケール化されています。")
            else:
                print(f"❌ グレースケールではありません。(モード: {img.mode})")
            print("-" * 20) # 区切り線

except FileNotFoundError:
    print(f"エラー: {target_dir} が見つかりません。")


import numpy as np
from PIL import Image
import os
import random

# --- 準備：確認用の画像を選ぶ ---
target_dir = '/kaggle/working/train_128'
try:
    # ランダムに1枚選ぶ
    fname = random.choice(os.listdir(target_dir))
    image_path = os.path.join(target_dir, fname)
    print(f"確認するファイル: {image_path}")

    # 1. 画像をグレースケールで開く
    with Image.open(image_path) as img:
        # 2. 画像をNumPy配列に変換
        pixel_array = np.array(img)

    # 3. 正規化前のデータを確認
    print(f"\n--- 正規化前 ---")
    print(f"データ型: {pixel_array.dtype}")
    print(f"形状 (高さ, 幅): {pixel_array.shape}")
    print(f"最大値: {pixel_array.max()}, 最小値: {pixel_array.min()}")

    # 4. 正規化を実行
    #    255.0で割ることで、データ型が自動的に浮動小数点数(float)になる
    normalized_array = pixel_array / 255.0

    # 5. 正規化後のデータを確認
    print(f"\n--- 正規化後 ---")
    print(f"データ型: {normalized_array.dtype}")
    print(f"形状 (高さ, 幅): {normalized_array.shape}")
    print(f"最大値: {normalized_array.max()}, 最小値: {normalized_array.min()}")

except (FileNotFoundError, IndexError):
    print(f"エラー: {target_dir} に確認できるファイルがありません。")


