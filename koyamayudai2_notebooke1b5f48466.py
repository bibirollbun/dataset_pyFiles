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
zip_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
# 展開先のパス
extract_path = '/kaggle/working/train'

# 展開処理
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


import os

# 展開先のパス
image_dir = '/kaggle/working/train/train'
image_files = os.listdir(image_dir)
print("画像の総数:", len(image_files))

# ファイル名からラベルを作成
data = []
for fname in image_files:
    if fname.endswith('.jpg'):  # jpg画像のみ対象
        label = 'dog' if 'dog' in fname else 'cat'
        data.append({'filename': fname, 'label': label})

# DataFrameに変換
df = pd.DataFrame(data)
print(df.head())


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math

# 画像一覧
all_files = df['filename'].tolist()
n_images = len(all_files)
cols = 5  # 1行あたりの画像数
rows = math.ceil(n_images / cols)

plt.figure(figsize=(cols * 3, rows * 3))  # 画像サイズに応じて調整

for i, fname in enumerate(all_files):
    img_path = os.path.join(image_dir, fname)
    img = mpimg.imread(img_path)
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(fname)
    plt.axis('off')

plt.tight_layout()
plt.show()

