import numpy as np
import pandas as pd
import os
import time
from fastai.vision.all import *
import zipfile
import albumentations as Alb


# データセットの展開
with zipfile.ZipFile('../input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
with zipfile.ZipFile('../input/dogs-vs-cats-redux-kernels-edition/test.zip', 'r') as zip_ref:
    zip_ref.extractall('.')


# 不良画像の削除
bad_images = [
    'dog.10797.jpg', 'dog.10747.jpg', 'dog.10237.jpg', 'dog.9517.jpg',
    'dog.8736.jpg', 'dog.5604.jpg', 'dog.1043.jpg', 'cat.4338.jpg',
    'dog.10161.jpg', 'dog.10190.jpg'
]
for img in bad_images:
    os.remove(f'./train/{img}')


# データセットの読み込み
tpath = "./train"
ftrain = get_image_files(tpath)
print('Train set size:', len(ftrain))


# データ拡張の定義
class AlbTransform(Transform):
    def __init__(self, aug): self.aug = aug
    def encodes(self, img: PILImage):
        aug_img = self.aug(image=np.array(img))['image']
        return PILImage.create(aug_img)

# Albumentationsのデータ拡張（ぼかし処理含む）
def get_augs():
    return Alb.Compose([
        Alb.ShiftScaleRotate(rotate_limit=20, border_mode=0, value=(0,0,0)),
        Alb.Transpose(),
        Alb.Flip(),
        Alb.RandomRotate90(),
        Alb.RandomBrightnessContrast(),
        Alb.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5),
        Alb.CoarseDropout(),
        
        # ぼかし処理の追加（どちらかを確率で適用）
        Alb.OneOf([
            Alb.Blur(blur_limit=3, p=0.5),              # 平均ぼかし
            Alb.GaussianBlur(blur_limit=(3, 5), p=0.5)  # ガウスぼかし
        ], p=0.5)  # このOneOf自体を50%の確率で適用
    ])

# 画像1枚ごとの前処理
item_tfms = [Resize(224), AlbTransform(get_augs())]

# バッチ単位での標準化と追加変換
batch_tfms = [Normalize.from_stats(*imagenet_stats), *aug_transforms()]


# データローダーの作成
dls = ImageDataLoaders.from_name_re(
    path=tpath, fnames=ftrain, pat=r'(.+)\.\d+.jpg$', valid_pct=0.1, 
    item_tfms=item_tfms, batch_tfms=batch_tfms, bs=64, shuffle=True
)

print('train items:', len(dls.train.items), 'validation items:', len(dls.valid.items))


# モデルの作成
learn = cnn_learner(dls, resnet50, metrics=[error_rate, accuracy])


# 学習率のスケジューリングを追加
learn.fit_one_cycle(5, lr_max=slice(1e-3, 1e-2))


# テストデータの準備
ftest = get_image_files('test')
print('Testing', len(ftest), 'items')


# テストデータローダーの作成
tst_dl = dls.test_dl(ftest, with_labels=False, shuffle=False)
tst_dl.show_batch(max_n=12)


# 推論の実行
startTime = time.time()
preds = learn.tta(dl=tst_dl, n=5, use_max=False)
print('TTA in:', time.time()-startTime, 'secs')


# 結果の保存
subm_df = pd.DataFrame()
subm_df['id'] = [item.stem for item in tst_dl.items]
subm_df['label'] = preds[0][:,1].clip(0.005, 0.995)
subm_df.to_csv('submission.csv', header=True, index=False)


# ソフトマックスを使用した予測結果の保存
subm_df['label'] = torch.softmax(preds[0], dim=1)[:, 1]
subm_df.to_csv('submission-softmax.csv', header=True, index=False)


# 推論結果の可視化
tst_dl.show_batch(max_n=12)


# クリーンアップ
from shutil import rmtree
rmtree('./train', ignore_errors=True)
rmtree('./test', ignore_errors=True)

