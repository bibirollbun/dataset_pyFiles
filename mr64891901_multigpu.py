#ライブラリのimport
import numpy as np
import pandas as pd
import time
import argparse
from pathlib import Path

from tqdm.auto import tqdm
from torch.utils.data import DataLoader, default_collate
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms.functional as TTF
import timm
import yaml

import multiprocessing as mp
from queue import Empty


INPUT_PATH = Path('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025')
MODEL_PATH = Path('/kaggle/input/bacterial-public/weights/object/baseline')


#指定したフォルダ内のtomogramのpathを取得し、tomo_pathsに格納する
def get_tomos_path(input_path:Path, data_type:str,*,n=None) -> list[Path]:

    #input_path='input/', data_type='train'ならばdata_path = input/train/
    data_path = input_path / data_type

    #data_pathにあるフォルダまたはファイルを全件取得し、ソートしてリストに格納
    tomo_paths = sorted(data_path.glob('*'))

    #nが指定されていて、data_typeがtrainの場合、先頭のn件のtomoを使用
    if(n is not None) and (data_type == 'train'):
        tomo_path = tomo_paths[:n]

    return tomo_paths


#引数の画像を640×640にリサイズし、パーセンタイルに基づいて正規化する
def preprocess(img : torch.Tensor) -> torch.Tensor:

    #サイズの統一
    size = (640, 640)

    #型変換(PyTorchモデルではfloat32が基本)
    img = img.to(dtype=torch.float32)

    #一括でリサイズ
    img = TTF.resize(img, size)

    #quantile基準点の定義
    #テンソルを画像と同じでデバイスに載せる
    q = torch.Tensor([0.05, 0.95]).to(img.device)

    #shape(バッチサイズ、チャンネル数、高さ、幅)
    batch_size, nch, h, w = img.shape

    #viewで一枚の画像を一次元ベクトルにする
    #各画像の5%・95%の値を求める
    x_min, x_max = torch.quantile(img.view(batch_size, nch*h*w), q, dim=1)

    x_min = x_min.view(batch_size, 1, 1, 1)
    x_max = x_max.view(batch_size, 1, 1, 1)

    #スケーリング
    img = (img - x_min) / (x_max - x_min)

    #クリッピング
    img = torch.clamp(img, 0, 1)

    return img


#tomo_pathに含まれる画像を一枚ずつ読み込んで、DataLoaderで使える形式にする
class Dataset(torch.utils.data.Dataset):

    #画像ファイル一覧を取得
    def __init__(self, tomo_path: Path):
        self.filenames = sorted(tomo_path.glob('*'))
        
    #このデータセットに含まれる「データ数（スライス画像の数）」
    def __len__(self) -> int:
        return len(self.filenames)

    #インデックス番号の画像を読込み、辞書形式で返す
    def __getitem__(self, i: int) -> dict:
        filename = self.filenames[i]      # Path
        filebase = filename.stem          #ファイル名取得
        assert filebase[:6] == 'slice_'   #ファイル名チェック
        slice_number = int(filebase[6:])  # slice_0000 -> int(0000)

        #画像を開いて numpy 配列にし、チャンネル軸（C=1）を追加（PyTorchの形式に合わせる）
        img = Image.open(filename)
        W, H = img.size
        img = np.expand_dims(np.array(img), axis=0)

        #辞書で返す値を設定
        ret = {'img': img,
               'slice_number': slice_number,
               'shape': np.array((H, W), dtype=int),
        }

        return ret

    def loader(self, batch_size: int, num_workers: int):
        loader = DataLoader(self, batch_size=batch_size, num_workers=num_workers)
        return loader


class Model(nn.Module):
    #構造を作る
    def __init__(self, cfg_model: dict, *, pretrained=True, verbose=True):
        super().__init__()

        # Timm encoder
        name = cfg_model['encoder']   #使うモデルの種類を指定
        in_channels = 1               #チャンネル数(白黒)
        out_channels = 1              #マスク画像のチャンネル数

        self.encoder = timm.create_model(name,                    #モデル名
                                         in_chans=in_channels,    #入力チャンネル数
                                         features_only=True,      #中間の特徴マップを取り出す
                                         pretrained=pretrained)   #事前学習済みの重みを使用する?


        #encoderから出てくる各ステージの特徴マップのチャンネル数を取得する
        encoder_channels = self.encoder.feature_info.channels()

        self.segmentation_head = nn.Conv2d(encoder_channels[-1],  #入力チャンネル数
                                           out_channels,          #出力チャンネル数
                                           kernel_size=3,         #カーネルサイズ
                                           padding=1)             #出力サイズを維持する

        self.regression_head = nn.Conv2d(encoder_channels[-1], out_channels=2,
                                         kernel_size=3, padding=1)

        self.criterion_seg = nn.BCEWithLogitsLoss()
        self.criterion_reg = nn.MSELoss()

        if verbose:
            print(name)

    #データの流れを定義する
    def forward(self, img: torch.Tensor):

        features = self.encoder(img)
        out = features[-1]  # (batch_size, embed_dim, h, w)
        y_pred = self.segmentation_head(out)  # (batch_size, 1, h, w)
        t_pred = self.regression_head(out)    # (batch_size, 2, h, w)

        return y_pred, t_pred


#モーターが一番ありそうなスライスと位置を特定
def predict(tomo_path: Path,models: list[nn.Module],cfg: dict) -> dict:
    
    #各種設定
    assert len(models) > 0
    tomo_id = tomo_path.name                       #ファイルの名前
    batch_size = cfg['batch_size']                 #バッチサイズ
    num_workers = cfg['num_workers']               #データ読込みの並列数
    use_amp = cfg['use_amp']                       #AMPを使うか？
    preprocess_device = cfg['preprocess_device']   #デバイス
    assert preprocess_device == 'cpu' or preprocess_device.startswith('cuda')

    #DataLoaderを生成(スライス画像をバッチで取り出す準備)
    dataset = Dataset(tomo_path)
    loader = dataset.loader(batch_size=batch_size, num_workers=num_workers)

    #モデルの載っているデバイスを確認(GPUを優先する)
    device = next(models[0].parameters()).device

    #予測値の格納場所
    best = (0, None)

    #バッチごとの処理
    for d in loader:        
        if preprocess_device.startswith('cuda'):
            img = d['img'].to(device)  # 画像をまずGPUへ
            img = preprocess(img)      # GPU上で処理
        elif preprocess_device == 'cpu':
            img = preprocess(d['img']) # CPU上で処理
            img = img.to(device)       # 処理後にGPUへ
        else:
            raise ValueError(f"Unknown preprocess_device: {preprocess_device}")


        #アンサンブル平均をとる
        y_pred_sum, t_pred_sum = None, None
        for model in models:
            with torch.no_grad():                            #勾配計算なし
                with torch.amp.autocast(device_type='cuda',  #AMP
                                        enabled=use_amp,
                                        dtype=torch.float16):
                    y_pred, t_pred = model(img) 

            y_pred = y_pred.sigmoid()  

            if y_pred_sum is None:
                y_pred_sum = y_pred
                t_pred_sum = t_pred
            else:
                y_pred_sum += y_pred
                t_pred_sum += t_pred

        #バッチの中で一番モーターっぽいやつ
        y_pred_max = y_pred_sum.max().item() / len(models)
        del y_pred, t_pred

        # マスクの中から一番スコアが高い場所を探し、(スライス番号+Y座標+X座標)を返す
        if y_pred_max > best[0]:             #best=(スコア, スライス番号, y, x)
            bs, _, h, w = y_pred_sum.shape   #バッチサイズ, チャンネル数, 特徴マップの高さと幅
        
            argmax = torch.unravel_index(y_pred_sum.argmax(), y_pred_sum.shape)  # b, ch, iy, ix
            i, _, iy, ix = [t.item() for t in argmax]    
            slice_number = d['slice_number'][i].item()   #スライス番号を取得
            offset = t_pred_sum[i, :, iy, ix].cpu().numpy() / len(models)  

            #元画像のピクセル単位に変換
            H, W = d['shape'][i].numpy()    
            x = (ix + offset[0]) * (W / w)
            y = (iy + offset[1]) * (H / h)

            #結果の更新
            best = (y_pred_max, slice_number, y, x)

    assert best[1] is not None

    # 予測結果を辞書にまとめて返す
    n_slices = len(dataset.filenames)
    pred = {'tomo_id': tomo_id,
            'n_slices': n_slices,
            'y_pred': best[0],
            'zyx': best[1:]}
    return pred


#スコアを判定
def create_submission(preds: list, th: float, ofilename: str) -> pd.DataFrame:

    rows = []             #結果をためる
    count_positive = 0    #モーターが見つかった件数
    for pred in preds:
        if pred['y_pred'] < th:
            zyx = (-1, -1, -1)
        else:
            count_positive += 1
            zyx = pred['zyx']

        row = {'tomo_id': pred['tomo_id'],
               'Motor axis 0': zyx[0],
               'Motor axis 1': zyx[1],
               'Motor axis 2': zyx[2]}
        rows.append(row)

    submit = pd.DataFrame(rows)
    submit.to_csv(ofilename, float_format='%.8e', index=False)

    print('Submit %s: %d positives / %d tomo_ids' % (ofilename, count_positive, len(rows)))

    return submit


def process_fn(process_id: int,  #GPUの番号
               tomo_queue,       #推論すべきtomoのリストが入ったキュー
               pred_queue,       #予測結果を格納するキュー
               cfg: dict):       #各種設定(モデル・バッチサイズ・AMPなどの設定が入った辞書)




    cfg = cfg.copy()
    cfg['preprocess_device'] = 'cuda' 

    device = torch.device('cuda:%d' % process_id)

    #モデルの読込み
    model_path = cfg['model_path']
    folds = cfg['folds']


    with open(model_path / 'config.yml', 'r') as f:
        cfg_model = yaml.safe_load(f)
            
    models = []
    for ifold in folds:
        model_filename = '%s/model%d.pytorch' % (model_path, ifold)            
        model = Model(cfg_model['model'], pretrained=False, verbose=False)     #モデル構造をインスタンス化
        model.load_state_dict(torch.load(model_filename, weights_only=True))   #重みを読込み
        model.to(device)                                                       #GPUに載せる
        model.eval()                                                           #推論モードに切り替え
        models.append(model)

        if process_id == 0:
            print('Load model', model_filename)
        
    #キューから一つずつとって推論
    while not tomo_queue.empty():
        try:
            tomo_path = tomo_queue.get(timeout=1)
            pred = predict(tomo_path, models, cfg)
            pred_queue.put(pred)

        except Empty:
            break

tb = time.time()


cfg = {
    'model_path': MODEL_PATH,    #モデルの保存先
    'folds': [0,1,2,3,4],        #使用するモデルの番号リスト
    'batch_size': 16,            #一度に処理するスライス数
    'num_workers': 1,            #DataLoaderの並列データ読込みスレッド数
    'use_amp': True,             #AMPをON
    'preprocess_device': 'cuda', #GPUを使う
}

tomo_paths = get_tomos_path(INPUT_PATH, 'test')

#テストデータの場合trainからデータを足す
if len(tomo_paths) == 3:
     #Some experiment when test is dummy (optional)
     #tomo_paths = get_tomos(INPUT_PATH, 'train', n=20, random_sample=True)
    pass

print('Data %d' % len(tomo_paths))

manager = mp.Manager()
#予測したいtomoを詰めるキュー
tomo_queue = manager.Queue()
#各GPUが出した予測結果を詰めるキュー
pred_queue = manager.Queue()
#処理対象のtomoを予測したいtomoキューに格納
for tomo_path in tomo_paths:
    tomo_queue.put(tomo_path)

time.sleep(1)
assert not tomo_queue.empty()

#GPUの数だけprcess_fnを同時実行するために、プロセスを作ってリストにまとめる
num_processes = 2
tb = time.time()

workers = [mp.Process(target=process_fn,
                      args=(i, tomo_queue, pred_queue, cfg))
           for i in range(num_processes)]

for w in workers:
    w.start()

for w in workers:
    w.join()


dt = time.time() - tb
print('%.2f sec for %d tomos' % (dt, len(tomo_paths)))

#処理を集めてキューに格納
preds = []
try:
    while not pred_queue.empty():
        preds.append(pred_queue.get(timeout=1))
except Empty:
    pass

assert len(preds) == len(tomo_paths)


th = 0.5        #閾値
ofilename = 'submission.csv'
create_submission(preds, th, 'submission.csv')
print(ofilename, 'written')

