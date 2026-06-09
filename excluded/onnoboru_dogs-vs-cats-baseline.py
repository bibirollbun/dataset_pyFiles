# いつもの
import os
import gc
import re
import sys
import time
import copy
import random 
import glob 

# zip ファイル用
import zipfile
import shutil

# いつもの
import numpy as np
import pandas as pd

# pytorch 関連
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 画像関連
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2   
from PIL import Image

# スケジューラのためだけにいれる
import transformers

# 進捗バー
from tqdm import tqdm

# loss と KFold
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold

# 図示
import matplotlib.pyplot as plt

# 警告の無視
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class CFG :
    debug_one_epoch = True
    debug_one_fold = False
    only_infer = False
    num_workers = 16
    batch_size = 64
    num_epochs = 10
    lr = 1e-3
    early_stopping_round = 5
    random_seed = 42
    n_splits = 5
    model_name = "resnet18" # timm で使うモデル名
    pretrained_path = None
    train_dir = None # 学習データセットのパス
    test_dir = None # テストデータセットのパス
    optimizer = torch.optim.AdamW
    criterion = nn.BCEWithLogitsLoss()
    scheduler = transformers.get_linear_schedule_with_warmup
    input_imgsize = 224
    data_dir = "../input/dogs-vs-cats-redux-kernels-edition/"
    kaggle_working_dir = "/kaggle/working/"
    
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(CFG.random_seed)

if  CFG.debug_one_epoch :
    CFG.num_epochs = 1

print('KAGGLE_URL_BASE' in set(os.environ.keys()))



submission = pd.read_csv(os.path.join(CFG.data_dir, "sample_submission.csv"))
submission.head(3)


if 'KAGGLE_URL_BASE' in set(os.environ.keys()) :
    kaggle_train_dir = os.path.join(CFG.kaggle_working_dir, "train")
    # すでに解凍されている場合は解凍しない
    if not os.path.exists(kaggle_train_dir) :
        shutil.unpack_archive(os.path.join(CFG.data_dir, "train.zip"), CFG.kaggle_working_dir)
    
    kaggle_test_dir = os.path.join(CFG.kaggle_working_dir, "test")
    if not os.path.exists(kaggle_test_dir) :
        shutil.unpack_archive(os.path.join(CFG.data_dir, "test.zip"), CFG.kaggle_working_dir)
        
    CFG.data_dir = CFG.kaggle_working_dir
    
CFG.train_dir = os.path.join(CFG.data_dir, "train")
CFG.test_dir = os.path.join(CFG.data_dir, "test")


train_list = glob.glob(os.path.join(CFG.data_dir, "train", "*.jpg"))
test_list = glob.glob(os.path.join(CFG.data_dir, "test", "*.jpg"))

print(f"train data : {len(train_list)}")
print(f"test data : {len(test_list)}")


print("the number of dog : ", len([i for i in train_list if "dog" in i]))
print("the number of cat : ", len([i for i in train_list if "cat" in i]))


random_img = random.choice(train_list)
img = Image.open(random_img)
print(random_img)
print(img.size)
plt.imshow(img)


img_array = np.array(img)
print(img_array.shape)


print(img_array[:,:,0])


transform_tmp = A.Compose([
    A.Resize(CFG.input_imgsize, CFG.input_imgsize),
])
img_transformed_tmp = transform_tmp(image = np.array(img)) # numpy 配列以外受け取ってくれない
print(img_transformed_tmp.keys())
img_transformed_tmp = Image.fromarray(img_transformed_tmp["image"])
print(img_transformed_tmp.size)
plt.imshow(img_transformed_tmp)


transform_tmp = A.Compose([
    A.Resize(CFG.input_imgsize, CFG.input_imgsize),
    A.HorizontalFlip(p=1.0),
])
img_transformed_tmp = transform_tmp(image = np.array(img))
img_transformed_tmp = Image.fromarray(img_transformed_tmp["image"])
plt.imshow(img_transformed_tmp)



transform_tmp = A.Compose([
    A.Resize(CFG.input_imgsize, CFG.input_imgsize),
    A.Normalize(),
])
img_transformed_tmp = transform_tmp(image = np.array(img))["image"]
plt.imshow(img_transformed_tmp)


transform_tmp = A.Compose([
    A.Resize(CFG.input_imgsize, CFG.input_imgsize),
    ToTensorV2()
])
img_transformed_tmp = transform_tmp(image = np.array(img))
print(img_transformed_tmp.keys())
print(type(img_transformed_tmp["image"]))
print(img_transformed_tmp["image"].shape)


# パスと class をまとめた DataFrame を作成
train_df = pd.DataFrame(train_list, columns=["path"])
train_df["class"] = train_df["path"].apply(lambda x : x.split("/")[-1].split(".")[0])
train_df["class"] = train_df["class"].map({"dog" : 1, "cat" : 0})
test_df = pd.DataFrame(test_list, columns=["path"])
test_df["class"] = -1
# test_df に対しては path の数字が昇順であることを保証するために id を追加
test_df["id"] = test_df["path"].apply(lambda x : int(x.split("/")[-1].split(".")[0]))
test_df = test_df.sort_values("id").reset_index(drop=True)

train_df.head(3)


test_df.head(3)


train_transform = A.Compose([
    A.Resize(CFG.input_imgsize, CFG.input_imgsize),
    A.HorizontalFlip(p=0.5), # 50% の確率で水平反転
    A.Normalize(),
    ToTensorV2()
])
test_transform = A.Compose([
    A.Resize(CFG.input_imgsize, CFG.input_imgsize),
    A.Normalize(),
    ToTensorV2()
])


class DogsCatsDataset(Dataset) :
    def __init__(self, df, transform=None) :
        self.df = df # さっきの pandas dataframe を受け取る
        self.transform = transform # 画像の変換処理を受け取る

    def __len__(self) :
        return len(self.df)
    
    def __getitem__(self, idx) :
        img = Image.open(self.df.iloc[idx, 0])
        img = self.transform(image = np.array(img))["image"]
        label = self.df.iloc[idx, 1].astype(np.float32)
        return img, label


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion) :
    model.train()
    losses = []
    for img, label in tqdm(dataloader) :
        img = img.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output.squeeze(-1),label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        
    return np.mean(losses)


def eval_one_epoch(model, dataloader, criterion) :
    model.eval()
    losses = []
    all_labels = []
    all_outputs = []
    with torch.no_grad() :
        for img, label in tqdm(dataloader) :
            img = img.to(device)
            label = label.to(device)
            output = model(img) # 予測
            loss = criterion(output.squeeze(-1), label) # いらない次元を潰して, loss を計算
            losses.append(loss.item()) # loss をリストに追加
            all_labels.extend(label.cpu().numpy()) # ラベルをリストに追加
            pred = torch.sigmoid(output).cpu().numpy()
            all_outputs.extend(pred) # 予測をリストに追加、 sigmoid で確率に変換
    
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    
    return {
        "bce_loss" : np.mean(losses),
        "log_loss" : log_loss(all_labels, all_outputs),
        "labels" : all_labels,
        "outputs" : all_outputs
    }


def infer(model, dataloader, test=False) :
    model.eval()
    all_outputs = []
    with torch.no_grad() :
        for img, label in tqdm(dataloader) :
            if test :
                assert(label[0] == -1)
            img = img.to(device)
            output = model(img)
            all_outputs.extend(torch.sigmoid(output).cpu().numpy()) # 予測をリストに追加、 sigmoid で確率に変換
            
            
    all_outputs = np.array(all_outputs)
    return all_outputs


def run_train_cv(train, test):
    kf = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.random_seed)
    oof = np.zeros((len(train), 1)) # out of fold の予測
    predictions =[]
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train)) :
        print(f"====================fold : {fold}====================")
        # df を分割
        train_df = train.iloc[train_idx].reset_index(drop=True)
        valid_df = train.iloc[valid_idx].reset_index(drop=True)
        
        # dataset と dataloader を作成
        train_dataset = DogsCatsDataset(train_df, transform=train_transform)
        valid_dataset = DogsCatsDataset(valid_df, transform=test_transform)
        test_dataset = DogsCatsDataset(test_df, transform=test_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, drop_last=True, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
        
        # timm でモデルを作成、学習済みの重みを読み込む
        model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)
        model.to(device)
        
        # optimizer と scheduler を作成
        optimizer = CFG.optimizer(model.parameters(), lr=CFG.lr)
        training_steps = len(train_loader) * CFG.num_epochs
        warmup_steps = int(training_steps * 0.1)
        scheduler = CFG.scheduler(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)
        
        # early stopping のための変数
        best_loss = np.inf
        early_stopping_round = 0
        
        # 学習
        for epoch in range(CFG.num_epochs):
            start_time = time.time() # 時間計測
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, CFG.criterion)
            valid_result = eval_one_epoch(model, valid_loader, CFG.criterion)
            print(f"epoch : {epoch} - train loss : {train_loss} - valid loss : {valid_result['bce_loss']} - valid log loss : {valid_result['log_loss']}")
            
            # valid の loss が改善した場合にモデルを保存
            if valid_result["bce_loss"] < best_loss :
                best_loss = valid_result["bce_loss"]
                early_stopping_round = 0
                torch.save(model.state_dict(), f"{CFG.model_name}_fold{fold}.pth")
                
            # 改善しない場合 early stopping の判定をいれる
            else :
                early_stopping_round += 1
                if early_stopping_round > CFG.early_stopping_round :
                    break
            
            print(f"spend time for epoch {epoch} : {time.time() - start_time}")
            
        oof[valid_idx] = infer(model, valid_loader) # valid に対する予測、これを 5 つの fold で集めて合体することで、リークなしで train に対して予測が出来る
        
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        
        if CFG.debug_one_fold :
            break
        
    for fold in range(CFG.n_splits) :
        model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)
        model.load_state_dict(torch.load(f"{CFG.model_name}_fold{fold}.pth"))
        model.to(device)
        predictions.append(infer(model, test_loader, test=True))
        if CFG.debug_one_fold :
            break
        
    predictions = np.mean(predictions, axis=0)
    
        
    return {
        "oof" : oof,
        "predictions" : predictions
    }



def main() :
    if CFG.only_infer :
        test_dataset = DogsCatsDataset(test_df, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
        model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)
        model.to(device)
        predictions = infer(model, test_loader)
        submission["label"] = predictions
        submission.to_csv("submission.csv", index=False)
       
        
    else :
        result = run_train_cv(train_df, test_df)
        oof_preds = result["oof"]
        predictions = result["predictions"]
        submission["label"] = predictions
        submission.to_csv("submission.csv", index=False)
        train_df["oof_preds"] = oof_preds   
        train_df.to_csv("oof_preds.csv", index=False)
        if CFG.debug_one_fold == False :
            print(f"oof log loss : {log_loss(train_df['class'], oof_preds)}")
        
if __name__ == "__main__" :
    main()

