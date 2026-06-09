import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch import nn, optim
from  torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from matplotlib import pyplot as plt
import os, random, gc
import json
from  ast import literal_eval
from sklearn.metrics import label_ranking_average_precision_score
from tqdm.notebook import tqdm
import joblib
import seaborn as sns


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


df = pd.read_csv('/kaggle/input/birdclef-2021/train_metadata.csv')


train_labels = df['primary_label']
label_counts = train_labels.value_counts()
top_100_counts = label_counts[:100]

plt.figure(figsize=(15, 7))
sns.barplot(x=top_100_counts.index, y=top_100_counts.values)
plt.xticks(rotation=90)
plt.xlabel("ĞœĞµÑ‚ĞºĞ° ĞºĞ»Ğ°Ñ�Ñ�Ğ°")
plt.ylabel("ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ°ÑƒĞ´Ğ¸Ğ¾")
plt.title("Ğ¢Ğ¾Ğ¿ 100 ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ² Ğ¿Ğ¾ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ñƒ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… Ğ² Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰ĞµĞ¼ Ğ½Ğ°Ğ±Ğ¾Ñ€Ğµ")
plt.tight_layout()
plt.show()


NUM_CLASSES = 397
SR = 32000
DURATION = 7
MAX_READ_SAMPLES = 5
DATA_ROOT = Path('/kaggle/input/birdclef-2021')
MEL_PATHS = sorted(Path('/kaggle/input').glob('mels-birds-train*/rich_train_metadata.csv'))
TRAIN_LABEL_PATHS = sorted(Path('/kaggle/input').glob('mels-birds-train*/LABEL_IDS.json'))
MODEL_ROOT = Path('.')
TRAIN_BATCH_SIZE = 100
TRAIN_NUM_WORKERS = 2
VAL_BATCH_SIZE = 128
VAL_NUM_WORKERS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)


def get_df(mel_paths=MEL_PATHS, train_label_paths=TRAIN_LABEL_PATHS):
  df_list = []
  LABEL_IDS = {}
    
  for file_path in mel_paths:
    temp = pd.read_csv(str(file_path), index_col=0)
    temp['impath'] = temp.apply(
        lambda row: file_path.parent/'audio_images/{}/{}.npy'.format(row.primary_label, row.filename), 
        axis=1
    ) 
    df_list.append(temp)

  df = pd.concat(df_list, ignore_index=True)
  df['secondary_labels'] = df['secondary_labels'].apply(literal_eval)

  for file_path in train_label_paths:
    with open(str(file_path)) as f:
      LABEL_IDS.update(json.load(f))

  return LABEL_IDS, df


def get_efficientnet_model(name, num_classes=NUM_CLASSES):
    if 'efficientnet' in name:
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    else:
        raise RuntimeError('Ğ�ĞµĞ·Ğ½Ğ°ĞºĞ¾Ğ¼Ğ°Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ')
    if hasattr(model, 'fc'):
        nb_ft = model.fc.in_features
        model.fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, '_fc'): 
        nb_ft = model._fc.in_features
        model._fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            for layer in reversed(model.classifier):
                if hasattr(layer, 'in_features'):
                    nb_ft = layer.in_features
                    break
            model.classifier = nn.Linear(nb_ft, num_classes)
        else:
            nb_ft = model.classifier.in_features
            model.classifier = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, 'last_linear'):
        nb_ft = model.last_linear.in_features
        model.last_linear = nn.Linear(nb_ft, num_classes)

    return model


def load_data(df):
    def load_row(row):
        return row.filename, np.load(str(row.impath))[:MAX_READ_SAMPLES]

    pool = joblib.Parallel(4)
    mapper = joblib.delayed(load_row)
    tasks = [mapper(row) for row in df.itertuples(False)]
    res = pool(tqdm(tasks))
    res = dict(res)
    return res


class BirdClefDataset(Dataset):
    def __init__(self, audio_image_store, meta, sr=SR, is_train=True, num_classes=NUM_CLASSES, duration=DURATION):
        self.audio_image_store = audio_image_store
        self.meta = meta.copy().reset_index(drop=True)
        self.sr = sr
        self.is_train = is_train
        self.num_classes = num_classes
        self.duration = duration
        self.audio_length = self.duration*self.sr
    
    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])
        return image

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        image = self.audio_image_store[row.filename]

        image = image[np.random.choice(len(image))]
        image = self.normalize(image)
        
        t = np.zeros(self.num_classes, dtype=np.float32) + 0.0025
        t[row.label_id] = 0.995
        
        return image, t


@torch.no_grad()
def evaluate(net, criterion, val_laoder):
    net.eval()
    os, y = [], []
    val_laoder = tqdm(val_laoder, leave = False, total=len(val_laoder))

    for icount, (xb, yb) in  enumerate(val_laoder):
        y.append(yb.to(DEVICE))
        xb = xb.to(DEVICE)
        o = net(xb)
        os.append(o)

    y = torch.cat(y)
    o = torch.cat(os)

    l = criterion(o, y).item()
    
    o = o.sigmoid()
    y = (y > 0.5)*1.0

    lrap = label_ranking_average_precision_score(y.cpu().numpy(), o.cpu().numpy())

    o = (o > 0.5)*1.0

    prec = ((o*y).sum()/(1e-6 + o.sum())).item()
    rec = ((o*y).sum()/(1e-6 + y.sum())).item()
    f1 = 2*prec*rec/(1e-6+prec+rec)

    return l, lrap, f1, rec, prec


class AutoSave:
    def __init__(self, top_k=3, metric="f1_val", mode="max", root=None, name="ckpt", save_best=True):
        self.top_k = top_k
        self.logs = []
        self.metric = metric
        self.mode = mode
        self.root = Path(root or MODEL_ROOT)
        assert self.root.exists()
        self.name = name
        self.save_best = save_best

        self.top_models = []
        self.top_metrics = []
        self.best_metric = -float('inf') if mode == "max" else float('inf')
        self.best_epoch = -1

    def log(self, model, metrics):
        metric = metrics[self.metric]
        rank = self.rank(metric)

        self.top_metrics.insert(rank+1, metric)
        if len(self.top_metrics) > self.top_k:
            self.top_metrics.pop(0)

        self.logs.append(metrics)
        
        # Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ² Ñ‚Ğ¾Ğ¿-K
        self.save(model, metric, rank, metrics["epoch"])
        
        # Ğ�Ñ‚Ğ´ĞµĞ»ÑŒĞ½Ğ¾ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ»ÑƒÑ‡ÑˆÑƒÑ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ
        if self.save_best:
            self.save_best_model(model, metric, metrics["epoch"])

    def save_best_model(self, model, metric, epoch):
        """Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµÑ‚ Ğ»ÑƒÑ‡ÑˆÑƒÑ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ¾Ñ‚Ğ´ĞµĞ»ÑŒĞ½Ğ¾"""
        is_better = (self.mode == "max" and metric > self.best_metric) or \
                   (self.mode == "min" and metric < self.best_metric)
        
        if is_better:
            self.best_metric = metric
            self.best_epoch = epoch
            
            # Ğ£Ğ´Ğ°Ğ»Ñ�ĞµĞ¼ Ğ¿Ñ€ĞµĞ´Ñ‹Ğ´ÑƒÑ‰ÑƒÑ� Ğ»ÑƒÑ‡ÑˆÑƒÑ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ
            best_pattern = f"{self.name}_best_*.pth"
            for old_best in self.root.glob(best_pattern):
                old_best.unlink()
            
            # Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ½Ğ¾Ğ²ÑƒÑ� Ğ»ÑƒÑ‡ÑˆÑƒÑ� (Ğ±ĞµĞ· Ğ»Ğ¸ÑˆĞ½Ğ¸Ñ… Ñ�Ğ¸Ğ¼Ğ²Ğ¾Ğ»Ğ¾Ğ²)
            best_name = f"{self.name}_best_epoch{epoch:02d}_{metric:.4f}.pth"
            best_path = self.root / best_name
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'metric': metric,
                'metric_name': self.metric,
                'logs': self.logs
            }, best_path.as_posix())
            
            print(f"ğŸ�† NEW BEST! Epoch {epoch}, {self.metric}: {metric:.4f}")
            print(f"ğŸ’¾ Saved: {best_name}")

    def save(self, model, metric, rank, epoch):
        """Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµÑ‚ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ² Ñ‚Ğ¾Ğ¿-K"""
        # Ğ‘Ğ¾Ğ»ĞµĞµ Ğ¿Ñ€Ğ¾Ñ�Ñ‚Ğ¾Ğµ Ğ¸Ğ¼Ñ� Ñ„Ğ°Ğ¹Ğ»Ğ°
        name = f"{self.name}_epoch{epoch:02d}_{metric:.4f}.pth"
        path = self.root / name

        old_model = None
        self.top_models.insert(rank+1, name)
        if len(self.top_models) > self.top_k:
            old_model = self.root / self.top_models[0]
            self.top_models.pop(0)      

        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metric': metric,
            'metric_name': self.metric
        }, path.as_posix())

        if old_model and old_model.exists():
            old_model.unlink()
            print(f"ğŸ—‘ï¸� Removed: {old_model.name}")

        self.to_json()

    def rank(self, val):
        if self.mode == "max":
            for i, top_val in enumerate(self.top_metrics):
                if val <= top_val:
                    return i - 1
            return len(self.top_metrics) - 1
        else:
            for i, top_val in enumerate(self.top_metrics):
                if val >= top_val:
                    return i - 1
            return len(self.top_metrics) - 1

    def to_json(self):
        log_name = f"{self.name}_logs.json"
        log_path = self.root / log_name
        with log_path.open("w") as f:
            json.dump(self.logs, f, indent=2)

    def get_best_model_info(self):
        if self.best_epoch == -1:
            return None
        return {
            'epoch': self.best_epoch,
            'metric': self.best_metric,
            'metric_name': self.metric
        }


def one_fold(model_name, fold, train_set, val_set, epochs=50, save=True, save_root=None, patience=7):
    save_root = Path(save_root) or MODEL_ROOT
    saver = AutoSave(
        root=save_root, 
        name=f"birdclef_{model_name}_fold{fold}", 
        metric="f1_val",
        mode="max",
        top_k=3,
        save_best=True
    )
    
    net = get_efficientnet_model(model_name).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-6, T_max=epochs)
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=1, min_lr=1e-6, threshold=0.002, threshold_mode='rel'
    )
    
    train_data = BirdClefDataset(audio_image_store, meta=df.iloc[train_set].reset_index(drop=True),
                             sr=SR, duration=DURATION, is_train=True)
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, num_workers=TRAIN_NUM_WORKERS, 
                            shuffle=True, pin_memory=True, drop_last=True)
    
    val_data = BirdClefDataset(audio_image_store, meta=df.iloc[val_set].reset_index(drop=True),  
                             sr=SR, duration=DURATION, is_train=False)
    val_loader = DataLoader(val_data, batch_size=VAL_BATCH_SIZE, num_workers=VAL_NUM_WORKERS, shuffle=False)
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\n--> [EPOCH {epoch:02d}]")
        net.train()

        (l, l_val), (lrap, lrap_val), (f1, f1_val), (rec, rec_val), (prec, prec_val) = one_epoch(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader
        )

        scheduler_plateau.step(f1_val)
        scheduler_cosine.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(
            "[{epoch:02d}] loss: {loss} lrap: {lrap} f1: {f1} rec: {rec} prec: {prec} lr: {lr:.2e}".format(
                epoch=epoch,
                loss="({:.4f}, {:.4f})".format(l, l_val),
                prec="({:.3f}, {:.3f})".format(prec, prec_val),
                rec="({:.3f}, {:.3f})".format(rec, rec_val),
                f1="({:.3f}, {:.3f})".format(f1, f1_val),
                lrap="({:.3f}, {:.3f})".format(lrap, lrap_val),
                lr=current_lr
            )
        )

        if save:
            metrics = {
                "loss": l, "lrap": lrap, "f1": f1, "rec": rec, "prec": prec,
                "loss_val": l_val, "lrap_val": lrap_val, "f1_val": f1_val, "rec_val": rec_val, "prec_val": prec_val,
                "epoch": epoch, "lr": current_lr
            }
            saver.log(net, metrics)

        # Ğ Ğ°Ğ½Ğ½Ñ�Ñ� Ğ¾Ñ�Ñ‚Ğ°Ğ½Ğ¾Ğ²ĞºĞ°
        if f1_val > best_f1:
            best_f1 = f1_val
            patience_counter = 0
            print(f"ğŸ�¯ New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"â�³ No improvement: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"ğŸ›‘ Early stopping at epoch {epoch}")
            best_info = saver.get_best_model_info()
            if best_info:
                print(f"ğŸ�† Best model: epoch {best_info['epoch']}, {best_info['metric_name']}: {best_info['metric']:.4f}")
            break
    
    best_info = saver.get_best_model_info()
    if best_info:
        print(f"\nğŸ�‰ Training finished! Best model: epoch {best_info['epoch']}, {best_info['metric_name']}: {best_info['metric']:.4f}")


def one_epoch(net, criterion, optimizer, train_loader, val_loader):
    net.train()
    l, lrap, prec, rec, f1, icount = 0., 0., 0., 0., 0., 0
    train_loader_tqdm = tqdm(train_loader, leave=False)
    epoch_bar = train_loader_tqdm
    
    for (xb, yb) in epoch_bar:
        _l, _lrap, _f1, _rec, _prec = one_step(xb, yb, net, criterion, optimizer)
        l += _l
        lrap += _lrap
        f1 += _f1
        rec += _rec
        prec += _prec
        icount += 1
            
        if hasattr(epoch_bar, "set_postfix") and not icount % 10:
            epoch_bar.set_postfix(
                loss="{:.6f}".format(l/icount),
                lrap="{:.3f}".format(lrap/icount),
                prec="{:.3f}".format(prec/icount),
                rec="{:.3f}".format(rec/icount),
                f1="{:.3f}".format(f1/icount),
            )
    
    l /= icount
    lrap /= icount
    f1 /= icount
    rec /= icount
    prec /= icount
    
    l_val, lrap_val, f1_val, rec_val, prec_val = evaluate(net, criterion, val_loader)
    
    return (l, l_val), (lrap, lrap_val), (f1, f1_val), (rec, rec_val), (prec, prec_val)


def train(model_name, epochs=20, save=True, n_splits=5, seed=177, save_root=None, suffix="", folds=None):
  gc.collect()
  torch.cuda.empty_cache()

  save_root = save_root or MODEL_ROOT/f"{model_name}{suffix}"
  save_root.mkdir(exist_ok=True, parents=True)
  
  fold_bar = tqdm(df.reset_index().groupby("fold").index.apply(list).items(), total=df.fold.max()+1)
  
  for fold, val_set in fold_bar:
      if folds and not fold in folds:
        continue
      
      print(f"\n [FOLD {fold}]")
      fold_bar.set_description(f"[FOLD {fold}]")
      train_set = np.setdiff1d(df.index, val_set)
        
      one_fold(model_name, fold=fold, train_set=train_set , val_set=val_set , epochs=epochs, save=save, save_root=save_root)
    
      gc.collect()
      torch.cuda.empty_cache()


LABEL_IDS, df = get_df()
audio_image_store = load_data(df)


ds = BirdClefDataset(audio_image_store, meta=df, sr=SR, duration=DURATION, is_train=True)


x, y = ds[np.random.choice(len(ds))]


try:
    train('efficientnet', epochs=30, suffix=f"_sr{SR}_d{DURATION}_v1_v1", folds=[3])
except Exception as e:
    raise ValueError() from  e

