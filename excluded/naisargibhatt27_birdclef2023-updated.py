import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection
import torchvision.transforms as transforms
import torchvision.io 
import librosa
from PIL import Image
import albumentations as alb
import torch.multiprocessing as mp
import warnings

warnings.filterwarnings('ignore')


from pytorch_lightning.callbacks import ModelCheckpoint, BackboneFinetuning, EarlyStopping



!pip install -q torchtoolbox timm


class Config:
    use_aug = False
    num_classes = 264
    batch_size = 64
    epochs = 20
    PRECISION = 16    
    PATIENCE = 8    
    seed = 64
    model = "tf_efficientnet_b1_ns"
    pretrained = True            
    weight_decay = 1e-3
    use_mixup = True
    mixup_alpha = 0.6
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    data_root = "/kaggle/input/birdclef-2023/"
    train_images = "/kaggle/input/split-creating-melspecs-stage-1/specs/train/"
    valid_images = "/kaggle/input/split-creating-melspecs-stage-1/specs/valid/"
    train_path = "/kaggle/input/bc2023-train-val-df/train.csv"
    valid_path = "/kaggle/input/bc2023-train-val-df/valid.csv"
    
    
    SR = 32000
    DURATION = 5
    MAX_READ_SAMPLES = 5
    LR = 10e-4
    


pl.seed_everything(Config.seed, workers=True)


def config_to_dict(cfg):
    return dict((name, getattr(cfg, name)) for name in dir(cfg) if not name.startswith('__'))


df_train = pd.read_csv(Config.train_path)
df_valid = pd.read_csv(Config.valid_path)
df_train.head()


Config.num_classes = len(df_train.primary_label.unique())


df_train = pd.concat([df_train, pd.get_dummies(df_train['primary_label'])], axis=1)
df_valid = pd.concat([df_valid, pd.get_dummies(df_valid['primary_label'])], axis=1)


birds = list(df_train.primary_label.unique())


missing_birds = list(set(list(df_train.primary_label.unique())).difference(list(df_valid.primary_label.unique())))


non_missing_birds = list(set(list(df_train.primary_label.unique())).difference(missing_birds))


len(non_missing_birds)


df_valid[missing_birds] = 0
df_valid = df_valid[df_train.columns] ## Fix order


# df_train.iloc[:,17:]


import albumentations as A
from torchtoolbox.transform import Cutout
def get_train_transform():
    return A.Compose([
        Cutout()
    ])


class BirdDataset(torch.utils.data.Dataset):

    def __init__(self, df, sr = Config.SR, duration = Config.DURATION, augmentations = None, train = True):

        self.df = df
        self.sr = sr 
        self.train = train
        self.duration = duration
        self.augmentations = augmentations
        if train:
            self.img_dir = Config.train_images
        else:
            self.img_dir = Config.valid_images

    def __len__(self):
        return len(self.df)

    @staticmethod
    def normalize(image):
        image = image / 255.0
        #image = torch.stack([image, image, image])
        return image

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        impath = self.img_dir + f"{row.filename}.npy"

        image = np.load(str(impath))[:Config.MAX_READ_SAMPLES]
        
        ########## RANDOM SAMPLING ################
        if self.train:
            image = image[np.random.choice(len(image))]
        else:
            image = image[0]
            
        #####################################################################
        
        image = torch.tensor(image).float()

        if self.augmentations:
            image = self.augmentations(image.unsqueeze(0)).squeeze()
            
        image.size()
        
        image = torch.stack([image])

        image = self.normalize(image)


        return image, torch.tensor(row[17:]).float()



def get_fold_dls(df_train, df_valid, aug=None):

    ds_train = BirdDataset(
        df_train, 
        sr = Config.SR,
        duration = Config.DURATION,
        augmentations = aug,
        train = True
    )
    ds_val = BirdDataset(
        df_valid, 
        sr = Config.SR,
        duration = Config.DURATION,
        augmentations = None,
        train = False
    )
    dl_train = DataLoader(ds_train, batch_size=Config.batch_size , shuffle=True, num_workers = 0)    
    dl_val = DataLoader(ds_val, batch_size=Config.batch_size, num_workers = 0)
    return dl_train, dl_val, ds_train, ds_val


def show_batch(img_ds, num_items, num_rows, num_cols, predict_arr=None):
    fig = plt.figure(figsize=(12, 6))    
    img_index = np.random.randint(0, len(img_ds)-1, num_items)
    for index, img_index in enumerate(img_index):  # list first 9 images
        img, lb = img_ds[img_index]        
        ax = fig.add_subplot(num_rows, num_cols, index + 1, xticks=[], yticks=[])
        if isinstance(img, torch.Tensor):
            img = img.detach().numpy()
        if isinstance(img, np.ndarray):
            img = img.transpose(1, 2, 0)
            ax.imshow(img)        
            
        title = f"Spec"
        ax.set_title(title)  


aug = get_train_transform()
dl_train, dl_val, ds_train, ds_val = get_fold_dls(df_train, df_valid)
show_batch(ds_val, 8, 2, 4)


from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR

def get_optimizer(lr, params):
    model_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, params), 
            lr=lr,
            weight_decay=Config.weight_decay
        )
    interval = "epoch"
    
    lr_scheduler = CosineAnnealingWarmRestarts(
                            model_optimizer, 
                            T_0=Config.epochs, 
                            T_mult=1, 
                            eta_min=1e-6, 
                            last_epoch=-1
                        )

    return {
        "optimizer": model_optimizer, 
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "interval": interval,
            "monitor": "val_loss",
            "frequency": 1
        }
    }


from torchtoolbox.tools import mixup_data, mixup_criterion
import torch.nn as nn
from torch.nn.functional import cross_entropy
import torchmetrics
import timm


import sklearn.metrics

def padded_cmap(solution, submission, padding_factor=5):
    solution = solution#.drop(['row_id'], axis=1, errors='ignore')
    submission = submission#.drop(['row_id'], axis=1, errors='ignore')
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = sklearn.metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro',
    )
    return score

def map_score(solution, submission):
    solution = solution#.drop(['row_id'], axis=1, errors='ignore')
    submission = submission#.drop(['row_id'], axis=1, errors='ignore')
    score = sklearn.metrics.average_precision_score(
        solution.values,
        submission.values,
        average='micro',
    )
    return score


dummy = df_valid[birds].copy()
dummy[birds] = np.random.rand(dummy.shape[0],dummy.shape[1])


padded_cmap(df_valid[birds], dummy[birds], padding_factor = 5)


padded_cmap(df_valid[birds], dummy[birds], padding_factor = 1)


map_score(df_valid[birds], dummy[birds])


class BirdClefModel(pl.LightningModule):
    def __init__(self, model_name=Config.model, num_classes = Config.num_classes, pretrained = Config.pretrained):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=1)
        
#         self.train_losses = []
#         self.val_losses = []
        self.train_acc_history = []
        self.val_acc_history = []

        if 'res' in model_name:
            self.in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(self.in_features, num_classes)
        elif 'dense' in model_name:
            self.in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(self.in_features, num_classes)
        elif 'efficientnet' in model_name:
            self.in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(self.in_features, num_classes)
            )
        
        self.loss_function = nn.BCEWithLogitsLoss() 

    def forward(self,images):
        logits = self.backbone(images)
        return logits
        
    def configure_optimizers(self):
        return get_optimizer(lr=Config.LR, params=self.parameters())

    def train_with_mixup(self, X, y):
        X, y_a, y_b, lam = mixup_data(X, y, alpha=Config.mixup_alpha)
        y_pred = self(X)
        loss_mixup = mixup_criterion(cross_entropy, y_pred, y_a, y_b, lam)
        return loss_mixup

    def training_step(self, batch, batch_idx):
        image, target = batch    
        y_pred = self(image)

        if Config.use_mixup:
            loss = self.train_with_mixup(image, target)
        else:
            y_pred = self(image)
            loss = self.loss_function(y_pred,target)
        
#         self.train_losses.append(loss.item())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate training accuracy
        y_pred_class = (y_pred > 0.5).float()
        train_acc = (y_pred_class == target).float().mean()
        self.train_acc_history.append(train_acc.item())
        self.log("train_acc", train_acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss        

    def validation_step(self, batch, batch_idx):
        image, target = batch     
        y_pred = self(image)
        val_loss = self.loss_function(y_pred, target)
#         self.val_losses.append(val_loss.item())
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        
        # Calculate validation accuracy
        y_pred_class = (y_pred > 0.5).float()
        val_acc = (y_pred_class == target).float().mean()
        self.val_acc_history.append(val_acc.item())
        self.log("val_acc", val_acc, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        return {"val_loss": val_loss, "logits": y_pred, "targets": target}
    
    def train_dataloader(self):
        return self._train_dataloader 
    
    def validation_dataloader(self):
        return self._validation_dataloader
    
  #     def plot_species_accuracy(self, true, pred):
#         species_acc = {}
#         for i, species in enumerate(birds):
#             species_acc[species] = (true[:, i] == pred[:, i]).mean()

#         plt.figure(figsize=(12, 8))
#         plt.bar(list(species_acc.keys()), list(species_acc.values()))
#         plt.xticks(rotation=90)
#         plt.xlabel('Species')
#         plt.ylabel('Accuracy')
#         plt.title('Accuracy per Species')
#         plt.savefig('species_accuracy.png')
#         plt.close()
    
        
    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        output_val = torch.cat([x['logits'] for x in outputs],dim=0).sigmoid().cpu().detach().numpy()
        target_val = torch.cat([x['targets'] for x in outputs],dim=0).cpu().detach().numpy()
        
        # print(output_val.shape)
        val_df = pd.DataFrame(target_val, columns = birds)
        pred_df = pd.DataFrame(output_val, columns = birds)
        
        avg_score = padded_cmap(val_df, pred_df, padding_factor = 5)
        avg_score2 = padded_cmap(val_df, pred_df, padding_factor = 3)
        avg_score3 = sklearn.metrics.label_ranking_average_precision_score(target_val,output_val)
            
#         competition_metrics(output_val,target_val)
        print(f'epoch {self.current_epoch} validation loss {avg_loss}')
        print(f'epoch {self.current_epoch} validation C-MAP score pad 5 {avg_score}')
        print(f'epoch {self.current_epoch} validation C-MAP score pad 3 {avg_score2}')
        print(f'epoch {self.current_epoch} validation AP score {avg_score3}')
        
        
        val_df.to_pickle('val_df.pkl')
        pred_df.to_pickle('pred_df.pkl')
        
#         self.plot_species_accuracy(target_val, output_val)
#         self. plot_species_cmap(target_val, output_val)
#         self.plot_high_species_cmap(target_val, output_val)
    
        return {'val_loss': avg_loss,'val_cmap':avg_score}
    
    def on_train_end(self):
#         self.plot_train_loss()
#         self.plot_val_loss()
        self.plot_train_val_accuracy()
#         self.plot_loss_curves()

    def plot_train_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses[:20], label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig('train_loss_plot.png')
        plt.close()

    def plot_val_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.val_losses[:20], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.savefig('val_loss_plot.png')
        plt.close()
        
        
    def plot_train_val_accuracy(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_acc_history[:20], label='Training Accuracy')
        plt.plot(self.val_acc_history[:20], label='Validation Accuracy')
        plt.xticks([0, 5, 10, 15, 20])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.ylim(0, 1) 
        plt.legend()
        plt.savefig('accuracy_plot.png')
        plt.close()
    

    def plot_loss_curves(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig('loss_plot.png')
        plt.close() 
        
    def plot_high_species_cmap(self, true, pred):
        species_cmap = {}
        val_df = pd.DataFrame(true, columns=birds)
        pred_df = pd.DataFrame(pred, columns=birds)

        for species in birds:
            cmap_score = padded_cmap(val_df[[species]], pred_df[[species]], padding_factor=5)
            if cmap_score > 0.8:
                species_cmap[species] = cmap_score

        if species_cmap:
            plt.figure(figsize=(12, 8))
            plt.bar(list(species_cmap.keys()), list(species_cmap.values()))
            plt.xticks(rotation=90)
            plt.xlabel('Species')
            plt.ylabel('cMAP')
            plt.title('cMAP per Species (cMAP > 0.8)')
            plt.savefig('species_cmap_gt_0.8.png')
            plt.close()
    
    def plot_species_cmap(self, true, pred):
        species_cmap = {}
        val_df = pd.DataFrame(true, columns=birds)
        pred_df = pd.DataFrame(pred, columns=birds)

        for species in birds:
            species_cmap[species] = padded_cmap(val_df[[species]], pred_df[[species]], padding_factor=5)

        plt.figure(figsize=(12, 8))
        plt.bar(list(species_cmap.keys()), list(species_cmap.values()))
        plt.xticks(rotation=90)
        plt.xlabel('Species')
        plt.ylabel('cMAP')
        plt.title('cMAP per Species')
        plt.savefig('species_cmap.png')
        plt.close()


from pytorch_lightning.loggers import WandbLogger
import gc

def run_training():
    print(f"Running training...")
    logger = None
    
    
    dl_train, dl_val, ds_train, ds_val = get_fold_dls(df_train, df_valid)
    
    audio_model = BirdClefModel()

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=Config.PATIENCE, verbose= True, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath= "/kaggle/working/exp1/",
                                      save_top_k=1,
                                      save_last= True,
                                      save_weights_only=True,
                                      filename= f'./{Config.model}_loss',
                                      verbose= True,
                                      mode='min')
    
    callbacks_to_use = [checkpoint_callback,early_stop_callback]


    trainer = pl.Trainer(
        gpus=1,
        val_check_interval=0.5,
        deterministic=True,
        max_epochs=Config.epochs,
        logger=logger,
        auto_lr_find=False,    
        callbacks=callbacks_to_use,
        precision=Config.PRECISION, accelerator="gpu" 
    )

    print("Running trainer.fit")
    trainer.fit(audio_model, train_dataloaders = dl_train, val_dataloaders = dl_val)                

    gc.collect()
    torch.cuda.empty_cache()



run_training()


pred = pd.read_pickle('/kaggle/working/pred_df.pkl')
true = pd.read_pickle('/kaggle/working/val_df.pkl')


true.sum(axis=1)


pred.sum(axis=1)


padded_cmap(true, pred, padding_factor = 5)


import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(true, pred, labels):
    # Check the shapes of true and pred
    print(f"true shape: {true.shape}")
    print(f"pred shape: {pred.shape}")
    print(f"labels length: {len(labels)}")

    # Convert true and pred to the same format
    if true.shape[1] == len(labels):
        true_labels = true.idxmax(axis=1)
    else:
        true_labels = np.argmax(true.values, axis=1)

    if pred.shape[1] == len(labels):
        pred_labels = pred.idxmax(axis=1)
    else:
        pred_labels = np.argmax(pred.values, axis=1)

    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    # Plot the confusion matrix
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=labels,
        y=labels,
        colorscale='RdBu'
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=800,
        height=800
    )

    fig.show()


plot_confusion_matrix(true, pred, birds)


# --- Convert Lightning .ckpt to plain PyTorch .pt ---
import os
from pathlib import Path
import torch

# 1) Choose checkpoint path
ckpt_dir = Path("/kaggle/working/exp1")
best_ckpt = ckpt_dir / "tf_efficientnet_b1_ns_loss.ckpt"
last_ckpt = ckpt_dir / "last.ckpt"
ckpt_path = best_ckpt if best_ckpt.exists() else last_ckpt
print("Using checkpoint:", ckpt_path)

# 2) Recreate your LightningModule class exactly as in training (already defined above)
#    BirdClefModel uses timm EfficientNet with in_chans=1 and replaces .classifier.
#    Config.model should be "tf_efficientnet_b1_ns" (or whatever you trained with).
lit_model = BirdClefModel(model_name=Config.model,
                          num_classes=Config.num_classes,
                          pretrained=False)

# 3) Load weights from Lightning checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu")
lit_model.load_state_dict(ckpt["state_dict"], strict=True)

# 4) Save only the *backbone* (plain nn.Module) as .pt for deployment
pt_out = "/kaggle/working/birdclef_model.pt"
torch.save(lit_model.backbone.state_dict(), pt_out)
print("Saved:", pt_out)

# 5) (Optional) also ensure labels.json is next to it for API
# If you already created labels.json earlier, just copy it:
src_labels = Path("/kaggle/working/labels.json")
if src_labels.exists():
    print("labels.json already present:", src_labels)
else:
    # Create from your training dataframe if needed:
    import pandas as pd, json
    df = pd.read_csv("/kaggle/working/train.csv")  # adjust if you stored metadata elsewhere
    labels = sorted(df["primary_label"].unique())
    id2label = {i: lab for i, lab in enumerate(labels)}
    with open("/kaggle/working/labels.json", "w") as f:
        json.dump(id2label, f)
    print("Wrote labels.json at /kaggle/working/labels.json")





