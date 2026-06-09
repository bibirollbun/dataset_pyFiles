import sys
sys.path.append('/kaggle/input/shopee-competition-code/main_folder/')


import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


import torch
import math
import torch.nn as nn
import pandas as pd 
import numpy as np
import albumentations
import torch.optim as optim
import tqdm.notebook as tqdm
from torch.nn import Parameter
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import f1_score, accuracy_score
from code_base.pipeline import SHOPEEImageDataset, SHOPEETextDataset, ImgEncoder, TextEncoder
from code_base.utils import CFG, WarmupScheduler


img_dir = '/kaggle/input/shopee-product-matching/train_images'
df = pd.read_csv('/kaggle/input/shopee-product-matching/train.csv')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG.device = device
EPOCHS = 40
fold_id = 0
LR = 3e-5
debug = False
BATCH_SIZE = 64
img_backbone = "timm/eca_nfnet_l1.ra2_in1k"
text_backbone = "FacebookAI/xlm-roberta-base"
img_size = 256
search_space = np.arange(40, 100, 10)
# valid_img_size = 512


le = LabelEncoder()
df.label_group = le.fit_transform(df.label_group)


gkf = GroupKFold(n_splits=5)
df['fold'] = -1
for fold, (train_idx, valid_idx) in enumerate(gkf.split(df, None, df.label_group)):
    df.loc[valid_idx, 'fold'] = fold

df_train = df[df['fold'] != fold_id]
df_train = df_train.reset_index(drop=True)
df_valid = df[df['fold'] == fold_id]
df_valid = df_valid.reset_index(drop=True)

df_valid['count'] = df_valid.label_group.map(df_valid.label_group.value_counts().to_dict())


transforms = albumentations.Compose([
    albumentations.Resize(img_size, img_size),
    albumentations.Normalize() 
])


def row_wise_f1_score(labels, preds):
    scores = []
    for label, pred in zip(labels, preds):
        n = len(np.intersect1d(label, pred))
        score = 2 * n / (len(label)+len(pred))
        scores.append(score)
    return scores, np.mean(scores)


# img model train and validation data
train_data_img = SHOPEEImageDataset(df_train, img_dir, transform = transforms)
train_dataloader_img = DataLoader(train_data_img, batch_size=BATCH_SIZE, shuffle=True)

valid_data_img = SHOPEEImageDataset(df_valid, img_dir, transform = transforms, gen_feat_only = True)
valid_dataloader_img = DataLoader(valid_data_img, batch_size=BATCH_SIZE, shuffle=False)


img_model = ImgEncoder(df.label_group.nunique(), 
                       backbone = img_backbone, 
                       embed_size = 1792,
                       scale = 10.0,
                       margin = 0.5)
if torch.cuda.device_count() > 1:
    img_model = nn.DataParallel(img_model)
    print("Multiple GPU detected")
img_model = img_model.to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(img_model.parameters(), lr = LR)
scheduler = WarmupScheduler(optimizer, warmup_epochs=20, plateau_lr=LR)


# training loop

def train_func(dataloader):
    
    img_model.train()
    bar = tqdm.tqdm(dataloader)
    losses = []
    for batch_idx, (images, targets) in enumerate(bar):
        images, targets = images.to(CFG.device), targets.to(CFG.device).long()

        if debug and batch_idx == 100:
            print('Debug Mode. Only train on first 100 batches.')
            break

        optimizer.zero_grad()
        
        logits = img_model(images, targets)
        loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()
        
        
        print(
            f"loss : {loss.item():.4f} ",
            end="\r",
            flush=True,
        )
        losses.append(loss.item())
    
    net_loss = np.mean(losses)
    
    return net_loss


def find_threshold(df, lower_count_thresh, upper_count_thresh, search_space):
    '''
    Compute the optimal threshold for the given count threshold.
    '''
    score_by_threshold = []
    best_score = 0
    best_threshold = -1
    for i in tqdm.tqdm(search_space):
        sim_thresh = i/100
        selection = ((FEAS@FEAS.T) > sim_thresh).cpu().numpy()
        matches = []
        oof = []
        for row in selection:
            oof.append(df.iloc[row].posting_id.tolist())
            matches.append(' '.join(df.iloc[row].posting_id.tolist()))
        tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
        df['target'] = df.label_group.map(tmp)
        scores, score = row_wise_f1_score(df.target, oof)
        df['score'] = scores
        df['oof'] = oof
        
        selected_score = df.query(f'count > {lower_count_thresh} and count < {upper_count_thresh}').score.mean()
        score_by_threshold.append(selected_score)
        if selected_score > best_score:
            best_score = selected_score
            best_threshold = i
            
    print(f'Best score is {best_score} and best threshold is {best_threshold/100}')


def gen_feas(dataloader):

    img_model.eval()
    bar = tqdm.tqdm(dataloader)
    
    FEAS = []
    
    with torch.no_grad():
        for batch_idx, (images) in enumerate(bar):
            images = images.to(CFG.device)
            
            logits = img_model(images)
            FEAS += [logits.detach().cpu()]
            
    FEAS = torch.cat(FEAS).cpu().numpy()
    
    return FEAS


# train img model
for epoch in range(EPOCHS):
    
    print(f'epoch no : {epoch+1}')
   
    tl = train_func(train_dataloader_img)
    scheduler.step()

    print(f"Net loss : {tl:.4f}")

    if((epoch+1)% 5 == 0):
        FEAS = gen_feas(valid_dataloader_img)
        FEAS = torch.tensor(FEAS).cuda()
    
        find_threshold(df = df_valid, 
           lower_count_thresh = 0, 
           upper_count_thresh = 999,
           search_space = search_space)


#save img model
torch.save(img_model.state_dict(), f"img_model_{img_backbone.split('/')[-1]}.pth")


# oof CV
# net_loss, net_score = valid_img_model(valid_dataloader_img, img_model, loss_fn)
# print(f'net_loss : {net_loss : .4f}  net_score : {net_score : .4f}')


# training_data_text = SHOPEETextDataset(train_df, tokenizer = text_backbone)
# text_train_dataloader = DataLoader(training_data_text, batch_size=32, shuffle=True)


# text_model = TextEncoder(train_df.label_group.nunique(), 
#                    backbone = text_backbone)
# text_model = nn.DataParallel(text_model)
# _ = text_model.to(device)


# train_text_model(text_train_dataloader, n_epochs, text_model, loss_fn, optimizer)







