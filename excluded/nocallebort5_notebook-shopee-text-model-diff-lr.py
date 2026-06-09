import sys
sys.path.append('/kaggle/input/shopee-competition-code/main_folder/')


import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


!pip install unidecode==1.4.0


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
EPOCHS = 35
fold_id = 0
LR = [3e-5, 1e-4] #[bb_lr, neck_lr]
validate = False
debug = False
BATCH_SIZE = 64
text_backbone = ["google-bert/bert-base-uncased", 
                "FacebookAI/xlm-roberta-large", 
                "cahya/bert-base-indonesian-1.5G", 
                "indobenchmark/indobert-large-p1", 
                "google-bert/bert-base-multilingual-uncased",
                "FacebookAI/xlm-roberta-base"] # not used just for reference
max_len = 35
NUM = 3
search_space = np.arange(40, 100, 10)
clean=True


le = LabelEncoder()
df.label_group = le.fit_transform(df.label_group)


if validate:
    gkf = GroupKFold(n_splits=5)
    df['fold'] = -1
    for fold, (train_idx, valid_idx) in enumerate(gkf.split(df, None, df.label_group)):
        df.loc[valid_idx, 'fold'] = fold
    
    df_train = df[df['fold'] != fold_id]
    df_train = df_train.reset_index(drop=True)
    df_valid = df[df['fold'] == fold_id]
    df_valid = df_valid.reset_index(drop=True)
    
    df_valid['count'] = df_valid.label_group.map(df_valid.label_group.value_counts().to_dict())


def row_wise_f1_score(labels, preds):
    scores = []
    for label, pred in zip(labels, preds):
        n = len(np.intersect1d(label, pred))
        score = 2 * n / (len(label)+len(pred))
        scores.append(score)
    return scores, np.mean(scores)


# txt model train and validation data
if validate:
    train_data_txt = SHOPEETextDataset(df_train, tokenizer = text_backbone[NUM])
    train_dataloader_txt = DataLoader(train_data_txt, batch_size=BATCH_SIZE, shuffle=True)
    
    valid_data_txt = SHOPEETextDataset(df_valid, tokenizer = text_backbone[NUM], gen_feat_only = True)
    valid_dataloader_txt = DataLoader(valid_data_txt, batch_size=BATCH_SIZE, shuffle=False)
else:
    train_data_txt = SHOPEETextDataset(df, tokenizer = text_backbone[NUM], clean = clean)
    train_dataloader_txt = DataLoader(train_data_txt, batch_size=BATCH_SIZE, shuffle=True)


txt_model = TextEncoder(df.label_group.nunique(), 
                       backbone = text_backbone[NUM], 
                       embed_size = 1024,
                       scale = 32.0,
                       margin = 0.5,
                       device=device)
if torch.cuda.device_count() > 1:
    txt_model = nn.DataParallel(txt_model)
    print("Multiple GPU detected")
text_model = txt_model.to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': txt_model.module.backbone.parameters(), 'lr': LR[0]},
    {'params': txt_model.module.fc.parameters(), 'lr': LR[1]},
    {'params': txt_model.module.final.parameters(), 'lr': LR[1]},
])
scheduler = WarmupScheduler(optimizer, warmup_epochs=20, 
                            plateau_lr_bb = LR[0], plateau_lr_neck = LR[1])


# training loop

def train_func(dataloader):
    
    txt_model.train()
    bar = tqdm.tqdm(dataloader)
    losses = []
    for batch_idx, (inp_ids, att_masks, targets) in enumerate(bar):
        inp_ids, att_masks, targets = inp_ids.to(CFG.device), att_masks.to(CFG.device), targets.to(CFG.device).long()

        if debug and batch_idx == 100:
            print('Debug Mode. Only train on first 100 batches.')
            break

        optimizer.zero_grad()
        
        logits = txt_model(inp_ids, att_masks, targets)
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

    txt_model.eval()
    bar = tqdm.tqdm(dataloader)
    
    FEAS = []
    
    with torch.no_grad():
        for batch_idx, (inp_ids, att_masks) in enumerate(bar):
            inp_ids, att_masks = inp_ids.to(CFG.device), att_masks.to(CFG.device)
            
            logits = txt_model(inp_ids, att_masks)
            FEAS += [logits.detach().cpu()]
            
    FEAS = torch.cat(FEAS).cpu().numpy()
    
    return FEAS


# train txt model
for epoch in range(EPOCHS):
    
    print(f'epoch no : {epoch+1}')
   
    tl = train_func(train_dataloader_txt)
    scheduler.step()

    print(f"Net loss : {tl:.4f}")

    # if validate:
    #     FEAS = gen_feas(valid_dataloader_txt)
    #     FEAS = torch.tensor(FEAS).to(device) 
    
    #     find_threshold(df = df_valid, 
    #        lower_count_thresh = 0, 
    #        upper_count_thresh = 999,
    #        search_space = search_space)


#save img model
torch.save(txt_model.state_dict(), f"txt_model_{text_backbone[NUM].split('/')[-1]}.pth")


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







