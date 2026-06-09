import sys
sys.path.append('../input/shopee-competition-code/main_folder/')


import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import warnings
warnings.filterwarnings("ignore")


!pip install --no-index --find-links /kaggle/input/faiss-gpu-whl faiss-gpu-cu12==1.11.0


import torch
import math
import gc
import re
import faiss
import faiss.contrib.torch_utils
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
from code_base.pipeline import SHOPEEImageDataset, SHOPEETextDataset, ImgEncoder, TextEncoder
from code_base.utils import CFG, WarmupScheduler
import lightgbm as lgb
from lightgbm import LGBMRegressor
import editdistance
import string
import cv2


compute_cv = False
img_dir = '../input/shopee-product-matching/train_images'
df = pd.read_csv('../input/shopee-product-matching/train.csv')    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG.device = device
BATCH_SIZE = 64
n_batch = 10 # load in batches for computation

img_backbone = ["timm/eca_nfnet_l1.ra2_in1k",
                "timm/dm_nfnet_f0.dm_in1k",
                "timm/swin_large_patch4_window7_224.ms_in22k_ft_in1k",
               "timm/dm_nfnet_f1.dm_in1k",
               "timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k"]

txt_backbone = ["google-bert/bert-large-uncased", 
                "FacebookAI/xlm-roberta-large", 
                "cahya/bert-base-indonesian-1.5G", 
                "indobenchmark/indobert-large-p1", 
                "google-bert/bert-base-multilingual-uncased"] # not used just for reference

img_size = 256
img_size_2 = 224
max_len = 35
num_workers = 4
num_classes = 11014
sim_thresh = 0.7
img = True
img_dim = 1792
txt_dim = 1024
k = 51 # k for knn search
di = img_dim*len(img_backbone)
dt = txt_dim*len(txt_backbone)
dc = di+dt
ngpu = torch.cuda.device_count()
min2 = False
th_lst = np.arange(80, 100, 10)
debug = False


if debug:
    df = df.head(100)


tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
df['target'] = df.label_group.map(tmp)


single_gpu = ["timm/dm_nfnet_f0.dm_in1k", "timm/dm_nfnet_f1.dm_in1k"]
permute = ["timm/swin_large_patch4_window7_224.ms_in22k_ft_in1k", 
          "timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k"]
diff_img_size = ["timm/swin_large_patch4_window7_224.ms_in22k_ft_in1k",
                "timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k"]


PATH_I0 = "../input/shopee-image-models/img_model_eca_nfnet_l1.ra2_in1k.pth"
PATH_I1 = "../input/shopee-image-models/img_model_dm_nfnet_f0.dm_in1k.pth"
PATH_I2 = "../input/shopee-image-models/img_model_swin_large_patch4_window7_224.ms_in22k_ft_in1k.pth"
PATH_I3 = "../input/shopee-image-models/img_model_dm_nfnet_f1.dm_in1k.pth"
PATH_I4 = "../input/shopee-image-models/img_model_swin_base_patch4_window7_224.ms_in22k_ft_in1k.pth"

img_ckpt = [PATH_I0, PATH_I1, PATH_I2, PATH_I3, PATH_I4]

PATH_T0 = "../input/shopee-text-models/txt_model_bert-base-uncased.pth"
PATH_T1 = "../input/shopee-text-models/txt_model_xlm-roberta-large.pth"
PATH_T2 = "../input/shopee-text-models/txt_model_bert-base-indonesian-1.5G.pth"
PATH_T3 = "../input/shopee-text-models/txt_model_indobert-large-p1.pth"
PATH_T4 = "../input/shopee-text-models/txt_model_bert-base-multilingual-uncased.pth"

txt_ckpt = [PATH_T0, PATH_T1, PATH_T2, PATH_T3, PATH_T4]

tkn_pth = ["../input/shopee-text-models-bb-offline/bert-base-uncased",
          "../input/shopee-text-models-bb-offline/xlm-roberta-large",
          "../input/shopee-text-models-bb-offline/bert-base-indonesian-1.5G",
          "../input/shopee-text-models-bb-offline/indobert-large-p1",
          "../input/shopee-text-models-bb-offline/bert-base-multilingual-uncased"]


def clean():
    gc.collect()
    torch.cuda.empty_cache()


# image and text data
def gen_img_data(backbone, size=img_size):
    if backbone in diff_img_size:
        size = img_size_2
    transforms = albumentations.Compose([
    albumentations.Resize(size, size),
    albumentations.Normalize() 
    ])
    data_img = SHOPEEImageDataset(df, img_dir, transform = transforms, gen_feat_only = True)
    dataloader_img = DataLoader(data_img, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    return dataloader_img
    
                            
def gen_text_data(tkn_pth):
    data_txt = SHOPEETextDataset(df, tokenizer = tkn_pth, gen_feat_only = True)
    dataloader_txt = DataLoader(data_txt, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    return dataloader_txt


def load_model(backbone, ckpt_path, img = False):
    if img:
        if backbone in permute:
            model = ImgEncoder(num_classes, backbone = backbone, 
                               pretrained = False, permute=True, p=4)
        else:
            model = ImgEncoder(num_classes, backbone = backbone, pretrained = False, p=4)
    else:
        model = TextEncoder(num_classes, backbone = backbone, eval_model=True)

    if ngpu > 1 and backbone not in single_gpu:
        model = nn.DataParallel(model)

    ckpt = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(ckpt)
    model = model.to(device)
    print(f"model {backbone} loaded successfully")
    return model


class gen_feas:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def gen_img_feas(self):

        self.model.eval()
        bar = tqdm.tqdm(self.dataloader)
        
        FEAS = []
        
        with torch.no_grad():
            for batch_idx, (images) in enumerate(bar):
                images = images.to(CFG.device)
                
                logits = self.model(images)
                FEAS += [logits.detach().cpu()]
                
        FEAS = torch.cat(FEAS).cpu().numpy()
        return FEAS

    def gen_txt_feas(self):
    
        self.model.eval()
        bar = tqdm.tqdm(self.dataloader)
        
        FEAS = []
        
        with torch.no_grad():
            for batch_idx, (inp_ids, att_masks) in enumerate(bar):
                inp_ids, att_masks = inp_ids.to(CFG.device), att_masks.to(CFG.device)
                
                logits = self.model(inp_ids, att_masks)
                FEAS += [logits.detach().cpu()]
                
        FEAS = torch.cat(FEAS).cpu().numpy()
        return FEAS


def return_feas(model, dataloader, img=False):
    if img:
        feas = gen_feas(model, dataloader).gen_img_feas()
    else:
        feas = gen_feas(model, dataloader).gen_txt_feas()
    feas = torch.tensor(feas).cuda()
    return feas

img_model = [load_model(backbone=img_backbone[i], ckpt_path=img_ckpt[i], img=img)
             for i in range(len(img_backbone))]

img_feas = torch.cat([return_feas(img_model[i], gen_img_data(img_backbone[i]), img=img) 
            for i in range(len(img_backbone))], dim=1)

if not compute_cv:
    del img_model
    clean()

txt_model = [load_model(backbone=tkn_pth[i], ckpt_path=txt_ckpt[i])
             for i in range(len(txt_backbone))]

txt_feas = torch.cat([return_feas(txt_model[i], gen_text_data(tkn_pth[i]))
            for i in range(len(txt_backbone))], dim=1)

if not compute_cv:
    del txt_model
    clean()


img_feas, txt_feas = F.normalize(img_feas).cuda(1), F.normalize(txt_feas).cuda(1)

torch.save(img_feas, 'img_feas.pt')
torch.save(txt_feas, 'txt_feas.pt')

# comb_feas = F.normalize(torch.cat([img_feas, txt_feas], dim=1)).cuda(1)
# torch.save(comb_feas, 'comb_feas.pt')





# def build_faiss(feas, dim):
#     res = faiss.StandardGpuResources()
#     index = faiss.GpuIndexFlatIP(res, dim)
#     index.add(feas)
#     return index


# def get_batches(bs, n_batch, feas):
#     batches = []
#     for i in range(n_batch):
#         left = bs * i
#         right = bs * (i+1)
#         if i == n_batch - 1:
#             right = feas.shape[0]
#         batches.append(feas[left:right,:])
#     return batches


# def get_matches(bs, n_batch, feas, dim, k):
#     index = build_faiss(feas, dim)
#     m=[]
#     s=[]
#     for batch in tqdm.tqdm(get_batches(bs, n_batch, feas)):
#         batch = batch.cuda() 
#         sims, matches = index.search(batch, k)
#         m.append(matches)
#         s.append(sims)
#     m = torch.cat(m, dim=0)
#     s = torch.cat(s, dim=0)
#     return m,s


# #index IP
# bs = comb_feas.shape[0] // n_batch
# img_matches, img_sims = get_matches(bs, n_batch, img_feas, di, k)    
# text_matches, text_sims = get_matches(bs, n_batch, txt_feas, dt, k)    
# comb_matches, comb_sims = get_matches(bs, n_batch, comb_feas, dc, k)

# if not compute_cv:
#     del img_feas, txt_feas, comb_feas
#     clean()


# def create_lgb_df(df, matches, sim):

#     matches = matches.detach().cpu().numpy()
#     sim = sim.detach().cpu().numpy()
    
#     rows=[]
#     titles = [
#         title.translate(str.maketrans({_: ' ' for _ in string.punctuation}))
#         for title in df['title'].str.lower().values
#     ]
#     # row_id = df['posting_id'].values
#     row_titles = df['title'].values
#     row_phash = df['image_phash'].values
#     label_group = df['target'].values
#     st_sizes = [os.stat(f'{img_dir}/{image}').st_size for image in df['image'].values]
#     image_dim = [cv2.imread(f'{img_dir}/{image}').shape[:2] for image in df['image'].values]
    
#     mean_sim_top5 = sim[:, :5].mean(1)
#     mean_mean_sim_top5 = mean_sim_top5[matches[:, :5]].mean(1)
#     mean_sim_top5 = (mean_sim_top5 - mean_sim_top5.mean()) / mean_sim_top5.std()
    
#     mean_sim_top15 = sim[:, :15].mean(1)
#     mean_mean_sim_top15 = mean_sim_top15[matches[:, :15]].mean(1)
#     mean_sim_top15 = (mean_sim_top15 - mean_sim_top15.mean()) / mean_sim_top15.std()
    
#     mean_sim_top30 = sim[:, :30].mean(1)
#     mean_mean_sim_top30 = mean_sim_top30[matches[:, :30]].mean(1)
#     mean_sim_top30 = (mean_sim_top30 - mean_sim_top30.mean()) / mean_sim_top30.std()

    
#     for i in tqdm.tqdm(range(len(df))):
#         match_dict = dict(zip(matches[i], sim[i]))
#         for j in match_dict:
#             if i == j:
#                 continue
#             rows.append({
#                 'i' : np.int32(i),
#                 'j' : np.int32(j),
#                 'sim' : match_dict[j],
#                 'edit_distance_title': np.int32(editdistance.eval(titles[i], titles[j])),
#                 'title_len': np.int32(len(row_titles[i])),
#                 'title_len_target': np.int32(len(row_titles[j])),
#                 'title_num_words': np.int32(len(row_titles[i].split())),
#                 'title_num_words_target': np.int32(len(row_titles[j].split())),
#                 'edit_distance_phash': np.int32(editdistance.eval(row_phash[i], row_phash[j])),
#                 'mean_sim_top5': mean_sim_top5[i],
#                 'mean_sim_target_top5': mean_sim_top5[j],
#                 'mean_mean_sim_top5': mean_mean_sim_top5[i],
#                 'mean_mean_sim_target_top5': mean_mean_sim_top5[j],
#                 'mean_sim_top15': mean_sim_top15[i],
#                 'mean_sim_target_top15': mean_sim_top15[j],
#                 'mean_mean_sim_top15': mean_mean_sim_top15[i],
#                 'mean_mean_sim_target_top15': mean_mean_sim_top15[j],            
#                 'mean_sim_top30': mean_sim_top30[i],
#                 'mean_sim_target_top30': mean_sim_top30[j],
#                 'mean_mean_sim_top30': mean_mean_sim_top30[i],
#                 'mean_mean_sim_target_top30': mean_mean_sim_top30[j],
#                 'st_size': np.int32(st_sizes[i]),
#                 'target_st_size': np.int32(st_sizes[j]),
#                 'wxh/st_size': np.float32(image_dim[i][1] * image_dim[i][0] / st_sizes[i]),
#                 'wxh/st_size_target': np.float32(image_dim[j][1] * image_dim[j][0] / st_sizes[j]),
#                 'y' : np.int8(1 if row_id[j] in label_group[i] else 0)
#             })
#     lgb_df = pd.DataFrame(rows)
#     return lgb_df     


# df_new = create_lgb_df(df, img_matches, img_sims)


# features = df_new.drop('y', axis=1)
# target = df['y']
# X_train, X_val, Y_train, Y_val = train_test_split(features, target,
#                                                   random_state=2023,
#                                                   test_size=0.25)
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_val = scaler.transform(X_val)

# train_data = lgb.Dataset(X_train, label=Y_train)
# test_data = lgb.Dataset(X_val, label=Y_val, reference=train_data)

# params = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'boosting_type': 'gbdt',
#     'num_leaves': 31,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
# num_round = 100
# model = lgb.train(params, train_data, num_round, valid_sets=[
#                 test_data], early_stopping_rounds=10)













