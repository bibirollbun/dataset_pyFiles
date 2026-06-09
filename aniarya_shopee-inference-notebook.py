import sys
sys.path.append('../input/shopee-competition-code/main_folder/')


import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import warnings
warnings.filterwarnings("ignore")


!pip install --no-index --find-links ../input/unidecode-whl unidecode==1.4.0
!pip install --no-index --find-links ../input/faiss-gpu-whl faiss-gpu-cu12==1.11.0


import torch
import math
import gc
import regex
import faiss
from functools import reduce
import faiss.contrib.torch_utils
import torch.nn as nn
import pandas as pd 
import numpy as np
import albumentations
import torch.optim as optim
import tqdm.notebook as tqdm
from torch.nn import Parameter
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from code_base.pipeline import SHOPEEImageDataset, SHOPEETextDataset, ImgEncoder, TextEncoder
from code_base.utils import CFG, WarmupScheduler, clean_text


compute_cv = False
if compute_cv:
    img_dir = '../input/shopee-product-matching/train_images'
    df = pd.read_csv('../input/shopee-product-matching/train.csv')
else:
    img_dir = '../input/shopee-product-matching/test_images'
    df = pd.read_csv('../input/shopee-product-matching/test.csv')
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG.device = device
BATCH_SIZE = 64
n_batch = 10 # load in batches for computation

img_backbone = ["timm/eca_nfnet_l1.ra2_in1k", 
                "timm/dm_nfnet_f0.dm_in1k",
                "timm/swin_large_patch4_window7_224.ms_in22k_ft_in1k",
               "timm/dm_nfnet_f1.dm_in1k",
               "timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k"]

txt_backbone = ["google-bert/bert-base-uncased",
                "FacebookAI/xlm-roberta-large",
                "cahya/bert-base-indonesian-1.5G", 
                "indobenchmark/indobert-large-p1", 
                "google-bert/bert-base-multilingual-uncased",
                "FacebookAI/xlm-roberta-base"] # not used just for reference

img_size = 256
img_size_2 = 224
img_size_3 = 384
max_len = 35
num_workers = 4
num_classes = 11014
img = True
img_dim = 1792
txt_dim = 1024
K = 51 # k for knn search
di = img_dim*len(img_backbone)
dt = txt_dim*len(txt_backbone)
dc = di+dt
bs = len(df) // n_batch
ngpu = torch.cuda.device_count()
min2 = False
th_lst = np.arange(80, 100, 10)


single_gpu = ["timm/dm_nfnet_f0.dm_in1k", "timm/dm_nfnet_f1.dm_in1k"]

permute = ["timm/swin_large_patch4_window7_224.ms_in22k_ft_in1k", 
          "timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k",
          "timm/swin_base_patch4_window12_384.ms_in1k"]

diff_img_size = ["timm/swin_large_patch4_window7_224.ms_in22k_ft_in1k",
                "timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k"]

diff_img_size_2 = ["timm/swin_base_patch4_window12_384.ms_in1k", 
                  "timm/swin_base_patch4_window12_384.ms_in22k_ft_in1k"]

cleaned = []


PATH_I0 = "../input/shopee-image-models/img_model_eca_nfnet_l1.ra2_in1k.pth"
PATH_I1 = "../input/shopee-image-models/img_model_dm_nfnet_f0.dm_in1k.pth"
PATH_I2 = "../input/shopee-image-models/img_model_swin_large_patch4_window7_224.ms_in22k_ft_in1k.pth"
PATH_I3 = "../input/shopee-image-models/img_model_dm_nfnet_f1.dm_in1k.pth"
PATH_I4 = "../input/shopee-image-models/img_model_swin_base_patch4_window7_224.ms_in22k_ft_in1k.pth" 

img_ckpt = [PATH_I0, PATH_I1, PATH_I2, PATH_I3, PATH_I4]

PATH_T0 = "../input/shopee-text-models/txt_model_bert-base-uncased_35.pth"
PATH_T1 = "../input/shopee-text-models/txt_model_xlm-roberta-large_35.pth"
PATH_T2 = "../input/shopee-text-models/txt_model_bert-base-indonesian-1.5G_35.pth"
PATH_T3 = "../input/shopee-text-models/txt_model_indobert-large-p1_35.pth"
PATH_T4 = "../input/shopee-text-models/txt_model_bert-base-multilingual-uncased_35.pth"
path_t5 = "../input/shopee-text-models/txt_model_xlm-roberta-base_35.pth"

txt_ckpt = [PATH_T0, PATH_T1, PATH_T2, PATH_T3, PATH_T4, path_t5]

tkn_pth = ["../input/shopee-text-models-bb-offline/bert-base-uncased",
           "../input/shopee-text-models-bb-offline/xlm-roberta-large",
          "../input/shopee-text-models-bb-offline/bert-base-indonesian-1.5G",
          "../input/shopee-text-models-bb-offline/indobert-large-p1",
          "../input/shopee-text-models-bb-offline/bert-base-multilingual-uncased",
          "../input/shopee-text-models-bb-offline/xlm-roberta-base"]


def clean():
    gc.collect()
    torch.cuda.empty_cache()


# image and text data
def gen_img_data(backbone, size=img_size):
    if backbone in diff_img_size:
        size = img_size_2
    elif backbone in diff_img_size_2:
        size = img_size_3
    transforms = albumentations.Compose([
    albumentations.Resize(size, size),
    albumentations.Normalize() 
    ])
    data_img = SHOPEEImageDataset(df, img_dir, transform = transforms, gen_feat_only = True)
    dataloader_img = DataLoader(data_img, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    return dataloader_img
    
                            
def gen_text_data(tkn_pth, clean=False):
    data_txt = SHOPEETextDataset(df, tokenizer = tkn_pth, gen_feat_only = True, clean = clean)
    dataloader_txt = DataLoader(data_txt, batch_size=BATCH_SIZE, shuffle= False, 
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

txt_feas = torch.cat([return_feas(txt_model[i], gen_text_data(tkn_pth[i], 
                      clean=txt_backbone[i] in cleaned))
            for i in range(len(txt_backbone))], dim=1)

if not compute_cv:
    del txt_model
    clean()


img_feas_, txt_feas_ = F.normalize(img_feas).cuda(1), F.normalize(txt_feas).cuda(1)

comb_feas = F.normalize(torch.cat([img_feas_, txt_feas_], dim=1)).cuda(1)

del img_feas, txt_feas
clean()

img_feas, txt_feas = img_feas_, txt_feas_


def build_faiss(feas, dim):
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, dim)
    index.add(feas)
    return index


def get_batches(bs, n_batch, feas):
    batches = []
    for i in range(n_batch):
        left = bs * i
        right = bs * (i+1)
        if i == n_batch - 1:
            right = feas.shape[0]
        batches.append(feas[left:right,:])
    return batches


def get_matches(bs, n_batch, feas, dim, k=K):
    index = build_faiss(feas, dim)
    m=[]
    s=[]
    for batch in tqdm.tqdm(get_batches(bs, n_batch, feas)):
        batch = batch.cuda() 
        sims, matches = index.search(batch, k)
        m.append(matches)
        s.append(sims)
    m = torch.cat(m, dim=0).to(torch.int32)
    s = torch.cat(s, dim=0)
    return m,s


def filter_embeddings(feas, matches, sims):
    feas = feas.detach().cpu()
    new_feas = feas.clone()
    
    for i in range(feas.shape[0]):
        cur_feas = feas[matches[i]]
        weights = torch.unsqueeze(torch.Tensor(sims[i]), 1)
        new_feas[i] = weights.T@cur_feas
    new_feas = F.normalize(new_feas)
    return new_feas.cuda()


img_matches, img_sims = get_matches(bs, n_batch, img_feas, di, k=K)     
text_matches, text_sims = get_matches(bs, n_batch, txt_feas, dt, k=K)    
comb_matches, comb_sims = get_matches(bs, n_batch, comb_feas, dc, k=K)

if not compute_cv:
    del img_feas, txt_feas
    clean()


def th_matches(bs, n_batch, matches, sims, th):
    matches = get_batches(bs, n_batch, matches)
    sims = get_batches(bs, n_batch, sims)
    m = []
    s=[]
    for (batch_m, batch_s) in zip(matches, sims):
        batch_m = batch_m.cpu().numpy()
        batch_s = batch_s.cpu().numpy()
        mask = (batch_s > th)
        for row in range(len(mask)):
            m.append(batch_m[row][mask[row]].tolist())
            s.append(batch_s[row][mask[row]].tolist())
    return m, s


img_final, img_sims = th_matches(bs, n_batch, img_matches, img_sims, 0.704)
text_final, text_sims = th_matches(bs, n_batch, text_matches, text_sims, 0.764)
comb_final, comb_sims = th_matches(bs, n_batch, comb_matches, comb_sims, 0.52)


#filter embeddings

comb_feas = filter_embeddings(comb_feas, comb_final, comb_sims)
comb_matches, comb_sims = get_matches(bs, n_batch, comb_feas, dc, k=K)
comb_final, comb_sims = th_matches(bs, n_batch, comb_matches, comb_sims, 0.9)

if not compute_cv:
    del comb_feas
    clean()


def filter_matches(matches, sims, th=1.0, k=3, dist=1e-2):
    top_matches = [row[:k] for row in matches]
    top_sims = [row[:k] for row in sims]
    for i in range(len(matches)):
        if len(matches[i]) < k+1:
            continue
        dist_1 = sims[i][k-2] - sims[i][k-1]
        dist_2 = sims[i][k-1] - sims[i][k]
        if dist_2 < dist:
            continue
        if th*dist_1 < dist_2:
            matches[i] = top_matches[i]
            sims[i] = top_sims[i]
    return matches, sims

img_final,_ = filter_matches(img_final, img_sims, 1.1, 4, 2e-2)
text_final,_ = filter_matches(text_final, text_sims, 1.2, 4, 2e-2)
comb_final,_ = filter_matches(comb_final, comb_sims, 1.0, 3, 2e-2)


def union_matches(*lists):
    matches = []
    for group in zip(*lists):
        matches.append(reduce(np.union1d, group).tolist())
    return matches


# adapted from kaggle.com/code/slawekbiel/resnet18-0-772-public-lb/notebook
measurements = {
    'weight': [('mg',1), ('g', 1000), ('gr', 1000), ('gram', 1000), ('kg', 1000000)],
    'length': [('mm',1), ('cm', 10), ('m',1000), ('meter', 1000)],
    'pieces': [ ('pc',1)],
    'memory': [('gb', 1)],
    'volume': [('ml', 1), ('l', 1000), ('liter',1000)]
}

def to_num(x, mult=1):
    x = x.replace(',','.')
    return int(float(x)*mult)

def extract_unit(tit, m):
    pat = f'\W(\d+(?:[\,\.]\d+)?) ?{m}s?\W'
    matches = regex.findall(pat, tit, overlapped=True)
    return set(matches)

def extract(tit):
    res =dict()
    tit = ' '+tit.lower()+' '
    for cat, units in measurements.items():
        cat_values=set()
        for unit_name, mult in units:
            values = extract_unit(tit, unit_name)
            values = {to_num(v, mult) for v in values}
            cat_values = cat_values.union(values)
        if cat_values:
            res[cat] = cat_values
    return res

def match_measures(m1, m2):
    k1,k2 = set(m1.keys()), set(m2.keys())
    common = k1.intersection(k2)
    if not common: 
        return True
    for key in common:
        s1,s2 = m1[key], m2[key]
        if s1.intersection(s2):
            return True
    return False

def filter_matches(matches : list): # filter matches override
    for i in range(len(matches)):
        item_title = extract(df.iloc[i].title)
        l=[]
        for match in matches[i]:
            if match == i:
                l.append(i)
                continue
            match_title = extract(df.iloc[match].title)
            if (match_measures(item_title, match_title)):
                l.append(match)
        matches[i] = l
    return matches


match_final = union_matches(img_final, text_final, comb_final)
match_final = filter_matches(match_final)


if not compute_cv:
    matches=[]
    for match in match_final: 
        matches.append(' '.join(df.iloc[match].posting_id.tolist()))


if not compute_cv:
    submission = pd.read_csv("../input/shopee-product-matching/sample_submission.csv")[0:0]
    submission['posting_id'] = df.posting_id
    submission['matches'] = matches
    
    submission.to_csv('submission.csv', index=False)







