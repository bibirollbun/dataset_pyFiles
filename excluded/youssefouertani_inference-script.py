try:
    import zarr
except: 
    !cp -r '/kaggle/input/hengck-czii-cryo-et-01/wheel_file' '/kaggle/working/'
    !pip install /kaggle/working/wheel_file/asciitree-0.3.3/asciitree-0.3.3
    !pip install --no-index --find-links=/kaggle/working/wheel_file zarr
    !pip install --no-index --find-links=/kaggle/working/wheel_file connected-components-3d


from typing import List, Tuple, Union
deps_path = '/kaggle/input/czii-cryoet-dependencies'
! pip install -q --no-index --find-links {deps_path} --requirement {deps_path}/requirements.txt

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import torch
import torch.nn as nn
import gc
import random
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import glob

import zarr
import cc3d

print('PIP INSTALL OK!!!')


DATA_KAGGLE_DIR = '/kaggle/input/czii-cryo-et-object-identification'
TRAIN_DIR = f'{DATA_KAGGLE_DIR}/train'

device = "cuda" if torch.cuda.is_available() else "cpu"

scale = 10.012444196428572

EPOCH = 35

blob_factor = 7

do_xy = False

OBJECT_DICT = {
    'apo-ferritin': {'label': 1, 'radius': 60/scale}, 
    'beta-galactosidase': {'label': 2, 'radius': 90/scale}, 
    'ribosome': {'label': 3, 'radius': 150/scale}, 
    'thyroglobulin': {'label': 4, 'radius': 130/scale}, 
    'virus-like-particle': {'label': 5, 'radius': 135/scale},
    #'beta-amylase' : {'label': 6, 'radius':65/scale},
}

MODE='submit'

if MODE=='local':
    valid_dir =f'{DATA_KAGGLE_DIR}/train'
    valid_id = ["TS_5_4"]
    
if MODE=='submit':
    valid_dir =f'{DATA_KAGGLE_DIR}/test'
    valid_id = glob.glob(f'{valid_dir}/static/ExperimentRuns/*')
    valid_id = [f.split('/')[-1] for f in valid_id]

print('valid_id:',len(valid_id), valid_id)


def calculate_sphere_volume(radius):
    """
    Calculate the volume of a sphere given its radius.

    Parameters:
    radius (float): The radius of the sphere.

    Returns:
    float: The volume of the sphere.
    """
    if radius < 0:
        raise ValueError("Radius cannot be negative.")
    volume = (4 / 3) * np.pi * np.power(radius * 0.8, 3)
    return volume


for k in OBJECT_DICT.keys():
    OBJECT_DICT[k]["blob"] = calculate_sphere_volume(np.log2(OBJECT_DICT[k]["radius"]))/blob_factor


OBJECT_DICT


def read_one_data(id, static_dir):
    zarr_dir = f'{static_dir}/{id}/VoxelSpacing10.000'
    zarr_file = f'{zarr_dir}/denoised.zarr'
    zarr_data = zarr.open(zarr_file, mode='r')
    volume = zarr_data[0][:]

    pmin,pmax = -1.1769942479337e-05 , 1.2801160441345688e-05
    # mean = volume.mean()
    # std = volume.std()
    # volume = (volume - mean) / std
    return ((volume-pmin)/(pmax-pmin)).astype(np.float16)


def read_one_truth(id, overlay_dir):
    location={}
    json_dir = f'{overlay_dir}/{id}/Picks'
    for p in OBJECT_DICT.keys():
        json_file = f'{json_dir}/{p}.json'
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        num_point = len(json_data['points'])
        loc = [list(json_data['points'][i]['location'].values())  for i in range(num_point)]
        location[p] = [[coo for coo in coos] for coos in loc ]
    return location

def do_one_eval(truth, predict, threshold = 3):
    P=len(predict)
    T=len(truth)

    if P==0:
        hit=[[],[]]
        miss=np.arange(T).tolist()
        fp=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    if T==0:
        hit=[[],[]]
        fp=np.arange(P).tolist()
        miss=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    #---
    distance = predict.reshape(P,1,3)-truth.reshape(1,T,3)
    distance = distance**2
    distance = distance.sum(axis=2)
    distance = np.sqrt(distance)
    p_index, t_index = linear_sum_assignment(distance)

    valid = distance[p_index, t_index] <= threshold
    p_index = p_index[valid]
    t_index = t_index[valid]
    hit = [p_index.tolist(), t_index.tolist()]
    miss = np.arange(T)
    miss = miss[~np.isin(miss,t_index)].tolist()
    fp = np.arange(P)
    fp = fp[~np.isin(fp,p_index)].tolist()

    metric = [P,T,len(hit[0]),len(miss),len(fp)] #for lb metric F-beta copmutation
    return hit, fp, miss, metric


patch_size = (184,128,128)
def calculate_patch_starts(dimension_size: int, patch_size: int):
    if dimension_size <= patch_size:
        return [0]
        

    n_patches = np.ceil(dimension_size / patch_size)  +1 if dimension_size>300 else 1

    
    if n_patches == 1:
        return [0]
    
    # Calculate overlap
    total_overlap = (n_patches * patch_size - dimension_size) / (n_patches - 1)
    
    # Generate starting positions
    positions = []
    for i in range(int(n_patches)):
        pos = int(i * (patch_size - total_overlap))
        if pos + patch_size > dimension_size:
            pos = dimension_size - patch_size
        if pos not in positions:  # Avoid duplicates
            positions.append(pos)
        
    return positions


    
class PredDataset(Dataset):
    def __init__(self,experiment ,patch_size = patch_size):
        self.is_local = MODE == "local"
        self.patch_size = patch_size
        #self.zyx = np.ones((3, 184+patch_size , 630+patch_size, 630+patch_size))*-1
        pad_size = [patch_size[i]//3 for i in range(3)]
        #self.zyx [:,pad_size:184+pad_size, pad_size:630+pad_size, pad_size:630+pad_size] = np.indices((184,630,630))
        
        self.volume = read_one_data(experiment, static_dir=f'{valid_dir}/static/ExperimentRuns')
        
        self.locations = read_one_truth(experiment, overlay_dir=f'{TRAIN_DIR}/overlay/ExperimentRuns') if self.is_local else None

        self.indexes = [[z,y,x] 
                       for z in calculate_patch_starts(184,patch_size[0])
                       for y in calculate_patch_starts(630, patch_size[1])
                       for x in calculate_patch_starts(630, patch_size[2])]
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self,idx):

        zyx = self.indexes [idx]
        patch = self.volume[zyx[0]:zyx[0]+self.patch_size[0],zyx[1]:zyx[1]+self.patch_size[1],zyx[2]:zyx[2]+self.patch_size[2]]

        x_flip = np.flip(patch,-1)
        y_flip = np.flip(patch,-2)
        z_flip = np.flip(patch,-3)
        patch = torch.tensor(patch,dtype = torch.float32)
        rot_1 = torch.rot90(patch , k = 1, dims = (-1,-2))
        rot_2 = torch.rot90(patch , k = 2, dims = (-1,-2))
        rot_3 = torch.rot90(patch , k = 3, dims = (-1,-2))
        
        return {
            "volume": patch,
            "x_flip":torch.tensor(x_flip.copy(),dtype = torch.float32),
            "y_flip":torch.tensor(y_flip.copy(),dtype = torch.float32),
            "z_flip":torch.tensor(z_flip.copy(),dtype = torch.float32),
            "rot_1":rot_1,
            "rot_2":rot_2,
            "rot_3":rot_3,
            'zyx':  torch.tensor(zyx,dtype = torch.long)}

def evaluate_predictions(stats, pred_loader, distance_threshold=3, beta=4 , particle_name = None):
    best_f_beta = 0
    best_metric = None
    # Filter predictions based on voxel count
    pred = np.array([centroid for i, centroid in enumerate(stats[particle_name]["centroids"]) if i != 0 and stats[particle_name]["voxel_counts"][i] > OBJECT_DICT[particle_name]["blob"]])
    pred *= scale
    if len(pred)==0:
        return {
            "truth": 0,
            "predict": 0,
            "hit": 0,
            "fp": 0,
            "miss": 0,
            "f_b": 0
        }
    pred = pred[:,::-1]
    truth_locations = np.array(pred_loader.dataset.locations[particle_name])
    # Perform evaluation
    hit, fp, miss, metric = do_one_eval(truth_locations, pred, distance_threshold)

    # Calculate precision, recall, and F-beta score
    precision = len(hit[0]) / (len(hit[0]) + len(fp)) if (len(hit[0]) + len(fp)) > 0 else 0
    recall = len(hit[0]) / (len(hit[0]) + len(miss)) if (len(hit[0]) + len(miss)) > 0 else 0

    beta_squared = beta ** 2
    f_beta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall) if (precision + recall) > 0 else 0
    if f_beta>= best_f_beta:
        best_f_beta = f_beta
        best_metric = {
            "truth": len(truth_locations),
            "predict": len(pred),
            "hit": len(hit[0]),
            "fp": len(fp),
            "miss": len(miss),
            "f_b": f_beta
        }
    # Return results as JSON-like dictionary
    return best_metric
    
def do_one_eval(truth, predict, threshold = 3):
    P=len(predict)
    T=len(truth)

    if P==0:
        hit=[[],[]]
        miss=np.arange(T).tolist()
        fp=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    if T==0:
        hit=[[],[]]
        fp=np.arange(P).tolist()
        miss=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    #---
    distance = predict.reshape(P,1,3)-truth.reshape(1,T,3)
    distance = distance**2
    distance = distance.sum(axis=2)
    distance = np.sqrt(distance)
    p_index, t_index = linear_sum_assignment(distance)

    valid = distance[p_index, t_index] <= threshold
    p_index = p_index[valid]
    t_index = t_index[valid]
    hit = [p_index.tolist(), t_index.tolist()]
    miss = np.arange(T)
    miss = miss[~np.isin(miss,t_index)].tolist()
    fp = np.arange(P)
    fp = fp[~np.isin(fp,p_index)].tolist()

    metric = [P,T,len(hit[0]),len(miss),len(fp)] #for lb metric F-beta copmutation
    return hit, fp, miss, metric



maps = {
    "cuda:0":[],
    "cuda:1":[]
}
for i,exp_name in enumerate(valid_id):
    if i%2 == 0:
        maps["cuda:0"].append(exp_name)
    else:
        maps["cuda:1"].append(exp_name)
print(maps)

loaders = {
    "cuda:0":None,
    "cuda:1":None
}
print(loaders)


#maps["cuda:0"] = [valid_id[0] for _ in range(250)]
#maps["cuda:1"] = [valid_id[0] for _ in range(250)]





class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

import torch.nn as nn

class ConvBNReLU2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU2D, self).__init__()
        if kernel_size == 5:
            padding = 2

        if kernel_size == 7:
            padding = 3
            
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
        
class ConvBNReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
            

import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels = 0, num_conv3d=1, 
                 do_up = True , do_down = True ,use_transpose = False):
        super(EncoderBlock, self).__init__()
        self.do_up = do_up
        
        if self.do_up:
            self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode='trilinear')
            
        if use_transpose:
            self.upsample = nn.Sequential(
                nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
            
        self.do_down = do_down 
        if self.do_down:
            self.downsample = lambda x: F.interpolate(x, scale_factor=.5, mode='trilinear')

        
        self.conv3d_layers = nn.Sequential(
            *[ConvBNReLU3D(out_channels if i!=0 else in_channels + skip_channels, out_channels, stride= (1, 1, 1)) 
              for i in range(num_conv3d)]
        )

    def forward(self, x , xskip = None):
        if xskip is not None:
            x = torch.cat([x, xskip], dim=1)

        out = self.conv3d_layers(x)
        output = {
            "out": out,
            "up":None,
            "down":None
        }

        if self.do_up:
            output["up"] = self.upsample(out)   
        if self.do_down:
            output["down"] = self.downsample(out)   
        return dotdict(output)
    
class Model(nn.Module):
    def __init__(self , channels = [28,32,36]):
        super(Model, self).__init__()
        self.register_buffer('D', torch.tensor(0))
        self.output_type = ['particle', 'loss']

        self.norm = nn.BatchNorm3d(1)
        
        self.encoder1 = EncoderBlock(in_channels = 1, out_channels = channels[0], num_conv3d=2 , do_up = False, do_down=True)
        self.encoder2 = EncoderBlock(in_channels = channels[0], out_channels = channels[1], num_conv3d=2 , do_up = True, do_down=True)
        
        self.decoder1 = EncoderBlock(in_channels = channels[1], out_channels = channels[2], num_conv3d=4 , do_up = True, do_down=False)
        self.decoder2 = EncoderBlock(in_channels = channels[2], out_channels = channels[1], num_conv3d=2 , skip_channels = channels[1], do_up = True, do_down=False , use_transpose = True)

        self.pre = EncoderBlock(in_channels = channels[1], out_channels = channels[0], num_conv3d=2 , do_up = False, do_down=False)

        self.mask = nn.Conv3d(channels[0], 6, 1, 1, bias=False)
        
        

    def forward(self,batch):
        device = self.D.device
        volume = batch["volume"].to(device).unsqueeze(1)

        input_ = self.norm(volume)
        
        encode1 = self.encoder1(input_)
        encode2 = self.encoder2(encode1.down)
        
        decode1 = self.decoder1(encode2.down)
        #print(encode2.out.shape , decode1.up.shape)
        decode2 = self.decoder2(encode2.out , decode1.up)

        pre = self.pre(decode2.up)

        logit = self.mask(pre.out)
        #print(mask.shape)

        output = {}
        
        if "loss" in self.output_type and "label" in batch.keys():
        
            # Apply weighted cross-entropy loss
            output["loss"] = F.cross_entropy(
                logit, 
                batch['label'].to(device), 
                label_smoothing=0.01,
            )

        if "particle" in self.output_type:
            output['particle'] = F.softmax(logit,1)
            
        return output


class AVGModel(nn.Module):
    def __init__(self, models):
        super(AVGModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self,batch):
        output = {"particle": 0}
        volume = batch["volume"].to(device)
        b = len(volume)
        z_flip = batch["z_flip"].to(device)
        y_flip = batch["y_flip"].to(device)
        x_flip = batch["x_flip"].to(device)
        rot_1 = batch["rot_1"].to(device)
        rot_2 = batch["rot_2"].to(device)
        rot_3 = batch["rot_3"].to(device)
        
        batch["volume"] = torch.cat([volume ,x_flip , y_flip ,z_flip , rot_1, rot_2, rot_3] , 0)
        all_ = 0
        for model in self.models:
            all_ += model(batch)["particle"]
        all_ /= len(self.models)
        
        for i in range(7):
            if i ==0:
                output['particle'] += all_[b*i:b*(i+1)]
            elif i<4 :
                output['particle'] += torch.flip(all_[b*i:b*(i+1)], dims = [-i])
            else :
                rot = i-3
                output['particle'] += torch.rot90(all_[b*i:b*(i+1)], k = rot, dims = (-2,-1))
        output['particle'] /= 7
        return output


paths = [
    "/kaggle/input/deepfinder-120-seed/model_all_35.bin",
    "/kaggle/input/deepfinder-128/model_all_35.bin",
    "/kaggle/input/deep-finder-128-80-seed/model_all_35.bin",
    "/kaggle/input/deep-finder-128-90/model_all_35.bin"
]

paths = [
    #"/kaggle/input/deepfinder-111-seed",
    "/kaggle/input/train-script-42-90/model_all_35_42.bin",
    "/kaggle/input/train-script-42-90/model_all_35_90.bin",
    "/kaggle/input/train-script-80-120/model_all_35_120.bin",
    "/kaggle/input/train-script-80-120/model_all_35_80.bin"
]



def get_deep(path):
    model = Model()

    state_dict = torch.load(path,weights_only=True , map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    return model



model = {
    "cuda:0": AVGModel([get_deep(path).to("cuda:0") for path in paths]),
    "cuda:1": AVGModel([get_deep(path).to("cuda:1") for path in paths]),
}

weight = torch.zeros((patch_size[0], patch_size[1], patch_size[2]) , dtype =torch.float16)
weight[8:patch_size[0]-8, 8:patch_size[1]-8, 8:patch_size[2]-8] += 1
weight += .1

weights = {
    "cuda:0": weight.to("cuda:0"),
    "cuda:1": weight.to("cuda:1")
}

logits = {
    "cuda:0": torch.zeros((6, 184, 630, 630) , dtype =torch.float16).to("cuda:0"),
    "cuda:1": torch.zeros((6, 184, 630, 630 ) , dtype =torch.float16).to("cuda:1")
}

count = {
    "cuda:0": torch.zeros((184 , 630 , 630 ) , dtype =torch.float16).to("cuda:0"),
    "cuda:1": torch.zeros((184 , 630 , 630 ) , dtype =torch.float16).to("cuda:1")
}


#stats,evals = get_stats(probs,loaders["cuda:0"])
best_eval = {
    'apo-ferritin': {'thresh': 0.0575000000000000005},
    'beta-galactosidase': {'thresh': 0.050000000000004},
    'ribosome': {'thresh': 0.0575000000000000005},
    'thyroglobulin':{'thresh': 0.05000000000000},
    'virus-like-particle': {'thresh': 0.125}
}

def get_probs(exp_name, device):
    global loaders, weight, logits, count
    #set the loader
    del loaders[device]
    gc.collect()
    loaders[device] = DataLoader(PredDataset(experiment = exp_name ),batch_size=1,shuffle=False,num_workers=2 )
    logits[device] = logits[device].zero_()
    count[device] = count[device].zero_()

    with torch.no_grad():
        with torch.amp.autocast(device):
            for batch in tqdm(loaders[device]):
                local_logits = model[device](batch)["particle"]
                for i, logits_patch in enumerate(local_logits):
                    
                    z, y, x = batch["zyx"][i]
                    z_slice = slice(z, z + patch_size[0])
                    y_slice = slice(y, y + patch_size[1])
                    x_slice = slice(x, x + patch_size[2])
                    
                    count[device][z_slice, y_slice, x_slice] += weights[device]
                    logits[device][:, z_slice, y_slice, x_slice] += logits_patch * weights[device]


            probs = (logits[device]/count[device]).detach().cpu().numpy()
    return probs

def get_stats(probs,pred_loader,search_thresh = False):
    stats = {}
    evals = {}
    for particle in OBJECT_DICT.keys():
        label = OBJECT_DICT[particle]["label"]
        if MODE == "local" and search_thresh:
            for prob_thresh in np.arange(0.05, 0.1, 0.01):
                thresh = OBJECT_DICT[particle]["radius"]/2*scale
                labels_out = cc3d.connected_components(probs[label, :, :, :] > prob_thresh, connectivity=18)
                stats[particle] = cc3d.statistics(labels_out)
                
                eval_ = evaluate_predictions(stats, pred_loader, distance_threshold = thresh, particle_name = particle)
                if particle not in evals.keys():
                    evals[particle] = eval_
                    evals[particle]["thresh"] = prob_thresh
                elif eval_["f_b"]>=evals[particle]["f_b"]:
                    evals[particle] = eval_
                    evals[particle]["thresh"] = prob_thresh

        elif MODE == "local":
            
            thresh = OBJECT_DICT[particle]["radius"]/2*scale
            labels_out = cc3d.connected_components(probs[label, :, :, :] > best_eval[particle]["thresh"], connectivity=18)
            stats[particle] = cc3d.statistics(labels_out)
            
            evals[particle] = evaluate_predictions(stats, pred_loader, distance_threshold = thresh, particle_name = particle)
        
        else :
            labels_out = cc3d.connected_components(probs[label, :, :, :] > best_eval[particle]["thresh"], connectivity=18)
            stats[particle] = cc3d.statistics(labels_out)
            
    return stats,evals

def stats_to_df(stats,exp_name):
    result = pd.DataFrame(columns=["x","y","z","particle_type","experiment"])
    for particle_name in OBJECT_DICT.keys():
        pred = np.array([centroid for i, centroid in enumerate(stats[particle_name]["centroids"]) if i != 0 and stats[particle_name]["voxel_counts"][i] > OBJECT_DICT[particle_name]["blob"]])
        if len(pred)==0:
            continue


        pred *= scale
        pred = pred[:,::-1]
        df = pd.DataFrame(pred, columns=["x","y","z"])
        df["experiment"] = [exp_name for _ in range(len(df))]
        df["particle_type"] = [particle_name for _ in range(len(df))]
        result = pd.concat([result,df])
    return result





if MODE == "local":
    for k in OBJECT_DICT.keys():
        OBJECT_DICT[k]["blob"] = calculate_sphere_volume(np.log2(OBJECT_DICT[k]["radius"]))/100
    probs = get_probs(valid_id[0],"cuda:0")
    s,e = get_stats(probs, loaders["cuda:0"])
    fb = 0
    s,e = get_stats(probs, loaders["cuda:0"])
    print(e)
    for p in e.keys():
        if p == "beta-galactosidase" or p == "thyroglobulin":
            fb+= 2*e[p]["f_b"]
    
        else :
            fb+= e[p]["f_b"]
    
        print(fb)
    
    print(fb/7)


if MODE == "local":
    fb = 0
    for k in OBJECT_DICT.keys():
        OBJECT_DICT[k]["blob"] = calculate_sphere_volume(np.log2(OBJECT_DICT[k]["radius"]))/3
    s,e = get_stats(probs, loaders["cuda:0"])
    print(e)
    for p in e.keys():
        if p == "beta-galactosidase" or p == "thyroglobulin":
            fb+= 2*e[p]["f_b"]
    
        else :
            fb+= e[p]["f_b"]
    
        print(fb)
    
    print(fb/7)


dfs = {
    "cuda:0":pd.DataFrame(columns=["x","y","z","particle_type","experiment"]),
    "cuda:1":pd.DataFrame(columns=["x","y","z","particle_type","experiment"])
}

def predict(exp_name,device, search_thresh=False):
    probs = get_probs(exp_name,device)
    stats,evals = get_stats(probs , loaders[device], search_thresh)
    if MODE == "local":
        print(exp_name,evals)
    prediction_df = stats_to_df(stats,exp_name)
    if prediction_df is not None:
        dfs[device] = pd.concat([dfs[device],prediction_df])

def predict_all(device):
    
    print("predicting" , device)
    for exp_name in maps[device] :
        predict(exp_name,device)


from concurrent.futures import ThreadPoolExecutor

# Initialize DataFrames
dfs = {
    "cuda:0": pd.DataFrame(columns=["x", "y", "z", "particle_type", "experiment"]),
    "cuda:1": pd.DataFrame(columns=["x", "y", "z", "particle_type", "experiment"]),
}

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [
        executor.submit(predict_all, "cuda:0"),
        executor.submit(predict_all, "cuda:1"),
    ]

# Wait for all threads to finish
for future in futures:
    future.result()  # Ensures exceptions in threads are raised

print("Processing complete!")


submit_df = pd.concat([dfs["cuda:0"],dfs["cuda:1"]])
submit_df["id"] = [i for i in range(len(submit_df))]


submit_df.to_csv("submission.csv",index=False)


submit_df




