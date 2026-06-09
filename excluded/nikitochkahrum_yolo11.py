!cp -r '/kaggle/input/hengck-czii-cryo-et-01/wheel_file' '/kaggle/working/'
!pip install /kaggle/working/wheel_file/asciitree-0.3.3/asciitree-0.3.3
!pip install --no-index --find-links=/kaggle/working/wheel_file zarr


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zarr
from tqdm import tqdm
import glob, os
import cv2


runs = sorted(glob.glob('/kaggle/input/czii-cryo-et-object-identification/train/overlay/ExperimentRuns/*'))
runs = [os.path.basename(x) for x in runs]
i2r_dict = {i:r for i, r in zip(range(len(runs)), runs)}
r2t_dict = {r:i for i, r in zip(range(len(runs)), runs)}
i2r_dict


def convert_to_8bit(x):
    lower, upper = np.percentile(x, (0.5, 99.5))
    x = np.clip(x, lower, upper)
    x = (x - x.min()) / (x.max() - x.min() + 1e-12) * 255
    return x.round().astype("uint8")


p2i_dict = {
        'apo-ferritin': 0,
        'beta-amylase': 1,
        'beta-galactosidase': 2,
        'ribosome': 3,
        'thyroglobulin': 4,
        'virus-like-particle': 5
    }

i2p = {v:k for k, v in p2i_dict.items()}

particle_radius = {
        'apo-ferritin': 60,
        'beta-amylase': 65,
        'beta-galactosidase': 90,
        'ribosome': 150,
        'thyroglobulin': 130,
        'virus-like-particle': 135,
    }


particle_names = ['apo-ferritin', 'beta-amylase', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']


def make_annotate_yolo(run_name, is_train_path=True):
    # to split validation
    is_train_path = 'train' if is_train_path else 'val'

    # read a volume
    vol = zarr.open(f'/kaggle/input/czii-cryo-et-object-identification/train/static/ExperimentRuns/{run_name}/VoxelSpacing10.000/denoised.zarr', mode='r') #bug fixed. Thanks to @pratyushh
    # use largest images
    vol = vol[0]
    # normalize [0, 255]
    vol2 = convert_to_8bit(vol)
    
    n_imgs = vol2.shape[0]
    # process each slices
    for j in range(n_imgs):
        newvol = vol2[j]
        newvolf = np.stack([newvol]*3, axis=-1)
        # YOLO requires image_size is multiple of 32
        newvolf = cv2.resize(newvolf, (640,640))
        # save as 1 slice
        cv2.imwrite(f'images/{is_train_path}/{run_name}_{j*10}.png', newvolf)
        # make txt file for annotation
        with open(f'labels/{is_train_path}/{run_name}_{j*10}.txt', 'w'):
            pass # make empty file
            
    # process each paticle types
    for p, particle in enumerate(tqdm(particle_names)):
        # we do not have to detect beta-amylase which weight is 0
        if particle=="beta-amylase":
            continue
        json_each_paticle = f"/kaggle/input/czii-cryo-et-object-identification/train/overlay/ExperimentRuns/{run_name}/Picks/{particle}.json"
        df = pd.read_json(json_each_paticle) 
        # pick each coordinate of particles
        for axis in "x", "y", "z":
            df[axis] = df.points.apply(lambda x: x["location"][axis])

        
        radius = particle_radius[particle]
        for i, row in df.iterrows():
            # The radius from the center of the particle is used to determine the slices present.
            start_z = np.round(row['z'] - radius).astype(np.int32)
            start_z = max(0, start_z//10) # 10 means pixelspacing
            end_z = np.round(row['z'] + radius).astype(np.int32)
            end_z = min(n_imgs, end_z//10) # 10 means pixelspacing
            
            for j in range(start_z+1, end_z+1-1, 1):
                # white the results of annotation
                with open(f'labels/{is_train_path}/{run_name}_{j*10}.txt', 'a') as f:
                    f.write(f'{p2i_dict[particle]} {row["x"]/10/vol2.shape[1]} {row["y"]/10/vol2.shape[2]} {radius/10/vol2.shape[1]*2} {radius/10/vol2.shape[2]*2} \n')


os.makedirs("images/train", exist_ok=True)
os.makedirs("images/val", exist_ok=True)
os.makedirs("labels/val", exist_ok=True)
os.makedirs("labels/train", exist_ok=True)


# use TS_5_4 as validation
for i, r in enumerate(runs):
    make_annotate_yolo(r, is_train_path=False if i==0 else True)


import shutil
os.makedirs('datasets/czii_det2d', exist_ok=True)
shutil.move('images/train', 'datasets/czii_det2d/images/train')
shutil.move('images/val', 'datasets/czii_det2d/images')
shutil.move('labels/train', 'datasets/czii_det2d/labels/train')
shutil.move('labels/val', 'datasets/czii_det2d/labels')


%%writefile czii_conf.yaml

path: /kaggle/working/datasets/czii_det2d # dataset root dir
train: images/train # train images (relative to 'path') 
val: images/val # val images (relative to 'path') 

# Classes
names:
  0: apo-ferritin
  1: beta-amylase
  2: beta-galactosidase
  3: ribosome
  4: thyroglobulin
  5: virus-like-particle


!tar xfvz /kaggle/input/ultralytics-for-offline-install/archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages


from tqdm import tqdm
import glob, os
from ultralytics import YOLO


# Load a pretrained model
model = YOLO("/kaggle/input/yolo11/pytorch/default/1/yolo11l.pt")  # load a pretrained model (recommended for training)


# Train the model
_ = model.train(
    data="/kaggle/working/czii_conf.yaml",
    epochs=120,
    warmup_epochs=10,
    optimizer='AdamW',
    cos_lr=True,
    lr0=3e-4,
    lrf=0.03,
    imgsz=640,
    device="0,1",
    weight_decay=0.005,
    batch=32,
    scale=0,
    flipud=0.5,
    fliplr=0.5,
    degrees=45,
    shear=5,
    mixup=0.2,
    copy_paste=0.25,
    seed=8620, # (｡•◡•｡)
)


model = YOLO("/kaggle/working/runs/detect/train/weights/best.pt")
metrics = model.val(data="/kaggle/working/czii_conf.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0", save_json=True)  # no arguments needed, dataset and settings remembered
print(metrics.box.map)  # map50-95
print(metrics.box.map50)  # map50
print(metrics.box.map75)  # map75
print(metrics.box.maps)


results = model("/kaggle/working/datasets/czii_det2d/images/val/TS_5_4_920.png")
results[0].show()


import zarr
from ultralytics import YOLO
from tqdm import tqdm
import glob, os
import torch


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


import sys
sys.setrecursionlimit(10000)

import warnings
warnings.simplefilter('ignore')
np.warnings = warnings


runs = sorted(glob.glob('/kaggle/input/czii-cryo-et-object-identification/test/static/ExperimentRuns/*'))
runs = [os.path.basename(x) for x in runs]
#change by @minfuka
# runs[:5]
sp = len(runs)//2
runs1 = runs[:sp]
runs1[:5]


runs2 = runs[sp:]
runs2[:5]


assert torch.cuda.device_count() == 2


particle_names = ['apo-ferritin', 'beta-amylase', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']


p2i_dict = {
        'apo-ferritin': 0,
        'beta-amylase': 1,
        'beta-galactosidase': 2,
        'ribosome': 3,
        'thyroglobulin': 4,
        'virus-like-particle': 5
    }

i2p = {v:k for k, v in p2i_dict.items()}


particle_radius = {
        'apo-ferritin': 60,
        'beta-amylase': 65,
        'beta-galactosidase': 90,
        'ribosome': 150,
        'thyroglobulin': 130,
        'virus-like-particle': 135,
    }


class PredAggForYOLO:
    def __init__(self, first_conf=0.2, final_conf=0.3, conf_coef=0.75):
        self.first_conf = first_conf # threshold of confidence yolo
        self.final_conf = final_conf # final threshold score (not be used in version 14)
        self.conf_coef = conf_coef # if found many points, give bonus
        self.particle_confs = [0.5, 0.0, 0.2, 0.5, 0.2, 0.5] # be strict to easy labels 

    def convert_to_8bit(self, x):
        lower, upper = np.percentile(x, (0.5, 99.5))
        x = np.clip(x, lower, upper)
        x = (x - x.min()) / (x.max() - x.min() + 1e-12) * 255
        return x.round().astype("uint8")

    # depth first search.
    # aggregate the coordinates and confidence scores of connected graphs.
    def dfs(self, v):
        self.passed[v] = True
        self.conf_sum += self.pdf.iloc[v].confidence
        self.cx += self.pdf.iloc[v].x
        self.cy += self.pdf.iloc[v].y
        self.cz += self.pdf.iloc[v].z
        self.nv += 1
        for next_v in self.adjacency_list[v]:
            if (self.passed[next_v]): continue
            self.dfs(next_v)

    # main routine.
    # change by @minfuka
    # def make_predict_yolo(self, r, model):
    def make_predict_yolo(self, r, model, device_no):
        vol = zarr.open(f'/kaggle/input/czii-cryo-et-object-identification/test/static/ExperimentRuns/{r}/VoxelSpacing10.000/denoised.zarr', mode='r')
        vol = vol[0]
        vol2 = self.convert_to_8bit(vol)
        n_imgs = vol2.shape[0]
    
        df = pd.DataFrame()
    
        pts = []
        confs = []
        xs = []
        ys = []
        zs = []
        
        for i in range(n_imgs):
            # Unfortunately the image size needs to be a multiple of 32.
            tmp_img = np.zeros((630, 630))
            tmp_img[:] = vol2[i]
    
            inp_arr = np.stack([tmp_img]*3,axis=-1)
            inp_arr = cv2.resize(inp_arr, (640,640))

            # change by @minfuka
            # res = model.predict(inp_arr, save=False, imgsz=640, conf=self.first_conf, device="0", batch=1, verbose=False)
            res = model.predict(inp_arr, save=False, imgsz=640, conf=self.first_conf, device=device_no, batch=1, verbose=False)
            for j, result in enumerate(res):
                boxes = result.boxes # Boxes object for bounding box outputs    
                for k in range(len(boxes.cls)):
                    ptype = i2p[boxes.cls.cpu().numpy()[k]] # particle type
                    conf = boxes.conf.cpu().numpy()[k] # confidence score
                    # YOLO can infer (start_x, end_x, start_y, end_y)
                    xc = (boxes.xyxy[k,0] + boxes.xyxy[k,2]) / 2.0 * 10 * (63/64)
                    yc = (boxes.xyxy[k,1] + boxes.xyxy[k,3]) / 2.0 * 10 * (63/64)
                    zc = i * 10 + 0
    
                    pts.append(ptype)
                    confs.append(conf)
                    xs.append(xc.cpu().numpy())
                    ys.append(yc.cpu().numpy())
                    zs.append(zc)           
                
        df['particle_type'] = pts
        df['confidence'] = confs
        df['x'] = xs
        df['y'] = ys
        df['z'] = zs

        # df includes overall canditate of CIRCLE. 
        df = df.sort_values(['particle_type', 'z'], ascending=[True, True])
    
        agg_df = []

        # infer center of sphere each particle types
        for pidx, p in enumerate(particle_names):
            if p == 'beta-amylase':
                continue
            pdf = df[df['particle_type']==p].reset_index(drop=True)
            self.pdf = pdf
            p_rad = particle_radius[p]

            # The distance between the x and y coordinates of adjacent slices is expected to be very small.
            xy_tol = p_rad / 16.0
            xy_tol_p2 = xy_tol ** 2

            # define the graph
            self.adjacency_list = [[] for _ in range(len(pdf))]
            # which already passed in dfs
            self.passed = [False for _ in range(len(pdf))]

            # Connect two points when they are close enough
            for i in range(len(pdf)):
                x1 = pdf['x'].iloc[i]
                y1 = pdf['y'].iloc[i]
                z1 = pdf['z'].iloc[i]
                for j in range(i+1, len(pdf), 1):
                    x2 = pdf['x'].iloc[j]
                    y2 = pdf['y'].iloc[j]
                    z2 = pdf['z'].iloc[j]
                    # Can be pruned. thanks to min fuka (@minfuka)
                    if abs(z1-z2)>20:
                        break
    
                    dist_p2 = (x1-x2)**2 + (y1-y2)**2
                    if dist_p2<xy_tol_p2 and dist_p2+(z1-z2)**2 < p_rad**2 and abs(z1-z2)<=20:
                        self.adjacency_list[i].append(j)
                        self.adjacency_list[j].append(i)

            rdf = pd.DataFrame()
            cxs = []
            cys = []
            czs = []

            # Perform DFS on all points and find the center of the sphere from the average of the coordinates
            for i in range(len(pdf)):
                self.conf_sum = 0
                self.nv = 0
                self.cx = 0
                self.cy = 0
                self.cz = 0
                if not self.passed[i]:
                    self.dfs(i)

                # Different confidence for different particle types
                if self.nv>=2 and self.conf_sum / (self.nv**self.conf_coef) > self.particle_confs[pidx]:
                    cxs.append(self.cx / self.nv)
                    cys.append(self.cy / self.nv)
                    czs.append(self.cz / self.nv)

            rdf['experiment'] = [r] * len(cxs)
            rdf['particle_type'] = [p] * len(cys)
            rdf['x'] = cxs
            rdf['y'] = cys
            rdf['z'] = czs

            agg_df.append(rdf)

       
        return pd.concat(agg_df, axis=0)


# instance main class
agent = PredAggForYOLO(first_conf=0.15, final_conf=0.2, conf_coef=0.5) # final_conf is not used after version 14


import time
#add by @minfuka
from concurrent.futures import ProcessPoolExecutor #add by @minfuka


#add by @minfuka
def inference(runs, model, device_no):
    subs = []
    for r in tqdm(runs, total=len(runs)):
        df = agent.make_predict_yolo(r, model, device_no)
        subs.append(df)
    
    return subs


subs = inference(runs, model, "0")


# print(f'estimated predict time is {(tock-tick)/3*500:.4f} seconds')


#submission = pd.concat(subs).reset_index(drop=True)
#change by @minfuka
submission = pd.concat(subs).reset_index(drop=True)
submission.insert(0, 'id', range(len(submission)))


sample = pd.read_csv('/kaggle/input/czii-cryo-et-object-identification/sample_submission.csv')


!rm -rf /kaggle/working/*


sample.drop(sample.index, inplace=True)  # Удаляем все строки
sample = pd.concat([sample, submission], ignore_index=True)  # Добавляем строки submission
sample.to_csv("submission.csv", index=False)

