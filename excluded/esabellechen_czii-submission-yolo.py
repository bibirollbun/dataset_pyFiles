!tar xfvz /kaggle/input/czii-ultralytics-packages/archieve.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ,/packages


!cp -r '/kaggle/input/hengck-czii-cryo-et-01/wheel_file' '/kaggle/working/'
!pip install /kaggle/working/wheel_file/asciitree-0.3.3/asciitree-0.3.3
!pip install --no-index --find-links=/kaggle/working/wheel_file zarr


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
warnings.filterwarnings('ignore')


model = YOLO("/kaggle/input/czii-yolo-model/runs/detect/train/weights/best.pt")


runs = sorted(glob.glob('/kaggle/input/czii-cryo-et-object-identification/test/static/ExperimentRuns/*'))
runs = [os.path.basename(x) for x in runs]
runs[:5]


# Information of labels

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

#Radius
particle_radius = {
        'apo-ferritin': 60,
        'beta-amylase': 65,
        'beta-galactosidase': 90,
        'ribosome': 150,
        'thyroglobulin': 130,
        'virus-like-particle': 135,
    }


class PredAggForYOLO:
    def __init__(self,first_conf=0.2, final_conf=0.3,conf_coef=0.75):
        self.first_conf = first_conf
        self.final_conf = final_conf
        self.conf_coef = conf_coef
        self.particle_confs = [0.5,0,0.2,0.5,0.2,0.5]

    def convert_to_8bit(self,x):
        lower,upper = np.percentile(x,(0.5,99.5))
        x = np.clip(x,lower,upper)
        x = (x-x.min())/(x.max()-x.min() + 1e-12) * 255
        return x.round().astype("uint8")

    # depth first search.
    # aggregate the coordinates and confidence scores of connected graphs.
    def dfs(self,v):
        self.passed[v]=True
        self.conf_sum += self.pdf.iloc[v].confidence
        self.cx += self.pdf.iloc[v].x
        self.cy += self.pdf.iloc[v].y
        self.cz += self.pdf.iloc[v].z
        self.nv += 1
        for next_v in self.adjacency_list[v]:
            if (self.passed[next_v]): 
                continue
            self.dfs(next_v)

    def make_predict_yolo(self,r,model):
        vol = zarr.open(f'/kaggle/input/czii-cryo-et-object-identification/test/static/ExperimentRuns/{r}/VoxelSpacing10.000/denoised.zarr',mode='r')
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
            tmp_img = np.zeros((630,630))
            tmp_img[:] = vol2[i]

            inp_arr = np.stack([tmp_img]*3,axis=-1)
            inp_arr = cv2.resize(inp_arr, (640,640))
            res = model.predict(inp_arr, save=False, imgsz=640, conf=self.first_conf, device="0", batch=1, verbose=False)
            for j ,result in enumerate(res):
                boxes = result.boxes # Boxes object for bounding box outputs
                for k in range(len(boxes.cls)):
                    ptype = i2p[boxes.cls.cpu().numpy()[k]] # particle type
                    conf = boxes.conf.cpu().numpy()[k] # confidence score
                    # YOLO can infer (start_x, end_x, start_y, end_y)
                    xc = (boxes.xyxy[k,0] + boxes.xyxy[k,2])/2.0*10*(63/64)
                    yc = (boxes.xyxy[k,1] + boxes.xyxy[k,3])/2.0*10*(63/64)
                    zc = i*10+5

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
        
        df= df.sort_values(['particle_type','z'],ascending=[True,True])
        agg_df=[]
        for pidx,p in enumerate(particle_names):
            if p =='beta-amylase':
                continue
            pdf = df[df['particle_type']==p].reset_index(drop = True)
            self.pdf = pdf
            p_rad = particle_radius[p]
            # The distance between the x and y coordinates of adjacent slices is expected to be very small.
            xy_tol = p_rad/16.0
            xy_tol_p2 = xy_tol**2
            # Define the graph
            self.adjacency_list = [[]for _ in range(len(pdf))]
            # which already passed in dfs
            self.passed = [False for _ in range(len(pdf))]
            #Connect two points when they are close enough
            for i in range(len(pdf)):
                x1=pdf['x'].iloc[i]
                y1=pdf['y'].iloc[i]
                z1=pdf['z'].iloc[i]
                for j in range(i+1,len(pdf),1):
                    x2 = pdf['x'].iloc[j]
                    y2 = pdf['y'].iloc[j]
                    z2 = pdf['z'].iloc[j]
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
                if self.nv >= 2 and self.conf_sum/(self.nv**self.conf_coef)> self.particle_confs[pidx]:
                    cxs.append(self.cx/self.nv)
                    cys.append(self.cy/self.nv)
                    czs.append(self.cz/self.nv)
            
            rdf['experiment'] = [r]* len(cxs)
            rdf['particle_type'] = [p]*len(cys)
            rdf['x'] = cxs
            rdf['y'] = cys
            rdf['z'] = czs
            agg_df.append(rdf)
                        

        return pd.concat(agg_df, axis=0)


# instance main class
agent = PredAggForYOLO(first_conf=0.15, final_conf=0.2, conf_coef=0.5) 
# final_conf is not used after version 14



subs=[]


import time



%%time
tick = time.time()
for r in tqdm(runs,total=len(runs)):
    df =  agent.make_predict_yolo(r, model)
    subs.append(df)
tock = time.time()


print(f'estimated predict time is {(tock-tick)/3*500:.4f} seconds')


submission = pd.concat(subs).reset_index(drop=True)
submission.insert(0, 'id', range(len(submission)))


submission.to_csv("submission.csv", index=False)
submission.head()

