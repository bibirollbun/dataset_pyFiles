!git clone https://github.com/ffyyytt/EAR-WACV25-DAKiet-TSM.git
%cd EAR-WACV25-DAKiet-TSM


!git clone https://huggingface.co/fdfyaytkt/ear-wacv25-tsm-rgb-resnext50_32x4d


import os
import cv2
import glob
import math

import pandas as pd
from tqdm import *

def mp4_to_jpg(filename, idx):
    os.makedirs(f"data/test_img/{idx}")
    vidcap = cv2.VideoCapture(filename)
    c = 0
    while True:
        success, image = vidcap.read()
        if not success:
            return c
        c += 1
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(f"data/test_img/{idx}/{c:06d}.jpg", image)


os.makedirs(f"data/")
allfiles = sorted( glob.glob("/kaggle/input/elderly-action-recognition-challenge-at-wacv-2025/eval_FO_ids/*") )
with open("data/classInd.txt", "w") as f:
    f.write("\n".join(map(str, range(6))))

step = 100
for i in range(math.ceil(len(allfiles)/step)):
    files = allfiles[i*step: (i+1)*step]
    txtfile = []
    for file in tqdm(files):
        idx = file.split("/")[-1][:-4]
        txtfile.append([idx, mp4_to_jpg(file, idx), 0])
    with open("data/test_img/test.txt", "w") as f:
        f.write("\n".join([" ".join(map(str, s)) for s in txtfile]))

    os.popen(f"python generate_submission.py elderly --arch=resnext50_32x4d --csv_file=submission_{i}.csv  --weights=/kaggle/working/EAR-WACV25-DAKiet-TSM/ear-wacv25-tsm-rgb-resnext50_32x4d/TSM_elderly_RGB_resnext50_32x4d_shift8_blockres_avg_segment8.tar --test_segments=8 --batch_size=1 --test_crops=1").read()
    os.popen("rm -rf data/test_img/").read()


csvfiles = sorted(glob.glob("submission_*.csv"), key = lambda x: int( x[x.find("_")+1:x.find(".")] ))
video_name = []
action_category = []
for file in csvfiles:
    df = pd.read_csv(file)
    video_name += df["video_name"].values.tolist()
    action_category += df["action_category"].values.tolist()

df = pd.DataFrame()
df["video_name"] = video_name
df["action_category"] = action_category


%cd ..


df.to_csv("submission.csv", index=False)

