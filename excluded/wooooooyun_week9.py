# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


fpath ='/kaggle/input/siim-acr-pneumothorax-segmentation/stage_2_images/ID_003206608.dcm'
import pydicom as dicom
img_dcm = dicom.dcmread(fpath)
#影像.dcm


img = img_dcm.pixel_array
#dcm讀取圖就用pixel_array


import matplotlib.pyplot as plt
plt.imshow (img,cmap='gray')


img_dcm


img_fpath = '/kaggle/input/chest-ct-segmentation/images/images/ID00007637202177411956430_11.jpg'
mask_fpath = '/kaggle/input/chest-ct-segmentation/masks/masks/ID00007637202177411956430_mask_11.jpg'


import cv2
img = cv2.imread(img_fpath)
img.shape
#cv2 opencv的簡寫，opencv用來讀影像的，顏色係調整
#讀出來即numpy檔，type(img) 


plt.imshow(img)
#讀出來是BGR，會先讀出藍色


img.shape
#512高，512寬，3片pixel疊在一起 (512,512,3)
#其實是立體的東西


mask = cv2.imread(mask_fpath)


plt.imshow(mask[:,:,0])
#（a,b,c）a第一維度 b第二維度 c第三維度，:是全選的意思，c=0代表第一片channel




#如果不是channel要轉成灰階
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


plt.imshow(mask)


#值域分布
np.bincount(mask.flatten())
#flatten把立體的改成平面的


mask[mask<10] =0 #[]中的是條件，當位置小於10！，都變成0，不是數字比大小，是位置
#mask<10 用numpy去選，條件句，對應矩陣的某個位置，true代表數字有<10


#再做一次值域分布
np.bincount(mask.flatten())
#flatten把立體的改成平面的


mask[mask>70] = 255


np.bincount(mask.flatten())


mask[np.logical_and(mask>=10, mask<=70)]=128
# logical_and 兩個都要符合


np.bincount(mask.flatten())


plt.imshow(mask, cmap='gray')


contours,_=cv2.findContours(
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)
# contours等高線圖


len(contours)
# 2 代表抓到2個座標


colors = [(255,0,0),(0,255,0),(0,0,255)]
#順序是BGR
for i, ent in enumerate(contours):
    print (ent.shape)
    cv2.drawContours(img,[ent],-1, colors[i],5)
    # -1                            顏色     粗細


plt.imshow(img)


cv2.contourArea(contours[1])


#找質心
M = cv2.moments(contours[1]) #moments就是要找質心
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
#cx=m10/m00,cy=m01/m00
#找座標
#寫字在圖上
cv2.putText(img, '({},{})'.format(cx,cy),(cx,cy), 
           cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255),2)
plt.imshow(img)
#5:11
#'({},{})'.format(cx,cy)：要顯示的文字內容（這裡是座標）
#(cx, cy)：文字要放的位置（就是質心的位置）
#cv2.FONT_HERSHEY_COMPLEX_SMALL：字型
#1：文字大小（scale）
#(255,255,255)：字的顏色（白色，BGR格式）
#2：字的線條粗細




