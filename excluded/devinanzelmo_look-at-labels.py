# The main thing to point out is that while only single slice is indicated as labeled
# multiple slices contain aneurysm. This can introduce label noise if too many/not enough slices are given
# positive label for each series. The number of slices with visible aneurism is different between series.


import polars as pl
import glob
import os 
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json


tr = pl.read_csv("../input/rsna-intracranial-aneurysm-detection/train.csv")
trl = pl.read_csv("../input/rsna-intracranial-aneurysm-detection/train_localizers.csv")


def plot_around_label(row_number, num_around, trl=trl):
    row = trl[row_number]
    a,b,c,d = row[:,:4].transpose()[:,0].to_list()
    
    folders = glob.glob("../input/rsna-intracranial-aneurysm-detection/series/*")
    
    for i in folders:
        if a in i:
            break
    files = sorted(glob.glob(os.path.join(i, "*")))
    i_n = list()
    for e, f in enumerate(files):
        ex = pydicom.dcmread(f)
        i_n.append([ex.InstanceNumber,f])
    
    i_n = sorted(i_n)
    
    for e, f in enumerate(i_n):
        if b in f[1]:
            break
    
    c = json.loads(c.replace("'",'"'))
    

    for i in range(e-num_around,e+num_around+1):
        if i ==e:
            print("labeled slice")
        ex = pydicom.dcmread(i_n[i][1])
        fig, ax = plt.subplots()
        if len(ex.pixel_array.shape) > 2:
            continue
        ax.imshow(ex.pixel_array)
        rect = patches.Rectangle((c["x"]-10, c["y"]-10), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()
    
   


# as can be seen the aneurysm is visible on several nearby slices. 
plot_around_label(55,4)


# some nearby slices do not contain the aneurysm, and its not always clear which ones do. 
plot_around_label(8,4)




