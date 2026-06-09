# Reduced dataset containing ROIs of 4346 DICOM


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


roi_df = pd.read_csv('/kaggle/input/rsna-raw-roi/rsna_roi.csv')
roi_df.tail()


N = 50
source_path = '/kaggle/input/rsna-raw-roi/NPZ/'
volumes = roi_df.case
samples =  np.random.permutation(np.arange(len(volumes)))[:N]
v = [volumes[k] for k in samples]
plt.imshow(np.arange(14).reshape(1,14),cmap='turbo')
plt.gca().get_yaxis().set_visible(False)
plt.xticks(ticks=np.arange(14), labels=np.arange(14))
plt.show() 
for k in range(len(v)):
    print(v[k])
    npz = np.load(source_path + v[k] + '.npz')
    print(pd.DataFrame({
        'AP':npz['AP'],
        'loc':npz['loc'],
        'z':npz['t'][:,0],
        'y':npz['t'][:,1],
        'x':npz['t'][:,2]
    }).sort_values('loc').reset_index(drop=True))
#   ROI
    volume = npz['volume']
#   Segmentation quality checks
    YX = npz['YX']
    ZY = npz['ZY']
    ZX = npz['ZX']
#   Legend and cmap sanity check
    YX[0,:14] = ZY[0,:14] = ZX[0,:14] = np.arange(14)
#   Visualization
    _, axs = plt.subplots(2, 3)
    axs[0,0].imshow(volume.max(0))
    axs[0,1].imshow(volume.max(1))
    axs[0,2].imshow(volume.max(2))
    axs[1,0].imshow(YX,cmap='turbo')
    axs[1,1].imshow(ZX,cmap='turbo')
    axs[1,2].imshow(ZY,cmap='turbo')
    plt.show()

