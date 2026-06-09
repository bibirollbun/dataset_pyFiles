# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


sub = pd.read_csv('/kaggle/input/waveform-inversion/sample_submission.csv')
sub.tail()


img = np.load('../input/waveform-inversion/train_samples/FlatFault_A/seis4_1_0.npy')


img.dtype


print(img.shape)
print(3 * 4096)
npshape = 3 * 4096


img = tf.io.read_file('../input/waveform-inversion/train_samples/FlatFault_A/seis4_1_0.npy')


img = tf.io.decode_raw(img, tf.float64)


tensorshape = img.shape[0]


tensorshape - npshape


print(img[16:]) # npy value


dtype=32
remove_len = 1024//dtype
img = img[remove_len:]


!pip install pycbc


#Code by Nayu. T.S.https://www.kaggle.com/nayuts/let-s-visualize-data-to-understand

train_id = "seis4_1_0"
signal_array = np.load(f"../input/waveform-inversion/train_samples/FlatFault_A/{train_id}.npy")


from IPython import display
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


#By Lupin11 https://www.kaggle.com/code/lupin11/weight-maps-for-better-boundaries

def get_ids(tar_path):
    ids = []
    for img_id in os.listdir(tar_path):
        ids.append(img_id)
    print(f"{len(ids)} samples in {tar_path}")
    return ids

tar_path = "/kaggle/input/waveform-inversion/train_samples"
ids = get_ids(tar_path)


#By Lupin11 https://www.kaggle.com/code/lupin11/weight-maps-for-better-boundaries

def gen_weight_map(image, sigma=0.2):
    """
    Generates a weight map based on the ground truth.
    
    Args:
        image (numpy.ndarray): Ground truth, binary image of shape (height, width).
        sigma (float): Controls the range of boundaries.
        
    Returns:
        weight_map (numpy.ndarray): Weight map of the same shape as the ground truth.
    """
    distance = distance_transform_edt(1 - image)
    distance = distance / np.max(distance)
    weight_map = np.exp(-0.5 * (distance / sigma) ** 2)
    weight_map[image == 1] = 0
    return weight_map


#By Lupin11 https://www.kaggle.com/code/lupin11/weight-maps-for-better-boundaries

for img_id in ids[:20]:  # visualize 20 images
    sample_path = f"{tar_path}/{img_id}"
    gt = np.load(f"{sample_path}//seis4_1_0.npy")
    if np.all(gt == 0):  # skip images without contrails
        continue
    weight_map = gen_weight_map(gt)  # generate the weight map
    plt.figure(figsize=(6, 3))
    ax = plt.subplot(1, 2, 1)
    ax.imshow(gt, interpolation='none')
    ax.set_title('GroundTruth')
    ax = plt.subplot(1, 2, 2)
    ax.imshow(weight_map, interpolation='none')
    ax.set_title('WeightMap')
    plt.show()


!python -m pip install gwpy


#By Geir Drange https://www.kaggle.com/mistag/data-preprocessing-with-gwpy

from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot

def read_file(fname):
    data = np.load(fname)
    d1 = TimeSeries(data[0,:], sample_rate=2048)
    d2 = TimeSeries(data[1,:], sample_rate=2048)
    d3 = TimeSeries(data[2,:], sample_rate=2048)
    return d1, d2, d3


#By Geir Drange https://www.kaggle.com/mistag/data-preprocessing-with-gwpy

def create_rgb(fname):
    r1, r2, r3 = read_file(fname)
    p1, p2, p3 = preprocess(r1, r2, r3)
    hq1 = p1.q_transform(outseg=(0, 2))
    hq2 = p2.q_transform(outseg=(0, 2))
    hq3 = p3.q_transform(outseg=(0, 2))
    img = np.zeros([hq1.shape[0], hq1.shape[1], 3], dtype=np.uint8)
    scaler = MinMaxScaler()
    img[:,:,0] = 255*scaler.fit_transform(hq1)
    img[:,:,1] = 255*scaler.fit_transform(hq2)
    img[:,:,2] = 255*scaler.fit_transform(hq3)
    return Image.fromarray(img).rotate(90, expand=1)


create_rgb('../input/waveform-inversion/train_samples/FlatFault_A/seis4_1_0.npy')

