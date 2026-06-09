!pip install dicomsdl


import dicomsdl
import pydicom
import os
import pandas
from random import choice
import numpy as np


data_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection'
base_series_dir = os.path.join(data_dir,'series')


sample_series = choice(os.listdir(base_series_dir))
series_dir = os.path.join(base_series_dir,sample_series)
sample_file = os.path.join(series_dir,choice(os.listdir(series_dir)))
print(sample_file)


%%timeit
ds = pydicom.dcmread(sample_file)
pixels_pydicom =ds.pixel_array


%%timeit
ds = dicomsdl.open(sample_file)
pixels_dicomsdl = ds.pixelData()


ds = pydicom.dcmread(sample_file)
pixels_pydicom =ds.pixel_array
ds = dicomsdl.open(sample_file)
pixels_dicomsdl = ds.pixelData()

assert np.allclose(pixels_pydicom,pixels_dicomsdl)




