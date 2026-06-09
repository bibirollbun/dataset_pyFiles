# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


star = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train_star_info.csv')
star.tail()


train = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train.csv')
train.tail()


axis = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/axis_info.parquet")
axis.tail()


dead = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/train/1010375142/AIRS-CH0_calibration_0/dead.parquet")
dead.tail()


dar = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/train/1024292144/FGS1_calibration_0/dark.parquet")
dar.tail(3)


#Cyberia https://www.kaggle.com/code/cyberia/eda-adc2024/notebook

dark = dar.values.reshape(32, 32)  # Reshape to 2D array 32 columns
plt.matshow(dark)
plt.show()


flat = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/train/1029552010/AIRS-CH0_calibration_0/flat.parquet")
flat.tail(3)


fla = flat.values  
plt.matshow(fla)
plt.show()


linCorr = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/train/1031303815/FGS1_calibration_0/linear_corr.parquet")
linCorr.tail(3)


linear = linCorr.values  
plt.matshow(linear)
plt.show()


read = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/train/104891231/AIRS-CH0_calibration_0/read.parquet")
read.tail(3)


rea = read.values  
plt.matshow(rea)
plt.show()


air = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/train/1029552010/FGS1_signal_0.parquet")
air.tail(3)


AIRS = air.values  
plt.matshow(AIRS)
plt.show()


submission = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/sample_submission.csv')
submission.tail()


wave = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/wavelengths.csv', delimiter=',', encoding='utf-8')
#pd.set_option('display.max_columns', None)
wave.head()


adc = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv', delimiter=',', encoding='utf-8')
#pd.set_option('display.max_columns', None)
adc.tail()


#By Ambros https://www.kaggle.com/code/ambrosm/adc24-intro-training/notebook

planet_id = 1124057774
f_signal = pd.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/FGS1_signal_0.parquet')
f_signal


#By Ambros https://www.kaggle.com/code/ambrosm/adc24-intro-training/notebook

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
sns.heatmap(f_signal.iloc[0].values.reshape(32, 32), ax=ax1, vmin=0, vmax=52000)
ax1.set_aspect('equal')
sns.heatmap(f_signal.iloc[1].values.reshape(32, 32), ax=ax2, vmin=0, vmax=52000)
ax2.set_aspect('equal')
plt.suptitle('A pair of FGS1 images')
plt.show()

