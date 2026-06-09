import numpy as np
import pandas as pd


data1 = pd.read_csv('/kaggle/input/d/irakozekelly/fertilizer-prediction/Fertilizer Prediction.csv')
data1 # original dataset of this competition


data2 = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
data2


data3 = pd.read_csv('/kaggle/input/fertilizer-prediction-agriculture/synthetic_fertilizer_data.csv')
data3


data4 = pd.read_csv('/kaggle/input/crop-yield-prediction-dataset/data_core_with_yield.csv').drop(columns='Crop Yield')
data4 = data4.iloc[99:] # Rows of number 0~98 are included in data2.
data4


data1_list = np.array(data1).tolist()
data2_list = np.array(data2).tolist()
data3_list = np.array(data3).tolist()
data4_list = np.array(data4).tolist()


# Comfirm that they don't have the same content.
for data in data1_list:
    if data in data2_list:
        print('They have a same content!:1,2')
    if data in data3_list:
        print('They have a same content!:1,3')
    if data in data4_list:
        print('They have a same content!:1,4')

for data in data2_list:
    if data in data3_list:
        print('They have a same content!:2,3')
    if data in data4_list:
        print('They have a same content!:2,4')

for data in data3_list:
    if data in data4_list:
        print('They have a same content!:3,4')

