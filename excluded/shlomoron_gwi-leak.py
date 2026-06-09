import numpy as np
import pandas as pd
import os


folders = os.listdir('/kaggle/input/waveform-inversion/train_samples')


seis_path = ['/kaggle/input/waveform-inversion/train_samples/CurveFault_A/seis2_1_0.npy',
             '/kaggle/input/waveform-inversion/train_samples/CurveFault_B/seis6_1_0.npy',
             '/kaggle/input/waveform-inversion/train_samples/CurveVel_A/data/data1.npy',
             '/kaggle/input/waveform-inversion/train_samples/CurveVel_B/data/data1.npy',
             '/kaggle/input/waveform-inversion/train_samples/FlatFault_A/seis2_1_0.npy',
             '/kaggle/input/waveform-inversion/train_samples/FlatFault_B/seis6_1_0.npy',
             '/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data1.npy',
             '/kaggle/input/waveform-inversion/train_samples/FlatVel_B/data/data1.npy',
             '/kaggle/input/waveform-inversion/train_samples/Style_A/data/data1.npy',
             '/kaggle/input/waveform-inversion/train_samples/Style_B/data/data1.npy']

datasets = ['CurveFault_A', 'CurveFault_B', 'CurveVel_A', 'CurveVel_B',
            'FlatFault_A', 'FlatFault_B', 'FlatVel_A', 'FlatVel_B', 'Style_A', 'Style_B']


for i in range(10):
    print('############################')
    print(datasets[i]+' ###############')
    seis = np.load(seis_path[i])
    print(np.min(seis[:,:,-1,:]))
    print(np.max(seis[:,:,-1,:]))


for i in range(10):
    print('############################')
    print(datasets[i]+' ###############')
    seis = np.load(seis_path[i])
    print(np.min(seis[:,:,-2,:]))
    print(np.max(seis[:,:,-2,:]))


test_files = os.listdir('/kaggle/input/waveform-inversion/test')


counter = 0
for i in range(0, 65818, 77):
    seis = np.load('/kaggle/input/waveform-inversion/test/'+test_files[i])
    if (np.min(seis[:,-1,:]) >= 0) & (np.max(seis[:,-1,:]) <= 0):
        counter+=1
print(counter)

