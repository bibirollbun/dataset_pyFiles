import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns

import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import numpy as np


def first_eigenvector_weightage(data):
    wlst = []
    for i in range(len(data)):
        x = data[i][0]
        x = (x-3000)/1000
        U, S, Vt = np.linalg.svd(x)
        wlst.append(
            np.abs(S[0])/np.sum(np.abs(S))
        )
    return wlst


curve_faultb = np.load("/kaggle/input/waveform-inversion/train_samples/CurveFault_B/vel8_1_0.npy")
curve_velb = np.load("/kaggle/input/waveform-inversion/train_samples/CurveVel_B/model/model2.npy")
flatfaultb = np.load("/kaggle/input/waveform-inversion/train_samples/FlatFault_B/vel8_1_0.npy")
flatvelb = np.load("/kaggle/input/waveform-inversion/train_samples/FlatVel_B/model/model2.npy")
styleb = np.load("/kaggle/input/waveform-inversion/train_samples/Style_B/model/model2.npy")

print(curve_velb.shape, curve_faultb.shape)
print()
print(flatfaultb.shape, flatvelb.shape)
print()
print(styleb.shape)


curve_faulta = np.load("/kaggle/input/waveform-inversion/train_samples/CurveFault_A/vel4_1_0.npy")
curve_vela = np.load("/kaggle/input/waveform-inversion/train_samples/CurveVel_A/model/model2.npy")
flatfaulta = np.load("/kaggle/input/waveform-inversion/train_samples/FlatFault_A/vel4_1_0.npy")
flatvela = np.load("/kaggle/input/waveform-inversion/train_samples/FlatVel_A/model/model2.npy")
stylea = np.load("/kaggle/input/waveform-inversion/train_samples/Style_A/model/model2.npy")


curvevelb_weightage = first_eigenvector_weightage(curve_velb)
curve_faultb_weightage = first_eigenvector_weightage(curve_faultb)

flatfaultb_weightage = first_eigenvector_weightage(flatfaultb)
flatvelb_weightage = first_eigenvector_weightage(flatvelb)
styleb_weightage = first_eigenvector_weightage(styleb)


curvevela_weightage = first_eigenvector_weightage(curve_vela)
curve_faulta_weightage = first_eigenvector_weightage(curve_faulta)

flatfaulta_weightage = first_eigenvector_weightage(flatfaulta)
flatvela_weightage = first_eigenvector_weightage(flatvela)
stylea_weightage = first_eigenvector_weightage(stylea)


fig, ax = plt.subplots(1, 3, figsize=(16, 5))

ax[0].hist(flatfaulta_weightage)
ax[1].hist(flatvela_weightage)
ax[2].hist(stylea_weightage)

ax[0].set_title("Flat Fault - A")
ax[1].set_title("Flat Vel-A")
ax[2].set_title("Style-A")

plt.show()


fig, ax = plt.subplots(1, 2, figsize=(16, 5))
ax[0].hist(curvevela_weightage)
ax[1].hist(curve_faulta_weightage)

ax[0].set_title("Curve Vel-A")
ax[1].set_title("Curve Fault-A")

plt.show()



fig, ax = plt.subplots(1, 3, figsize=(16, 5))

ax[0].hist(flatfaultb_weightage)
ax[1].hist(flatvelb_weightage)
ax[2].hist(styleb_weightage)

ax[0].set_title("Flat Fault-B")
ax[1].set_title("Flat Vel-B")
ax[2].set_title("Style-B")

plt.show()


fig, ax = plt.subplots(1, 2, figsize=(16, 5))
ax[0].hist(curvevelb_weightage)
ax[1].hist(curve_faultb_weightage)

ax[0].set_title("Curve Vel-B")
ax[1].set_title("Curve Fault-B")

plt.show()



def eigenvalue_count_by_threshold(data, th=0.85):
    count_list = []
    for i in range(len(data)):
        x = data[i][0]
        x = (x-3000)/1000
        U, S, Vt = np.linalg.svd(x)
        W = np.abs(S)/np.sum(np.abs(S))
        W = np.cumsum(W)
        
        count_list.append( (W<=th).sum()+1 )
        
    return count_list


curvevelb_counts = eigenvalue_count_by_threshold(curve_velb ,0.7)
curve_faultb_counts = eigenvalue_count_by_threshold(curve_faultb,0.7)

flatfaultb_counts = eigenvalue_count_by_threshold(flatfaultb,0.7)
flatvelb_counts = eigenvalue_count_by_threshold(flatvelb,0.7)
styleb_counts = eigenvalue_count_by_threshold(styleb,0.7)


fig, ax = plt.subplots(1, 3, figsize=(16, 5))

ax[0].hist(flatfaultb_counts)
ax[1].hist(flatvelb_counts)
ax[2].hist(styleb_counts)

ax[0].set_title("Flat Fault-B Counts")
ax[1].set_title("Flat Vel-B Counts")
ax[2].set_title("Style-B Counts")

plt.show()


fig, ax = plt.subplots(1, 2, figsize=(16, 5))
ax[0].hist(curvevelb_counts)
ax[1].hist(curve_faultb_counts)

ax[0].set_title("Curve Vel-B Counts")
ax[1].set_title("Curve Fault-B Counts")

plt.show()




