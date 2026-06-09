import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


path = Path('/kaggle/input/waveform-inversion')
seis = np.load(path/'train_samples/CurveFault_A/seis2_1_0.npy')
vel  = np.load(path/'train_samples/CurveFault_A/vel2_1_0.npy').squeeze()


plt.imshow(vel[0, :, :]); plt.axis('off');


vel.shape


fig, axs = plt.subplots(3, 3, figsize=(6, 6), constrained_layout=True)
np.random.seed(0)
for ax in axs.flatten():
    i = np.random.randint(0, vel.shape[0])
    ax.imshow(vel[i, :, :])
    ax.axis('off')
    ax.text(0.05, .95, f'Example  {i}', transform=ax.transAxes, color='white', ha='left', va='top', bbox={"facecolor":'black'})


plt.imshow(vel[0, :, :]); plt.axis('off');


fig, ax = plt.subplots(1,1, figsize=(12, 3))
ax.plot(seis[0, 1, :, 34])


fig, axs = plt.subplots(7, 1, figsize=(12, 10),sharex=True, sharey=True)
for i, ax in enumerate(axs):
    ax.plot(seis[0, 1, :, i*10])
    ax.text(0.01, .95, f'Receivers  {i*10}', transform=ax.transAxes,ha='left', va='top')


plt.imshow(seis[0, 1, :, :].T, cmap='seismic')


fig, ax = plt.subplots(figsize=(12,3))
ax.imshow(seis[0, 1, :, :].T, aspect='auto', cmap='seismic', extent=[0, seis.shape[2], seis.shape[3], 0])


fig, ax = plt.subplots(figsize=(3, 5))
ax.imshow(seis[0, 1, :, :], aspect='auto', cmap='seismic', extent=[0, seis.shape[3], seis.shape[2], 0])


fig, axes = plt.subplots(1, seis.shape[1], figsize=(4 * seis.shape[1], 5)) 

for i, ax in enumerate(axes.flatten()):
    ax.imshow(seis[0, i, :, :], aspect='auto', cmap='seismic', extent=[0, seis.shape[3], seis.shape[2], 0])


seis.shape

