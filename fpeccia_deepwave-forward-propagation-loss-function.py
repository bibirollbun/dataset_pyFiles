!pip install deepwave


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import deepwave


# https://arxiv.org/pdf/2111.02926
# https://csim.kaust.edu.sa/files/SeismicInversion/Chapter.FD/lab.FD2.8/lab.html

def ricker(f, dt, nt=None):
    nw = int(2.2 / f / dt)
    nw = 2 * (nw // 2) + 1
    nc = nw // 2 + 1  # 중심 인덱스를 1-based 기준으로 설정

    k = np.arange(1, nw + 1)  # 1-based index
    alpha = (nc - k) * f * dt * np.pi
    beta = alpha ** 2
    w0 = (1.0 - 2.0 * beta) * np.exp(-beta)

    # 1-based wavelet 생성
    if nt is not None:
        if nt < len(w0):
            raise ValueError("nt is smaller than condition!")
        w = np.zeros(nt + 1)  # dummy 포함
        w[1:len(w0) + 1] = w0
    else:
        w = np.zeros(len(w0) + 1)
        w[1:] = w0

    # 1-based time axis 생성
    if nt is not None:
        tw = np.arange(1, len(w)) * dt
    else:
        tw = np.arange(1, len(w)) * dt

    return w, tw

def padvel(v0, nbc):
    v_padded = np.pad(v0, ((nbc, nbc), (nbc, nbc)), mode='edge')
    nz, nx = v_padded.shape
    v = np.zeros((nz + 1, nx + 1))
    v[1:, 1:] = v_padded
    return v
	
def expand_source(s0, nt):
    s0 = np.asarray(s0).flatten()
    s = np.zeros(nt + 1)
    s[1:len(s0) + 1] = s0
    return s
	
def adjust_sr(coord, dx, nbc):
    isx = int(round(coord['sx'] / dx)) + 1 + nbc
    isz = int(round(coord['sz'] / dx)) + 1 + nbc
    igx = (np.round(np.array(coord['gx']) / dx) + 1 + nbc).astype(int)
    igz = (np.round(np.array(coord['gz']) / dx) + 1 + nbc).astype(int)

    if abs(coord['sz']) < 0.5:
        isz += 1
    igz = igz + (np.abs(np.array(coord['gz'])) < 0.5).astype(int)
    return isx, isz, igx, igz
	
def AbcCoef2D(vel, nbc, dx):
    nzbc, nxbc = vel.shape[1] - 1, vel.shape[0] - 1  # 실제 사이즈
    velmin = np.min(vel[1:, 1:])
    nz = nzbc - 2 * nbc
    nx = nxbc - 2 * nbc

    a = (nbc - 1) * dx
    kappa = 3.0 * velmin * np.log(1e7) / (2.0 * a)

    damp1d = kappa * (((np.arange(1, nbc + 1) - 1) * dx / a) ** 2)
    damp = np.zeros((nzbc + 1, nxbc + 1))

    for iz in range(1, nzbc + 1):
        damp[iz, 1:nbc + 1] = damp1d[::-1]
        damp[iz, nx + nbc + 1 : nx + 2 * nbc + 1] = damp1d

    for ix in range(nbc + 1, nbc + nx + 1):
        damp[1:nbc + 1, ix] = damp1d[::-1]
        damp[nz + nbc + 1 : nz + 2 * nbc + 1, ix] = damp1d

    return damp
	
def a2d_mod_abc24(v, nbc, dx, nt, dt, s, coord, isFS):
    ng = len(coord['gx'])
    seis = np.zeros((nt + 1, ng))  # 1-based time axis

    c1 = -2.5
    c2 = 4.0 / 3.0
    c3 = -1.0 / 12.0

    v = padvel(v, nbc)
    abc = AbcCoef2D(v, nbc, dx)

    alpha = (v * dt / dx) ** 2
    kappa = abc * dt
    temp1 = 2 + 2 * c1 * alpha - kappa
    temp2 = 1 - kappa
    beta_dt = (v * dt) ** 2
    s = expand_source(s, nt)
    isx, isz, igx, igz = adjust_sr(coord, dx, nbc)

    p0 = np.zeros_like(v)
    p1 = np.zeros_like(v)

    # Time Loop (1-based)
    for it in range(1, nt + 1):
        p = (temp1 * p1 - temp2 * p0 +
             alpha * (
                 c2 * (np.roll(p1, 1, axis=1) + np.roll(p1, -1, axis=1) +
                       np.roll(p1, 1, axis=0) + np.roll(p1, -1, axis=0)) +
                 c3 * (np.roll(p1, 2, axis=1) + np.roll(p1, -2, axis=1) +
                       np.roll(p1, 2, axis=0) + np.roll(p1, -2, axis=0))
             ))

        # Source
        p[isz, isx] += beta_dt[isz, isx] * s[it]

        # Free Surface
        if isFS:
            p[nbc, :] = 0.0
            p[nbc - 1 : nbc + 1, :] = -p[nbc + 1 : nbc + 3, :]

        for ig in range(ng):
            seis[it, ig] = p[igz[ig], igx[ig]]

        p0, p1 = p1, p

    return seis
    
def vel_to_seis(vel):
    """
    vel : (70, 70)
    output : (5, 1001, 70)
    """
    # 1. 모델 및 파라미터 설정
    nz = 70
    nx = 70
    dx = 10
    nbc = 120
    nt = 1001
    dt = 1e-3
    freq = 15
    isFS = False  # 자유표면 사용 여부
    
    # 2. Ricker 파형 생성
    s, _ = ricker(freq, dt)
    
    # 3. 소스 및 수신기 설정
    coord = {}
    coord['sz'] = 1 * dx
    coord['gx'] = np.arange(1, nx + 1) * dx
    coord['gz'] = np.ones_like(coord['gx']) * dx
    
    # 4. 파동장 시뮬레이션 수행 : 소스 위치만 바꿔서
    seis_data = []
    for source_x in [1, 18, 35, 53, 70]:
        coord['sx'] = source_x * dx        
        
        # 시뮬레이션
        seis = a2d_mod_abc24(vel, nbc, dx, nt, dt, s, coord, isFS)

        seis_data += [seis]
        
    return np.stack(seis_data, axis=0)
    
    
def plot_seis(seis):
    """
    seis : (5, 1000, 70)
    """
    fig,ax=plt.subplots(1,5,figsize=(20,5))
    timesteps = seis.shape[1]
    ax[0].imshow(seis[0, :, :],extent=[0,70,timesteps,0],aspect='auto',cmap='gray',vmin=-0.5,vmax=0.5)
    ax[1].imshow(seis[1, :, :],extent=[0,70,timesteps,0],aspect='auto',cmap='gray',vmin=-0.5,vmax=0.5)
    ax[2].imshow(seis[2, :, :],extent=[0,70,timesteps,0],aspect='auto',cmap='gray',vmin=-0.5,vmax=0.5)
    ax[3].imshow(seis[3, :, :],extent=[0,70,timesteps,0],aspect='auto',cmap='gray',vmin=-0.5,vmax=0.5)
    ax[4].imshow(seis[4, :, :],extent=[0,70,timesteps,0],aspect='auto',cmap='gray',vmin=-0.5,vmax=0.5)
    for axis in ax:
        axis.set_xticks(range(0, 70, 10))
        axis.set_xticklabels(range(0, 700, 100))
        axis.set_yticks(range(0, timesteps*2, timesteps))
        axis.set_yticklabels(range(0, timesteps*2, timesteps))
        axis.set_ylabel('Time (s)' if timesteps > 500 else 'Time (ms)', fontsize=12)
        axis.set_xlabel('Offset (m)', fontsize=12)
    plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F

class WavePropagationLoss(nn.Module):
    def __init__(self, freq=15, dx=10, dt=1e-3, nt=1001, nx=70, debug=False):
        super(WavePropagationLoss, self).__init__()
        self.dx = dx
        self.dt = dt
        self.nt = nt
        self.nx = nx
        self.debug = debug

        rickers = np.zeros((5, 1, nt + 1), dtype=np.float32)
        for i in range(5):
            rickers[i,0,:148] = deepwave.wavelets.ricker(freq, 148, dt, 74*dt)

        self.rickers = torch.from_numpy(rickers)
        
        receiver_locations = np.zeros((5,nx,2))
        for source in range(5):
            for i in range(nx):
                receiver_locations[source,i,:] = [0,i]

        self.receiver_locations = torch.from_numpy(receiver_locations)
        
        source_locations = np.zeros((5,1,2))
        source_locations[0,0,:] = [0,0]
        source_locations[1,0,:] = [0,17]
        source_locations[2,0,:] = [0,34]
        source_locations[3,0,:] = [0,52]
        source_locations[4,0,:] = [0,69]

        self.source_locations = torch.from_numpy(source_locations)

    
    def forward(self, v_batch, target_seis_batch):
        """
        v: torch.Tensor [batch, nz, nx]  (velocity model)
        target_seis: torch.Tensor [batch, 5, nt+1, n_receivers] (observed seismograms)

        Returns:
            loss: scalar tensor
        """
        batch_size = v_batch.shape[0]

        if self.debug:
            torch_seis_data = []
            
        losses = []
        
        for batch in range(batch_size):
            target_seis = target_seis_batch[batch]

            pred_seis = -deepwave.scalar(
                v_batch[batch,0], grid_spacing=self.dx, dt=1e-3,
                source_amplitudes=self.rickers,
                source_locations=self.source_locations,
                receiver_locations=self.receiver_locations
            )[-1].transpose(2,1)

            if self.debug:
                torch_seis_data.append(pred_seis[:, 2:,:])
            losses.append(F.l1_loss(pred_seis[:, 2:,:], target_seis))
        loss = torch.stack(losses).mean()
        if self.debug:
            return loss, torch.cat(torch_seis_data)
        else:
            return loss



seis_files = [
    "/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data1.npy",
    "/kaggle/input/waveform-inversion/train_samples/FlatVel_B/data/data1.npy",
    
    "/kaggle/input/waveform-inversion/train_samples/CurveVel_A/data/data1.npy",
    "/kaggle/input/waveform-inversion/train_samples/CurveVel_B/data/data1.npy",

    "/kaggle/input/waveform-inversion/train_samples/FlatFault_A/seis2_1_0.npy",
    "/kaggle/input/waveform-inversion/train_samples/FlatFault_B/seis6_1_0.npy",
    
    "/kaggle/input/waveform-inversion/train_samples/CurveFault_A/seis2_1_0.npy",
    "/kaggle/input/waveform-inversion/train_samples/CurveFault_B/seis6_1_0.npy",

    "/kaggle/input/waveform-inversion/train_samples/Style_A/data/data1.npy",
    "/kaggle/input/waveform-inversion/train_samples/Style_B/data/data1.npy",
    
]
vel_files = [
    "/kaggle/input/waveform-inversion/train_samples/FlatVel_A/model/model1.npy",
    "/kaggle/input/waveform-inversion/train_samples/FlatVel_B/model/model1.npy",
    
    "/kaggle/input/waveform-inversion/train_samples/CurveVel_A/model/model1.npy",
    "/kaggle/input/waveform-inversion/train_samples/CurveVel_B/model/model1.npy",

    "/kaggle/input/waveform-inversion/train_samples/FlatFault_A/vel2_1_0.npy",
    "/kaggle/input/waveform-inversion/train_samples/FlatFault_B/vel6_1_0.npy",
    
    "/kaggle/input/waveform-inversion/train_samples/CurveFault_A/vel2_1_0.npy",
    "/kaggle/input/waveform-inversion/train_samples/CurveFault_B/vel6_1_0.npy",

    "/kaggle/input/waveform-inversion/train_samples/Style_A/model/model1.npy",
    "/kaggle/input/waveform-inversion/train_samples/Style_B/model/model1.npy",
    
]
types = [
    'FlatVel_A', 'FlatVel_B', 'CurveVel_A', 'CurveVel_B', 'FlatFault_A', 'FlatFault_B', 'CurveFault_A', 'CurveFault_B', 'Style_A', 'Style_B'
]


%%time
for i, tp in enumerate(types):
    seis_data = np.load(seis_files[i])
    vel_data = np.load(vel_files[i])
    seis_data_sim = vel_to_seis(vel_data[0][0])
    print(f"TYPE : {tp}")
    print("\n< VELOCITY MAP >")
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    ax.imshow(vel_data[0, 0])
    plt.show()

    print(f"\n< FORWARD SIMULATION (from velocity map) : shape={seis_data_sim.shape} >")
    plot_seis(seis_data_sim[:, 2:, :])
    
    print("\n< INPUT (origin) >")
    plot_seis(seis_data[0])
    
    print("\n< ERROR >")
    errors = []
    for j in range(5):
        error = np.mean(np.abs(seis_data[0, j, :, :] - seis_data_sim[j, 2:, :])) # MAE
        errors.append(error)
        print(f"Receiver{j+1} Error : {error:.6f}")

    loss_fn = WavePropagationLoss(debug=True)
    vel_data_torch = torch.from_numpy(vel_data[:1,:,:,:]).to('cpu')
    seis_data_torch = torch.from_numpy(seis_data[:1,:,:,:]).to('cpu')
    torch_error, torch_seis_data = loss_fn(vel_data_torch, seis_data_torch)

    print(f"\n< DEEPWAVE FORWARD SIMULATION (from velocity map) : shape={torch_seis_data.shape} >")
    plot_seis(torch_seis_data)

    print(f"Mean Error : {np.mean(errors):.6f} - Mean DeepWave Error : {torch_error:.6f}")
    print("#########################################")
    print()

    # if i==2:
    #     break
    # break

