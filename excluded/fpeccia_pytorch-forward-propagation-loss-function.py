import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


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

def AbcCoef2D(vel, nbc, dx):
    """
    Calculates coefficients for a 2D Absorbing Boundary Condition (ABC).
    This is a Python/NumPy translation of the provided MATLAB function.

    Args:
        vel (np.ndarray): The padded 2D velocity model.
        nbc (int): The number of padding cells (boundary width).
        dx (float): The spatial grid interval.

    Returns:
        np.ndarray: A 2D array of damping coefficients.
    """
    nzbc, nxbc = vel.shape
    velmin = np.min(vel)
    nz = nzbc - 2 * nbc
    nx = nxbc - 2 * nbc

    if nbc <= 1:
        return np.zeros_like(vel)

    a = (nbc - 1) * dx
    kappa = 3.0 * velmin * np.log(1e7) / (2.0 * a)

    damp1d = kappa * ((np.arange(nbc) * dx / a) ** 2)
    damp = np.zeros((nzbc, nxbc), dtype=np.float64)

    # Fill left and right damping zones
    for iz in range(nzbc):
        damp[iz, 0:nbc] = damp1d[::-1]
        damp[iz, nx + nbc : nx + 2 * nbc] = damp1d

    # Fill top and bottom damping zones
    for ix in range(nbc, nbc + nx):
        damp[0:nbc, ix] = damp1d[::-1]
        damp[nbc + nz : nz + 2 * nbc, ix] = damp1d
        
    return damp

def padvel(v0, nbc):
    """
    Pads the velocity model by extending the edge values outward.
    """
    return np.pad(v0, pad_width=nbc, mode='edge')

def expand_source(s0, nt):
    """
    Ensures the source time function has length 'nt'.
    """
    nt0 = s0.size
    if nt0 < nt:
        s = np.zeros(nt, dtype=np.float64)
        s[:nt0] = s0
        return s
    else:
        return s0[:nt].astype(np.float64)

def adjust_sr(coord, dx, nbc):
    """
    Converts physical source/receiver coordinates to grid indices.
    """
    # MATLAB's round(x.5) rounds away from zero. NumPy's np.round(x.5)
    # rounds to the nearest even integer. Using np.floor(x + 0.5) for
    # positive numbers emulates MATLAB's behavior.
    round_to_int = lambda x: np.floor(x + 0.5).astype(int)

    isx = round_to_int(coord['sx'] / dx) + nbc
    isz = round_to_int(coord['sz'] / dx) + nbc
    igx = round_to_int(coord['gx'] / dx) + nbc
    igz = round_to_int(coord['gz'] / dx) + nbc

    if np.abs(coord['sz']) < 0.5:
        isz += 1
        
    igz += (np.abs(coord['gz']) < 0.5).astype(int)
    
    return isx, isz, igx, igz

def a2d_mod_abc24(v, nbc, dx, nt, dt, s, coord, isFS):
    """
    Performs a 2D acoustic wave finite-difference simulation (4th order).
    """
    n_receivers = coord['gx'].size
    seis = np.zeros((nt, n_receivers), dtype=np.float64)
    
    c1 = -2.5
    c2 = 4.0 / 3.0
    c3 = -1.0 / 12.0
    
    v_padded = padvel(v, nbc)
    abc = AbcCoef2D(v_padded, nbc, dx)
    
    alpha = (v_padded * dt / dx)**2
    kappa = abc * dt
    temp1 = 2 + 2 * c1 * alpha - kappa
    temp2 = 1 - kappa
    beta_dt = (v_padded * dt)**2
    
    s = expand_source(s, nt)
    isx, isz, igx, igz = adjust_sr(coord, dx, nbc)

    p0 = np.zeros_like(v_padded, dtype=np.float64)
    p1 = np.zeros_like(v_padded, dtype=np.float64)
    
    for it in range(nt):
        laplacian = (
            c2 * (np.roll(p1, 1, axis=1) + np.roll(p1, -1, axis=1) +
                  np.roll(p1, 1, axis=0) + np.roll(p1, -1, axis=0)) +
            c3 * (np.roll(p1, 2, axis=1) + np.roll(p1, -2, axis=1) +
                  np.roll(p1, 2, axis=0) + np.roll(p1, -2, axis=0))
        )
        
        p = temp1 * p1 - temp2 * p0 + alpha * laplacian
        p[isz, isx] += beta_dt[isz, isx] * s[it]
        
        if isFS:
            p[nbc, :] = 0.0
            p[nbc-1, :] = -p[nbc+1, :]
            p[nbc-2, :] = -p[nbc+2, :]

        seis[it, :] = p[igz, igx]
        
        p0 = p1.copy()
        p1 = p.copy()
        
    return seis
    
def vel_to_seis(vel, method='abc24'):
    """
    Runs the simulation for multiple sources and collects the seismograms.
    
    Args:
        vel (np.ndarray): The (70, 70) velocity model.
        method (str): The simulation function to use ('abc24').
        
    Returns:
        np.ndarray: Stacked seismogram data of shape (5, 1001, 70).
    """
    # 1. Model and Simulation Parameters
    nz, nx = vel.shape
    dx = 10.0
    nbc = 120
    nt = 1001
    dt = 1e-3
    freq = 15.0
    isFS = False  # Use free surface condition or not

    # 2. Generate Ricker wavelet source
    s, _ = ricker(freq, dt)
    
    # 3. Setup Receiver Coordinates
    # Receivers are placed at every grid point horizontally at a fixed depth.
    coord = {}
    coord['sz'] = 1 * dx
    coord['gx'] = np.arange(nx) * dx
    coord['gz'] = np.ones(nx) * dx
    
    # 4. Loop over source positions and run simulation
    seis_data = []
    source_x_locations = [0, 17, 34, 52, 69] # Using 0-based indices now
    
    for sx_idx in source_x_locations:
        coord['sx'] = sx_idx * dx
        
        if method == 'abc24':
            seis = a2d_mod_abc24(vel, nbc, dx, nt, dt, s, coord, isFS)
        else:
            raise ValueError(f"Invalid method: {method}")

        seis_data.append(seis)
        
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
    def __init__(self, freq=15, nbc=120, dx=10, dt=1e-3, nt=1001, nx=70, isFS=False):
        super(WavePropagationLoss, self).__init__()
        self.nbc = nbc
        self.dx = dx
        self.dt = dt
        self.nt = nt
        self.nx = nx
        self.isFS = isFS
        self.s0 = torch.from_numpy(self.ricker(freq, dt))

    def ricker(self, f, dt):
        nw = int(2.2 / f / dt)
        nw = 2 * (nw // 2) + 1
        nc = nw // 2 + 1  # 중심 인덱스를 1-based 기준으로 설정
    
        k = np.arange(1, nw + 1)  # 1-based index
        alpha = (nc - k) * f * dt * np.pi
        beta = alpha ** 2
        w0 = (1.0 - 2.0 * beta) * np.exp(-beta)
    
        # 1-based wavelet 생성
        w = np.zeros(len(w0) + 1)
        w[1:] = w0
    
        return w
    
    def forward(self, v_batch, target_seis_batch):
        """
        v: torch.Tensor [batch, nz, nx]  (velocity model)
        target_seis: torch.Tensor [batch, 5, nt+1, n_receivers] (observed seismograms)

        Returns:
            loss: scalar tensor
        """
        batch_size = v_batch.shape[0]
        source_x_idxs = [0, 17, 34, 52, 69]
        losses = []

        s = self.s0.to(v_batch.device).unsqueeze(0).repeat(batch_size,1)

        torch_seis_data = []
        for j in range(5):
            target_seis = target_seis_batch[:,j]

            coord = {}
            coord['sz'] = 1 * self.dx
            coord['gx'] = np.arange(0, self.nx) * self.dx
            coord['gz'] = np.ones_like(coord['gx']) * self.dx
            coord['sx'] = source_x_idxs[j] * self.dx 
                
            pred_seis = self.simulate_batch(v_batch, s, coord)
            torch_seis_data.append(pred_seis[:, 1:,:])
            losses.append(F.l1_loss(pred_seis[:, 1:,:], target_seis[:,:,:]))
        loss = torch.stack(losses).mean()
        return loss, torch.cat(torch_seis_data)
    
    def simulate_batch(self, v_batch, s_batch, coord):
        batch_size, nz, nx = v_batch.shape
        ng = len(coord['gx'])
        device = v_batch.device

        # Prepare constants
        c1 = -2.5
        c2 = 4.0 / 3.0
        c3 = -1.0 / 12.0
        #c1 = -205.0/72.0 
        #c2 = 8.0/5.0
        #c3 = -1.0/5.0
        #c4 = 8.0/315.0
        #c5 = -1.0/560.0;

        # Pad velocity
        v_batch = self.padvel(v_batch)  # [batch_size, nz_p, nx_p]
        abc_batch = self.AbcCoef2D(v_batch)

        alpha = (v_batch * self.dt / self.dx) ** 2
        kappa = abc_batch * self.dt
        temp1 = 2 + 2 * c1 * alpha - kappa
        temp2 = 1 - kappa
        beta_dt = (v_batch * self.dt) ** 2

        s_batch = self.expand_source_batch(s_batch)  # [batch_size, nt+1]
        isx, isz, igx, igz = self.adjust_sr(device, coord)

        nz_p, nx_p = v_batch.shape[-2], v_batch.shape[-1]

        p0 = torch.zeros_like(v_batch)
        p1 = torch.zeros_like(v_batch)
        seis_batch = torch.zeros((batch_size, self.nt, ng), device=device)

        # Build Laplacian kernel for conv2d: shape (1, 1, 3, 3)
        #laplace_kernel = torch.tensor([[0, c2, 0],
        #                               [c2, c1 * 4, c2],
        #                               [0, c2, 0]], device=device).view(1, 1, 3, 3)
    
        # Expand to batch: group conv
        #laplace_kernel = laplace_kernel.repeat(batch_size, 1, 1, 1)  # (B, 1, 3, 3)
    
        # Reshape for conv2d: (B, 1, Z, X)
        #def laplacian(u):
        #    return F.conv2d(u.unsqueeze(1), laplace_kernel, padding=1, groups=batch_size).squeeze(1)
            
        for it in range(self.nt):
            # This for loop is the bottleneck.
            # The laplacian calculated using convolutions is actually quite slow, so this idea was discarded. 
            #lap_u = self.laplacian_9pt(p1)
            #p = temp1 * p1 - temp2 * p0 + alpha * lap_u

            p = (temp1 * p1 - temp2 * p0 +
                 alpha * (
                     c2 * (torch.roll(p1, 1, dims=2) + torch.roll(p1, -1, dims=2) +
                           torch.roll(p1, 1, dims=1) + torch.roll(p1, -1, dims=1)) +
                     c3 * (torch.roll(p1, 2, dims=2) + torch.roll(p1, -2, dims=2) +
                           torch.roll(p1, 2, dims=1) + torch.roll(p1, -2, dims=1))
                     #c4 * (torch.roll(p1, 3, dims=2) + torch.roll(p1, -3, dims=2) +
                     #      torch.roll(p1, 3, dims=1) + torch.roll(p1, -3, dims=1)) +
                     #c5 * (torch.roll(p1, 4, dims=2) + torch.roll(p1, -4, dims=2) +
                     #      torch.roll(p1, 4, dims=1) + torch.roll(p1, -4, dims=1))
                 ))

            # Source injection (vectorized)
            p[torch.arange(batch_size), isz, isx] += beta_dt[torch.arange(batch_size), isz, isx] * s_batch[:, it]

            if self.isFS:
                p[:, self.nbc, :] = 0.0
                p[:, self.nbc-1:self.nbc+1, :] = -p[:, self.nbc+1:self.nbc+3, :]

            # Receiver sampling (vectorized)
            #print(torch.max(p[torch.arange(batch_size).unsqueeze(1), igz.unsqueeze(0), igx.unsqueeze(0)].view(-1)))
            #seis_batch[:, it, :] = p[torch.arange(batch_size).unsqueeze(1), igz.unsqueeze(0), igx.unsqueeze(0)]
            for ig in range(ng):
                seis_batch[:, it, ig] = p[torch.arange(batch_size).unsqueeze(1), igz[ig], igx[ig]]
            # Record receivers: vectorized gather
            #batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, G)  # (B, G)
            #seis[:, it, :] = p[batch_idx, igz, igx]

            p0, p1 = p1, p

        return seis_batch

    #def padvel(self, v0):
    #    v_padded = torch.squeeze(F.pad(torch.unsqueeze(v0,0), (self.nbc, self.nbc, self.nbc, self.nbc), mode='replicate'))
    #    nz, nx = v_padded.shape
    #    v = torch.zeros((nz + 1, nx + 1), device=v0.device, dtype=v0.dtype)
    #    v[1:, 1:] = v_padded
    #    return v

    def padvel(self, v_batch):
        # Pad with replicate mode
        v_padded = F.pad(v_batch, (self.nbc, self.nbc, self.nbc, self.nbc), mode='replicate')
        #batch_size, nz_p, nx_p = v_padded.shape
        #v = torch.zeros((batch_size, nz_p + 1, nx_p + 1), device=v_batch.device, dtype=v_batch.dtype)
        #v[:, 1:, 1:] = v_padded
        return v_padded

    #def expand_source(self, s0):
    #    s0 = torch.as_tensor(s0, dtype=torch.float32, device=s0.device if isinstance(s0, torch.Tensor) else 'cpu').flatten()
    #    s = torch.zeros(self.nt + 1, device=s0.device, dtype=s0.dtype)
    #    s[1:len(s0) + 1] = s0
    #    return s

    def expand_source_batch(self, s_batch):
        batch_size = s_batch.shape[0]
        device = s_batch.device
        s = torch.zeros((batch_size, self.nt), device=device)
        for b in range(batch_size):
            ns = s_batch[b].numel()
            s[b, :ns] = s_batch[b].flatten()
        return s

    def adjust_sr(self, device, coord):
        def round_away_from_zero(x):
            return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)
    
        #device = self.dx.device if isinstance(self.dx, torch.Tensor) else 'cpu'
        sx = torch.tensor(coord['sx'], dtype=torch.float32, device=device)
        sz = torch.tensor(coord['sz'], dtype=torch.float32, device=device)
        gx = torch.tensor(coord['gx'], dtype=torch.float32, device=device)
        gz = torch.tensor(coord['gz'], dtype=torch.float32, device=device)

        isx = round_away_from_zero(sx / self.dx).int() + self.nbc
        isz = round_away_from_zero(sz / self.dx).int() + self.nbc
        igx = (round_away_from_zero(gx / self.dx) + self.nbc).int()
        igz = (round_away_from_zero(gz / self.dx) + self.nbc).int()

        if torch.abs(sz) < 0.5:
            isz += 1
        igz += (torch.abs(gz) < 0.5).int()

        return isx, isz, igx, igz

    def AbcCoef2D(self, vel_batch):
        batch_size = vel_batch.shape[0]
        nzbc, nxbc = vel_batch.shape[2], vel_batch.shape[1]
        #velmin = torch.min(vel_batch[:, 1:, 1:], dim=(1,2)).values
        #velmin = vel_batch[:, 1:, 1:].view(-1,nzbc*nxbc).min(dim=1).values
        velmin = vel_batch.reshape(batch_size, -1).min(dim=1).values
        nz = nzbc - 2 * self.nbc
        nx = nxbc - 2 * self.nbc

        a = (self.nbc - 1) * self.dx
        kappa = 3.0 * velmin * torch.log(torch.tensor(1e7, device=vel_batch.device)) / (2.0 * a)

        damp1d = []
        for k in kappa:
            d = k * ((torch.arange(0, self.nbc, device=vel_batch.device) * self.dx / a) ** 2)
            damp1d.append(d)
        damp1d = torch.stack(damp1d)  # [batch_size, nbc]
        
        #damp1d = kappa * (((torch.arange(1, self.nbc + 1, device=vel.device, dtype=vel.dtype) - 1) * self.dx / a) ** 2)
        damp = torch.zeros((batch_size, nzbc, nxbc), device=vel_batch.device, dtype=vel_batch.dtype)

        for iz in range(nzbc):
            damp[:, iz, :self.nbc] = torch.flip(damp1d, dims=[1])
            damp[:, iz, nx + self.nbc : nx + 2 * self.nbc] = damp1d

        for ix in range(self.nbc, self.nbc + nx):
            damp[:, :self.nbc, ix] = torch.flip(damp1d, dims=[1])
            damp[:, nz + self.nbc : nz + 2 * self.nbc, ix] = damp1d

        return damp


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
    plot_seis(seis_data_sim[:, 1:, :])
    
    print("\n< INPUT (origin) >")
    plot_seis(seis_data[0])
    
    print("\n< ERROR >")
    errors = []
    for j in range(5):
        error = np.mean(np.abs(seis_data[0, j, :, :] - seis_data_sim[j, 1:, :])) # MAE
        errors.append(error)
        print(f"Receiver{j+1} Error : {error:.6f}")

    loss_fn = WavePropagationLoss()#, timesteps_to_compare=100)
    vel_data_torch = torch.from_numpy(vel_data[0]).to('cpu')
    seis_data_torch = torch.from_numpy(seis_data[:1,:,:,:]).to('cpu')
    torch_error, torch_seis_data = loss_fn(vel_data_torch, seis_data_torch)

    print(f"\n< TORCH FORWARD SIMULATION (from velocity map) : shape={torch_seis_data.shape} >")
    plot_seis(torch_seis_data)

    print(f"Mean Error : {np.mean(errors):.6f} - Mean Torch Error : {torch_error:.6f}")
    print("#########################################")
    print()

    # if i==2:
    #     break
    # break

