import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# https://arxiv.org/pdf/2111.02926
# https://csim.kaust.edu.sa/files/SeismicInversion/Chapter.FD/lab.FD2.8/lab.html

# no modification
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
    ax[0].imshow(seis[0, :, :],extent=[0,70,1000,0],aspect='auto',cmap='gray',vmin=-0.5,vmax=0.5)
    ax[1].imshow(seis[1, :, :],extent=[0,70,1000,0],aspect='auto',cmap='gray',vmin=-0.5,vmax=0.5)
    ax[2].imshow(seis[2, :, :],extent=[0,70,1000,0],aspect='auto',cmap='gray',vmin=-0.5,vmax=0.5)
    ax[3].imshow(seis[3, :, :],extent=[0,70,1000,0],aspect='auto',cmap='gray',vmin=-0.5,vmax=0.5)
    ax[4].imshow(seis[4, :, :],extent=[0,70,1000,0],aspect='auto',cmap='gray',vmin=-0.5,vmax=0.5)
    for axis in ax:
        axis.set_xticks(range(0, 70, 10))
        axis.set_xticklabels(range(0, 700, 100))
        axis.set_yticks(range(0, 2000, 1000))
        axis.set_yticklabels(range(0, 2,1))
        axis.set_ylabel('Time (s)', fontsize=12)
        axis.set_xlabel('Offset (m)', fontsize=12)
    plt.show()


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
    print(f"Mean Error : {np.mean(errors):.6f}")
    print("#########################################")
    print()

    # if i==2:
    #     break
    # break

