!pip install pymap3d
!pip install --upgrade "ipywidgets==7.7.0"  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymap3d as pm
import pymap3d.vincenty as pmv
import os
import glob as gl
import scipy.optimize
from tqdm.auto import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.spatial import distance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import time
from numpy.lib.stride_tricks import sliding_window_view

# Constants
CLIGHT = 299_792_458   # speed of light (m/s)
RE_WGS84 = 6_378_137   # earth semimajor axis (WGS84) (m)
OMGE = 7.2921151467E-5  # earth angular velocity (IS-GPS) (rad/s)


# Satellite selection using carrier frequency error, elevation angle, and C/N0
def satellite_selection(df, column):
    """
    Args:
        df : DataFrame from device_gnss.csv
        column : Column name
    Returns:
        df: DataFrame with eliminated satellite signals
    """
    idx = df[column].notnull()
    idx &= df['CarrierErrorHz'] < 2.0e6  # carrier frequency error (Hz)
    idx &= df['SvElevationDegrees'] > 10.0  # elevation angle (deg)
    idx &= df['Cn0DbHz'] > 15.0  # C/N0 (dB-Hz)
    idx &= df['MultipathIndicator'] == 0 # Multipath flag

    return df[idx]

# Compute line-of-sight vector from user to satellite
def los_vector(xusr, xsat):
    """
    Args:
        xusr : user position in ECEF (m)
        xsat : satellite position in ECEF (m)
    Returns:
        u: unit line-of-sight vector in ECEF (m)
        rng: distance between user and satellite (m)
    """
    u = xsat - xusr
    rng = np.linalg.norm(u, axis=1).reshape(-1, 1)
    u /= rng
    
    return u, rng.reshape(-1)


# Compute Jacobian matrix
def jac_pr_residuals(x, xsat, pr, W):
    """
    Args:
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        pr : pseudorange (m)
        W : weight matrix
    Returns:
        W*J : Jacobian matrix
    """
    u, _ = los_vector(x[:3], xsat)
    J = np.hstack([-u, np.ones([len(pr), 1])])  # J = [-ux -uy -uz 1]

    return W @ J


# Compute pseudorange residuals
def pr_residuals(x, xsat, pr, W):
    """
    Args:
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        pr : pseudorange (m)
        W : weight matrix
    Returns:
        residuals*W : pseudorange residuals
    """
    u, rng = los_vector(x[:3], xsat)

    # Approximate correction of the earth rotation (Sagnac effect) often used in GNSS positioning
    rng += OMGE * (xsat[:, 0] * x[1] - xsat[:, 1] * x[0]) / CLIGHT

    # Add GPS L1 clock offset
    residuals = rng - (pr - x[3])

    return residuals @ W


# Compute Jacobian matrix
def jac_prr_residuals(v, vsat, prr, x, xsat, W):
    """
    Args:
        v : current velocity in ECEF (m/s)
        vsat : satellite velocity in ECEF (m/s)
        prr : pseudorange rate (m/s)
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        W : weight matrix
    Returns:
        W*J : Jacobian matrix
    """
    u, _ = los_vector(x[:3], xsat)
    J = np.hstack([-u, np.ones([len(prr), 1])])

    return np.dot(W, J)


# Compute pseudorange rate residuals
def prr_residuals(v, vsat, prr, x, xsat, W):
    """
    Args:
        v : current velocity in ECEF (m/s)
        vsat : satellite velocity in ECEF (m/s)
        prr : pseudorange rate (m/s)
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        W : weight matrix
    Returns:
        residuals*W : pseudorange rate residuals
    """
    u, rng = los_vector(x[:3], xsat)
    rate = np.sum((vsat-v[:3])*u, axis=1) \
          + OMGE / CLIGHT * (vsat[:, 1] * x[0] + xsat[:, 1] * v[0]
                           - vsat[:, 0] * x[1] - xsat[:, 0] * v[1])

    residuals = rate - (prr - v[3])

    return residuals @ W

# Carrier smoothing of pseudarange
def carrier_smoothing(gnss_df):
    """
    Args:
        df : DataFrame from device_gnss.csv
    Returns:
        df: DataFrame with carrier-smoothing pseudorange 'pr_smooth'
    """
    carr_th = 1.2# carrier phase jump threshold [m] 2->1.5 (best)->1.0
    pr_th =  15.0 # pseudorange jump threshold [m] 20->15

    prsmooth = np.full_like(gnss_df['RawPseudorangeMeters'], np.nan)
    # Loop for each signal
    for (i, (svid_sigtype, df)) in enumerate(gnss_df.groupby(['Svid', 'SignalType'])):
        df = df.replace(
            {'AccumulatedDeltaRangeMeters': {0: np.nan}})  # 0 to NaN

        # Compare time difference between pseudorange/carrier with Doppler
        drng1 = df['AccumulatedDeltaRangeMeters'].diff() - df['PseudorangeRateMetersPerSecond']
        drng2 = df['RawPseudorangeMeters'].diff() - df['PseudorangeRateMetersPerSecond']

        # Check cycle-slip
        slip1 = (df['AccumulatedDeltaRangeState'].to_numpy() & 2**1) != 0  # reset flag
        slip2 = (df['AccumulatedDeltaRangeState'].to_numpy() & 2**2) != 0  # cycle-slip flag
        slip3 = np.fabs(drng1.to_numpy()) > carr_th # Carrier phase jump
        slip4 = np.fabs(drng2.to_numpy()) > pr_th # Pseudorange jump

        idx_slip = slip1 | slip2 | slip3 | slip4
        idx_slip[0] = True

        # groups with continuous carrier phase tracking
        df['group_slip'] = np.cumsum(idx_slip)

        # Psudorange - carrier phase
        df['dpc'] = df['RawPseudorangeMeters'] - df['AccumulatedDeltaRangeMeters']

        # Absolute distance bias of carrier phase
        meandpc = df.groupby('group_slip')['dpc'].mean()
        df = df.merge(meandpc, on='group_slip', suffixes=('', '_Mean'))

        # Index of original gnss_df
        idx = (gnss_df['Svid'] == svid_sigtype[0]) & (
            gnss_df['SignalType'] == svid_sigtype[1])

        # Carrier phase + bias
        prsmooth[idx] = df['AccumulatedDeltaRangeMeters'] + df['dpc_Mean']

    # If carrier smoothing is not possible, use original pseudorange
    idx_nan = np.isnan(prsmooth)
    prsmooth[idx_nan] = gnss_df['RawPseudorangeMeters'][idx_nan]
    gnss_df['pr_smooth'] = prsmooth

    return gnss_df

# Compute distance by Vincenty's formulae
def vincenty_distance(llh1, llh2):
    """
    Args:
        llh1 : [latitude,longitude] (deg)
        llh2 : [latitude,longitude] (deg)
    Returns:
        d : distance between llh1 and llh2 (m)
    """
    d, az = np.array(pmv.vdist(llh1[:, 0], llh1[:, 1], llh2[:, 0], llh2[:, 1]))

    return d


# Compute score
def calc_score(llh, llh_gt):
    """
    Args:
        llh : [latitude,longitude] (deg)
        llh_gt : [latitude,longitude] (deg)
    Returns:
        score : (m)
    """
    d = vincenty_distance(llh, llh_gt)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])

    return score

# GNSS single point positioning using pseudorange
def point_positioning(gnss_df):
    # Add nominal frequency to each signal
    # Note: GLONASS is an FDMA signal, so each satellite has a different frequency
    CarrierFrequencyHzRef = gnss_df.groupby(['Svid', 'SignalType'])[
        'CarrierFrequencyHz'].median()
    gnss_df = gnss_df.merge(CarrierFrequencyHzRef, how='left', on=[
                            'Svid', 'SignalType'], suffixes=('', 'Ref'))
    gnss_df['CarrierErrorHz'] = np.abs(
        (gnss_df['CarrierFrequencyHz'] - gnss_df['CarrierFrequencyHzRef']))

    # Carrier smoothing
    gnss_df = carrier_smoothing(gnss_df)

    # GNSS single point positioning
    utcTimeMillis = gnss_df['utcTimeMillis'].unique()
    nepoch = len(utcTimeMillis)
    x0 = np.zeros(4)  # [x,y,z,tGPSL1]
    v0 = np.zeros(4)  # [vx,vy,vz,dtGPSL1]
    x_wls = np.full([nepoch, 3], np.nan)  # For saving position
    v_wls = np.full([nepoch, 3], np.nan)  # For saving velocity

    # Loop for epochs
    for i, (t_utc, df) in enumerate(tqdm(gnss_df.groupby('utcTimeMillis'), total=nepoch)):
        # Valid satellite selection
        df_pr = satellite_selection(df, 'pr_smooth')
        df_prr = satellite_selection(df, 'PseudorangeRateMetersPerSecond')

        # Corrected pseudorange/pseudorange rate
        pr = (df_pr['pr_smooth'] + df_pr['SvClockBiasMeters'] - df_pr['IsrbMeters'] - 
              df_pr['IonosphericDelayMeters'] - df_pr['TroposphericDelayMeters']).to_numpy()
        prr = (df_prr['PseudorangeRateMetersPerSecond'] + 
               df_prr['SvClockDriftMetersPerSecond']).to_numpy()

        # Satellite position/velocity
        xsat_pr = df_pr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 
                         'SvPositionZEcefMeters']].to_numpy()
        xsat_prr = df_prr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 
                           'SvPositionZEcefMeters']].to_numpy()
        vsat = df_prr[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 
                       'SvVelocityZEcefMetersPerSecond']].to_numpy()

        # Weight matrix for peseudorange/pseudorange rate
        Wx = np.diag(1 / df_pr['RawPseudorangeUncertaintyMeters'].to_numpy())
        Wv = np.diag(1 / df_prr['PseudorangeRateUncertaintyMetersPerSecond'].to_numpy())

        # Robust WLS requires accurate initial values for convergence,
        # so perform normal WLS for the first time
        if len(df_pr) >= 4:
            # Normal WLS
            if np.all(x0 == 0):
                opt = scipy.optimize.least_squares(
                    pr_residuals, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx))
                x0 = opt.x 
            # Robust WLS for position estimation
            opt = scipy.optimize.least_squares(
                 pr_residuals, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx), loss='soft_l1')
            if opt.status < 1 or opt.status == 2:
                 print(f'i = {i} position lsq status = {opt.status}')
            else:
                 x_wls[i, :] = opt.x[:3]
                 x0 = opt.x
                 
        # Velocity estimation
        if len(df_prr) >= 4:
            if np.all(v0 == 0): # Normal WLS
                opt = scipy.optimize.least_squares(
                    prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv))
                v0 = opt.x
            # Robust WLS for velocity estimation
            opt = scipy.optimize.least_squares(
                prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv), loss='soft_l1')
            if opt.status < 1:
                print(f'i = {i} velocity lsq status = {opt.status}')
            else:
                v_wls[i, :] = opt.x[:3]
                v0 = opt.x

    return utcTimeMillis, x_wls, v_wls

# Simple outlier detection and interpolation
def exclude_interpolate_outlier(x_wls, v_wls):
    # Up velocity threshold
    v_up_th = 2.0 # m/s

    # Coordinate conversion
    x_llh = np.array(pm.ecef2geodetic(x_wls[:, 0], x_wls[:, 1], x_wls[:, 2])).T
    v_enu = np.array(pm.ecef2enuv(
        v_wls[:, 0], v_wls[:, 1], v_wls[:, 2], x_llh[0, 0], x_llh[0, 1])).T

    # Up velocity jump detection
    # Cars don't jump suddenly!
    idx_v_out = np.abs(v_enu[:, 2]) > v_up_th
    v_wls[idx_v_out, :] = np.nan
    
    # Interpolate NaNs at beginning and end of array
    x_df = pd.DataFrame({'x': x_wls[:, 0], 'y': x_wls[:, 1], 'z': x_wls[:, 2]})
    x_df = x_df.interpolate(limit_area='outside', limit_direction='both')
    
    # Interpolate all NaN data
    v_df = pd.DataFrame({'x': v_wls[:, 0], 'y': v_wls[:, 1], 'z': v_wls[:, 2]})
    v_df = v_df.interpolate(limit_area='outside', limit_direction='both')
    v_df = v_df.interpolate('spline', order=3)

    return x_df.to_numpy(), v_df.to_numpy()

# Kalman filter
def Kalman_filter(zs, us, phone):
 
    # I don't know why only XiaomiMi8 seems to be inaccurate ... 
    sigma_v = 0.6 if phone == 'XiaomiMi8' else 0.1 # velocity SD m/s
    sigma_x = 5.0  # position SD m
    sigma_mahalanobis = 30.0 # Mahalanobis distance for rejecting innovation
    
    n, dim_x = zs.shape
    F = np.eye(3)  # Transition matrix
    Q = sigma_v**2 * np.eye(3)  # Process noise

    H = np.eye(3)  # Measurement function
    R = sigma_x**2 * np.eye(3)  # Measurement noise

    # Initial state and covariance
    x = zs[0, :3].T  # State
    P = sigma_x**2 * np.eye(3)  # State covariance
    I = np.eye(dim_x)

    x_kf = np.zeros([n, dim_x])
    P_kf = np.zeros([n, dim_x, dim_x])

    # Kalman filtering
    for i, (u, z) in enumerate(zip(us, zs)):
        # First step
        if i == 0:
            x_kf[i] = x.T
            P_kf[i] = P
            continue

        # Prediction step
        x = F @ x + u.T
        P = (F @ P) @ F.T + Q

        # Check outliers for observation
        d = distance.mahalanobis(z, H @ x, np.linalg.pinv(P))

        # Update step
        if d < sigma_mahalanobis:
            y = z.T - H @ x
            S = (H @ P) @ H.T + R
            K = (P @ H.T) @ np.linalg.inv(S)
            x = x + K @ y
            P = (I - (K @ H)) @ P
        else:
            # If no observation update is available, increase covariance
            P += 10**2*Q

        x_kf[i] = x.T
        P_kf[i] = P

    return x_kf, P_kf


# Forward + backward Kalman filter and smoothing
def Kalman_smoothing(x_wls, v_wls, phone):
    n, dim_x = x_wls.shape

    # Forward
    v = np.vstack([np.zeros([1, 3]), (v_wls[:-1, :] + v_wls[1:, :])/2])
    x_f, P_f = Kalman_filter(x_wls, v, phone)

    # Backward
    v = -np.flipud(v_wls)
    v = np.vstack([np.zeros([1, 3]), (v[:-1, :] + v[1:, :])/2])
    x_b, P_b = Kalman_filter(np.flipud(x_wls), v, phone)

    # Smoothing
    x_fb = np.zeros_like(x_f)
    P_fb = np.zeros_like(P_f)
    for (f, b) in zip(range(n), range(n-1, -1, -1)):
        P_fi = np.linalg.inv(P_f[f])
        P_bi = np.linalg.inv(P_b[b])

        P_fb[f] = np.linalg.inv(P_fi + P_bi)
        x_fb[f] = P_fb[f] @ (P_fi @ x_f[f] + P_bi @ x_b[b])

    return x_fb, x_f, np.flipud(x_b)


class PositionLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=16, num_layers=1, output_size=2):
        super(PositionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Fast distance calculation function
def fast_distance(llh1, llh2):
    lat1, lon1 = llh1
    lat2, lon2 = llh2
    avg_lat = np.radians((lat1 + lat2) / 2)
    dx = (lon2 - lon1) * 111000 * np.cos(avg_lat)
    dy = (lat2 - lat1) * 111000
    return np.sqrt(dx**2 + dy**2)

# Preparing LSTM training data
def prepare_data(kf_points, gt_points, seq_length=5):
    """
   Prepare LSTM training data: use Kalman filter output as input sequence and ground truth as target

Args:
kf_points: Kalman filter output latitude and longitude sequence [n, 2]
gt_points: ground truth latitude and longitude sequence [n, 2]
seq_length: sequence length

Returns:
X: input sequence [num_samples, seq_length, 2]
y: target value [num_samples, 2]
    """
    n = kf_points.shape[0]
    X, y = [], []
    
# Create a sliding window sequence (each window predicts the correction value of the center point of the window)
    half_len = seq_length // 2
    
    for i in range(half_len, n - half_len):
        seq = kf_points[i-half_len:i+half_len+1]
        target = gt_points[i]  # Use the current ground_truth as the target
        
        X.append(seq)
        y.append(target)
    
    return np.array(X), np.array(y)

# Optimized LSTM correction function
def optimized_lstm_correction(kf_points, model, scaler, seq_length=5, distance_threshold=3.0):
    """
   Optimized LSTM correction function: Use Kalman filter output as input and apply LSTM correction

Args:
kf_points: Latitude and longitude sequence of Kalman filter output [n, 2]
model: trained LSTM model
scaler: normalizer
seq_length: sequence length
distance_threshold: distance threshold (meters)
        
    Returns:
        corrected: Corrected position sequence [n, 2]
    """
    device = next(model.parameters()).device
    model.eval()
    
    n = kf_points.shape[0]
    corrected = np.copy(kf_points)
    
    if n <= seq_length:
        return corrected
    
   # Create a sliding window sequence (each window predicts the correction value of the center point of the window)
    half_len = seq_length // 2
    sequences = []
    valid_indices = []
    
    for i in range(half_len, n - half_len):
        seq = kf_points[i-half_len:i+half_len+1]
        sequences.append(seq)
        valid_indices.append(i)
    
    sequences = np.array(sequences)
    
# Standardization
    sequences_scaled = scaler.transform(sequences.reshape(-1, 2)).reshape(sequences.shape)
    sequences_tensor = torch.tensor(sequences_scaled, dtype=torch.float32).to(device)
    
# Batch prediction
    batch_size = 16
    predictions_scaled = []
    with torch.no_grad():
        for i in range(0, len(sequences_tensor), batch_size):
            batch = sequences_tensor[i:i+batch_size]
            pred_batch = model(batch)
            predictions_scaled.append(pred_batch.cpu().numpy())
    
    predictions_scaled = np.vstack(predictions_scaled)
    predictions = scaler.inverse_transform(predictions_scaled)
    
# Apply conditional filtering and weighted fusion
    for idx, center_idx in enumerate(valid_indices):
        kf_point = kf_points[center_idx]
        lstm_point = predictions[idx]
        
# Calculate the distance between the Kalman point and the LSTM prediction point
        d = fast_distance(kf_point, lstm_point)
        
        # Distance threshold screening + weighted fusion
        if d < distance_threshold:
            # Adaptive weight: the smaller the distance, the larger the LSTM weight
            alpha = max(0.5, 1.0 - d/distance_threshold)
            corrected[center_idx] = alpha * lstm_point + (1-alpha) * kf_point
    
    return corrected


print("="*50)
print("Step 1/4: Processing training data")
print("="*50)

# Configuration path
train_path =  '/kaggle/input/smartphone-decimeter-2023/sdc2023/train/2023-09-07-22-48-us-ca-routebc2/pixel4xl'
val_path =  '/kaggle/input/smartphone-decimeter-2023/sdc2023/train/2023-09-07-22-47-us-ca-routebc2/pixel6pro'
model_path = 'gnss_lstm_model.pth'
scaler_path = 'gnss_scaler.pkl'

# Process training data

train_gnss_df = pd.read_csv(os.path.join(train_path, 'device_gnss.csv'), low_memory=False)

#train_gnss_df = pd.read_csv(os.path.join(train_path, 'device_gnss.csv'))
train_gt_df = pd.read_csv(os.path.join(train_path, 'ground_truth.csv'))
train_utc, train_x_wls, train_v_wls = point_positioning(train_gnss_df)
    
train_x_wls, train_v_wls = exclude_interpolate_outlier(train_x_wls, train_v_wls)
train_x_kf, _, _ = Kalman_smoothing(train_x_wls, train_v_wls, 'pixel4xl')
train_llh_kf = np.array(pm.ecef2geodetic(train_x_kf[:, 0], train_x_kf[:, 1], train_x_kf[:, 2])).T[:, :2]
train_llh_gt = train_gt_df[['LatitudeDegrees', 'LongitudeDegrees']].to_numpy()

# Processing verification data
val_gnss_df = pd.read_csv(os.path.join(val_path, 'device_gnss.csv'), low_memory=False)
#val_gnss_df = pd.read_csv(os.path.join(val_path, 'device_gnss.csv'))
val_gt_df = pd.read_csv(os.path.join(val_path, 'ground_truth.csv'))
val_utc, val_x_wls, val_v_wls = point_positioning(val_gnss_df)
val_x_wls, val_v_wls = exclude_interpolate_outlier(val_x_wls, val_v_wls)
val_x_kf, _, _ = Kalman_smoothing(val_x_wls, val_v_wls, 'pixel6pro')
val_llh_kf = np.array(pm.ecef2geodetic(val_x_kf[:, 0], val_x_kf[:, 1], val_x_kf[:, 2])).T[:, :2]
val_llh_gt = val_gt_df[['LatitudeDegrees', 'LongitudeDegrees']].to_numpy()


print("=" * 50)
print("Step 2/4: Train the LSTM model")
print("=" * 50)

# Configuration parameters
seq_length = 5
hidden_size = 16
num_layers = 2
num_epochs = 7
batch_size = 16
input_size = 2
output_size = 2

# Prepare training data - use Kalman filter output as input and ground truth as target
X_train, y_train = prepare_data(train_llh_kf, train_llh_gt, seq_length)
X_val, y_val = prepare_data(val_llh_kf, val_llh_gt, seq_length)

# Data Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, input_size)).reshape(X_train.shape)
y_train_scaled = scaler.transform(y_train)
X_val_scaled = scaler.transform(X_val.reshape(-1, input_size)).reshape(X_val.shape)
y_val_scaled = scaler.transform(y_val)

# Save the normalizer
scaler_path = "/kaggle/working/normalizer.pkl"
joblib.dump(scaler, scaler_path)
print(f"Normalizer saved to: {scaler_path}")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device used: {device}")

# Model definition
class PositionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PositionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Instantiate model
model = PositionLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=output_size
).to(device)

# Optimizer and Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
model_path = '/kaggle/working/gnss_lstm_model.pth'
train_losses, val_losses = [], []
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training
    model.train()
    epoch_train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * inputs.size(0)

    # Validation
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item() * inputs.size(0)

    # Compute average losses
    train_loss = epoch_train_loss / len(train_loader.dataset)
    val_loss = epoch_val_loss / len(val_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'hidden_size': hidden_size,
            'num_layers': num_layers
        }, model_path)

    print(f"Epoch {epoch+1}/{num_epochs}: Training loss = {train_loss:.6f}, Validation loss = {val_loss:.6f}")

# View model parameters
print("\nModel weight information:")
print("=" * 50)
for name, param in model.named_parameters():
    print(f"Layer Name: {name}")
    print(f"Shape: {param.shape}")
    print(f"First 5 weight values: {param.data.flatten()[:5].cpu().numpy()}")
    print("-" * 50)

# Draw loss curve
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Training and validation losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('/kaggle/working/loss_curve.png')
plt.show()

print(f"Model saved at: {model_path}")



print("="*50)
print("Step 2/4: Train the LSTM model")
print("="*50)

# Configuration parameters
seq_length = 5
hidden_size = 16
num_layers = 2
num_epochs = 7
batch_size = 16

input_size = 2
output_size = 2

# Prepare training and validation data
X_train, y_train = prepare_data(train_llh_kf, train_llh_gt, seq_length)
X_val, y_val = prepare_data(val_llh_kf, val_llh_gt, seq_length)

# Data Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, input_size)).reshape(X_train.shape)
y_train_scaled = scaler.transform(y_train)
X_val_scaled = scaler.transform(X_val.reshape(-1, input_size)).reshape(X_val.shape)
y_val_scaled = scaler.transform(y_val)

# Save the normalizer
scaler_path = '/kaggle/working/scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Normalizer saved to: {scaler_path}")

# Convert to PyTorch tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

# Dataset and DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model
model = PositionLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=output_size
).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model_path = '/kaggle/working/gnss_lstm_model.pth'
train_losses, val_losses = [], []
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training
    model.train()
    epoch_train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * inputs.size(0)
    
    # Validation
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item() * inputs.size(0)
    
    train_loss = epoch_train_loss / len(train_loader.dataset)
    val_loss = epoch_val_loss / len(val_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'hidden_size': hidden_size,
            'num_layers': num_layers
        }, model_path)
    
    print(f'Epoch {epoch+1}/{num_epochs}: Training loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}')

# Load best model (optional step)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Print model weights
print("\nModel weight information:")
print("="*50)
for name, param in model.named_parameters():
    print(f"Layer Name: {name}")
    print(f"Shape: {param.shape}")
    print(f"First 5 values: {param.data.flatten()[:5].cpu().numpy()}")
    print("-"*50)

# Draw loss curve
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('/kaggle/working/loss_curve.png')
plt.show()


# Configuration
path = '/kaggle/input/smartphone-decimeter-2023/sdc2023'
model_path = 'gnss_lstm_model.pth'
scaler_path = 'gnss_scaler.pkl'
submission_path = 'submission.csv'

# Sample submission
sample_df = pd.read_csv(f'{path}/sample_submission.csv')

# Load model and scaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (make sure these are defined)
hidden_size = 16
num_layers = 2

model = PositionLSTM(
    input_size=2,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=2
).to(device)

# Load model weights
checkpoint = torch.load(model_path, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)
model.eval()

scaler = joblib.load(scaler_path)

# Initialize submission list
test_dfs = []

# Loop through test data
for dirname in tqdm(sorted(gl.glob(f'{path}/test/*/*/')), desc="Processing itinerary"):
    drive, phone = dirname.split('/')[-3:-1]
    tripID = f"{drive}/{phone}"
    print(f"\nProcessing itinerary: {tripID}")

    gnss_df = pd.read_csv(f'{dirname}/device_gnss.csv', low_memory=False)

    # Positioning and preprocessing
    utc, x_wls, v_wls = point_positioning(gnss_df)
    x_wls, v_wls = exclude_interpolate_outlier(x_wls, v_wls)

    try:
        x_kf, _, _ = Kalman_smoothing(x_wls, v_wls, phone)
    except:
        print("Warning: Kalman smoothing failed, using original positions")
        x_kf = x_wls

    # Convert ECEF to Lat-Lon
    llh_kf = np.array(pm.ecef2geodetic(x_kf[:, 0], x_kf[:, 1], x_kf[:, 2])).T[:, :2]

    # Apply LSTM correction
    llh_corrected = optimized_lstm_correction(
        llh_kf, model, scaler, seq_length=5, distance_threshold=3.0
    )

    # Interpolate positions for submission timestamps
    trip_timestamps = sample_df[sample_df['tripId'] == tripID]['UnixTimeMillis'].values

    if len(utc) > 1:
        spline_lat = InterpolatedUnivariateSpline(utc, llh_corrected[:, 0], k=1, ext=3)
        spline_lng = InterpolatedUnivariateSpline(utc, llh_corrected[:, 1], k=1, ext=3)
        lat = spline_lat(trip_timestamps)
        lng = spline_lng(trip_timestamps)
    else:
        lat = np.full(len(trip_timestamps), llh_corrected[0, 0])
        lng = np.full(len(trip_timestamps), llh_corrected[0, 1])

    trip_df = pd.DataFrame({
        'tripId': tripID,
        'UnixTimeMillis': trip_timestamps,
        'LatitudeDegrees': lat,
        'LongitudeDegrees': lng
    })
    test_dfs.append(trip_df)

# Final submission
test_df = pd.concat(test_dfs)
test_df.to_csv(submission_path, index=False)


