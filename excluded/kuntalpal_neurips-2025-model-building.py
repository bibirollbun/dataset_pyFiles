import numpy as np
from sklearn.linear_model import LinearRegression



# Calibrated Pixel Cube
def calibrate_cube(raw_cube, gain, offset, dark_map, flat_map, lin_corr_map):
    """
    Restore calibrated pixel values.
    
    Parameters
    ----------
    raw_cube : np.ndarray
        Raw pixel cube, shape (T, H, W), dtype uint16.
    gain : float
        Detector gain.
    offset : float
        Detector offset.
    dark_map : np.ndarray
        Dark current + bias map, shape (H, W).
    flat_map : np.ndarray
        Flat-field response map, shape (H, W).
    lin_corr_map : np.ndarray
        Linearity correction factors, shape (H, W).
    
    Returns
    -------
    I_cal : np.ndarray
        Calibrated pixel cube, shape (T, H, W), dtype float.
    """
    # Recover dynamic range
    I = raw_cube.astype(float) * gain + offset
    # Subtract dark current, divide flat field
    I = (I - dark_map[None, :, :]) / flat_map[None, :, :]
    # Apply linearity correction
    I *= lin_corr_map[None, :, :]
    return I

# Transit-Phase Mask (approximate)
def transit_phase_mask(times, P, sma, inc_deg):
    """
    Compute in-transit and out-of-transit masks without external packages.
    
    Parameters
    ----------
    times : np.ndarray
        Time stamps of observations, shape (T,), same units as P.
    P : float
        Orbital period (days).
    sma : float
        Semi-major axis in units of stellar radii (R*).
    inc_deg : float
        Orbital inclination in degrees.
    
    Returns
    -------
    in_transit : np.ndarray
        Boolean mask for in-transit frames, shape (T,).
    oot : np.ndarray
        Boolean mask for out-of-transit frames, shape (T,).
    """
    # Compute orbital phase in [0,1)
    phase = ((times - times[0]) / P) % 1.0
    
    # Approximate full transit duration (T14) using small-planet formula
    i = np.deg2rad(inc_deg)
    b = sma * np.cos(i)  # impact parameter
    arg = np.clip(np.sqrt(max(0, (1.0) - b**2)) / (sma * np.sin(i)), -1, 1)
    T14 = (P / np.pi) * np.arcsin(arg)
    
    # Fractional duration of transit
    frac = T14 / P
    
    # Center transit at phase = 0.5
    in_transit = (phase >= 0.5 - frac/2) & (phase <= 0.5 + frac/2)
    oot = ~in_transit
    return in_transit, oot

# Example usage (replace raw_cube, times, star parameters as needed):
# I_cal = calibrate_cube(raw_cube, gain, offset, dark_map, flat_map, lin_corr_map)
# in_tr, oot = transit_phase_mask(times, P, sma, inclination)


# Frame-level (pixel-domain) statistics
def frame_stats(I_cal, in_transit, oot):
    """
    Compute per-frame and aggregated statistics:
      - Total flux
      - Centroids x_c, y_c
      - Centroid scatter
      - Background mean (border pixels)
      - Spatial RMS
      - Flux derivative
    
    Aggregates mean, std, skew for IN and OOT segments.
    
    Parameters
    ----------
    I_cal : np.ndarray, shape (T, H, W)
        Calibrated pixel cube.
    in_transit : np.ndarray, bool, shape (T,)
        In-transit mask.
    oot : np.ndarray, bool, shape (T,)
        Out-of-transit mask.
    
    Returns
    -------
    stats : dict
        Feature dictionary with keys like 'F_IN_mean', 'F_OOT_std', etc.
    """
    T, H, W = I_cal.shape
    flat = I_cal.reshape(T, -1)
    
    # 1. Total flux
    F = flat.sum(axis=1)
    
    # 2. Centroids
    ys, xs = np.indices((H, W))
    x_c = (I_cal * xs[None]).sum(axis=(1,2)) / F
    y_c = (I_cal * ys[None]).sum(axis=(1,2)) / F
    
    # 3. Background mean (outer border)
    border = np.concatenate([
        I_cal[:, 0, :], I_cal[:, -1, :], 
        I_cal[:, :, 0], I_cal[:, :, -1]
    ], axis=0).reshape(4, T, -1)
    B = border.mean(axis=(0,2))
    
    # 4. Spatial RMS
    med = np.median(flat, axis=1)
    RMS = np.sqrt(((flat - med[:, None])**2).mean(axis=1))
    
    # 5. Flux derivative
    dotF = np.concatenate(([0], np.diff(F)))  # assume uniform Δt
    
    # Helper for aggregation
    def agg(arr):
        return arr.mean(), arr.std(), skew(arr)
    
    stats = {}
    for name, arr in [('F', F),
                      ('x_c', x_c),
                      ('y_c', y_c),
                      ('B', B),
                      ('RMS', RMS),
                      ('dotF', dotF)]:
        m_in, s_in, k_in = agg(arr[in_transit])
        m_oot, s_oot, k_oot = agg(arr[oot])
        stats[f'{name}_IN_mean'] = m_in
        stats[f'{name}_IN_std']  = s_in
        stats[f'{name}_IN_skew'] = k_in
        stats[f'{name}_OOT_mean'] = m_oot
        stats[f'{name}_OOT_std']  = s_oot
        stats[f'{name}_OOT_skew'] = k_oot
    
    return stats


# Example usage:
# pixel_feats = frame_stats(I_cal, in_tr, oot)




def spectro_lightcurve_features(I_cal_airs, I_cal_fgs, in_transit, oot, d_phys_airs=None, N_beta=10):
    """
    Extract spectrophotometric light-curve features for AIRS-CH0 and FGS1.
    
    Parameters
    ----------
    I_cal_airs : np.ndarray, shape (T, H, W)
        Calibrated cube for AIRS-CH0.
    I_cal_fgs : np.ndarray, shape (T, H, W')
        Calibrated cube for FGS1 (H'=W'=32).
    in_transit : np.ndarray, bool, shape (T,)
        In-transit mask.
    oot : np.ndarray, bool, shape (T,)
        Out-of-transit mask.
    d_phys_airs : np.ndarray or None, shape (W,)
        Analytic physics depth per AIRS channel. If None, zeros used.
    N_beta : int
        Bin size for beta factor calculation.
    
    Returns
    -------
    feats : dict
        Dictionary containing per-channel arrays and aggregated stats:
        - 'd_emp': empirical depth array for AIRS shape (W,)
        - 'r': residual array for AIRS shape (W,)
        - 'a': white-trend slopes array for AIRS shape (W,)
        - 'sigma_corr': correlated-noise RMS array shape (W,)
        - 'beta': Pont beta-factor array shape (W,)
        - '<feat>_mean', '<feat>_std' for each feat in [d_emp, r, a, sigma_corr, beta]
        - 'd_emp_fgs', 'r_fgs' for FGS1 empirical depth & residual
    """
    T = I_cal_airs.shape[0]
    # 1) Light curves
    L_airs = I_cal_airs.sum(axis=1)              # shape (T, W)
    L_fgs  = I_cal_fgs.sum(axis=(1,2))           # shape (T,)
    
    # 2) Analytic physics depth
    W = L_airs.shape[1]
    if d_phys_airs is None:
        d_phys_airs = np.zeros(W)
    
    # 3) Empirical depth & residual for AIRS
    med_oot = np.median(L_airs[oot], axis=0)
    med_in  = np.median(L_airs[in_transit], axis=0)
    d_emp   = (med_oot - med_in) / med_oot
    r       = d_emp - d_phys_airs
    
    # 4) Trend & correlated noise per channel
    t = np.arange(T)[:, None]
    a      = np.zeros(W)
    sigma_corr = np.zeros(W)
    beta   = np.zeros(W)
    for i in range(W):
        # slope fit on OOT
        lr = LinearRegression().fit(t[oot], L_airs[oot, i])
        a[i] = lr.coef_[0]
        resid = L_airs[:, i] - lr.predict(t)
        sigma_corr[i] = np.sqrt(np.mean(resid[oot]**2))
        # beta factor
        M = (T // N_beta) * N_beta
        binned = resid[:M].reshape(-1, N_beta).mean(axis=1)
        beta[i] = np.std(binned) / np.std(resid[:M])
    
    # 5) Empirical depth & residual for FGS1
    med_oot_fgs = np.median(L_fgs[oot])
    med_in_fgs  = np.median(L_fgs[in_transit])
    d_emp_fgs   = (med_oot_fgs - med_in_fgs) / med_oot_fgs
    r_fgs       = d_emp_fgs  # if no physics model for FGS, residual = empirical
    
    # 6) Aggregates
    def agg(arr):
        return arr.mean(), arr.std()
    
    feats = {
        'd_emp': d_emp,
        'r': r,
        'a': a,
        'sigma_corr': sigma_corr,
        'beta': beta,
        'd_emp_fgs': d_emp_fgs,
        'r_fgs': r_fgs,
    }
    
    for name in ['d_emp', 'r', 'a', 'sigma_corr', 'beta']:
        m, s = agg(feats[name])
        feats[f'{name}_mean'] = m
        feats[f'{name}_std']  = s
    
    return feats

# Example usage:
# feats_spec = spectro_lightcurve_features(I_cal_airs, I_cal_fgs, in_tr, oot, d_phys_airs=None)




