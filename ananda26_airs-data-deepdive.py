import pandas as pd
import numpy as np

import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from astropy.stats import sigma_clip


# Functions

def plot_signal(signal):

    if signal.ndim == 3:
        # visualize the first timestep
        sample0 = signal[0]  # shape (32, 282)
        sample1 = signal[1]  # shape (32, 282)
    else:
        # Visulaize the whole image
        sample0 = signal  # shape (32, 282)
        sample1 = signal  # shape (32, 282)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot index 0
    im0 = axes[0].imshow(sample0, aspect='auto', cmap='viridis')
    axes[0].set_title("Index 0")
    plt.colorbar(im0, ax=axes[0], shrink=0.7)
    
    # Plot index 1
    im1 = axes[1].imshow(sample1, aspect='auto', cmap='viridis')
    axes[1].set_title("Index 1")
    plt.colorbar(im1, ax=axes[1], shrink=0.7)
    
    plt.tight_layout()
    plt.show()


input_path = r'/kaggle/input/ariel-data-challenge-2025/'

image_id = 1010375142

df = pd.read_parquet(input_path + f'/train/{image_id}/AIRS-CH0_signal_0.parquet')

axis_info = pd.read_parquet(input_path + 'axis_info.parquet')
adc_info = pd.read_csv(input_path + 'adc_info.csv')


axis_info.info()


df


axis_info[axis_info['AIRS-CH0-axis0-h'].notnull()]


diff_axis0 = axis_info['AIRS-CH0-axis0-h'].diff()
diff_axis0


cut_inf, cut_sup = 39, 321
l = cut_sup - cut_inf

signal = df.values.reshape((df.shape[0], 32, 356))
print(signal.shape)

chopped_signal = signal[:, :, cut_inf:cut_sup]
print(chopped_signal.shape)


plot_signal(signal=chopped_signal)


# Define a function to perform analog to dgital conversion.

def ADC_convertion(signal, gain, offset):
    '''
    Function to convert a signal from Analog to Digital
    '''
    signal = signal / gain
    signal = signal + offset

    return signal

gain = adc_info['AIRS-CH0_adc_gain'].loc[0]
offset = adc_info['AIRS-CH0_adc_offset'].loc[0]

cleaned_signal = ADC_convertion(chopped_signal, gain, offset)


plot_signal(signal=cleaned_signal)


dead_airs = pd.read_parquet(input_path + f'train/{image_id}/AIRS-CH0_calibration_0/dead.parquet').values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]

dark = pd.read_parquet(input_path + f'train/{image_id}/AIRS-CH0_calibration_0/dark.parquet').values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]


## Visulaize the dead_airs and dark pixels

plot_signal(signal=dead_airs)


plot_signal(signal=dark)


print(dead_airs.shape)
print(dark.shape)


AIRS_CH0_clean = np.ma.zeros((1, 11250, 32, l))


# Masking of hot/dead pixels

def mask_hot_dead_signal(signal, dead, dark):

    # determine the hot and dead pixels.
    hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    
    # Expand the hot pixel mask to match the number of frames in signal
    hot = np.tile(hot, (signal.shape[0], 1, 1))  
    
    # Expand dead pixel mask too
    dead = np.tile(dead, (signal.shape[0], 1, 1))  
    
    # Apply dead pixel mask
    signal = np.ma.masked_where(dead, signal)
    
    # Apply hot pixel mask
    signal = np.ma.masked_where(hot, signal)
    
    return signal

cleaned_signal = mask_hot_dead_signal(cleaned_signal, dead_airs, dark)

AIRS_CH0_clean[0] = cleaned_signal


plot_signal(signal=AIRS_CH0_clean[0])


linear_corr_raw = pd.read_parquet(input_path + f'train/{image_id}/AIRS-CH0_calibration_0/linear_corr.parquet').values.astype(np.float64).reshape((6, 32, 356))[:, :, cut_inf:cut_sup]
linear_corr_raw.shape


AIRS_CH0_clean[0].shape


# Applying linearity correction

def apply_linear_corr(linear_corr,clean_signal):
    linear_corr = np.flip(linear_corr, axis=0)
    for x, y in itertools.product(
                range(clean_signal.shape[1]), range(clean_signal.shape[2])
            ):
        poli = np.poly1d(linear_corr[:, x, y])
        clean_signal[:, x, y] = poli(clean_signal[:, x, y])
    return clean_signal


linear_corr_signal = apply_linear_corr(linear_corr=linear_corr_raw,
                                   clean_signal=AIRS_CH0_clean[0])

AIRS_CH0_clean[0] = linear_corr_signal


plot_signal(AIRS_CH0_clean[0])


dark = pd.read_parquet(input_path + f'train/{image_id}/AIRS-CH0_calibration_0/dark.parquet').values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]


dark.shape


dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values


dt_airs


def clean_dark(signal, dead, dark, dt):

    dark = np.ma.masked_where(dead, dark)
    dark = np.tile(dark, (signal.shape[0], 1, 1))

    signal -= dark* dt[:, np.newaxis, np.newaxis]
    return signal


AIRS_CH0_clean[0] = clean_dark(AIRS_CH0_clean[0],dead_airs, dark, dt_airs)


plot_signal(AIRS_CH0_clean[0])


def get_cds(signal):
    cds = signal[:,1::2,:,:] - signal[:,::2,:,:]
    return cds


AIRS_CH0_clean[0].shape


AIRS_cds = get_cds(AIRS_CH0_clean)


AIRS_cds.shape


plot_signal(AIRS_cds[0])


AIRS_cds.shape


flat_airs = pd.read_parquet(input_path + f'train/{image_id}/AIRS-CH0_calibration_0/flat.parquet').values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]


flat_airs.shape


dead_airs = pd.read_parquet(input_path + f'train/{image_id}/AIRS-CH0_calibration_0/dead.parquet').values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]


dead_airs.shape


def correct_flat_field(flat,dead, signal):
    flat = flat.transpose(1, 0)
    dead = dead.transpose(1, 0)
    flat = np.ma.masked_where(dead, flat)
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    signal = signal / flat
    return signal


AIRS_cds = AIRS_cds.transpose(0,1,3,2)
AIRS_cds_binned = AIRS_cds

corrected_AIRS_cds_binned = correct_flat_field(flat_airs,dead_airs, AIRS_cds_binned[0])
AIRS_cds_binned[0] = corrected_AIRS_cds_binned


plot_signal(AIRS_cds_binned[0])


for i in range(len(AIRS_cds_binned)) : 
    light_curve = AIRS_cds_binned[i,:,:,:].sum(axis=(1,2))
    plt.plot(light_curve/light_curve.mean(), '-', alpha=0.3)

plt.xlabel('Time (frame index)')
plt.ylabel('Normalized flux in the frame')


#

