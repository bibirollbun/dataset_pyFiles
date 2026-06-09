import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.stats import sigma_clip



input_path = r'/kaggle/input/ariel-data-challenge-2025/'
output_path = r'/kaggle/working/tmp/data_processed/'


if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f'Directory made : {output_path}')
else:
    print(f'Directory exists : {output_path}')


# Understanding the dataset.

# 1. adc Info.
adc_info = pd.read_csv(input_path + 'adc_info.csv')
# 2. axis Info.
axis_info = pd.read_parquet(input_path + 'axis_info.parquet')
# 3. Sample submission
sample_submission = pd.read_csv(input_path + 'sample_submission.csv')
#4. test_star_info
test_star_info = pd.read_csv(input_path + 'test_star_info.csv')
#5. train data sample
train_data_sample = pd.read_csv(input_path + 'train.csv')
#7. test_star_info
train_star_info = pd.read_csv(input_path + 'train_star_info.csv')
#8. Wavelength
wavelength = pd.read_csv(input_path + 'wavelengths.csv')


# Define a function to perform analog to doigotal conversion.

def ADC_convertion(signal, gain, offset):
    '''
    Function to convert a signal from Analog to Digital
    '''
    signal = signal / gain
    signal = signal + offset

    return signal


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
    


def apply_linear_corr(linear_corr,clean_signal):
    linear_corr = np.flip(linear_corr, axis=0)
    for x, y in itertools.product(
                range(clean_signal.shape[1]), range(clean_signal.shape[2])
            ):
        poli = np.poly1d(linear_corr[:, x, y])
        clean_signal[:, x, y] = poli(clean_signal[:, x, y])
    return clean_signal


def clean_dark(signal, dead, dark, dt):

    dark = np.ma.masked_where(dead, dark)
    dark = np.tile(dark, (signal.shape[0], 1, 1))

    signal -= dark* dt[:, np.newaxis, np.newaxis]
    return signal


def get_cds(signal):
    cds = signal[:,1::2,:,:] - signal[:,::2,:,:]
    return cds


def correct_flat_field(flat,dead, signal):
    flat = flat.transpose(1, 0)
    dead = dead.transpose(1, 0)
    flat = np.ma.masked_where(dead, flat)
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    signal = signal / flat
    return signal


def bin_obs(cds_signal,binning):
    cds_transposed = cds_signal.transpose(0,1,3,2)
    cds_binned = np.zeros((cds_transposed.shape[0], cds_transposed.shape[1]//binning, cds_transposed.shape[2], cds_transposed.shape[3]))
    for i in range(cds_transposed.shape[1]//binning):
        cds_binned[:,i,:,:] = np.sum(cds_transposed[:,i*binning:(i+1)*binning,:,:], axis=1)
    return cds_binned


adc_info


axis_info


wavelength


do_mask = True
do_nl_corr = True
do_dark = True
do_flat = True
time_binning = True

cut_inf, cut_sup = 39, 321
l = cut_sup - cut_inf

AIRS_CH0_clean = np.zeros((1, 11250, 32, l))
FGS1_clean = np.zeros((1, 135000, 32, 32))

image_id = 1010375142

df = pd.read_parquet(input_path + f'/train/{image_id}/AIRS-CH0_signal_0.parquet')
signal = df.values.reshape((df.shape[0], 32, 356))

gain = adc_info['AIRS-CH0_adc_gain'].loc[0]
offset = adc_info['AIRS-CH0_adc_offset'].loc[0]

signal = ADC_convertion(signal, gain, offset)

dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values

chopped_signal = signal[:, :, cut_inf:cut_sup]

del signal, df


# Read the necessary data for cleaning

flat = pd.read_parquet(input_path + f'train/{image_id}/AIRS-CH0_calibration_0/flat.parquet').values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
dark = pd.read_parquet(input_path + f'train/{image_id}/AIRS-CH0_calibration_0/dark.parquet').values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
dead_airs = pd.read_parquet(input_path + f'train/{image_id}/AIRS-CH0_calibration_0/dead.parquet').values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
linear_corr = pd.read_parquet(input_path + f'train/{image_id}/AIRS-CH0_calibration_0/linear_corr.parquet').values.astype(np.float64).reshape((6, 32, 356))[:, :, cut_inf:cut_sup]


if do_mask:
    chopped_signal = mask_hot_dead_signal(chopped_signal, dead_airs, dark)
    AIRS_CH0_clean[0] = chopped_signal
else:
    AIRS_CH0_clean[0] = chopped_signal

if do_nl_corr: 
    linear_corr_signal = apply_linear_corr(linear_corr,AIRS_CH0_clean[0])
    AIRS_CH0_clean[0] = linear_corr_signal
del linear_corr

if do_dark: 
    cleaned_signal = clean_dark(AIRS_CH0_clean[0], dead_airs, dark,dt_airs)
    AIRS_CH0_clean[0] = cleaned_signal
else: 
    pass
del dark


# FGS

df = pd.read_parquet(os.path.join(input_path + f'train/{image_id}/FGS1_signal_0.parquet'))
fgs_signal = df.values.reshape((df.shape[0], 32, 32))

FGS1_gain = adc_info['FGS1_adc_gain'].loc[0]
FGS1_offset = adc_info['FGS1_adc_offset'].loc[0]

fgs_signal = ADC_convertion(fgs_signal, FGS1_gain, FGS1_offset)

dt_fgs1 = np.ones(len(fgs_signal))*0.1  ## please refer to data documentation for more information

chopped_FGS1 = fgs_signal

del fgs_signal, df


# CLEANING THE DATA: FGS1
flat = pd.read_parquet(input_path + f'train/{image_id}/FGS1_calibration_0/flat.parquet').values.astype(np.float64).reshape((32, 32))
dark = pd.read_parquet(input_path + f'train/{image_id}/FGS1_calibration_0/dark.parquet').values.astype(np.float64).reshape((32, 32))
dead_fgs1 = pd.read_parquet(input_path + f'train/{image_id}/FGS1_calibration_0/dead.parquet').values.astype(np.float64).reshape((32, 32))
linear_corr = pd.read_parquet(input_path + f'train/{image_id}/FGS1_calibration_0/linear_corr.parquet').values.astype(np.float64).reshape((6, 32, 32))


if do_mask:
    chopped_FGS1 = mask_hot_dead_signal(chopped_FGS1, dead_fgs1, dark)
    FGS1_clean[0] = chopped_FGS1
else:
    FGS1_clean[0] = chopped_FGS1

if do_nl_corr: 
    linear_corr_signal = apply_linear_corr(linear_corr,FGS1_clean[0])
    FGS1_clean[0,:, :, :] = linear_corr_signal
del linear_corr

if do_dark: 
    cleaned_signal = clean_dark(FGS1_clean[0], dead_fgs1, dark,dt_fgs1)
    FGS1_clean[0] = cleaned_signal
else: 
    pass
del dark 


AIRS_cds = get_cds(AIRS_CH0_clean)
FGS1_cds = get_cds(FGS1_clean)

del AIRS_CH0_clean, FGS1_clean


# Save the data before binning if necessary


## (Optional) Time Binning to reduce space
if time_binning:
    AIRS_cds_binned = bin_obs(AIRS_cds,binning=30)
    FGS1_cds_binned = bin_obs(FGS1_cds,binning=30*12)
else:
    AIRS_cds = AIRS_cds.transpose(0,1,3,2) ## this is important to make it consistent for flat fielding, but you can always change it
    AIRS_cds_binned = AIRS_cds
    FGS1_cds = FGS1_cds.transpose(0,1,3,2)
    FGS1_cds_binned = FGS1_cds

del AIRS_cds, FGS1_cds


flat_airs = pd.read_parquet(input_path + f'train/{image_id}/AIRS-CH0_calibration_0/flat.parquet').values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
flat_fgs = pd.read_parquet(input_path + f'train/{image_id}/FGS1_calibration_0/flat.parquet').values.astype(np.float64).reshape((32, 32))
if do_flat:
    corrected_AIRS_cds_binned = correct_flat_field(flat_airs,dead_airs, AIRS_cds_binned[0])
    AIRS_cds_binned[0] = corrected_AIRS_cds_binned
    corrected_FGS1_cds_binned = correct_flat_field(flat_fgs,dead_fgs1, FGS1_cds_binned[0])
    FGS1_cds_binned[0] = corrected_FGS1_cds_binned
else:
    pass


# Save the sample data




for i in range(len(AIRS_cds_binned)) : 
    light_curve = AIRS_cds_binned[i,:,:,:].sum(axis=(1,2))
    plt.plot(light_curve/light_curve.mean(), '-', alpha=0.3)

plt.xlabel('Time (frame index)')
plt.ylabel('Normalized flux in the frame')




