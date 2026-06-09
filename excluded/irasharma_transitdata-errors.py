import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

import glob
import os
import time
# import 
path_folder = '/kaggle/input/ariel-data-challenge-2024/'
path_out = "/kaggle/working/"




def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            absolute_path = os.path.abspath(os.path.join(root, file))
            if 'AIRS_clean_train_' in absolute_path:
                file_paths.append(absolute_path.split('AIRS_clean_train_')[1].split('.npy')[0])
    return file_paths


# ids = get_file_paths('/kaggle/input/long-run-neurips')

# for n, id in enumerate(ids):
#     data = np.load(f'/kaggle/input/long-run-neurips/data_light_raw/AIRS_clean_train_{id}.npy')[0]
#     data[np.isnan(data)] = 0
    
#     series = data.sum(axis=1).sum(axis=1)
#     series /= np.mean(series)
#     x_axis = np.arange(len(series))
    
    
#     def taylor_series(x, a, b, c, d): #, d, e, f):
#         return a + b * x + c * x**2 + d * x**3 #+ d * np.sin(e*x + f)# + d * x**3 
    
#     param, param_cov = curve_fit(taylor_series, list(x_axis[:40]) + list(x_axis[len(x_axis)-40:]), 
#                                                 list(series[:40]) + list(series[len(series)-40:]))
    
#     series /= taylor_series(x_axis, *param)
    
#     grad = np.gradient(series)
#     cutoff = 1
    
#     x_start, x_end = np.where(grad == min(grad))[0][0], np.where(grad == max(grad))[0][0]
    
#     zero_to_t1, t1_to_t4, t4_to_end = series[40:x_start], series[x_start:x_end], series[x_end:len(series)-40]
    
#     index_in_transit = np.where(np.abs(series - np.mean(t1_to_t4)) < cutoff * np.std(t1_to_t4))[0]
#     index_out_transit = np.where((np.abs(series - np.mean(zero_to_t1)) < cutoff * np.std(zero_to_t1)) | \
#                                (np.abs(series - np.mean(t4_to_end)) < cutoff * np.std(t4_to_end)))[0]
    
#     spectrum_in_transit = data[index_in_transit, :, :].sum(axis=2).sum(axis=0)
#     spectrum_out_transit = data[index_out_transit, :, :].sum(axis=2).sum(axis=0)

#     spectrum_error_in_transit = np.array([data[index_in_transit, i, :].flatten().std(axis=0) for i in range(data.shape[1])])
#     spectrum_error_out_transit = np.array([data[index_in_transit, i, :].flatten().std(axis=0) for i in range(data.shape[1])])
    
#     # plt.plot(x_axis[index_out_transit], series[index_out_transit], 'b')
#     # plt.plot(x_axis[index_in_transit], series[index_in_transit], 'r')

    
    
#     if not os.path.exists(path_out + f"AIRS_spectrum_in_transit/"):
#         os.mkdir(path_out + f"AIRS_spectrum_in_transit/")
    
#     np.save(path_out + f"AIRS_spectrum_in_transit/data_{id}",  spectrum_in_transit)
#     np.save(path_out + f"AIRS_spectrum_in_transit/error_{id}",  spectrum_error_in_transit)

    
#     if not os.path.exists(path_out + f"AIRS_spectrum_out_transit/"):
#         os.mkdir(path_out + f"AIRS_spectrum_out_transit/")
    
#     np.save(path_out + f"AIRS_spectrum_out_transit/data_{id}", spectrum_out_transit)
#     np.save(path_out + f"AIRS_spectrum_out_transit/error_{id}",  spectrum_error_out_transit)



from scipy.optimize import curve_fit

ids = get_file_paths('/kaggle/input/long-run-neurips')

for n, id in enumerate(ids):
    data = np.load(f'/kaggle/input/long-run-neurips/data_light_raw/AIRS_clean_train_{id}.npy')[0]
    data[np.isnan(data)] = 0
    
    series = data.sum(axis=1).sum(axis=1)
    series /= np.mean(series)
    x_axis = np.arange(len(series))
    
    
    def taylor_series(x, a, b, c, d): #, d, e, f):
        return a + b * x + c * x**2 + d * x**3 #+ d * np.sin(e*x + f)# + d * x**3 
    
    param, param_cov = curve_fit(taylor_series, list(x_axis[:40]) + list(x_axis[len(x_axis)-40:]), 
                                                list(series[:40]) + list(series[len(series)-40:]))
    
    series /= taylor_series(x_axis, *param)
    
    grad = np.gradient(series)
    cutoff = 1
    
    x_start, x_end = np.where(grad == min(grad))[0][0], np.where(grad == max(grad))[0][0]
    
    zero_to_t1, t1_to_t4, t4_to_end = series[40:x_start], series[x_start:x_end], series[x_end:len(series)-40]
    
    index_in_transit = np.where(np.abs(series - np.mean(t1_to_t4)) < cutoff * np.std(t1_to_t4))[0]
    index_out_transit = np.where((np.abs(series - np.mean(zero_to_t1)) < cutoff * np.std(zero_to_t1)) | \
                               (np.abs(series - np.mean(t4_to_end)) < cutoff * np.std(t4_to_end)))[0]
    
    spectrum_in_transit = data[index_in_transit, :, :].sum(axis=2).mean(axis=0)
    spectrum_out_transit = data[index_out_transit, :, :].sum(axis=2).mean(axis=0)
    
    if not os.path.exists(path_out + f"{id}/"):
        os.mkdir(path_out + f"{id}/")

    np.save(path_out + f"{id}/full_data",  data)


    np.save(path_out + f"{id}/in_transit_spectrum_data",  spectrum_in_transit)
    np.save(path_out + f"{id}/in_transit_index",  index_in_transit)


    np.save(path_out + f"{id}/out_transit_spectrum_data", spectrum_out_transit)
    np.save(path_out + f"{id}/out_transit_index",  index_out_transit)




