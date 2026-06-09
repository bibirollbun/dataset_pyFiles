import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats
from tqdm import tqdm
import pickle
import itertools
import os
import glob
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error



train_labels=pd.read_csv("/kaggle/input/ariel-data-challenge-2024/train_labels.csv")
train_labels


import os

# Define the path to the dataset directory
dataset_path = '/kaggle/input/long-run-neurips/data_light_raw'

# List all files in the directory
files = os.listdir(dataset_path)

# Filter out the files that end with 'AIRS_clean_train_' and '.npy'
airs_clean_files = [file for file in files if file.startswith('AIRS_clean_train_')]
fgs_clean_files = [file for file in files if file.startswith('FGS1_train_')]
# Display the number of matching files
print(f"Number of AIRS_clean_train_ files: {len(airs_clean_files)}")
print(f"Number of FGS_clean_train_ files: {len(fgs_clean_files)}")
# print(f"Files found: {airs_clean_files}")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(10, 6))

# Iterate over the first 5 planets
for planet_id in train_labels['planet_id'].values[:1]:
    # Construct the file path
    file_path = f"/kaggle/input/long-run-neurips/data_light_raw/AIRS_clean_train_{planet_id}.npy"
    
    # Load the data for the planet
    data = np.load(file_path)[0]
    
    # Replace NaN values with 0
    data[np.isnan(data)] = 0
    
    # Plot the heatmap on top of others
    sns.heatmap(data[0, :, :].T, cbar=True)

# Customize the plot
plt.title("Intensity Heatmap for a Planet")
plt.ylabel('Spatial Dimension')
plt.xlabel('Wavelength Dimension')
plt.show()




# Brightness Variation Information Showing the Approximate In-Transit and Out-Transit Regions

plt.figure(figsize=(10, 6))

# Iterate for all the planets
for planet_id in train_labels['planet_id'].values[:]:
    # Construct the file path
    file_path = f"/kaggle/input/long-run-neurips/data_light_raw/AIRS_clean_train_{planet_id}.npy"
    
    # Load the data for the planet
    data = np.load(file_path)[0]
    
    # Replace NaN values with 0
    data[np.isnan(data)] = 0
    
    # Sum over the wavelength 
    series = data.sum(axis=1).sum(axis=1)

    plt.plot(series/series.mean())

for time_step in [51, 75, 115, 140]:
   plt.axvline(time_step, color='black', linewidth=1)
plt.title("For 673 Planets")
plt.xlabel("Time (frame index)")
plt.ylabel("Normalized Flux")
plt.show()



# Smoothened Gradient showing the Approximate Transition Points

plt.figure(figsize=(10, 6))

# Iterate for all the planets
for planet_id in train_labels['planet_id'].values[:]:
    # Construct the file path
    file_path = f"/kaggle/input/long-run-neurips/data_light_raw/AIRS_clean_train_{planet_id}.npy"
    
    # Load the data for the planet
    data = np.load(file_path)[0]
    
    # Replace NaN values with 0
    data[np.isnan(data)] = 0
    
    # Sum over the wavelength 
    tsignal=pd.Series(data.sum(axis=1).sum(axis=1)).rolling(window=5).sum()
    gsignal=np.gradient(tsignal/tsignal.max())

    plt.plot(gsignal)

for time_step in [51, 75, 115, 140]:
   plt.axvline(time_step, color='black', linewidth=1)
plt.title("For 673 Planets")
plt.xlabel("Time (frame index)")
plt.ylabel("Normalized Gradient Flux")
plt.show()


# Spectral Information showing the absorption dips

plt.figure(figsize=(10, 6))

# Iterate for all the planets
for planet_id in train_labels['planet_id'].values[:]:
    # Construct the file path
    file_path = f"/kaggle/input/long-run-neurips/data_light_raw/AIRS_clean_train_{planet_id}.npy"
    
    # Load the data for the planet
    data = np.load(file_path)[0]
    
    # Replace NaN values with 0
    data[np.isnan(data)] = 0
    
    # Sum over the time frame
    series = data.sum(axis=0).sum(axis=1)

    plt.plot(series/series.mean())

# Customize the plot
plt.title("For 673 Planets")
plt.xlabel("Wavelength")
plt.ylabel("Normalized Flux")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(10, 6))

# Iterate over the first 5 planets
for planet_id in train_labels['planet_id'].values[:1]:
    # Construct the file path
    file_path = f"/kaggle/input/long-run-neurips/data_light_raw/FGS1_train_{planet_id}.npy"
    
    # Load the data for the planet
    data = np.load(file_path)[0]
    
    # Replace NaN values with 0
    data[np.isnan(data)] = 0
    
    # Plot the heatmap on top of others
    sns.heatmap(data[0, :, :].T, cbar=True)

# Customize the plot
plt.title("Intensity Heatmap for a Planet")
plt.ylabel('Spatial Dimension')
plt.xlabel('Wavelength Dimension')
plt.show()


# Brightness Variation Information Showing the Approximate In-Transit and Out-Transit Regions

plt.figure(figsize=(10, 6))

# Iterate for all the planets
for planet_id in train_labels['planet_id'].values[:]:
    # Construct the file path
    file_path = f"/kaggle/input/long-run-neurips/data_light_raw/FGS1_train_{planet_id}.npy"
    
    # Load the data for the planet
    data = np.load(file_path)[0]
    
    # Replace NaN values with 0
    data[np.isnan(data)] = 0
    
    # Sum over the wavelength 
    series = data.sum(axis=1).sum(axis=1)

    plt.plot(series/series.mean())

for time_step in [51, 75, 115, 140]:
   plt.axvline(time_step, color='black', linewidth=1)
plt.title("For 673 Planets")
plt.xlabel("Time (frame index)")
plt.ylabel("Normalized Flux")
plt.show()


in_data=np.load("/kaggle/input/transitdata-errors/1005054328/in_transit_spectrum_data.npy")
out_data= np.load("/kaggle/input/transitdata-errors/1005054328/out_transit_spectrum_data.npy")
planet_id = 1005054328  # Example planet ID as an integer

# Filter the DataFrame to get the row corresponding to this planet_id
planet_data = train_labels[train_labels['planet_id'] == planet_id]

# Drop the 'planet_id' column, as you only need the wavelength columns
wavelength_data = planet_data.drop(columns=['planet_id'])



tr_data = []
# Iterate over the first 5 planets
for planet_id in tqdm(train_labels['planet_id'].values[:]):
    # Construct the file path
    in_data = np.load(f"/kaggle/input/transitdata-errors/{planet_id}/in_transit_spectrum_data.npy")
    out_data = np.load(f"/kaggle/input/transitdata-errors/{planet_id}/out_transit_spectrum_data.npy")
    transit_depth= (out_data-in_data)/out_data
    tr_data.append(transit_depth)


tr_data = np.array(tr_data)
train_data, val_data, train_label, val_label = train_test_split(tr_data, train_labels.values[:, 1:], \
                                                               test_size=0.4, shuffle=True)


from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Define Ridge regression model
model = Ridge(alpha=1E-3)

# Cross-validation predictions for train data
oof_pred = cross_val_predict(model, train_data, train_label)

# Calculate R2 score and RMSE for train data
print(f"# R2 score for Train Data: {r2_score(train_label, oof_pred):.3f}")
sigma_pred = mean_squared_error(train_label, oof_pred, squared=False)
print(f"# Root mean squared error for Train Data: {sigma_pred:.6f}")

# Cross-validation predictions for validation data
oof_pred_val = cross_val_predict(model, val_data, val_label)

# Calculate R2 score and RMSE for validation data
print(f"# R2 score on Validation Data: {r2_score(val_label, oof_pred_val):.3f}")
sigma_pred_val = mean_squared_error(val_label, oof_pred_val, squared=False)
print(f"# Root mean squared error for Validation Data: {sigma_pred_val:.6f}")

# Plotting
plt.figure(figsize=(8, 6))
col = 1  # Ensure this column exists in both train_label and val_label

# Error bar plot for train data
plt.errorbar(
    oof_pred[:, col], train_label[:, col], yerr=sigma_pred, 
    fmt='o', markersize=4, alpha=1, label='Predictions on Train Data', 
    color='deepskyblue', markeredgecolor='blue', ecolor='gray', elinewidth=2
)

# Error bar plot for validation data
plt.errorbar(
    oof_pred_val[:, col], val_label[:, col], yerr=sigma_pred_val, 
    fmt='o', markersize=4, alpha=1, label='Predictions on Validation Data', 
    color='lightgreen', markeredgecolor='green', ecolor='gray', elinewidth=2
)

# Reference line y = x
y_min = min(train_label.min(), val_label.min())
y_max = max(train_label.max(), val_label.max())
plt.plot([y_min, y_max], [y_min, y_max], color='red', linestyle='--', linewidth=3, label='y = x')

# Labels, title, and legend
plt.xlabel('y_pred')
plt.ylabel('y_true')
plt.title('Comparing y_true and y_pred')
plt.legend()
plt.show()



import pickle

# save
with open('/kaggle/working/model.pkl','wb') as f:
    pickle.dump(model,f)

