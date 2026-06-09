import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


import os
os.chdir(SIMU_DIR)


#domain: 50.02 26 0.02 
#dx_dy_dz: 0.02 0.02 0.02 
#time_window: 15000 
#time_step_stability_factor: 0.99 
#waveform: gaussiandotdot 1 60.0e6 my_pulse
#hertzian_dipole: z 2 22.8 0 my_pulse 
#rx: 2.02 22.8 0 
#rx: 2.22 22.8 0 
#rx: 2.42 22.8 0 
#rx: 2.62 22.8 0 
#rx: 2.82 22.8 0 
#rx: 3.02 22.8 0 
#rx: 3.22 22.8 0 
#rx: 3.42 22.8 0 
#rx: 3.62 22.8 0 
#rx: 3.82 22.8 0 
#rx: 4.02 22.8 0 
#rx: 4.22 22.8 0 
#rx: 4.42 22.8 0 
#rx: 4.62 22.8 0 
#rx: 4.82 22.8 0 
#rx: 5.02 22.8 0 
#rx: 5.22 22.8 0 
#rx: 5.42 22.8 0 
#rx: 5.62 22.8 0 
#rx: 5.82 22.8 0 
#rx: 6.02 22.8 0 
#rx: 6.22 22.8 0 
#rx: 6.42 22.8 0 
#rx: 6.62 22.8 0 
#rx: 6.82 22.8 0 
#rx: 7.02 22.8 0 
#rx: 7.22 22.8 0 
#rx: 7.42 22.8 0 
#rx: 7.62 22.8 0 
#rx: 7.82 22.8 0 
#rx: 8.02 22.8 0 
#rx: 8.22 22.8 0 
#rx: 8.42 22.8 0 
#rx: 8.62 22.8 0 
#rx: 8.82 22.8 0 
#rx: 9.02 22.8 0 
#rx: 9.22 22.8 0 
#rx: 9.42 22.8 0 
#rx: 9.62 22.8 0 
#rx: 9.82 22.8 0 
#rx: 10.02 22.8 0 
#rx: 10.22 22.8 0 
#rx: 10.42 22.8 0 
#rx: 10.62 22.8 0 
#rx: 10.82 22.8 0 
#rx: 11.02 22.8 0 
#rx: 11.22 22.8 0 
#rx: 11.42 22.8 0 
#rx: 11.62 22.8 0 
#rx: 11.82 22.8 0 
#rx: 12.02 22.8 0 
#rx: 12.22 22.8 0 
#rx: 12.42 22.8 0 
#rx: 12.62 22.8 0 
#rx: 12.82 22.8 0 
#rx: 13.02 22.8 0 
#rx: 13.22 22.8 0 
#rx: 13.42 22.8 0 
#rx: 13.62 22.8 0 
#rx: 13.82 22.8 0 
#rx: 14.02 22.8 0 
#rx: 14.22 22.8 0 
#rx: 14.42 22.8 0 
#rx: 14.62 22.8 0 
#rx: 14.82 22.8 0 
#rx: 15.02 22.8 0 
#rx: 15.22 22.8 0 
#rx: 15.42 22.8 0 
#rx: 15.62 22.8 0 
#rx: 15.82 22.8 0 
#rx: 16.02 22.8 0 
#rx: 16.22 22.8 0 
#rx: 16.42 22.8 0 
#rx: 16.62 22.8 0 
#rx: 16.82 22.8 0 
#rx: 17.02 22.8 0 
#rx: 17.22 22.8 0 
#rx: 17.42 22.8 0 
#rx: 17.62 22.8 0 
#rx: 17.82 22.8 0 
#rx: 18.02 22.8 0 
#rx: 18.22 22.8 0 
#rx: 18.42 22.8 0 
#rx: 18.62 22.8 0 
#rx: 18.82 22.8 0 
#rx: 19.02 22.8 0 
#rx: 19.22 22.8 0 
#rx: 19.42 22.8 0 
#rx: 19.62 22.8 0 
#rx: 19.82 22.8 0 
#rx: 20.02 22.8 0 
#rx: 20.22 22.8 0 
#rx: 20.42 22.8 0 
#rx: 20.62 22.8 0 
#rx: 20.82 22.8 0 
#rx: 21.02 22.8 0 
#rx: 21.22 22.8 0 
#rx: 21.42 22.8 0 
#rx: 21.62 22.8 0 
#rx: 21.82 22.8 0 
#rx: 22.02 22.8 0 
#rx: 22.22 22.8 0 
#rx: 22.42 22.8 0 
#rx: 22.62 22.8 0 
#rx: 22.82 22.8 0 
#rx: 23.02 22.8 0 
#rx: 23.22 22.8 0 
#rx: 23.42 22.8 0 
#rx: 23.62 22.8 0 
#rx: 23.82 22.8 0 
#rx: 24.02 22.8 0 
#rx: 24.22 22.8 0 
#rx: 24.42 22.8 0 
#rx: 24.62 22.8 0 
#rx: 24.82 22.8 0 
#rx: 25.02 22.8 0 
#rx: 25.22 22.8 0 
#rx: 25.42 22.8 0 
#rx: 25.62 22.8 0 
#rx: 25.82 22.8 0 
#rx: 26.02 22.8 0 
#rx: 26.22 22.8 0 
#rx: 26.42 22.8 0 
#rx: 26.62 22.8 0 
#rx: 26.82 22.8 0 
#rx: 27.02 22.8 0 
#rx: 27.22 22.8 0 
#rx: 27.42 22.8 0 
#rx: 27.62 22.8 0 
#rx: 27.82 22.8 0 
#rx: 28.02 22.8 0 
#rx: 28.22 22.8 0 
#rx: 28.42 22.8 0 
#rx: 28.62 22.8 0 
#rx: 28.82 22.8 0 
#rx: 29.02 22.8 0 
#rx: 29.22 22.8 0 
#rx: 29.42 22.8 0 
#rx: 29.62 22.8 0 
#rx: 29.82 22.8 0 
#rx: 30.02 22.8 0 
#rx: 30.22 22.8 0 
#rx: 30.42 22.8 0 
#rx: 30.62 22.8 0 
#rx: 30.82 22.8 0 
#rx: 31.02 22.8 0 
#rx: 31.22 22.8 0 
#rx: 31.42 22.8 0 
#rx: 31.62 22.8 0 
#rx: 31.82 22.8 0 
#rx: 32.02 22.8 0 
#rx: 32.22 22.8 0 
#rx: 32.42 22.8 0 
#rx: 32.62 22.8 0 
#rx: 32.82 22.8 0 
#rx: 33.02 22.8 0 
#rx: 33.22 22.8 0 
#rx: 33.42 22.8 0 
#rx: 33.62 22.8 0 
#rx: 33.82 22.8 0 
#rx: 34.02 22.8 0 
#rx: 34.22 22.8 0 
#rx: 34.42 22.8 0 
#rx: 34.62 22.8 0 
#rx: 34.82 22.8 0 
#rx: 35.02 22.8 0 
#rx: 35.22 22.8 0 
#rx: 35.42 22.8 0 
#rx: 35.62 22.8 0 
#rx: 35.82 22.8 0 
#rx: 36.02 22.8 0 
#rx: 36.22 22.8 0 
#rx: 36.42 22.8 0 
#rx: 36.62 22.8 0 
#rx: 36.82 22.8 0 
#rx: 37.02 22.8 0 
#rx: 37.22 22.8 0 
#rx: 37.42 22.8 0 
#rx: 37.62 22.8 0 
#rx: 37.82 22.8 0 
#rx: 38.02 22.8 0 
#rx: 38.22 22.8 0 
#rx: 38.42 22.8 0 
#rx: 38.62 22.8 0 
#rx: 38.82 22.8 0 
#rx: 39.02 22.8 0 
#rx: 39.22 22.8 0 
#rx: 39.42 22.8 0 
#rx: 39.62 22.8 0 
#rx: 39.82 22.8 0 
#rx: 40.02 22.8 0 
#rx: 40.22 22.8 0 
#rx: 40.42 22.8 0 
#rx: 40.62 22.8 0 
#rx: 40.82 22.8 0 
#rx: 41.02 22.8 0 
#rx: 41.22 22.8 0 
#rx: 41.42 22.8 0 
#rx: 41.62 22.8 0 
#rx: 41.82 22.8 0 
#rx: 42.02 22.8 0 
#rx: 42.22 22.8 0 
#rx: 42.42 22.8 0 
#rx: 42.62 22.8 0 
#rx: 42.82 22.8 0 
#rx: 43.02 22.8 0 
#rx: 43.22 22.8 0 
#rx: 43.42 22.8 0 
#rx: 43.62 22.8 0 
#rx: 43.82 22.8 0 
#rx: 44.02 22.8 0 
#rx: 44.22 22.8 0 
#rx: 44.42 22.8 0 
#rx: 44.62 22.8 0 
#rx: 44.82 22.8 0 
#rx: 45.02 22.8 0 
#rx: 45.22 22.8 0 
#rx: 45.42 22.8 0 
#rx: 45.62 22.8 0 
#rx: 45.82 22.8 0 
#rx: 46.02 22.8 0 
#rx: 46.22 22.8 0 
#rx: 46.42 22.8 0 
#rx: 46.62 22.8 0 
#rx: 46.82 22.8 0 
#rx: 47.02 22.8 0 
#rx: 47.22 22.8 0 
#rx: 47.42 22.8 0 
#rx: 47.62 22.8 0 
#rx: 47.82 22.8 0 
#pml_cells: 60 60 0 60 60 0 
#material: 3 0.00003 1 0 my_sand 
#box: 0 0 0 50.02 22.0 0.02 my_sand 



import os
import re

# Parameter definition
wave = {0: 'gaussiandot', 1: 'ricker', 2: 'gaussiandotnorm', 3: 'gaussiandotdot'}
cf_values = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0] # Center frequency
v_values  = [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Dielectric constant

# template file: box_model.in
with open(os.path.join(SRC_DIR, r"box_model.in"), "r") as f:
    template = f.read()

exp_counter = 0

for ind in wave.keys(): # ind is the wave type key 0 to 3
    for cf in cf_values:
        for v in v_values:
            # Generate new folder names (e.g. box_exp0001, box_exp0002, ...)
            folder_name = f"box_exp{exp_counter:04d}"
            os.makedirs(folder_name, exist_ok=True)
            
            new_waveform = "#waveform: {} 1 {}e6 my_pulse \n".format(wave[ind], cf)
            # Set ε_r, σ=ε_r/10000, Round to 7 decimal places
            new_material = "#material: {} {} 1 0 my_sand \n".format(
                format(round(v, 3), ".3f"),
                format(round(v / 10000, 7), ".7f")
            )
            
            # Replace the #waveform and #material lines in the template
            content = re.sub(r"#waveform:.*\n", new_waveform, template)
            content = re.sub(r"#material:.*\n", new_material, content)
            
            with open(os.path.join(folder_name, "box_model.in"), "w") as f_out:
                f_out.write(content)
            
            exp_counter += 1


from skimage.transform import resize
import numpy as np
from matplotlib import pyplot as plt

import argparse
import glob
import os
import copy
import h5py
import numpy as np
from sklearn import preprocessing as p

from scipy.interpolate import NearestNDInterpolator

def get_output_data(filename, rxnumber, rxcomponent):
    """Gets B-scan output data from a model.

    Args:
        filename (string): Filename (including path) of output file.
        rxnumber (int): Receiver output number.
        rxcomponent (str): Receiver output field/current component.

    Returns:
        outputdata (array): Array of A-scans, i.e. B-scan data.
        dt (float): Temporal resolution of the model.
    """

    # Open output file and read some attributes
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    dt = f.attrs['dt']

    # Check there are any receivers
    if nrx == 0:
        raise CmdInputError('No receivers found in {}'.format(filename))

    path = '/rxs/rx' + str(rxnumber) + '/'
    availableoutputs = list(f[path].keys())

    # Check if requested output is in file
    if rxcomponent not in availableoutputs:
        raise CmdInputError('{} output requested to plot, but the available output for receiver 1 is {}'.format(rxcomponent, ', '.join(availableoutputs)))

    outputdata = f[path + '/' + rxcomponent]
    outputdata = np.array(outputdata)
    f.close()

    return outputdata, dt


def make_bscan(fil,plot=False):
    n_receivers = np.shape(np.arange(101,2400,10))[0]

    Bscan=[]
    for i in range(0,n_receivers):
        [fi, t]=get_output_data(fil, i+1, "Ez")
        gain = np.arange(0,np.shape(fi)[0],40, dtype=np.float64)**3
        Bscan.append(fi[0:np.shape(fi)[0]:40]*gain)

    Bs = np.array(Bscan)
    image = Bs.T

    # Resize your BScan to 230x230 dimensions.
    B = resize(image, (230, 230))
    B2 = copy.copy(B)
    if plot:
        B[B>np.max(B)*0.05] = np.max(B)*0.05
        B[B<np.min(B)*0.05] = np.min(B)*0.05
        plt.imshow(B,aspect='auto',cmap= 'bone')
        plt.show()

    # B2 is a numpy array with the resized (230x230) and processed BScan
    return B2


import os
import numpy as np

Bscan_data_path = BSCAN_DATA_PATH
num_Bscan_samples = 360

Bscan_data_list = []

for i in range(num_Bscan_samples):
    filename = os.path.join(Bscan_data_path, f"box_model_Bscan_{i}.npy")
    data = np.load(filename)
    Bscan_data_list.append(data)

box_inputs = np.array(Bscan_data_list)  # Shape: (N, H, W)
del Bscan_data_list


import itertools

wave = {0: 'gaussiandot', 1: 'ricker', 2: 'gaussiandotnorm', 3: 'gaussiandotdot'}
cf_values = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0] # Center frequency
ep_values  = [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Dielectric constant

for exp_counter, (ind, cf, ep) in enumerate(itertools.product(wave.keys(), cf_values, ep_values)):
    print(exp_counter, wave[ind], cf, ep)


def bscan_line_sum3(inputs_index0, inputs, offset, vn, t0m):
    result = 0
    for i in range(230):
        t = calc_direct_t(offset[i], vn, t0m)
            
        if t <= 700:
            result += np.interp(t, np.linspace(0, 700, 230), inputs[inputs_index0, :, i])
        else:
            break
    return result

def calc_direct_t(offset,  v, t0=0.0):
    return  ( offset / v ) + t0


import itertools
from scipy.signal import argrelextrema
from scipy.optimize import minimize
import matplotlib.pyplot as plt

offset = np.linspace(0.02, 45.82, 230)
t_index = np.linspace(0, 700, 230)

input_peak_time_list2 = []
local_max_min_dict = {}

wave = {0: 'gaussiandot', 1: 'ricker', 2: 'gaussiandotnorm', 3: 'gaussiandotdot'}
cf_values = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0] # Center frequency
ep_values  = [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Dielectric constant

for inputs_index0, (ind, cf, ep) in enumerate(itertools.product(wave.keys(), cf_values, ep_values)):
    category = ind

    # Calculate the value of bscan_line_sum when t0m is changed from 0 to 30 in increments of 0.2
    t0m_values = np.arange(0, 30.2, 0.2)
    vn = 0.3
    bscan_values = [bscan_line_sum3(inputs_index0, box_inputs, offset, vn, t) for t in t0m_values]
    
    # Get the indices of local maximum and minimum
    local_max_idx = argrelextrema(np.array(bscan_values), np.greater)[0]  # Local maximum
    local_min_idx = argrelextrema(np.array(bscan_values), np.less)[0]     # Local minimum
    
    local_max_min_dict[inputs_index0] = {
        "local_max_idx": local_max_idx.tolist(),
        "local_min_idx": local_min_idx.tolist()
    }

    if category == 1:
        initial_guess = t0m_values[np.argmin(np.array(bscan_values))]
    else:
        initial_guess = t0m_values[np.argmax(np.array(bscan_values))]
                
    bounds = [(initial_guess-0.5, initial_guess+0.5)]
    vn = 0.3

    if category ==1:
        objective = lambda t0m, inputs_index0=inputs_index0: bscan_line_sum3(inputs_index0, box_inputs, offset, vn, t0m[0])
    else:
        objective = lambda t0m, inputs_index0=inputs_index0: -bscan_line_sum3(inputs_index0, box_inputs, offset, vn, t0m[0])
                
    result = minimize(objective, initial_guess, bounds=bounds)
    optimal_t0m = result.x[0]
    print(f"inputs_index0 {inputs_index0}, category {category}, 最適な peak_time = {optimal_t0m}")
    input_peak_time_list2.append(optimal_t0m)

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(t0m_values, bscan_values, label="bscan_line_sum")
    
    plt.plot(t0m_values[local_max_idx], np.array(bscan_values)[local_max_idx], 'ro', label='Local Maxima')
    plt.plot(t0m_values[local_min_idx], np.array(bscan_values)[local_min_idx], 'go', label='Local Minima')
    
    plt.axvline(optimal_t0m, color='red', linestyle='--', label=f'Optimal peak_time = {optimal_t0m:.2f}')
    plt.xlabel("shifted time")
    plt.ylabel("bscan_line_sum")
    plt.title("bscan_line_sum Box No.{} category:{} cf:{:.0f} ep:{:.1f}".format(inputs_index0, category, cf, ep))
    plt.legend()
    plt.grid(True)
    #plt.show()
    folder_path = FOLDER_PATH
    output_file = os.path.join(folder_path,"Box_gpr_peak_time_{:04d}.jpg".format(inputs_index0))
    plt.savefig(output_file)
    plt.close()


import json
folder_path = FOLDER_PATH
json_output_file = os.path.join(folder_path, "Box_exp_local_max_min_dict.json")
with open(json_output_file, "w") as f:
    json.dump(local_max_min_dict, f, indent=4)

print(f"JSONファイルとして保存しました: {json_output_file}")


import pandas as pd
import itertools

local_max_min_dict_rows = []

for index, (wave_ind, cf, ep) in enumerate(itertools.product(wave.keys(), cf_values, ep_values)):

    data = local_max_min_dict.get(index, {})
    local_max_idx = data.get('local_max_idx', [])
    local_min_idx = data.get('local_min_idx', [])
    
    # Check the length of the list and assign a value such as None if it does not exist.
    local_high0 = local_max_idx[0] if len(local_max_idx) > 0 else None
    local_high1 = local_max_idx[1] if len(local_max_idx) > 1 else None
    local_low0  = local_min_idx[0]  if len(local_min_idx)  > 0 else None
    local_low1  = local_min_idx[1]  if len(local_min_idx)  > 1 else None
    
    local_max_min_dict_rows.append({
        'wave_ind': wave_ind,
        'cf': cf,
        'ep': ep,
        'local_high0': local_high0,
        'local_high1': local_high1,
        'local_low0': local_low0,
        'local_low1': local_low1
    })

df = pd.DataFrame(local_max_min_dict_rows)
print(df)


df[df['wave_ind'] == 0]['local_high0'].to_numpy()


df[df['wave_ind'] == 0]['local_high0'].to_numpy().reshape(9,10)


print(df[df['wave_ind'] == 0]['local_high0'].to_numpy().reshape(9,10)[0,2])


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cf_values = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0] # Center frequency
ep_values  = [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Dielectric constant

t0m_values = np.arange(0, 30.2, 0.2)

data = t0m_values[df[df['wave_ind'] == 0]['local_high0'].to_numpy().reshape(len(cf_values),len(ep_values))]

x = np.array(cf_values)
y = np.array(ep_values)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=30, azim=50)

# meshgrid default indexing='xy' return (len(y), len(x)) , data need to transpose
ax.plot_wireframe(X, Y, data.T, color='r')

ax.set_ylabel(r'$\epsilon_{r}$')
ax.set_xlabel('cf')
ax.set_zlabel('Z Axis')
ax.set_title('3D Surface Plot')

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cf_values = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0] # Center frequency
ep_values  = [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Dielectric constant

t0m_values = np.arange(0, 30.2, 0.2)

data = t0m_values[df[df['wave_ind'] == 0]['local_high0'].to_numpy().reshape(len(cf_values),len(ep_values))]
data3 = t0m_values[df[df['wave_ind'] == 0]['local_low0'].to_numpy().reshape(len(cf_values),len(ep_values))]

x = np.array(cf_values)
y = np.array(ep_values)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=30, azim=50)

# meshgrid default indexing='xy' return (len(y), len(x)) , data need to transpose
ax.plot_wireframe(X, Y, data.T, color='r')
ax.plot_wireframe(X, Y, data3.T, color='b')

ax.set_ylabel(r'$\epsilon_{r}$')
ax.set_xlabel('cf')
ax.set_zlabel('Z Axis')
ax.set_title('3D Surface Plot')

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cf_values = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0] # Center frequency
ep_values  = [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Dielectric constant

t0m_values = np.arange(0, 30.2, 0.2)

data = t0m_values[df[df['wave_ind'] == 1]['local_high0'].to_numpy().reshape(len(cf_values),len(ep_values))]
data3 = t0m_values[df[df['wave_ind'] == 1]['local_low0'].to_numpy().reshape(len(cf_values),len(ep_values))]

x = np.array(cf_values)
y = np.array(ep_values)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=30, azim=50)

# meshgrid default indexing='xy' return (len(y), len(x)) , data need to transpose
ax.plot_wireframe(X, Y, data.T, color='r')
ax.plot_wireframe(X, Y, data3.T, color='b')

ax.set_ylabel(r'$\epsilon_{r}$')
ax.set_xlabel('cf')
ax.set_zlabel('Z Axis')
ax.set_title('3D Surface Plot')

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cf_values = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0] # Center frequency
ep_values  = [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Dielectric constant

t0m_values = np.arange(0, 30.2, 0.2)

data = t0m_values[df[df['wave_ind'] == 2]['local_high0'].to_numpy().reshape(len(cf_values),len(ep_values))]
data3 = t0m_values[df[df['wave_ind'] == 2]['local_low0'].to_numpy().reshape(len(cf_values),len(ep_values))]

x = np.array(cf_values)
y = np.array(ep_values)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=30, azim=50)

# meshgrid default indexing='xy' return (len(y), len(x)) , data need to transpose
ax.plot_wireframe(X, Y, data.T, color='r')
ax.plot_wireframe(X, Y, data3.T, color='b')

ax.set_ylabel(r'$\epsilon_{r}$')
ax.set_xlabel('cf')
ax.set_zlabel('Z Axis')
ax.set_title('3D Surface Plot')

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cf_values = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0] # Center frequency
ep_values  = [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Dielectric constant

t0m_values = np.arange(0, 30.2, 0.2)

data = t0m_values[df[df['wave_ind'] == 3]['local_high0'].to_numpy().reshape(len(cf_values),len(ep_values))]
data3 = t0m_values[df[df['wave_ind'] == 3]['local_low0'].to_numpy().reshape(len(cf_values),len(ep_values))]

x = np.array(cf_values)
y = np.array(ep_values)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=30, azim=50)

# meshgrid default indexing='xy' return (len(y), len(x)) , data need to transpose
ax.plot_wireframe(X, Y, data.T, color='r')
ax.plot_wireframe(X, Y, data3.T, color='b')

ax.set_ylabel(r'$\epsilon_{r}$')
ax.set_xlabel('cf')
ax.set_zlabel('Z Axis')
ax.set_title('3D Surface Plot')

plt.show()


import itertools
from scipy.signal import argrelextrema
from scipy.optimize import minimize
import matplotlib.pyplot as plt

offset = np.linspace(0.02, 45.82, 230)
t_index = np.linspace(0, 700, 230)

input_peak_time_list2 = []
local_max_min_dict = {}
local_max_min_time_dict = {}

wave = {0: 'gaussiandot', 1: 'ricker', 2: 'gaussiandotnorm', 3: 'gaussiandotdot'}
cf_values = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0] # Center frequency
ep_values  = [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Dielectric constant

for inputs_index0, (ind, cf, ep) in enumerate(itertools.product(wave.keys(), cf_values, ep_values)):
    category = ind

    # Calculate the value of bscan_line_sum when t0m is changed from 0 to 30 in increments of 0.2
    t0m_values = np.arange(0, 30.2, 0.2)
    vn = 0.3
    bscan_values = [bscan_line_sum3(inputs_index0, box_inputs, offset, vn, t) for t in t0m_values]
    
    # Get the indices of local maximum and minimum
    local_max_idx = argrelextrema(np.array(bscan_values), np.greater)[0]  # Local maximum
    local_min_idx = argrelextrema(np.array(bscan_values), np.less)[0]     # Local minimum
    
    local_max_min_dict[inputs_index0] = {
        "local_max_idx": local_max_idx.tolist(),
        "local_min_idx": local_min_idx.tolist()
    }

    lp_high_time = []
    lp_low_time = []
    vn = 0.3
    
    #local_max
    for lp_high in local_max_idx.tolist():
        initial_guess = t0m_values[lp_high]
        bounds = [(t0m_values[lp_high]-0.5, t0m_values[lp_high]+0.5)]
        objective = lambda t0m, inputs_index0=inputs_index0: -bscan_line_sum3(inputs_index0, box_inputs, offset, vn, t0m[0])
        result = minimize(objective, initial_guess, bounds=bounds)
        optimal_time = result.x[0]
        lp_high_time.append(optimal_time)
        
    #local_min
    for lp_low in local_min_idx.tolist():
        initial_guess = t0m_values[lp_low]
        bounds = [(t0m_values[lp_low]-0.5, t0m_values[lp_low]+0.5)]
        objective = lambda t0m, inputs_index0=inputs_index0: bscan_line_sum3(inputs_index0, box_inputs, offset, vn, t0m[0])
        result = minimize(objective, initial_guess, bounds=bounds)
        optimal_time = result.x[0]
        lp_low_time.append(optimal_time)

    local_max_min_time_dict[inputs_index0] = {
        "local_max_time": lp_high_time,
        "local_min_time": lp_low_time
    }
        
    bounds = [(initial_guess-0.5, initial_guess+0.5)]
    vn = 0.3
    result = minimize(objective, initial_guess, bounds=bounds)
    

    if category == 1:
        initial_guess = t0m_values[np.argmin(np.array(bscan_values))]
    else:
        initial_guess = t0m_values[np.argmax(np.array(bscan_values))]
                
    bounds = [(initial_guess-0.5, initial_guess+0.5)]
    vn = 0.3

    if category ==1:
        objective = lambda t0m, inputs_index0=inputs_index0: bscan_line_sum3(inputs_index0, box_inputs, offset, vn, t0m[0])
    else:
        objective = lambda t0m, inputs_index0=inputs_index0: -bscan_line_sum3(inputs_index0, box_inputs, offset, vn, t0m[0])
                
    result = minimize(objective, initial_guess, bounds=bounds)
    optimal_t0m = result.x[0]
    print(f"inputs_index0 {inputs_index0}, category {category}, 最適な peak_time = {optimal_t0m}")
    input_peak_time_list2.append(optimal_t0m)


import json

folder_path = FOLDER_PATH
json_output_file = os.path.join(folder_path, "Box_exp_local_max_min_time_dict.json")
with open(json_output_file, "w") as f:
    json.dump(local_max_min_time_dict, f, indent=4)

print(f"JSONファイルとして保存しました: {json_output_file}")


import pandas as pd
import itertools

local_max_min_time_dict_rows = []

for index, (wave_ind, cf, ep) in enumerate(itertools.product(wave.keys(), cf_values, ep_values)):

    data = local_max_min_time_dict.get(index, {})
    local_max_time = data.get('local_max_time', [])
    local_min_time = data.get('local_min_time', [])
    
    # Check the length of the list and assign a value such as None if it does not exist.
    local_high0 = local_max_time[0] if len(local_max_time) > 0 else None
    local_high1 = local_max_time[1] if len(local_max_time) > 1 else None
    local_low0  = local_min_time[0]  if len(local_min_time)  > 0 else None
    local_low1  = local_min_time[1]  if len(local_min_time)  > 1 else None
    
    local_max_min_time_dict_rows.append({
        'wave_ind': wave_ind,
        'cf': cf,
        'ep': ep,
        'local_high0': local_high0,
        'local_high1': local_high1,
        'local_low0': local_low0,
        'local_low1': local_low1
    })

time_df = pd.DataFrame(local_max_min_time_dict_rows)
print(time_df)





import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cf_values = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0] # Center frequency
ep_values  = [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Dielectric constant

data = time_df[time_df['wave_ind'] == 0]['local_high0'].to_numpy().reshape(len(cf_values),len(ep_values))
data2 = time_df[time_df['wave_ind'] == 0]['local_low0'].to_numpy().reshape(len(cf_values),len(ep_values))

x = np.array(cf_values)
y = np.array(ep_values)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=30, azim=50)

# meshgrid default indexing='xy' return (len(y), len(x)) , data need to transpose
ax.plot_wireframe(X, Y, data.T, color='r')
ax.plot_wireframe(X, Y, data2.T, color='b')

ax.set_ylabel(r'$\epsilon_{r}$')
ax.set_xlabel('cf')
ax.set_zlabel('Z Axis')
ax.set_title('3D Surface Plot')

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cf_values = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0] # Center frequency
ep_values  = [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Dielectric constant

data = time_df[time_df['wave_ind'] == 1]['local_high0'].to_numpy().reshape(len(cf_values),len(ep_values))
data2 = time_df[time_df['wave_ind'] == 1]['local_low0'].to_numpy().reshape(len(cf_values),len(ep_values))

x = np.array(cf_values)
y = np.array(ep_values)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=30, azim=50)

# meshgrid default indexing='xy' return (len(y), len(x)) , data need to transpose
ax.plot_wireframe(X, Y, data.T, color='r')
ax.plot_wireframe(X, Y, data2.T, color='b')

ax.set_ylabel(r'$\epsilon_{r}$')
ax.set_xlabel('cf')
ax.set_zlabel('Z Axis')
ax.set_title('3D Surface Plot')

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cf_values = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0] # Center frequency
ep_values  = [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Dielectric constant

data = time_df[time_df['wave_ind'] == 2]['local_high0'].to_numpy().reshape(len(cf_values),len(ep_values))
data2 = time_df[time_df['wave_ind'] == 2]['local_low0'].to_numpy().reshape(len(cf_values),len(ep_values))

x = np.array(cf_values)
y = np.array(ep_values)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=30, azim=50)

# meshgrid default indexing='xy' return (len(y), len(x)) , data need to transpose
ax.plot_wireframe(X, Y, data.T, color='r')
ax.plot_wireframe(X, Y, data2.T, color='b')

ax.set_ylabel(r'$\epsilon_{r}$')
ax.set_xlabel('cf')
ax.set_zlabel('Z Axis')
ax.set_title('3D Surface Plot')

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cf_values = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0] # Center frequency
ep_values  = [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Dielectric constant

data = time_df[time_df['wave_ind'] == 3]['local_high0'].to_numpy().reshape(len(cf_values),len(ep_values))
data2 = time_df[time_df['wave_ind'] == 3]['local_low0'].to_numpy().reshape(len(cf_values),len(ep_values))

x = np.array(cf_values)
y = np.array(ep_values)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=30, azim=50)

# meshgrid default indexing='xy' return (len(y), len(x)) , data need to transpose
ax.plot_wireframe(X, Y, data.T, color='r')
ax.plot_wireframe(X, Y, data2.T, color='b')

ax.set_ylabel(r'$\epsilon_{r}$')
ax.set_xlabel('cf')
ax.set_zlabel('Z Axis')
ax.set_title('3D Surface Plot')

plt.show()





from scipy.interpolate import RegularGridInterpolator

cf_values = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0] # Center frequency
ep_values  = [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Dielectric constant

data = time_df[time_df['wave_ind'] == 0]['local_high0'].to_numpy().reshape(len(cf_values),len(ep_values))
data2 = time_df[time_df['wave_ind'] == 0]['local_low0'].to_numpy().reshape(len(cf_values),len(ep_values))

local_high0_t_interp_func = RegularGridInterpolator((cf_values, ep_values), data, method='linear')
local_min0_t_interp_func = RegularGridInterpolator((cf_values, ep_values), data2, method='linear')





import numpy as np
from scipy.optimize import minimize

# Objective function
def obj_time_function(freq, epsilon_r):
    global obs_local_high0_t, obs_local_min0_t
    tmp = (obs_local_high0_t - local_high0_t_interp_func((epsilon_r, freq)) )**2
    + (obs_local_min0_t - local_min0_t_interp_func((epsilon_r, freq)) )**2
    return tmp


test_data_path = TEST_DATA_PATH
num_test_samples = 100

test_list = []
test_list_nan_mask = []
for i in range(num_test_samples):
    filename = os.path.join(test_data_path, f"Testing_Bscan_{i}.npy")
    data = np.load(filename)
    
    # Replace NaN with 0
    data_nan_mask = np.isnan(data)
    data = np.nan_to_num(data, nan=0.0)
    
    test_list.append(data)
    test_list_nan_mask.append(data_nan_mask)

if test_list:
    test_data = np.array(test_list)
    test_nan_mask_data = np.array(test_list_nan_mask)
else:
    print("No valid test data found.")
    test_data = np.empty((0, 0, 0, 0))


def bscan_line_sum2(inputs_index0, test_data, offset, vn, t0m):
    result = 0
    for i in range(230):
        t = calc_direct_t(offset[i], vn, t0m)
        if test_nan_mask_data[inputs_index0, 0, i ]:
            continue
            
        if t <= 700:
            result += np.interp(t, np.linspace(0, 700, 230), test_data[inputs_index0, :, i])
        else:
            break
    return result


import pandas as pd

num_samples = 100

obs_local_max_min_time_dict = {}

# Read category information from a CSV file
# CSV has columns "Index,Value,Category"
df = pd.read_csv(os.path.join(CSV_DIR, r"modify_test_dataframe_output.csv"))

def calc_direct_t(offset,  v, t0=0.0):
    return  ( offset / v ) + t0

# Define vmin and vmax for each category in a dictionary
v_limits = {
    0: {"vmin": -1.5 * 10**18, "vmax": 1.5 * 10**18},
    1: {"vmin": -1.5 * 10**10, "vmax": 1.5 * 10**10},
    2: {"vmin": -1.5 * 10**10, "vmax": 1.5 * 10**10},
    3: {"vmin": -1.5 * 10**27, "vmax": 1.5 * 10**27},
}

v_limits40 = {
    0: {"vmin": -6 * 10**19, "vmax": 6 * 10**19},
    1: {"vmin": -6 * 10**11, "vmax": 6 * 10**11},
    2: {"vmin": -6 * 10**11, "vmax": 6 * 10**11},
    3: {"vmin": -6 * 10**28, "vmax": 6 * 10**28},
}

offset = np.linspace(0.02, 45.82, 230)
t_index = np.linspace(0, 700, 230)

peak_time_list = []

#for inputs_index0 in range(num_samples):
for inputs_index0 in [1]:
    
    row = df[df['Index'] == inputs_index0]
    if row.empty:
        print(f"Index {inputs_index0} のカテゴリー情報が見つかりません。")
        continue
    category = int(row['Category'].values[0])
    print('inputs_index0:', inputs_index0, 'category:', category)
    
    v = np.arange(0.25, 0.3+0.01, 0.01)  # Velocity
    t0 = np.arange(-30.0, 30.0, 0.5)
    result = np.zeros((len(v), len(t0)))

    # Calculate the value of bscan_line_sum when t0m is changed from 0 to 30 in increments of 0.2
    t0m_values = np.arange(0, 30.2, 0.2)
    vn = 0.3
    bscan_values = [bscan_line_sum2(inputs_index0, test_data, offset, vn, t) for t in t0m_values]

    max_v_index , max_t_index = np.unravel_index(np.argmax(result), result.shape)
    min_v_index , min_t_index = np.unravel_index(np.argmin(result), result.shape)
    
    # Get the indices of local maximum and minimum
    local_max_idx = argrelextrema(np.array(bscan_values), np.greater)[0]  # Local maximum
    local_min_idx = argrelextrema(np.array(bscan_values), np.less)[0]     # Local minimum

    obs_lp_high_time = []
    obs_lp_low_time = []
    vn = 0.3
    
    #local_max
    for lp_high in local_max_idx.tolist():
        initial_guess = t0m_values[lp_high]
        bounds = [(t0m_values[lp_high]-0.5, t0m_values[lp_high]+0.5)]
        objective = lambda t0m, inputs_index0=inputs_index0: -bscan_line_sum2(inputs_index0, test_data, offset, vn, t0m[0])
        result = minimize(objective, initial_guess, bounds=bounds)
        optimal_time = result.x[0]
        obs_lp_high_time.append(optimal_time)
        
    #local_min
    for lp_low in local_min_idx.tolist():
        initial_guess = t0m_values[lp_low]
        bounds = [(t0m_values[lp_low]-0.5, t0m_values[lp_low]+0.5)]
        objective = lambda t0m, inputs_index0=inputs_index0: bscan_line_sum2(inputs_index0, test_data, offset, vn, t0m[0])
        result = minimize(objective, initial_guess, bounds=bounds)
        optimal_time = result.x[0]
        obs_lp_low_time.append(optimal_time)

    obs_local_max_min_time_dict[inputs_index0] = {
        "local_max_time": lp_high_time,
        "local_min_time": lp_low_time
    }
    
    # observational data
    obs_local_high0_t = obs_lp_high_time[0]
    obs_local_min0_t = obs_lp_low_time[0]
    print('obs_local_high0_t:{}, obs_local_min0_t:{}'.format(obs_local_high0_t, obs_local_min0_t ))

    # init
    x0 = np.array([75.0, 5.0])

    bounds = [(np.array(cf_values).min(), np.array(cf_values).max()), (np.array(ep_values).min(), np.array(ep_values).max())]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    # result
    print("最適化結果:", result.x)
    print("最小値:", result.fun)
    
    continue
    
    #################################################

    img = plt.imshow(np.flipud(result.T),  extent=[0.25, 0.3, -30, 30], cmap='gray', aspect = 'auto' , **v_limits40[category])
    plt.xlabel("Velocity [m/ns]")
    plt.ylabel("Time [ns]")

    cbar = plt.colorbar(img)

    folder_path = FOLDER_PATH
    output_file = os.path.join(folder_path,"Test_vt_image_{:04d}.jpg".format(inputs_index0))
    plt.savefig(output_file)
    plt.close()

    if category == 0:
        peak_time = t0[max_t_index]
    elif category == 1:
        peak_time = t0[max_t_index]
    elif category == 2:
        peak_time = t0[min_t_index]
    else:
        peak_time = t0[max_t_index]
        
    print('peak_time:', peak_time)
    peak_time_list.append(peak_time)

    t_direct = calc_direct_t(offset, v=0.3, t0=peak_time)

    plt.figure(figsize=(6, 6))
    img = plt.imshow(test_data[inputs_index0], cmap='gray_r', extent=[0.02, 45.82, 700, 0], aspect = 'auto', **v_limits[category])  # カラーマップを指定可能 (例: 'viridis')

    plt.imshow(test_nan_mask_data[inputs_index0], cmap='Blues', extent=[0.02, 45.82, 700, 0], aspect = 'auto', alpha=test_nan_mask_data[inputs_index0].astype(float) )  # NaN の部分だけ青く塗る
    
    cbar = plt.colorbar(img)
    cbar.set_label('Amplitude')

    plt.plot(offset, t_direct , color ='red')

    plt.title('GPR Image {:04d}'.format(inputs_index0))
    plt.xlabel("Offset [m]")
    plt.ylabel("Time [ns]")

    folder_path = FOLDER_PATH
    output_file = os.path.join(folder_path,"Test_gpr_direct_line_{:04d}.jpg".format(inputs_index0))
    plt.savefig(output_file)
    plt.close()

    for i in range(230):
        plt.plot(t_index-peak_time-(offset[i]/0.3),test_data[inputs_index0,:,i])

    plt.xlim(-20,20)
    plt.ylim(v_limits[category]['vmin'],v_limits[category]['vmax'])

    folder_path = FOLDER_PATH
    output_file = os.path.join(folder_path,"Test_direct_wave_{:04d}.jpg".format(inputs_index0))
    plt.savefig(output_file)
    plt.close()


import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

cf_values = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0]  # Center frequency
ep_values = [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Dielectric constant

interp_funcs = {}

for wave_ind in range(4):
    df_wave = time_df[time_df['wave_ind'] == wave_ind]
    
    # Reshape array to interpolate on grid
    data_high = df_wave['local_high0'].to_numpy().reshape(len(cf_values), len(ep_values))
    data_low = df_wave['local_low0'].to_numpy().reshape(len(cf_values), len(ep_values))
    
    local_high0_t_interp_func = RegularGridInterpolator((cf_values, ep_values), data_high, method='linear')
    local_low0_t_interp_func  = RegularGridInterpolator((cf_values, ep_values), data_low, method='linear')
    
    interp_funcs[wave_ind] = {
        'local_high0_t': local_high0_t_interp_func,
        'local_low0_t': local_low0_t_interp_func
    }


import pandas as pd

num_samples = 100

obs_local_max_min_time_dict = {}

# Read category information from a CSV file
# CSV has columns "Index,Value,Category"
df = pd.read_csv(os.path.join(CSV_DIR, r"modify_test_dataframe_output.csv"))

def calc_direct_t(offset,  v, t0=0.0):
    return  ( offset / v ) + t0

# Define vmin and vmax for each category
v_limits = {
    0: {"vmin": -1.5 * 10**18, "vmax": 1.5 * 10**18},
    1: {"vmin": -1.5 * 10**10, "vmax": 1.5 * 10**10},
    2: {"vmin": -1.5 * 10**10, "vmax": 1.5 * 10**10},
    3: {"vmin": -1.5 * 10**27, "vmax": 1.5 * 10**27},
}

v_limits40 = {
    0: {"vmin": -6 * 10**19, "vmax": 6 * 10**19},
    1: {"vmin": -6 * 10**11, "vmax": 6 * 10**11},
    2: {"vmin": -6 * 10**11, "vmax": 6 * 10**11},
    3: {"vmin": -6 * 10**28, "vmax": 6 * 10**28},
}

offset = np.linspace(0.02, 45.82, 230)
t_index = np.linspace(0, 700, 230)

test_param_dict = {}

for inputs_index0 in range(num_samples):
    
    row = df[df['Index'] == inputs_index0]
    if row.empty:
        print(f"Index {inputs_index0} のカテゴリー情報が見つかりません。")
        continue
    category = int(row['Category'].values[0])
    print('inputs_index0:', inputs_index0, 'category:', category)


    # Calculate the value of bscan_line_sum when t0m is changed from 0 to 30 in increments of 0.2
    t0m_values = np.arange(0, 30.2, 0.2)
    vn = 0.3
    bscan_values = [bscan_line_sum2(inputs_index0, test_data, offset, vn, t) for t in t0m_values]

    # Get the indices of local maximum and minimum
    local_max_idx = argrelextrema(np.array(bscan_values), np.greater)[0]  # Local maximum
    local_min_idx = argrelextrema(np.array(bscan_values), np.less)[0]     # Local minimum

    obs_lp_high_time = []
    obs_lp_low_time = []
    vn = 0.3
    
    #local_max
    for lp_high in local_max_idx.tolist():
        initial_guess = t0m_values[lp_high]
        bounds = [(t0m_values[lp_high]-0.5, t0m_values[lp_high]+0.5)]
        objective = lambda t0m, inputs_index0=inputs_index0: -bscan_line_sum2(inputs_index0, test_data, offset, vn, t0m[0])
        result = minimize(objective, initial_guess, bounds=bounds)
        optimal_time = result.x[0]
        obs_lp_high_time.append(optimal_time)
        
    #local_min
    for lp_low in local_min_idx.tolist():
        initial_guess = t0m_values[lp_low]
        bounds = [(t0m_values[lp_low]-0.5, t0m_values[lp_low]+0.5)]
        objective = lambda t0m, inputs_index0=inputs_index0: bscan_line_sum2(inputs_index0, test_data, offset, vn, t0m[0])
        result = minimize(objective, initial_guess, bounds=bounds)
        optimal_time = result.x[0]
        obs_lp_low_time.append(optimal_time)

    obs_local_max_min_time_dict[inputs_index0] = {
        "local_max_time": lp_high_time,
        "local_min_time": lp_low_time
    }
    
    # observational data
    obs_local_high0_t = obs_lp_high_time[0]
    obs_local_low0_t = obs_lp_low_time[0]

    print('obs_local_high0_t:{}, obs_local_low0_t:{}'.format(obs_local_high0_t, obs_local_low0_t ))
    
    def obj_time_func(x, 
                          interp_high = interp_funcs[category]['local_high0_t'], 
                          interp_low = interp_funcs[category]['local_low0_t'],
                          obs_high = obs_local_high0_t, 
                          obs_low = obs_local_low0_t):

        error_high = obs_high - interp_high((x[0], x[1]))
        error_low  = obs_low  - interp_low((x[0], x[1]))
        return error_high**2 + error_low**2

    # init
    x0 = np.array([75.0, 5.0])  
    
    bounds = [(np.array(cf_values).min(), np.array(cf_values).max()), (np.array(ep_values).min(), np.array(ep_values).max())]

    result = minimize(obj_time_func, x0, method='L-BFGS-B', bounds=bounds)

    # result
    print("最適化結果:", result.x, ", 最小値:", result.fun)

    test_param_dict[inputs_index0] = {
        "obs_local_high0_t": obs_local_high0_t,
        "obs_local_low0_t": obs_local_low0_t,
        "cf": result.x[0],
        "ep": result.x[1],
        "obj_time_func": result.fun
    }



import json
folder_path = FOLDER_PATH
json_output_file = os.path.join(folder_path, "test_param_dict.json")
with open(json_output_file, "w") as f:
    json.dump(test_param_dict, f, indent=4)

print(f"JSONファイルとして保存しました: {json_output_file}")

