from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
import ipywidgets as widgets
from IPython.display import display


data_dir = '/kaggle/input/waveform-inversion/train_samples'


data_dict = defaultdict(lambda: defaultdict(list))

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.npy'):
            category = os.path.basename(root)  # last part of the path (e.g., model, data)
            scenario = os.path.basename(os.path.dirname(root)) if category in ['model', 'data'] else os.path.basename(root)
            file_type = None

            if 'seis' in file:
                file_type = 'seismic'
            elif 'vel' in file:
                file_type = 'velocity'
            elif 'model' in file:
                file_type = 'model'
            elif 'data' in file:
                file_type = 'data'

            if file_type:
                full_path = os.path.join(root, file)
                data_dict[scenario][file_type].append(full_path)

# Example: print file paths in FlatVel_A
for key, value in data_dict.items():
    print(f"Scenario: {key}")
    for ftype, flist in value.items():
        print(f"  {ftype}:")
        for f in flist:
            print(f"    {f}")


# Example: Load one seismic and one model from data_dict
seis_sample = np.load(data_dict['FlatFault_A']['seismic'][0])
vel_sample = np.load(data_dict['FlatFault_A']['velocity'][0])
model_sample = np.load(data_dict['FlatVel_A']['model'][0])
data_sample = np.load(data_dict['FlatVel_A']['data'][0])

print("Seismic shape:", seis_sample.shape)
print("Velocity shape:", vel_sample.shape)
print("Model shape:", model_sample.shape)
print("Data shape:", data_sample.shape)


sample_index = 50
num_sources = 5
seis_sample = seis_sample[sample_index]  # shape: (num_sources, time_steps, num_receivers)

plt.figure(figsize=(12, 4))
for i in range(num_sources):
    plt.subplot(1, 5, i + 1)
    plt.imshow(seis_sample[i], aspect='auto', cmap='seismic')
    plt.title(f'Source {i}')
    plt.gca().xaxis.set_label_position('top') 
    plt.gca().xaxis.tick_top()  # move ticks to top
    if i == 0:
        plt.ylabel('Time Step')
    else:
        plt.yticks([])
plt.suptitle(f'Seismic Waveforms for Sample {sample_index}', fontsize=16)
plt.tight_layout()
plt.show()



def plot_all(scenario, sample_index):
    seis_array = np.load(data_dict[scenario]['seismic'][0])
    vel_array = np.load(data_dict[scenario]['velocity'][0])
    
    # Related velocity and model might be stored in *_Vel scenario (example: FlatFault_A -> FlatVel_A)
    base_name = scenario.replace("Fault", "Vel") if "Fault" in scenario else scenario.replace("Curve", "Vel")
    model_array = np.load(data_dict[base_name]['model'][0])
    data_array = np.load(data_dict[base_name]['data'][0])
    
    seis_sample = seis_array[sample_index]     # (5, 1000, 70)
    vel_sample = vel_array[sample_index, 0]    # (70, 70)
    model_sample = model_array[sample_index, 0]  # (70, 70)
    data_sample = data_array[sample_index]     # (5, 1000, 70)

    num_sources = seis_sample.shape[0]
    
    fig, axes = plt.subplots(3, num_sources + 1, figsize=(15, 10))

    for i in range(num_sources):
        # Seismic plot
        ax = axes[0, i]
        ax.imshow(seis_sample[i], aspect='auto', cmap='seismic')
        ax.set_title(f'Seismic {i+1}')
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.set_xlabel('Receiver')
        ax.set_ylabel('Time' if i == 0 else "")
        if i != 0:
            ax.set_yticks([])

        # Data plot
        ax = axes[1, i]
        ax.imshow(data_sample[i], aspect='auto', cmap='seismic')
        ax.set_title(f'Data {i+1}')
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.set_xlabel('Receiver')
        ax.set_ylabel('Time' if i == 0 else "")
        if i != 0:
            ax.set_yticks([])

    # Add Velocity and Model to last column
    ax = axes[0, -1]
    im1 = ax.imshow(vel_sample, aspect='auto', cmap='viridis')
    ax.set_title('Velocity')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, -1]
    im2 = ax.imshow(model_sample, aspect='auto', cmap='viridis')
    ax.set_title('Model')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # Hide third row if not used
    for ax in axes[2]:
        ax.axis('off')

    fig.suptitle(f'Visualization for {scenario} - Sample {sample_index}', fontsize=16)
    plt.tight_layout()
    plt.show()


scenarios = [s for s in data_dict.keys() if 'seismic' in data_dict[s]]
sample_index = 25 # 1 to 500
plot_all(scenarios[0], sample_index)




