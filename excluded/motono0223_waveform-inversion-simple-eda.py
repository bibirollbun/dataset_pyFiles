import os

base_path = "/kaggle/input/waveform-inversion"
print(os.listdir(base_path))
if 'train_samples' in os.listdir(base_path):
    print("Train folder contents:", os.listdir(os.path.join(base_path, 'train_samples')))


import numpy as np

data_batch = np.load( "/kaggle/input/waveform-inversion/train_samples/CurveVel_A/data/data1.npy" )
model_batch = np.load( "/kaggle/input/waveform-inversion/train_samples/CurveVel_A/model/model1.npy" )
print("Data batch shape:", data_batch.shape)
print("Model batch shape:", model_batch.shape)

sample_seismic = data_batch[0]
sample_velocity = model_batch[0]
print("Single sample seismic data shape:", sample_seismic.shape)
print("Single sample velocity model shape:", sample_velocity.shape)


import matplotlib.pyplot as plt

def viz_waveform_data(sample_seismic):
    shot0 = sample_seismic[0]
    
    plt.figure(figsize=(6,4))
    plt.imshow(shot0.T, aspect='auto', cmap='seismic', origin='upper')
    plt.colorbar(label="Amplitude")
    plt.title("Waveform amplitudes (Time vs Receiver)")
    plt.xlabel("Time index")
    plt.ylabel("Receiver index")
    plt.show()
    
    plt.figure(figsize=(6,4))
    for rec_idx in [0, 35, 69]:
        trace = shot0[:, rec_idx]
        plt.plot(trace, label=f"Receiver {rec_idx}")
    plt.title("Sample waveforms at selected receivers")
    plt.xlabel("Time index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

viz_waveform_data(sample_seismic)


def viz_velocity_model( sample_velocity ):
    velocity_grid = sample_velocity.squeeze()
    
    plt.figure(figsize=(5,5))
    plt.imshow(velocity_grid, origin='upper', cmap='viridis')
    plt.colorbar(label="Velocity (m/s)")
    plt.title("Example Velocity Model")
    plt.xlabel("Horizontal position")
    plt.ylabel("Depth")
    plt.show()

viz_velocity_model( sample_velocity )


def viz_velocity_value_distribution(model_batch):
    all_velocities = model_batch.reshape(-1)
    print("Velocity stats – min:", all_velocities.min(), "max:", all_velocities.max(), 
          "mean:", all_velocities.mean())
    
    plt.figure(figsize=(6,4))
    plt.hist(all_velocities, bins=50, color='gray')
    plt.title("Distribution of Velocity Values (Sample Batch)")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Frequency")
    plt.show()

viz_velocity_value_distribution(model_batch)


series = [
    "CurveVel_A", 
    "CurveVel_B",
    "FlatVel_A",
    "FlatVel_B",
    "Style_A",
    "Style_B",
]

for family_name in series:
    data_file = f"/kaggle/input/waveform-inversion/train_samples/{family_name}/data/data1.npy"
    model_file = f"/kaggle/input/waveform-inversion/train_samples/{family_name}/model/model1.npy"
    print( "-" * 30 )
    print( f"\n// {family_name}" )
    data_batch = np.load( data_file )
    model_batch = np.load( model_file )
    sample_seismic = data_batch[0]
    sample_velocity = model_batch[0]

    viz_waveform_data(sample_seismic)
    viz_velocity_model( sample_velocity )
    viz_velocity_value_distribution(model_batch)

