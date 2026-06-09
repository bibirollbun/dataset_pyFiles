import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path


fpath = Path("/kaggle/input/waveform-inversion")


test_dir = fpath/'test'
train_dir = fpath/'train_samples'
samp_sub = pd.read_csv(fpath/'sample_submission.csv')


print("\nSample submission head:")
print(samp_sub.head())


data_test = np.load(test_dir/os.listdir(test_dir)[0])


# Plot all 5 slices as subplots in a single figure:
plt.clf() # Clear previous figure settings if any for this new subplot example
fig, axes = plt.subplots(nrows=1, ncols=data_test.shape[0], figsize=(20, 5))
if data_test.shape[0] == 1: axes = [axes]
else: axes = axes.flatten()

for i in range(data_test.shape[0]):
    ax = axes[i]
    im = ax.imshow(data_test[i, :, :], aspect='auto', cmap='viridis')
    ax.set_title(f'Slice {i}')
    ax.set_xlabel('Dim 70')
    if i == 0: ax.set_ylabel('Dim 1000')
    else: ax.set_yticks([])
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)


plt.suptitle('Visualization of 5 Slices (each 1000x70)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
plt.show()


# List available training family folders
if train_dir.exists():
    family_folders = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print(f"Available training families: {family_folders}")
else:
    print(f"ERROR: Training directory not found at {train_dir}")
    family_folders = []


# Define get_sample_pair_paths Function
def get_sample_pair_paths(family_path: Path):
    """
    Finds a pair of (seismic_data_file, velocity_model_file) paths from a given family folder.
    Handles two structures:
    1. Vel/Style families: data/*.npy in 'data/' subdir, model/*.npy in 'model/' subdir.
    2. Fault families: seis_*.npy and vel_*.npy directly in family_path.
    """
    data_subdir = family_path / 'data'
    model_subdir = family_path / 'model'

    seismic_file_path = None
    model_file_path = None

    if data_subdir.exists() and model_subdir.exists():
        # Vel or Style family structure
        print(f"Found 'data' and 'model' subdirectories in {family_path.name}. (Vel/Style family)")
        try:
            # Get the first .npy file from data subdir
            data_files = sorted([f for f in data_subdir.iterdir() if f.suffix == '.npy'])
            if not data_files:
                print(f"No .npy files found in {data_subdir}")
                return None, None
            seismic_file_path = data_files[0]

            # Try to find corresponding model file
            # e.g., data1.npy -> model1.npy
            model_file_name = seismic_file_path.name.replace('data', 'model', 1)
            potential_model_path = model_subdir / model_file_name
            
            if potential_model_path.exists():
                model_file_path = potential_model_path
            else:
                # Fallback: take the first model file if direct match fails
                print(f"Warning: Could not find direct match {model_file_name}, trying first model file.")
                model_files = sorted([f for f in model_subdir.iterdir() if f.suffix == '.npy'])
                if model_files:
                    model_file_path = model_files[0]
                else:
                    print(f"No .npy files found in {model_subdir}")
                    return None, None
            
        except IndexError:
            print(f"Error accessing files in {data_subdir} or {model_subdir}")
            return None, None
    else:
        # Fault family structure (files directly in family_path)
        print(f"No 'data'/'model' subdirs in {family_path.name}. Assuming Fault family structure.")
        all_npy_files_in_family = sorted([f for f in family_path.iterdir() if f.suffix == '.npy'])
        
        seis_files = [f for f in all_npy_files_in_family if f.name.startswith('seis_')]
        vel_files = [f for f in all_npy_files_in_family if f.name.startswith('vel_')]

        if seis_files and vel_files:
            # Try to find a matching pair
            for s_file in seis_files:
                expected_v_file_name = s_file.name.replace('seis_', 'vel_')
                # Check if the corresponding vel file exists by constructing its full path
                potential_v_file_path = family_path / expected_v_file_name
                if potential_v_file_path in vel_files: # Check if Path object is in list of Path objects
                    seismic_file_path = s_file
                    model_file_path = potential_v_file_path
                    break # Found a pair
            if not seismic_file_path:
                # Fallback: if no direct match, take the first of each
                print("Warning: Could not find a directly name-matched seis/vel pair. Using first found files.")
                seismic_file_path = seis_files[0]
                model_file_path = vel_files[0]
        else:
            print(f"No 'seis_'/'vel_' .npy file pairs found directly in {family_path.name}")
            return None, None
            
    return seismic_file_path, model_file_path


# Choose a Family and Get Specific File Paths

# --- You can change this to any family name from 'family_folders' ---
# Example for a Fault family:
# chosen_family_name = 'CurveFault_A'
# Example for a Vel family:
chosen_family_name = 'CurveVel_A'
# Example for a Style family:
# chosen_family_name = 'Style_A'

# Fallback if chosen_family_name isn't available
if not family_folders:
    print("Error: family_folders list is empty. Cannot proceed.")
    # Handle error appropriately, perhaps skip subsequent snippets
    seismic_data_filepath = None
    velocity_model_filepath = None
elif chosen_family_name not in family_folders:
    print(f"Warning: Chosen family '{chosen_family_name}' not found. Defaulting to first available: {family_folders[0]}")
    chosen_family_name = family_folders[0]
# --------------------------------------------------------------------

if family_folders: # Proceed only if family_folders is not empty
    chosen_family_path = train_dir / chosen_family_name
    print(f"\nExploring family: {chosen_family_name} (Path: {chosen_family_path})")

    seismic_data_filepath, velocity_model_filepath = get_sample_pair_paths(chosen_family_path)

    if seismic_data_filepath and velocity_model_filepath:
        print(f"  Seismic data file: {seismic_data_filepath}")
        print(f"  Velocity model file: {velocity_model_filepath}")
    else:
        print(f"Could not retrieve a valid file pair for {chosen_family_name}.")
else:
    # This else corresponds to the "if not family_folders:" check earlier
    print("Skipping file path retrieval as no family folders were found.")


# Load Data and Select a Sample

# Initialize variables to avoid NameError if previous snippet failed
seismic_data_batch = None
velocity_model_batch = None
sample_seismic_data = None
sample_velocity_model = None

if seismic_data_filepath and velocity_model_filepath:
    try:
        seismic_data_batch = np.load(seismic_data_filepath)
        velocity_model_batch = np.load(velocity_model_filepath)

        print(f"\nShape of loaded seismic data batch: {seismic_data_batch.shape}")
        print(f"Shape of loaded velocity model batch: {velocity_model_batch.shape}")

        # Pick the first sample from the batch for visualization
        # (batch_size is typically the first dimension)
        if seismic_data_batch.shape[0] > 0:
            sample_seismic_data = seismic_data_batch[0]
            print(f"Shape of one seismic data sample: {sample_seismic_data.shape}")
            # Expected: (num_sources, time_steps, num_receivers)
        else:
            print("Seismic data batch is empty.")

        if velocity_model_batch.shape[0] > 0:
            sample_velocity_model = velocity_model_batch[0]
            print(f"Shape of one velocity model sample: {sample_velocity_model.shape}")
            # Expected: (height, width)
        else:
            print("Velocity model batch is empty.")

    except Exception as e:
        print(f"Error loading .npy files: {e}")
else:
    print("\nSkipping data loading as file paths were not resolved.")


# Visualize Velocity Model

if sample_velocity_model is not None:
    plt.figure(figsize=(8, 6))
    plt.imshow(sample_velocity_model.squeeze(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Velocity (e.g., m/s)')
    plt.title(f'Velocity Model (Sample 0 from {chosen_family_name})')
    plt.xlabel('Width Index')
    plt.ylabel('Height (Depth) Index')
    plt.tight_layout()
    plt.show()
else:
    print("\nSkipping velocity model visualization as data is not loaded.")


# Visualize a Single Seismogram

if sample_seismic_data is not None:
    try:
        num_sources, time_steps, num_receivers = sample_seismic_data.shape
        
        # Define which source and receiver to plot
        source_idx_to_plot = 0 
        receiver_idx_to_plot = 0

        if num_sources > source_idx_to_plot and num_receivers > receiver_idx_to_plot:
            single_seismogram = sample_seismic_data[source_idx_to_plot, :, receiver_idx_to_plot]
            
            plt.figure(figsize=(12, 4))
            plt.plot(single_seismogram)
            plt.title(f'Seismogram: Source {source_idx_to_plot}, Receiver {receiver_idx_to_plot}\n(Sample 0 from {chosen_family_name})')
            plt.xlabel('Time Step Index')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print(f"Cannot plot seismogram: Source index {source_idx_to_plot} or Receiver index {receiver_idx_to_plot} out of bounds.")
            print(f"(Available: {num_sources} sources, {num_receivers} receivers)")
            
    except ValueError: # If sample_seismic_data doesn't have 3 dimensions
        print(f"Error: sample_seismic_data does not have the expected 3 dimensions. Shape is {sample_seismic_data.shape}")
else:
    print("\nSkipping seismogram visualization as data is not loaded.")


# Visualize a Shot Record

if sample_seismic_data is not None:
    try:
        num_sources, time_steps, num_receivers = sample_seismic_data.shape
        source_idx_to_view = 0 # View data from the first source

        if num_sources > source_idx_to_view:
            shot_record = sample_seismic_data[source_idx_to_view, :, :] # Shape: (time_steps, num_receivers)

            plt.figure(figsize=(10, 7))
            # Plot with time on y-axis, receivers on x-axis (a common convention)
            # Transpose shot_record to have time_steps as columns for imshow if receivers are the primary "image width"
            # Or plot directly if time_steps are the "image width"
            # Let's plot time_steps along x-axis, receivers along y-axis for consistency with previous imshow.
            # extent=[x_min, x_max, y_min, y_max]
            plt.imshow(shot_record.T, aspect='auto', cmap='seismic', extent=[0, time_steps, num_receivers, 0])
            plt.colorbar(label='Amplitude')
            plt.title(f'Shot Record for Source {source_idx_to_view}\n(Sample 0 from {chosen_family_name})')
            plt.xlabel('Time Step Index')
            plt.ylabel('Receiver Index')
            plt.tight_layout()
            plt.show()
        else:
            print(f"Cannot plot shot record: Source index {source_idx_to_view} out of bounds (Num sources: {num_sources}).")
    except ValueError:
         print(f"Error: sample_seismic_data does not have the expected 3 dimensions. Shape is {sample_seismic_data.shape}")
else:
    print("\nSkipping shot record visualization as data is not loaded.")


# Visualize Multiple Shot Records

if sample_seismic_data is not None:
    try:
        num_sources, time_steps, num_receivers = sample_seismic_data.shape
        
        # Safety check: plot up to 'max_sources_to_plot' or all if fewer
        max_sources_to_plot = min(num_sources, 5) 
        
        if max_sources_to_plot > 0:
            fig, axes = plt.subplots(nrows=1, ncols=max_sources_to_plot, figsize=(4 * max_sources_to_plot, 7), squeeze=False)
            # squeeze=False ensures axes is always 2D, even if nrows or ncols is 1. Access with axes[0, i]
            
            for i in range(max_sources_to_plot):
                ax = axes[0, i] # Access subplot from the 2D array
                current_shot_record = sample_seismic_data[i, :, :] # Data for i-th source
                
                im = ax.imshow(current_shot_record.T, aspect='auto', cmap='seismic', extent=[0, time_steps, num_receivers, 0])
                ax.set_title(f'Source {i}')
                ax.set_xlabel('Time Steps')
                if i == 0:
                    ax.set_ylabel('Receivers')
                else:
                    ax.set_yticks([]) # Hide y-ticks for subsequent subplots for clarity
            
            # Add a single colorbar for the entire figure
            # Adjust 'right' to make space for colorbar, 'wspace' for space between subplots
            fig.subplots_adjust(right=0.90, wspace=0.3) 
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
            fig.colorbar(im, cax=cbar_ax, label='Amplitude')

            plt.suptitle(f'Shot Records (First {max_sources_to_plot} Sources)\n(Sample 0 from {chosen_family_name})', fontsize=16)
            # Adjust layout to make space for suptitle. rect=[left, bottom, right, top]
            # This might need fine-tuning after seeing the plot with the colorbar.
            # plt.tight_layout(rect=[0, 0, 0.9, 0.95]) # tight_layout might conflict with add_axes, adjust manually or remove
            plt.show()
        else:
            print("No sources found in the sample seismic data to plot multiple shot records.")
    except ValueError:
        print(f"Error: sample_seismic_data does not have the expected 3 dimensions. Shape is {sample_seismic_data.shape}")

else:
    print("\nSkipping multiple shot records visualization as data is not loaded.")




