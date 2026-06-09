!pip install -q zarr ome-zarr copick


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zarr
from pathlib import Path


train_zarr_path = Path('/kaggle/input/czii-cryo-et-object-identification/train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr')
train_zarr_data = zarr.open(str(train_zarr_path))

# printing data about zarr store
print("Zarr file contains:")
print(train_zarr_data.tree())


tomogram_ts_5_4 = train_zarr_data[0][:]

print(f"Tomogram shape: {tomogram_ts_5_4.shape}")
print(f"Tomogram DataType: {tomogram_ts_5_4.dtype}")
print(f"Tomogram minimum value: {tomogram_ts_5_4.min()}")
print(f"Tomogram maximum value: {tomogram_ts_5_4.max()}")
print(f"Tomogram Mean value: {tomogram_ts_5_4.mean()}")



import json

apo_ferritin_json_path = Path('/kaggle/input/czii-cryo-et-object-identification/train/overlay/ExperimentRuns/TS_5_4/Picks/apo-ferritin.json')

with open(apo_ferritin_json_path, 'r') as f:
    data = json.load(f)

# Visualize the structure
print("Keys in the JSON file: ", data.keys())
print("First few points:");
print(json.dumps(data['points'][:2], indent=2))


apo_data = data
coords = list()
# inside each point we have location
for point in apo_data["points"]:
    x, y, z = (point["location"][key] for key in ['x', 'y', 'z'])
    coords.append([x, y, z])
coords = np.array(coords)
print(f"Total points coordinates: {len(coords)}")
print(f"X range: {coords[:,0].min()} to {coords[0].max()}")
print(f"Y range: {coords[:,1].min()} to {coords[1].max()}")
print(f"Z range: {coords[:,2].min()} to {coords[2].max()}")



# tomogram shape
tomogram_ts_5_4.shape


def scale_coordinates(coords, tomogram_shape):
    scaled_coords = coords.copy()
    
    # Scale factors for each dimension
    scale_z = tomogram_shape[0] / coords[:, 2].max() # 184 / 5744.509
    scale_y = tomogram_shape[1] / coords[:, 1].max()
    scale_x = tomogram_shape[2] / coords[:, 0].max()
    
    # Apply scaling
    scaled_coords[:, 2] = coords[:, 2] * scale_z
    scaled_coords[:, 1] = coords[:, 1] * scale_y
    scaled_coords[:, 0] = coords[:, 0] * scale_x
    
    return scaled_coords

# Scale the coordinates
scaled_coords = scale_coordinates(coords, tomogram_ts_5_4.shape)

print("\nCoordinate ranges after scaling:")
print(f"X range: {scaled_coords[:, 0].min():.1f} to {scaled_coords[:, 0].max():.1f}")
print(f"Y range: {scaled_coords[:, 1].min():.1f} to {scaled_coords[:, 1].max():.1f}")
print(f"Z range: {scaled_coords[:, 2].min():.1f} to {scaled_coords[:, 2].max():.1f}")


import matplotlib.pyplot as plt
import numpy as np

def visualize_apo_ferritin_particle(tomogram, coords, n_slices=3, slice_thickness=10):
    """
    Visualizing apo-ferritin particles in tomogram slices using subplots.
    
    Parameters:
        tomogram (ndarray): 3D tomogram array.
        coords (ndarray): Array of particle coordinates [x, y, z].
        n_slices (int): Number of slices to display.
        slice_thickness (int): Range around each slice to include particles.
    """
    # Normalize tomogram for consistent visualization
    vmin, vmax = np.percentile(tomogram, (1, 99))
    normalized_tomogram = np.clip((tomogram - vmin) / (vmax - vmin), 0, 1)

    # Calculate z-positions for the slices
    z_positions = np.linspace(0, tomogram.shape[0] - 1, n_slices, dtype=int)

    # Create subplots
    fig, axes = plt.subplots(1, n_slices, figsize=(20, 8), constrained_layout=True)
    
    for idx, ax in enumerate(axes):
        z = z_positions[idx]

        # Display the tomogram slice
        im = ax.imshow(normalized_tomogram[z, :, :], cmap='gray', vmin=0, vmax=1)
        
        # Highlight particles near the current slice
        mask = np.abs(coords[:, 2] - z) < slice_thickness
        if np.any(mask):
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                color='red', marker='o', s=100,
                facecolors='none', linewidth=2,
                label='apo-ferritin'
            )
        
        # Customize the subplot
        ax.set_title(f"Slice Z={z}\n({np.sum(mask)} particles visible)")
        ax.set_xlim(0, tomogram.shape[2])
        ax.set_ylim(tomogram.shape[1], 0)  # Invert y-axis for correct orientation
        ax.axis('off')  # Turn off axes for cleaner visualization
    
    # Add a single colorbar for the figure
    cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.7, pad=0.05)
    cbar.set_label('Normalized Intensity', fontsize=12)

    # Add a shared title
    fig.suptitle(
        f"Apo-ferritin Particles in Tomogram Slices\n"
        f"Displaying particles within ±{slice_thickness} units of each slice",
        fontsize=16
    )

    # Show legend on the first subplot
    axes[0].legend(loc='upper right')
    
    plt.show()



# Create the visualization with scaled coordinates
visualize_apo_ferritin_particle(tomogram_ts_5_4, scaled_coords)


PARTICLE_MAP = {
    'apo-ferritin': {'color': '#FF3333', 'marker': 'o', 'difficulty': 'easy'},          # Bright red
    'beta-amylase': {'color': '#FFFFFF', 'marker': 's', 'difficulty': 'impossible'},     # White
    'beta-galactosidase': {'color': '#33FFFF', 'marker': '^', 'difficulty': 'hard'},    # Cyan
    'ribosome': {'color': '#33FF33', 'marker': 'D', 'difficulty': 'easy'},              # Bright green
    'thyroglobulin': {'color': '#FF33FF', 'marker': 'p', 'difficulty': 'hard'},         # Magenta
    'virus-like-particle': {'color': '#FFFF33', 'marker': '*', 'difficulty': 'easy'}     # Yellow
}

EXPERIMENT_NAMES = ["TS_5_4", "TS_69_2", "TS_6_4", "TS_6_6", "TS_73_6", "TS_86_3", "TS_99_9"]


def load_particle_coordinates(experiment_name="TS_5_4"):
    base_path = Path("/kaggle/input/czii-cryo-et-object-identification/train/overlay/ExperimentRuns")
    particle_coords = {}
    print("-" * 50)
    for particle in PARTICLE_MAP:
        particle_path = base_path / experiment_name / 'Picks' / f'{particle}.json'
        try:
            with open(particle_path, 'r') as f:
                data = json.load(f)
                coords = list()
                # inside each point we have location
                for point in data["points"]:
                    x, y, z = (point["location"][key] for key in ['x', 'y', 'z'])
                    coords.append([x, y, z])
                coords = np.array(coords)
                particle_coords[particle] = coords
                print(f"Loaded {len(coords)} {particle} coordinates")
        except Exception as e:
            print(f"Error reading {particle} coordinates: {e}")
            particle_coords[particle] = np.array([])
    return particle_coords


def visualize_all_particles(tomogram, particle_coords, experiment_name="TS_5_4", print_statistics=True, n_slices=3, slice_thickness=10):
    """
    Visualizing all particles in tomogram slices using subplots.
    
    Parameters:
        tomogram (ndarray): 3D tomogram array.
        coords (ndarray): Array of particle coordinates [x, y, z].
        n_slices (int): Number of slices to display.
        slice_thickness (int): Range around each slice to include particles.
    """
    # Normalize tomogram for consistent visualization
    vmin, vmax = np.percentile(tomogram, (1, 99))
    normalized_tomogram = np.clip((tomogram - vmin) / (vmax - vmin), 0, 1)

    # Calculate z-positions for the slices
    all_z_coords = []
    for coords in particle_coords.values():
        if len(coords):
            all_z_coords.extend(coords[:, 2])
    
    if all_z_coords:
        z_coords = np.array(all_z_coords)
        z_density = np.histogram(z_coords, bins=50)[0]
        highest_density_indices = np.argsort(z_density)[-n_slices:]
        z_positions = np.linspace(z_coords.min(), z_coords.max(), 51)[highest_density_indices]
    else:
        z_positions = np.linspace(0, tomogram.shape[0]-1, n_slices, dtype=int)
    

    # Create subplots
    fig, axes = plt.subplots(1, n_slices, figsize=(20, 8), constrained_layout=True)
    
    for idx, ax in enumerate(axes):
        z = int(z_positions[idx])

        # Display the tomogram slice
        im = ax.imshow(normalized_tomogram[z, :, :], cmap='gray', vmin=0, vmax=1)

        # plot each particle type
        particles_in_slice = 0
        particle_counts = {}
        for particle_type, coords in particle_coords.items():
            if len(coords):
                
        
                # Highlight particles near the current slice
                mask = np.abs(coords[:, 2] - z) < slice_thickness
                if np.any(mask):
                    style = PARTICLE_MAP[particle_type]
                    ax.scatter(
                        coords[mask, 0], coords[mask, 1],
                        color=style['color'], marker=style['marker'], s=100,
                        facecolors='none', linewidth=2,
                        label=f"{particle_type}\n({style['difficulty']})"
                    )
                    count = np.sum(mask)
                    particles_in_slice += count
                    particle_counts[particle_type] = count 
        
        # Customize the subplot
        # Create detailed title showing counts for each particle type
        title_parts = [f'Slice Z={z}']
        if particle_counts:
            for ptype, count in particle_counts.items():
                if count > 0:
                    title_parts.append(f'{ptype}: {count}')
        title = '\n'.join(title_parts)
        ax.set_title(title, fontsize=8)

        
        ax.set_xlim(0, tomogram.shape[2])
        ax.set_ylim(tomogram.shape[1], 0)  # Invert y-axis for correct orientation
        ax.axis('off')  # Turn off axes for cleaner visualization

        # Add legend with semi-transparent background for better visibility
        if idx == 0:  # Only add legend to first subplot
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(handles, labels,
                             bbox_to_anchor=(0.02, 0.98), 
                             loc='upper left',
                             borderaxespad=0.,
                             framealpha=0.8,
                             facecolor='black',
                             edgecolor='white',
                             labelcolor='white',
                             fontsize=8)
            
            for handle in handles:
                handle.set_edgecolor('black')
                handle.set_linewidth(1.5)
    
    # Add a single colorbar for the figure
    cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.7, pad=0.05)
    cbar.set_label('Normalized Intensity', fontsize=12)

    # Add a shared title
    fig.suptitle(f'All Particle Types for experiment {experiment_name} in Tomogram Slices\n' + 
             f'Showing particles within ±{slice_thickness} units of each slice',
             fontsize=16, y=1.05)

    # Show legend on the first subplot
    axes[0].legend(loc='upper right')
    
    plt.show()
    if print_statistics:
        # Print overall particle statistics
        print("\nOverall Particle Statistics:")
        print("-" * 50)
        for particle_type, coords in particle_coords.items():
            if len(coords) > 0:
                print(f"\n{particle_type} ({PARTICLE_MAP[particle_type]['difficulty']}):")
                print(f"Total particles: {len(coords)}")
                print(f"Z range: {coords[:, 2].min():.1f} to {coords[:, 2].max():.1f}")


all_particle_coords = load_particle_coordinates()
scale_particle_coords = { particle_type: scale_coordinates(coords, tomogram_ts_5_4.shape)
    for particle_type, coords in all_particle_coords.items() }
visualize_all_particles(tomogram_ts_5_4, scale_particle_coords)
# Print statistics for each particle type
print("\nParticle Statistics:")
print("-" * 50)
for particle_type, coords in scale_particle_coords.items():
    if len(coords) > 0:
        print(f"\n{particle_type} ({PARTICLE_MAP[particle_type]['difficulty']}):")
        print(f"Number of particles: {len(coords)}")
        print(f"X range: {coords[:, 0].min():.1f} to {coords[:, 0].max():.1f}")
        print(f"Y range: {coords[:, 1].min():.1f} to {coords[:, 1].max():.1f}")
        print(f"Z range: {coords[:, 2].min():.1f} to {coords[:, 2].max():.1f}")


# FIRST findout all the tomogram shapes for each experiment
# we already have tomogram data for ts_5_4
EXPERIMENT_MAP = {} # name: tomogram
for experiment_name in EXPERIMENT_NAMES:
    train_zarr_path = Path(f'/kaggle/input/czii-cryo-et-object-identification/train/static/ExperimentRuns/{experiment_name}/VoxelSpacing10.000/denoised.zarr')
    train_zarr_data = zarr.open(str(train_zarr_path))
    
    # printing data about zarr store
    print(f"{experiment_name} Zarr file contains:")
    print(train_zarr_data.tree())
    tomogram = train_zarr_data[0][:]
    EXPERIMENT_MAP[experiment_name] = tomogram



# visualizing
for experiment_name, tomogram in EXPERIMENT_MAP.items():
    all_particle_coords = load_particle_coordinates(experiment_name)
    scale_particle_coords = { particle_type: scale_coordinates(coords, tomogram.shape)
        for particle_type, coords in all_particle_coords.items() }
    visualize_all_particles(tomogram, scale_particle_coords, experiment_name, print_statistics=False)
    # Print statistics for each particle type
    print(f"\nParticle Statistics for {experiment_name}:")
    print("-" * 50)
    for particle_type, coords in scale_particle_coords.items():
        if len(coords) > 0:
            print(f"\n{particle_type} ({PARTICLE_MAP[particle_type]['difficulty']}):")
            print(f"Number of particles: {len(coords)}")
            print(f"X range: {coords[:, 0].min():.1f} to {coords[:, 0].max():.1f}")
            print(f"Y range: {coords[:, 1].min():.1f} to {coords[:, 1].max():.1f}")
            print(f"Z range: {coords[:, 2].min():.1f} to {coords[:, 2].max():.1f}")

