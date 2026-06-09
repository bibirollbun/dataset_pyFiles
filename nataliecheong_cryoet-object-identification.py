!pip install -q zarr 

!pip install -q copick


import os
import glob
import zarr
import numpy as np
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Set paths
base_dir = '/kaggle/input/czii-cryo-et-object-identification'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')


# Function to explore directory structure
def explore_directory(directory, max_depth=3):
    """
    Recursively explore a directory structure up to a specified depth
    """
    result = []
    
    def _explore(dir_path, current_depth=0):
        if current_depth > max_depth:
            return
        
        for item in os.listdir(dir_path):
            path = os.path.join(dir_path, item)
            prefix = "  " * current_depth
            
            if os.path.isdir(path):
                result.append(f"{prefix}ğŸ“� {item}/")
                _explore(path, current_depth + 1)
            else:
                result.append(f"{prefix}ğŸ“„ {item}")
    
    try:
        _explore(directory)
    except Exception as e:
        result.append(f"Error exploring {directory}: {str(e)}")
    
    return result


# Function to visualize data
def visualize_data(data, title="Data Visualization", n_slices=3):
    """
    Visualize slices of a 3D or 4D array
    """
    if data is None:
        print("No data to visualize")
        return
    
    # Handle different data dimensions
    if len(data.shape) == 4:
        print(f"4D data of shape {data.shape}, taking first channel/time point")
        data = data[0]  # Take first channel/time point
    
    if len(data.shape) != 3:
        print(f"Cannot visualize data of shape {data.shape}")
        return
    
    # Get dimensions
    depth, height, width = data.shape
    print(f"Data dimensions: {depth} x {height} x {width}")
    
    # Choose slice indices
    slice_indices = np.linspace(depth // 4, 3 * depth // 4, n_slices).astype(int)
    
    # Create figure
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5))
    if n_slices == 1:
        axes = [axes]
    
    # Plot each slice
    for i, slice_idx in enumerate(slice_indices):
        im = axes[i].imshow(data[slice_idx], cmap='gray')
        axes[i].set_title(f'Slice {slice_idx}/{depth}')
        axes[i].axis('off')
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    # Histogram of values
    plt.figure(figsize=(10, 6))
    plt.hist(data.flatten(), bins=100, alpha=0.7, color='blue')
    plt.title(f"Value Distribution - {title}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.show()



# Function to explore a zarr file's structure
def explore_zarr(zarr_path, max_items=5):
    """
    Explore a zarr file's structure and metadata
    """
    print(f"Exploring zarr at: {zarr_path}")
    
    # First, explore the file system structure
    print("\nFile system structure:")
    structure = explore_directory(zarr_path, max_depth=4)
    for line in structure[:30]:  # Limit to first 30 lines
        print(line)
    if len(structure) > 30:
        print("... (truncated)")
    
    # Check for .zattrs file which contains metadata
    zattrs_path = os.path.join(zarr_path, '.zattrs')
    if os.path.exists(zattrs_path):
        print("\nFound .zattrs file. Contents:")
        try:
            with open(zattrs_path, 'r') as f:
                zattrs = json.load(f)
            print(json.dumps(zattrs, indent=2)[:1000] + "..." if len(json.dumps(zattrs)) > 1000 else "")
            
            # Check for multiscales metadata
            if 'multiscales' in zattrs:
                print("\nMultiscales metadata:")
                multiscales = zattrs['multiscales']
                print(json.dumps(multiscales, indent=2)[:1000] + "..." if len(json.dumps(multiscales)) > 1000 else "")
                
                # Extract dataset paths
                if multiscales and 'datasets' in multiscales[0]:
                    datasets = multiscales[0]['datasets']
                    print(f"\nDataset paths: {[d.get('path') for d in datasets]}")
        except Exception as e:
            print(f"Error reading .zattrs: {str(e)}")
    
    # Look for .zarray files which define arrays
    zarray_files = glob.glob(os.path.join(zarr_path, "**", ".zarray"), recursive=True)
    if zarray_files:
        print(f"\nFound {len(zarray_files)} .zarray files:")
        for i, zarray_file in enumerate(zarray_files[:max_items]):
            print(f"  {i+1}. {os.path.relpath(zarray_file, zarr_path)}")
            try:
                with open(zarray_file, 'r') as f:
                    zarray = json.load(f)
                print(f"    Shape: {zarray.get('shape')}")
                print(f"    Chunks: {zarray.get('chunks')}")
                print(f"    Data type: {zarray.get('dtype')}")
            except Exception as e:
                print(f"    Error reading .zarray: {str(e)}")
    
    # Try to open the zarr file
    try:
        z = zarr.open(zarr_path, mode='r')
        
        # Check if it's a Group
        if isinstance(z, zarr.Group):
            print("\nZarr opened as a Group")
            print(f"Keys: {list(z.keys())}")
            
            # Explore each key
            for key in list(z.keys())[:max_items]:
                print(f"\nExploring group '{key}':")
                try:
                    item = z[key]
                    if isinstance(item, zarr.Array):
                        print(f"  Array shape: {item.shape}")
                        print(f"  Data type: {item.dtype}")
                        print(f"  Chunks: {item.chunks}")
                    elif isinstance(item, zarr.Group):
                        print(f"  Subgroup with keys: {list(item.keys())}")
                except Exception as e:
                    print(f"  Error exploring group: {str(e)}")
        
        # Check if it's an Array
        elif isinstance(z, zarr.Array):
            print("\nZarr opened as an Array")
            print(f"Shape: {z.shape}")
            print(f"Data type: {z.dtype}")
            print(f"Chunks: {z.chunks}")
            
            # Print some sample data if small enough
            if np.prod(z.shape) < 100:
                print(f"Data sample: {z[:]}")
            else:
                print("Data too large to display sample")
        
        else:
            print(f"\nZarr opened as unknown type: {type(z)}")
    
    except Exception as e:
        print(f"\nError opening zarr: {str(e)}")
    
    return



# Function to try loading and visualizing data from a zarr file
def try_load_visualize(zarr_path):
    """
    Try different approaches to load and visualize data from a zarr file
    """
    print(f"Attempting to load and visualize data from: {zarr_path}")
    
    # Try approach 1: Direct array loading if zarr_path points to a zarr array
    try:
        print("\nApproach 1: Direct array loading")
        z1 = zarr.open(zarr_path, mode='r')
        if isinstance(z1, zarr.Array):
            print(f"Found array of shape: {z1.shape}")
            data1 = z1[:]
            visualize_data(data1, title="Approach 1: Direct array")
            return data1
    except Exception as e:
        print(f"Approach 1 failed: {str(e)}")
    
    # Try approach 2: Using multiscales metadata if available
    try:
        print("\nApproach 2: Using multiscales metadata")
        zattrs_path = os.path.join(zarr_path, '.zattrs')
        if os.path.exists(zattrs_path):
            with open(zattrs_path, 'r') as f:
                zattrs = json.load(f)
            
            if 'multiscales' in zattrs and 'datasets' in zattrs['multiscales'][0]:
                datasets = zattrs['multiscales'][0]['datasets']
                if datasets:
                    dataset_path = datasets[0].get('path')
                    if dataset_path:
                        full_path = os.path.join(zarr_path, dataset_path)
                        print(f"Loading dataset from: {full_path}")
                        z2 = zarr.open(full_path, mode='r')
                        if isinstance(z2, zarr.Array):
                            print(f"Found array of shape: {z2.shape}")
                            data2 = z2[:]
                            visualize_data(data2, title="Approach 2: Multiscales")
                            return data2
    except Exception as e:
        print(f"Approach 2 failed: {str(e)}")
    
    # Try approach 3: Find .zarray files and try to load data from their parent directories
    try:
        print("\nApproach 3: Find .zarray files and load from their parent directories")
        zarray_files = glob.glob(os.path.join(zarr_path, "**", ".zarray"), recursive=True)
        if zarray_files:
            for i, zarray_file in enumerate(zarray_files[:3]):  # Try first 3 .zarray files
                data_dir = os.path.dirname(zarray_file)
                print(f"Trying to load from: {data_dir}")
                try:
                    z3 = zarr.open(data_dir, mode='r')
                    if isinstance(z3, zarr.Array):
                        print(f"Found array of shape: {z3.shape}")
                        data3 = z3[:]
                        visualize_data(data3, title=f"Approach 3: Array {i+1}")
                        return data3
                except Exception as e:
                    print(f"Failed to load from {data_dir}: {str(e)}")
    except Exception as e:
        print(f"Approach 3 failed: {str(e)}")
    
    # Try approach 4: Traverse numeric subdirectories to find data
    try:
        print("\nApproach 4: Traverse numeric subdirectories")
        # Open zarr root
        z4 = zarr.open(zarr_path, mode='r')
        
        # Check if it's a Group
        if isinstance(z4, zarr.Group):
            # Look for numeric keys that might contain the data
            numeric_keys = [k for k in z4.keys() if k.isdigit()]
            if numeric_keys:
                print(f"Found numeric keys: {numeric_keys}")
                
                # Try to traverse the structure through a chain of numeric keys
                current_group = z4
                traversal_path = []
                
                # Try up to depth 5
                for _ in range(5):
                    numeric_keys = [k for k in current_group.keys() if k.isdigit()]
                    if not numeric_keys:
                        break
                    
                    next_key = numeric_keys[0]  # Take first numeric key
                    traversal_path.append(next_key)
                    
                    current_group = current_group[next_key]
                    if isinstance(current_group, zarr.Array):
                        print(f"Found array at path: {'/'.join(traversal_path)}")
                        print(f"Array shape: {current_group.shape}")
                        data4 = current_group[:]
                        visualize_data(data4, title="Approach 4: Numeric traversal")
                        return data4
    except Exception as e:
        print(f"Approach 4 failed: {str(e)}")
    
    print("\nAll approaches failed to load data")
    return None



# Main exploration
print("Exploring zarr file structure in the dataset")

# List all denoised.zarr files in training data
denoised_zarrs = glob.glob(os.path.join(train_dir, 'static/ExperimentRuns/*/VoxelSpacing10.000/denoised.zarr'))
print(f"Found {len(denoised_zarrs)} denoised.zarr files in training data:")
for i, zarr_path in enumerate(denoised_zarrs):
    parts = zarr_path.split('/')
    experiment = parts[-3]
    print(f"  {i+1}. {experiment}")

# Choose first experiment for exploration
if denoised_zarrs:
    sample_zarr_path = denoised_zarrs[0]
    print(f"\nExploring sample zarr file: {sample_zarr_path}")
    
    # Explore zarr structure
    explore_zarr(sample_zarr_path)
    
    # Try to load and visualize data
    print("\nAttempting to load and visualize data...")
    try_load_visualize(sample_zarr_path)
else:
    print("No denoised.zarr files found in training data")

# Explore a test zarr file as well
test_zarrs = glob.glob(os.path.join(test_dir, 'static/ExperimentRuns/*/VoxelSpacing10.000/denoised.zarr'))
if test_zarrs:
    test_zarr_path = test_zarrs[0]
    print(f"\nExploring test zarr file: {test_zarr_path}")
    
    # Just do a basic exploration
    explore_zarr(test_zarr_path)
else:
    print("\nNo denoised.zarr files found in test data")


import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Set paths
base_dir = '/kaggle/input/czii-cryo-et-object-identification'
train_dir = os.path.join(base_dir, 'train')

# Particle types and their properties
particle_types = {
    'apo-ferritin': {'difficulty': 'easy', 'weight': 1},
    'beta-amylase': {'difficulty': 'impossible', 'weight': 0},
    'beta-galactosidase': {'difficulty': 'hard', 'weight': 2},
    'ribosome': {'difficulty': 'easy', 'weight': 1},
    'thyroglobulin': {'difficulty': 'hard', 'weight': 2},
    'virus-like-particle': {'difficulty': 'easy', 'weight': 1}
}


# Find all JSON files with particle annotations
print("Looking for particle annotation JSON files...")
particle_jsons = glob.glob(os.path.join(train_dir, 'overlay/ExperimentRuns/*/Picks/*.json'))
print(f"Found {len(particle_jsons)} JSON files")


# List the first few files
for i, json_path in enumerate(particle_jsons[:10]):
    # Extract experiment and particle type from path
    parts = json_path.split('/')
    experiment = parts[-3]
    particle_type = os.path.splitext(os.path.basename(json_path))[0]
    print(f"  {i+1}. {experiment} - {particle_type}")


# Function to explore a JSON file's structure
def explore_json(json_path):
    """
    Explore the structure of a JSON file containing particle annotations
    """
    print(f"Exploring JSON file: {json_path}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Print overall structure
        print("\nTop level keys:")
        for key in data.keys():
            if isinstance(data[key], list):
                print(f"  {key}: list with {len(data[key])} items")
            elif isinstance(data[key], dict):
                print(f"  {key}: dict with {len(data[key])} keys")
            else:
                print(f"  {key}: {type(data[key]).__name__}")
        
        # Check for 'picks' which should contain particle coordinates
        if 'picks' in data:
            picks = data['picks']
            print(f"\nFound {len(picks)} picks")
            
            # Look at the first few picks
            if picks:
                print("\nSample pick entries:")
                for i, pick in enumerate(picks[:3]):
                    print(f"  Pick {i+1}:")
                    print(json.dumps(pick, indent=4))
                
                # Extract a sample of the coordinates
                coords = []
                for pick in picks[:100]:  # Limit to first 100 picks
                    if 'position' in pick:
                        pos = pick['position']
                        if all(k in pos for k in ['x', 'y', 'z']):
                            coords.append((pos['x'], pos['y'], pos['z']))
                
                if coords:
                    print(f"\nExtracted {len(coords)} coordinates")
                    
                    # Basic statistics
                    coords_array = np.array(coords)
                    print("\nCoordinate statistics:")
                    print(f"  X: min={coords_array[:, 0].min():.2f}, max={coords_array[:, 0].max():.2f}, mean={coords_array[:, 0].mean():.2f}")
                    print(f"  Y: min={coords_array[:, 1].min():.2f}, max={coords_array[:, 1].max():.2f}, mean={coords_array[:, 1].mean():.2f}")
                    print(f"  Z: min={coords_array[:, 2].min():.2f}, max={coords_array[:, 2].max():.2f}, mean={coords_array[:, 2].mean():.2f}")
                    
                    return True, coords
                else:
                    print("Could not extract valid coordinates from pick entries")
            else:
                print("No pick entries found")
        else:
            print("No 'picks' key found in JSON")
        
    except Exception as e:
        print(f"Error exploring JSON: {str(e)}")
    
    return False, []


# Collect all particle coordinates in a DataFrame for analysis
all_coords = []

# Try to process a few JSON files
for json_path in particle_jsons[:10]:  # Process first 10 files
    # Extract experiment and particle type from path
    parts = json_path.split('/')
    experiment = parts[-3]
    particle_type = os.path.splitext(os.path.basename(json_path))[0]
    
    print(f"\n{'='*80}\nProcessing {experiment} - {particle_type}\n{'='*80}")
    success, coords = explore_json(json_path)
    
    if success and coords:
        # Add experiment and particle type info to each coordinate
        for x, y, z in coords:
            all_coords.append({
                'experiment': experiment,
                'particle_type': particle_type,
                'x': x,
                'y': y,
                'z': z,
                'difficulty': particle_types[particle_type]['difficulty'],
                'weight': particle_types[particle_type]['weight']
            })



# Convert to DataFrame
coords_df = pd.DataFrame(all_coords)

if not coords_df.empty:
    print(f"\nCollected {len(coords_df)} particle coordinates")
    print("\nCoordinates DataFrame sample:")
    print(coords_df.head())
    
    # Count particles by type and experiment
    particle_counts = coords_df.groupby(['experiment', 'particle_type']).size().reset_index(name='count')
    print("\nParticle counts by experiment and type:")
    print(particle_counts)
    
    # Visualize particle counts by type
    plt.figure(figsize=(12, 6))
    type_counts = coords_df['particle_type'].value_counts()
    plt.bar(type_counts.index, type_counts.values)
    plt.title('Number of Particles by Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Plot coordinates in 3D for a single experiment
    experiments = coords_df['experiment'].unique()
    if len(experiments) > 0:
        exp_to_plot = experiments[0]
        exp_coords = coords_df[coords_df['experiment'] == exp_to_plot]
        
        if len(exp_coords) > 0:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot each particle type with a different color
            for p_type, group in exp_coords.groupby('particle_type'):
                ax.scatter(group['x'], group['y'], group['z'], 
                          label=p_type, 
                          alpha=0.7, s=10)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'3D Distribution of Particles in Experiment {exp_to_plot}')
            plt.legend()
            plt.tight_layout()
            plt.show()
else:
    print("\nFailed to collect any valid particle coordinates")
    
    # Try a more direct approach
    print("\nTrying a direct approach to read one JSON file...")
    if particle_jsons:
        first_json = particle_jsons[0]
        print(f"Reading: {first_json}")
        
        try:
            with open(first_json, 'r') as f:
                content = f.read()
            
            print(f"File size: {len(content)} bytes")
            print("First 1000 characters:")
            print(content[:1000])
            
            # Try parsing the JSON
            try:
                data = json.loads(content)
                print("\nSuccessfully parsed JSON")
                print(f"Top level keys: {list(data.keys())}")
                
                # Fully print a small part of the structure
                if 'picks' in data and data['picks']:
                    first_pick = data['picks'][0]
                    print("\nFirst pick entry:")
                    print(json.dumps(first_pick, indent=2))
            except json.JSONDecodeError as e:
                print(f"\nJSON parsing error: {str(e)}")
        except Exception as e:
            print(f"Error reading file: {str(e)}")


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set paths
base_dir = '/kaggle/input/czii-cryo-et-object-identification'

# Load the sample submission file
sample_submission_path = os.path.join(base_dir, 'sample_submission.csv')



if os.path.exists(sample_submission_path):
    print(f"Loading sample submission from: {sample_submission_path}")
    sample_submission = pd.read_csv(sample_submission_path)
    
    # Print basic info
    print(f"\nSample submission shape: {sample_submission.shape}")
    print("\nSample submission columns:")
    for col in sample_submission.columns:
        print(f"  - {col}")
    
    print("\nFirst 10 rows of the sample submission:")
    print(sample_submission.head(10))
    
    # Analyze the unique values in each column
    print("\nUnique values in each column:")
    for col in sample_submission.columns:
        unique_vals = sample_submission[col].nunique()
        print(f"  - {col}: {unique_vals} unique values")
        
        # For columns with few unique values, print them
        if unique_vals < 10 and col != 'id':
            print(f"    Values: {sorted(sample_submission[col].unique())}")
    
    # Check for missing values
    missing = sample_submission.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing values in sample submission:")
        for col, count in missing.items():
            if count > 0:
                print(f"  - {col}: {count} missing values")
    else:
        print("\nNo missing values in the sample submission")
    
    # Analyze coordinates
    print("\nCoordinate statistics:")
    for col in ['x', 'y', 'z']:
        if col in sample_submission.columns:
            print(f"  - {col}:")
            print(f"    Min: {sample_submission[col].min()}")
            print(f"    Max: {sample_submission[col].max()}")
            print(f"    Mean: {sample_submission[col].mean()}")
            print(f"    Std: {sample_submission[col].std()}")
    
    # Count rows per experiment and particle type
    if 'experiment' in sample_submission.columns and 'particle_type' in sample_submission.columns:
        exp_counts = sample_submission.groupby('experiment').size()
        print("\nSubmission entries per experiment:")
        for exp, count in exp_counts.items():
            print(f"  - {exp}: {count} entries")
        
        type_counts = sample_submission.groupby('particle_type').size()
        print("\nSubmission entries per particle type:")
        for p_type, count in type_counts.items():
            print(f"  - {p_type}: {count} entries")
        
        # Cross-tabulation of experiment and particle type
        cross_tab = pd.crosstab(sample_submission['experiment'], sample_submission['particle_type'])
        print("\nCross-tabulation of experiment and particle type:")
        print(cross_tab)
        
        # Visualize particle types in sample submission
        plt.figure(figsize=(10, 6))
        type_counts.plot(kind='bar')
        plt.title('Particle Types in Sample Submission')
        plt.xlabel('Particle Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
else:
    print(f"Sample submission file not found at: {sample_submission_path}")


import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import zarr
import json
from tqdm.notebook import tqdm
import seaborn as sns


# Set paths to the data directories
base_dir = '/kaggle/input/czii-cryo-et-object-identification' 
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')


# Function to explore directory structure
def explore_directory(directory, max_depth=3, current_depth=0):
    if current_depth > max_depth:
        return []
    
    result = []
    try:
        for item in os.listdir(directory):
            path = os.path.join(directory, item)
            if os.path.isdir(path):
                result.append(f"{'  ' * current_depth}ğŸ“� {item}/")
                result.extend(explore_directory(path, max_depth, current_depth + 1))
            else:
                result.append(f"{'  ' * current_depth}ğŸ“„ {item}")
    except Exception as e:
        result.append(f"{'  ' * current_depth}â�Œ Error: {str(e)}")
    
    return result



# Explore train directory structure
print("Train Directory Structure:")
train_structure = explore_directory(train_dir, max_depth=4)
for line in train_structure[:50]:  # Limit output to first 50 lines
    print(line)
print("..." if len(train_structure) > 50 else "")


# Explore test directory structure
print("\nTest Directory Structure:")
test_structure = explore_directory(test_dir, max_depth=4)
for line in test_structure[:50]:  # Limit output to first 50 lines
    print(line)
print("..." if len(test_structure) > 50 else "")


# List all experiments in train and test
train_experiments = [os.path.basename(p) for p in glob.glob(os.path.join(train_dir, 'static/ExperimentRuns/*'))]
test_experiments = [os.path.basename(p) for p in glob.glob(os.path.join(test_dir, 'static/ExperimentRuns/*'))]



print(f"\nNumber of experiments in training data: {len(train_experiments)}")
print(f"Number of experiments in test data: {len(test_experiments)}")

print("\nSample of training experiments:")
for exp in train_experiments[:5]:
    print(f"  - {exp}")

print("\nSample of test experiments:")
for exp in test_experiments[:5]:
    print(f"  - {exp}")


# Check sample submission format
sample_submission_path = os.path.join(base_dir, 'sample_submission.csv')
if os.path.exists(sample_submission_path):
    sample_submission = pd.read_csv(sample_submission_path)
    print("\nSample Submission Format:")
    print(sample_submission.head())
    print(f"Sample submission shape: {sample_submission.shape}")
    print(f"Sample submission columns: {sample_submission.columns.tolist()}")


import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from collections import Counter, defaultdict


# Set paths
base_dir = '/kaggle/input/czii-cryo-et-object-identification'
train_dir = os.path.join(base_dir, 'train')

# Particle types and their properties
particle_types = {
    'apo-ferritin': {'difficulty': 'easy', 'weight': 1, 'color': 'red'},
    'beta-amylase': {'difficulty': 'impossible', 'weight': 0, 'color': 'gray'},
    'beta-galactosidase': {'difficulty': 'hard', 'weight': 2, 'color': 'blue'},
    'ribosome': {'difficulty': 'easy', 'weight': 1, 'color': 'green'},
    'thyroglobulin': {'difficulty': 'hard', 'weight': 2, 'color': 'purple'},
    'virus-like-particle': {'difficulty': 'easy', 'weight': 1, 'color': 'orange'}
}


# Function to load particle coordinates from JSON with the correct structure
def load_particle_coords(json_path):
    """
    Load particle coordinates from JSON file
    Uses 'points' key and 'location' field which is the actual structure in the data
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract coordinates from points
        coords = []
        if 'points' in data:
            for point in data['points']:
                if 'location' in point:
                    loc = point['location']
                    coords.append((loc.get('x', 0), loc.get('y', 0), loc.get('z', 0)))
            
            print(f"Extracted {len(coords)} coordinates from {os.path.basename(json_path)}")
            return coords
        else:
            print(f"No 'points' key found in {json_path}")
            return []
    except Exception as e:
        print(f"Error loading {json_path}: {str(e)}")
        return []



# Collect particle coordinates data
all_particle_data = []
experiment_counts = defaultdict(lambda: defaultdict(int))
particle_counts = Counter()
experiments = []

# Find all JSON files with particle annotations
json_files = glob.glob(os.path.join(train_dir, 'overlay/ExperimentRuns/*/Picks/*.json'))
print(f"Found {len(json_files)} JSON files")



# Process all JSON files
for json_path in json_files:
    # Extract experiment and particle type from path
    parts = json_path.split('/')
    experiment = parts[-3]
    particle_type = os.path.splitext(os.path.basename(json_path))[0]
    
    # Load coordinates
    coords = load_particle_coords(json_path)
    count = len(coords)
    
    # Update counts
    if experiment not in experiments:
        experiments.append(experiment)
    
    experiment_counts[experiment][particle_type] = count
    particle_counts[particle_type] += count
    
    # Add coordinates to dataset
    for x, y, z in coords:
        all_particle_data.append({
            'experiment': experiment,
            'particle_type': particle_type,
            'x': x,
            'y': y,
            'z': z,
            'difficulty': particle_types[particle_type]['difficulty'],
            'weight': particle_types[particle_type]['weight']
        })

# Convert to DataFrame for easier analysis
particle_df = pd.DataFrame(all_particle_data)
experiment_df = pd.DataFrame(experiment_counts).T.fillna(0)
experiment_df = experiment_df.astype(int)


# Print overall statistics
print(f"\nTotal number of experiments: {len(experiments)}")
print(f"Total number of particles found: {len(particle_df)}")
print("\nParticle distribution:")
for particle, count in particle_counts.most_common():
    difficulty = particle_types.get(particle, {}).get('difficulty', 'unknown')
    weight = particle_types.get(particle, {}).get('weight', 'unknown')
    print(f"  - {particle}: {count} particles (Difficulty: {difficulty}, Weight: {weight})")



if not particle_df.empty:
    # Plot particle distribution
    plt.figure(figsize=(12, 6))
    type_counts = particle_df['particle_type'].value_counts()
    bars = plt.bar(type_counts.index, type_counts.values, 
                  color=[particle_types[p]['color'] for p in type_counts.index])
    
    # Add difficulty labels
    for i, particle in enumerate(type_counts.index):
        difficulty = particle_types.get(particle, {}).get('difficulty', 'unknown')
        weight = particle_types.get(particle, {}).get('weight', 'unknown')
        plt.text(i, type_counts[particle] + 5, f"{difficulty}\nweight: {weight}", 
                ha='center', va='bottom', fontweight='bold')
    
    plt.title('Particle Distribution in Training Data')
    plt.ylabel('Number of Particles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Particle counts per experiment
    experiment_df['total'] = experiment_df.sum(axis=1)
    
    print("\nParticle counts per experiment:")
    print(experiment_df)
    
    print("\nParticle count statistics per experiment:")
    print(experiment_df.describe())
    
    # Plot total particles per experiment
    plt.figure(figsize=(10, 6))
    experiment_df['total'].plot(kind='bar', color='teal')
    plt.title('Total Particles per Experiment')
    plt.xlabel('Experiment')
    plt.ylabel('Number of Particles')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot distribution of particles by type and experiment
    plt.figure(figsize=(14, 8))
    experiment_df_subset = experiment_df.drop(columns=['total'])
    experiment_df_subset.plot(kind='bar', 
                             color=[particle_types[col]['color'] for col in experiment_df_subset.columns])
    plt.title('Particle Distribution by Type and Experiment')
    plt.xlabel('Experiment')
    plt.ylabel('Number of Particles')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Particle Type')
    plt.tight_layout()
    plt.show()
    
    # 3D scatter plot of one experiment
    if len(experiments) > 0:
        sample_exp = experiments[0]
        sample_data = particle_df[particle_df['experiment'] == sample_exp]
        
        if len(sample_data) > 0:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            for p_type, group in sample_data.groupby('particle_type'):
                ax.scatter(group['x'], group['y'], group['z'], 
                          label=p_type, 
                          color=particle_types[p_type]['color'],
                          alpha=0.7)
            
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.set_zlabel('Z coordinate')
            ax.set_title(f'3D Distribution of Particles in Experiment {sample_exp}')
            plt.legend()
            plt.tight_layout()
            plt.show()
    
    # Examine the coordinate ranges for each experiment
    coord_ranges = {}
    for exp in experiments:
        exp_data = particle_df[particle_df['experiment'] == exp]
        
        if not exp_data.empty:
            coord_ranges[exp] = {
                'x': {'min': exp_data['x'].min(), 'max': exp_data['x'].max()},
                'y': {'min': exp_data['y'].min(), 'max': exp_data['y'].max()},
                'z': {'min': exp_data['z'].min(), 'max': exp_data['z'].max()}
            }
    
    print("\nCoordinate ranges by experiment:")
    for exp, ranges in coord_ranges.items():
        print(f"  {exp}:")
        print(f"    X: {ranges['x']['min']:.1f} to {ranges['x']['max']:.1f}")
        print(f"    Y: {ranges['y']['min']:.1f} to {ranges['y']['max']:.1f}")
        print(f"    Z: {ranges['z']['min']:.1f} to {ranges['z']['max']:.1f}")
else:
    print("No particle data found after extraction.")


import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm.notebook import tqdm

# Set paths
base_dir = '/kaggle/input/czii-cryo-et-object-identification'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')



# Function to load a tomogram using the correct zarr structure
def load_tomogram(zarr_path, resolution=0):
    """
    Load a tomogram from a zarr file using the multiscale structure
    
    Parameters:
    zarr_path (str): Path to the zarr directory
    resolution (int): Resolution level (0 = highest, 1 = medium, 2 = lowest)
    
    Returns:
    numpy.ndarray: The loaded tomogram data
    """
    try:
        # Open the zarr group
        z = zarr.open(zarr_path, mode='r')
        
        # Access the resolution level directly (based on the structure we observed)
        # Resolution level is a direct key in the group
        if str(resolution) in z:
            tomo_data = z[str(resolution)][:]
            print(f"Loaded tomogram with shape {tomo_data.shape}")
            return tomo_data
        else:
            print(f"Resolution level {resolution} not found in zarr file")
            return None
    except Exception as e:
        print(f"Error loading tomogram: {str(e)}")
        return None



# Function to visualize tomogram slices
def visualize_tomogram(tomo_data, title="Tomogram Slices", n_slices=3, figsize=(15, 5)):
    """
    Visualize slices of a 3D tomogram
    
    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    title (str): Plot title
    n_slices (int): Number of slices to visualize
    figsize (tuple): Figure size
    """
    if tomo_data is None:
        print("No tomogram data to visualize")
        return
    
    # Get tomogram dimensions
    depth, height, width = tomo_data.shape
    print(f"Tomogram dimensions: {depth} x {height} x {width}")
    
    # Choose slice indices at different depths
    slice_indices = np.linspace(depth // 4, 3 * depth // 4, n_slices).astype(int)
    
    # Create figure
    fig, axes = plt.subplots(1, n_slices, figsize=figsize)
    if n_slices == 1:
        axes = [axes]
    
    # Plot each slice
    for i, slice_idx in enumerate(slice_indices):
        im = axes[i].imshow(tomo_data[slice_idx], cmap='gray')
        axes[i].set_title(f'Z-Slice {slice_idx}/{depth}')
        axes[i].axis('off')
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    # Show orthogonal views (XY, XZ, YZ planes)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # XY plane (middle slice in Z)
    z_mid = depth // 2
    axes[0].imshow(tomo_data[z_mid], cmap='gray')
    axes[0].set_title(f'XY Plane (Z={z_mid})')
    axes[0].axis('off')
    
    # XZ plane (middle slice in Y)
    y_mid = height // 2
    axes[1].imshow(tomo_data[:, y_mid, :], cmap='gray')
    axes[1].set_title(f'XZ Plane (Y={y_mid})')
    axes[1].axis('off')
    
    # YZ plane (middle slice in X)
    x_mid = width // 2
    axes[2].imshow(tomo_data[:, :, x_mid], cmap='gray')
    axes[2].set_title(f'YZ Plane (X={x_mid})')
    axes[2].axis('off')
    
    plt.suptitle(f"{title} - Orthogonal Views")
    plt.tight_layout()
    plt.show()



# Function to visualize tomogram with particle positions
def visualize_tomogram_with_particles(tomo_data, json_paths, title="Tomogram with Particles", slice_idx=None):
    """
    Visualize a tomogram slice with particle positions overlaid
    
    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    json_paths (list): List of paths to JSON files with particle coordinates
    title (str): Plot title
    slice_idx (int, optional): Specific slice to visualize. If None, middle slice is used.
    """
    if tomo_data is None:
        print("No tomogram data to visualize")
        return
    
    # Get tomogram dimensions
    depth, height, width = tomo_data.shape
    
    # Choose slice index if not specified
    if slice_idx is None:
        slice_idx = depth // 2
    
    # Load particle coordinates from all JSON files
    particles_by_type = {}
    
    for json_path in json_paths:
        particle_type = os.path.splitext(os.path.basename(json_path))[0]
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            coords = []
            if 'points' in data:
                for point in data['points']:
                    if 'location' in point:
                        loc = point['location']
                        coords.append((loc.get('x', 0), loc.get('y', 0), loc.get('z', 0)))
            
            particles_by_type[particle_type] = coords
        except Exception as e:
            print(f"Error loading {json_path}: {str(e)}")
    
    # Define colors for different particle types
    colors = {
        'apo-ferritin': 'red',
        'beta-amylase': 'gray',
        'beta-galactosidase': 'blue',
        'ribosome': 'green',
        'thyroglobulin': 'purple',
        'virus-like-particle': 'orange'
    }
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Show the tomogram slice
    plt.imshow(tomo_data[slice_idx], cmap='gray')
    
    # Overlay particle positions near the slice
    slice_range = 10  # Consider particles within Â±10 slices
    
    for p_type, coords in particles_by_type.items():
        # Filter coordinates near the current slice
        slice_coords = []
        for x, y, z in coords:
            z_idx = int(z / 10)  # Convert physical z to index (assuming 10 Ã… voxel spacing)
            if abs(z_idx - slice_idx) <= slice_range:
                slice_coords.append((x, y))
        
        if slice_coords:
            x_coords = [x for x, _ in slice_coords]
            y_coords = [y for _, y in slice_coords]
            
            plt.scatter(x_coords, y_coords, 
                      c=colors.get(p_type, 'white'), 
                      label=f'{p_type} ({len(slice_coords)})', 
                      alpha=0.7, s=30, edgecolors='white')
    
    plt.title(f'{title}\nSlice {slice_idx}/{depth}')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# List all experiments in training data
train_experiments = [os.path.basename(p) for p in glob.glob(os.path.join(train_dir, 'static/ExperimentRuns/*'))]
print(f"Found {len(train_experiments)} experiments in training data: {train_experiments}")



# Choose one experiment for visualization
if train_experiments:
    sample_experiment = train_experiments[0]
    print(f"\nVisualizing data for experiment: {sample_experiment}")
    
    # Path to the denoised tomogram
    denoised_path = os.path.join(train_dir, 'static/ExperimentRuns', sample_experiment, 'VoxelSpacing10.000/denoised.zarr')
    
    # Check if the path exists
    if os.path.exists(denoised_path):
        print(f"Loading denoised tomogram from: {denoised_path}")
        
        # Load the tomogram at different resolutions
        for resolution in range(3):  # 0, 1, 2
            tomo_data = load_tomogram(denoised_path, resolution=resolution)
            
            if tomo_data is not None:
                # Visualize the tomogram
                visualize_tomogram(tomo_data, 
                                 title=f"Experiment {sample_experiment} - Denoised (Resolution {resolution})",
                                 n_slices=3)
                
                # Show histogram of voxel values
                plt.figure(figsize=(10, 6))
                plt.hist(tomo_data.flatten(), bins=100, alpha=0.7, color='blue')
                plt.title(f"Voxel Value Distribution - {sample_experiment} (Resolution {resolution})")
                plt.xlabel("Voxel Value")
                plt.ylabel("Frequency")
                plt.grid(True, alpha=0.3)
                plt.show()
                
                # Get particle annotation files for this experiment
                particle_jsons = glob.glob(os.path.join(train_dir, 'overlay/ExperimentRuns', sample_experiment, 'Picks/*.json'))
                
                if particle_jsons:
                    print(f"Found {len(particle_jsons)} particle annotation files")
                    
                    # Visualize tomogram with particles
                    visualize_tomogram_with_particles(tomo_data, particle_jsons, 
                                                   title=f"Experiment {sample_experiment} - With Particles")
                else:
                    print("No particle annotation files found for this experiment")
                
                # Only visualize highest resolution (level 0)
                if resolution == 0:
                    # Also compare with other tomogram types if available
                    tomogram_types = ['ctfdeconvolved.zarr', 'isonetcorrected.zarr', 'wbp.zarr']
                    
                    for tomo_type in tomogram_types:
                        tomo_path = os.path.join(train_dir, 'static/ExperimentRuns', sample_experiment, 
                                              f'VoxelSpacing10.000/{tomo_type}')
                        
                        if os.path.exists(tomo_path):
                            print(f"\nLoading {tomo_type} tomogram...")
                            other_tomo = load_tomogram(tomo_path, resolution=0)
                            
                            if other_tomo is not None:
                                # Visualize middle slice
                                mid_slice = other_tomo.shape[0] // 2
                                plt.figure(figsize=(10, 8))
                                plt.imshow(other_tomo[mid_slice], cmap='gray')
                                plt.title(f"{sample_experiment} - {tomo_type} (Slice {mid_slice})")
                                plt.axis('off')
                                plt.colorbar(fraction=0.046, pad=0.04)
                                plt.tight_layout()
                                plt.show()
                
                # Break after visualizing level 0 (highest resolution)
                if resolution > 0:
                    break
    else:
        print(f"Tomogram path does not exist: {denoised_path}")
else:
    print("No experiment directories found in training data")


# Also check a test experiment
test_experiments = [os.path.basename(p) for p in glob.glob(os.path.join(test_dir, 'static/ExperimentRuns/*'))]
if test_experiments:
    test_experiment = test_experiments[0]
    print(f"\nVisualizing data for test experiment: {test_experiment}")
    
    # Path to the denoised tomogram
    test_denoised_path = os.path.join(test_dir, 'static/ExperimentRuns', test_experiment, 'VoxelSpacing10.000/denoised.zarr')
    
    if os.path.exists(test_denoised_path):
        print(f"Loading test tomogram from: {test_denoised_path}")
        
        # Load and visualize test tomogram
        test_tomo = load_tomogram(test_denoised_path, resolution=0)
        
        if test_tomo is not None:
            visualize_tomogram(test_tomo, 
                              title=f"Test Experiment {test_experiment} - Denoised",
                              n_slices=3)
    else:
        print(f"Test tomogram path does not exist: {test_denoised_path}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

# Particle types and their weights for scoring
particle_types = {
    'apo-ferritin': {'difficulty': 'easy', 'weight': 1},
    'beta-amylase': {'difficulty': 'impossible', 'weight': 0},
    'beta-galactosidase': {'difficulty': 'hard', 'weight': 2},
    'ribosome': {'difficulty': 'easy', 'weight': 1},
    'thyroglobulin': {'difficulty': 'hard', 'weight': 2},
    'virus-like-particle': {'difficulty': 'easy', 'weight': 1}
}



# Approximate particle radii in Angstroms
# Values estimated from the literature
particle_radii = {
    'apo-ferritin': 60,
    'beta-amylase': 45,
    'beta-galactosidase': 80,
    'ribosome': 100,
    'thyroglobulin': 85,
    'virus-like-particle': 120
}


# Define the F-beta metric calculation function
def calculate_fbeta(precision, recall, beta=4):
    """
    Calculate F-beta score from precision and recall values.
    
    Parameters:
    precision (float): Precision value
    recall (float): Recall value
    beta (float): Beta value (defaults to 4 as per competition requirements)
    
    Returns:
    float: F-beta score
    """
    if precision == 0 and recall == 0:
        return 0
    
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)


# Function to determine if a predicted particle is a true positive
def is_true_positive(pred_coords, true_coords, particle_type, radius_factor=0.5):
    """
    Determine if a predicted particle is a true positive.
    
    Parameters:
    pred_coords (tuple): Coordinates of the predicted particle (x, y, z)
    true_coords (list): List of coordinates of true particles
    particle_type (str): Type of particle
    radius_factor (float): Factor of particle radius for matching (0.5 means within half radius)
    
    Returns:
    bool: True if the predicted particle is a true positive, False otherwise
    int: Index of the matched true particle if found, -1 otherwise
    """
    if not true_coords:
        return False, -1
    
    # Get particle radius
    particle_radius = particle_radii.get(particle_type, 60)  # Default to 60 Angstroms if unknown
    
    # Calculate the distance threshold
    threshold = particle_radius * radius_factor
    
    # Convert to numpy arrays for vectorized operations
    pred_coords_array = np.array(pred_coords).reshape(1, 3)
    true_coords_array = np.array(true_coords)
    
    # Calculate distances to all true particles
    distances = cdist(pred_coords_array, true_coords_array)[0]
    
    # Find the minimum distance
    min_dist_idx = np.argmin(distances)
    min_dist = distances[min_dist_idx]
    
    # Check if the minimum distance is below the threshold
    if min_dist <= threshold:
        return True, min_dist_idx
    
    return False, -1



# Function to evaluate predictions against ground truth
def evaluate_predictions(predictions, ground_truth, beta=4):
    """
    Evaluate predictions against ground truth using the F-beta metric.
    
    Parameters:
    predictions (pd.DataFrame): DataFrame with columns 'experiment', 'particle_type', 'x', 'y', 'z'
    ground_truth (pd.DataFrame): DataFrame with columns 'experiment', 'particle_type', 'x', 'y', 'z'
    beta (float): Beta value for F-beta calculation (default: 4)
    
    Returns:
    dict: Dictionary with evaluation results
    """
    results = {'overall': {}, 'by_type': {}, 'by_experiment': {}}
    
    # Initialize counters
    total_tp = 0
    total_fp = 0
    total_fn = 0
    weighted_tp = 0
    weighted_fp = 0
    weighted_fn = 0
    
    # Initialize counters for each particle type
    type_stats = {}
    for p_type in particle_types:
        if particle_types[p_type]['weight'] > 0:  # Only consider scored particles
            type_stats[p_type] = {'tp': 0, 'fp': 0, 'fn': 0, 'weight': particle_types[p_type]['weight']}
    
    # Process each experiment
    experiments = ground_truth['experiment'].unique()
    for experiment in experiments:
        # Get predictions and ground truth for this experiment
        exp_pred = predictions[predictions['experiment'] == experiment]
        exp_true = ground_truth[ground_truth['experiment'] == experiment]
        
        # Process each particle type
        for p_type in particle_types:
            if particle_types[p_type]['weight'] == 0:
                continue  # Skip particles with weight 0 (beta-amylase)
            
            # Get predictions and ground truth for this type
            type_pred = exp_pred[exp_pred['particle_type'] == p_type]
            type_true = exp_true[exp_true['particle_type'] == p_type]
            
            # Extract coordinates
            pred_coords = type_pred[['x', 'y', 'z']].values.tolist()
            true_coords = type_true[['x', 'y', 'z']].values.tolist()
            
            # Count true positives, false positives, and false negatives
            tp = 0
            fp = len(pred_coords)  # Start assuming all predictions are false positives
            fn = len(true_coords)  # Start assuming all true particles are false negatives
            
            # Track which true particles have been matched
            matched_true = [False] * len(true_coords)
            
            # Check each prediction
            for pred_coord in pred_coords:
                is_tp, match_idx = is_true_positive(pred_coord, true_coords, p_type)
                
                if is_tp and not matched_true[match_idx]:
                    tp += 1
                    fp -= 1  # One less false positive
                    fn -= 1  # One less false negative
                    matched_true[match_idx] = True
            
            # Update type statistics
            type_stats[p_type]['tp'] += tp
            type_stats[p_type]['fp'] += fp
            type_stats[p_type]['fn'] += fn
            
            # Update total counters
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # Update weighted counters
            weight = particle_types[p_type]['weight']
            weighted_tp += tp * weight
            weighted_fp += fp * weight
            weighted_fn += fn * weight
    
    # Calculate overall micro-averaged precision, recall, and F-beta
    if weighted_tp + weighted_fp > 0:
        weighted_precision = weighted_tp / (weighted_tp + weighted_fp)
    else:
        weighted_precision = 0
    
    if weighted_tp + weighted_fn > 0:
        weighted_recall = weighted_tp / (weighted_tp + weighted_fn)
    else:
        weighted_recall = 0
    
    weighted_fbeta = calculate_fbeta(weighted_precision, weighted_recall, beta)
    
    # Store overall results
    results['overall'] = {
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
        'weighted_true_positives': weighted_tp,
        'weighted_false_positives': weighted_fp,
        'weighted_false_negatives': weighted_fn,
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f_beta': weighted_fbeta
    }
    
    # Calculate results for each particle type
    for p_type, stats in type_stats.items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
        
        fbeta = calculate_fbeta(precision, recall, beta)
        
        results['by_type'][p_type] = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f_beta': fbeta,
            'weight': stats['weight']
        }
    
    return results


# Function to visualize the F-beta metric
def visualize_fbeta():
    """
    Create visualizations to understand the F-beta metric with beta=4.
    """
    # Create a grid of precision and recall values
    precision_values = np.linspace(0.01, 1.0, 100)
    recall_values = np.linspace(0.01, 1.0, 100)
    P, R = np.meshgrid(precision_values, recall_values)
    
    # Calculate F-beta for each precision-recall pair
    F_beta = np.zeros_like(P)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            F_beta[i, j] = calculate_fbeta(P[i, j], R[i, j], beta=4)
    
    # 3D surface plot of F-beta
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(P, R, F_beta, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_zlabel('F-beta (beta=4)')
    ax.set_title('F-beta Metric (beta=4) for Different Precision and Recall Values')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()
    
    # Contour plot of F-beta
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(P, R, F_beta, 20, cmap='viridis')
    plt.colorbar(contour, label='F-beta (beta=4)')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Contour Plot of F-beta Metric (beta=4)')
    plt.grid(True, alpha=0.3)
    
    # Add some contour lines with labels
    contour_lines = plt.contour(P, R, F_beta, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
                             colors='white', linestyles='dashed')
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
    plt.tight_layout()
    plt.show()
    
    # Plot F-beta for fixed precision or recall values
    plt.figure(figsize=(12, 6))
    
    # For fixed precision values
    plt.subplot(1, 2, 1)
    for precision in [0.2, 0.4, 0.6, 0.8, 1.0]:
        fbeta_values = [calculate_fbeta(precision, r, beta=4) for r in recall_values]
        plt.plot(recall_values, fbeta_values, label=f'Precision = {precision:.1f}')
    
    plt.xlabel('Recall')
    plt.ylabel('F-beta (beta=4)')
    plt.title('F-beta for Fixed Precision Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # For fixed recall values
    plt.subplot(1, 2, 2)
    for recall in [0.2, 0.4, 0.6, 0.8, 1.0]:
        fbeta_values = [calculate_fbeta(p, recall, beta=4) for p in precision_values]
        plt.plot(precision_values, fbeta_values, label=f'Recall = {recall:.1f}')
    
    plt.xlabel('Precision')
    plt.ylabel('F-beta (beta=4)')
    plt.title('F-beta for Fixed Recall Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compare different beta values
    plt.figure(figsize=(10, 6))
    
    # Fixed recall of 0.8
    recall = 0.8
    for beta in [0.5, 1, 2, 4, 8]:
        fbeta_values = [calculate_fbeta(p, recall, beta=beta) for p in precision_values]
        plt.plot(precision_values, fbeta_values, label=f'Beta = {beta}')
    
    plt.xlabel('Precision')
    plt.ylabel('F-beta')
    plt.title(f'F-beta Metrics for Different Beta Values (Recall = {recall})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



# Function to create a sample prediction and ground truth for demonstration
def create_sample_data():
    """
    Create sample prediction and ground truth data for demonstration.
    """
    # Ground truth
    ground_truth_data = []
    
    # Experiment 1: 20 particles of each type (except beta-amylase)
    for _ in range(20):
        for p_type in ['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']:
            x = np.random.uniform(100, 1000)
            y = np.random.uniform(100, 1000)
            z = np.random.uniform(50, 150)
            ground_truth_data.append({'experiment': 'TS_5_4', 'particle_type': p_type, 'x': x, 'y': y, 'z': z})
    
    # Experiment 2: 15 particles of each type
    for _ in range(15):
        for p_type in ['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']:
            x = np.random.uniform(100, 1000)
            y = np.random.uniform(100, 1000)
            z = np.random.uniform(50, 150)
            ground_truth_data.append({'experiment': 'TS_6_4', 'particle_type': p_type, 'x': x, 'y': y, 'z': z})
    
    # Create ground truth DataFrame
    ground_truth_df = pd.DataFrame(ground_truth_data)
    
    # Create predictions with some noise and missing particles
    predictions_data = []
    
    # Copy 80% of ground truth with some noise
    for idx, row in ground_truth_df.iterrows():
        if np.random.random() < 0.8:  # 80% chance of detecting the particle
            noise_x = np.random.normal(0, 10)  # Add some noise
            noise_y = np.random.normal(0, 10)
            noise_z = np.random.normal(0, 5)
            predictions_data.append({
                'experiment': row['experiment'],
                'particle_type': row['particle_type'],
                'x': row['x'] + noise_x,
                'y': row['y'] + noise_y,
                'z': row['z'] + noise_z
            })
    
    # Add some false positives (10% of the total)
    num_false_positives = int(0.1 * len(ground_truth_df))
    for _ in range(num_false_positives):
        experiment = np.random.choice(['TS_5_4', 'TS_6_4'])
        p_type = np.random.choice(['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle'])
        x = np.random.uniform(100, 1000)
        y = np.random.uniform(100, 1000)
        z = np.random.uniform(50, 150)
        predictions_data.append({'experiment': experiment, 'particle_type': p_type, 'x': x, 'y': y, 'z': z})
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame(predictions_data)
    
    return predictions_df, ground_truth_df



# Visualize the F-beta metric properties
print("Visualizing F-beta metric properties...")
visualize_fbeta()

# Demonstrate the evaluation with sample data
print("\nDemonstrating the evaluation metric with sample data...")
predictions, ground_truth = create_sample_data()

print(f"Ground truth shape: {ground_truth.shape}")
print(f"Predictions shape: {predictions.shape}")

# Evaluate the predictions
results = evaluate_predictions(predictions, ground_truth, beta=4)

# Print overall results
print("\nOverall Results:")
print(f"Precision: {results['overall']['precision']:.4f}")
print(f"Recall: {results['overall']['recall']:.4f}")
print(f"F-beta (beta=4): {results['overall']['f_beta']:.4f}")


# Print results by particle type
print("\nResults by Particle Type:")
for p_type, stats in results['by_type'].items():
    print(f"{p_type}:")
    print(f"  Precision: {stats['precision']:.4f}")
    print(f"  Recall: {stats['recall']:.4f}")
    print(f"  F-beta: {stats['f_beta']:.4f}")
    print(f"  TP: {stats['true_positives']}, FP: {stats['false_positives']}, FN: {stats['false_negatives']}")
    print(f"  Weight: {stats['weight']}")



# Visualize results by particle type
plt.figure(figsize=(12, 6))
particle_types_list = list(results['by_type'].keys())
precision_values = [results['by_type'][p]['precision'] for p in particle_types_list]
recall_values = [results['by_type'][p]['recall'] for p in particle_types_list]
fbeta_values = [results['by_type'][p]['f_beta'] for p in particle_types_list]

x = np.arange(len(particle_types_list))
width = 0.25

plt.bar(x - width, precision_values, width, label='Precision')
plt.bar(x, recall_values, width, label='Recall')
plt.bar(x + width, fbeta_values, width, label='F-beta')

plt.xlabel('Particle Type')
plt.ylabel('Score')
plt.title('Precision, Recall, and F-beta by Particle Type')
plt.xticks(x, particle_types_list, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


import os
import glob
import zarr
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import gc
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom, gaussian_filter


# Set paths
base_dir = '/kaggle/input/czii-cryo-et-object-identification'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
output_dir = '/kaggle/working/preprocessed'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


# Particle types and their properties
particle_types = {
    'apo-ferritin': {'difficulty': 'easy', 'weight': 1, 'radius': 60},
    'beta-amylase': {'difficulty': 'impossible', 'weight': 0, 'radius': 45},
    'beta-galactosidase': {'difficulty': 'hard', 'weight': 2, 'radius': 80},
    'ribosome': {'difficulty': 'easy', 'weight': 1, 'radius': 100},
    'thyroglobulin': {'difficulty': 'hard', 'weight': 2, 'radius': 85},
    'virus-like-particle': {'difficulty': 'easy', 'weight': 1, 'radius': 120}
}

# Get scored particle types (weight > 0)
scored_particle_types = [p for p, props in particle_types.items() if props['weight'] > 0]



# Function to load a tomogram from a zarr file
def load_tomogram(zarr_path, resolution=0):
    """
    Load a tomogram from a zarr file using the multiscale structure
    
    Parameters:
    zarr_path (str): Path to the zarr directory
    resolution (int): Resolution level (0 = highest, 1 = medium, 2 = lowest)
    
    Returns:
    numpy.ndarray: The loaded tomogram data
    """
    try:
        # Open the zarr group
        z = zarr.open(zarr_path, mode='r')
        
        # Access the resolution level directly
        if str(resolution) in z:
            tomo_data = z[str(resolution)][:]
            return tomo_data
        else:
            print(f"Resolution level {resolution} not found in zarr file")
            return None
    except Exception as e:
        print(f"Error loading tomogram: {str(e)}")
        return None


# Function to load particle coordinates from JSON
def load_particle_coords(json_path):
    """
    Load particle coordinates from JSON file
    Uses 'points' key and 'location' field
    
    Parameters:
    json_path (str): Path to the JSON file
    
    Returns:
    list: List of (x, y, z) coordinate tuples
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract coordinates from points
        coords = []
        if 'points' in data:
            for point in data['points']:
                if 'location' in point:
                    loc = point['location']
                    coords.append((loc.get('x', 0), loc.get('y', 0), loc.get('z', 0)))
        
        return coords
    except Exception as e:
        print(f"Error loading {json_path}: {str(e)}")
        return []


# Function to preprocess a tomogram
def preprocess_tomogram(tomo_data):
    """
    Preprocess a tomogram
    
    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    
    Returns:
    numpy.ndarray: Preprocessed tomogram data
    """
    # Make a copy to avoid modifying the original
    processed = tomo_data.copy()
    
    # Apply Gaussian filtering to reduce noise
    processed = gaussian_filter(processed, sigma=1.0)
    
    # Normalize to [0, 1] range
    min_val = processed.min()
    max_val = processed.max()
    if max_val > min_val:
        processed = (processed - min_val) / (max_val - min_val)
    
    # Enhance contrast
    processed = np.clip((processed - 0.1) * 1.25, 0, 1)
    
    return processed


# Function to create a density map for particles
def create_density_map(shape, coords, radius=10):
    """
    Create a density map for particle locations
    
    Parameters:
    shape (tuple): Shape of the output density map (depth, height, width)
    coords (list): List of (x, y, z) coordinate tuples
    radius (float): Radius of particles in voxels
    
    Returns:
    numpy.ndarray: Density map with Gaussian-like peaks at particle locations
    """
    # Initialize empty map
    density_map = np.zeros(shape, dtype=np.float32)
    
    # Convert coordinates to indices
    # Assuming 10 Angstrom voxel spacing
    voxel_spacing = 10.0
    
    depth, height, width = shape
    
    # Add Gaussian peaks for each particle
    for x, y, z in coords:
        # Convert physical coordinates to voxel indices
        z_idx, y_idx, x_idx = int(z / voxel_spacing), int(y / voxel_spacing), int(x / voxel_spacing)
        
        # Skip if outside the volume
        if not (0 <= z_idx < depth and 0 <= y_idx < height and 0 <= x_idx < width):
            continue
        
        # Create a spherical mask around the particle
        z_min = max(0, z_idx - radius)
        z_max = min(depth, z_idx + radius + 1)
        y_min = max(0, y_idx - radius)
        y_max = min(height, y_idx + radius + 1)
        x_min = max(0, x_idx - radius)
        x_max = min(width, x_idx + radius + 1)
        
        # Create coordinate grids
        z_grid, y_grid, x_grid = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Calculate distance from center
        dist_from_center = np.sqrt(
            (z_grid - z_idx)**2 + 
            (y_grid - y_idx)**2 + 
            (x_grid - x_idx)**2
        )
        
        # Use a Gaussian-like function to create soft spheres
        mask = np.exp(-(dist_from_center**2) / (2 * (radius/2)**2))
        
        # Add to density map
        density_map[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
            density_map[z_min:z_max, y_min:y_max, x_min:x_max],
            mask
        )
    
    return density_map


# Function to extract subvolumes (patches) from a tomogram
def extract_subvolumes(tomo_data, coords, patch_size=64, target_radius=6, max_patches_per_tomo=1000):
    """
    Extract subvolumes (patches) from a tomogram
    
    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    coords (dict): Dictionary mapping particle types to coordinate lists
    patch_size (int): Size of cubic patches to extract
    target_radius (int): Radius of the target in the output density map (in voxels)
    max_patches_per_tomo (int): Maximum number of patches to extract per tomogram
    
    Returns:
    tuple: (patches, labels) where patches are subvolumes and labels are target density maps
    """
    depth, height, width = tomo_data.shape
    half_size = patch_size // 2
    
    patches = []
    labels = []
    metadata = []
    
    # Combine all coordinates
    all_coords = []
    for p_type, coords_list in coords.items():
        for coord in coords_list:
            all_coords.append((coord[0], coord[1], coord[2], p_type))
    
    # Shuffle to ensure random selection if we hit max_patches_per_tomo
    np.random.shuffle(all_coords)
    
    # Extract patches around particle centers
    patch_count = 0
    for x, y, z, p_type in all_coords:
        if patch_count >= max_patches_per_tomo:
            break
            
        # Convert physical coordinates to voxel indices
        z_idx, y_idx, x_idx = int(z / 10.0), int(y / 10.0), int(x / 10.0)
        
        # Check if the patch is fully within the tomogram
        if (z_idx - half_size < 0 or z_idx + half_size >= depth or
            y_idx - half_size < 0 or y_idx + half_size >= height or
            x_idx - half_size < 0 or x_idx + half_size >= width):
            continue
        
        # Extract the patch
        patch = tomo_data[
            z_idx - half_size:z_idx + half_size,
            y_idx - half_size:y_idx + half_size,
            x_idx - half_size:x_idx + half_size
        ]
        
        # Skip if patch is invalid
        if patch.shape != (patch_size, patch_size, patch_size):
            continue
        
        # Create a label (target density map)
        if p_type in scored_particle_types:
            # Only create a target for scored particle types
            label = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
            
            # Create a spherical mask
            z_grid, y_grid, x_grid = np.ogrid[
                :patch_size, 
                :patch_size, 
                :patch_size
            ]
            center = patch_size // 2
            
            dist_from_center = np.sqrt(
                (z_grid - center)**2 + 
                (y_grid - center)**2 + 
                (x_grid - center)**2
            )
            
            # Create Gaussian-like target
            label = np.exp(-(dist_from_center**2) / (2 * (target_radius/2)**2))
        else:
            # For non-scored particles, use empty labels
            label = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
        
        patches.append(patch)
        labels.append(label)
        metadata.append({
            'particle_type': p_type,
            'x': x,
            'y': y,
            'z': z,
            'weight': particle_types[p_type]['weight']
        })
        
        patch_count += 1
    
    # Also add some random negative patches (background)
    num_negative = min(len(patches) // 4, max_patches_per_tomo - patch_count)
    
    for _ in range(num_negative):
        # Generate random coordinates away from particles
        while True:
            z_idx = np.random.randint(half_size, depth - half_size)
            y_idx = np.random.randint(half_size, height - half_size)
            x_idx = np.random.randint(half_size, width - half_size)
            
            # Check if this point is far from all particles
            physical_x = x_idx * 10.0
            physical_y = y_idx * 10.0
            physical_z = z_idx * 10.0
            
            # Check distance from all particles
            min_dist = float('inf')
            for x, y, z, p_type in all_coords:
                dist = np.sqrt((x - physical_x)**2 + (y - physical_y)**2 + (z - physical_z)**2)
                min_dist = min(min_dist, dist)
            
            # If far enough from particles, use this location
            if min_dist > 100:  # 100 Angstroms away from any particle
                break
        
        # Extract the patch
        patch = tomo_data[
            z_idx - half_size:z_idx + half_size,
            y_idx - half_size:y_idx + half_size,
            x_idx - half_size:x_idx + half_size
        ]
        
        # Skip if patch is invalid
        if patch.shape != (patch_size, patch_size, patch_size):
            continue
        
        # Use empty label for negative patches
        label = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
        
        patches.append(patch)
        labels.append(label)
        metadata.append({
            'particle_type': 'background',
            'x': physical_x,
            'y': physical_y,
            'z': physical_z,
            'weight': 0
        })
    
    return np.array(patches), np.array(labels), metadata


# Process training data
def process_training_data(patch_size=64, split_ratio=0.2):
    """
    Process all training data, extract patches, and save them
    
    Parameters:
    patch_size (int): Size of cubic patches to extract
    split_ratio (float): Ratio for validation split
    
    Returns:
    tuple: (train_patches, train_labels, val_patches, val_labels, metadata)
    """
    all_patches = []
    all_labels = []
    all_metadata = []
    
    # Get list of training experiments
    train_experiments = [os.path.basename(p) for p in glob.glob(os.path.join(train_dir, 'static/ExperimentRuns/*'))]
    print(f"Found {len(train_experiments)} training experiments: {train_experiments}")
    
    # Process each experiment
    for experiment in tqdm(train_experiments, desc="Processing experiments"):
        # Load tomogram
        zarr_path = os.path.join(train_dir, 'static/ExperimentRuns', experiment, 'VoxelSpacing10.000/denoised.zarr')
        
        if not os.path.exists(zarr_path):
            print(f"Tomogram not found: {zarr_path}")
            continue
        
        tomo_data = load_tomogram(zarr_path)
        
        if tomo_data is None:
            print(f"Failed to load tomogram for experiment {experiment}")
            continue
        
        # Preprocess tomogram
        tomo_data = preprocess_tomogram(tomo_data)
        
        # Load particle coordinates for each particle type
        coords = {}
        
        for p_type in particle_types.keys():
            json_path = os.path.join(train_dir, 'overlay/ExperimentRuns', experiment, 'Picks', f"{p_type}.json")
            
            if os.path.exists(json_path):
                coords_list = load_particle_coords(json_path)
                if coords_list:
                    coords[p_type] = coords_list
        
        # Extract patches
        patches, labels, metadata = extract_subvolumes(tomo_data, coords, patch_size)
        
        # Add experiment to metadata
        for m in metadata:
            m['experiment'] = experiment
        
        # Collect patches, labels, and metadata
        all_patches.append(patches)
        all_labels.append(labels)
        all_metadata.extend(metadata)
        
        # Free memory
        del tomo_data, patches, labels
        gc.collect()
    
    # Combine all patches and labels
    all_patches = np.concatenate(all_patches, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(all_metadata)
    
    # Save metadata
    os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
    metadata_df.to_csv(os.path.join(output_dir, 'metadata', 'training_metadata.csv'), index=False)
    
    # Split into training and validation sets
    train_patches, val_patches, train_labels, val_labels = train_test_split(
        all_patches, all_labels, test_size=split_ratio, random_state=42
    )
    
    # Save patches and labels
    os.makedirs(os.path.join(output_dir, 'training'), exist_ok=True)
    np.save(os.path.join(output_dir, 'training', 'train_patches.npy'), train_patches)
    np.save(os.path.join(output_dir, 'training', 'train_labels.npy'), train_labels)
    np.save(os.path.join(output_dir, 'training', 'val_patches.npy'), val_patches)
    np.save(os.path.join(output_dir, 'training', 'val_labels.npy'), val_labels)
    
    # Print statistics
    print(f"Processed {len(all_patches)} patches")
    print(f"Training set: {len(train_patches)} patches")
    print(f"Validation set: {len(val_patches)} patches")
    
    # Count number of patches per particle type
    particle_counts = metadata_df['particle_type'].value_counts()
    print("\nPatch distribution by particle type:")
    for p_type, count in particle_counts.items():
        print(f"  - {p_type}: {count} patches")
    
    # Plot a few random patches
    plot_random_patches(train_patches, train_labels, metadata_df, n_samples=5)
    
    return train_patches, train_labels, val_patches, val_labels, metadata_df



# Function to process test data
def process_test_data(patch_size=64, stride=32):
    """
    Process test data by extracting overlapping patches
    
    Parameters:
    patch_size (int): Size of cubic patches to extract
    stride (int): Stride between patches
    
    Returns:
    dict: Dictionary of test data by experiment
    """
    test_data = {}
    
    # Get list of test experiments
    test_experiments = [os.path.basename(p) for p in glob.glob(os.path.join(test_dir, 'static/ExperimentRuns/*'))]
    print(f"Found {len(test_experiments)} test experiments: {test_experiments}")
    
    # Process each experiment
    for experiment in tqdm(test_experiments, desc="Processing test experiments"):
        # Load tomogram
        zarr_path = os.path.join(test_dir, 'static/ExperimentRuns', experiment, 'VoxelSpacing10.000/denoised.zarr')
        
        if not os.path.exists(zarr_path):
            print(f"Test tomogram not found: {zarr_path}")
            continue
        
        tomo_data = load_tomogram(zarr_path)
        
        if tomo_data is None:
            print(f"Failed to load test tomogram for experiment {experiment}")
            continue
        
        # Preprocess tomogram
        tomo_data = preprocess_tomogram(tomo_data)
        
        # Extract overlapping patches
        depth, height, width = tomo_data.shape
        half_size = patch_size // 2
        
        # Initialize data structures
        patches = []
        coordinates = []
        
        # Extract patches with overlap (stride)
        for z in range(half_size, depth - half_size, stride):
            for y in range(half_size, height - half_size, stride):
                for x in range(half_size, width - half_size, stride):
                    # Extract the patch
                    patch = tomo_data[
                        z - half_size:z + half_size,
                        y - half_size:y + half_size,
                        x - half_size:x + half_size
                    ]
                    
                    # Skip if patch is invalid
                    if patch.shape != (patch_size, patch_size, patch_size):
                        continue
                    
                    patches.append(patch)
                    coordinates.append((x, y, z))
        
        # Convert to numpy arrays
        patches = np.array(patches)
        
        # Store test data
        test_data[experiment] = {
            'patches': patches,
            'coordinates': coordinates,
            'shape': tomo_data.shape
        }
        
        # Save the test data
        os.makedirs(os.path.join(output_dir, 'test', experiment), exist_ok=True)
        np.save(os.path.join(output_dir, 'test', experiment, 'patches.npy'), patches)
        np.save(os.path.join(output_dir, 'test', experiment, 'coordinates.npy'), coordinates)
        np.save(os.path.join(output_dir, 'test', experiment, 'shape.npy'), tomo_data.shape)
        
        # Print statistics
        print(f"Processed {len(patches)} patches for experiment {experiment}")
        
        # Free memory
        del tomo_data, patches
        gc.collect()
    
    return test_data



# Function to plot random patches
def plot_random_patches(patches, labels, metadata_df, n_samples=5):
    """
    Plot random patches and their labels
    
    Parameters:
    patches (numpy.ndarray): Array of patches
    labels (numpy.ndarray): Array of labels
    metadata_df (pandas.DataFrame): DataFrame with metadata
    n_samples (int): Number of samples to plot
    """
    # Get random indices
    indices = np.random.choice(len(patches), size=n_samples, replace=False)
    
    # Plot each sample
    for i, idx in enumerate(indices):
        patch = patches[idx]
        label = labels[idx]
        
        # Get metadata for this patch
        if i < len(metadata_df):
            p_type = metadata_df.iloc[idx]['particle_type']
            p_weight = metadata_df.iloc[idx]['weight']
        else:
            p_type = "Unknown"
            p_weight = "Unknown"
        
        # Plot the middle slice of the patch and label
        middle_slice = patch.shape[0] // 2
        
        plt.figure(figsize=(12, 6))
        
        # Plot patch
        plt.subplot(1, 2, 1)
        plt.imshow(patch[middle_slice], cmap='gray')
        plt.title(f"Patch {idx}: {p_type} (Weight: {p_weight})")
        plt.axis('off')
        
        # Plot label
        plt.subplot(1, 2, 2)
        plt.imshow(label[middle_slice], cmap='hot')
        plt.title(f"Label {idx}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()



# Main execution
print("Starting data preprocessing...")

# Set parameters
patch_size = 64  # Size of cubic patches
split_ratio = 0.2  # Ratio for validation split

# Process training data
train_patches, train_labels, val_patches, val_labels, metadata_df = process_training_data(patch_size, split_ratio)

# Process test data
test_data = process_test_data(patch_size, stride=32)

print("Data preprocessing complete.")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Set paths
output_dir = '/kaggle/working/preprocessed'
model_dir = '/kaggle/working/models'

# Create models directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# Particle types and their properties
particle_types = {
    'apo-ferritin': {'difficulty': 'easy', 'weight': 1, 'radius': 60},
    'beta-amylase': {'difficulty': 'impossible', 'weight': 0, 'radius': 45},
    'beta-galactosidase': {'difficulty': 'hard', 'weight': 2, 'radius': 80},
    'ribosome': {'difficulty': 'easy', 'weight': 1, 'radius': 100},
    'thyroglobulin': {'difficulty': 'hard', 'weight': 2, 'radius': 85},
    'virus-like-particle': {'difficulty': 'easy', 'weight': 1, 'radius': 120}
}

# Get scored particle types (weight > 0)
scored_particle_types = [p for p, props in particle_types.items() if props['weight'] > 0]



# 3D U-Net model
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super(UNet3D, self).__init__()
        
        features = init_features
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = self._block(features * 4, features * 8, name="bottleneck")
        
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        
        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.conv(dec1))
    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )



# Dataset class for 3D patches
class PatchDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels = labels
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        # Add channel dimension and convert to torch tensors
        patch = torch.FloatTensor(self.patches[idx]).unsqueeze(0)
        label = torch.FloatTensor(self.labels[idx]).unsqueeze(0)
        
        return patch, label


# Dice loss for 3D segmentation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Flatten the predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


# Focal loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# Combined loss function
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
    
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)
        
        return self.dice_weight * dice + self.focal_weight * focal



# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, patience=10):
    """
    Train the 3D detection model
    
    Parameters:
    model (nn.Module): Model to train
    train_loader (DataLoader): Training data loader
    val_loader (DataLoader): Validation data loader
    criterion (nn.Module): Loss function
    optimizer (optim.Optimizer): Optimizer
    scheduler (optim.lr_scheduler): Learning rate scheduler
    num_epochs (int): Maximum number of epochs
    patience (int): Early stopping patience
    
    Returns:
    model: Trained model
    history: Training history
    """
    # Initialize variables
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    # Move model to device
    model = model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Training step
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for patches, labels in progress_bar:
            # Move data to device
            patches = patches.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(patches)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update training loss
            train_loss += loss.item() * patches.size(0)
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
        
        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        
        # Validation step
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for patches, labels in val_loader:
                # Move data to device
                patches = patches.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(patches)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Update validation loss
                val_loss += loss.item() * patches.size(0)
        
        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Free up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history



# Function to visualize training history
def plot_training_history(history):
    """
    Plot training history
    
    Parameters:
    history (dict): Training history
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    plt.show()


# Function to visualize model predictions
def visualize_predictions(model, val_loader, num_samples=5):
    """
    Visualize model predictions
    
    Parameters:
    model (nn.Module): Trained model
    val_loader (DataLoader): Validation data loader
    num_samples (int): Number of samples to visualize
    """
    model.eval()
    
    # Get random samples
    samples = []
    with torch.no_grad():  # This ensures no gradients are tracked
        for patches, labels in val_loader:
            samples.append((patches, labels))
            if len(samples) >= num_samples:
                break
    
    # Visualize each sample
    for i, (patches, labels) in enumerate(samples):
        # Move data to device
        patches = patches.to(device)
        
        # Forward pass
        with torch.no_grad():  # Add this to be extra safe
            outputs = model(patches)
        
        # Move back to CPU and convert to numpy
        patches = patches.cpu().numpy()
        labels = labels.cpu().numpy()
        outputs = outputs.detach().cpu().numpy()  # Use detach() before converting to numpy
        
        # Plot middle slice of first batch item
        middle_slice = patches.shape[2] // 2
        
        plt.figure(figsize=(15, 5))
        
        # Plot patch
        plt.subplot(1, 3, 1)
        plt.imshow(patches[0, 0, middle_slice], cmap='gray')
        plt.title(f"Input Patch {i+1}")
        plt.axis('off')
        
        # Plot ground truth
        plt.subplot(1, 3, 2)
        plt.imshow(labels[0, 0, middle_slice], cmap='hot')
        plt.title("Ground Truth")
        plt.axis('off')
        
        # Plot prediction
        plt.subplot(1, 3, 3)
        plt.imshow(outputs[0, 0, middle_slice], cmap='hot')
        plt.title("Prediction")
        plt.axis('off')
        
        plt.savefig(os.path.join(model_dir, f'prediction_{i+1}.png'))
        plt.show()


# Main execution
print("Starting model training...")

# Load training and validation data
train_patches = np.load(os.path.join(output_dir, 'training', 'train_patches.npy'))
train_labels = np.load(os.path.join(output_dir, 'training', 'train_labels.npy'))
val_patches = np.load(os.path.join(output_dir, 'training', 'val_patches.npy'))
val_labels = np.load(os.path.join(output_dir, 'training', 'val_labels.npy'))

print(f"Training data shape: {train_patches.shape}")
print(f"Validation data shape: {val_patches.shape}")


# Create datasets and dataloaders
train_dataset = PatchDataset(train_patches, train_labels)
val_dataset = PatchDataset(val_patches, val_labels)

batch_size = 16  # Adjust based on available memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)



# Initialize model
model = UNet3D(in_channels=1, out_channels=1, init_features=16)

# Initialize loss function and optimizer
criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Train model
model, history = train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    scheduler,
    num_epochs=50,
    patience=10
)

# Save model
torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

# Visualize training history
plot_training_history(history)

# Visualize predictions
#visualize_predictions(model, val_loader, num_samples=5)

print("Model training complete. Saved model to", os.path.join(model_dir, 'model.pth'))

# Free up memory
#del train_patches, train_labels, val_patches, val_labels
#del train_dataset, val_dataset, train_loader, val_loader
#gc.collect()


visualize_predictions(model, val_loader, num_samples=5)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import zarr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gc
from tqdm.notebook import tqdm


# Set paths
base_dir = '/kaggle/input/czii-cryo-et-object-identification'
train_dir = os.path.join(base_dir, 'train')
output_dir = '/kaggle/working/preprocessed'
model_dir = '/kaggle/working/models'
visualization_dir = '/kaggle/working/val_visualizations'

# Create visualization directory if it doesn't exist
os.makedirs(visualization_dir, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Particle types and their properties
particle_types = {
    'apo-ferritin': {'difficulty': 'easy', 'weight': 1, 'radius': 60, 'color': 'red'},
    'beta-amylase': {'difficulty': 'impossible', 'weight': 0, 'radius': 45, 'color': 'yellow'},
    'beta-galactosidase': {'difficulty': 'hard', 'weight': 2, 'radius': 80, 'color': 'green'},
    'ribosome': {'difficulty': 'easy', 'weight': 1, 'radius': 100, 'color': 'blue'},
    'thyroglobulin': {'difficulty': 'hard', 'weight': 2, 'radius': 85, 'color': 'purple'},
    'virus-like-particle': {'difficulty': 'easy', 'weight': 1, 'radius': 120, 'color': 'orange'}
}



class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super(UNet3D, self).__init__()
        
        features = init_features
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = self._block(features * 4, features * 8, name="bottleneck")
        
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        
        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.conv(dec1))
    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )



# Dataset class for validation data
class ValDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels = labels
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        # Add channel dimension and convert to torch tensors
        patch = torch.FloatTensor(self.patches[idx]).unsqueeze(0)
        label = torch.FloatTensor(self.labels[idx]).unsqueeze(0)
        
        return patch, label


# Function to load a tomogram
def load_tomogram(zarr_path, resolution=0):
    """
    Load a tomogram from a zarr file
    
    Parameters:
    zarr_path (str): Path to the zarr directory
    resolution (int): Resolution level (0 = highest, 1 = medium, 2 = lowest)
    
    Returns:
    numpy.ndarray: The loaded tomogram data
    """
    try:
        # Open the zarr group
        z = zarr.open(zarr_path, mode='r')
        
        # Access the resolution level directly (based on the structure we observed)
        if str(resolution) in z:
            tomo_data = z[str(resolution)][:]
            print(f"Loaded tomogram with shape {tomo_data.shape}")
            return tomo_data
        else:
            print(f"Resolution level {resolution} not found in zarr file")
            return None
    except Exception as e:
        print(f"Error loading tomogram: {str(e)}")
        return None


# Function to preprocess a tomogram
def preprocess_tomogram(tomo_data):
    """
    Preprocess a tomogram for better visualization
    
    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    
    Returns:
    numpy.ndarray: Preprocessed tomogram data
    """
    # Make a copy to avoid modifying the original
    processed = tomo_data.copy()
    
    # Normalize to [0, 1] range
    min_val = processed.min()
    max_val = processed.max()
    if max_val > min_val:
        processed = (processed - min_val) / (max_val - min_val)
    
    # Enhance contrast
    processed = np.clip((processed - 0.1) * 1.25, 0, 1)
    
    return processed



# Function to find local maxima in the density map
def find_local_maxima(density_map, min_distance=10, threshold_abs=0.3, threshold_rel=0.2):
    """
    Find local maxima in the density map
    
    Parameters:
    density_map (numpy.ndarray): Density map
    min_distance (int): Minimum distance between peaks
    threshold_abs (float): Minimum absolute threshold for peak
    threshold_rel (float): Minimum relative threshold for peak
    
    Returns:
    numpy.ndarray: Array of peak coordinates [z, y, x]
    """
    # Import here to keep dependencies clean
    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max
    
    # Apply Gaussian smoothing to reduce noise
    smoothed_map = gaussian_filter(density_map, sigma=1.0)
    
    # Find local maxima
    coordinates = peak_local_max(
        smoothed_map,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=False
    )
    
    return coordinates


# Function to prepare a sample of validation tomograms for visualization
def prepare_validation_tomograms():
    """
    Prepare a few validation tomograms and their particle annotations for visualization
    
    This is a simplified approach that visualizes a few tomograms with their ground truth annotations
    """
    # Get train experiments
    train_experiments = [os.path.basename(p) for p in glob.glob(os.path.join(train_dir, 'static/ExperimentRuns/*'))]
    
    if not train_experiments:
        print("No training experiments found.")
        return
    
    # Use the first 3 experiments for visualization (or fewer if less are available)
    viz_experiments = train_experiments[:min(3, len(train_experiments))]
    print(f"Using {len(viz_experiments)} experiments for visualization: {viz_experiments}")
    
    # For each experiment, load the tomogram and particle annotations
    for experiment in viz_experiments:
        print(f"\nProcessing experiment: {experiment}")
        
        # Load tomogram
        zarr_path = os.path.join(train_dir, 'static/ExperimentRuns', experiment, 'VoxelSpacing10.000/denoised.zarr')
        
        if not os.path.exists(zarr_path):
            print(f"Tomogram not found for experiment {experiment}")
            continue
        
        tomo_data = load_tomogram(zarr_path)
        if tomo_data is None:
            print(f"Failed to load tomogram for experiment {experiment}")
            continue
        
        tomo_data = preprocess_tomogram(tomo_data)
        
        # Load particle annotations
        ground_truth_coords = {}
        
        for p_type in particle_types.keys():
            json_path = os.path.join(train_dir, 'overlay/ExperimentRuns', experiment, 'Picks', f"{p_type}.json")
            
            if not os.path.exists(json_path):
                print(f"No annotations found for {p_type} in experiment {experiment}")
                continue
            
            # Load coordinates from JSON
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Extract coordinates from points
                coords = []
                if 'points' in data:
                    for point in data['points']:
                        if 'location' in point:
                            loc = point['location']
                            coords.append((loc.get('x', 0), loc.get('y', 0), loc.get('z', 0)))
                
                if coords:
                    ground_truth_coords[p_type] = coords
                    print(f"Loaded {len(coords)} {p_type} coordinates")
            except Exception as e:
                print(f"Error loading {json_path}: {str(e)}")
        
        # Visualize ground truth
        visualize_ground_truth(tomo_data, ground_truth_coords, experiment)
        
        # Now visualize model predictions on the same tomogram
        visualize_model_predictions(tomo_data, experiment, ground_truth_coords)
        
        # Free memory
        #del tomo_data
        #gc.collect()



# Function to visualize ground truth
def visualize_ground_truth(tomo_data, ground_truth_coords, experiment):
    """
    Visualize ground truth particles on the tomogram
    
    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    ground_truth_coords (dict): Dictionary mapping particle types to coordinates
    experiment (str): Experiment name
    """
    # Get tomogram dimensions
    depth, height, width = tomo_data.shape
    
    # Choose slices for visualization
    slices = [depth // 4, depth // 2, 3 * depth // 4]
    
    # Create figure with subplots for different slices
    fig, axes = plt.subplots(1, len(slices), figsize=(6 * len(slices), 6))
    if len(slices) == 1:
        axes = [axes]
    
    # For each slice
    for i, slice_idx in enumerate(slices):
        # Show the tomogram slice
        axes[i].imshow(tomo_data[slice_idx], cmap='gray')
        axes[i].set_title(f'Ground Truth - Z-Slice {slice_idx}/{depth}')
        
        # Get slice range (particles near this slice)
        slice_range = 10  # Consider particles within Â±10 slices
        z_min = (slice_idx - slice_range) * 10.0  # Convert to physical coordinates
        z_max = (slice_idx + slice_range) * 10.0
        
        # Add circles for each particle type
        for p_type, coords in ground_truth_coords.items():
            # Skip if particle type not in our dictionary
            if p_type not in particle_types:
                continue
                
            # Get color and radius for this particle type
            color = particle_types[p_type]['color']
            radius = particle_types[p_type]['radius'] * 0.1  # Scale down for visualization
            
            # Count particles in this slice
            slice_particles = [(x, y, z) for x, y, z in coords if z_min <= z <= z_max]
            n_particles = len(slice_particles)
            
            # Skip if no particles of this type in this slice
            if n_particles == 0:
                continue
            
            # Add to legend
            axes[i].plot([], [], 'o', color=color, label=f'{p_type} ({n_particles})')
            
            # Add circle for each particle
            for x, y, z in slice_particles:
                # Convert physical coordinates to pixel coordinates
                y_px = y / 10.0
                x_px = x / 10.0
                
                # Add circle
                circle = patches.Circle((x_px, y_px), radius, color=color, fill=False, linewidth=1.5, alpha=0.7)
                axes[i].add_patch(circle)
        
        # Add legend
        axes[i].legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, f'{experiment}_ground_truth.png'), dpi=200, bbox_inches='tight')
    plt.show()


# Function to visualize model predictions
def visualize_model_predictions(tomo_data, experiment, ground_truth_coords):
    """
    Visualize model predictions on a tomogram
    
    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    experiment (str): Experiment name
    ground_truth_coords (dict): Dictionary mapping particle types to ground truth coordinates (for comparison)
    """
    # Load the trained model
    model_path = os.path.join(model_dir, 'model.pth')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    # Initialize model
    model = UNet3D(in_channels=1, out_channels=1, init_features=16)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Get tomogram dimensions
    depth, height, width = tomo_data.shape
    
    # Create a simplified 3D density map
    print("Generating simplified density map for visualization...")
    density_map = np.zeros_like(tomo_data)
    patch_size = 64
    half_size = patch_size // 2
    stride = 32  # Use a stride to reduce computation
    
    # Use a sliding window approach to generate density map
    with torch.no_grad():
        for z in tqdm(range(half_size, depth - half_size, stride)):
            for y in range(half_size, height - half_size, stride):
                for x in range(half_size, width - half_size, stride):
                    # Extract patch
                    patch = tomo_data[
                        z - half_size:z + half_size,
                        y - half_size:y + half_size,
                        x - half_size:x + half_size
                    ]
                    
                    # Skip if patch is invalid
                    if patch.shape != (patch_size, patch_size, patch_size):
                        continue
                    
                    # Convert to tensor
                    patch_tensor = torch.FloatTensor(patch).unsqueeze(0).unsqueeze(0).to(device)
                    
                    # Forward pass
                    output = model(patch_tensor)
                    
                    # Convert to numpy
                    output = output.detach().cpu().numpy()[0, 0]
                    
                    # Add to density map
                    density_map[
                        z - half_size:z + half_size,
                        y - half_size:y + half_size,
                        x - half_size:x + half_size
                    ] = np.maximum(
                        density_map[
                            z - half_size:z + half_size,
                            y - half_size:y + half_size,
                            x - half_size:x + half_size
                        ],
                        output
                    )
    
    # Create a combined visualization with side-by-side ground truth and predictions
    visualize_comparison(tomo_data, density_map, ground_truth_coords, experiment)


# Function to visualize comparison between ground truth and predictions
def visualize_comparison(tomo_data, density_map, ground_truth_coords, experiment):
    """
    Visualize comparison between ground truth and model predictions
    
    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    density_map (numpy.ndarray): 3D density map from model predictions
    ground_truth_coords (dict): Dictionary mapping particle types to ground truth coordinates
    experiment (str): Experiment name
    """
    # Get tomogram dimensions
    depth, height, width = tomo_data.shape
    
    # Choose slices for visualization
    slices = [depth // 4, depth // 2, 3 * depth // 4]
    
    # Find local maxima in density map
    print("Finding local maxima in density map...")
    coords = find_local_maxima(density_map, min_distance=10, threshold_abs=0.3, threshold_rel=0.2)
    print(f"Found {len(coords)} local maxima")
    
    # Simplified particle type assignment for visualization
    # This doesn't replicate the full clustering logic but is sufficient for visualization
    predicted_coords = {}
    
    # Basic heuristic: stronger peaks are more likely to be larger/easier particles
    peak_values = np.array([density_map[z, y, x] for z, y, x in coords])
    peak_order = np.argsort(peak_values)[::-1]  # Sort in descending order
    
    # Assign top 20% to easier particles, next 30% to harder particles
    n_peaks = len(coords)
    easy_threshold = int(n_peaks * 0.2)
    hard_threshold = int(n_peaks * 0.5)
    
    easy_particles = ['apo-ferritin', 'ribosome', 'virus-like-particle']
    hard_particles = ['beta-galactosidase', 'thyroglobulin']
    
    # Randomly assign among the categories
    np.random.seed(42)  # For reproducibility
    
    for i, idx in enumerate(peak_order):
        z, y, x = coords[idx]
        
        # Convert voxel coordinates to physical coordinates
        physical_x = x * 10.0
        physical_y = y * 10.0
        physical_z = z * 10.0
        
        if i < easy_threshold:
            # Assign to an easy particle type
            p_type = np.random.choice(easy_particles)
        elif i < hard_threshold:
            # Assign to a hard particle type
            p_type = np.random.choice(hard_particles)
        else:
            # Skip the rest
            continue
        
        if p_type not in predicted_coords:
            predicted_coords[p_type] = []
        
        predicted_coords[p_type].append((physical_x, physical_y, physical_z))
    
    # For each slice, create a side-by-side comparison
    for slice_idx in slices:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Ground truth visualization
        ax1.imshow(tomo_data[slice_idx], cmap='gray')
        ax1.set_title(f'Ground Truth - Z-Slice {slice_idx}/{depth}')
        
        # Add a semi-transparent overlay of the density map
        ax2.imshow(tomo_data[slice_idx], cmap='gray')
        density_overlay = ax2.imshow(density_map[slice_idx], cmap='hot', alpha=0.5)
        fig.colorbar(density_overlay, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title(f'Model Predictions - Z-Slice {slice_idx}/{depth}')
        
        # Get slice range (particles near this slice)
        slice_range = 10  # Consider particles within Â±10 slices
        z_min = (slice_idx - slice_range) * 10.0  # Convert to physical coordinates
        z_max = (slice_idx + slice_range) * 10.0
        
        # Add ground truth particles
        for p_type, coords in ground_truth_coords.items():
            # Skip if particle type not in our dictionary
            if p_type not in particle_types:
                continue
                
            # Get color and radius for this particle type
            color = particle_types[p_type]['color']
            radius = particle_types[p_type]['radius'] * 0.1  # Scale down for visualization
            
            # Count particles in this slice
            slice_particles = [(x, y, z) for x, y, z in coords if z_min <= z <= z_max]
            n_particles = len(slice_particles)
            
            # Skip if no particles of this type in this slice
            if n_particles == 0:
                continue
            
            # Add to legend
            ax1.plot([], [], 'o', color=color, label=f'{p_type} ({n_particles})')
            
            # Add circle for each particle
            for x, y, z in slice_particles:
                # Convert physical coordinates to pixel coordinates
                y_px = y / 10.0
                x_px = x / 10.0
                
                # Add circle
                circle = patches.Circle((x_px, y_px), radius, color=color, fill=False, linewidth=1.5, alpha=0.7)
                ax1.add_patch(circle)
        
        # Add predicted particles
        for p_type, coords in predicted_coords.items():
            # Skip if particle type not in our dictionary
            if p_type not in particle_types:
                continue
                
            # Get color and radius for this particle type
            color = particle_types[p_type]['color']
            radius = particle_types[p_type]['radius'] * 0.1  # Scale down for visualization
            
            # Count particles in this slice
            slice_particles = [(x, y, z) for x, y, z in coords if z_min <= z <= z_max]
            n_particles = len(slice_particles)
            
            # Skip if no particles of this type in this slice
            if n_particles == 0:
                continue
            
            # Add to legend
            ax2.plot([], [], 'o', color=color, label=f'{p_type} ({n_particles})')
            
            # Add circle for each particle
            for x, y, z in slice_particles:
                # Convert physical coordinates to pixel coordinates
                y_px = y / 10.0
                x_px = x / 10.0
                
                # Add circle
                circle = patches.Circle((x_px, y_px), radius, color=color, fill=False, linewidth=1.5, alpha=0.7)
                ax2.add_patch(circle)
        
        # Add legends
        ax1.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        
        # Remove axes
        ax1.axis('off')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_dir, f'{experiment}_slice_{slice_idx}_comparison.png'), dpi=200, bbox_inches='tight')
        plt.show()


# Main execution
print("Starting validation tomogram visualization...")
prepare_validation_tomograms()
print("Validation tomogram visualization complete.")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import zarr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gc
import glob
from tqdm.notebook import tqdm
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
import json


# Set paths
base_dir = '/kaggle/input/czii-cryo-et-object-identification'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
output_dir = '/kaggle/working/preprocessed'
model_dir = '/kaggle/working/models'
submission_dir = '/kaggle/working/final_submission'
visualization_dir = '/kaggle/working/final_visualizations'

# Create directories if they don't exist
os.makedirs(submission_dir, exist_ok=True)
os.makedirs(visualization_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)



# Particle types and their properties
particle_types = {
    'apo-ferritin': {'difficulty': 'easy', 'weight': 1, 'radius': 60, 'color': 'red'},
    'beta-amylase': {'difficulty': 'impossible', 'weight': 0, 'radius': 45, 'color': 'yellow'},
    'beta-galactosidase': {'difficulty': 'hard', 'weight': 2, 'radius': 80, 'color': 'green'},
    'ribosome': {'difficulty': 'easy', 'weight': 1, 'radius': 100, 'color': 'blue'},
    'thyroglobulin': {'difficulty': 'hard', 'weight': 2, 'radius': 85, 'color': 'purple'},
    'virus-like-particle': {'difficulty': 'easy', 'weight': 1, 'radius': 120, 'color': 'orange'}
}

# Get scored particle types (weight > 0)
scored_particle_types = [p for p, props in particle_types.items() if props['weight'] > 0]



# 3D U-Net model (same as in training)
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super(UNet3D, self).__init__()
        
        features = init_features
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = self._block(features * 4, features * 8, name="bottleneck")
        
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        
        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.conv(dec1))
    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )



# Function to load a tomogram
def load_tomogram(zarr_path, resolution=0):
    """
    Load a tomogram from a zarr file
    
    Parameters:
    zarr_path (str): Path to the zarr directory
    resolution (int): Resolution level (0 = highest, 1 = medium, 2 = lowest)
    
    Returns:
    numpy.ndarray: The loaded tomogram data
    """
    try:
        # Open the zarr group
        z = zarr.open(zarr_path, mode='r')
        
        # Access the resolution level directly
        if str(resolution) in z:
            tomo_data = z[str(resolution)][:]
            print(f"Loaded tomogram with shape {tomo_data.shape}")
            return tomo_data
        else:
            print(f"Resolution level {resolution} not found in zarr file")
            return None
    except Exception as e:
        print(f"Error loading tomogram: {str(e)}")
        return None



# Function to preprocess a tomogram
def preprocess_tomogram(tomo_data):
    """
    Preprocess a tomogram for better prediction
    
    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    
    Returns:
    numpy.ndarray: Preprocessed tomogram data
    """
    # Make a copy to avoid modifying the original
    processed = tomo_data.copy()
    
    # Normalize to [0, 1] range
    min_val = processed.min()
    max_val = processed.max()
    if max_val > min_val:
        processed = (processed - min_val) / (max_val - min_val)
    
    # Enhance contrast
    processed = np.clip((processed - 0.1) * 1.25, 0, 1)
    
    return processed


# Function to load particle coordinates from JSON
def load_particle_coords(json_path):
    """
    Load particle coordinates from JSON file
    
    Parameters:
    json_path (str): Path to the JSON file
    
    Returns:
    list: List of (x, y, z) coordinate tuples
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract coordinates from points
        coords = []
        if 'points' in data:
            for point in data['points']:
                if 'location' in point:
                    loc = point['location']
                    coords.append((loc.get('x', 0), loc.get('y', 0), loc.get('z', 0)))
        
        return coords
    except Exception as e:
        print(f"Error loading {json_path}: {str(e)}")
        return []


# Function to generate a density map for a full tomogram
def generate_density_map(model, tomo_data, patch_size=64, stride=32, batch_size=8):
    """
    Generate a density map for a full tomogram
    
    Parameters:
    model (nn.Module): Trained model
    tomo_data (numpy.ndarray): 3D tomogram data
    patch_size (int): Size of cubic patches
    stride (int): Stride between patches
    batch_size (int): Batch size for prediction
    
    Returns:
    numpy.ndarray: Density map
    """
    model.eval()
    
    # Get dimensions
    depth, height, width = tomo_data.shape
    
    # Initialize density map and count array
    density_map = np.zeros_like(tomo_data, dtype=np.float32)
    count = np.zeros_like(tomo_data, dtype=np.float32)
    
    # Half size for patch extraction
    half_size = patch_size // 2
    
    # Extract patches
    patches = []
    coordinates = []
    
    print("Extracting patches...")
    for z in tqdm(range(half_size, depth - half_size, stride)):
        for y in range(half_size, height - half_size, stride):
            for x in range(half_size, width - half_size, stride):
                # Extract the patch
                patch = tomo_data[
                    z - half_size:z + half_size,
                    y - half_size:y + half_size,
                    x - half_size:x + half_size
                ]
                
                # Skip if patch is invalid
                if patch.shape != (patch_size, patch_size, patch_size):
                    continue
                
                patches.append(patch)
                coordinates.append((z, y, x))
                
                # Process in batches to avoid memory issues
                if len(patches) >= batch_size:
                    # Convert to tensor
                    batch_patches = torch.FloatTensor(np.array(patches)).unsqueeze(1).to(device)
                    
                    # Predict
                    with torch.no_grad():
                        batch_outputs = model(batch_patches)
                    
                    # Move to CPU and convert to numpy
                    batch_outputs = batch_outputs.detach().cpu().numpy()
                    
                    # Add to density map
                    for i, (z, y, x) in enumerate(coordinates):
                        output = batch_outputs[i, 0]
                        
                        # Add to density map
                        density_map[
                            z - half_size:z + half_size,
                            y - half_size:y + half_size,
                            x - half_size:x + half_size
                        ] += output
                        
                        # Increment count
                        count[
                            z - half_size:z + half_size,
                            y - half_size:y + half_size,
                            x - half_size:x + half_size
                        ] += 1
                    
                    # Clear lists
                    patches = []
                    coordinates = []
    
    # Process remaining patches
    if patches:
        # Convert to tensor
        batch_patches = torch.FloatTensor(np.array(patches)).unsqueeze(1).to(device)
        
        # Predict
        with torch.no_grad():
            batch_outputs = model(batch_patches)
        
        # Move to CPU and convert to numpy
        batch_outputs = batch_outputs.detach().cpu().numpy()
        
        # Add to density map
        for i, (z, y, x) in enumerate(coordinates):
            output = batch_outputs[i, 0]
            
            # Add to density map
            density_map[
                z - half_size:z + half_size,
                y - half_size:y + half_size,
                x - half_size:x + half_size
            ] += output
            
            # Increment count
            count[
                z - half_size:z + half_size,
                y - half_size:y + half_size,
                x - half_size:x + half_size
            ] += 1
    
    # Average overlapping regions
    density_map = np.divide(density_map, count, out=np.zeros_like(density_map), where=count > 0)
    
    return density_map


# Function to analyze validation tomograms for threshold calibration
def analyze_validation_tomograms():
    """
    Analyze validation tomograms to calibrate thresholds
    
    Returns:
    dict: Dictionary of calibrated thresholds
    """
    # Load the trained model
    model_path = os.path.join(model_dir, 'model.pth')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return None
    
    # Initialize model
    model = UNet3D(in_channels=1, out_channels=1, init_features=16)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Get list of training experiments
    train_experiments = [os.path.basename(p) for p in glob.glob(os.path.join(train_dir, 'static/ExperimentRuns/*'))]
    
    if not train_experiments:
        print("No training experiments found.")
        return None
    
    # Sample a few experiments for analysis
    sample_size = min(3, len(train_experiments))
    validation_experiments = train_experiments[:sample_size]
    print(f"Using {len(validation_experiments)} experiments for threshold calibration: {validation_experiments}")
    
    # Dictionary to store signal intensities for each particle type
    particle_intensities = {p_type: [] for p_type in particle_types}
    
    # Process each validation experiment
    for experiment in validation_experiments:
        print(f"\nProcessing validation experiment: {experiment}")
        
        # Load tomogram
        zarr_path = os.path.join(train_dir, 'static/ExperimentRuns', experiment, 'VoxelSpacing10.000/denoised.zarr')
        
        if not os.path.exists(zarr_path):
            print(f"Tomogram not found for experiment {experiment}")
            continue
        
        tomo_data = load_tomogram(zarr_path)
        if tomo_data is None:
            print(f"Failed to load tomogram for experiment {experiment}")
            continue
        
        tomo_data = preprocess_tomogram(tomo_data)
        
        # Generate density map
        print("Generating density map...")
        density_map = generate_density_map(model, tomo_data, patch_size=64, stride=32, batch_size=8)
        
        # Load ground truth particle positions
        ground_truth = {}
        
        for p_type in particle_types:
            json_path = os.path.join(train_dir, 'overlay/ExperimentRuns', experiment, 'Picks', f"{p_type}.json")
            
            if os.path.exists(json_path):
                coords = load_particle_coords(json_path)
                if coords:
                    ground_truth[p_type] = coords
                    print(f"Loaded {len(coords)} {p_type} coordinates")
        
        # Sample intensity values at ground truth positions
        for p_type, coords in ground_truth.items():
            for x, y, z in coords:
                # Convert physical coordinates to voxel indices
                z_idx, y_idx, x_idx = int(z / 10), int(y / 10), int(x / 10)
                
                # Check if the position is within the density map
                if (0 <= z_idx < density_map.shape[0] and 
                    0 <= y_idx < density_map.shape[1] and 
                    0 <= x_idx < density_map.shape[2]):
                    
                    # Get the intensity value
                    intensity = density_map[z_idx, y_idx, x_idx]
                    particle_intensities[p_type].append(intensity)
        
        # Free memory
        del tomo_data, density_map
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculate intensity statistics for each particle type
    intensity_stats = {}
    
    print("\nParticle intensity statistics:")
    for p_type, intensities in particle_intensities.items():
        if intensities:
            stats = {
                'mean': np.mean(intensities),
                'median': np.median(intensities),
                'min': np.min(intensities),
                'max': np.max(intensities),
                'p25': np.percentile(intensities, 25),
                'p10': np.percentile(intensities, 10),
                'p5': np.percentile(intensities, 5),
                'count': len(intensities)
            }
            intensity_stats[p_type] = stats
            
            print(f"  - {p_type} ({len(intensities)} particles):")
            print(f"      Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}")
            print(f"      Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
            print(f"      P5: {stats['p5']:.4f}, P10: {stats['p10']:.4f}, P25: {stats['p25']:.4f}")
    
    # Visualize intensity distributions
    plt.figure(figsize=(12, 8))
    
    for p_type, intensities in particle_intensities.items():
        if len(intensities) > 10:  # Only plot if we have enough samples
            plt.hist(intensities, bins=20, alpha=0.6, label=f"{p_type} (n={len(intensities)})")
    
    plt.xlabel('Density Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Density Values at Ground Truth Particle Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(visualization_dir, 'intensity_distributions.png'), dpi=200)
    plt.close()
    
    # Calculate calibrated thresholds based on validation analysis
    # Use much lower percentiles to ensure we find all types
    calibrated_thresholds = {}
    
    for p_type in scored_particle_types:
        if p_type in intensity_stats:
            stats = intensity_stats[p_type]
            
            # Use very low percentiles - prioritize recall over precision
            if p_type == 'apo-ferritin' or p_type == 'ribosome':
                # For easier particle types
                calibrated_thresholds[p_type] = stats['p5'] * 0.7  # 70% of 5th percentile
            elif p_type == 'beta-galactosidase':
                # For harder particle types
                calibrated_thresholds[p_type] = stats['p5'] * 0.6  # 60% of 5th percentile
            elif p_type == 'thyroglobulin':
                # More difficult to detect
                calibrated_thresholds[p_type] = stats['p5'] * 0.5  # 50% of 5th percentile
            elif p_type == 'virus-like-particle':
                # Most difficult to detect
                calibrated_thresholds[p_type] = stats['p5'] * 0.4  # 40% of 5th percentile
            else:
                # Default
                calibrated_thresholds[p_type] = stats['p5'] * 0.6
    
    # If we don't have statistics for some particle types, use defaults
    default_threshold = 0.002  # Very low default threshold
    for p_type in scored_particle_types:
        if p_type not in calibrated_thresholds:
            if p_type == 'thyroglobulin':
                calibrated_thresholds[p_type] = 0.0015  # Very low threshold
            elif p_type == 'virus-like-particle':
                calibrated_thresholds[p_type] = 0.0010  # Extremely low threshold
            elif p_type == 'beta-galactosidase':
                calibrated_thresholds[p_type] = 0.0018  # Low threshold
            else:
                calibrated_thresholds[p_type] = default_threshold
    
    print("\nCalibrated thresholds:")
    for p_type, threshold in calibrated_thresholds.items():
        print(f"  - {p_type}: {threshold:.4f}")
    
    return calibrated_thresholds



# Function to find local maxima in the density map
def find_local_maxima(density_map, min_distance=6, threshold_abs=0.05, threshold_rel=0.03):
    """
    Find local maxima in the density map
    
    Parameters:
    density_map (numpy.ndarray): Density map
    min_distance (int): Minimum distance between peaks
    threshold_abs (float): Minimum absolute threshold for peak
    threshold_rel (float): Minimum relative threshold for peak
    
    Returns:
    numpy.ndarray: Array of peak coordinates [z, y, x]
    """
    # Apply Gaussian smoothing to reduce noise
    smoothed_map = gaussian_filter(density_map, sigma=1.0)
    
    # Find local maxima
    coordinates = peak_local_max(
        smoothed_map,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=False
    )
    
    return coordinates


# Enhanced clustering function to detect all particle types
def calibrated_cluster_particles(coordinates, volume, particle_types, calibrated_thresholds):
    """
    Cluster particles using calibrated thresholds with a multi-pass approach to ensure all types are detected
    
    Parameters:
    coordinates (numpy.ndarray): Array of peak coordinates [z, y, x]
    volume (numpy.ndarray): Density map
    particle_types (list): List of particle types to consider
    calibrated_thresholds (dict): Calibrated thresholds for each particle type
    
    Returns:
    dict: Dictionary mapping particle type to list of coordinates
    """
    if len(coordinates) == 0:
        print("No coordinates found for clustering")
        return {p_type: [] for p_type in particle_types}
    
    # Sort peaks by intensity
    peak_values = np.array([volume[z, y, x] for z, y, x in coordinates])
    sorted_indices = np.argsort(peak_values)[::-1]  # Sort in descending order
    
    sorted_coordinates = coordinates[sorted_indices]
    sorted_values = peak_values[sorted_indices]
    
    # Dictionary to store particles by type
    particles_by_type = {p_type: [] for p_type in particle_types}
    
    # Assign peaks to particle types based on calibrated thresholds
    n_particles = {p_type: 0 for p_type in particle_types}
    
    # Set target and maximum number of particles per tomogram
    # Increased target numbers to ensure we find all types
    target_particles = {
        'apo-ferritin': 80,
        'beta-galactosidase': 60,
        'ribosome': 80,
        'thyroglobulin': 60,
        'virus-like-particle': 40
    }
    
    max_particles = {
        'apo-ferritin': 120,
        'beta-galactosidase': 100,
        'ribosome': 120,
        'thyroglobulin': 100,
        'virus-like-particle': 80
    }
    
    # Define size categories by radius for more balanced distribution
    size_categories = {
        'small': ['apo-ferritin'],
        'medium': ['beta-galactosidase', 'thyroglobulin'],
        'large': ['ribosome', 'virus-like-particle']
    }
    
    # Create reverse mapping of particle type to category
    particle_to_category = {}
    for category, p_types in size_categories.items():
        for p_type in p_types:
            particle_to_category[p_type] = category
    
    # Track assigned peaks
    already_assigned = set()
    
    # First pass: Assign top peaks to different size categories as evenly as possible
    size_counts = {cat: 0 for cat in size_categories}
    category_quotas = {
        'small': 0.30,  # 30% for small particles
        'medium': 0.40,  # 40% for medium particles
        'large': 0.30    # 30% for large particles
    }
    
    total_expected = min(len(sorted_coordinates), sum(target_particles.values()))
    category_limits = {
        cat: int(total_expected * quota) for cat, quota in category_quotas.items()
    }
    
    print("\nCategory distribution targets:")
    for cat, limit in category_limits.items():
        print(f"  - {cat}: {limit} particles")
    
    # First pass: Assign peaks to categories based on thresholds and type quotas
    for i, (z, y, x) in enumerate(sorted_coordinates):
        peak_value = sorted_values[i]
        
        # Skip if already assigned
        if i in already_assigned:
            continue
        
        # Skip if this peak is too close to an already assigned peak
        too_close = False
        for idx in already_assigned:
            z2, y2, x2 = sorted_coordinates[idx]
            dist = np.sqrt((z - z2)**2 + (y - y2)**2 + (x - x2)**2)
            if dist < 5:  # 5 voxels minimum distance (reduced from original 6)
                too_close = True
                break
        
        if too_close:
            continue
        
        # Try to assign to particle types that are under their limits
        assigned = False
        
        # Prioritize underrepresented categories
        categories_sorted = sorted(size_categories.keys(), key=lambda cat: size_counts[cat] / max(1, category_limits[cat]))
        
        for category in categories_sorted:
            if size_counts[category] >= category_limits[category] * 1.5:  # Allow going over by 50%
                continue  # Skip if category is well beyond limit
            
            # Try each particle type in this category
            p_types = size_categories[category]
            
            # Sort types by how far they are from their target
            p_types_sorted = sorted(p_types, 
                                  key=lambda p: (n_particles[p] / max(1, target_particles[p])))
            
            for p_type in p_types_sorted:
                if p_type not in calibrated_thresholds:
                    continue  # Skip if we don't have a threshold
                
                if n_particles[p_type] >= max_particles[p_type]:
                    continue  # Skip if at limit
                
                # Check threshold - use a lower threshold as we go deeper into the sorted list
                threshold_factor = 1.0 - (i / len(sorted_coordinates)) * 0.2  # Gradually decrease threshold (reduced from 0.3)
                effective_threshold = calibrated_thresholds[p_type] * threshold_factor
                
                if peak_value >= effective_threshold:
                    # Convert voxel coordinates to physical coordinates
                    physical_coords = (x * 10.0, y * 10.0, z * 10.0)
                    particles_by_type[p_type].append(physical_coords)
                    n_particles[p_type] += 1
                    already_assigned.add(i)
                    size_counts[category] += 1
                    assigned = True
                    break
            
            if assigned:
                break
    
    # Print distribution after first pass
    print("\nParticle distribution after first pass:")
    for p_type in particle_types:
        print(f"  - {p_type}: {n_particles[p_type]} particles")
    
    # Check for missing or underrepresented types
    underrepresented = []
    for p_type in particle_types:
        # If we have less than 20% of target for this type, consider it underrepresented
        if n_particles[p_type] < target_particles.get(p_type, 50) * 0.2:
            underrepresented.append(p_type)
    
    print(f"Underrepresented types: {underrepresented}")
    
    # Second pass: Focus on underrepresented types with much lower thresholds
    if underrepresented:
        print("Running second pass for underrepresented types...")
        
        # Sort by intensity again - we'll use the remaining strong signals
        remaining_indices = [i for i in range(len(sorted_coordinates)) if i not in already_assigned]
        
        for i in remaining_indices:
            z, y, x = sorted_coordinates[i]
            peak_value = sorted_values[i]
            
            # Skip if too close to an already assigned peak
            too_close = False
            for idx in already_assigned:
                z2, y2, x2 = sorted_coordinates[idx]
                dist = np.sqrt((z - z2)**2 + (y - y2)**2 + (x - x2)**2)
                if dist < 5:  # 5 voxels minimum distance
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Try to assign to underrepresented types with very low thresholds
            for p_type in underrepresented:
                # Target at least 20% of the expected count for each type
                min_target = target_particles.get(p_type, 50) * 0.2
                
                if n_particles[p_type] >= min_target:
                    continue
                
                # Use a very low threshold - just to get some representation
                very_low_threshold = calibrated_thresholds[p_type] * 0.4  # 40% of calibrated threshold
                
                if peak_value >= very_low_threshold:
                    # Convert voxel coordinates to physical coordinates
                    physical_coords = (x * 10.0, y * 10.0, z * 10.0)
                    particles_by_type[p_type].append(physical_coords)
                    n_particles[p_type] += 1
                    already_assigned.add(i)
                    
                    # Update category count
                    category = particle_to_category.get(p_type)
                    if category:
                        size_counts[category] += 1
                    
                    break  # Assign at most one particle per coordinate
        
        # Print distribution after second pass
        print("\nParticle distribution after second pass:")
        for p_type in particle_types:
            print(f"  - {p_type}: {n_particles[p_type]} particles")
    
    # Check if we still have missing particle types
    missing_types = [p_type for p_type in particle_types if n_particles[p_type] == 0]
    
    # Third pass: Desperate measures for still-missing types
    if missing_types:
        print(f"Still missing types: {missing_types}. Using desperate measures.")
        
        # Use remaining peaks with extremely low thresholds
        remaining_indices = [i for i in range(len(sorted_coordinates)) if i not in already_assigned]
        
        # For each missing type, try to find at least a few candidates
        for p_type in missing_types:
            # Count needed for minimum representation
            min_count = 5  # At least 5 particles of each type
            
            # Find peaks for this type, sorted by intensity
            candidate_indices = []
            for i in remaining_indices:
                z, y, x = sorted_coordinates[i]
                # Use almost no threshold - we just need some representation
                candidate_indices.append((i, sorted_values[i], z, y, x))
            
            # Sort candidates by intensity
            candidate_indices.sort(key=lambda x: x[1], reverse=True)
            
            # Take top candidates that aren't too close to existing particles
            for i, value, z, y, x in candidate_indices:
                if n_particles[p_type] >= min_count:
                    break
                
                # Check if too close to already assigned particles
                too_close = False
                for idx in already_assigned:
                    z2, y2, x2 = sorted_coordinates[idx]
                    dist = np.sqrt((z - z2)**2 + (y - y2)**2 + (x - x2)**2)
                    if dist < 5:  # 5 voxels minimum distance
                        too_close = True
                        break
                
                if too_close:
                    continue
                
                # Convert voxel coordinates to physical coordinates
                physical_coords = (x * 10.0, y * 10.0, z * 10.0)
                particles_by_type[p_type].append(physical_coords)
                n_particles[p_type] += 1
                already_assigned.add(i)
                
                # Update category count
                category = particle_to_category.get(p_type)
                if category:
                    size_counts[category] += 1
            
            print(f"Added {n_particles[p_type]} emergency particles for {p_type}")
    
    # Fourth pass: Fill in with remaining high-value peaks to reach targets
    remaining_indices = [i for i in range(len(sorted_coordinates)) if i not in already_assigned]
    
    # For each particle type that's below target
    below_target_types = [(p, target_particles.get(p, 50) - n_particles[p]) 
                          for p in particle_types 
                          if n_particles[p] < target_particles.get(p, 50)]
    
    # Sort by how far below target they are
    below_target_types.sort(key=lambda x: x[1], reverse=True)
    
    if below_target_types:
        print("\nRunning fourth pass to reach targets for types below target...")
        
        for p_type, deficit in below_target_types:
            if deficit <= 0:
                continue
                
            # Use lower thresholds for final pass
            final_threshold = calibrated_thresholds[p_type] * 0.3  # 30% of calibrated threshold
            
            # Count particles added for this type
            added = 0
            
            for i in remaining_indices[:]:  # Create a copy to modify during iteration
                if added >= deficit:
                    break
                    
                z, y, x = sorted_coordinates[i]
                peak_value = sorted_values[i]
                
                # Skip if already assigned (could happen during this pass)
                if i in already_assigned:
                    continue
                
                # Skip if too close to an already assigned peak
                too_close = False
                for idx in already_assigned:
                    z2, y2, x2 = sorted_coordinates[idx]
                    dist = np.sqrt((z - z2)**2 + (y - y2)**2 + (x - x2)**2)
                    if dist < 5:  # 5 voxels minimum distance
                        too_close = True
                        break
                
                if too_close:
                    continue
                
                # Check if this peak meets our threshold
                if peak_value >= final_threshold:
                    # Convert voxel coordinates to physical coordinates
                    physical_coords = (x * 10.0, y * 10.0, z * 10.0)
                    particles_by_type[p_type].append(physical_coords)
                    n_particles[p_type] += 1
                    already_assigned.add(i)
                    added += 1
                    
                    # Update category count
                    category = particle_to_category.get(p_type)
                    if category:
                        size_counts[category] += 1
            
            print(f"Added {added} additional particles for {p_type}")
    
    # Print final statistics
    print("\nFinal particle distribution:")
    for p_type in particle_types:
        print(f"  - {p_type}: {n_particles[p_type]} particles")
    
    print(f"Total particles found: {sum(n_particles.values())}")
    
    return particles_by_type


# Function to write predictions to JSON
def write_predictions_to_json(particles_by_type, output_path):
    """
    Write particle predictions to JSON
    
    Parameters:
    particles_by_type (dict): Dictionary mapping particle type to list of coordinates
    output_path (str): Path to save the JSON file
    """
    # Format the predictions according to the submission format
    prediction = {"points": []}
    
    for p_type, coords in particles_by_type.items():
        for x, y, z in coords:
            prediction["points"].append({
                "location": {"x": x, "y": y, "z": z},
                "type": p_type
            })
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(prediction, f)
    
    print(f"Wrote {len(prediction['points'])} particles to {output_path}")


# Function to visualize predictions with different colors for different particle types
def visualize_tomogram_with_particles(tomo_data, particles_by_type, output_path, slices=None):
    """
    Visualize tomogram slices with colored particle markers
    
    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    particles_by_type (dict): Dictionary mapping particle type to list of coordinates
    output_path (str): Path to save the visualization
    slices (list): List of slice indices to visualize (default is middle slice)
    """
    # Get tomogram dimensions
    depth, height, width = tomo_data.shape
    
    # Choose slices if not provided
    if slices is None:
        slices = [depth // 4, depth // 2, 3 * depth // 4]
    
    # Create figure
    fig, axes = plt.subplots(1, len(slices), figsize=(6 * len(slices), 6))
    if len(slices) == 1:
        axes = [axes]
    
    # For each slice
    for i, slice_idx in enumerate(slices):
        # Show the tomogram slice
        axes[i].imshow(tomo_data[slice_idx], cmap='gray')
        axes[i].set_title(f'Z-Slice {slice_idx}/{depth}')
        
        # Get slice range (particles near this slice)
        slice_range = 10  # Consider particles within Â±10 slices
        z_min = (slice_idx - slice_range) * 10.0  # Convert to physical coordinates
        z_max = (slice_idx + slice_range) * 10.0
        
        # Add circles for each particle type
        for p_type, coords in particles_by_type.items():
            # Skip if particle type not in our dictionary or no coordinates
            if p_type not in particle_types or not coords:
                continue
                
            # Get color and radius for this particle type
            color = particle_types[p_type]['color']
            radius = particle_types[p_type]['radius'] / 10.0  # Convert to voxel units
            
            # Count particles in this slice
            slice_particles = [(x, y, z) for x, y, z in coords if z_min <= z <= z_max]
            n_particles = len(slice_particles)
            
            # Skip if no particles of this type in this slice
            if n_particles == 0:
                continue
            
            # Add to legend
            axes[i].plot([], [], 'o', color=color, label=f'{p_type} ({n_particles})')
            
            # Add circle for each particle
            for x, y, z in slice_particles:
                # Convert physical coordinates to pixel coordinates
                y_px = y / 10.0
                x_px = x / 10.0
                
                # Calculate alpha based on distance from the slice
                z_px = z / 10.0
                alpha = 1.0 - abs(z_px - slice_idx) / slice_range
                
                # Add circle
                circle = plt.Circle((x_px, y_px), radius, color=color, fill=False, alpha=alpha, linewidth=1.5)
                axes[i].add_patch(circle)
        
        # Add legend
        axes[i].legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()



# Function to visualize density map with particle locations
def visualize_density_map_with_particles(density_map, particles_by_type, output_path, slice_idx=None):
    """
    Visualize density map with particle locations
    
    Parameters:
    density_map (numpy.ndarray): 3D density map
    particles_by_type (dict): Dictionary mapping particle type to list of coordinates
    output_path (str): Path to save the visualization
    slice_idx (int): Index of slice to visualize (default is middle slice)
    """
    # Get dimensions
    depth, height, width = density_map.shape
    
    # Choose slice if not provided
    if slice_idx is None:
        slice_idx = depth // 2
    
    # Create a custom colormap for density
    cmap_name = 'hot_alpha'
    colors = [(0, 0, 0, 0)]  # Start with transparent
    for i in range(1, 256):
        # Red-yellow colormap with increasing alpha
        alpha = i / 255.0
        if i < 128:
            # From transparent to red
            colors.append((i / 127.0, 0, 0, alpha * 0.7))
        else:
            # From red to yellow
            colors.append((1, (i - 128) / 127.0, 0, alpha * 0.7))
    
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Show the density map slice
    plt.imshow(density_map[slice_idx], cmap=custom_cmap)
    
    # Get slice range (particles near this slice)
    slice_range = 10  # Consider particles within Â±10 slices
    z_min = (slice_idx - slice_range) * 10.0  # Convert to physical coordinates
    z_max = (slice_idx + slice_range) * 10.0
    
    # Add markers for each particle type
    for p_type, coords in particles_by_type.items():
        # Skip if particle type not in our dictionary or no coordinates
        if p_type not in particle_types or not coords:
            continue
            
        # Get color for this particle type
        color = particle_types[p_type]['color']
        
        # Count particles in this slice
        slice_particles = [(x, y, z) for x, y, z in coords if z_min <= z <= z_max]
        n_particles = len(slice_particles)
        
        # Skip if no particles of this type in this slice
        if n_particles == 0:
            continue
        
        # Extract x, y coordinates for this slice
        x_coords = [x / 10.0 for x, y, z in slice_particles]
        y_coords = [y / 10.0 for x, y, z in slice_particles]
        
        # Plot particles
        plt.scatter(x_coords, y_coords, color=color, marker='o', s=50, facecolors='none', 
                   label=f'{p_type} ({n_particles})', linewidth=1.5)
    
    plt.title(f'Density Map with Particles (Z-Slice {slice_idx}/{depth})', fontsize=14)
    plt.colorbar(label='Density Value')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()



# Function to evaluate the prediction quality
def evaluate_prediction(particles_by_type):
    """
    Evaluate prediction quality using heuristic metrics
    
    Parameters:
    particles_by_type (dict): Dictionary mapping particle type to list of coordinates
    
    Returns:
    float: Heuristic quality score (0-1)
    """
    # Calculate score based on particle distribution
    scores = []
    
    # Expected particle counts based on observations
    expected_counts = {
        'apo-ferritin': 80,
        'beta-galactosidase': 60,
        'ribosome': 80,
        'thyroglobulin': 60,
        'virus-like-particle': 40
    }
    
    for p_type, coords in particles_by_type.items():
        if p_type not in expected_counts:
            continue
            
        # Count
        count = len(coords)
        expected = expected_counts[p_type]
        
        # Calculate normalized score (1.0 if count matches expected, less otherwise)
        # Use min-max normalization
        ratio = min(count / expected, 1.0) if expected > 0 else 0.0
        
        # Weight by particle importance
        weight = particle_types[p_type].get('weight', 1.0)
        
        # Add to scores
        scores.append(ratio * weight)
    
    # Calculate final score
    total_weight = sum(particle_types[p_type].get('weight', 1.0) for p_type in expected_counts if p_type in particle_types)
    
    if total_weight > 0:
        return sum(scores) / total_weight
    else:
        return 0.0


# Main function to process test tomograms
def process_test_tomograms():
    """
    Process test tomograms and generate predictions
    """
    # Load the trained model
    model_path = os.path.join(model_dir, 'model.pth')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    # Initialize model
    model = UNet3D(in_channels=1, out_channels=1, init_features=16)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Get calibrated thresholds
    calibrated_thresholds = analyze_validation_tomograms()
    
    if calibrated_thresholds is None:
        print("Failed to calibrate thresholds. Using default values.")
        calibrated_thresholds = {
            'apo-ferritin': 0.0020,
            'beta-galactosidase': 0.0018,
            'ribosome': 0.0020,
            'thyroglobulin': 0.0015,
            'virus-like-particle': 0.0010
        }
    
    # Modify thresholds to ensure better detection of all types
    # Further lower thresholds for all types to ensure better detection
    print("\nAdjusting thresholds for better particle type detection:")
    
    # Hardcode lower thresholds to ensure all types are detected
    for p_type in calibrated_thresholds:
        # Reduce all thresholds by 20%
        calibrated_thresholds[p_type] *= 0.8
    
    # Specifically lower thresholds for problematic types
    if 'thyroglobulin' in calibrated_thresholds:
        calibrated_thresholds['thyroglobulin'] = min(calibrated_thresholds['thyroglobulin'], 0.0015)
    
    if 'virus-like-particle' in calibrated_thresholds:
        calibrated_thresholds['virus-like-particle'] = min(calibrated_thresholds['virus-like-particle'], 0.0008)
    
    if 'beta-galactosidase' in calibrated_thresholds:
        calibrated_thresholds['beta-galactosidase'] = min(calibrated_thresholds['beta-galactosidase'], 0.0015)
    
    for p_type, threshold in calibrated_thresholds.items():
        print(f"  - {p_type}: {threshold:.4f}")
    
    # Get list of test experiments
    test_experiments = [os.path.basename(p) for p in glob.glob(os.path.join(test_dir, 'static/ExperimentRuns/*'))]
    
    if not test_experiments:
        print("No test experiments found.")
        return
    
    print(f"\nFound {len(test_experiments)} test experiments: {test_experiments}")
    
    # List to store all particle predictions for final submission
    all_particles = []
    
    # Process each test experiment
    for experiment in test_experiments:
        print(f"\nProcessing test experiment: {experiment}")
        
        # Create experiment output directory
        experiment_output_dir = os.path.join(output_dir, 'test', experiment)
        os.makedirs(experiment_output_dir, exist_ok=True)
        
        # Load tomogram
        zarr_path = os.path.join(test_dir, 'static/ExperimentRuns', experiment, 'VoxelSpacing10.000/denoised.zarr')
        
        if not os.path.exists(zarr_path):
            print(f"Tomogram not found for experiment {experiment}")
            continue
        
        tomo_data = load_tomogram(zarr_path)
        if tomo_data is None:
            print(f"Failed to load tomogram for experiment {experiment}")
            continue
        
        tomo_data = preprocess_tomogram(tomo_data)
        
        # Generate density map
        print("Generating density map...")
        density_map = generate_density_map(model, tomo_data, patch_size=64, stride=32, batch_size=8)
        
        # Save density map
        np.save(os.path.join(experiment_output_dir, 'density_map.npy'), density_map)
        
        # Find local maxima
        print("Finding local maxima...")
        # Use very low thresholds to ensure we detect harder particles
        coordinates = find_local_maxima(
            density_map, 
            min_distance=6,  # Decreased from 8 to detect smaller particles
            threshold_abs=0.05,  # Decreased to be more sensitive
            threshold_rel=0.03   # Decreased to be more sensitive
        )
        
        print(f"Found {len(coordinates)} potential particle locations")
        
        # Cluster particles
        print("Clustering particles...")
        particles_by_type = calibrated_cluster_particles(
            coordinates, 
            density_map, 
            scored_particle_types,
            calibrated_thresholds
        )
        
        # Evaluate prediction quality
        quality_score = evaluate_prediction(particles_by_type)
        print(f"Prediction quality score: {quality_score:.4f}")
        
        # Check for missing particle types
        missing_types = [p_type for p_type in scored_particle_types if len(particles_by_type[p_type]) == 0]
        
        # If quality score is too low or we're missing particle types, try again with even lower thresholds
        if quality_score < 0.5 or missing_types:
            print(f"Low quality prediction (score: {quality_score:.4f}) or missing types: {missing_types}. Retrying with lower thresholds...")
            
            # Try with even lower thresholds
            lower_thresholds = {p_type: t * 0.6 for p_type, t in calibrated_thresholds.items()}
            
            # Especially lower thresholds for missing types
            for p_type in missing_types:
                lower_thresholds[p_type] = calibrated_thresholds[p_type] * 0.4
            
            # Find local maxima again with even lower thresholds
            coordinates = find_local_maxima(
                density_map, 
                min_distance=5,  # Even smaller minimum distance
                threshold_abs=0.03,  # Much lower threshold
                threshold_rel=0.02   # Much lower relative threshold
            )
            
            print(f"Found {len(coordinates)} potential particle locations (retry)")
            
            # Cluster particles again
            particles_by_type = calibrated_cluster_particles(
                coordinates, 
                density_map, 
                scored_particle_types,
                lower_thresholds
            )
            
            # Re-evaluate
            quality_score = evaluate_prediction(particles_by_type)
            print(f"New prediction quality score: {quality_score:.4f}")
            
            # If still missing particle types, use desperate measures
            missing_types = [p_type for p_type in scored_particle_types if len(particles_by_type[p_type]) == 0]
            if missing_types:
                print(f"Still missing types: {missing_types}. Trying desperate measures...")
                
                # For each missing type, find at least a few candidates
                for p_type in missing_types:
                    # Use peak_local_max with very low thresholds specific to this type
                    desperate_coords = peak_local_max(
                        density_map,
                        min_distance=4,
                        threshold_abs=0.01,
                        threshold_rel=0.01,
                        exclude_border=False,
                        num_peaks=10  # Limit to top 10 peaks
                    )
                    
                    # Add top 5 coordinates to this particle type
                    for z, y, x in desperate_coords[:5]:
                        physical_coords = (x * 10.0, y * 10.0, z * 10.0)
                        particles_by_type[p_type].append(physical_coords)
                    
                    print(f"Added {min(5, len(desperate_coords))} emergency particles for {p_type}")
        
        # Write predictions to JSON
        output_path = os.path.join(submission_dir, f"{experiment}.json")
        write_predictions_to_json(particles_by_type, output_path)
        
        # Visualize tomogram with particles
        vis_path = os.path.join(visualization_dir, f"{experiment}_tomogram_with_particles.png")
        visualize_tomogram_with_particles(tomo_data, particles_by_type, vis_path)
        
        # Visualize density map with particles
        density_vis_path = os.path.join(visualization_dir, f"{experiment}_density_map_with_particles.png")
        visualize_density_map_with_particles(density_map, particles_by_type, density_vis_path)
        
        # Add to list of all particles for CSV submission
        for p_type, coords in particles_by_type.items():
            for x, y, z in coords:
                all_particles.append({
                    'experiment': experiment,
                    'particle_type': p_type,
                    'x': x,
                    'y': y,
                    'z': z
                })
        
        # Free memory
        del tomo_data, density_map
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create CSV submission file
    if all_particles:
        # Create DataFrame
        submission_df = pd.DataFrame(all_particles)
        
        # Add id column
        submission_df['id'] = range(len(submission_df))
        
        # Reorder columns
        submission_df = submission_df[['id', 'experiment', 'particle_type', 'x', 'y', 'z']]
        
        # Save submission
        submission_path = os.path.join(submission_dir, 'submission.csv')
        submission_df.to_csv(submission_path, index=False)
        
        print(f"\nSaved submission to {submission_path}")
        print(f"Total predictions: {len(submission_df)}")
        
        # Print submission statistics
        print("\nSubmission statistics:")
        print(submission_df.groupby(['experiment', 'particle_type']).size().unstack(fill_value=0))
    
    print("\nAll test tomograms processed.")



# Run the main function
if __name__ == "__main__":
    print("Starting test tomogram processing...")
    process_test_tomograms()
    print("Test tomogram processing completed.")


# Function to display test images
def display_test_images():
    """
    Display all test images generated in the visualization directory
    """
    import glob
    import matplotlib.pyplot as plt
    from IPython.display import display
    
    # Get all visualization files
    vis_files = glob.glob(os.path.join(visualization_dir, '*.png'))
    
    if not vis_files:
        print("No visualization files found in", visualization_dir)
        return
    
    print(f"Found {len(vis_files)} visualization files.")
    
    # Display each visualization file
    for vis_file in sorted(vis_files):
        filename = os.path.basename(vis_file)
        print(f"\nDisplaying: {filename}")
        
        # Load and display the image
        img = plt.imread(vis_file)
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(filename, fontsize=14)
        plt.tight_layout()
        plt.show()

# Call the function to display the images
display_test_images()




