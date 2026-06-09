%%capture
print("Hidden")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------
# 1. Dataset Path Setup (Adjust as needed)
# ---------------------------------------------------
DATA_DIR = Path("/kaggle/input/waveform-inversion")
TRAIN_DIR = DATA_DIR / "train_samples"
TEST_DIR = DATA_DIR / "test"

# ---------------------------------------------------
# 2. Load Vel/Style Family Data (FlatVel_A, CurveVel_A, etc.)
# ---------------------------------------------------
def load_vel_style_data(family_type="FlatVel_A", sample_num=1):
    """Load Vel/Style family data from structured directories"""
    data_path = TRAIN_DIR / family_type / "data" / f"data{sample_num}.npy"
    model_path = TRAIN_DIR / family_type / "model" / f"model{sample_num}.npy"
    
    seismic = np.load(data_path)  # Shape: (500, sources, time, receivers)
    velocity = np.load(model_path) # Shape: (500, height, width)
    return seismic, velocity

# Example: Load FlatVel_A's first sample
flat_vel_seismic, flat_vel_maps = load_vel_style_data("FlatVel_A", 1)

# ---------------------------------------------------
# 3. Load Fault Family Data (FlatFault_A, CurveFault_A etc.)
# ---------------------------------------------------
def load_fault_data(family_type="FlatFault_A", n=2, file_index=0):
    """Load Fault family data with complex naming convention"""
    seis_path = TRAIN_DIR / family_type / f"seis{n}_1_{file_index}.npy"
    vel_path = TRAIN_DIR / family_type / f"vel{n}_1_{file_index}.npy"
    
    seismic = np.load(seis_path)    # Shape: (500, sources, time, receivers)
    velocity = np.load(vel_path)    # Shape: (500, height, width)
    return seismic, velocity

# Example: Load FlatFault_A's seis2_1_0.npy and vel2_1_0.npy
flat_fault_seismic, flat_fault_maps = load_fault_data("FlatFault_A", n=2, file_index=0)

# ---------------------------------------------------
# 4. Basic Data Exploration
# ---------------------------------------------------
# Updated visualization function
def explore_data(seismic, velocity, family_name):
    """Print basic stats and visualize samples"""
    print(f"\n=== {family_name} Data ===")
    print(f"Seismic shape: {seismic.shape}")
    print(f"Velocity maps shape: {velocity.shape}")
    
    # Remove singleton dimension from velocity maps
    velocity = velocity.squeeze()  # Shape becomes (500, 70, 70)
    
    # Visualize first sample
    plt.figure(figsize=(12,5))
    
    # Seismic plot (first source, first receiver)
    plt.subplot(121)
    plt.plot(seismic[0,0,:,0]) 
    plt.title(f"{family_name} Waveform")
    
    # Velocity map plot
    plt.subplot(122)
    plt.imshow(velocity[0], cmap='viridis')  # Now 2D: (70,70)
    plt.title(f"{family_name} Velocity")
    plt.colorbar()
    plt.show()

# Explore different families
# Load data with corrected shape
flat_vel_seismic, flat_vel_maps = load_vel_style_data("FlatVel_A", 1)
flat_vel_maps = flat_vel_maps.squeeze()  # Apply squeeze globally

# Now visualization will work
explore_data(flat_vel_seismic, flat_vel_maps, "FlatVel_A")

# ---------------------------------------------------
# 5. Test Data Handling
# ---------------------------------------------------
# ---------------------------------------------------
# 5. Updated Test Data Handling with Error Checking
# ---------------------------------------------------
def load_test_data():
    """Load test seismic data with proper path handling"""
    # Check available test files
    test_files = list(TEST_DIR.glob("*.npy")) + list(TEST_DIR.glob("*/*.npy"))
    
    if not test_files:
        raise FileNotFoundError(f"No .npy files found in {TEST_DIR}")
        
    print("Available test files:")
    for f in test_files[:3]:  # Show first 3 files to avoid clutter
        print(f" - {f.name}")
    
    # Load first test file as example
    return np.load(test_files[0])

# Example usage
try:
    test_seismic = load_test_data()
    print("\n=== Test Data Shape ===")
    print(f"Test seismic shape: {test_seismic.shape}")
except Exception as e:
    print(f"\nError: {str(e)}")
    print("Check the exact test data path structure in the competition dataset!")


# 1️⃣ Count and list family types in train_samples
train_family_types = [f.name for f in TRAIN_DIR.iterdir() if f.is_dir()]
print(f"Total family types in train_samples: {len(train_family_types)}")
print("Train family types:")
for f in train_family_types:
    print(f" - {f}")

# 2️⃣ For each family, show its subfolders (like data, model/vel)
print("\nSubfolders in each train family:")
for fam in train_family_types:
    subfolders = [sub.name for sub in (TRAIN_DIR / fam).iterdir() if sub.is_dir()]
    print(f"{fam}: {subfolders}")

# 3️⃣ List test files (to see test data structure)
print("\nTest directory files:")
test_items = list(TEST_DIR.glob("*"))
for item in test_items[:5]:  # Show first 5 items only
    print(f" - {item.name}")



import numpy as np
from pathlib import Path

# Set the training directory path
#DATA_DIR = Path("/kaggle/input/waveform-inversion")
#TRAIN_DIR = DATA_DIR / "train_samples"

# Get all family folders inside train_samples
family_dirs = [f for f in TRAIN_DIR.iterdir() if f.is_dir()]
print(f"Total families found: {len(family_dirs)}\n")

for family in sorted(family_dirs):
    print(f"\n=== {family.name} ===")
    
    # Case 1: If it has 'data' and 'model' or 'vel' folders
    data_dir = family / "data"
    model_dir = family / "model"
    if data_dir.exists() and model_dir.exists():
        data_files = list(data_dir.glob("*.npy"))
        model_files = list(model_dir.glob("*.npy"))
        
        print(f"Data samples: {len(data_files)}, Model samples: {len(model_files)}")
        if data_files:
            data_sample = np.load(data_files[0])
            model_sample = np.load(model_files[0])
            print(f" - Data sample shape: {data_sample.shape}")
            print(f" - Model sample shape: {model_sample.shape}")
    
    # Case 2: If files are directly inside (e.g., FlatFault_*, CurveFault_*)
    else:
        npy_files = list(family.glob("*.npy"))
        print(f"Raw .npy files: {len(npy_files)}")
        
        # Analyze file naming pattern
        print("Example file names:")
        for f in npy_files[:3]:  # Show up to 3
            print(f" - {f.name}")
        
        # Check shape of the first 1 or 2 files (if available)
        if npy_files:
            sample_data = np.load(npy_files[0])
            print(f" - First sample shape: {sample_data.shape}")
            if len(npy_files) > 1:
                sample_data2 = np.load(npy_files[1])
                print(f" - Second sample shape: {sample_data2.shape}")



import pandas as pd

data_summary = []
for family in family_dirs:
    subdirs = [d.name for d in family.iterdir() if d.is_dir()]
    data_files = list((family/"data").glob("*.npy")) if "data" in subdirs else list(family.glob("seis*.npy"))
    model_files = list((family/"model").glob("*.npy")) if "model" in subdirs else list(family.glob("vel*.npy"))
    
    data_summary.append({
        "Family": family.name,
        "Data Files": len(data_files),
        "Model Files": len(model_files),
        "Has Subfolders": subdirs if subdirs else "No",
    })

df_summary = pd.DataFrame(data_summary)
print(df_summary)



import numpy as np
from pathlib import Path

# Set train data directory
DATA_DIR = Path("/kaggle/input/waveform-inversion/train_samples")

# List all family folders inside train_samples
family_dirs = [f for f in DATA_DIR.iterdir() if f.is_dir()]

# Initialize results dictionary
stats_summary = []

# Loop through each family
for family_path in family_dirs:
    family_name = family_path.name
    subfolders = [f.name for f in family_path.iterdir() if f.is_dir()]
    
    # Initialize holders for statistics
    seismic_shapes = []
    velocity_shapes = []
    seismic_vals = []
    velocity_vals = []
    
    # Case 1: Structured folder with 'data' and 'model' subfolders
    if "data" in subfolders and "model" in subfolders:
        data_files = sorted((family_path / "data").glob("*.npy"))
        model_files = sorted((family_path / "model").glob("*.npy"))
        
        for data_file, model_file in zip(data_files, model_files):
            try:
                seismic = np.load(data_file)
                velocity = np.load(model_file)
                seismic_vals.append([np.min(seismic), np.max(seismic), np.mean(seismic), np.std(seismic)])
                velocity_vals.append([np.min(velocity), np.max(velocity), np.mean(velocity), np.std(velocity)])
                seismic_shapes.append(seismic.shape)
                velocity_shapes.append(velocity.shape)
            except Exception as e:
                print(f"Error loading file in {family_name}: {e}")
                
    # Case 2: Flat folder (e.g. FlatFault_*) with 'seis*.npy' and 'vel*.npy'
    else:
        seis_files = sorted(family_path.glob("seis*.npy"))
        vel_files = sorted(family_path.glob("vel*.npy"))
        
        for seis_file, vel_file in zip(seis_files, vel_files):
            try:
                seismic = np.load(seis_file)
                velocity = np.load(vel_file)
                seismic_vals.append([np.min(seismic), np.max(seismic), np.mean(seismic), np.std(seismic)])
                velocity_vals.append([np.min(velocity), np.max(velocity), np.mean(velocity), np.std(velocity)])
                seismic_shapes.append(seismic.shape)
                velocity_shapes.append(velocity.shape)
            except Exception as e:
                print(f"Error loading file in {family_name}: {e}")

    # Summarize statistics for this family
    if seismic_vals and velocity_vals:
        seismic_vals = np.array(seismic_vals)
        velocity_vals = np.array(velocity_vals)
        stats_summary.append({
            "Family": family_name,
            "Num Samples": len(seismic_vals),
            "Seismic Shape Example": seismic_shapes[0],
            "Velocity Shape Example": velocity_shapes[0],
            "Seismic Mean ± Std": f"{np.mean(seismic_vals[:,2]):.3f} ± {np.mean(seismic_vals[:,3]):.3f}",
            "Velocity Mean ± Std": f"{np.mean(velocity_vals[:,2]):.3f} ± {np.mean(velocity_vals[:,3]):.3f}",
            "Seismic Min/Max": f"{np.min(seismic_vals[:,0]):.2f} / {np.max(seismic_vals[:,1]):.2f}",
            "Velocity Min/Max": f"{np.min(velocity_vals[:,0]):.2f} / {np.max(velocity_vals[:,1]):.2f}"
        })

# Print summary
import pandas as pd
df_stats = pd.DataFrame(stats_summary)
df_stats.head()



import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Set the train directory
#DATA_DIR = Path("/kaggle/input/waveform-inversion/train_samples")
fault_families = [f for f in DATA_DIR.iterdir() if f.is_dir() and ('Fault' in f.name)]

print("=== Naming Pattern Checker ===\n")
pattern_summary = []

for family in fault_families:
    files = list(family.glob("*.npy"))
    print(f"Family: {family.name}")
    for f in files[:3]:  # Show first 3 files only
        print(f" - {f.name}")
        match = re.match(r"(seis|vel)(\d+)_(\d+)_(\d+)\.npy", f.name)
        if match:
            prefix, version, set_num, index = match.groups()
            pattern_summary.append((family.name, prefix, version, set_num, index))
    print()

# Convert to DataFrame (optional)
import pandas as pd
df_patterns = pd.DataFrame(pattern_summary, columns=["Family", "Type", "Version", "Set", "Index"])
print("Pattern Summary:\n", df_patterns.head())





