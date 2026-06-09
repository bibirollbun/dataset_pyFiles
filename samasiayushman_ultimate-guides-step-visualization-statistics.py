import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


df = pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')
print(df.head())


sample = df.iloc[0]
print(f"\nSample tomogram: {sample['tomo_id']}")
print(f"Motor coordinates: {sample['Motor axis 0']}, {sample['Motor axis 1']}, {sample['Motor axis 2']}")
print(f"Volume shape: {sample['Array shape (axis 0)']}x{sample['Array shape (axis 1)']}x{sample['Array shape (axis 2)']}")
print(f"Voxel spacing: {sample['Voxel spacing']} nm")


class SyntheticTomogramDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.volume_shape = (64, 64, 64) 
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
      
        volume = np.zeros(self.volume_shape, dtype=np.float32)
        
       
        num_motors = np.random.randint(1, 4)
        coordinates = []
        
        for _ in range(num_motors):
            
            x, y, z = np.random.randint(10, 54, size=3)
            coordinates.append([x, y, z])
            
            
            xx, yy, zz = np.mgrid[:64, :64, :64]
            blob = np.exp(-((xx-x)**2 + (yy-y)**2 + (zz-z)**2) / 8.0)
            volume += blob
        
      
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        
      
        volume = torch.from_numpy(volume).unsqueeze(0)  # Add channel dim
        
        
        coordinates = torch.FloatTensor(coordinates[0]) if coordinates else torch.zeros(3)
        
        return volume, coordinates

dataset = SyntheticTomogramDataset()
sample_vol, sample_coords = dataset[0]

print(f"\nSample volume shape: {sample_vol.shape}")
print(f"Sample motor coordinates: {sample_coords}")



fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (ax, slice_idx) in enumerate(zip(axes, [32, 32, 32])):
    if i == 0:
        ax.imshow(sample_vol[0, slice_idx, :, :], cmap='gray')
    elif i == 1:
        ax.imshow(sample_vol[0, :, slice_idx, :], cmap='gray')
    else:
        ax.imshow(sample_vol[0, :, :, slice_idx], cmap='gray')
    ax.set_title(f"Slice {slice_idx}")
plt.show()




class MotorLocalizer(nn.Module):
    def __init__(self):
        super(MotorLocalizer, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*8*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  
        )
        
    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)

model = MotorLocalizer()
print(model)





train_data, val_data = train_test_split(dataset, test_size=0.2)


train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
val_loader = DataLoader(val_data, batch_size=4)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)




def train_model(num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for volumes, coords in train_loader:
            optimizer.zero_grad()
            
            outputs = model(volumes)
            loss = criterion(outputs, coords)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for volumes, coords in val_loader:
                outputs = model(volumes)
                val_loss += criterion(outputs, coords).item()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f}")


def predict_motor_location(volume):
    model.eval()
    with torch.no_grad():
        volume = volume.unsqueeze(0)  
        pred_coords = model(volume)
    return pred_coords.squeeze(0)

test_vol, true_coords = dataset[5]
pred_coords = predict_motor_location(test_vol)

print(f"\nTrue coordinates: {true_coords}")
print(f"Predicted coordinates: {pred_coords}")
print(f"Euclidean distance error: {torch.norm(pred_coords - true_coords):.2f} voxels")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from skimage import io, exposure
import seaborn as sns


df = pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')


IMAGE_DIR = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train'
os.makedirs('visualizations', exist_ok=True)


def plot_metadata_distributions(df):
    """Visualize distributions of key metadata features"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(df['Number of motors'], bins=20, kde=False)
    plt.title('Distribution of Motors per Tomogram')
    
    plt.subplot(2, 2, 2)
    sns.histplot(df['Voxel spacing'], bins=20, kde=False)
    plt.title('Voxel Spacing Distribution')
    
    plt.subplot(2, 2, 3)
    sizes = df['Array shape (axis 0)'] * df['Array shape (axis 1)'] * df['Array shape (axis 2)']
    sns.histplot(sizes, bins=20, kde=False)
    plt.title('Tomogram Volume Sizes (voxels)')
    
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='Array shape (axis 0)', y='Voxel spacing', data=df)
    plt.title('Resolution vs Volume Size')
    
    plt.tight_layout()
    plt.savefig('visualizations/metadata_distributions.png')
    plt.show()

plot_metadata_distributions(df)


def visualize_tomogram(tomo_id, df, slice_frac=0.5):
    """Visualize a tomogram with motor locations marked"""
    # Get tomogram info
    tomo_info = df[df['tomo_id'] == tomo_id].iloc[0]
    motor_locs = df[df['tomo_id'] == tomo_id][['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values
    
    try:
        print(f"Warning: Using synthetic data - replace with actual tomogram loading")
        volume = np.random.rand(tomo_info['Array shape (axis 0)'], 
                              tomo_info['Array shape (axis 1)'], 
                              tomo_info['Array shape (axis 2)'])
    except Exception as e:
        print(f"Error loading {tomo_id}: {e}")
        return
    
    
    slice_z = int(volume.shape[0] * slice_frac)
    slice_y = int(volume.shape[1] * slice_frac)
    slice_x = int(volume.shape[2] * slice_frac)
    
   
    fig = plt.figure(figsize=(15, 5))
    
   
    ax1 = fig.add_subplot(131)
    ax1.imshow(volume[slice_z,:,:], cmap='gray')
    
    for motor in motor_locs:
        if motor[0] >= 0: 
            ax1.plot(motor[2], motor[1], 'r+', markersize=10)
    ax1.set_title(f'XY Slice (Z={slice_z})')
    
   
    ax2 = fig.add_subplot(132)
    ax2.imshow(volume[:,slice_y,:], cmap='gray')
    for motor in motor_locs:
        if motor[0] >= 0:
            ax2.plot(motor[2], motor[0], 'r+', markersize=10)
    ax2.set_title(f'XZ Slice (Y={slice_y})')
    
   
    ax3 = fig.add_subplot(133)
    ax3.imshow(volume[:,:,slice_x], cmap='gray')
    for motor in motor_locs:
        if motor[0] >= 0:
            ax3.plot(motor[1], motor[0], 'r+', markersize=10)
    ax3.set_title(f'YZ Slice (X={slice_x})')
    
    plt.suptitle(f"Tomogram {tomo_id}\nVoxel size: {tomo_info['Voxel spacing']}nm | Motors: {len(motor_locs)}")
    plt.tight_layout()
    plt.savefig(f'visualizations/{tomo_id}_slices.png')
    plt.show()

sample_tomos = df['tomo_id'].unique()[:3]  
for tomo in sample_tomos:
    visualize_tomogram(tomo, df)


def plot_3d_motor_locations(tomo_id, df):
    """Create 3D plot of motor locations within a tomogram"""
    motor_locs = df[df['tomo_id'] == tomo_id][['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values
    tomo_info = df[df['tomo_id'] == tomo_id].iloc[0]
    
    
    valid_locs = motor_locs[(motor_locs >= 0).all(axis=1)]
    
    if len(valid_locs) == 0:
        print(f"No valid motor coordinates for {tomo_id}")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    
    ax.scatter(valid_locs[:,0], valid_locs[:,1], valid_locs[:,2], 
               c='r', s=50, marker='o', label='Flagellar Motors')
    
   
    ax.set_xlabel('Axis 0')
    ax.set_ylabel('Axis 1')
    ax.set_zlabel('Axis 2')
    ax.set_title(f'3D Motor Locations in {tomo_id}\n{len(valid_locs)} motors | Voxel size: {tomo_info["Voxel spacing"]}nm')
    
   
    ax.set_xlim(0, tomo_info['Array shape (axis 0)'])
    ax.set_ylim(0, tomo_info['Array shape (axis 1)'])
    ax.set_zlim(0, tomo_info['Array shape (axis 2)'])
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'visualizations/{tomo_id}_3d.png')
    plt.show()
    
for tomo in sample_tomos:
    plot_3d_motor_locations(tomo, df)


def analyze_motor_density(df):
    """Analyze spatial distribution patterns of motors"""
    plt.figure(figsize=(12, 8))
    
    all_motors = []
    for tomo_id in df['tomo_id'].unique():
        motor_locs = df[df['tomo_id'] == tomo_id][['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values
        valid_locs = motor_locs[(motor_locs >= 0).all(axis=1)]
        if len(valid_locs) > 0:
            all_motors.extend(valid_locs)
    
    if not all_motors:
        print("No valid motor coordinates found")
        return
    
    all_motors = np.array(all_motors)
    
    # Create 2D histograms for each axis pair
    plt.subplot(2, 2, 1)
    plt.hist2d(all_motors[:,0], all_motors[:,1], bins=50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Axis 0')
    plt.ylabel('Axis 1')
    plt.title('Motor Density (Axis 0 vs 1)')
    
    plt.subplot(2, 2, 2)
    plt.hist2d(all_motors[:,0], all_motors[:,2], bins=50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Axis 0')
    plt.ylabel('Axis 2')
    plt.title('Motor Density (Axis 0 vs 2)')
    
    plt.subplot(2, 2, 3)
    plt.hist2d(all_motors[:,1], all_motors[:,2], bins=50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Axis 1')
    plt.ylabel('Axis 2')
    plt.title('Motor Density (Axis 1 vs 2)')
    
    plt.tight_layout()
    plt.savefig('visualizations/motor_density_analysis.png')
    plt.show()

analyze_motor_density(df)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols



df = pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')

df['volume_size'] = df['Array shape (axis 0)'] * df['Array shape (axis 1)'] * df['Array shape (axis 2)']
df['has_motors'] = df['Number of motors'] > 0
valid_motors = df[df['Motor axis 0'] >= 0]  # Filter valid motor coordinates


def compute_descriptive_stats(df):
    """Calculate comprehensive descriptive statistics"""
    print("=== Global Descriptive Statistics ===")
    print(f"Total tomograms: {len(df)}")
    print(f"Tomograms with motors: {df['has_motors'].sum()} ({df['has_motors'].mean()*100:.1f}%)")
    
    print("\n=== Motor Characteristics ===")
    motor_stats = valid_motors[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].describe()
    print(motor_stats)
    
    print("\n=== Tomogram Characteristics ===")
    print(df[['volume_size', 'Voxel spacing', 'Number of motors']].describe())
    
    
    plt.figure(figsize=(12, 5))
    sns.boxplot(x='Number of motors', y='Voxel spacing', data=df)
    plt.title('Voxel Spacing Distribution by Motor Count')
    plt.tight_layout()
    plt.show()

compute_descriptive_stats(df)


def analyze_spatial_distribution(valid_motors):
    """Analyze spatial distribution patterns of motors"""
    print("\n=== Spatial Distribution Analysis ===")
    
    
    for axis in ['Motor axis 0', 'Motor axis 1', 'Motor axis 2']:
        valid_motors[f'{axis}_norm'] = valid_motors[axis] / valid_motors[f'Array shape (axis {axis[-1]})']
    
    
    print("\nUniform Distribution Tests (per axis):")
    for axis in ['0', '1', '2']:
        stat, p = stats.kstest(valid_motors[f'Motor axis {axis}_norm'], 'uniform')
        print(f"Axis {axis}: KS stat = {stat:.3f}, p = {p:.4f}")
    

    try:
        from libpysal.weights import DistanceBand
        from esda.moran import Moran
        
        # Create spatial weights matrix (using first 500 motors for demo)
        sample = valid_motors.iloc[:500]
        coords = sample[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values
        w = DistanceBand(coords, threshold=50, binary=False)
        moran = Moran(sample['Motor axis 0'], w)
        print(f"\nSpatial Autocorrelation (Moran's I): {moran.I:.3f}, p = {moran.p_norm:.4f}")
    except ImportError:
        print("\nSpatial autocorrelation analysis requires libpysal and esda packages")

analyze_spatial_distribution(valid_motors)


def motor_location_regression(valid_motors):
    """Analyze spatial relationships between motor coordinates"""
    print("\n=== Motor Coordinate Regression ===")
    
    
    for axis in ['0', '1', '2']:
        valid_motors[f'motor_{axis}_z'] = stats.zscore(valid_motors[f'Motor axis {axis}'])
    
    
    model = ols('motor_2_z ~ motor_0_z + motor_1_z', data=valid_motors).fit()
    print(model.summary())
    
    
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    sample = valid_motors.sample(500) if len(valid_motors) > 500 else valid_motors
    ax1.scatter(sample['motor_0_z'], sample['motor_1_z'], sample['motor_2_z'], alpha=0.5)
    ax1.set_xlabel('X (Axis 0)')
    ax1.set_ylabel('Y (Axis 1)')
    ax1.set_zlabel('Z (Axis 2)')
    ax1.set_title('Standardized Motor Coordinates')
    
    ax2 = fig.add_subplot(122)
    sm.graphics.plot_partregress_grid(model, fig=fig, exog_idx=['motor_0_z', 'motor_1_z'])
    plt.tight_layout()
    plt.show()

motor_location_regression(valid_motors)

