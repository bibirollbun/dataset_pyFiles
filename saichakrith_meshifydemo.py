import numpy as np
import pydicom
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import plotly.graph_objects as go
from scipy import ndimage

# Dataset path
BASE_PATH = '/kaggle/input/rsna-2022-cervical-spine-fracture-detection'
TRAIN_IMAGES_PATH = os.path.join(BASE_PATH, 'train_images')

def load_dicom_series(patient_id):
    """Load all DICOM slices for a patient"""
    patient_path = os.path.join(TRAIN_IMAGES_PATH, patient_id)
    
    if not os.path.exists(patient_path):
        raise ValueError(f"Patient folder not found: {patient_id}")
    
    # Get all DICOM files
    dicom_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.dcm')])
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found for patient: {patient_id}")
    
    print(f"Loading {len(dicom_files)} DICOM slices...")
    
    # Load slices
    slices = []
    for dicom_file in dicom_files:
        filepath = os.path.join(patient_path, dicom_file)
        ds = pydicom.dcmread(filepath)
        slices.append(ds)
    
    # Sort by ImagePositionPatient (Z coordinate)
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    return slices

def get_pixels_hu(slices):
    """Convert DICOM pixel data to Hounsfield Units"""
    # Stack all slices into 3D array
    image = np.stack([s.pixel_array for s in slices])
    
    # Convert to int16
    image = image.astype(np.int16)
    
    # Set outside-of-scan pixels to 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def resample_volume(image, slices, new_spacing=[1, 1, 1]):
    """Resample volume to isotropic voxels"""
    # Get current spacing
    spacing = np.array([
        slices[0].SliceThickness,
        slices[0].PixelSpacing[0],
        slices[0].PixelSpacing[1]
    ], dtype=np.float32)
    
    resize_factor = spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor)
    
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = ndimage.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def render_3d_volume_mip(volume, title="Maximum Intensity Projection"):
    """Create Maximum Intensity Projection views"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)
    
    # Axial view (top-down)
    mip_axial = np.max(volume, axis=0)
    axes[0, 0].imshow(mip_axial, cmap='gray')
    axes[0, 0].set_title('Axial MIP (Top View)')
    axes[0, 0].axis('off')
    
    # Sagittal view (side)
    mip_sagittal = np.max(volume, axis=2)
    axes[0, 1].imshow(mip_sagittal, cmap='gray')
    axes[0, 1].set_title('Sagittal MIP (Side View)')
    axes[0, 1].axis('off')
    
    # Coronal view (front)
    mip_coronal = np.max(volume, axis=1)
    axes[1, 0].imshow(mip_coronal, cmap='gray')
    axes[1, 0].set_title('Coronal MIP (Front View)')
    axes[1, 0].axis('off')
    
    # 3D visualization info
    axes[1, 1].text(0.5, 0.5, f'Volume Shape: {volume.shape}\nHU Range: [{volume.min()}, {volume.max()}]',
                    ha='center', va='center', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def render_3d_surface(volume, threshold=-300, title="3D Surface Rendering"):
    """Create 3D surface rendering using marching cubes"""
    print(f"Creating 3D surface with threshold: {threshold} HU...")
    
    # Use marching cubes to obtain the surface mesh
    verts, faces, normals, values = measure.marching_cubes(volume, threshold)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create mesh
    mesh = Poly3DCollection(verts[faces], alpha=0.7, linewidth=0)
    face_color = [0.8, 0.8, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    
    # Set plot limits
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

def render_interactive_3d(volume, threshold=-300):
    """Create interactive 3D rendering using Plotly"""
    print(f"Creating interactive 3D visualization with threshold: {threshold} HU...")
    
    # Use marching cubes
    verts, faces, normals, values = measure.marching_cubes(volume, threshold)
    
    # Create mesh
    x, y, z = verts.T
    i, j, k = faces.T
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=0.5,
            color='lightblue',
            flatshading=True
        )
    ])
    
    fig.update_layout(
        title='Interactive 3D Spine CT Visualization',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=900,
        height=700
    )
    
    fig.show()

def render_slices_montage(volume, num_slices=16):
    """Display a montage of slices"""
    step = max(1, volume.shape[0] // num_slices)
    slice_indices = range(0, volume.shape[0], step)[:num_slices]
    
    rows = int(np.ceil(np.sqrt(num_slices)))
    cols = int(np.ceil(num_slices / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, slice_idx in enumerate(slice_indices):
        axes[idx].imshow(volume[slice_idx], cmap='gray')
        axes[idx].set_title(f'Slice {slice_idx}')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(slice_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('CT Slices Montage', fontsize=16)
    plt.tight_layout()
    plt.show()

def apply_bone_window(volume):
    """Apply bone window settings (W:1800, L:400)"""
    lower = 400 - 1800/2
    upper = 400 + 1800/2
    windowed = np.clip(volume, lower, upper)
    windowed = (windowed - lower) / (upper - lower) * 255
    return windowed.astype(np.uint8)

def main():
    """Main function to run 3D volumetric rendering"""
    
    # Example patient IDs from the dataset
    # You can find these by listing the train_images directory
    print("Available patients:")
    patients = sorted(os.listdir(TRAIN_IMAGES_PATH))[:5]  # Show first 5
    for i, patient in enumerate(patients):
        print(f"{i+1}. {patient}")
    
    # Select a patient (change this to any patient ID in your dataset)
    patient_id = patients[0]  # Use first patient
    print(f"\nProcessing patient: {patient_id}")
    
    # Load DICOM series
    slices = load_dicom_series(patient_id)
    print(f"Loaded {len(slices)} slices")
    
    # Get pixel data in Hounsfield Units
    volume = get_pixels_hu(slices)
    print(f"Volume shape: {volume.shape}")
    print(f"HU range: [{volume.min()}, {volume.max()}]")
    
    # Resample to isotropic voxels (optional, for better 3D rendering)
    print("\nResampling volume to isotropic voxels...")
    volume_resampled, new_spacing = resample_volume(volume, slices, [1, 1, 1])
    print(f"Resampled volume shape: {volume_resampled.shape}")
    
    # Apply bone window
    volume_windowed = apply_bone_window(volume_resampled)
    
    # 1. Maximum Intensity Projection
    print("\n1. Creating Maximum Intensity Projections...")
    render_3d_volume_mip(volume_resampled, "Spine CT - MIP Views")
    
    # 2. Slice montage
    print("\n2. Creating slice montage...")
    render_slices_montage(volume_resampled, num_slices=16)
    
    # 3. 3D Surface rendering (bones)
    print("\n3. Creating 3D surface rendering...")
    # Threshold for bone: typically > 200 HU
    render_3d_surface(volume_resampled, threshold=200, title="3D Bone Rendering")
    
    # 4. Interactive 3D rendering
    print("\n4. Creating interactive 3D visualization...")
    render_interactive_3d(volume_resampled, threshold=200)
    
    print("\n✓ All visualizations complete!")

if __name__ == "__main__":
    main()

