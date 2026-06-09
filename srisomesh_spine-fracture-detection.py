!pip install pyvista


"""
Enhanced 3D Cervical Spine Visualization Suite
Combines multiple visualization techniques with robust processing
Optimized for RSNA Kaggle competition data
"""

import numpy as np
import pydicom
import os
import gc
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, morphology, exposure
from scipy import ndimage as ndi

# Try optional imports
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available - interactive 3D will be skipped")

# ==================== PARAMETERS ====================
BASE_PATH = '/kaggle/input/rsna-2022-cervical-spine-fracture-detection'
TRAIN_IMAGES_PATH = os.path.join(BASE_PATH, 'train_images')

# Processing parameters
PATIENT_INDEX = 0
BONE_THRESHOLD = 200          # HU threshold for bone
SOFT_TISSUE_THRESHOLD = -300  # HU threshold for soft tissue
RESAMPLE = True               # Resample to isotropic voxels
TARGET_SPACING = [1, 1, 1]    # Target voxel spacing in mm
MAX_DIM = 256                 # Max dimension for marching cubes
LARGEST_COMPONENT = True      # Keep only largest bone structure
APPLY_MORPHOLOGY = True       # Apply morphological operations

# Visualization parameters
SAVE_OUTPUTS = True
OUTPUT_DIR = '/kaggle/working'
VERBOSE = True
# ===================================================

def log(msg):
    """Print with prefix if verbose"""
    if VERBOSE:
        print(f"[3D-VIZ] {msg}")

def load_dicom_series(patient_id):
    """Load and sort DICOM slices for a patient"""
    patient_path = os.path.join(TRAIN_IMAGES_PATH, patient_id)
    
    if not os.path.exists(patient_path):
        raise ValueError(f"Patient folder not found: {patient_id}")
    
    dicom_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.dcm')])
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found for patient: {patient_id}")
    
    log(f"Loading {len(dicom_files)} DICOM slices...")
    
    # Load slices
    slices = []
    for dicom_file in dicom_files:
        filepath = os.path.join(patient_path, dicom_file)
        ds = pydicom.dcmread(filepath)
        slices.append(ds)
    
    # Robust multi-method sorting
    try:
        slices.sort(key=lambda x: int(x.InstanceNumber))
    except Exception:
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except Exception:
            slices.sort(key=lambda x: float(x.SliceLocation))
    
    return slices

def get_pixels_hu(slices):
    """Convert DICOM pixel data to Hounsfield Units"""
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    
    # Set outside-of-scan pixels to 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units
    for slice_number in range(len(slices)):
        intercept = float(slices[slice_number].RescaleIntercept)
        slope = float(slices[slice_number].RescaleSlope)
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def get_spacing(slices):
    """Extract voxel spacing from DICOM headers"""
    try:
        px = slices[0].PixelSpacing
        
        # Calculate z-spacing from consecutive slices
        if len(slices) > 1:
            positions = [float(ds.ImagePositionPatient[2]) for ds in slices[:5]]
            z_diffs = [abs(positions[i+1] - positions[i]) for i in range(len(positions)-1)]
            z_spacing = np.median(z_diffs)
        else:
            z_spacing = float(slices[0].SliceThickness)
        
        spacing = [z_spacing, float(px[0]), float(px[1])]
        return spacing
    except Exception as e:
        log(f"Warning: Could not extract spacing ({e}), using defaults")
        return [1.0, 1.0, 1.0]

def resample_volume(image, original_spacing, new_spacing=[1, 1, 1]):
    """Resample volume to isotropic voxels"""
    spacing = np.array(original_spacing, dtype=np.float32)
    new_spacing = np.array(new_spacing, dtype=np.float32)
    
    resize_factor = spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor).astype(int)
    
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    log(f"Resampling from {image.shape} to {tuple(new_shape)}...")
    image = ndi.zoom(image, real_resize_factor, order=1)
    
    return image, new_spacing.tolist()

def apply_bone_window(volume):
    """Apply bone window settings (W:1800, L:400)"""
    lower = 400 - 1800/2
    upper = 400 + 1800/2
    windowed = np.clip(volume, lower, upper)
    windowed = (windowed - lower) / (upper - lower)
    return windowed

def apply_soft_tissue_window(volume):
    """Apply soft tissue window settings (W:400, L:40)"""
    lower = 40 - 400/2
    upper = 40 + 400/2
    windowed = np.clip(volume, lower, upper)
    windowed = (windowed - lower) / (upper - lower)
    return windowed

def create_bone_mask(hu_volume, threshold, apply_morphology=True):
    """Create binary bone mask with optional morphological cleanup"""
    mask = hu_volume > threshold
    
    if apply_morphology:
        # Aggressive morphological operations
        mask = ndi.binary_closing(mask, structure=np.ones((3,3,3)), iterations=2)
        mask = ndi.binary_opening(mask, structure=np.ones((3,3,3)), iterations=1)
        
        # Remove small objects
        mask = morphology.remove_small_objects(mask, min_size=1000)
    
    return mask

def extract_largest_component(mask):
    """Keep only the largest connected component"""
    labeled = measure.label(mask, connectivity=2)
    if labeled.max() == 0:
        return mask
    
    props = measure.regionprops(labeled)
    largest = max(props, key=lambda x: x.area)
    
    log(f"Kept largest component: {largest.area:,} voxels out of {mask.sum():,}")
    return labeled == largest.label

def downsample_mask(mask, target_max_dim):
    """Downsample mask to target dimension"""
    max_dim = max(mask.shape)
    if max_dim <= target_max_dim:
        return mask, 1
    
    factor = int(np.ceil(max_dim / target_max_dim))
    downsampled = mask[::factor, ::factor, ::factor]
    
    log(f"Downsampled from {mask.shape} to {downsampled.shape} (factor={factor})")
    return downsampled, factor

# ==================== VISUALIZATION FUNCTIONS ====================

def render_mip_views(volume, title="Maximum Intensity Projection", save_path=None):
    """Create Maximum Intensity Projection views"""
    log("Creating MIP views...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Axial MIP (top-down)
    mip_axial = np.max(volume, axis=0)
    axes[0, 0].imshow(mip_axial, cmap='gray')
    axes[0, 0].set_title('Axial MIP (Top View)', fontsize=12)
    axes[0, 0].axis('off')
    
    # Sagittal MIP (side)
    mip_sagittal = np.max(volume, axis=2)
    axes[0, 1].imshow(mip_sagittal, cmap='gray')
    axes[0, 1].set_title('Sagittal MIP (Side View)', fontsize=12)
    axes[0, 1].axis('off')
    
    # Coronal MIP (front)
    mip_coronal = np.max(volume, axis=1)
    axes[1, 0].imshow(mip_coronal, cmap='gray')
    axes[1, 0].set_title('Coronal MIP (Front View)', fontsize=12)
    axes[1, 0].axis('off')
    
    # Volume info
    info_text = f'Volume Shape: {volume.shape}\n'
    info_text += f'HU Range: [{volume.min():.0f}, {volume.max():.0f}]\n'
    info_text += f'Mean HU: {volume.mean():.0f}\n'
    info_text += f'Std HU: {volume.std():.0f}'
    axes[1, 1].text(0.5, 0.5, info_text, ha='center', va='center', 
                    fontsize=12, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path and SAVE_OUTPUTS:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log(f"Saved MIP views: {save_path}")
    
    plt.show()

def render_slice_montage(volume, num_slices=16, window_func=None, save_path=None):
    """Display a montage of slices"""
    log("Creating slice montage...")
    
    step = max(1, volume.shape[0] // num_slices)
    slice_indices = range(0, volume.shape[0], step)[:num_slices]
    
    rows = int(np.ceil(np.sqrt(num_slices)))
    cols = int(np.ceil(num_slices / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    axes = axes.flatten()
    
    # Apply windowing if provided
    display_volume = window_func(volume) if window_func else volume
    
    for idx, slice_idx in enumerate(slice_indices):
        axes[idx].imshow(display_volume[slice_idx], cmap='gray')
        axes[idx].set_title(f'Slice {slice_idx}/{volume.shape[0]}', fontsize=10)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(slice_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('CT Slices Montage', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path and SAVE_OUTPUTS:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log(f"Saved slice montage: {save_path}")
    
    plt.show()

def render_multiplanar(volume, title="Multiplanar Reconstruction", save_path=None):
    """Display orthogonal slices at volume center"""
    log("Creating multiplanar reconstruction...")
    
    z, y, x = volume.shape
    mid_z, mid_y, mid_x = z//2, y//2, x//2
    
    fig = plt.figure(figsize=(16, 12))
    
    # Axial slice
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(volume[mid_z, :, :], cmap='bone')
    ax1.set_title(f'Axial (Z={mid_z})', fontsize=12)
    ax1.axhline(mid_y, color='r', linestyle='--', alpha=0.5)
    ax1.axvline(mid_x, color='g', linestyle='--', alpha=0.5)
    ax1.axis('off')
    
    # Sagittal slice
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(volume[:, :, mid_x].T, cmap='bone', aspect='auto')
    ax2.set_title(f'Sagittal (X={mid_x})', fontsize=12)
    ax2.axhline(mid_y, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(mid_z, color='b', linestyle='--', alpha=0.5)
    ax2.axis('off')
    
    # Coronal slice
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(volume[:, mid_y, :].T, cmap='bone', aspect='auto')
    ax3.set_title(f'Coronal (Y={mid_y})', fontsize=12)
    ax3.axhline(mid_x, color='g', linestyle='--', alpha=0.5)
    ax3.axvline(mid_z, color='b', linestyle='--', alpha=0.5)
    ax3.axis('off')
    
    # Bone window views
    bone_windowed = apply_bone_window(volume)
    
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(bone_windowed[mid_z, :, :], cmap='gray')
    ax4.set_title('Axial (Bone Window)', fontsize=12)
    ax4.axis('off')
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(bone_windowed[:, :, mid_x].T, cmap='gray', aspect='auto')
    ax5.set_title('Sagittal (Bone Window)', fontsize=12)
    ax5.axis('off')
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(bone_windowed[:, mid_y, :].T, cmap='gray', aspect='auto')
    ax6.set_title('Coronal (Bone Window)', fontsize=12)
    ax6.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path and SAVE_OUTPUTS:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log(f"Saved multiplanar view: {save_path}")
    
    plt.show()

def render_3d_surface_matplotlib(verts, faces, title="3D Surface", save_path=None):
    """Create 3D surface rendering using matplotlib"""
    log("Creating 3D surface with matplotlib...")
    
    fig = plt.figure(figsize=(14, 12))
    
    # Main view
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Subsample faces if too many
    max_faces = 30000
    if len(faces) > max_faces:
        indices = np.random.choice(len(faces), max_faces, replace=False)
        faces_plot = faces[indices]
    else:
        faces_plot = faces
    
    mesh = Poly3DCollection(verts[faces_plot], alpha=0.8, linewidths=0.1, edgecolors='darkgray')
    mesh.set_facecolor([0.8, 0.9, 1.0])
    ax1.add_collection3d(mesh)
    
    ax1.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax1.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax1.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax1.set_xlabel('X (mm)', fontsize=10)
    ax1.set_ylabel('Y (mm)', fontsize=10)
    ax1.set_zlabel('Z (mm)', fontsize=10)
    ax1.set_title('3D View', fontsize=12)
    ax1.view_init(elev=20, azim=45)
    
    # Side view
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    mesh2 = Poly3DCollection(verts[faces_plot], alpha=0.8, linewidths=0.1, edgecolors='darkgray')
    mesh2.set_facecolor([0.8, 0.9, 1.0])
    ax2.add_collection3d(mesh2)
    ax2.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax2.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax2.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax2.set_xlabel('X (mm)', fontsize=10)
    ax2.set_ylabel('Y (mm)', fontsize=10)
    ax2.set_zlabel('Z (mm)', fontsize=10)
    ax2.set_title('Side View', fontsize=12)
    ax2.view_init(elev=0, azim=90)
    
    # Top view
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    mesh3 = Poly3DCollection(verts[faces_plot], alpha=0.8, linewidths=0.1, edgecolors='darkgray')
    mesh3.set_facecolor([0.8, 0.9, 1.0])
    ax3.add_collection3d(mesh3)
    ax3.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax3.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax3.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax3.set_xlabel('X (mm)', fontsize=10)
    ax3.set_ylabel('Y (mm)', fontsize=10)
    ax3.set_zlabel('Z (mm)', fontsize=10)
    ax3.set_title('Top View', fontsize=12)
    ax3.view_init(elev=90, azim=0)
    
    # Front view
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    mesh4 = Poly3DCollection(verts[faces_plot], alpha=0.8, linewidths=0.1, edgecolors='darkgray')
    mesh4.set_facecolor([0.8, 0.9, 1.0])
    ax4.add_collection3d(mesh4)
    ax4.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax4.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax4.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax4.set_xlabel('X (mm)', fontsize=10)
    ax4.set_ylabel('Y (mm)', fontsize=10)
    ax4.set_zlabel('Z (mm)', fontsize=10)
    ax4.set_title('Front View', fontsize=12)
    ax4.view_init(elev=0, azim=0)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path and SAVE_OUTPUTS:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log(f"Saved 3D surface: {save_path}")
    
    plt.show()

def render_interactive_3d_plotly(verts, faces, title="Interactive 3D Spine"):
    """Create interactive 3D rendering using Plotly"""
    if not PLOTLY_AVAILABLE:
        log("Plotly not available - skipping interactive 3D")
        return
    
    log("Creating interactive 3D visualization with Plotly...")
    
    x, y, z = verts.T
    i, j, k = faces.T
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=0.7,
            color='lightblue',
            flatshading=False,
            lighting=dict(
                ambient=0.5,
                diffuse=0.8,
                specular=0.5,
                roughness=0.5,
                fresnel=0.2
            ),
            lightposition=dict(
                x=100,
                y=200,
                z=300
            )
        )
    ])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800
    )
    
    if SAVE_OUTPUTS:
        html_path = os.path.join(OUTPUT_DIR, 'interactive_3d_spine.html')
        fig.write_html(html_path)
        log(f"Saved interactive 3D: {html_path}")
    
    fig.show()

def save_stl(verts, faces, filename):
    """Save mesh to STL format"""
    try:
        with open(filename, 'w') as f:
            f.write('solid mesh\n')
            for face in faces:
                v0, v1, v2 = verts[face]
                normal = np.cross(v1 - v0, v2 - v0)
                normal = normal / (np.linalg.norm(normal) + 1e-8)
                
                f.write(f'  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n')
                f.write('    outer loop\n')
                f.write(f'      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n')
                f.write(f'      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n')
                f.write(f'      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n')
                f.write('    endloop\n')
                f.write('  endfacet\n')
            f.write('endsolid mesh\n')
        log(f"Saved STL mesh: {filename}")
        return True
    except Exception as e:
        log(f"Failed to save STL: {e}")
        return False

# ==================== MAIN PIPELINE ====================

def main():
    """Main visualization pipeline"""
    
    log("="*60)
    log("3D SPINE VISUALIZATION SUITE")
    log("="*60)
    
    # 1. List and select patient
    log("\nAvailable patients:")
    patients = sorted(os.listdir(TRAIN_IMAGES_PATH))[:10]
    for i, patient in enumerate(patients[:5]):
        print(f"  {i+1}. {patient}")
    if len(patients) > 5:
        print(f"  ... and {len(patients)-5} more")
    
    patient_id = patients[PATIENT_INDEX]
    log(f"\nSelected patient: {patient_id} (index {PATIENT_INDEX})")
    
    # 2. Load DICOM series
    slices = load_dicom_series(patient_id)
    log(f"Loaded {len(slices)} slices")
    
    # 3. Convert to Hounsfield Units
    volume = get_pixels_hu(slices)
    log(f"Volume shape: {volume.shape}")
    log(f"HU range: [{volume.min()}, {volume.max()}]")
    
    # 4. Get spacing
    original_spacing = get_spacing(slices)
    log(f"Original spacing (Z×Y×X): {original_spacing} mm")
    
    # 5. Optional resampling
    if RESAMPLE:
        volume, new_spacing = resample_volume(volume, original_spacing, TARGET_SPACING)
        spacing = new_spacing
        log(f"Resampled to spacing: {spacing} mm")
    else:
        spacing = original_spacing
    
    # 6. VISUALIZATION 1: MIP Views
    log("\n" + "="*60)
    log("VISUALIZATION 1: Maximum Intensity Projections")
    log("="*60)
    render_mip_views(
        volume, 
        title=f"Patient {patient_id} - MIP Views",
        save_path=os.path.join(OUTPUT_DIR, 'mip_views.png')
    )
    
    # 7. VISUALIZATION 2: Multiplanar Reconstruction
    log("\n" + "="*60)
    log("VISUALIZATION 2: Multiplanar Reconstruction")
    log("="*60)
    render_multiplanar(
        volume,
        title=f"Patient {patient_id} - Multiplanar Views",
        save_path=os.path.join(OUTPUT_DIR, 'multiplanar.png')
    )
    
    # 8. VISUALIZATION 3: Slice Montage
    log("\n" + "="*60)
    log("VISUALIZATION 3: Slice Montage")
    log("="*60)
    render_slice_montage(
        volume, 
        num_slices=16,
        window_func=apply_bone_window,
        save_path=os.path.join(OUTPUT_DIR, 'slice_montage.png')
    )
    
    # 9. Create bone mask
    log("\n" + "="*60)
    log("MESH GENERATION: Creating bone surface")
    log("="*60)
    bone_mask = create_bone_mask(volume, BONE_THRESHOLD, APPLY_MORPHOLOGY)
    log(f"Initial bone voxels: {bone_mask.sum():,}")
    
    # Auto-adjust threshold if needed
    if bone_mask.sum() < 10000:
        log(f"Low bone count, reducing threshold from {BONE_THRESHOLD} to 100")
        bone_mask = create_bone_mask(volume, 100, APPLY_MORPHOLOGY)
        log(f"Adjusted bone voxels: {bone_mask.sum():,}")
    
    # 10. Extract largest component
    if LARGEST_COMPONENT and bone_mask.sum() > 0:
        bone_mask = extract_largest_component(bone_mask)
    
    # 11. Downsample and generate mesh
    bone_mask_small, down_factor = downsample_mask(bone_mask, MAX_DIM)
    final_spacing = [s * down_factor for s in spacing]
    
    log("Running marching cubes algorithm...")
    try:
        verts, faces, normals, vals = measure.marching_cubes(
            bone_mask_small.astype(np.uint8),
            level=0.5,
            spacing=final_spacing,
            step_size=1
        )
        faces = faces.astype(np.int64)
        log(f"Mesh created: {len(verts):,} vertices, {len(faces):,} faces")
        
        # 12. Save STL
        if SAVE_OUTPUTS:
            stl_path = os.path.join(OUTPUT_DIR, 'spine_mesh.stl')
            save_stl(verts, faces, stl_path)
        
        # 13. VISUALIZATION 4: 3D Surface (Matplotlib)
        log("\n" + "="*60)
        log("VISUALIZATION 4: 3D Surface Rendering (Matplotlib)")
        log("="*60)
        render_3d_surface_matplotlib(
            verts, faces,
            title=f"Patient {patient_id} - 3D Bone Surface",
            save_path=os.path.join(OUTPUT_DIR, '3d_surface.png')
        )
        
        # 14. VISUALIZATION 5: Interactive 3D (Plotly)
        log("\n" + "="*60)
        log("VISUALIZATION 5: Interactive 3D (Plotly)")
        log("="*60)
        render_interactive_3d_plotly(verts, faces, f"Patient {patient_id} - Interactive 3D")
        
    except Exception as e:
        log(f"ERROR: Mesh generation failed: {e}")
    
    # Cleanup
    del volume, bone_mask, bone_mask_small
    if 'verts' in locals():
        del verts, faces, normals
    gc.collect()
    
    log("\n" + "="*60)
    log("✓ ALL VISUALIZATIONS COMPLETE!")
    log("="*60)
    if SAVE_OUTPUTS:
        log(f"Output files saved to: {OUTPUT_DIR}")
        log("  - mip_views.png")
        log("  - multiplanar.png")
        log("  - slice_montage.png")
        log("  - 3d_surface.png")
        log("  - spine_mesh.stl")
        if PLOTLY_AVAILABLE:
            log("  - interactive_3d_spine.html")

if __name__ == "__main__":
    main()




