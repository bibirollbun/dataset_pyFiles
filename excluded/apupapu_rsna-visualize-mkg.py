import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# reading the processed CSV file
df = pd.read_csv('/kaggle/input/rsna-dataset/RSNA - Sheet1.csv')

# DICOM files path
base_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'


def view_dicom_images(sop_ids):
    """
    Load and display DICOM images
    """
    if not isinstance(sop_ids, list):
        sop_ids = [sop_ids]
    
    if len(sop_ids) == 0:
        print("No images found to display.")
        return
    
    max_images = min(9, len(sop_ids))
    rows = int(np.ceil(max_images / 3))
    cols = min(3, max_images)
    
    fig = plt.figure(figsize=(15, 5 * rows))
    
    for i, sop_id in enumerate(sop_ids[:max_images]):
       
        found = False
        series_id = ""
        file_path = ""
        
        for series_folder in os.listdir(base_dir):
            series_path = os.path.join(base_dir, series_folder)
            if os.path.isdir(series_path):
               
                for file_name in os.listdir(series_path):
                    if file_name.endswith('.dcm') and sop_id in file_name:
                        file_path = os.path.join(series_path, file_name)
                        series_id = series_folder
                        found = True
                        break
            if found:
                break
        
        plt.subplot(rows, cols, i+1)
        
        if found:
            try:
                ds = pydicom.dcmread(file_path)
                img = ds.pixel_array
                if len(img.shape) > 2 and img.shape[2] > 1:
                    # adjust for different size
                    img = np.mean(img, axis=2).astype(np.float32)
                if img.shape[0] < 10 or img.shape[1] < 10:
                    raise ValueError(f"Corrupted image size: {img.shape}")
                
                # check aspect ratio
                aspect_ratio = img.shape[0] / img.shape[1]
                if aspect_ratio < 0.1 or aspect_ratio > 10:
                    raise ValueError(f"Corrupted image ratio: {aspect_ratio:.2f}")
                
                # apply frame/contrast
                if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                    try:
                        img = apply_voi_lut(img, ds)
                    except Exception:
                        # if VOI LUT cannot be applied, adjust contrast directly
                        img = np.clip(img, 0, np.percentile(img, 99.5))
                
                if img.max() > 0:
                    img = img / img.max() * 255
                
                # display image
                plt.imshow(img, cmap='gray')
                plt.title(f"SOPInstanceUID: {sop_id[-8:]}", fontsize=10)
                
                # add SeriesInstanceUID below the image in red
                plt.xlabel(f"SeriesInstanceUID: {series_id[-7:]}", fontsize=8, color='red')
                
            except Exception as e:
                error_msg = str(e)
                # attempt to repair DICOM array
                try:
                    # get correct dimensions from metadata
                    if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                        rows_dicom = ds.Rows
                        cols_dicom = ds.Columns
                        
                        # some DICOM files are not flat, could be 3D or other formats
                        # try reshaping
                        try:
                            # flatten first
                            flat_pixels = ds.pixel_array.flatten()
                            
                            # reshape into correct dimensions
                            if len(flat_pixels) >= rows_dicom * cols_dicom:
                                reshaped_img = flat_pixels[:rows_dicom * cols_dicom].reshape(rows_dicom, cols_dicom)
                                
                                # normalize
                                if reshaped_img.max() > 0:
                                    reshaped_img = reshaped_img / reshaped_img.max() * 255
                                
                                # display repaired image
                                plt.imshow(reshaped_img, cmap='gray')
                                plt.title(f"SOPInstanceUID: {sop_id[-8:]} (Repaired)", fontsize=10)
                                plt.xlabel(f"SeriesInstanceUID: {series_id[-7:]}", fontsize=8, color='red')
                            else:
                                # alternative approach: transpose
                                if hasattr(ds, 'pixel_array'):
                                    orig_shape = ds.pixel_array.shape
                                    
                                    transposed = np.transpose(ds.pixel_array)
                                    
                                    if transposed.max() > 0:
                                        transposed = transposed / transposed.max() * 255
                                    
                                    plt.imshow(transposed, cmap='gray')
                                    plt.title(f"SOPInstanceUID: {sop_id[-8:]} (Transposed)", fontsize=10)
                                    plt.xlabel(f"SeriesInstanceUID: {series_id[-7:]}", fontsize=8, color='red')
                                else:
                                    plt.text(0.5, 0.5, f"No pixel data", ha='center', va='center', color='red')
                        except Exception as reshape_error:
                            plt.text(0.5, 0.5, f"Shape error: {str(reshape_error)[:30]}...", ha='center', va='center', color='red')
                    else:
                        plt.text(0.5, 0.5, f"Missing metadata", ha='center', va='center', color='red')
                except Exception as meta_error:
                    plt.text(0.5, 0.5, f"Error: {str(meta_error)[:30]}...", ha='center', va='center', color='red')
        else:
            plt.text(0.5, 0.5, "File not found", ha='center', va='center')
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# filter
cta_aneurysm = df[(df['Aneurysm Present'] == 1) & (df['Modality'] == 'CTA')]

# SOPInstanceUID 
cta_sop_ids = []

for _, row in cta_aneurysm.iterrows():
    if pd.notna(row['SOPInstanceUID']):
        # Could be multiple UIDs separated by commas
        ids = str(row['SOPInstanceUID']).split(',')
        cta_sop_ids.extend([id.strip() for id in ids])

print(f"In CTA modality {len(cta_aneurysm)} total for the patient {len(cta_sop_ids)} dcm find:")
for id in cta_sop_ids:
    print(id)

# show DICOM files 
view_dicom_images(cta_sop_ids)


# MRA AND Aneurysm = 1 
mra_aneurysm = df[(df['Aneurysm Present'] == 1) & (df['Modality'] == 'MRA')]

# SOPInstanceUID 
mra_sop_ids = []

for _, row in mra_aneurysm.iterrows():
    if pd.notna(row['SOPInstanceUID']):
        # Could be multiple UIDs separated by commas 
        ids = str(row['SOPInstanceUID']).split(',')
        mra_sop_ids.extend([id.strip() for id in ids])

print(f"In MRA modality {len(mra_aneurysm)} total for the patient {len(mra_sop_ids)} dcm find.:")
for id in mra_sop_ids:
    print(id)

# show DICOM files 
view_dicom_images(mra_sop_ids)


# MRI T1post AND Aneurysm = 1 
mri_t1_aneurysm = df[(df['Aneurysm Present'] == 1) & (df['Modality'] == 'MRI T1post')]

# SOPInstanceUID 
mri_t1_sop_ids = []

for _, row in mri_t1_aneurysm.iterrows():
    if pd.notna(row['SOPInstanceUID']):
        # Could be multiple UIDs separated by commas
        ids = str(row['SOPInstanceUID']).split(',')
        mri_t1_sop_ids.extend([id.strip() for id in ids])

print(f"MRI T1post modalitesinde {len(mri_t1_aneurysm)} total for the patient {len(mri_t1_sop_ids)} dcm findu:")
for id in mri_t1_sop_ids:
    print(id)

# show DICOM files 
view_dicom_images(mri_t1_sop_ids)


# MRI T2 AND Aneurysm = 1 
mri_t2_aneurysm = df[(df['Aneurysm Present'] == 1) & (df['Modality'] == 'MRI T2')]

# SOPInstanceUID değerlerini al
mri_t2_sop_ids = []

for _, row in mri_t2_aneurysm.iterrows():
    if pd.notna(row['SOPInstanceUID']):
        # Virgülle ayrılmış birden fazla UID olabilir
        ids = str(row['SOPInstanceUID']).split(',')
        mri_t2_sop_ids.extend([id.strip() for id in ids])

print(f"In MRI T2 modality {len(mri_t2_aneurysm)} total for the patient {len(mri_t2_sop_ids)} dcm find:")
for id in mri_t2_sop_ids:
    print(id)

# show DICOM files 
view_dicom_images(mri_t2_sop_ids)


# Function to display DICOM images with aneurysm locations marked
def view_dicom_images_with_markers(sop_ids):
    """
    Load and display DICOM images, marking aneurysm locations as red dots.
    """
    # Load localizer data
    localizers_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv')
    
    # Convert coordinate strings to dict
    def parse_coordinates(coord_str):
        try:
            # Convert string-formatted coordinates to Python dict
            if isinstance(coord_str, str):
                # Convert {'x': 123.45, 'y': 678.90} string to dict
                # Using eval is generally not recommended, but here we use it for controlled data
                return eval(coord_str)
            return None
        except:
            return None
    
    # Process coordinates
    localizers_df['coords_dict'] = localizers_df['coordinates'].apply(parse_coordinates)
    
    if not isinstance(sop_ids, list):
        sop_ids = [sop_ids]
    
    if len(sop_ids) == 0:
        print("No images to display.")
        return
    
    # Set number of images to display
    max_images = min(9, len(sop_ids))
    rows = int(np.ceil(max_images / 3))
    cols = min(3, max_images)
    
    fig = plt.figure(figsize=(15, 5 * rows))
    
    for i, sop_id in enumerate(sop_ids[:max_images]):
        # Check SeriesInstanceUID folders
        found = False
        series_id = ""
        file_path = ""
        
        for series_folder in os.listdir(base_dir):
            series_path = os.path.join(base_dir, series_folder)
            if os.path.isdir(series_path):
                # Check DICOM files in this folder
                for file_name in os.listdir(series_path):
                    if file_name.endswith('.dcm') and sop_id in file_name:
                        file_path = os.path.join(series_path, file_name)
                        series_id = series_folder
                        found = True
                        break
            if found:
                break
        
        plt.subplot(rows, cols, i+1)
        
        if found:
            try:
                # Load DICOM file
                ds = pydicom.dcmread(file_path)
                
                # Get and process image
                img = ds.pixel_array
                
                # Check image shape
                if len(img.shape) > 2 and img.shape[2] > 1:
                    # Convert multi-channel image to grayscale
                    img = np.mean(img, axis=2).astype(np.float32)
                
                # Detect images that look like thin lines (very small width or height)
                if img.shape[0] < 10 or img.shape[1] < 10:
                    raise ValueError(f"Corrupted image size: {img.shape}")
                
                # Check aspect ratio (very thin or very wide)
                aspect_ratio = img.shape[0] / img.shape[1]
                if aspect_ratio < 0.1 or aspect_ratio > 10:
                    raise ValueError(f"Corrupted image aspect ratio: {aspect_ratio:.2f}")
                
                # Apply standard window
                if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                    try:
                        img = apply_voi_lut(img, ds)
                    except Exception:
                        # If VOI LUT cannot be applied, adjust contrast directly
                        img = np.clip(img, 0, np.percentile(img, 99.5))
                
                if img.max() > 0:
                    img = img / img.max() * 255
                
                # Display image
                ax = plt.gca()
                ax.imshow(img, cmap='gray')
                
                # Find coordinates for this SeriesInstanceUID and SOPInstanceUID
                markers = localizers_df[(localizers_df['SeriesInstanceUID'] == series_id) & 
                                      (localizers_df['SOPInstanceUID'] == sop_id)]
                
                # Image dimensions - for correct scaling of coordinates
                img_height, img_width = img.shape
                
                # If coordinates exist, mark the points
                if not markers.empty:
                    for _, marker in markers.iterrows():
                        if marker['coords_dict'] is not None:
                            # Original coordinates
                            orig_x = marker['coords_dict']['x']
                            orig_y = marker['coords_dict']['y']
                            
                            # Check and limit coordinates 
                            # (some coordinates may exceed image dimensions)
                            if orig_x > img_width * 1.5 or orig_y > img_height * 1.5:
                                # If coordinates are much larger than image size,
                                # scale to original image size
                                x = orig_x * img_width / 512 if orig_x > 512 else orig_x
                                y = orig_y * img_height / 512 if orig_y > 512 else orig_y
                            else:
                                # Use coordinates directly
                                x = orig_x
                                y = orig_y
                            
                            # Ensure coordinates are within image bounds
                            x = min(max(0, x), img_width - 1)
                            y = min(max(0, y), img_height - 1)
                            
                            # Red, semi-transparent, small dot
                            ax.scatter(x, y, color='red', alpha=0.6, s=20, marker='o', 
                                      edgecolor='white', linewidth=0.5)
                            
                            # Optionally add location info
                            location = marker['location']
                            if location:
                                # Show location info on hover
                                ax.annotate(location, xy=(x, y), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=6, 
                                          color='white', backgroundcolor='black', alpha=0.7)
                
                plt.title(f"SOPInstanceUID: {sop_id[-8:]}", fontsize=10)
                # Add SeriesInstanceUID below the image in red
                plt.xlabel(f"SeriesInstanceUID: {series_id[-7:]}", fontsize=8, color='red')
                plt.axis('off')
                
            except Exception as e:
                error_msg = str(e)
                # Try to repair DICOM array
                try:
                    # Get correct dimensions from metadata
                    if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                        rows_dicom = ds.Rows
                        cols_dicom = ds.Columns
                        
                        # Try to reshape
                        try:
                            # Flatten
                            flat_pixels = ds.pixel_array.flatten()
                            
                            # Reshape to correct dimensions
                            if len(flat_pixels) >= rows_dicom * cols_dicom:
                                reshaped_img = flat_pixels[:rows_dicom * cols_dicom].reshape(rows_dicom, cols_dicom)
                                
                                # Normalize image
                                if reshaped_img.max() > 0:
                                    reshaped_img = reshaped_img / reshaped_img.max() * 255
                                
                                # Display the repaired image
                                ax = plt.gca()
                                ax.imshow(reshaped_img, cmap='gray')
                                
                                # Find coordinates for this SeriesInstanceUID and SOPInstanceUID
                                markers = localizers_df[(localizers_df['SeriesInstanceUID'] == series_id) & 
                                                      (localizers_df['SOPInstanceUID'] == sop_id)]
                                
                                # If coordinates exist, mark the points (scaled)
                                if not markers.empty:
                                    for _, marker in markers.iterrows():
                                        if marker['coords_dict'] is not None:
                                            x = min(marker['coords_dict']['x'], reshaped_img.shape[1] - 1)
                                            y = min(marker['coords_dict']['y'], reshaped_img.shape[0] - 1)
                                            # Red, semi-transparent, small dot
                                            ax.scatter(x, y, color='red', alpha=0.6, s=20, marker='o',
                                                      edgecolor='white', linewidth=0.5)
                                
                                plt.title(f"SOPInstanceUID: {sop_id[-8:]} (Repaired)", fontsize=10)
                                plt.xlabel(f"SeriesInstanceUID: {series_id[-7:]}", fontsize=8, color='red')
                                plt.axis('off')
                            else:
                                # Alternative approach: Transpose
                                if hasattr(ds, 'pixel_array'):
                                    # Transpose the image
                                    transposed = np.transpose(ds.pixel_array)
                                    
                                    # Normalize
                                    if transposed.max() > 0:
                                        transposed = transposed / transposed.max() * 255
                                    
                                    ax = plt.gca()
                                    ax.imshow(transposed, cmap='gray')
                                    
                                    # Find coordinates for this SeriesInstanceUID and SOPInstanceUID
                                    markers = localizers_df[(localizers_df['SeriesInstanceUID'] == series_id) & 
                                                          (localizers_df['SOPInstanceUID'] == sop_id)]
                                    
                                    # If coordinates exist, mark the points for transposed image
                                    if not markers.empty:
                                        for _, marker in markers.iterrows():
                                            if marker['coords_dict'] is not None:
                                                # Swap coordinates for transposed image
                                                y = min(marker['coords_dict']['x'], transposed.shape[0] - 1)
                                                x = min(marker['coords_dict']['y'], transposed.shape[1] - 1)
                                                # Red, semi-transparent, small dot
                                                ax.scatter(x, y, color='red', alpha=0.6, s=20, marker='o',
                                                          edgecolor='white', linewidth=0.5)
                                    
                                    plt.title(f"SOPInstanceUID: {sop_id[-8:]} (Transposed)", fontsize=10)
                                    plt.xlabel(f"SeriesInstanceUID: {series_id[-7:]}", fontsize=8, color='red')
                                    plt.axis('off')
                                else:
                                    plt.text(0.5, 0.5, f"No pixel data", ha='center', va='center', color='red')
                        except Exception as reshape_error:
                            plt.text(0.5, 0.5, f"Shape error: {str(reshape_error)[:30]}...", ha='center', va='center', color='red')
                    else:
                        plt.text(0.5, 0.5, f"Missing metadata", ha='center', va='center', color='red')
                except Exception as meta_error:
                    plt.text(0.5, 0.5, f"Error: {str(meta_error)[:30]}...", ha='center', va='center', color='red')
        else:
            plt.text(0.5, 0.5, "File not found", ha='center', va='center')
        
    plt.tight_layout()
    plt.show()


view_dicom_images_with_markers(cta_sop_ids)


view_dicom_images_with_markers(mra_sop_ids)


view_dicom_images_with_markers(mri_t1_sop_ids)


view_dicom_images_with_markers(mri_t2_sop_ids)




