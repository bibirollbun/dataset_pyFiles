!pip install duckdb --no-index --find-links=/kaggle/input/polars-and-duckdb/kaggle/working/mysitepackages/duck_pkg
!pip install python-gdcm
!pip install pylibjpeg
!pip install pylibjpeg-libjpeg==2.2.0
!pip install pylibjpeg-openjpeg==2.3.0
!pip install matplotlib==3.10.3
!pip install scikit-learn==1.7.0
!pip install polars --no-index --find-links=/kaggle/input/polars-and-duckdb/kaggle/working/mysitepackages/polars_pkg
!pip install pydicom


from pydicom import dcmread
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ImplicitVRLittleEndian

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import polars as pl
import duckdb as dd
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import pickle
import gc
import ctypes
from pathlib import Path
import logging
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras


mp.cpu_count()


print(tf.__version__)
print(tfio.__version__)


tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)

print("Number of accelerators: ", tpu_strategy.num_replicas_in_sync)


pl.Config(fmt_str_lengths=1000)
pl.Config.set_tbl_rows(1000)


train_meta_data = pl.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv'\
                              , low_memory=True)

train_locale_meta_data = pl.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv'\
                              , low_memory=True)

def parse_coordinates(coord_str):
    if coord_str is None:
        return None, None
    try:
        coord_dict = json.loads(coord_str.replace("'", '"'))
        return float(coord_dict.get('x', 0.0)), float(coord_dict.get('y', 0.0)), int(coord_dict.get('f', 0.0))
    except (json.JSONDecodeError, KeyError, ValueError, AttributeError):
        return None, None

train_locale_meta_data = train_locale_meta_data.with_columns([
    pl.col("coordinates")
    .map_elements(lambda x: parse_coordinates(x)[0], return_dtype=pl.Float64)
    .cast(pl.Float64)
    .alias("coordinates_x"),
    
    pl.col("coordinates")
    .map_elements(lambda x: parse_coordinates(x)[1], return_dtype=pl.Float64)
    .cast(pl.Float64)
    .alias("coordinates_y"),
    
    pl.col("coordinates")
    .map_elements(lambda x: parse_coordinates(x)[2], return_dtype=pl.Int32)
    .cast(pl.Int32)
    .alias("coordinates_f")
])

print("Train CSV shape : ", train_meta_data.shape)
print("Train Localizers CSV shape : ", train_locale_meta_data.shape)
# Show the first few rows
print(train_locale_meta_data.filter(pl.col('coordinates_f') != 0.0)\
      .select(["coordinates", "coordinates_x", "coordinates_y", "coordinates_f"]).head(5))


train_meta_data.head(10)


train_locale_meta_data.head(10)


print(train_locale_meta_data.select(["coordinates_x", "coordinates_y", "coordinates_f"]).describe())


allowed_tags = ['BitsAllocated', 'BitsStored', 'Rows', 'Columns', 'FrameOfReferenceUID', 'HighBit', 'ImageOrientationPatient'
                , 'ImagePositionPatient', 'InstanceNumber', 'Modality', 'PhotometricInterpretation'
                , 'PixelRepresentation', 'PixelSpacing', 'PlanarConfiguration', 'RescaleIntercept', 'RescaleSlope'
                , 'RescaleType', 'SamplesPerPixel', 'SliceThickness', 'SpacingBetweenSlices']


class DicomRecord:
    """
    Memory-efficient class for storing DICOM metadata using __slots__
    """
    __slots__ = ['folder_name', 'file_name', 'file_path', 'image_shape', 'min_max_diff'] + [
        'BitsAllocated', 'BitsStored', 'Rows', 'Columns', 'FrameOfReferenceUID',
        'HighBit', 'ImageOrientationPatient', 'ImagePositionPatient', 'InstanceNumber',
        'Modality', 'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing',
        'PlanarConfiguration', 'RescaleIntercept', 'RescaleSlope', 'RescaleType',
        'SamplesPerPixel', 'SliceThickness', 'SpacingBetweenSlices'
    ]
    
    def __init__(self, folder_name, file_name, file_path, image_shape, min_max_diff):
        self.folder_name = folder_name
        self.file_name = file_name
        self.file_path = file_path
        self.image_shape = image_shape
        self.min_max_diff = min_max_diff
        for tag in self.__slots__[5:]:  
            setattr(self, tag, None)
    
    def to_dict(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}


def process_single_folder(folder_path, allowed_tags):
    """
    Process a single folder of DICOM files and save image arrays
    """
    try:
        data = []
        dcm_files = list(Path(folder_path).glob("*.dcm"))
        folder_name = Path(folder_path).name
        
        for dcm_file in dcm_files:
            try:
                # Read DICOM file
                ds = dcmread(str(dcm_file))
                original_shape = str(ds.pixel_array.shape)
                min_max_diff = round(np.max(ds.pixel_array) - np.min(ds.pixel_array),5)
                
                # Create record
                record = DicomRecord(folder_name, dcm_file.name, str(dcm_file), original_shape, min_max_diff)
                
                # Fill in tags
                for tag in allowed_tags:
                    try:
                        value = getattr(ds, tag)
                        if hasattr(value, '__iter__') and not isinstance(value, str):
                            value = str(list(map(str, value)))
                        else:
                            value = str(value)
                        setattr(record, tag, value)
                    except (AttributeError, TypeError):
                        continue
                
                data.append(record.to_dict())
                
            except Exception as e:
                print(f"Error processing file {dcm_file}: {e}")
                continue
                
        return data
        
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
        return []


def create_dicom_dataset(root_folder, allowed_tags, num_processes=None, chunk_size=100):
    """
    Create dataset with metadata DataFrame and memory-mapped image arrays
    """
    root_path = Path(root_folder)
    folders = [f for f in root_path.iterdir() if f.is_dir()]
    
    if not num_processes:
        num_processes = mp.cpu_count()
    
    # Create directories for temporary and array storage
    temp_dir = Path("temp_chunks")
    temp_dir.mkdir(exist_ok=True)
    
    # Create schema
    schema = {
        'folder_name': pl.Utf8,
        'file_name': pl.Utf8,
        'file_path': pl.String,
        'image_shape': pl.String,
        'min_max_diff': pl.Float32
    }
    schema.update({tag: pl.Utf8 for tag in allowed_tags})
    
    # Process folders in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for i in range(0, len(folders), chunk_size):
            chunk_folders = folders[i:i+chunk_size]
            chunk_data = []
            
            futures = [
                executor.submit(
                    process_single_folder, 
                    str(folder), 
                    allowed_tags
                    #arrays_dir
                )
                for folder in chunk_folders
            ]
            
            for future in tqdm(futures, 
                             desc=f"Processing chunk {i//chunk_size + 1}/{(len(folders)-1)//chunk_size + 1}"):
                chunk_data.extend(future.result())
            
            if chunk_data:
                chunk_df = pl.DataFrame(
                    chunk_data,
                    schema=schema,
                    infer_schema_length=None
                )
                
                chunk_df.write_parquet(
                    temp_dir / f"dicom_metadata_chunk_{i//chunk_size}.parquet",
                    compression="snappy"
                )
                
                del chunk_data
                del chunk_df
    
    # Combine chunks
    print("\nCombining chunks...")
    chunk_files = list(temp_dir.glob("dicom_metadata_chunk_*.parquet"))
    final_df = pl.concat([
        pl.scan_parquet(str(chunk_file))
        for chunk_file in chunk_files
    ]).collect()
    
    # Clean up temporary files
    for f in chunk_files:
        f.unlink()
    temp_dir.rmdir()
    
    return final_df


#with tpu_strategy.scope():
root_folder = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"

try:
    metadata_df = create_dicom_dataset(
        root_folder, 
        allowed_tags, 
        num_processes=mp.cpu_count(),
        chunk_size=192
    )
except Exception as e:
    print(f"Error: {e}")


metadata_df.write_parquet('metadata_df.parquet')


metadata_df.columns


metadata_df.filter(pl.col('min_max_diff') == 0).select(['file_path'])


df_all_coordinates = dd.sql( \
    "select t2.coordinates_x, t2.coordinates_y, coalesce(t2.coordinates_f,0) as coordinates_f, t1.* \
    from (select * from metadata_df where min_max_diff != 0)t1 \
    left join train_locale_meta_data t2 \
    on t1.folder_name = t2.SeriesInstanceUID \
    and replace(t1.file_name, '.dcm','') = t2.SOPInstanceUID "\
).pl()

print(df_all_coordinates.shape)
print(df_all_coordinates.columns)


new_columns = [col.lower().replace(" ", "_") for col in train_meta_data.columns]
train_meta_data.columns = new_columns
print(train_meta_data.columns)


df_all_data = dd.sql( \
    "select t2.file_name, t2.file_path, t2.image_shape, t2.coordinates_x, t2.coordinates_y, t2.coordinates_f, t2.min_max_diff \
    , t1.aneurysm_present as aneurysm_present_in_series \
    , case when t2.coordinates_x is not null then 1 else 0 end as aneurysm_present_in_image \
    , t1.seriesinstanceuid, t1.patientage, t1.patientsex, t1.modality \
    , case when t2.coordinates_x is not null then t1.left_infraclinoid_internal_carotid_artery \
    else 0 end as left_infraclinoid_internal_carotid_artery \
    , case when t2.coordinates_x is not null then t1.right_infraclinoid_internal_carotid_artery \
    else 0 end as right_infraclinoid_internal_carotid_artery \
    , case when t2.coordinates_x is not null then t1.left_supraclinoid_internal_carotid_artery \
    else 0 end as left_supraclinoid_internal_carotid_artery \
    , case when t2.coordinates_x is not null then t1.right_supraclinoid_internal_carotid_artery \
    else 0 end as right_supraclinoid_internal_carotid_artery \
    , case when t2.coordinates_x is not null then t1.left_middle_cerebral_artery \
    else 0 end as left_middle_cerebral_artery \
    , case when t2.coordinates_x is not null then t1.right_middle_cerebral_artery \
    else 0 end as right_middle_cerebral_artery \
    , case when t2.coordinates_x is not null then t1.anterior_communicating_artery \
    else 0 end as anterior_communicating_artery \
    , case when t2.coordinates_x is not null then t1.left_anterior_cerebral_artery \
    else 0 end as left_anterior_cerebral_artery \
    , case when t2.coordinates_x is not null then t1.right_anterior_cerebral_artery \
    else 0 end as right_anterior_cerebral_artery \
    , case when t2.coordinates_x is not null then t1.left_posterior_communicating_artery \
    else 0 end as left_posterior_communicating_artery \
    , case when t2.coordinates_x is not null then t1.right_posterior_communicating_artery \
    else 0 end as right_posterior_communicating_artery \
    , case when t2.coordinates_x is not null then t1.basilar_tip \
    else 0 end as basilar_tip \
    , case when t2.coordinates_x is not null then t1.other_posterior_circulation \
    else 0 end as other_posterior_circulation \
    from train_meta_data t1 \
    join df_all_coordinates t2 \
    on t1.SeriesInstanceUID = t2.folder_name" \
).pl()

print("Full training data: ", df_all_data.shape)
print("Full training data columns: ", df_all_data.columns)
print("Aneurysm not present in {0} series".format(df_all_data.filter(pl.col("coordinates_x").is_null()).shape[0]))

print("Aneurysm present in {0} series".format(df_all_data.filter(pl.col("coordinates_x").is_not_null()).shape[0]))

print("Aneurysm not shown in {0} images".format(df_all_data.filter(pl.col("aneurysm_present_in_image")==0).shape[0]))

print("Aneurysm shown in {0} images".format(df_all_data.filter(pl.col("aneurysm_present_in_image")==1).shape[0]))

print(df_all_data.select(["coordinates_x", "coordinates_y", "coordinates_f"]).describe())


df_all_data.head(5)


dd.sql(" \
select t1.modality, t1.aneurysm_present_in_image, t1.per_mod_count, \
round(t1.per_mod_count/t2.total_count ,3) as modality_pct \
from \
( \
select modality, aneurysm_present_in_image, cast(count(1) as float) as per_mod_count from df_all_data \
group by modality, aneurysm_present_in_image \
)t1 \
join \
(select cast(count(1) as float) as total_count from df_all_data)t2 \
on 1=1 \
order by 1 \
").pl()


dd.sql(" \
select t1.image_shape, t1.aneurysm_present_in_image, t1.per_mod_count, \
round(t1.per_mod_count/t2.total_count ,3) as modality_pct \
from \
( \
select image_shape, aneurysm_present_in_image, cast(count(1) as float) as per_mod_count from df_all_data \
group by image_shape, aneurysm_present_in_image \
)t1 \
join \
(select cast(count(1) as float) as total_count from df_all_data)t2 \
on 1=1 \
order by 3 desc \
").pl()


df_all_data = dd.sql(" select \
case when left_infraclinoid_internal_carotid_artery = 1 then \'left_infraclinoid_internal_carotid_artery\' \
when right_infraclinoid_internal_carotid_artery = 1 then \'right_infraclinoid_internal_carotid_artery\' \
when left_supraclinoid_internal_carotid_artery = 1 then \'left_supraclinoid_internal_carotid_artery\' \
when right_supraclinoid_internal_carotid_artery = 1 then \'right_supraclinoid_internal_carotid_artery\' \
when left_middle_cerebral_artery = 1 then \'left_middle_cerebral_artery\' \
when right_middle_cerebral_artery = 1 then \'right_middle_cerebral_artery\' \
when anterior_communicating_artery = 1 then \'anterior_communicating_artery\' \
when left_anterior_cerebral_artery = 1 then \'left_anterior_cerebral_artery\' \
when right_anterior_cerebral_artery = 1 then \'right_anterior_cerebral_artery\' \
when left_posterior_communicating_artery = 1 then \'left_posterior_communicating_artery\' \
when right_posterior_communicating_artery = 1 then \'right_posterior_communicating_artery\' \
when basilar_tip = 1 then \'basilar_tip\' \
when other_posterior_circulation = 1 then \'other_posterior_circulation\' \
else \'no_aneurysm\' end as aneurysm_position \
, * \
from df_all_data" \
).pl()


df_all_data.head(5)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_all_data = df_all_data.with_columns(
        pl.Series(
            "aneurysm_position_encoded",
            le.fit_transform(df_all_data["aneurysm_position"].to_numpy())
        )
    )


df_all_data.write_parquet('full_training_data.parquet')


df_all_data = pl.read_parquet('/kaggle/input/rsna-aneurysm-train-metadata-suman/full_training_data_v2.parquet')
print("Shape of training metadata", df_all_data.shape)
df_all_data.columns


pl.Config(fmt_str_lengths=1000)
pl.Config.set_tbl_rows(1000)


df_all_data.filter((pl.col('modality')=='MRI T1post') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('file_path')).item(0, 0)


df_all_data.filter((pl.col('modality')=='MRI T1post') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('coordinates_x')).item(0, 0)


df_all_data.filter((pl.col('modality')=='MRI T1post') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('coordinates_y')).item(0, 0)


dd.sql("select * from df_all_data where modality = 'MRI T1post' and aneurysm_present_in_image = 1 limit 5 ").pl()


dd.sql("select * from df_all_data where modality = 'MRI T2' and aneurysm_present_in_image = 1 limit 5 ").pl()


dd.sql("select * from df_all_data where modality = 'MRA' and aneurysm_present_in_image = 1 limit 5 ").pl()


dd.sql("select * from df_all_data where modality = 'CTA' and aneurysm_present_in_image = 1 limit 5 ").pl()


def extract_single_frame(multiframe_path, slice_number, output_path=None):
    """
    Extract a single frame from a multi-frame DICOM
    
    Args:
        multiframe_path: Path to multi-frame DICOM file
        slice_number: The slice number to extract (0-based index)
        output_path: Path to save the single-frame DICOM. If None, returns the dataset
    """
    try:
        # Read the multi-frame DICOM with force=True to handle potentially corrupted files
        multi_ds = dcmread(multiframe_path, force=True)
        
        # Verify it's a multi-frame image
        if not hasattr(multi_ds, 'NumberOfFrames'):
            raise ValueError("Input DICOM is not a multi-frame image")
        
        # Check if slice number is valid
        if slice_number >= multi_ds.NumberOfFrames:
            raise ValueError(f"Slice number {slice_number} is out of range. "
                           f"Image has {multi_ds.NumberOfFrames} frames")
        
        # Create new dataset for single frame
        single_ds = FileDataset(output_path or "temp.dcm", {}, 
                              file_meta=FileMetaDataset(), 
                              preamble=b"\0" * 128)
        
        # Copy attributes from multi-frame dataset
        attrs_to_copy = allowed_tags
        
        for attr in attrs_to_copy:
            if hasattr(multi_ds, attr):
                setattr(single_ds, attr, getattr(multi_ds, attr))
        
        # Generate new UIDs
        single_ds.SOPInstanceUID = generate_uid()
        single_ds.file_meta.MediaStorageSOPInstanceUID = single_ds.SOPInstanceUID
        
        # Set transfer syntax to uncompressed little endian
        single_ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        single_ds.file_meta.MediaStorageSOPClassUID = multi_ds.file_meta.MediaStorageSOPClassUID
        if hasattr(multi_ds.file_meta, 'ImplementationClassUID'):
            single_ds.file_meta.ImplementationClassUID = multi_ds.file_meta.ImplementationClassUID
        
        # Set instance-specific attributes
        single_ds.InstanceNumber = slice_number + 1
        
        try:
            # Try to get pixel array directly
            pixel_array = multi_ds.pixel_array[slice_number]
        except Exception as e:
            #print(f"Warning: Could not directly access pixel_array: {e}")
            # Alternative approach: decompress and get pixels
            if hasattr(multi_ds, 'decompress'):
                multi_ds.decompress()
            pixel_array = multi_ds.pixel_array[slice_number]
        
        # Set pixel data
        single_ds.PixelData = pixel_array.tobytes()
        
        # Update image-specific attributes
        single_ds.NumberOfFrames = 1
        
        # Try to copy position and orientation
        try:
            if hasattr(multi_ds, 'PerFrameFunctionalGroupsSequence'):
                frame_content = multi_ds.PerFrameFunctionalGroupsSequence[slice_number]
                
                if hasattr(frame_content, 'PlanePositionSequence'):
                    position = frame_content.PlanePositionSequence[0].ImagePositionPatient
                    single_ds.ImagePositionPatient = position
                
                if hasattr(frame_content, 'PlaneOrientationSequence'):
                    orientation = frame_content.PlaneOrientationSequence[0].ImageOrientationPatient
                    single_ds.ImageOrientationPatient = orientation
        except Exception as e:
            #print(f"Warning: Could not copy position/orientation: {e}")
            raise
        
        # Add creation timestamp
        dt = datetime.datetime.now()
        single_ds.ContentDate = dt.strftime('%Y%m%d')
        single_ds.ContentTime = dt.strftime('%H%M%S.%f')
        
        # Save or return the dataset
        if output_path:
            single_ds.save_as(output_path, write_like_original=False)
            return None
        return single_ds
    
    except Exception as e:
        #print(f"Error extracting frame: {e}")
        raise

# Alternative version using different approach for compressed files
def extract_single_frame_alternative(multiframe_path, slice_number, output_path=None):
    """
    Alternative version for handling problematic files
    """
    try:
        # Read with force and stop before pixels
        multi_ds = dcmread(multiframe_path, force=True, stop_before_pixels=True)
        
        # Read pixel data separately
        with open(multiframe_path, 'rb') as f:
            multi_ds.PixelData = f.read()
        
        # Decompress if needed
        if hasattr(multi_ds, 'decompress'):
            multi_ds.decompress()
        
        # Get pixel array
        pixel_array = multi_ds.pixel_array[slice_number]
        
        # Create new dataset
        single_ds = FileDataset(output_path or "temp.dcm", {}, 
                              file_meta=FileMetaDataset(), 
                              preamble=b"\0" * 128)
        
        # Copy attributes (same as before)
        attrs_to_copy = allowed_tags
        
        for attr in attrs_to_copy:
            if hasattr(multi_ds, attr):
                setattr(single_ds, attr, getattr(multi_ds, attr))
        
        # Generate new UIDs
        single_ds.SOPInstanceUID = generate_uid()
        single_ds.file_meta.MediaStorageSOPInstanceUID = single_ds.SOPInstanceUID
        
        # Set transfer syntax to uncompressed little endian
        single_ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        single_ds.file_meta.MediaStorageSOPClassUID = multi_ds.file_meta.MediaStorageSOPClassUID
        if hasattr(multi_ds.file_meta, 'ImplementationClassUID'):
            single_ds.file_meta.ImplementationClassUID = multi_ds.file_meta.ImplementationClassUID
        
        # Set instance-specific attributes
        single_ds.InstanceNumber = slice_number + 1
        
        try:
            # Try to get pixel array directly
            pixel_array = multi_ds.pixel_array[slice_number]
        except Exception as e:
            #print(f"Warning: Could not directly access pixel_array: {e}")
            # Alternative approach: decompress and get pixels
            if hasattr(multi_ds, 'decompress'):
                multi_ds.decompress()
            pixel_array = multi_ds.pixel_array[slice_number]
        
        # Set pixel data
        single_ds.PixelData = pixel_array.tobytes()
        
        # Update image-specific attributes
        single_ds.NumberOfFrames = 1
        
        # Try to copy position and orientation
        try:
            if hasattr(multi_ds, 'PerFrameFunctionalGroupsSequence'):
                frame_content = multi_ds.PerFrameFunctionalGroupsSequence[slice_number]
                
                if hasattr(frame_content, 'PlanePositionSequence'):
                    position = frame_content.PlanePositionSequence[0].ImagePositionPatient
                    single_ds.ImagePositionPatient = position
                
                if hasattr(frame_content, 'PlaneOrientationSequence'):
                    orientation = frame_content.PlaneOrientationSequence[0].ImageOrientationPatient
                    single_ds.ImageOrientationPatient = orientation
        except Exception as e:
            #print(f"Warning: Could not copy position/orientation: {e}")
            raise
        
        # Add creation timestamp
        dt = datetime.datetime.now()
        single_ds.ContentDate = dt.strftime('%Y%m%d')
        single_ds.ContentTime = dt.strftime('%H%M%S.%f')
        
        # Save or return the dataset
        if output_path:
            single_ds.save_as(output_path, write_like_original=False)
            return None
        return single_ds
        
    except Exception as e:
        #print(f"Error in alternative extraction: {e}")
        raise

# Function to try both methods
def safe_extract_single_frame(multiframe_path, slice_number, output_path=None):
    """
    Try both extraction methods
    """
    try:
        return extract_single_frame(multiframe_path, slice_number, output_path)
    except Exception as e:
        #print(f"Primary method failed: {e}")
        #print("Trying alternative method...")
        try:
            return extract_single_frame_alternative(multiframe_path, slice_number, output_path)
        except Exception as e2:
            #print(f"Alternative method also failed: {e2}")
            raise

# Version with zoom functionality
def load_and_view_single_slice_with_zoom(dcm_path, x_coord, y_coord, f_coord=None, zoom_size=100):
    """
    Load and display a single DICOM slice with crosshair and zoomed inset
    
    Args:
        dcm_path: Path to the DICOM file
        x_coord: x coordinate for the crosshair
        y_coord: y coordinate for the crosshair
        zoom_size: Size of the zoom window in pixels
    """
    # Read DICOM file
    if f_coord:
        ds = safe_extract_single_frame(dcm_path, f_coord)
    else:
        ds = dcmread(dcm_path)
    img = ds.pixel_array
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Main image with crosshair
    ax1.imshow(img, cmap='gray')
    ax1.axvline(x=x_coord, color='red', alpha=0.5)
    ax1.axhline(y=y_coord, color='red', alpha=0.5)
    ax1.plot(x_coord, y_coord, 'r+', markersize=10, markeredgewidth=2)
    
    # Zoomed region
    x_start = int(max(0, x_coord - zoom_size/2))
    x_end = int(min(img.shape[1], x_coord + zoom_size/2))
    y_start = int(max(0, y_coord - zoom_size/2))
    y_end = int(min(img.shape[0], y_coord + zoom_size/2))
    
    zoomed = img[y_start:y_end, x_start:x_end]
    ax2.imshow(zoomed, cmap='gray')
    
    # Add crosshair to zoomed region
    center_x = x_coord - x_start
    center_y = y_coord - y_start
    ax2.axvline(x=center_x, color='red', alpha=0.5)
    ax2.axhline(y=center_y, color='red', alpha=0.5)
    ax2.plot(center_x, center_y, 'r+', markersize=10, markeredgewidth=2)
    
    ax1.axis('off')
    ax2.axis('off')
    ax1.set_title('Full Image')
    ax2.set_title('Zoomed Region')
    
    plt.tight_layout()
    plt.show()


dcm_path = df_all_data.filter((pl.col('modality')=='MRI T1post') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('file_path')).item(0, 0)

x_coord = df_all_data.filter((pl.col('modality')=='MRI T1post') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('coordinates_x')).item(0, 0)

y_coord = df_all_data.filter((pl.col('modality')=='MRI T1post') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('coordinates_y')).item(0, 0)

f_coord = df_all_data.filter((pl.col('modality')=='MRI T1post') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('coordinates_f')).item(0, 0)

load_and_view_single_slice_with_zoom(dcm_path, x_coord, y_coord, f_coord)


if f_coord:
    ds = safe_extract_single_frame(dcm_path, f_coord)
else:
    ds = dcmread(dcm_path)

for tag in allowed_tags:
    print("{0} = {1}".format(tag, getattr(ds, tag, None)))


dcm_path = df_all_data.filter((pl.col('modality')=='MRI T2') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('file_path')).item(0, 0)

x_coord = df_all_data.filter((pl.col('modality')=='MRI T2') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('coordinates_x')).item(0, 0)

y_coord = df_all_data.filter((pl.col('modality')=='MRI T2') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('coordinates_y')).item(0, 0)

f_coord = df_all_data.filter((pl.col('modality')=='MRI T2') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('coordinates_f')).item(0, 0)

load_and_view_single_slice_with_zoom(dcm_path, x_coord, y_coord, f_coord)


if f_coord:
    ds = safe_extract_single_frame(dcm_path, f_coord)
else:
    ds = dcmread(dcm_path)

for tag in allowed_tags:
    print("{0} = {1}".format(tag, getattr(ds, tag, None)))


dcm_path = df_all_data.filter((pl.col('modality')=='MRA') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('file_path')).item(0, 0)

x_coord = df_all_data.filter((pl.col('modality')=='MRA') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('coordinates_x')).item(0, 0)

y_coord = df_all_data.filter((pl.col('modality')=='MRA') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('coordinates_y')).item(0, 0)

f_coord = df_all_data.filter((pl.col('modality')=='MRA') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('coordinates_f')).item(0, 0)

load_and_view_single_slice_with_zoom(dcm_path, x_coord, y_coord, f_coord)


if f_coord:
    ds = safe_extract_single_frame(dcm_path, f_coord)
else:
    ds = dcmread(dcm_path)

for tag in allowed_tags:
    print("{0} = {1}".format(tag, getattr(ds, tag, None)))


dcm_path = df_all_data.filter((pl.col('modality')=='CTA') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('file_path')).item(0, 0)

x_coord = df_all_data.filter((pl.col('modality')=='CTA') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('coordinates_x')).item(0, 0)

y_coord = df_all_data.filter((pl.col('modality')=='CTA') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('coordinates_y')).item(0, 0)

f_coord = df_all_data.filter((pl.col('modality')=='CTA') & (pl.col('aneurysm_present_in_image')==1))\
.select(pl.col('coordinates_f')).item(0, 0)

load_and_view_single_slice_with_zoom(dcm_path, x_coord, y_coord, f_coord)


if f_coord:
    ds = safe_extract_single_frame(dcm_path, f_coord)
else:
    ds = dcmread(dcm_path)

for tag in allowed_tags:
    print("{0} = {1}".format(tag, getattr(ds, tag, None)))


if f_coord:
    ds = safe_extract_single_frame(dcm_path, f_coord)
else:
    ds = dcmread(dcm_path)

arr = ds.pixel_array.astype(np.float32)

# Apply rescale if present
slope = float(getattr(ds, "RescaleSlope", 1) or 1)
intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
arr = arr * slope + intercept

# Handle MONOCHROME1 (invert)
if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
    arr = arr.max() - arr

image = tf.convert_to_tensor(arr)

expanded_image = tf.expand_dims(image, -1)
m, M=tf.math.reduce_min(expanded_image), tf.math.reduce_max(expanded_image)
expanded_image = (tf.image.grayscale_to_rgb(expanded_image)-m)/(M-m)
expanded_image = tf.image.resize(expanded_image, (128,128))
sqzd_image = tf.squeeze(expanded_image)

train_img = tf.reshape(sqzd_image, shape=(128, 128, 3))


image_np = train_img.numpy()
plt.imshow(image_np)
plt.title("TensorFlow Image Visualization")
plt.axis('off') # Hide axes for cleaner image display
plt.show()

