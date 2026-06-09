import os

import numpy as np 
import pandas as pd

import pydicom
import matplotlib.pyplot as plt
import nibabel as nib

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

rd = "../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/"

train_df = pd.read_csv(rd + "stage_2_train.csv")
print(train_df.head())

print('-'*120)
train_df['subtype'] = train_df.ID.apply(lambda x: x.split('_')[2])
train_df['ID'] = train_df.ID.apply(lambda x: '_'.join(x.split('_')[:2]))
print(train_df.shape)
print(train_df['Label'].value_counts())
train_df.head()


train_files = rd + "stage_2_train/"
print("Len of dicom files: ", len(os.listdir(train_files)))
print("Uniq IDs: ", len(train_df.ID.unique()))
print("Ratio: ", train_df.shape[0] / len(train_df.ID.unique()))


img = pydicom.dcmread(train_files + 'ID_5c8b5d701.dcm')
img


# https://www.kaggle.com/code/samanesharifi/eda-view-dicom-images-with-correct-windowing


def window_image(img, window_center,window_width, intercept, slope):

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    return img 

def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

window_center, window_width, intercept, slope = get_windowing(img)
print(window_center , window_width, intercept, slope)
img_arr = window_image(img.pixel_array, window_center, window_width, intercept, slope)
plt.imshow(img_arr, cmap='gray')
plt.grid(False)
plt.show()


# hs, ws, cs = set(), set(), set()
# for idx, filename in tqdm(enumerate(os.listdir(train_files)), total=10**3):
#     img_path = os.path.join(train_files, filename)
#     img = pydicom.dcmread(img_path).pixel_array
#     hs.add(img.shape[0])
#     ws.add(img.shape[1])
#     if len(img.shape)>2:
#         cs.add(img.shape[2])

#     if idx==10**3:
#         break

# print('uniq heights: ', hs)
# print('uniq widths: ', ws)
# print('uniq channels: ', cs)


def get_metadata(image_dir):

    labels = [
        'BitsAllocated', 'BitsStored', 'Columns', 'HighBit', 
        'ImageOrientationPatient_0', 'ImageOrientationPatient_1', 'ImageOrientationPatient_2',
        'ImageOrientationPatient_3', 'ImageOrientationPatient_4', 'ImageOrientationPatient_5',
        'ImagePositionPatient_0', 'ImagePositionPatient_1', 'ImagePositionPatient_2',
        'Modality', 'PatientID', 'PhotometricInterpretation', 'PixelRepresentation',
        'PixelSpacing_0', 'PixelSpacing_1', 'RescaleIntercept', 'RescaleSlope', 'Rows', 'SOPInstanceUID',
        'SamplesPerPixel', 'SeriesInstanceUID', 'StudyID', 'StudyInstanceUID', 
        'WindowCenter', 'WindowWidth', 'Image',
    ]

    data = {l: [] for l in labels}
    i = 0
    for image in tqdm(os.listdir(image_dir)):
        data["Image"].append(image[:-4])

        ds = pydicom.dcmread(os.path.join(image_dir, image))

        for metadata in ds.dir():
            if metadata != "PixelData":
                metadata_values = getattr(ds, metadata)
                if type(metadata_values) == pydicom.multival.MultiValue and metadata not in ["WindowCenter", "WindowWidth"]:
                    for i, v in enumerate(metadata_values):
                        data[f"{metadata}_{i}"].append(v)
                else:
                    if type(metadata_values) == pydicom.multival.MultiValue and metadata in ["WindowCenter", "WindowWidth"]:
                        data[metadata].append(metadata_values[0])
                    else:
                        data[metadata].append(metadata_values)
                        
        i+=1

    # print(data)
    return pd.DataFrame(data).set_index("Image")



train_df = pd.read_csv(f'{rd}/stage_2_train.csv').drop_duplicates()
train_df['ImageID'] = train_df['ID'].str.slice(stop=12)
train_df['Diagnosis'] = train_df['ID'].str.slice(start=13)
train_labels = train_df.pivot(index="ImageID", columns="Diagnosis", values="Label")
train_labels.head()


# train_metadata = get_metadata(os.path.join(rd, "stage_2_train"))
# test_metadata = get_metadata(os.path.join(rd, "stage_2_test"))

# train_metadata.to_parquet(f'./train_metadata.parquet.gzip', compression='gzip')
# test_metadata.to_parquet(f'./test_metadata.parquet.gzip', compression='gzip')


! ls /kaggle/input/rsna19-metadata-zip/


train_metadata = pd.read_parquet(f'/kaggle/input/rsna19-metadata-zip/train_metadata.parquet.gzip')
test_metadata = pd.read_parquet(f'/kaggle/input/rsna19-metadata-zip/test_metadata.parquet.gzip')

train_metadata["Dataset"] = "train"
test_metadata["Dataset"] = "test"

train_metadata = train_metadata.join(train_labels)

metadata = pd.concat([train_metadata, test_metadata], sort=True)
# metadata = test_metadata #pd.concat([train_metadata, test_metadata], sort=True)

metadata.sort_values(by="ImagePositionPatient_2", inplace=True, ascending=False)
metadata.head()


metadata["StudyInstanceUID"].nunique()


def window_img(dcm, width=None, center=None, norm=True):
    pixels = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    
    # Pad non-square images
    if pixels.shape[0] != pixels.shape[1]:
        (a,b) = pixels.shape
        if a > b:
            padding = ((0, 0), ((a-b) // 2, (a-b) // 2))
        else:
            padding = (((b-a) // 2, (b-a) // 2), (0, 0))
        pixels = np.pad(pixels, padding, mode='constant', constant_values=0)
            
    if not width:
        width = dcm.WindowWidth
        if type(width) != pydicom.valuerep.DSfloat:
            width = width[0]
    if not center:
        center = dcm.WindowCenter
        if type(center) != pydicom.valuerep.DSfloat:
            center = center[0]
    lower = center - (width / 2)
    upper = center + (width / 2)
    img = np.clip(pixels, lower, upper)
    
    if norm:
        return (img - lower) / (upper - lower)
    else:
        return img


studies = metadata.groupby("StudyInstanceUID")
studies_list = list(studies)

os.makedirs('npy_files', exist_ok=True)

for i in range(len(studies_list)): #range(1000):
    
    study_name, study_df = studies_list[i]

    volume, labels = [], []
    for index, row in study_df.iterrows():
        if row["Dataset"] == "train":
            dcm = pydicom.dcmread(os.path.join(rd, "stage_2_train", index+".dcm"))
        else:
            dcm = pydicom.dcmread(os.path.join(rd, "stage_2_test", index+".dcm"))
            
        img = dcm.pixel_array #window_img(dcm, width=80, center=40, norm=False)
        label = row[["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]]
        volume.append(img)
        labels.append(label)
        
    volume = np.array(volume)
    labels = np.array(labels)

    np.save(f'npy_files/{study_name}.npy', volume)
    np.save(f'npy_files/{study_name}_labels.npy', labels)

    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(volume, affine)
    output_file = f"{study_name}_nifti_file_from_numpy.nii.gz"  # Replace with desired output file name
    nib.save(nifti_img, output_file)

    
    break
    



i = 100
studies = metadata[metadata['Dataset']=='train'].groupby("StudyInstanceUID")
studies_list = list(studies)

study_name, study_df = studies_list[i]


import shutil
os.makedirs(study_name, exist_ok=True)

for index, row in study_df.iterrows():
    dicom_path = os.path.join(rd, "stage_2_train", index+".dcm")
    assert os.path.exists(dicom_path)

    shutil.copy(dicom_path, os.path.join(study_name, index+".dcm"))



! pip install dicom2nifti



! dicom2nifti ./ID_010b7f38e0/ ./ -M


! ls


import dicom2nifti
import nibabel as nib

dicom2nifti.convert_directory('./ID_010b7f38e0', ".")



dicom_directory = study_name
output_file = f'{study_name}_nifti/nifti_file.nii.gz'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
print(os.path.dirname(output_file))
# dicom2nifti.convert_directory(dicom_directory, os.path.dirname(output_file), compression=True, reorient=True)

dicom2nifti.dicom_series_to_nifti([dicom_directory], output_file, reorient_nifti=True)

print(os.listdir(os.path.dirname(output_file)))


# nifti_image = nib.load(output_file)
# print(nifti_image.shape)
# print(nifti_image.header)


import os
import numpy as np
import nibabel as nib
import pydicom

# Define the DICOM folder path
dicom_folder = "ID_010b7f38e0"  # Replace with your DICOM folder

# Load all DICOM files
dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm")]
dicom_files.sort()  # Ensure slices are in order

# Read the first DICOM file to get metadata
first_dicom = pydicom.dcmread(dicom_files[0])
pixel_spacing = first_dicom.PixelSpacing  # e.g., [row_spacing, col_spacing]
slice_thickness = 1.0 #float(first_dicom.SliceThickness)
affine = np.array([
    [pixel_spacing[0], 0, 0, 0],
    [0, pixel_spacing[1], 0, 0],
    [0, 0, slice_thickness, 0],
    [0, 0, 0, 1]
])

# Stack all slices into a 3D NumPy array
images = [pydicom.dcmread(f).pixel_array for f in dicom_files]
volume = np.stack(images, axis=-1)

# Convert to NIfTI and save
nifti_img = nib.Nifti1Image(volume, affine)
output_file = "output_file_gpt.nii.gz"
nib.save(nifti_img, output_file)

print(f"NIfTI file saved as {output_file}")



first_dicom


# https://www.kaggle.com/code/amirmohammadparvizi/identify-acute-intracranial-hemorrhage


# def correct_dcm(dcm):
#     """
#     Correct DICOM pixel values.
#     """
#     x = dcm.pixel_array + 1000
#     px_mode = 4096
#     x[x >= px_mode] -= px_mode
#     dcm.PixelData = x.tobytes()
#     dcm.RescaleIntercept = -1000

# def window_image(dcm, window_center, window_width):
#     """
#     Apply specified window level and width to the DICOM image.
#     """
#     if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
#         correct_dcm(dcm)
    
#     img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
#     img_min = window_center - window_width // 2
#     img_max = window_center + window_width // 2
#     img = np.clip(img, img_min, img_max)

#     return img

# def bsb_window(dcm):
#     """
#     Apply Brain, Subdural, and Soft tissue windowing to the DICOM image.
#     """
#     brain_img = window_image(dcm, 40, 80)
#     subdural_img = window_image(dcm, 80, 200)
#     soft_img = window_image(dcm, 40, 380)
    
#     brain_img = (brain_img - 0) / 80
#     subdural_img = (subdural_img - (-20)) / 200
#     soft_img = (soft_img - (-150)) / 380
#     bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

#     return bsb_img

# plt.subplot(1, 3, 1)
# plt.imshow(window_image(img, 40, 80), cmap='gray')
# plt.title('Brain img')

# plt.subplot(1, 3, 2)
# plt.imshow(window_image(img, 80, 200), cmap='gray')
# plt.title('Subdural img')

# plt.subplot(1, 3, 3)
# plt.imshow(window_image(img, 40, 380), cmap='gray')
# plt.title('Soft img')
# plt.show()

# plt.figure()
# plt.imshow(bsb_window(img))
# plt.show()


# def window_with_correction(dcm, window_center, window_width):
#     """
#     Apply windowing to DICOM image with correction for specific conditions.
#     """
#     if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
#         correct_dcm(dcm)
#     img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
#     img_min = window_center - window_width // 2
#     img_max = window_center + window_width // 2
#     img = np.clip(img, img_min, img_max)
#     return img

# def window_without_correction(dcm, window_center, window_width):
#     """
#     Apply windowing to DICOM image without correction.
#     """
#     img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
#     img_min = window_center - window_width // 2
#     img_max = window_center + window_width // 2
#     img = np.clip(img, img_min, img_max)
#     return img

# def window_testing(img, window):
#     """
#     Apply Brain, Subdural, and Soft tissue windowing to the DICOM image.
#     """
#     brain_img = window(img, 40, 80)
#     subdural_img = window(img, 80, 200)
#     soft_img = window(img, 40, 380)
    
#     brain_img = (brain_img - 0) / 80
#     subdural_img = (subdural_img - (-20)) / 200
#     soft_img = (soft_img - (-150)) / 380
#     bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

#     return bsb_img


# # Plot original and corrected images side by side
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(window_testing(img, window_without_correction), cmap='gray')
# ax[0].set_title("Original")
# ax[1].imshow(window_testing(img, window_with_correction), cmap='gray')
# ax[1].set_title("Corrected")
# plt.show()







