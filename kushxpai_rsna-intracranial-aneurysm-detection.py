import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path

# Load CSV
train_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
series_id = train_df.SeriesInstanceUID.iloc[0]

# Full path to series folder
series_path = Path(f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{series_id}")

# Get sorted DICOM files
dicom_paths = sorted(series_path.glob("*.dcm"),
                     key=lambda x: pydicom.dcmread(str(x)).InstanceNumber)

if not dicom_paths:
    raise FileNotFoundError(f"No DICOM files found for {series_id}")

# Read middle slice
mid_idx = len(dicom_paths) // 2
ds = pydicom.dcmread(str(dicom_paths[mid_idx]))

# Display
plt.imshow(ds.pixel_array, cmap="gray")
plt.title(f"Series: {series_id}")
plt.axis("off")
plt.show()



import pandas as pd
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk
import ipywidgets as widgets


def preprocess_volume(series_path):
    """Read DICOM series, resample to 1mm³, clip & normalize."""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(series_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Resample to isotropic 1mm spacing
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing((1.0, 1.0, 1.0))
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetSize([int(sz * sp / 1.0) 
                      for sz, sp in zip(image.GetSize(), image.GetSpacing())])
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    image = resample.Execute(image)

    # Convert to NumPy array
    arr = sitk.GetArrayFromImage(image).astype(np.float32)

    # Clip to [-100, 400] HU and normalize to [0, 1]
    arr = np.clip(arr, -100, 400)
    arr = (arr - arr.min()) / (arr.max() - arr.min())

    return arr


train_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
series_id = train_df.SeriesInstanceUID.iloc[0]  # pick first series
series_path = Path(f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{series_id}")


dicom_paths = sorted(series_path.glob("*.dcm"),
                     key=lambda x: pydicom.dcmread(str(x)).InstanceNumber)

if not dicom_paths:
    raise FileNotFoundError(f"No DICOM files found for {series_id}")

mid_idx = len(dicom_paths) // 2
ds = pydicom.dcmread(str(dicom_paths[mid_idx]))

plt.imshow(ds.pixel_array, cmap="gray")
plt.title(f"Series: {series_id} (Raw)")
plt.axis("off")
plt.show()


arr = preprocess_volume(str(series_path))


mid_idx = arr.shape[0] // 2
plt.imshow(arr[mid_idx], cmap="gray")
plt.title(f"Series: {series_id} (Preprocessed)")
plt.axis("off")
plt.show()


@widgets.interact(slice_idx=(0, arr.shape[0]-1))
def view_slice(slice_idx=0):
    plt.imshow(arr[slice_idx], cmap="gray")
    plt.axis("off")
    plt.show()

