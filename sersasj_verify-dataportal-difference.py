!pip install awscli --break-system-packages
!pip install zarr


# denoised tomogram 
# source https://cryoetdataportal.czscience.com/runs/16463?table-tab=Tomograms&metadata=tomogram&tab=metadata
!aws s3 --no-sign-request sync s3://cryoet-data-portal-public/10440/TS_5_4/Reconstructions/VoxelSpacing10.012/Tomograms/100/TS_5_4.zarr TS_5_4.zarr


kaggle_original_tomogram_path = "/kaggle/input/czii-cryo-et-object-identification/test/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr"
data_porta_denoised_tomogram_path = "/kaggle/working/TS_5_4.zarr"


import zarr
import numpy as np
import matplotlib.pyplot as plt

kaggle = zarr.open(kaggle_original_tomogram_path, mode="r")["0"][:]
portal = zarr.open(data_porta_denoised_tomogram_path, mode="r")["0"][:]

# middle slice
idx = kaggle.shape[0] // 2

slice_kaggle = kaggle[idx]
slice_portal = portal[idx]

# rotate portal tomogram by 180 degrees 
slice_portal_rotated = np.rot90(slice_portal, 2)


plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.imshow(slice_kaggle, cmap="gray")
plt.title("Kaggle Tomogram (middle slice)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(slice_portal, cmap="gray")
plt.title("Data Portal Tomogram (middle slice)")
plt.axis("off")

plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(slice_portal_rotated, cmap="gray")
plt.title("Data Portal Rotated 180째")
plt.axis("off")
plt.show()


plt.figure(figsize=(12, 5))
plt.hist(slice_kaggle.flatten(), bins=100, alpha=0.5, label="Kaggle", density=True)
plt.hist(slice_portal.flatten(), bins=100, alpha=0.2, label="Portal", density=True) # same histogram 
plt.hist(slice_portal_rotated.flatten(), bins=100, alpha=0.5, label="Portal Rot 180째", density=True) # same histogram
plt.title("Intensity Histograms Comparison")
plt.legend()
plt.show()

print("Kaggle tomogram intensity range:", kaggle.min(), kaggle.max())
print("Portal tomogram intensity range:", portal.min(), portal.max())

