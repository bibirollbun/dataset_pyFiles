import os
import pydicom
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import numpy as np
import seaborn as sns



seg_path = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/1.2.826.0.1.3680043.8.498.10759842474698331813589731619457567641/1.2.826.0.1.3680043.8.498.10759842474698331813589731619457567641.nii"
seg = nib.load(seg_path)
seg_data = seg.get_fdata()

# Show one slice with mask
plt.imshow(seg_data[:, :, seg_data.shape[2] // 2], cmap='gray')
plt.title("Middle slice of aneurysm mask from segmentation folder (.nii)")
plt.show()




study_uid = "1.2.826.0.1.3680043.8.498.10009383108068795488741533244914370182"
folder = f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{study_uid}"

# Load and sort all .dcm slices
slices = []
for file in os.listdir(folder):
    dcm = pydicom.dcmread(os.path.join(folder, file))
    slices.append(dcm)

# Sort by ImagePositionPatient[2] (z-axis)
slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

# Stack into a 3D volume
volume = np.stack([s.pixel_array for s in slices])

# Show a middle slice
plt.imshow(volume[len(volume)//2], cmap='gray')
plt.title("Middle slice of 3D volume from series folder (.dcm)")
plt.axis('off')
plt.show()



series_id = "1.2.826.0.1.3680043.8.498.10491885999343016971277789732392506995"
folder = f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{series_id}"
localizer_csv = "/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv"

# Load the localizers CSV
localizers_df = pd.read_csv(localizer_csv)

# Filter localizers for our series
localizers_series = localizers_df[localizers_df['SeriesInstanceUID'] == series_id]

if localizers_series.empty:
    print("No localizer data found for this series.")
else:
    first_localizer = localizers_series.iloc[0]
    sop_uid = first_localizer['SOPInstanceUID']
    coord = eval(first_localizer['coordinates'])  # convert string dict to actual dict

    slices = [pydicom.dcmread(os.path.join(folder, f)) for f in os.listdir(folder)]
    target_slice = None
    for s in slices:
        if s.SOPInstanceUID == sop_uid:
            target_slice = s
            break

    if target_slice is None:
        print("Slice with given SOPInstanceUID not found.")
    else:
        img = target_slice.pixel_array
        plt.imshow(img, cmap='gray')
        plt.annotate('Aneurysm', xy=(coord['x'], coord['y']),
                     xytext=(coord['x'] + 10, coord['y'] - 10),
                     arrowprops=dict(color='red', lw=1, shrink=0.05, headwidth=5, headlength=7))
        plt.title(f"Annotated DICOM slice\nSeries: {series_id}")
        plt.axis('off')
        plt.show()



import pandas as pd
data = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
data.head()


data1 = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv")
data1.head()


# Count how many cases have aneurysm present
num_positive =  data['Aneurysm Present'].sum()
print("Number of positive (aneurysm present) cases:", num_positive)




data['PatientAge'] = data['PatientAge'].astype(int)
plt.figure(figsize=(10,6))
sns.histplot(data['PatientAge'], bins=20, kde=False, color=sns.color_palette("rocket")[4])  
plt.xlabel('Patient Age')
plt.ylabel('Count')
plt.title('Distribution of Patient Age')
plt.show()



colors = sns.color_palette("pastel")  # soft pastel colors

pd.crosstab(data['PatientSex'], data['Aneurysm Present']).plot(
    kind='bar',
    stacked=True,
    color=colors,
    figsize=(8,6)
)

plt.xlabel('Patient Sex')
plt.ylabel('Count')
plt.title('Aneurysm Presence by Patient Sex')
plt.legend(title='Aneurysm Present', labels=['No', 'Yes'])
plt.xticks(rotation=0)
plt.show()




#Different modality present in the data 
data['Modality'].value_counts().plot(kind='bar')


modality_counts = data.groupby('SeriesInstanceUID')['Modality'].nunique()
uids_with_multiple_modalities = modality_counts[modality_counts > 1]
print(f"Number of UIDs with multiple modalities: {len(uids_with_multiple_modalities)}")


location_cols = data.columns[4:-1]
data[location_cols].sum().sort_values(ascending=False).plot(kind='bar')



#Some patients may have aneurysms in more than one location.
data['location_count'] = data[location_cols].sum(axis=1)
data['location_count'].value_counts().sort_index().plot(kind='bar')
plt.title("Number of Aneurysm Locations per Patient")
plt.xlabel("Number of Locations with Aneurysm")
plt.ylabel("Number of Patients")
plt.show()


