import shutil
import os

base = "/kaggle/input/rsna-intracranial-aneurysm-detection"

# List of SeriesInstanceId to download
series_to_download = [
    "1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647",
    "1.2.826.0.1.3680043.8.498.10004684224894397679901841656954650085",
    "1.2.826.0.1.3680043.8.498.10005158603912009425635473100344077317",   
]

# Working directory
temp_dir = "/kaggle/tempdir"
out_zip = "/kaggle/working/package.zip"

os.makedirs(temp_dir, exist_ok=True)


# Copy the series folders into the temporary directory
for s in series_to_download:
    src = os.path.join(base, "series", s)
    dst = os.path.join(temp_dir, "series", s)
    shutil.copytree(src, dst)

# Copy the train.csv file into the temporary directory
shutil.copy(
    os.path.join(base, "train.csv"),
    os.path.join(temp_dir, "train.csv")
)


shutil.make_archive(base_name=out_zip.replace(".zip", ""), format="zip", root_dir=temp_dir)
print(f"Created {out_zip}")

