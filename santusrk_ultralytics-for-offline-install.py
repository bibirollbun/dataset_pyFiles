import os
import subprocess
import shutil

# Define download directory
download_dir = "ultralytics_offline"
os.makedirs(download_dir, exist_ok=True)

# Download ultralytics and its dependencies
subprocess.run(f"pip download ultralytics -d {download_dir}", shell=True, check=True)

# Zip the downloaded packages
shutil.make_archive("ultralytics_offline", 'zip', download_dir)

print("✅ Packages downloaded and zipped successfully.")





