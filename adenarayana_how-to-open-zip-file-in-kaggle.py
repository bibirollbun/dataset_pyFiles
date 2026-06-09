# import the zipfile library
import zipfile


# Zipfile location
zip_file_directory = "/kaggle/input/dogs-vs-cats/train.zip"

# Extracted file location
extract_directory = "/kaggle/working/"


# Extract the zipfile
with zipfile.ZipFile(zip_file_directory, "r") as files:
    files.extractall(extract_directory)


# Get the files path

# import os library
import os

# list all the files in "train folder"
list_files = os.listdir("/kaggle/working/train/")

# create a list of each files, so we can access/open it
files_path = []
for f in list_files:
    files_path.append(os.path.join("/kaggle/working/train/", f))

# preview the first five files path
files_path[:5]


# randomly preview one file
from PIL import Image
import random

random_image_path = random.choice(files_path)
img = Image.open(random_image_path)
img

