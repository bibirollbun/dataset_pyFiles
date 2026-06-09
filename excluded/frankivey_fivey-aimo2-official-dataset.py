# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import shutil

# Define the input and output directories
input_dir = '/kaggle/input'
output_dir = '/kaggle/working/output'  # You can change this to your desired output location

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate through all files and directories in the input directory
for dirname, _, filenames in os.walk(input_dir):
    # Create the corresponding directory in the output directory
    relative_path = os.path.relpath(dirname, input_dir)
    output_dirname = os.path.join(output_dir, relative_path)
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)

    # Copy the files to the output directory
    for filename in filenames:
        input_filepath = os.path.join(dirname, filename)
        output_filepath = os.path.join(output_dirname, filename)
        shutil.copy2(input_filepath, output_filepath)  # Use copy2 to preserve metadata

print("Files copied successfully!")

