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

# Show files in the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



file_path = '/kaggle/input/agents-intensive-capstone-project/Hackathon dataset.txt'

with open(file_path, 'r', encoding='utf-8') as f:
    text_data = f.read()

text_data[:500]   # Show first 500 characters



def simple_summarizer(text):
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # simple logic: take the first 3 sentences
    summary = ". ".join(sentences[:3])
    return summary + "."



summary = simple_summarizer(text_data)
summary
with open('/kaggle/working/summary_output.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print("Summary saved as summary_output.txt")


