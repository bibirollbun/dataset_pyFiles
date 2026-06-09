# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#JSON
import json


#df = pd.read_json('../input/cusersmarildownloadsdata-json/data.json', encoding = 'utf-8-sig')

#input_df = pd.read_json('../input/traffic-violations/data.json', lines=True, orient="columns")

#data= pd.read_json('../input/traffic-violations/data.json', lines=True)

#data= pd.read_json('../input/traffic-violations/data.json', lines=True, orient='records')

#data= pd.read_json('../input/traffic-violations/data.json', orient=str)

#data= pd.read_json('../input/traffic-violations/data.json', typ="series")

#df = pd.read_json(path_or_buf='/kaggle/input/en-to-ko/EN_TO_KO.json')


data= pd.read_json('../input/google-code-golf-2025/task221.json', lines=True)
data.head()


df= pd.read_json('../input/google-code-golf-2025/task292.json', typ="series")


df.to_json('new_file.json')


df.tail()


# Convert a JSON File to a CSV File

import csv

with open('/kaggle/input/google-code-golf-2025/task292.json') as json_file:
    jsondata = json.load(json_file)


f2 = open('../input/google-code-golf-2025/task001.json')
golf = json.load(f2)

golf['train']


try:
    with open('/kaggle/input/google-code-golf-2025/code_golf_utils.py', 'r') as file:
        content = file.read()
        print(content)
except FileNotFoundError:
    print("Error: /kaggle/input/google-code-golf-2025/code_golf_utils.py not found.")
except Exception as e:
    print(f"An error occurred: {e}")


#by Michael D. Moffitt https://www.kaggle.com/code/mmoffitt/neurips-2025-google-code-golf-championship/notebook

import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *
show_legend()


task_num = 2  # trying task 2?? 


#by Michael D. Moffitt https://www.kaggle.com/code/mmoffitt/neurips-2025-google-code-golf-championship/notebook

examples = load_examples(task_num)
show_examples(examples['arc-gen'])


task_num = 0


#by Michael D. Moffitt https://www.kaggle.com/code/mmoffitt/neurips-2025-google-code-golf-championship/notebook

examples = load_examples(task_num)
show_examples(examples['arc-gen'])

