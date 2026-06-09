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
        os.path.join(dirname, filename)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install plotly


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


df=pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')


df.head()


df.columns


sns.histplot(df["Modality"], color='purple', multiple='dodge')


sns.histplot(data= df, x="PatientAge", 
             hue="PatientSex", 
             kde=True, 
             bins=30, 
             multiple='stack', 
             palette='Set2', 
             edgecolor='black')


fig = px.pie(
    df[df["Aneurysm Present"]==1],
    names="PatientSex",
    title="Patient Sex Distribution",
    color="PatientSex",  # This links to the color map
    color_discrete_map={
        "Male": "lightblue",
        "Female": "pink"
    }
)

fig.show(renderer='iframe_connected')

