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


!pip install --no-index --no-deps /kaggle/input/rdkit2025new/wheelhouse/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
import numpy as np

df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')

df.info()


df.describe()


import matplotlib.pyplot as plt
import seaborn as sns
columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


scols = int(len(columns)/2)
srows = 3
fig, axes = plt.subplots(scols, srows, figsize=(10,6))

for i, col in enumerate(columns):
    ax_col = int(i%scols)
    ax_row = int(i/scols)
    
    sns.histplot(data=df, x=col, kde=True, ax=axes[ax_col, ax_row])
    axes[ax_col, ax_row].set_title('Frequency distribution '+ col, fontsize=12)
    axes[ax_col, ax_row].set_xlabel(col, fontsize=8)
    axes[ax_col, ax_row].set_ylabel('Count', fontsize=8)
fig.tight_layout()
plt.show()


sns.pairplot(df[columns])


fig, axs = plt.subplots(5, 2, figsize=(10, 15))


for i, col in enumerate(columns):
    
   
    sns.histplot(data=df, x=col, kde=True, ax=axs[i, 0])
    axs[i, 0].set_title(f'Distribution of {col}') 
    
   
    df.boxplot(column=[col], ax=axs[i, 1])
    axs[i, 1].set_title(f'Boxplot of {col}') 

plt.tight_layout()
plt.show()




