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


import pandas as pd

df =pd.read_parquet('/kaggle/input/MABe-mouse-behavior-detection/train_tracking/ReflectiveManatee/315178669.parquet')
df.head()




import matplotlib.pyplot as plt 
x = df['x']
y = df['y']

plt.plot(x[:200],y[:200],marker='o',markersize=2,linestyle='-')
plt.title("Mouse's nose in 200 First frame")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()



print("coloumn:", len(df.columns))
for i, col in enumerate(df.columns[:50]):
    print(i, ":", repr(col))


print("\ncoloumns List:")
print([str(c) for c in df.columns[:200]])


display(df.head(3))


