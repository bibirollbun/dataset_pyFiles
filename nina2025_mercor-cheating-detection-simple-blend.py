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


df1 = pd.read_csv('/kaggle/input/mercor-cheating-detection-h-blend/submission.csv')
df2 = pd.read_csv('/kaggle/input/mercor-cheating-detection-ensemble-1570000/submission.csv')

df = pd.read_csv('/kaggle/input/mercor-cheating-detection/sample_submission.csv')


# df['prediction'] = df1['prediction'] *0.30+0.70* df2['prediction'] # LB = -1572785.00000
# df['prediction'] = df1['prediction'] *0.20+0.80* df2['prediction'] # LB = -1574440.00000
# df['prediction'] = df1['prediction'] *0.40+0.60* df2['prediction'] # LB = -1573540.00000
# df['prediction'] = df1['prediction'] *0.50+0.50* df2['prediction'] # LB = -1573165.00000
# df['prediction'] = df1['prediction'] *0.55+0.45* df2['prediction'] # LB = -1571990.00000
# df['prediction'] = df1['prediction'] *0.60+0.40* df2['prediction'] # LB = -1571680.00000
# df['prediction'] = df1['prediction'] *0.65+0.35* df2['prediction'] # LB = -1571750.00000
# df['prediction'] = df1['prediction'] *0.62+0.38* df2['prediction'] # LB = -1571655.00000

# df['prediction'] = df1['prediction']*0.626+0.374*df2['prediction'] # LB = -1572105.00000


df['prediction']  = df1['prediction'] *0.61 + 0.39* df2['prediction'] # LB =


df.to_csv('submission.csv',index=False)
df

