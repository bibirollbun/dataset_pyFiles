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


df = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

t = 'loan_paid_back'

df[t] = \
 pd.read_csv('/kaggle/input/17-11-2025-ps-s5e11/submission 7.a.csv')[t] * 0.98 + \
 pd.read_csv('/kaggle/input/17-11-2025-ps-s5e11/submission 7.b.csv')[t] * 0.005 + \
 pd.read_csv('/kaggle/input/17-11-2025-ps-s5e11/submission 7.c.csv')[t] * 0.005 + \
 pd.read_csv('/kaggle/input/17-11-2025-ps-s5e11/submission 4.e.csv')[t] * 0.005 + \
 pd.read_csv('/kaggle/input/17-11-2025-ps-s5e11/submission 5.d.csv')[t] * 0.005

df.to_csv('submission.csv',index=False)
df

