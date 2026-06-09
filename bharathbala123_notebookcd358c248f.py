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


submission = pd.DataFrame({
    'ID': ['img001', 'img002', 'img003'],
    'Aneurysm Present': [0.85, 0.01, 0.92],
    'label_1': [0.1, 0.03, 0.02],
    'label_2': [0.05, 0.02, 0.07],
    'label_3': [0.03, 0.04, 0.06],
    'label_4': [0.09, 0.05, 0.11],
    'label_5': [0.07, 0.11, 0.03],
    'label_6': [0.02, 0.06, 0.09],
    'label_7': [0.05, 0.03, 0.12],
    'label_8': [0.15, 0.12, 0.08],
    'label_9': [0.01, 0.01, 0.02],
    'label_10': [0.03, 0.05, 0.09],
    'label_11': [0.10, 0.03, 0.07],
    'label_12': [0.02, 0.12, 0.10],
    'label_13': [0.08, 0.06, 0.05]
})

