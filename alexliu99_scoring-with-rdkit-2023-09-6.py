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


import rdkit
print(rdkit.__version__)


import pandas as pd
sub = pd.read_csv('../input/neurips-open-polymer-prediction-2025/test.csv')
sub['Tg'] = 38.5
sub['FFV'] = 0.365
sub['Tc'] = 0.24
sub['Density'] = 0.96
sub['Rg'] = 13.47

from rdkit import Chem
for row in sub.itertuples():
    assert Chem.MolFromSmiles(row.SMILES)
    
del sub['SMILES']
sub.to_csv('submission.csv', index=False)
print(sub)




