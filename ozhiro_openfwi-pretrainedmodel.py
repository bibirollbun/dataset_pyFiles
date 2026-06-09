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


!git clone https://github.com/lanl/OpenFWI


#FlatVel
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/fva_l1.pth'
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/fva_l2.pth'
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/fvb_l1.pth'
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/fvb_l2.pth'

#FlatFault
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/ffa_l1.pth'
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/ffa_l2.pth'
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/ffb_l1.pth'
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/ffb_l2.pth'

#CurveVel
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/cva_l1.pth'
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/cva_l2.pth'
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/cvb_l1.pth'
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/cvb_l2.pth'

#CurveFault
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/cfa_l1.pth'
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/cfa_l2.pth'
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/cfb_l1.pth'
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/cfb_l2.pth'

#Style
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/sta_l1_new.pth'
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/sta_l2.pth'
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/stb_l1.pth'
! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/stb_l2.pth'




