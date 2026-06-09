# This Python environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# he example, here's several helpful packages to load

#the numpy as np # linear algebra
the pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)(kiri)
#my project is a one month work 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# my can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# my can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

