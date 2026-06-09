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


#upload the file, it's csv file (means it's saved in excel file as extension csv
MentalHealth_test = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")


MentalHealth_TestDemograp= pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")


MentalHealth_test.dtypes


MentalHealth_test.isna().sum()


MentalHealth_TestDemograp.isna().sum()


MentalHealth_TestDemograp.dtypes


MentalHealth_TestDemograp.head()


# Subject is the column name which exits on both dataset which is the primary key in 
# the mental health test deompgraph and subject coloumn is unique identifier

# the how is the used as a join so used as a "left join"

MentalHealth_test = MentalHealth_test.merge(MentalHealth_TestDemograp, on ="subject", how = "left")


MentalHealth_test.isna().sum()


#  I USE THE FORWARD FILL IN THIS TIME SERIES DATASET 

MentalHealth_test.fillna(method="ffill", inplace=True)







