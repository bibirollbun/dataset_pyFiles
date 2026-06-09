# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pathlib
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data_dir = Path.cwd().parent/'input' / 'geology-forecast-challenge-open' /'data'


data_dir


train_data = pd.read_csv(data_dir / "train.csv", index_col="geology_id")
test_data = pd.read_csv(data_dir / "test.csv", index_col="geology_id")
sample_sub = pd.read_csv(data_dir / "sample_submission.csv", index_col="geology_id")


overall_average=train_data.mean().mean()
X_Train = train_data.iloc[:,:300].fillna(overall_average)
X_Test = test_data.fillna(overall_average)
y_train = train_data.iloc[:, 300:]


firstReg = KNeighborsRegressor(p=1, n_neighbors=3)
firstReg.fit(X=X_Train, y=y_train)



output = pd.DataFrame(firstReg.predict(X_Test), columns = sample_sub.columns, index=sample_sub.index)
output.to_csv( "submission.csv")




