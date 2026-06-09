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


#Import pandas
import pandas as pd
#install autogluon library
!python -m pip install --upgrade pip
!python -m pip install autogluon
from autogluon.tabular import TabularDataset, TabularPredictor


#Load training data
train = pd.read_csv('/kaggle/input/molecular-machine-learning/train.csv')

#Keeping only important variables from Competition Details
#features = ['T80','TDOS4.0', 'NumHeteroatoms', 'Mass']
#Keeping only variables that I found to be important in another notebook
features = ['T80','PrimeState', 'SDOS4.5', 'SDOS2.5', 'O19', 'O1', 'SurfaceCharge',
       'SDOS3.7', 'SDOS2.6', 'TDOS1.5', 'O10', 'TDOS1.6', 'O12',
       'LUMO(eV)', 'HAcceptors', 'SDOS2.7']

train_X = train[features]

#train.info()
#train.describe()
#train.head()


#Create label for target variable
label = 'T80'


#Create model - time_limit should be changed to whatever amount of minutes you want to train the model for.
predictor = TabularPredictor(label = label).fit(train_X, time_limit = 300)


#Evaluate model performance
predictor.evaluate(train, silent=True)


#view model leaderboard
predictor.leaderboard(train)


#load test data
test = pd.read_csv('/kaggle/input/molecular-machine-learning/test.csv')

#Create predictions using test data
y_pred = predictor.predict(test)


#Create output for competitions
output = pd.DataFrame({'Batch_ID': test.Batch_ID,
                       'T80': y_pred})
output.to_csv('submission.csv', index=False)

