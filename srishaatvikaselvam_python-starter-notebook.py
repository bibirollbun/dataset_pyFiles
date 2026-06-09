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
df = pd.read_csv('/kaggle/input/orange-2/train.csv')

print(df.head())
df.info()
 


test = pd.read_csv("/kaggle/input/orange-2/test.csv")
test.head()


 #Your Answer


 #Your Answer


 #Your Answer 


  #Your Answer


 #Your Answer


 #Your Answer 


#Hint: Markov chains require continuos discreete values. How will you handle that? (Looking into forward fill and interpolation techniques)
#Your Answer


 #Your Answer 


  #Your Answer


  #Your Answer


  #Your Answer


 #Your Answer


 #Your Answer


 #Your Answer


#Your answer


  #Your Answer


#Your answer


submission = pd.DataFrame()
sample_submission = pd.read_csv('/kaggle/input/orange-2/sample_submission.csv')


submission['date'] = sample_submission['date']
submission['forecasted_open'] = sample_submission['forecasted_open']
submission['forecasted_close'] = sample_submission['forecasted_close']


submission.head()

#Convert to a csv file and name it submission.csv
#submission.to_csv('submission.csv', index = False)

