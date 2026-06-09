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


import sklearn   
print(sklearn.__version__)


from sklearn import svm
from sklearn import model_selection


data_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
data_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


data_train.head()



data_test.head()


data_test.fillna(0, inplace=True)


data_train_scale = (data_train - data_train.min())/(data_train.max()-data_train.min())
data_train_scale.head


x = data_train_scale[['pressure',	'maxtemp',	'temparature',	'mintemp',	'dewpoint',	'humidity',	'cloud',	'sunshine',	'winddirection',	'windspeed']]
y = data_train_scale['rainfall']


h1 = svm.LinearSVC(C=20)
h1.fit(x,y)
h1.score(x,y)


x_test = data_test[['pressure',	'maxtemp',	'temparature',	'mintemp',	'dewpoint',	'humidity',	'cloud',	'sunshine',	'winddirection',	'windspeed']]


y_pred = h1.predict(x_test)


my_submission = pd.DataFrame({'id': data_test.id, 'rainfall': y_pred})
my_submission.to_csv('submission-rain.csv', index=False)

print("Submission file created.")


my_submission.head


from IPython.display import HTML
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


create_download_link(my_submission)


create_download_link(data_test)

