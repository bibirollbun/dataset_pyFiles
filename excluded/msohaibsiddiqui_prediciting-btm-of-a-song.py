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


File_Path = '/kaggle/input/playground-series-s5e9/train.csv'
Btm_Train_Data = pd.read_csv(File_Path)


train_x = Btm_Train_Data.drop(columns=["BeatsPerMinute"])
x = pd.get_dummies(train_x)
y = Btm_Train_Data['BeatsPerMinute']


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x, y)



File_Path_Two = '/kaggle/input/playground-series-s5e9/test.csv'
Btm_Test_Data = pd.read_csv(File_Path_Two)
Encoded_Features_Test = pd.get_dummies(Btm_Test_Data)
Features_test = [['id', 'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
       'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
       'TrackDurationMs', 'Energy']]
predictions = model.predict(Encoded_Features_Test)
print(predictions)


submission = pd.DataFrame({
    "id": Btm_Test_Data["id"],
    "BeatsPerMinute": predictions
})

submission.to_csv("submission.csv", index=False)

print("✅ submission.csv file created!")

