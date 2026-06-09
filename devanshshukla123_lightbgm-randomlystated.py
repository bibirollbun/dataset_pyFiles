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


train=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
train.shape


train.columns



train.isnull().mean()
features=['id', 'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
       'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
       'TrackDurationMs', 'Energy']


test.shape



from sklearn.model_selection import train_test_split
x=train[features]
y=train['BeatsPerMinute']



x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.33, random_state=42)


# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# # x_test = std.transform(x_test)



x_test.shape



print("features --------- correlation")
for i in features:
    print(f"{i} --------- {train[i].corr(train['BeatsPerMinute'])} ")


from lightgbm import LGBMRegressor
lgbm = LGBMRegressor(     
    learning_rate=0.008,       
    max_depth=6,                         
    random_state=123,
    n_jobs=-1          
)






lgbm.fit(x_train,y_train)
pred=lgbm.predict(x_test)



from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, pred))
print("RMSE:", rmse)



submission = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': lgbm.predict(test[features])})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Done!")

