# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
print("data loaded")


train_data.info()


test_data.info()


train_data.describe()


test_data.describe()


robust_cols = ['AudioLoudness', 
               'VocalContent', 
               'AcousticQuality', 
               'InstrumentalScore', 
               'LivePerformanceLikelihood']
st_cols = ['AudioLoudness', 
           'VocalContent', 
           'AcousticQuality', 
           'InstrumentalScore', 
           'LivePerformanceLikelihood',
           'RhythmScore',
           'MoodScore',
           'TrackDurationMs']
robust_scale_transform = ColumnTransformer(
    transformers=[
        ('robust_transform', RobustScaler(), robust_cols),
        ('standard_scaler', StandardScaler(), st_cols)
    ],
    remainder='passthrough'
)


pipeline_w_model = Pipeline(
    steps=[
        ('scaling', robust_scale_transform),
        ('hist_grad_model', HistGradientBoostingRegressor(random_state=44))
    ]
)


X = train_data.drop(columns=['id', 'BeatsPerMinute'])
y = train_data['BeatsPerMinute']
val_X = test_data.drop(columns=['id'])


pipeline_w_model.fit(X,y)


sub_preds = pipeline_w_model.predict(val_X)


sub_ids = test_data['id']
data_dict = {
    'id': sub_ids,
    'BeatsPerMinute': sub_preds
}
data = pd.DataFrame(data_dict)


data.to_csv("submission.csv", index=False)

