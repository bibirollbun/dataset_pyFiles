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
import matplotlib.pyplot as plt
import seaborn as sns

import os
import polars as pl

import kaggle_evaluation.cmi_inference_server


train_data = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
test_data = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")


# Shape of the data:
print("train_data :", train_data.shape)
print("train :", train.shape)
print("test_data :", test_data.shape)
print("test :", test.shape)


train_data.head()


train_data['gesture'].value_counts()


train_data['sequence_id'].value_counts()


train.head()


test_data.head()


test.head()


train_data = train_data.merge(train,on='subject',how='left')
test_data = test_data.merge(test,on='subject',how='left')
train_data.shape, test_data.shape


# Encode target
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
target_le = LabelEncoder()
train_data['gesture'] = target_le.fit_transform(train_data['gesture'])


#cols = test_data.columns
cols = test_data.columns.tolist()
len(cols)


from sklearn.model_selection import train_test_split
X = train_data[cols]
X = train_data.drop(['gesture', 'phase', 'behavior', 'orientation', 'sequence_type', 'sequence_id','row_id','sequence_id','subject'],axis=1)
y = train_data['gesture']
test = test_data.drop(['sequence_id','row_id','sequence_id','subject'],axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


from lightgbm import LGBMClassifier
model = LGBMClassifier(verbosity=-1)
model.fit(X,y)
preds = model.predict(test)

predictions = target_le.inverse_transform(preds)


import joblib

# Save trained model and label encoder
joblib.dump(model, 'model.pkl')
joblib.dump(target_le, 'label_encoder.pkl')


def predict(sequence: pd.DataFrame, demographics: pd.DataFrame) -> str:
    # Convert from Polars to Pandas if needed
    if not isinstance(sequence, pd.DataFrame):
        sequence = sequence.to_pandas()
    if not isinstance(demographics, pd.DataFrame):
        demographics = demographics.to_pandas()
    
    # Merge with demographics
    sequence = sequence.merge(demographics, on="subject", how="left")
    
    # Store sequence_id if it exists
    sequence_id = None
    if 'sequence_id' in sequence.columns:
        sequence_id = sequence['sequence_id'].iloc[0]


     # Drop unnecessary columns but keep sequence_id for now
    drop_cols = ['row_id', 'subject','sequence_id']
    sequence = sequence.drop(columns=[col for col in drop_cols if col in sequence.columns])
    
    # Drop sequence_id before prediction if it exists
    if 'sequence_id' in sequence.columns:
        sequence = sequence.drop('sequence_id', axis=1)
    
    # Make prediction using our trained model
    prediction = model.predict(sequence)
    
    # Get the most common prediction (mode)
    mode_pred = int(pd.Series(prediction).mode()[0])


    return target_le.inverse_transform([mode_pred])[0]


import joblib

# Load pre-trained model and label encoder
model = joblib.load('/kaggle/input/model-data/model.pkl')
target_le = joblib.load('/kaggle/input/target-encoder-data/label_encoder.pkl')


inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

