# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
pd.set_option('display.max_columns', None)
train.tail(3)


#By Yulya Odintsova https://www.kaggle.com/code/yulyaodintsova/eda-logistic-regression-lgbmclassifier

train_info = pd.DataFrame({
    "DataType": train.dtypes,
    "MissingValues": train.isnull().sum(),
    "UniqueValues": train.nunique()
}).sort_values(by="MissingValues", ascending=False)

train_info['MissingValuesRatio'] = round(train_info['MissingValues'] / train.shape[0] ,2)

train_info


train.describe().loc[['mean','min','max']].T


#Numerical List
#only Integer: list(df.select_dtypes(include='int64').columns)
#Below could be df.select_dtypes(include=[np.number])

list(train.select_dtypes(include=['int64', 'float64']).columns)


numerical_cols = ['video_id','mouse1_id','mouse2_id', 'mouse3_id','mouse4_id',
 'frames_per_second','video_duration_sec','pix_per_cm_approx','video_width_pix',
 'video_height_pix','arena_width_cm','arena_height_cm']


# OutlierPandas https://www.kaggle.com/code/abhyudaya456/s5e6-eda-for-predicting-optimal-fertilizers/notebook 
plt.figure(figsize=(10,6))
sns.heatmap(train[numerical_cols].corr(), annot=True, cmap='summer')
plt.title("Correlation among Numerical Features")
plt.show()


list(train.select_dtypes(include='object').columns)


#By H-Z-Ning  https://www.kaggle.com/code/hzning/top-10-solution-0-97525-esay-is-all-you

categorical_columns = ["mouse1_sex", "mouse2_sex","mouse2_strain", "mouse1_color", "mouse2_color","mouse3_color", "mouse4_color", "arena_shape", "arena_type"]

plt.figure(figsize=(14, 12))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=column, data=train, palette='Set2')
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


train['tracking_method'].value_counts()


labels = 'custom HRnet', 'MARS', 'DeepLabCut', 'SLEAP'
sizes = [7926, 528, 211, 124]  #must have same number labels, sizes and explode
explode = (0, 0.2, 0, 0)  # only "explode" the 2nd slice 

fig1, ax1 = plt.subplots(figsize=(6,6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Tracking Methods')
plt.show()


#Code by Ducky https://www.kaggle.com/code/illgamhoduck/nfl-starter-eda

ENV_DIR = '../input'
DATA_DIR = f'{ENV_DIR}/elif-mars'


#Code by Ducky https://www.kaggle.com/code/illgamhoduck/nfl-starter-eda

from IPython.display import Video, display

def video(video_path, ratio=0.7):
    nfl_video = Video(f"{DATA_DIR}/{video_path}",
                      embed=True,
                      height=int(720 * ratio),
                      width=int(1280 * ratio))
    return nfl_video
    
video('elife-63720-video2 (1).mp4')


test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
test.tail()


sub = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/sample_submission.csv')
sub.tail()


#Read One parquet file. 
df = pd.read_parquet("../input/MABe-mouse-behavior-detection/train_tracking/DeliriousFly/1649549863.parquet")
df.tail()

