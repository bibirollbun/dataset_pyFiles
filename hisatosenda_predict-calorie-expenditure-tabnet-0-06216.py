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


!pip install pytorch-tabnet
!pip install torchvision


%matplotlib inline
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
from mlxtend.plotting import heatmap

import numpy as np
import pandas as pd
import seaborn as sns

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import torchvision.transforms as transforms
from torchvision import models
from torchvision import datasets
from torchinfo import summary
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer

from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

from glob import glob
import cv2
# from tqdm.auto import tqdm
from tqdm import tqdm
from PIL import Image
import pathlib

import warnings
warnings.filterwarnings('ignore')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def seed_torch(seed=42):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.use_deterministic_algorithms = True


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


# Yeo-Johnson transformer
pt = PowerTransformer(method='yeo-johnson')

# Box-Cox transformer
# pt = PowerTransformer(method='box-cox')

train_df['Age_trans'] = train_df['Age'].copy()
test_df['Age_trans'] = test_df['Age'].copy()
train_df['Height_trans'] = train_df['Height'].copy()
test_df['Height_trans'] = test_df['Height'].copy()
train_df['Body_Temp_trans'] = train_df['Body_Temp'].copy()
test_df['Body_Temp_trans'] = test_df['Body_Temp'].copy()

trans_cols = ['Age_trans', 'Height_trans', 'Body_Temp_trans']

pt.fit(train_df[trans_cols])
train_df[trans_cols] = pt.transform(train_df[trans_cols])
test_df[trans_cols] = pt.transform(test_df[trans_cols])


all_df = pd.concat([train_df, test_df])


all_df.iloc[:,1:].describe().T


# LabelEncoding

all_df['Sex'] = all_df['Sex'].map({'male': 0, 'female': 1})


# Male

all_df.loc[all_df.Sex.eq(0), ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']].describe().T


# Female

all_df.loc[all_df.Sex.eq(1), ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']].describe().T


# Outlier: Numerical features of training data and test data

# Male
# male_num_cols = ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']

# all_p99_0 = all_df.loc[all_df.Sex.eq(0), male_num_cols].quantile(0.99)
# all_p01_0 = all_df.loc[all_df.Sex.eq(0), male_num_cols].quantile(0.01)


# Female
# female_num_cols = ['Weight', 'Heart_Rate', 'Body_Temp']

# all_p99_1 = all_df.loc[all_df.Sex.eq(1), female_num_cols].quantile(0.99)
# all_p01_1 = all_df.loc[all_df.Sex.eq(1), female_num_cols].quantile(0.01)
all_p99_1_height = all_df.loc[all_df.Sex.eq(1), 'Height'].quantile(0.99)
all_p01_1_height = all_df.loc[all_df.Sex.eq(1), 'Height'].quantile(0.01)

# all_df.loc[all_df.Sex.eq(0), male_num_cols] = all_df.loc[all_df.Sex.eq(0), male_num_cols].clip(all_p01_0, None, axis=1)
# all_df.loc[all_df.Sex.eq(1), female_num_cols] = all_df.loc[all_df.Sex.eq(1), female_num_cols].clip(all_p01_1, None, axis=1)
all_df.loc[all_df['Sex'] == 1, 'Height'] = all_df.loc[all_df['Sex'] == 1, 'Height'].clip(all_p01_1_height, all_p99_1_height)



# Bucket age

boundaries_age = torch.tensor([all_df.Age.quantile(0.25), 
                               all_df.Age.quantile(0.5),
                               all_df.Age.quantile(0.75)])
v = torch.tensor(all_df['Age'].values)

all_df['Age_Bucketed'] = torch.bucketize(v, boundaries_age, right=True)
all_df['Age_Bucketed'] = all_df['Age_Bucketed'].astype('object')


# Bucket Duration

boundaries_du = torch.tensor([all_df.Duration.quantile(0.25),
                              all_df.Duration.quantile(0.5),
                              all_df.Duration.quantile(0.75)])
v = torch.tensor(all_df['Duration'].values)

all_df['Duration_Bucketed'] = torch.bucketize(v, boundaries_du, right=True)
all_df['Duration_Bucketed'] = all_df['Duration_Bucketed'].astype('object')


# BMI: Body Mass Indes  
# healty values: 18.5~24.9

all_df['BMI'] = np.round(all_df['Weight'] / ((all_df['Height']/100)**2), 2)


# Bucket BMI　by gender

all_df['BMI_Bucketed'] = 0

# Male
boundaries_bmi = torch.tensor([all_df.loc[all_df.Sex.eq(0), 'BMI'].quantile(0.25), 
                               all_df.loc[all_df.Sex.eq(0), 'BMI'].quantile(0.5), 
                               all_df.loc[all_df.Sex.eq(0), 'BMI'].quantile(0.75)])
v = torch.tensor(all_df['BMI'].values)

all_df['BMI_0'] = torch.bucketize(v, boundaries_bmi, right=True)
all_df.loc[all_df.Sex.eq(0), 'BMI_Bucketed'] = all_df.loc[all_df.Sex.eq(0), 'BMI_0']


# Female
boundaries_bmi = torch.tensor([all_df.loc[all_df.Sex.eq(1), 'BMI'].quantile(0.25), 
                               all_df.loc[all_df.Sex.eq(1), 'BMI'].quantile(0.5), 
                               all_df.loc[all_df.Sex.eq(1), 'BMI'].quantile(0.75)])
v = torch.tensor(all_df['BMI'].values)

all_df['BMI_1'] = torch.bucketize(v, boundaries_bmi, right=True)
all_df.loc[all_df.Sex.eq(1), 'BMI_Bucketed'] = all_df.loc[all_df.Sex.eq(1), 'BMI_1']


all_df = all_df.drop(['BMI_0', 'BMI_1'], axis=1)
all_df['BMI_Bucketed'] = all_df['BMI_Bucketed'].astype('object')


# BMR: Basal metabolic rate
# Men: BMR = 88.362 + (13.397 x weight in kg) + (4.799 x height in cm) – (5.677 x age in years)
# Women: BMR = 447.593 + (9.247 x weight in kg) + (3.098 x height in cm) – (4.330 x age in years)

all_df['BMR'] = 0

all_df['bmr_weight_0'] = 13.397 * all_df['Weight']
all_df['bmr_height_0'] = 4.799 * all_df['Height']
all_df['bmr_age_0'] = 5.677 * all_df['Age']
all_df['bmr_0'] = np.round(all_df['bmr_weight_0'] + all_df['bmr_height_0'] - all_df['bmr_age_0'] + 88.362, 2)
all_df.loc[all_df['Sex'].eq(0), 'BMR'] = all_df.loc[all_df['Sex'].eq(0), 'bmr_0']

all_df['bmr_weight_1'] = 9.247 * all_df['Weight']
all_df['bmr_height_1'] = 3.098 * all_df['Height']
all_df['bmr_age_1'] = 4.33 * all_df['Age']
all_df['bmr_1'] = np.round(all_df['bmr_weight_1'] + all_df['bmr_height_1'] - all_df['bmr_age_1'] + 447.593, 2)
all_df.loc[all_df['Sex'].eq(1), 'BMR'] = all_df.loc[all_df['Sex'].eq(1), 'bmr_1']

all_df = all_df.drop(['bmr_weight_0','bmr_height_0', 'bmr_age_0', 'bmr_0', 
                      'bmr_weight_1','bmr_height_1', 'bmr_age_1', 'bmr_1'], axis=1)



# Bucket BMR by gender

all_df['BMR_Bucketed'] = 0

# Male
boundaries_bmr = torch.tensor([all_df.loc[all_df.Sex.eq(0), 'BMR'].quantile(0.25), 
                               all_df.loc[all_df.Sex.eq(0), 'BMR'].quantile(0.5), 
                               all_df.loc[all_df.Sex.eq(0), 'BMR'].quantile(0.75)])
v = torch.tensor(all_df['BMR'].values)

all_df['BMR_0'] = torch.bucketize(v, boundaries_bmr, right=True)
all_df.loc[all_df['Sex'].eq(0), 'BMR_Bucketed'] = all_df.loc[all_df['Sex'].eq(0), 'BMR_0']

# Female
boundaries_bmr = torch.tensor([all_df.loc[all_df.Sex.eq(1), 'BMR'].quantile(0.25), 
                               all_df.loc[all_df.Sex.eq(1), 'BMR'].quantile(0.5), 
                               all_df.loc[all_df.Sex.eq(1), 'BMR'].quantile(0.75)])
v = torch.tensor(all_df['BMR'].values)

# 1026.38~, 1296.1~, 1368.24~, 1443.42~
all_df['BMR_1'] = torch.bucketize(v, boundaries_bmr, right=True)
all_df.loc[all_df['Sex'].eq(1), 'BMR_Bucketed'] = all_df.loc[all_df['Sex'].eq(1), 'BMR_1']

all_df = all_df.drop(['BMR_0', 'BMR_1'], axis=1)
all_df['BMR_Bucketed'] = all_df['BMR_Bucketed'].astype('object')


# Healty_Weight = Height/100 * Height/100 * 22

all_df['Healty_Weight'] = np.round(((all_df['Height']/100) ** 2) * 22, 2)


# Bucket Healty Weight by gender

all_df['Healty_Weight_Bucketed'] = 0

# Male
boundaries_hel = torch.tensor([all_df.loc[all_df.Sex.eq(0), 'Healty_Weight'].quantile(0.25), 
                               all_df.loc[all_df.Sex.eq(0), 'Healty_Weight'].quantile(0.5), 
                               all_df.loc[all_df.Sex.eq(0), 'Healty_Weight'].quantile(0.75)])
v = torch.tensor(all_df['Healty_Weight'].values)

all_df['hel_0'] = torch.bucketize(v, boundaries_hel, right=True)
all_df.loc[all_df['Sex'].eq(0), 'Healty_Weight_Bucketed'] = all_df.loc[all_df['Sex'].eq(0), 'hel_0']


# Female
boundaries_hel = torch.tensor([all_df.loc[all_df.Sex.eq(1), 'Healty_Weight'].quantile(0.25), 
                               all_df.loc[all_df.Sex.eq(1), 'Healty_Weight'].quantile(0.5), 
                               all_df.loc[all_df.Sex.eq(1), 'Healty_Weight'].quantile(0.75)])
v = torch.tensor(all_df['Healty_Weight'].values)

all_df['hel_1'] = torch.bucketize(v, boundaries_hel, right=True)
all_df.loc[all_df['Sex'].eq(1), 'Healty_Weight_Bucketed'] = all_df.loc[all_df['Sex'].eq(1), 'hel_1']


all_df = all_df.drop(['hel_0', 'hel_1'], axis=1)
all_df['Healty_Weight_Bucketed'] = all_df['Healty_Weight_Bucketed'].astype('object')



# Difference between 'Weight' and 'Healty_Weight'

all_df['Diff_Weight'] = np.round(all_df['Weight'] - all_df['Healty_Weight'], 2)


# 'Over_Weight':

all_df['Over_Weight'] = all_df['Diff_Weight']
all_df.loc[all_df['Diff_Weight'] <=0, 'Over_Weight'] = 0


# Bucket Over_Weight by gender

all_df['Over_Weight_Bucketed'] = 0

# Male
boundaries_ow = torch.tensor([all_df.loc[all_df.Sex.eq(0), 'Over_Weight'].quantile(0.25), 
                              all_df.loc[all_df.Sex.eq(0), 'Over_Weight'].quantile(0.5), 
                              all_df.loc[all_df.Sex.eq(0), 'Over_Weight'].quantile(0.75)])
v = torch.tensor(all_df['Over_Weight'].values)

all_df['ow_0'] = torch.bucketize(v, boundaries_ow, right=True)
all_df.loc[all_df['Sex'].eq(0), 'Over_Weight_Bucketed'] = all_df.loc[all_df['Sex'].eq(0), 'ow_0']


# Female
boundaries_ow = torch.tensor([all_df.loc[all_df.Sex.eq(1), 'Over_Weight'].quantile(0.25), 
                              all_df.loc[all_df.Sex.eq(1), 'Over_Weight'].quantile(0.5), 
                              all_df.loc[all_df.Sex.eq(1), 'Over_Weight'].quantile(0.75)])
v = torch.tensor(all_df['Over_Weight'].values)

all_df['ow_1'] = torch.bucketize(v, boundaries_ow, right=True)
all_df.loc[all_df['Sex'].eq(1), 'Over_Weight_Bucketed'] = all_df.loc[all_df['Sex'].eq(1), 'ow_1']


all_df = all_df.drop(['ow_0', 'ow_1'], axis=1)
all_df['Over_Weight_Bucketed'] = all_df['Over_Weight_Bucketed'].astype('object')


# Height　by gender

all_df['Height_Bucketed'] = 0

# Male
boundaries_he = torch.tensor([all_df.loc[all_df.Sex.eq(0), 'Height'].quantile(0.25), 
                              all_df.loc[all_df.Sex.eq(0), 'Height'].quantile(0.5), 
                              all_df.loc[all_df.Sex.eq(0), 'Height'].quantile(0.75)])
v = torch.tensor(all_df['Height'].values)

all_df['he_0'] = torch.bucketize(v, boundaries_he, right=True)
all_df.loc[all_df.Sex.eq(0), 'Height_Bucketed'] = all_df.loc[all_df.Sex.eq(0), 'he_0']


# Female
boundaries_he = torch.tensor([all_df.loc[all_df.Sex.eq(1), 'Height'].quantile(0.25), 
                              all_df.loc[all_df.Sex.eq(1), 'Height'].quantile(0.5), 
                              all_df.loc[all_df.Sex.eq(1), 'Height'].quantile(0.75)])
v = torch.tensor(all_df['Height'].values)

all_df['he_1'] = torch.bucketize(v, boundaries_he, right=True)
all_df.loc[all_df.Sex.eq(1), 'Height_Bucketed'] = all_df.loc[all_df.Sex.eq(1), 'he_1']

all_df = all_df.drop(['he_0', 'he_1'], axis=1)
all_df['Height_Bucketed'] = all_df['Height_Bucketed'].astype('object')


# Weight　by gender

all_df['Weight_Bucketed'] = 0

# Male
boundaries_we = torch.tensor([all_df.loc[all_df.Sex.eq(0), 'Weight'].quantile(0.25), 
                              all_df.loc[all_df.Sex.eq(0), 'Weight'].quantile(0.5), 
                              all_df.loc[all_df.Sex.eq(0), 'Weight'].quantile(0.75)])
v = torch.tensor(all_df['Weight'].values)

all_df['we_0'] = torch.bucketize(v, boundaries_we, right=True)
all_df.loc[all_df.Sex.eq(0), 'Weight_Bucketed'] = all_df.loc[all_df.Sex.eq(0), 'we_0']


# Female
boundaries_we = torch.tensor([all_df.loc[all_df.Sex.eq(1), 'Weight'].quantile(0.25), 
                              all_df.loc[all_df.Sex.eq(1), 'Weight'].quantile(0.5), 
                              all_df.loc[all_df.Sex.eq(1), 'Weight'].quantile(0.75)])
v = torch.tensor(all_df['Weight'].values)

# 50~, 59~, 63~, 68~
all_df['we_1'] = torch.bucketize(v, boundaries_we, right=True)
all_df.loc[all_df.Sex.eq(1), 'Weight_Bucketed'] = all_df.loc[all_df.Sex.eq(1), 'we_1']

all_df = all_df.drop(['we_0', 'we_1'], axis=1)
all_df['Weight_Bucketed'] = all_df['Weight_Bucketed'].astype('object')



all_df['Height_cat'] = all_df['Height_Bucketed']
all_df['Height_cat'] = all_df['Height_cat'].astype('str')
all_df['Weight_cat'] = all_df['Weight_Bucketed']
all_df['Weight_cat'] = all_df['Weight_cat'].astype('str')

all_df['Height_Weight'] = all_df['Height_cat'] + '_' + all_df['Weight_cat']
all_df = all_df.drop(['Height_cat', 'Weight_cat'], axis=1)


# all_df['Height_Weight'] = all_df['Height_Weight'].astype('object')


male_mapping = {
    '0_3': 3,
    '0_2': 3,
    '0_1': 3,
    '2_3': 3,
    '1_2': 2, 
    '1_3': 2, 
    '1_1': 2, 
    '3_3': 2, 
    '2_2': 1,
    '2_1': 1,
    '0_0': 1, 
    '1_0': 1, 
    '3_2': 0, 
    '3_1': 0,
    '2_0': 0,
    '3_0': 0,
}
female_mapping = {
    '0_1': 3,
    '0_2': 3,
    '0_0': 3,
    '1_1': 3,
    '2_1': 2,
    '1_0': 2, 
    '1_2': 2, 
    '2_0': 2, 
    '2_2': 1, 
    '0_3': 1,
    '2_3': 1,
    '1_3': 1, 
    '3_3': 0, 
    '3_2': 0, 
    '3_1': 0, 
    '3_0': 0,
}

all_df.loc[all_df.Sex.eq(0), 'Height_Weight'] = all_df.loc[all_df.Sex.eq(0), 'Height_Weight'].map(male_mapping)
all_df.loc[all_df.Sex.eq(1), 'Height_Weight'] = all_df.loc[all_df.Sex.eq(1), 'Height_Weight'].map(female_mapping)
all_df['Height_Weight'] = all_df['Height_Weight'].astype('object')



all_df['Duration_cat'] = all_df['Duration_Bucketed']
all_df['Duration_cat'] = all_df['Duration_cat'].astype('str')
all_df['Weight_cat'] = all_df['Weight_Bucketed']
all_df['Weight_cat'] = all_df['Weight_cat'].astype('str')

all_df['Duration_Weight'] = all_df['Duration_cat'] + '_' + all_df['Weight_cat']
all_df = all_df.drop(['Duration_cat', 'Weight_cat'], axis=1)


# all_df['Duration_Weight'] = all_df['Duration_Weight'].astype('object')



male_mapping = {
    '3_1': 3,
    '3_0': 3,
    '3_2': 3,
    '3_3': 3,
    '2_3': 2, 
    '2_1': 2, 
    '2_2': 2, 
    '2_0': 2, 
    '1_3': 1,
    '1_2': 1,
    '1_1': 1, 
    '1_0': 1, 
    '0_3': 0, 
    '0_2': 0,
    '0_1': 0,
    '0_0': 0,
}
female_mapping = {
    '3_3': 3,
    '3_2': 3,
    '3_1': 3,
    '3_0': 3,
    '2_0': 2, 
    '2_1': 2, 
    '2_2': 2, 
    '2_3': 2, 
    '1_1': 1,
    '1_2': 1,
    '1_0': 1, 
    '1_3': 1, 
    '0_1': 0, 
    '0_0': 0,
    '0_2': 0,
    '0_3': 0,
}
all_df.loc[all_df.Sex.eq(0), 'Duration_Weight'] = all_df.loc[all_df.Sex.eq(0), 'Duration_Weight'].map(male_mapping)
all_df.loc[all_df.Sex.eq(1), 'Duration_Weight'] = all_df.loc[all_df.Sex.eq(1), 'Duration_Weight'].map(female_mapping)
all_df['Duration_Weight'] = all_df['Duration_Weight'].astype('object')



all_df['Age_times_Duration'] = np.round(all_df['Age'] * all_df['Duration'], 2)
all_df['Height_times_Duration'] = np.round(all_df['Height'] * all_df['Duration'], 2)
all_df['Heart_Rate_times_Duration'] = np.round(all_df['Heart_Rate'] * all_df['Duration'], 2)
all_df['Body_Temp_times_Duration'] = np.round(all_df['Body_Temp'] * all_df['Duration'], 2)
all_df['Height_times_Weight'] = np.round(all_df['Height'] * all_df['Weight'], 2)
all_df['Weight_times_Duration'] = np.round(all_df['Weight'] * all_df['Duration'], 2)
all_df['Body_Temp_times_Age_Bucketed'] = np.round(all_df['Body_Temp'] * (all_df['Age_Bucketed']).astype('float64'), 2)
all_df['Sex_times_Duration'] = np.round(all_df['Sex'] * all_df['Duration'], 2)


all_df['Age'] = all_df['Age'].astype('object')
all_df['Duration'] = all_df['Duration'].astype('object')


all_df.info()


train_df = all_df[:750000]
test_df = all_df.iloc[750000:].drop(['Calories'], axis=1)


# standard scaler
# sc = StandardScaler()

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

numeric_cols = [
    'Weight', 
    'Heart_Rate',
    'BMI', 'BMR', 
    'Healty_Weight', 
    'Diff_Weight', 'Over_Weight',
    'Age_times_Duration', 'Height_times_Duration',
    'Heart_Rate_times_Duration', 'Body_Temp_times_Duration',
    'Height_times_Weight', 
    'Weight_times_Duration', 'Body_Temp_times_Age_Bucketed',
    'Sex_times_Duration',
]

sc.fit(train_df[numeric_cols])
train_df[numeric_cols] = sc.transform(train_df[numeric_cols])
test_df[numeric_cols] = sc.transform(test_df[numeric_cols])



# del all_df
print(train_df.shape, test_df.shape)


X = train_df.drop(['id', 'Sex', 'Body_Temp', 'Height',
                   'Calories',
#                   'Age_Bucketed', 'Duration_Bucketed',
#                   'BMI_Bucketed', 'BMR_Bucketed', 
                   'Healty_Weight',
                   'Healty_Weight_Bucketed', 
#                   'Diff_Weight', 'Over_Weight',
                   'Over_Weight_Bucketed', 'Height_Bucketed', 'Weight_Bucketed',
                   'Height_Weight', 
#                   'Duration_Weight',
#                   'Age_times_Duration',
#                   'Height_times_Duration', 'Heart_Rate_times_Duration',
#                   'Body_Temp_times_Duration', 'Height_times_Weight',
#                   'Weight_times_Duration', 'Body_Temp_times_Age_Bucketed'
                   ],
                   axis=1)
 
y = train_df['Calories']

test_X = test_df.drop(['id', 'Sex', 'Body_Temp', 'Height',
#                       'Age_Bucketed', 'Duration_Bucketed',
#                       'BMI_Bucketed', 'BMR_Bucketed', 
                       'Healty_Weight',
                       'Healty_Weight_Bucketed', 
#                       'Diff_Weight', 'Over_Weight',
                       'Over_Weight_Bucketed', 'Height_Bucketed', 'Weight_Bucketed',
                       'Height_Weight', 
#                       'Duration_Weight', 
#                       'Age_times_Duration',
#                       'Height_times_Duration', 'Heart_Rate_times_Duration',
#                       'Body_Temp_times_Duration', 'Height_times_Weight',
#                       'Weight_times_Duration', 'Body_Temp_times_Age_Bucketed'
                      ],
                       axis=1)


X.info()


from sklearn.preprocessing import LabelEncoder

nunique = X.nunique()
types = X.dtypes

categorical_columns = []
categorical_dims =  {}

for col in X.columns[X.dtypes == object]:
    print(col, X[col].nunique())
    l_enc = LabelEncoder()
#    X[col] = X[col].fillna("VV_likely")
    X[col] = l_enc.fit_transform(X[col].values)
    test_X[col] = l_enc.fit_transform(test_X[col].values)
    categorical_columns.append(col)
    categorical_dims[col] = len(l_enc.classes_)

features = [col for col in X.columns]
cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
# cat_emb_dim = [i // 2 if i <= 30 else 16 for i in cat_dims ]
cat_emb_dim = [i // 2 for i in cat_dims ]


print(cat_idxs)
print(cat_dims)
print(cat_emb_dim)


# Split training data
#   X_train: 70%, X_valid: 15%, X_test: 15%

train_rate, val_rate, test_rate = 0.7, 0.15, 0.15

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_rate, random_state=42)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_test, y_test, test_size=test_rate / (test_rate + val_rate), random_state=42)


X_train = X_train.values
X_valid = X_valid.values
X_test = X_test.values
test_X = test_X.values

y_train = y_train.values.reshape(-1, 1)
y_valid = y_valid.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)
print(test_X.shape)


from pytorch_tabnet.pretraining import TabNetPretrainer

max_epochs = 100

# TabNetPretrainer
unsupervised_model = TabNetPretrainer(
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=cat_emb_dim,    
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax',             # "sparsemax",
    verbose=2,
)

unsupervised_model.fit(
    X_train=X_train,
    eval_set=[X_valid],
    max_epochs=max_epochs ,
    patience=5,      
    num_workers=0,
    drop_last=False,
    pretraining_ratio=0.5,    # 0.8
)



max_epochs = 100
# max_epochs=200

# n_d = 14
# n_steps = 3
# gamma = 1.3
# lambda_sparse = 3.752055855124284e-05
# n_shared = 2

# n_d=54
# n_steps=7
# gamma=1.7
# lambda_sparse=1e-06
# n_shared=1

n_d = 8
n_steps = 3
gamma = 1.3
# gamma = 1.2
n_shared = 2
lambda_sparse = 1e-3

tabnet_params = dict(
    n_d=n_d,
    n_a=n_d,
    n_steps=n_steps,
    gamma=gamma,
    n_shared=n_shared,
    cat_idxs=cat_idxs, 
    cat_dims=cat_dims, 
    cat_emb_dim=cat_emb_dim, 
    optimizer_fn=torch.optim.Adam,
#    optimizer_params=dict(lr=2e-2),
    optimizer_params=dict(lr=1e-2),
    scheduler_params={'step_size': 10,  # how to use learning rate scheduler
                      'gamma': 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax',                 # This will be overwritten if using pretrain model  'sparsemax', 'entmax'
    lambda_sparse=lambda_sparse,
    seed=42,
    verbose=2,
)


model_tabnet_7 = TabNetRegressor(**tabnet_params)

model_tabnet_7.fit(
    X_train=X_train,
    y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['rmsle'],
    from_unsupervised=unsupervised_model,
    patience=20,
    max_epochs=max_epochs,
)


y_pred = np.clip(model_tabnet_7.predict(X_test), a_min=1, a_max=None)

tmp1_rmsle = mean_squared_log_error(y_test, y_pred, squared=False)

print(f'RMSLE: {tmp1_rmsle}')



for param in ['train_rmsle', 'valid_rmsle']:
    plt.plot(model_tabnet_7.history[param])
    plt.xlabel('epoch')
    plt.ylabel(param)
    
    plt.grid()
    plt.show()


importance_tabnet = sorted([(i, n) for i, n in enumerate(model_tabnet_7.feature_importances_)],
                            key=lambda x: x[1], 
                            reverse=True)

print(importance_tabnet)


# 1step Age,Age_Bucketed, BMI, Over_Weight ,Body_Temp_times_Age_Bucketed
# 2step Duration , Heart_Rate_times_Duration, Body_Temp_times_Duration 
# step3 Height ,Weight , BMR_Bucketed, Height_times_Weight
# なし　Body_Temp Duration_Bucketed Diff_Weight Duration_Weight Height_times_Duration Weight_times_Duration 


# (3, 0.0), (6, 0.0), (8, 0.0), (9, 0.0), (14, 0.0), (16, 0.0), (17, 0.0), (19, 0.0), (21, 0.0), (22, 0.0)]
# Heart_Rate Body_Temp_trans Duration_Bucketed BMI Over_Weight 
# Age_times_Duration Height_times_Duration Body_Temp_times_Duration Weight_times_Duration Body_Temp_times_Age_Bucketed
# step1 Age Age_trans  
# step2 Weight Duration Height_trans Age_Bucketed BMI_Bucketed BMR BMR_Bucketed Diff_Weight Height_times_Weight Sex_times_Duration     
# step3  Duration Duration_Weight Heart_Rate_times_Duration


explain_matrix, masks = model_tabnet_7.explain(X_test)

fig, ax = plt.subplots(1, 3, figsize=(20,20))     # n_steps=3 
# fig, ax = plt.subplots(1, len(masks.keys(), figsize=(20,20))     # n_steps=3 

for i in range(3):
    ax[i].imshow(masks[i][:30])   
    ax[i].set_title(f'mask {i}')

plt.tight_layout()
plt.show()


# Residual plot: Tabnet model 

target ='Calories'
features = X.columns

train_pred = np.clip(model_tabnet_7.predict(X_train), a_min=0, a_max=None)
valid_pred = np.clip(model_tabnet_7.predict(X_valid), a_min=0, a_max=None)
tabnet_pred = np.clip(model_tabnet_7.predict(X_test), a_min=0, a_max=None)

fig, ax = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

x_max = np.max([np.max(train_pred), np.max(valid_pred)])
x_max = np.max([np.max(train_pred), x_max])
x_min = np.min([np.min(train_pred), np.min(valid_pred)])
x_min = np.min([np.min(tabnet_pred), x_min])

ax[0].scatter(tabnet_pred, (tabnet_pred - y_test), 
              c='limegreen', marker='s', edgecolor='white', label='X_test data')
ax[0].set_ylabel('Residuals')

ax[1].scatter(valid_pred, (valid_pred - y_valid), 
              c='steelblue', marker='o', edgecolor='white', label='Validation data')
ax[2].scatter(train_pred, (train_pred - y_train), 
              c='blue', marker='o', edgecolor='white', label='Training data')

for ax in (ax[0], ax[1], ax[2]):
    ax.set_xlabel('Predicted values')
    ax.legend(loc='upper left', fontsize=8)
    ax.hlines(y=0, 
              xmin=(x_min - 100),
              xmax=(x_max + 100), color='black', lw=2)

plt.suptitle('Tabnet model residuals')
plt.tight_layout()
plt.show()


tabnet_test_pred = np.clip(model_tabnet_7.predict(test_X), a_min=1, a_max=None)
submission_df['Calories'] = np.array(tabnet_test_pred)

display(submission_df.head())
display(submission_df.tail())


submission_df.to_csv('predict_calorie_expenditure_7.csv', index=False)

