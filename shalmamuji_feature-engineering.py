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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
from itertools import combinations
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Set random seed for reproducibility
np.random.seed(42)

def mapk(actual, predicted, k=3):
    """Compute Mean Average Precision at k."""
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        num_hits = 0.0
        for i, pred in enumerate(p):
            if pred == a and pred not in p[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
original_df = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')  # Load the original dataset
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


# Feature Engineering
print("\nPerforming advanced feature engineering...")

# 1. Nutrient Features
for df in [train_df, test_df]:
    # Basic ratios
    df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
    df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
    df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-6)
    
    # Advanced nutrient features
    df['nutrient_sum'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    df['nutrient_balance'] = df['Nitrogen'] * df['Phosphorous'] * df['Potassium']
    df['nutrient_mean'] = df['nutrient_sum'] / 3
    df['nutrient_std'] = df[['Nitrogen', 'Phosphorous', 'Potassium']].std(axis=1)
    df['nutrient_max'] = df[['Nitrogen', 'Phosphorous', 'Potassium']].max(axis=1)
    df['nutrient_min'] = df[['Nitrogen', 'Phosphorous', 'Potassium']].min(axis=1)
    
    # Nutrient dominance features
    df['N_dominance'] = df['Nitrogen'] / df['nutrient_sum']
    df['P_dominance'] = df['Phosphorous'] / df['nutrient_sum']
    df['K_dominance'] = df['Potassium'] / df['nutrient_sum']
    
    # New nutrient interaction features
    df['NP_interaction'] = df['Nitrogen'] * df['Phosphorous']
    df['NK_interaction'] = df['Nitrogen'] * df['Potassium']
    df['PK_interaction'] = df['Phosphorous'] * df['Potassium']
    
    # Nutrient ratios with weather
    df['N_temp_ratio'] = df['Nitrogen'] / (df['Temparature'] + 1e-6)
    df['P_humidity_ratio'] = df['Phosphorous'] / (df['Humidity'] + 1e-6)
    df['K_moisture_ratio'] = df['Potassium'] / (df['Moisture'] + 1e-6)


# 2. Weather Features
for df in [train_df, test_df]:
    # Basic interactions
    df['temp_humidity'] = df['Temparature'] * df['Humidity']
    df['temp_moisture'] = df['Temparature'] * df['Moisture']
    df['humidity_moisture'] = df['Humidity'] * df['Moisture']
    
    # Advanced weather features
    df['weather_sum'] = df['Temparature'] + df['Humidity'] + df['Moisture']
    df['weather_mean'] = df['weather_sum'] / 3
    df['weather_std'] = df[['Temparature', 'Humidity', 'Moisture']].std(axis=1)
    
    # Weather extremes
    df['temp_humidity_ratio'] = df['Temparature'] / (df['Humidity'] + 1e-6)
    df['temp_moisture_ratio'] = df['Temparature'] / (df['Moisture'] + 1e-6)
    df['humidity_moisture_ratio'] = df['Humidity'] / (df['Moisture'] + 1e-6)
    
    # New weather features
    df['weather_product'] = df['Temparature'] * df['Humidity'] * df['Moisture']
    df['weather_max'] = df[['Temparature', 'Humidity', 'Moisture']].max(axis=1)
    df['weather_min'] = df[['Temparature', 'Humidity', 'Moisture']].min(axis=1)
    df['weather_range'] = df['weather_max'] - df['weather_min']


# 3. Interaction Features
for df in [train_df, test_df]:
    # Create interaction between soil type and crop type
    df['soil_crop_interaction'] = df['Soil Type'].astype(str) + '_' + df['Crop Type'].astype(str)
    
    # Create frequency encoding for categorical features
    for col in ['Soil Type', 'Crop Type', 'soil_crop_interaction']:
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq)
    
    # New interaction features
    df['soil_nutrient_mean'] = df.groupby('Soil Type')['nutrient_mean'].transform('mean')
    df['crop_nutrient_mean'] = df.groupby('Crop Type')['nutrient_mean'].transform('mean')
    df['soil_weather_mean'] = df.groupby('Soil Type')['weather_mean'].transform('mean')
    df['crop_weather_mean'] = df.groupby('Crop Type')['weather_mean'].transform('mean')


# Encode categorical features
label_encoders = {}
for col in ['Soil Type', 'Crop Type', 'soil_crop_interaction']:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# Encode target
target_le = LabelEncoder()
train_df['Fertilizer Name'] = target_le.fit_transform(train_df['Fertilizer Name'])


# Prepare features
feature_cols = [col for col in train_df.columns if col not in ['id', 'Fertilizer Name']]
X = train_df[feature_cols]
y = train_df['Fertilizer Name']
X_test = test_df[feature_cols]

# Scale numerical features
scaler = StandardScaler()
numerical_cols = [col for col in feature_cols if col not in ['Soil Type', 'Crop Type', 'soil_crop_interaction']]
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

