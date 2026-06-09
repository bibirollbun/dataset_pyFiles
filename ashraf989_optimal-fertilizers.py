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


import pandas as pd               # For data manipulation and analysis
import numpy as np                # For numerical operations
import matplotlib.pyplot as plt    # For data visualization
import seaborn as sns             # For enhanced data visualization
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # For feature scaling and encoding
from sklearn.compose import ColumnTransformer  # For applying transformations to specific columns
from sklearn.pipeline import Pipeline  # For creating a machine learning pipeline
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.metrics import classification_report, accuracy_score  # For evaluating model performance
from xgboost import XGBClassifier  # For using XGBoost


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Load the training data
train_data = pd.read_csv('../input/playground-series-s5e6/train.csv')

# Load the test data
test_data = pd.read_csv('../input/playground-series-s5e6/test.csv')


train_data.head()


test_data.head()


print("Training data shape:", train_data.shape)
print("Test data shape:", test_data.shape)


# train_data.isnull().sum()
# test_data.isnull().sum()


from sklearn.preprocessing import LabelEncoder

# Create LabelEncoders for both features
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

# Fit and transform the 'Soil Type' and 'Crop Type'
train_data['Soil Type'] = soil_encoder.fit_transform(train_data['Soil Type'])
train_data['Crop Type'] = crop_encoder.fit_transform(train_data['Crop Type'])

# Display the first few rows of the encoded dataset
train_data.head()


#Distribution of Target Variable

sns.countplot(x='Fertilizer Name', data=train_data)
plt.title('Distribution of Target Variable')
plt.show()




