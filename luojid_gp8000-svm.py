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


import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 50
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

plt.pcolormesh(np.random.rand(5,5))


INPUT_PATH = "/kaggle/input/forest-cover-type-kernels-only/"
train = pd.read_csv(f"{INPUT_PATH}train.csv.zip")
train


print(train.columns.shape)
print(train.dtypes)


numerical_features= ['Elevation', 'Aspect', 'Slope',
                    'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', # TODO: create a new feature dist_to_hyf
                    'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points',
                    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
categorical_features = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

print(len(numerical_features), len(categorical_features))



X_ds = train.drop(['Id','Cover_Type'], axis = 1)
X = X_ds.values
X.shape


X_num = X_ds[numerical_features]
for feature in X_num:
    plt.figure(figsize=(5,1))
    sns.histplot(X_num[feature])


X_num = X_ds[numerical_features]
X_num

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# scaling
s_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
X_num_s_scaled = s_scaler.fit_transform(X_num)
X_num_mm_scaled = mm_scaler.fit_transform(X_num)

print(X_num_s_scaled.max(),X_num_s_scaled.min())
print(X_num_mm_scaled.max(),X_num_mm_scaled.min())

print(X_num_s_scaled.shape)
print(X_num_mm_scaled.shape)

# combine binary features
X_cat = X_ds[categorical_features]

X_s_all = np.hstack([X_num_s_scaled, X_cat])
X_mm_all = np.hstack([X_num_s_scaled, X_cat])
print(X_s_all.shape)
print(X_mm_all.shape)






y = train.Cover_Type.values
y.shape


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X_s_all, y, test_size=0.2, random_state=42)

C_grid = {"C": [0.01, 0.1, 1, 10, 100]}
svm_model = SVC(kernel='linear')
grid_search = GridSearchCV(svm_model, C_grid, 
                           cv=5, 
                           scoring="accuracy",
                           n_jobs = -1)
grid_search.fit(X_train, y_train)


print(grid_search.best_params_)


svc_best = SVC(kernel="linear", C=100)
svc_best.fit(X_train, y_train)

accuracy = svc_best.score(X_test, y_test)
print(f"Test Accuracy: {accuracy}")




