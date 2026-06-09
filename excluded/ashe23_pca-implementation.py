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


import numpy as np 
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb, xgboost as xgb, catboost as cb
from gc import collect
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy
import seaborn as sns
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures, MinMaxScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_curve, make_scorer
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, SelectKBest
from sklearn.linear_model import LogisticRegression, RidgeClassifier,RidgeClassifierCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.manifold import TSNE
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


%%time
def get_data(mode='train'):
   
    data=pd.read_excel(f"/kaggle/input/widsdatathon2025/{mode}/{mode}_QUANTITATIVE_METADATA.xlsx")
    
    if mode=='TRAIN':
        cate=pd.read_excel(f"/kaggle/input/widsdatathon2025/{mode}/{mode}_CATEGORICAL_METADATA.xlsx")
    else:
        cate=pd.read_excel(f"/kaggle/input/widsdatathon2025/{mode}/{mode}_CATEGORICAL.xlsx")
    data=data.merge(cate,on='participant_id',how='left')
    
    func=pd.read_csv(f"/kaggle/input/widsdatathon2025/{mode}/{mode}_FUNCTIONAL_CONNECTOME_MATRICES.csv")
    data=data.merge(func,on='participant_id',how='left')

    # if mode=='TRAIN':
    #     solution=pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx")
    #     data=data.merge(solution,on='participant_id',how='left')
        
    return data
    
train=get_data(mode='TRAIN')
test=get_data(mode='TEST')
sample_sub= pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')
sol = pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx")


#First we have to make sure there are no missing values in Train and Test data
def fill_missing_with_mean(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns  # Select only numeric columns

    return df.fillna(df[columns].mean())


# Fill missing values using mean
train_filled = fill_missing_with_mean(train)
test_filled = fill_missing_with_mean(test)



train_pid = train_filled['participant_id']
test_pid = test_filled['participant_id']


train_filled.drop(columns = 'participant_id', inplace = True)
test_filled.drop(columns = 'participant_id', inplace = True)


scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_filled)
test_scaled = scaler.transform(test_filled)


# Plot the explained variance ratio
pca = PCA().fit(train_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()


pca = PCA(n_components=0.95, svd_solver='full')
train_pca = pca.fit_transform(train_scaled)

# Step 3: Transform the test data using the same PCA
test_pca = pca.transform(test_scaled)

# Print the results
print(f"Original number of features: {train_scaled.shape[1]}")
print(f"Reduced number of features after PCA: {train_pca.shape[1]}")
print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.2f}")




