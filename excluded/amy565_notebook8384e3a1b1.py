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
import seaborn as sns

import os
import matplotlib.pyplot as plt

import sklearn
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from scipy.stats import zscore, pearsonr, uniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV

from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


def get_feats(mode='TRAIN'):
    if mode == 'TRAIN':
        folder = 'TRAIN_NEW'
        prefix = 'TRAIN'
        suffix = '_new'
        connectome_file = f"{prefix}_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv"
    else:
        folder = 'TEST'
        prefix = 'TEST'
        suffix = ''  # TEST không có _new
        connectome_file = f"{prefix}_FUNCTIONAL_CONNECTOME_MATRICES.csv"

    # Đọc dữ liệu định lượng
    feats = pd.read_excel(f"/kaggle/input/widsdatathon2025/{folder}/{prefix}_QUANTITATIVE_METADATA{suffix}.xlsx")

    # Đọc dữ liệu phân loại
    if mode == 'TRAIN':
        cate = pd.read_excel(f"/kaggle/input/widsdatathon2025/{folder}/{prefix}_CATEGORICAL_METADATA{suffix}.xlsx")
    else:
        cate = pd.read_excel(f"/kaggle/input/widsdatathon2025/{folder}/{prefix}_CATEGORICAL.xlsx")

    # Gộp metadata
    feats = feats.merge(cate, on='participant_id', how='left')

    # Đọc và gộp connectome
    func = pd.read_csv(f"/kaggle/input/widsdatathon2025/{folder}/{connectome_file}")
    feats = feats.merge(func, on='participant_id', how='left')

    # Nếu là TRAIN, gộp thêm nhãn
    if mode == 'TRAIN':
        solution = pd.read_excel(f"/kaggle/input/widsdatathon2025/{folder}/TRAINING_SOLUTIONS.xlsx")
        feats = feats.merge(solution, on='participant_id', how='left')

    return feats



# Tạo dữ liệu
train = get_feats(mode='TRAIN')
test = get_feats(mode='TEST')

# Đọc file mẫu submission và nhãn thật
sub = pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')
y = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx')


train.set_index('participant_id',inplace=True)
test.set_index('participant_id',inplace=True)
targets = ['ADHD_Outcome','Sex_F']
features = test.columns


# Check for missing values in the training data
missing = train.isnull().sum()
missing_percent = 100 * missing / len(train)
missing_df = pd.DataFrame({
    'Missing Values': missing,
    'Percentage': missing_percent
})


# Display features with missing values
missing_features = missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False)
print("\nFeatures with missing values in training data:")
missing_features


# Gộp lại để xử lý đồng nhất
df_all = pd.concat([train, test], axis=0)

# ===============================
# 1. Impute median cho biến số liên tục
# ===============================
median_cols = ['MRI_Track_Age_at_Scan', 'EHQ_EHQ_Total'] + [
    'APQ_P_APQ_P_PP', 'APQ_P_APQ_P_PM', 'APQ_P_APQ_P_OPD',
    'APQ_P_APQ_P_INV', 'APQ_P_APQ_P_ID', 'APQ_P_APQ_P_CP',
    'SDQ_SDQ_Difficulties_Total', 'SDQ_SDQ_Emotional_Problems',
    'SDQ_SDQ_Externalizing', 'SDQ_SDQ_Conduct_Problems',
    'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Internalizing',
    'SDQ_SDQ_Peer_Problems', 'SDQ_SDQ_Prosocial',
    'SDQ_SDQ_Generating_Impact'
]

median_imputer = SimpleImputer(strategy='median')
df_all[median_cols] = median_imputer.fit_transform(df_all[median_cols])

# ===============================
# 2. Impute mode cho các biến phân loại
# ===============================
mode_cols = [
    'Barratt_Barratt_P2_Occ', 'Barratt_Barratt_P2_Edu',
    'Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P1_Edu',
    'PreInt_Demos_Fam_Child_Race', 'PreInt_Demos_Fam_Child_Ethnicity',
    'ColorVision_CV_Score', 'MRI_Track_Scan_Location'
]

mode_imputer = SimpleImputer(strategy='most_frequent')
df_all[mode_cols] = mode_imputer.fit_transform(df_all[mode_cols])

# ===============================
# 3. Tách lại thành Train/Test
# ===============================
train_imputed = df_all.iloc[:len(train)].copy()
test_imputed = df_all.iloc[len(train):].copy()


train.describe()


train.info()


#Gender distribution
sns.countplot(x='Sex_F', data=train[["Sex_F"]])
plt.xticks([0, 1], ['Male', 'Female'])
plt.title("Gender distribution")
plt.xlabel("Gender")
plt.show()


# Distribution of MRI_Track_Age_at_Scan
train['MRI_Track_Age_at_Scan'].hist(figsize=(12, 10), bins=20)
plt.suptitle("MRI_Track_Age_at_Scan Distributions")
plt.xlabel('MRI_Track_Age_at_Scan')
plt.ylabel('Frequency Count')
plt.show()


# Check for correlation with ADHD outcome
train_copy= train.copy()
train_copy['ADHD_Outcome'] = train['ADHD_Outcome']

plt.figure(figsize=(8, 6))
sns.boxplot(x='ADHD_Outcome', y='SDQ_SDQ_Emotional_Problems', data=train_copy)
plt.title('SDQ_SDQ_Emotional_Problems vs ADHD Outcome')
plt.xlabel('ADHD Outcome')
plt.ylabel('SDQ_SDQ_Emotional_Problems')
plt.show()




