### head
# dependency

# Try the way mentioned in https://www.kaggle.com/discussions/product-feedback/532336
# If fail, try : https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/113195
#
# mark when do "save and run(commit)", unmark when want to try edit current version
# 
# --no-index /kaggle/input/iterative-stratification/iterative_stratification-0.1.9-py3-none-any.whl
#
#
#


# !pip install iterative_stratification
# !pip install pytorch_tabnet
# !pip install pytorch_tabnet
# !pip install numpy pandas matplotlib joblib scikit-learn
# !pip install xgboost
# !pip install notebook
# !pip install catboost
# !pip install lightgbm
# !pip install tensorflow

# !pip uninstall tensorflow -y
# !pip install tensorflow==2.17.0



!pip install pytorch_tabnet
!pip install pytorch_tabnet
!pip install numpy pandas matplotlib joblib scikit-learn
!pip install xgboost
!pip install notebook
!pip install catboost
!pip install lightgbm
!pip install tensorflow
!pip install tensorflow==2.17.0
!pip install tensorflow==2.17.0


import sys
sys.path.append('/kaggle/input/iterativestratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


### General ###
import os
import copy
import tqdm
import pickle
import random
import warnings
warnings.filterwarnings("ignore")
sys.path.append("../input/rank-gauss")
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

### Data Wrangling ###
import numpy as np
import pandas as pd
from scipy import stats

### Machine Learning ###
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pickle import load,dump

### Deep Learning ###
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Tabnet 
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor

### Make prettier the prints ###
from colorama import Fore
c_ = Fore.CYAN
m_ = Fore.MAGENTA
r_ = Fore.RED
b_ = Fore.BLUE
y_ = Fore.YELLOW
g_ = Fore.GREEN

from sklearn.preprocessing import QuantileTransformer

os.listdir('../input/lish-moa')

train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')
df = pd.read_csv('../input/lish-moa/sample_submission.csv')

train_features2=train_features.copy()
test_features2=test_features.copy()

GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]

qt = QuantileTransformer(n_quantiles=100,random_state=42,output_distribution='normal')
train_features[GENES+CELLS] = qt.fit_transform(train_features[GENES+CELLS])
test_features[GENES+CELLS] = qt.transform(test_features[GENES+CELLS])

seed = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(seed)

# GENES
n_comp = 600  #<--Update
gpca= load(open('/kaggle/input/3-11-hning-train/gpca.pkl', 'rb'))
train2= (gpca.transform(train_features[GENES]))
test2 = (gpca.transform(test_features[GENES]))

train_gpca = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
test_gpca = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])

# drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
train_features = pd.concat((train_features, train_gpca), axis=1)
test_features = pd.concat((test_features, test_gpca), axis=1)

test_gpca

#CELLS
n_comp = 50  #<--Update

cpca= load(open('/kaggle/input/3-11-hning-train/cpca.pkl', 'rb'))
train2= (cpca.transform(train_features[CELLS]))
test2 = (cpca.transform(test_features[CELLS]))

train_cpca = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
test_cpca = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])

# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
train_features = pd.concat((train_features, train_cpca), axis=1)
test_features = pd.concat((test_features, test_cpca), axis=1)

from sklearn.feature_selection import VarianceThreshold

c_n = [f for f in list(train_features.columns) if f not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]
mask = (train_features[c_n].var() >= 0.85).values
tmp = train_features[c_n].loc[:, mask]
train_features = pd.concat([train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)
tmp = test_features[c_n].loc[:, mask]
test_features = pd.concat([test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)

from sklearn.cluster import KMeans
def fe_cluster_genes(train, test, n_clusters_g = 22, SEED = 42):
    
    features_g = GENES
    #features_c = CELLS
    
    def create_cluster(train, test, features, kind = 'g', n_clusters = n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        kmeans_genes = load(open('/kaggle/input/3-11-hning-train/kmeans_genes.pkl', 'rb'))
        train[f'clusters_{kind}'] = kmeans_genes.predict(train_.values)
        test[f'clusters_{kind}'] = kmeans_genes.predict(test_.values)
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        return train, test
    
    train, test = create_cluster(train, test, features_g, kind = 'g', n_clusters = n_clusters_g)
   # train, test = create_cluster(train, test, features_c, kind = 'c', n_clusters = n_clusters_c)
    return train, test

train_features2 ,test_features2=fe_cluster_genes(train_features2,test_features2)

def fe_cluster_cells(train, test, n_clusters_c = 4, SEED = 42):
    
    #features_g = GENES
    features_c = CELLS
    
    def create_cluster(train, test, features, kind = 'c', n_clusters = n_clusters_c):
        train_ = train[features].copy()
        test_ = test[features].copy()
        kmeans_cells = load(open('/kaggle/input/3-11-hning-train/kmeans_cells.pkl', 'rb'))
        train[f'clusters_{kind}'] = kmeans_cells.predict(train_.values)
        test[f'clusters_{kind}'] = kmeans_cells.predict(test_.values)
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        return train, test
    
   # train, test = create_cluster(train, test, features_g, kind = 'g', n_clusters = n_clusters_g)
    train, test = create_cluster(train, test, features_c, kind = 'c', n_clusters = n_clusters_c)
    return train, test

train_features2 ,test_features2=fe_cluster_cells(train_features2,test_features2)

train_pca=pd.concat((train_gpca,train_cpca),axis=1)
test_pca=pd.concat((test_gpca,test_cpca),axis=1)

def fe_cluster_pca(train, test,n_clusters=5,SEED = 42):
        kmeans_pca = load(open('/kaggle/input/3-11-hning-train/kmeans_pca.pkl', 'rb'))
        train[f'clusters_pca'] = kmeans_pca.predict(train.values)
        test[f'clusters_pca'] = kmeans_pca.predict(test.values)
        train = pd.get_dummies(train, columns = [f'clusters_pca'])
        test = pd.get_dummies(test, columns = [f'clusters_pca'])
        return train, test
train_cluster_pca ,test_cluster_pca = fe_cluster_pca(train_pca,test_pca)

train_cluster_pca = train_cluster_pca.iloc[:,650:]
test_cluster_pca = test_cluster_pca.iloc[:,650:]

train_features_cluster=train_features2.iloc[:,876:]
test_features_cluster=test_features2.iloc[:,876:]


gsquarecols=['g-574','g-211','g-216','g-0','g-255','g-577','g-153','g-389','g-60','g-370','g-248','g-167','g-203','g-177','g-301','g-332','g-517','g-6','g-744','g-224','g-162','g-3','g-736','g-486','g-283','g-22','g-359','g-361','g-440','g-335','g-106','g-307','g-745','g-146','g-416','g-298','g-666','g-91','g-17','g-549','g-145','g-157','g-768','g-568','g-396']

def fe_stats(train, test):
    
    features_g = GENES
    features_c = CELLS
    
    for df in train, test:
        df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
        df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
        
        df['c52_c42'] = df['c-52'] * df['c-42']
        df['c13_c73'] = df['c-13'] * df['c-73']
        df['c26_c13'] = df['c-23'] * df['c-13']
        df['c33_c6'] = df['c-33'] * df['c-6']
        df['c11_c55'] = df['c-11'] * df['c-55']
        df['c38_c63'] = df['c-38'] * df['c-63']
        df['c38_c94'] = df['c-38'] * df['c-94']
        df['c13_c94'] = df['c-13'] * df['c-94']
        df['c4_c52'] = df['c-4'] * df['c-52']
        df['c4_c42'] = df['c-4'] * df['c-42']
        df['c13_c38'] = df['c-13'] * df['c-38']
        df['c55_c2'] = df['c-55'] * df['c-2']
        df['c55_c4'] = df['c-55'] * df['c-4']
        df['c4_c13'] = df['c-4'] * df['c-13']
        df['c82_c42'] = df['c-82'] * df['c-42']
        df['c66_c42'] = df['c-66'] * df['c-42']
        df['c6_c38'] = df['c-6'] * df['c-38']
        df['c2_c13'] = df['c-2'] * df['c-13']
        df['c62_c42'] = df['c-62'] * df['c-42']
        df['c90_c55'] = df['c-90'] * df['c-55']
        df['c26_c38'] = df['c-26'] * df['c-38']
        df['c90_c13'] = df['c-90'] * df['c-13']
        df['c85_c31'] = df['c-85'] * df['c-31']
        df['c63_c42'] = df['c-63'] * df['c-42']
        df['c94_c11'] = df['c-94'] * df['c-11']
        df['c94_c60'] = df['c-94'] * df['c-60']
        df['c55_c42'] = df['c-55'] * df['c-42']
        df['g37_c50'] = df['g-37'] * df['g-50']
        
        for feature in features_c:
             df[f'{feature}_squared'] = df[feature] ** 2     
                
        for feature in gsquarecols:
            df[f'{feature}_squared'] = df[feature] ** 2        
        
    return train, test

train_features2,test_features2=fe_stats(train_features2,test_features2)

train_features_stats=train_features2.iloc[:,902:]
test_features_stats=test_features2.iloc[:,902:]

train_features = pd.concat((train_features, train_features_cluster,train_cluster_pca,train_features_stats), axis=1)
test_features = pd.concat((test_features, test_features_cluster,test_cluster_pca,test_features_stats), axis=1)

train_features

train = train_features.merge(train_targets_scored, on='sig_id')
train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

target = train[train_targets_scored.columns]

train = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)

target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

target=target[target_cols]

train = pd.get_dummies(train, columns=['cp_time','cp_dose'])
test_ = pd.get_dummies(test, columns=['cp_time','cp_dose'])

feature_cols = [c for c in train.columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['sig_id']]

# feature_cols.remove('pca_G-143')

train = train[feature_cols]
test = test_[feature_cols]

feature_cols[-1]

len(train.columns)

X_test = test.astype(np.float32).values

class LogitsLogLoss(Metric):

    def __init__(self):
        self._name = "logits_ll"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        logits = 1 / (1 + np.exp(-y_pred))
        aux = (1 - y_true) * np.log(1 - logits + 5e-5) + y_true * np.log(logits + 5e-5)
        return np.mean(-aux)

MAX_EPOCH = 200

tabnet_params = dict(
    n_d = 32,
    n_a = 32,
    n_steps = 1,
    gamma = 1.3,
    lambda_sparse = 0,
    optimizer_fn = optim.Adam,
    optimizer_params = dict(lr = 2e-2, weight_decay = 1e-5),
    mask_type = "entmax",
    scheduler_params = dict(mode = "min", patience = 5, min_lr = 1e-5, factor = 0.9),
    scheduler_fn = ReduceLROnPlateau,
    seed = seed,
    verbose = 10
)

test_cv_preds = []

NB_SPLITS = 7
mskf = MultilabelStratifiedKFold(n_splits = NB_SPLITS, random_state = 0, shuffle = True)
SEED = [0, 1, 2, 3, 4, 5, 6]
for s in SEED:
    tabnet_params['seed'] = s
    for fold_nb, (train_idx, val_idx) in enumerate(mskf.split(train, target)):
        
        model = TabNetRegressor()
        ### Predict on test ###
        model.load_model(f"/kaggle/input/3-11-hning-train/TabNet_seed_{s}_fold_{fold_nb+1}.zip")
        preds_test = model.predict(X_test)
        test_cv_preds.append(1 / (1 + np.exp(-preds_test)))

        print("finish one")

test_preds_all = np.stack(test_cv_preds)

all_feat = [col for col in df.columns if col not in ["sig_id"]]
# To obtain the same lenght of test_preds_all and submission
test = pd.read_csv("../input/lish-moa/test_features.csv")
sig_id = test[test["cp_type"] != "ctl_vehicle"].sig_id.reset_index(drop = True)
tmp = pd.DataFrame(test_preds_all.mean(axis = 0), columns = all_feat)
tmp["sig_id"] = sig_id

submission = pd.merge(test[["sig_id"]], tmp, on = "sig_id", how = "left")
submission.fillna(0, inplace = True)
submission.to_csv("submission_jj.csv", index = None)

sub_jj = submission.copy()

# submission.head()
# submission.isna().sum().sum()

# print(f"{b_}submission.shape: {r_}{submission.shape}")

# check=pd.read_csv('/kaggle/input/3-11-hning-train/submission_jj.csv')

# subm=pd.read_csv('/kaggle/input/3-11-hning-train/submission_jj.csv')

# subm.head()



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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import random
import pandas as pd
from copy import deepcopy as dp

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import _WeightedLoss

#
# already have the Tabnet import
#
# Tabnet 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor

# feature transformation - fit
def norm_fit(df_1,saveM = True, sc_name = 'zsco'):   
    from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler,Normalizer,QuantileTransformer,PowerTransformer
    ss_1_dic = {'zsco':StandardScaler(),
                'mima':MinMaxScaler(),
                'maxb':MaxAbsScaler(), 
                'robu':RobustScaler(),
                'norm':Normalizer(), 
                'quan':QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal"),
                'powe':PowerTransformer()}
    ss_1 = ss_1_dic[sc_name]
    df_2 = pd.DataFrame(ss_1.fit_transform(df_1),index = df_1.index,columns = df_1.columns)
    if saveM == False:
        return(df_2)
    else:
        return(df_2,ss_1)

# feature transformation - trans
def norm_tra(df_1,ss_x):
    df_2 = pd.DataFrame(ss_x.transform(df_1),index = df_1.index,columns = df_1.columns)
    return(df_2)

# frequency 
def f_table(list1):
    table_dic = {}
    for i in list1:
        if i not in table_dic.keys():
            table_dic[i] = 1
        else:
            table_dic[i] += 1
    return(table_dic)

# seed for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
import warnings
warnings.filterwarnings('ignore')

seed_everything(seed=42)

# input data dir
input_dir = '../input/lish-moa/'
# upload model dataset from training kernel outputs
mod_path1 = '../input/1d-cnn-train/'                 # 1D-CNN
mod_path2 = '../input/tabnet-train/' # TabNet
mod_path3 = '../input/dnn-train/'                       # DNN


# head
SEED = [0]
# SEED = [0, 1, 2, 3 ,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

seed_everything(seed=42)

sc_dic = {}
feat_dic = {}
train_features = pd.read_csv(input_dir+'train_features.csv').sample(n=15000, random_state=42)
train_targets_scored = pd.read_csv(input_dir+'train_targets_scored.csv').sample(n=15000, random_state=42)
train_targets_nonscored = pd.read_csv(input_dir+'train_targets_nonscored.csv').sample(n=15000, random_state=42)
test_features = pd.read_csv(input_dir+'test_features.csv')
sample_submission = pd.read_csv(input_dir+'sample_submission.csv')
train_drug = pd.read_csv(input_dir+'train_drug.csv').sample(n=15000, random_state=42)

target_cols = train_targets_scored.drop('sig_id', axis=1).columns.values.tolist()
target_nonsc_cols = train_targets_nonscored.drop('sig_id', axis=1).columns.values.tolist()

# non-score targets highly correlated with scored targets will be used in pretrain
nonctr_id = train_features.loc[train_features['cp_type']!='ctl_vehicle','sig_id'].tolist()
tmp_con1 = [i in nonctr_id for i in train_targets_scored['sig_id']]
mat_cor = pd.DataFrame(np.corrcoef(train_targets_scored.drop('sig_id',axis = 1)[tmp_con1].T,
                      train_targets_nonscored.drop('sig_id',axis = 1)[tmp_con1].T))
mat_cor2 = mat_cor.iloc[(train_targets_scored.shape[1]-1):,0:train_targets_scored.shape[1]-1]
mat_cor2.index = target_nonsc_cols
mat_cor2.columns = target_cols
mat_cor2 = mat_cor2.dropna()
mat_cor2_max = mat_cor2.abs().max(axis = 1)

q_n_cut = 0.9
target_nonsc_cols2 = mat_cor2_max[mat_cor2_max > np.quantile(mat_cor2_max,q_n_cut)].index.tolist()
print(len(target_nonsc_cols2))

GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]
feat_dic['gene'] = GENES
feat_dic['cell'] = CELLS

# sample normalization 
q2 = train_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = train_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.75).copy()
qmean = (q2+q7)/2
train_features[feat_dic['gene']] = (train_features[feat_dic['gene']].T - qmean.values).T
q2 = test_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = test_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.75).copy()
qmean = (q2+q7)/2
test_features[feat_dic['gene']] = (test_features[feat_dic['gene']].T - qmean.values).T

q2 = train_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = train_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.72).copy()
qmean = (q2+q7)/2
train_features[feat_dic['cell']] = (train_features[feat_dic['cell']].T - qmean.values).T
qmean2 = train_features[feat_dic['cell']].abs().apply(np.quantile,axis = 1,q = 0.75).copy()+4
train_features[feat_dic['cell']] = (train_features[feat_dic['cell']].T / qmean2.values).T.copy()

q2 = test_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = test_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.72).copy()
qmean = (q2+q7)/2
test_features[feat_dic['cell']] = (test_features[feat_dic['cell']].T - qmean.values).T
qmean2 = test_features[feat_dic['cell']].abs().apply(np.quantile,axis = 1,q = 0.75).copy()+4
test_features[feat_dic['cell']] = (test_features[feat_dic['cell']].T / qmean2.values).T.copy()

# remove ctr 
train = train_features.merge(train_targets_scored, on='sig_id')
train = train.merge(train_targets_nonscored[['sig_id']+target_nonsc_cols2], on='sig_id')

train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

target = train[['sig_id']+target_cols]
target_ns = train[['sig_id']+target_nonsc_cols2]

train0 = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)

target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

# drug ids
tar_sig = target['sig_id'].tolist()
train_drug = train_drug.loc[[i in tar_sig for i in train_drug['sig_id']]]
target = target.merge(train_drug, on='sig_id', how='left') 

# LOCATE DRUGS
vc = train_drug.drug_id.value_counts()
vc1 = vc.loc[vc <= 19].index
vc2 = vc.loc[vc > 19].index

feature_cols = []
for key_i in feat_dic.keys():
    value_i = feat_dic[key_i]
    print(key_i,len(value_i))
    feature_cols += value_i
feature_cols0 = dp(feature_cols)
    
oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

# Averaging on multiple SEEDS
for seed in SEED:
    seed_everything(seed=seed)
    folds = train0.copy()
    feature_cols = dp(feature_cols0)
    
    # Kfold - leave drug out
    target2 = target.copy()
    dct1 = {}; dct2 = {}
    skf = MultilabelStratifiedKFold(n_splits = 5) # , shuffle = True, random_state = seed
    tmp = target2.groupby('drug_id')[target_cols].mean().loc[vc1]
    tmp_idx = tmp.index.tolist()
    tmp_idx.sort()
    tmp_idx2 = random.sample(tmp_idx,len(tmp_idx))
    tmp = tmp.loc[tmp_idx2]
    for fold,(idxT,idxV) in enumerate(skf.split(tmp,tmp[target_cols])):
        dd = {k:fold for k in tmp.index[idxV].values}
        dct1.update(dd)

    skf = MultilabelStratifiedKFold(n_splits = 5) # , shuffle = True, random_state = seed
    tmp = target2.loc[target2.drug_id.isin(vc2)].reset_index(drop = True)
    tmp_idx = tmp.index.tolist()
    tmp_idx.sort()
    tmp_idx2 = random.sample(tmp_idx,len(tmp_idx))
    tmp = tmp.loc[tmp_idx2]
    for fold,(idxT,idxV) in enumerate(skf.split(tmp,tmp[target_cols])):
        dd = {k:fold for k in tmp.sig_id[idxV].values}
        dct2.update(dd)

    target2['kfold'] = target2.drug_id.map(dct1)
    target2.loc[target2.kfold.isna(),'kfold'] = target2.loc[target2.kfold.isna(),'sig_id'].map(dct2)
    target2.kfold = target2.kfold.astype(int)

    folds['kfold'] = target2['kfold'].copy()

    train = folds.copy()
    test_ = test.copy()

    # HyperParameters
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 25
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NFOLDS = 5
    EARLY_STOPPING_STEPS = 10
    EARLY_STOP = False

    n_comp1 = 50
    n_comp2 = 15

    num_features=len(feature_cols) + n_comp1 + n_comp2
    num_targets=len(target_cols)
    num_targets_0=len(target_nonsc_cols2)
    hidden_size=4096

    tar_freq = np.array([np.min(list(f_table(train[target_cols].iloc[:,i]).values())) for i in range(len(target_cols))])
    tar_weight0 = np.array([np.log(i+100) for i in tar_freq])
    tar_weight0_min = dp(np.min(tar_weight0))
    tar_weight = tar_weight0_min/tar_weight0
    np.mean(tar_weight)
    pos_weight = torch.tensor(tar_weight).to(DEVICE)
    
    class SmoothBCEwLogits(_WeightedLoss):
        def __init__(self, weight=None, reduction='mean', smoothing=0.0):
            super().__init__(weight=weight, reduction=reduction)
            self.smoothing = smoothing
            self.weight = weight
            self.reduction = reduction

        @staticmethod
        def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
            assert 0 <= smoothing < 1
            with torch.no_grad():
                targets = targets * (1.0 - smoothing) + 0.5 * smoothing
            return targets

        def forward(self, inputs, targets):
            targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
                self.smoothing)
            loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight,
                                                      pos_weight = pos_weight)

            if  self.reduction == 'sum':
                loss = loss.sum()
            elif  self.reduction == 'mean':
                loss = loss.mean()

            return loss

    class TrainDataset:
        def __init__(self, features, targets):
            self.features = features
            self.targets = targets

        def __len__(self):
            return (self.features.shape[0])

        def __getitem__(self, idx):
            dct = {
                'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
                'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            
            }
            return dct

    class TestDataset:
        def __init__(self, features):
            self.features = features

        def __len__(self):
            return (self.features.shape[0])

        def __getitem__(self, idx):
            dct = {
                'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
            }
            return dct


    def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
        model.train()
        final_loss = 0

        for data in dataloader:
            optimizer.zero_grad()
            inputs, targets = data['x'].to(device), data['y'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            final_loss += loss.item()

        final_loss /= len(dataloader)

        return final_loss


    def valid_fn(model, loss_fn, dataloader, device):
        model.eval()
        final_loss = 0
        valid_preds = []

        for data in dataloader:
            inputs, targets = data['x'].to(device), data['y'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            final_loss += loss.item()
            valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

        final_loss /= len(dataloader)
        valid_preds = np.concatenate(valid_preds)

        return final_loss, valid_preds

    def inference_fn(model, dataloader, device):
        model.eval()
        preds = []

        for data in dataloader:
            inputs = data['x'].to(device)
            with torch.no_grad():
                outputs = model(inputs)

            preds.append(outputs.sigmoid().detach().cpu().numpy())

        preds = np.concatenate(preds)

        return preds

    class Model(nn.Module):

        def __init__(self, num_features, num_targets, hidden_size):
            super(Model, self).__init__()
            cha_1 = 256
            cha_2 = 512
            cha_3 = 512

            cha_1_reshape = int(hidden_size/cha_1)
            cha_po_1 = int(hidden_size/cha_1/2)
            cha_po_2 = int(hidden_size/cha_1/2/2) * cha_3

            self.cha_1 = cha_1
            self.cha_2 = cha_2
            self.cha_3 = cha_3
            self.cha_1_reshape = cha_1_reshape
            self.cha_po_1 = cha_po_1
            self.cha_po_2 = cha_po_2

            self.batch_norm1 = nn.BatchNorm1d(num_features)
            self.dropout1 = nn.Dropout(0.1)
            self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

            self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
            self.dropout_c1 = nn.Dropout(0.1)
            self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1,cha_2, kernel_size = 5, stride = 1, padding=2,  bias=False),dim=None)

            self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = cha_po_1)

            self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
            self.dropout_c2 = nn.Dropout(0.1)
            self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

            self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
            self.dropout_c2_1 = nn.Dropout(0.3)
            self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

            self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
            self.dropout_c2_2 = nn.Dropout(0.2)
            self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_3, kernel_size = 5, stride = 1, padding=2, bias=True),dim=None)

            self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

            self.flt = nn.Flatten()

            self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
            self.dropout3 = nn.Dropout(0.2)
            self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

        def forward(self, x):

            x = self.batch_norm1(x)
            x = self.dropout1(x)
            x = F.celu(self.dense1(x), alpha=0.06)

            x = x.reshape(x.shape[0],self.cha_1,
                          self.cha_1_reshape)

            x = self.batch_norm_c1(x)
            x = self.dropout_c1(x)
            x = F.relu(self.conv1(x))

            x = self.ave_po_c1(x)

            x = self.batch_norm_c2(x)
            x = self.dropout_c2(x)
            x = F.relu(self.conv2(x))
            x_s = x

            x = self.batch_norm_c2_1(x)
            x = self.dropout_c2_1(x)
            x = F.relu(self.conv2_1(x))

            x = self.batch_norm_c2_2(x)
            x = self.dropout_c2_2(x)
            x = F.relu(self.conv2_2(x))
            x =  x * x_s

            x = self.max_po_c2(x)

            x = self.flt(x)

            x = self.batch_norm3(x)
            x = self.dropout3(x)
            x = self.dense3(x)

            return x

    def run_training(fold, seed):

        seed_everything(seed)

        trn_idx = train[train['kfold'] != fold].index
        val_idx = train[train['kfold'] == fold].index

        train_df = train[train['kfold'] != fold].reset_index(drop=True).copy()
        valid_df = train[train['kfold'] == fold].reset_index(drop=True).copy()

        x_train, y_train,y_train_ns = train_df[feature_cols], train_df[target_cols].values,train_df[target_nonsc_cols2].values
        x_valid, y_valid,y_valid_ns  =  valid_df[feature_cols], valid_df[target_cols].values,valid_df[target_nonsc_cols2].values
        x_test = test_[feature_cols]

        #------------ norm --------------
        col_num = list(set(feat_dic['gene'] + feat_dic['cell']) & set(feature_cols))
        col_num.sort()
        x_train[col_num],ss = norm_fit(x_train[col_num],True,'quan')
        x_valid[col_num]    = norm_tra(x_valid[col_num],ss)
        x_test[col_num]     = norm_tra(x_test[col_num],ss)

        #------------ pca --------------
        def pca_pre(tr,va,te,
                    n_comp,feat_raw,feat_new):
            pca = PCA(n_components=n_comp, random_state=42)
            tr2 = pd.DataFrame(pca.fit_transform(tr[feat_raw]),columns=feat_new)
            va2 = pd.DataFrame(pca.transform(va[feat_raw]),columns=feat_new)
            te2 = pd.DataFrame(pca.transform(te[feat_raw]),columns=feat_new)
            return(tr2,va2,te2)


        pca_feat_g = [f'pca_G-{i}' for i in range(n_comp1)]
        feat_dic['pca_g'] = pca_feat_g
        x_tr_g_pca,x_va_g_pca,x_te_g_pca = pca_pre(x_train,x_valid,x_test,
                                                   n_comp1,feat_dic['gene'],pca_feat_g)
        x_train = pd.concat([x_train,x_tr_g_pca],axis = 1)
        x_valid = pd.concat([x_valid,x_va_g_pca],axis = 1)
        x_test  = pd.concat([x_test,x_te_g_pca],axis = 1)

        pca_feat_g = [f'pca_C-{i}' for i in range(n_comp2)]
        feat_dic['pca_c'] = pca_feat_g
        x_tr_c_pca,x_va_c_pca,x_te_c_pca = pca_pre(x_train,x_valid,x_test,
                                                   n_comp2,feat_dic['cell'],pca_feat_g)
        x_train = pd.concat([x_train,x_tr_c_pca],axis = 1)
        x_valid = pd.concat([x_valid,x_va_c_pca],axis = 1)
        x_test  = pd.concat([x_test,x_te_c_pca], axis = 1)

        x_train,x_valid,x_test = x_train.values,x_valid.values,x_test.values
        
        train_dataset = TrainDataset(x_train, y_train)
        valid_dataset = TrainDataset(x_valid, y_valid)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = Model(
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=hidden_size,
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                                  max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))

        loss_tr = SmoothBCEwLogits(smoothing = 0.001)
        loss_va = nn.BCEWithLogitsLoss()    

        early_stopping_steps = EARLY_STOPPING_STEPS
        early_step = 0

        oof = np.zeros((len(train), len(target_cols)))
        best_loss = np.inf

        mod_name = mod_path1 + f"FOLD_mod11_{seed}_{fold}_.pth"
        
        model.load_state_dict(torch.load(mod_name, map_location=torch.device('cpu')))
        model.to(DEVICE)
        
        oof[val_idx] = inference_fn(model, validloader, DEVICE)
        
        #--------------------- PREDICTION---------------------
        testdataset = TestDataset(x_test)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

        predictions = np.zeros((len(test_), len(target_cols)))
        predictions = inference_fn(model, testloader, DEVICE)
        return oof, predictions

    def run_k_fold(NFOLDS, seed):
        oof = np.zeros((len(train), len(target_cols)))
        predictions = np.zeros((len(test), len(target_cols)))

        for fold in range(NFOLDS):
            oof_, pred_ = run_training(fold, seed)

            predictions += pred_ / NFOLDS
            oof += oof_

        return oof, predictions

    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)
    
    oof_tmp = dp(oof)
    oof_tmp = oof_tmp * len(SEED) / (SEED.index(seed)+1)
    sc_dic[seed] = np.mean([log_loss(train[target_cols].iloc[:,i],oof_tmp[:,i]) for i in range(len(target_cols))])


print(np.mean([log_loss(train[target_cols].iloc[:,i],oof[:,i]) for i in range(len(target_cols))]))

train0[target_cols] = oof
test[target_cols] = predictions

sub = sample_submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

### mod1 ###
train0_1 = train0.copy()
sub_1 = sub.copy()

pd.DataFrame(sc_dic,index=['sc']).T


# head
SEED = [100]
# SEED = [100,101,102,103,104,105,106,107,108,109]

seed_everything(seed=42)

sc_dic = {}
feat_dic = {}
train_features = pd.read_csv(input_dir+'train_features.csv').sample(n=15000, random_state=42)
train_targets_scored = pd.read_csv(input_dir+'train_targets_scored.csv').sample(n=15000, random_state=42)
train_targets_nonscored = pd.read_csv(input_dir+'train_targets_nonscored.csv').sample(n=15000, random_state=42)
test_features = pd.read_csv(input_dir+'test_features.csv')
sample_submission = pd.read_csv(input_dir+'sample_submission.csv')
train_drug = pd.read_csv(input_dir+'train_drug.csv').sample(n=15000, random_state=42)

GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]
feat_dic['gene'] = GENES
feat_dic['cell'] = CELLS

# sample normalization 
q2 = train_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = train_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.75).copy()
qmean = (q2+q7)/2
train_features[feat_dic['gene']] = (train_features[feat_dic['gene']].T - qmean.values).T

q2 = test_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = test_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.75).copy()
qmean = (q2+q7)/2
test_features[feat_dic['gene']] = (test_features[feat_dic['gene']].T - qmean.values).T

q2 = train_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = train_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.72).copy()
qmean = (q2+q7)/2
train_features[feat_dic['cell']] = (train_features[feat_dic['cell']].T - qmean.values).T
qmean2 = train_features[feat_dic['cell']].abs().apply(np.quantile,axis = 1,q = 0.75).copy()+4
train_features[feat_dic['cell']] = (train_features[feat_dic['cell']].T / qmean2.values).T.copy()

q2 = test_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = test_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.72).copy()
qmean = (q2+q7)/2
test_features[feat_dic['cell']] = (test_features[feat_dic['cell']].T - qmean.values).T
qmean2 = test_features[feat_dic['cell']].abs().apply(np.quantile,axis = 1,q = 0.75).copy()+4
test_features[feat_dic['cell']] = (test_features[feat_dic['cell']].T / qmean2.values).T.copy()

def fe_stats(train, test):
    features_g = GENES
    features_c = CELLS

    feat_raw = train.columns
    for df in train, test:
        df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
        df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)

        df['c52_c42'] = df['c-52'] * df['c-42']
        df['c13_c73'] = df['c-13'] * df['c-73']
        df['c26_c13'] = df['c-23'] * df['c-13']
        df['c33_c6'] = df['c-33'] * df['c-6']
        df['c11_c55'] = df['c-11'] * df['c-55']
        df['c38_c63'] = df['c-38'] * df['c-63']
        df['c38_c94'] = df['c-38'] * df['c-94']
        df['c13_c94'] = df['c-13'] * df['c-94']
        df['c4_c52'] = df['c-4'] * df['c-52']
        df['c4_c42'] = df['c-4'] * df['c-42']
        df['c13_c38'] = df['c-13'] * df['c-38']
        df['c55_c2'] = df['c-55'] * df['c-2']
        df['c55_c4'] = df['c-55'] * df['c-4']
        df['c4_c13'] = df['c-4'] * df['c-13']
        df['c82_c42'] = df['c-82'] * df['c-42']
        df['c66_c42'] = df['c-66'] * df['c-42']
        df['c6_c38'] = df['c-6'] * df['c-38']
        df['c2_c13'] = df['c-2'] * df['c-13']
        df['c62_c42'] = df['c-62'] * df['c-42']
        df['c90_c55'] = df['c-90'] * df['c-55']      

    feat_new = train.columns
    feat_stat = list(set(feat_new) - set(feat_raw))
    feat_stat.sort()
    return train, test, feat_stat

train_features,test_features, feat_stat=fe_stats(train_features,test_features)
feat_dic['stat'] = feat_stat

# remove ctr
train = train_features.merge(train_targets_scored, on='sig_id')
train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

target = train[train_targets_scored.columns]

train0 = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)

target_cols = target.drop('sig_id', axis=1).columns.values.tolist()


# drug ids
tar_sig = target['sig_id'].tolist()
train_drug = train_drug.loc[[i in tar_sig for i in train_drug['sig_id']]]
target = target.merge(train_drug, on='sig_id', how='left') 

# LOCATE DRUGS
vc = train_drug.drug_id.value_counts()
vc1 = vc.loc[vc <= 19].index
vc2 = vc.loc[vc > 19].index

feature_cols = []
for key_i in feat_dic.keys():
    value_i = feat_dic[key_i]
    print(key_i,len(value_i))
    feature_cols += value_i
len(feature_cols)
feature_cols0 = dp(feature_cols)

oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

# Averaging on multiple SEEDS
for seed in SEED:
    
    seed_everything(seed=seed)
    folds = train0.copy()
    feature_cols = dp(feature_cols0)
    
    # Kfold - leave drug out
    target2 = target.copy()
    dct1 = {}; dct2 = {}
    skf = MultilabelStratifiedKFold(n_splits = 5) # , shuffle = True, random_state = seed
    tmp = target2.groupby('drug_id')[target_cols].mean().loc[vc1]
    tmp_idx = tmp.index.tolist()
    tmp_idx.sort()
    tmp_idx2 = random.sample(tmp_idx,len(tmp_idx))
    tmp = tmp.loc[tmp_idx2]
    for fold,(idxT,idxV) in enumerate(skf.split(tmp,tmp[target_cols])):
        dd = {k:fold for k in tmp.index[idxV].values}
        dct1.update(dd)

    skf = MultilabelStratifiedKFold(n_splits = 5) # , shuffle = True, random_state = seed
    tmp = target2.loc[target2.drug_id.isin(vc2)].reset_index(drop = True)
    tmp_idx = tmp.index.tolist()
    tmp_idx.sort()
    tmp_idx2 = random.sample(tmp_idx,len(tmp_idx))
    tmp = tmp.loc[tmp_idx2]
    for fold,(idxT,idxV) in enumerate(skf.split(tmp,tmp[target_cols])):
        dd = {k:fold for k in tmp.sig_id[idxV].values}
        dct2.update(dd)

    target2['kfold'] = target2.drug_id.map(dct1)
    target2.loc[target2.kfold.isna(),'kfold'] = target2.loc[target2.kfold.isna(),'sig_id'].map(dct2)
    target2.kfold = target2.kfold.astype(int)

    folds['kfold'] = target2['kfold'].copy()

    train = folds.copy()
    test_ = test.copy()

    # HyperParameters
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    NFOLDS = 5

    n_comp1 = 600
    n_comp2 = 50
 
    tar_freq = np.array([np.min(list(f_table(train[target_cols].iloc[:,i]).values())) for i in range(len(target_cols))])
    tar_weight0 = np.array([np.log(i+100) for i in tar_freq])
    tar_weight0_min = dp(np.min(tar_weight0))
    tar_weight = tar_weight0_min/tar_weight0
    np.mean(tar_weight)
    pos_weight = torch.tensor(tar_weight).to(DEVICE)
    
    wgt_bce = dp(F.binary_cross_entropy_with_logits)
    wgt_bce.__defaults__ = (None, None, None, 'mean', pos_weight)
    
    class SmoothBCEwLogits(_WeightedLoss):
        def __init__(self, weight=None, reduction='mean', smoothing=0.0):
            super().__init__(weight=weight, reduction=reduction)
            self.smoothing = smoothing
            self.weight = weight
            self.reduction = reduction

        @staticmethod
        def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
            assert 0 <= smoothing < 1
            with torch.no_grad():
                targets = targets * (1.0 - smoothing) + 0.5 * smoothing
            return targets

        def forward(self, inputs, targets):
            targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
                self.smoothing)
            loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight,
                                                      pos_weight = pos_weight)
            if  self.reduction == 'sum':
                loss = loss.sum()
            elif  self.reduction == 'mean':
                loss = loss.mean()

            return loss
            
    def run_training(fold, seed):

        seed_everything(seed)

        trn_idx = train[train['kfold'] != fold].index
        val_idx = train[train['kfold'] == fold].index

        train_df = train[train['kfold'] != fold].reset_index(drop=True).copy()
        valid_df = train[train['kfold'] == fold].reset_index(drop=True).copy()

        x_train, y_train  = train_df[feature_cols], train_df[target_cols].values
        x_valid, y_valid =  valid_df[feature_cols], valid_df[target_cols].values
        x_test = test_[feature_cols]

        #------------ norm --------------
        col_num = list(set(feat_dic['gene'] + feat_dic['cell']) & set(feature_cols))
        col_num.sort()
        x_train[col_num],ss = norm_fit(x_train[col_num],True,'quan')
        x_valid[col_num]    = norm_tra(x_valid[col_num],ss)
        x_test[col_num]     = norm_tra(x_test[col_num],ss)

        #------------ pca --------------
        def pca_pre(tr,va,te,
                    n_comp,feat_raw,feat_new):
            pca = PCA(n_components=n_comp, random_state=42)
            tr2 = pd.DataFrame(pca.fit_transform(tr[feat_raw]),columns=feat_new)
            va2 = pd.DataFrame(pca.transform(va[feat_raw]),columns=feat_new)
            te2 = pd.DataFrame(pca.transform(te[feat_raw]),columns=feat_new)
            return(tr2,va2,te2)

        pca_feat_g = [f'pca_G-{i}' for i in range(n_comp1)]
        feat_dic['pca_g'] = pca_feat_g
        x_tr_g_pca,x_va_g_pca,x_te_g_pca = pca_pre(x_train,x_valid,x_test,
                                                   n_comp1,feat_dic['gene'],pca_feat_g)
        x_train = pd.concat([x_train,x_tr_g_pca],axis = 1)
        x_valid = pd.concat([x_valid,x_va_g_pca],axis = 1)
        x_test  = pd.concat([x_test,x_te_g_pca],axis = 1)

        pca_feat_g = [f'pca_C-{i}' for i in range(n_comp2)]
        feat_dic['pca_c'] = pca_feat_g
        x_tr_c_pca,x_va_c_pca,x_te_c_pca = pca_pre(x_train,x_valid,x_test,
                                                   n_comp2,feat_dic['cell'],pca_feat_g)
        x_train = pd.concat([x_train,x_tr_c_pca],axis = 1)
        x_valid = pd.concat([x_valid,x_va_c_pca],axis = 1)
        x_test  = pd.concat([x_test,x_te_c_pca], axis = 1)

        #------------ var --------------
        var_thresh = VarianceThreshold(0.8)
        var_thresh.fit(x_train)
        x_train = x_train.loc[:,var_thresh.variances_ > 0.8]
        x_valid = x_valid.loc[:,var_thresh.variances_ > 0.8]
        x_test  = x_test.loc[:,var_thresh.variances_ > 0.8]

        x_train,x_valid,x_test = x_train.values,x_valid.values,x_test.values

        class LogitsLogLoss(Metric):
            """
            LogLoss with sigmoid applied
            """
            def __init__(self):
                self._name = "logits_ll"
                self._maximize = False

            def __call__(self, y_true, y_pred):
                """
                Compute LogLoss of predictions.

                Parameters
                ----------
                y_true: np.ndarray
                    Target matrix or vector
                y_score: np.ndarray
                    Score matrix or vector

                Returns
                -------
                    float
                    LogLoss of predictions vs targets.
                """
                logits = 1 / (1 + np.exp(-y_pred))
                aux = (1 - y_true) * np.log(1 - logits + 1e-15) + y_true * np.log(logits + 1e-15)
                return np.mean(-aux)

        MAX_EPOCH = 120
        # n_d and n_a are different from the original work, 32 instead of 24
        # This is the first change in the code from the original
        tabnet_params = dict(
            n_d = 64,
            n_a = 128,
            n_steps = 1,
            gamma = 1.3,
            lambda_sparse = 0,
            n_independent = 2,
            n_shared = 1,
            optimizer_fn = optim.Adam,
            optimizer_params = dict(lr = 2e-2, weight_decay = 1e-5),
            mask_type = "entmax",
            scheduler_params = dict(
                mode = "min", patience = 5, min_lr = 1e-5, factor = 0.9),
            scheduler_fn = ReduceLROnPlateau,
            seed = seed,
            verbose = 10
        )

        mod_path = mod_path2 + f"mod21_{seed}_{fold}_.pth.zip"
        model =  TabNetRegressor()
        model.load_model(mod_path)

        oof = np.zeros((len(train), len(target_cols)))
        valid_preds = 1 / (1 + np.exp(-model.predict(x_valid)))
        oof[val_idx] = valid_preds
        predictions = 1 / (1 + np.exp(-model.predict(x_test)))
        
        return oof, predictions

    def run_k_fold(NFOLDS, seed):
        oof = np.zeros((len(train), len(target_cols)))
        predictions = np.zeros((len(test), len(target_cols)))

        for fold in range(NFOLDS):
            oof_, pred_ = run_training(fold, seed)

            predictions += pred_ / NFOLDS
            oof += oof_

        return oof, predictions

    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)
    
    oof_tmp = dp(oof)
    oof_tmp = oof_tmp * len(SEED) / (SEED.index(seed)+1)
    sc_dic[seed] = np.mean([log_loss(train[target_cols].iloc[:,i],oof_tmp[:,i]) for i in range(len(target_cols))])

print(np.mean([log_loss(train[target_cols].iloc[:,i],oof[:,i]) for i in range(len(target_cols))]))

train0[target_cols] = oof
test[target_cols] = predictions

sub = sample_submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

### mod2 ###
train0_2 = train0.copy()
sub_2 = sub.copy()

pd.DataFrame(sc_dic,index=['sc']).T


# head 

SEED = [200]
# SEED = [200, 201, 202, 203 ,204, 205, 206, 207, 208, 209]

NFOLDS = 7

seed_everything(seed=42)

sc_dic = {}
feat_dic = {}
train_features = pd.read_csv(input_dir+'train_features.csv').sample(n=15000, random_state=42)
train_targets_scored = pd.read_csv(input_dir+'train_targets_scored.csv').sample(n=15000, random_state=42)
train_targets_nonscored = pd.read_csv(input_dir+'train_targets_nonscored.csv').sample(n=15000, random_state=42)
test_features = pd.read_csv(input_dir+'test_features.csv')
sample_submission = pd.read_csv(input_dir+'sample_submission.csv')
train_drug = pd.read_csv(input_dir+'train_drug.csv').sample(n=15000, random_state=42)

target_cols = train_targets_scored.drop('sig_id', axis=1).columns.values.tolist()
aux_target_cols = train_targets_nonscored.drop('sig_id', axis=1).columns.values.tolist()
all_target_cols = target_cols + aux_target_cols

GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]
feat_dic['gene'] = GENES
feat_dic['cell'] = CELLS

## sample normalization ##
q2 = train_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = train_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.75).copy()
qmean = (q2+q7)/2
train_features[feat_dic['gene']] = (train_features[feat_dic['gene']].T - qmean.values).T
q2 = test_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = test_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.75).copy()
qmean = (q2+q7)/2
test_features[feat_dic['gene']] = (test_features[feat_dic['gene']].T - qmean.values).T

q2 = train_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = train_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.72).copy()
qmean = (q2+q7)/2
train_features[feat_dic['cell']] = (train_features[feat_dic['cell']].T - qmean.values).T
qmean2 = train_features[feat_dic['cell']].abs().apply(np.quantile,axis = 1,q = 0.75).copy()+4
train_features[feat_dic['cell']] = (train_features[feat_dic['cell']].T / qmean2.values).T.copy()

q2 = test_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = test_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.72).copy()
qmean = (q2+q7)/2
test_features[feat_dic['cell']] = (test_features[feat_dic['cell']].T - qmean.values).T
qmean2 = test_features[feat_dic['cell']].abs().apply(np.quantile,axis = 1,q = 0.75).copy()+4
test_features[feat_dic['cell']] = (test_features[feat_dic['cell']].T / qmean2.values).T.copy()

# remove ctr
train = train_features.merge(train_targets_scored, on='sig_id')
train = train.merge(train_targets_nonscored, on='sig_id')
train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

target = train[train_targets_scored.columns]

train0 = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)

# drug ids
tar_sig = target['sig_id'].tolist()
train_drug = train_drug.loc[[i in tar_sig for i in train_drug['sig_id']]]
target = target.merge(train_drug, on='sig_id', how='left') 

# LOCATE DRUGS
vc = train_drug.drug_id.value_counts()
vc1 = vc.loc[vc <= 19].index
vc2 = vc.loc[vc > 19].index

feature_cols = []
for key_i in feat_dic.keys():
    value_i = feat_dic[key_i]
    print(key_i,len(value_i))
    feature_cols += value_i
len(feature_cols)
feature_cols0 = dp(feature_cols)
    
oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

# Averaging on multiple SEEDS
for seed in SEED:

    seed_everything(seed=seed)
    folds = train0.copy()
    feature_cols = dp(feature_cols0)
    
    # Kfold - leave drug out
    target2 = target.copy()
    dct1 = {}; dct2 = {}
    skf = MultilabelStratifiedKFold(n_splits = NFOLDS) # , shuffle = True, random_state = seed
    tmp = target2.groupby('drug_id')[target_cols].mean().loc[vc1]
    tmp_idx = tmp.index.tolist()
    tmp_idx.sort()
    tmp_idx2 = random.sample(tmp_idx,len(tmp_idx))
    tmp = tmp.loc[tmp_idx2]
    for fold,(idxT,idxV) in enumerate(skf.split(tmp,tmp[target_cols])):
        dd = {k:fold for k in tmp.index[idxV].values}
        dct1.update(dd)

    skf = MultilabelStratifiedKFold(n_splits = NFOLDS) # , shuffle = True, random_state = seed
    tmp = target2.loc[target2.drug_id.isin(vc2)].reset_index(drop = True)
    tmp_idx = tmp.index.tolist()
    tmp_idx.sort()
    tmp_idx2 = random.sample(tmp_idx,len(tmp_idx))
    tmp = tmp.loc[tmp_idx2]
    for fold,(idxT,idxV) in enumerate(skf.split(tmp,tmp[target_cols])):
        dd = {k:fold for k in tmp.sig_id[idxV].values}
        dct2.update(dd)

    target2['kfold'] = target2.drug_id.map(dct1)
    target2.loc[target2.kfold.isna(),'kfold'] = target2.loc[target2.kfold.isna(),'sig_id'].map(dct2)
    target2.kfold = target2.kfold.astype(int)

    folds['kfold'] = target2['kfold'].copy()

    train = folds.copy()
    test_ = test.copy()

    # HyperParameters
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 24
    BATCH_SIZE = 128

    WEIGHT_DECAY = {'ALL_TARGETS': 1e-5, 'SCORED_ONLY': 3e-6}
    MAX_LR = {'ALL_TARGETS': 1e-2, 'SCORED_ONLY': 3e-3}
    DIV_FACTOR = {'ALL_TARGETS': 1e3, 'SCORED_ONLY': 1e2}
    PCT_START = 0.1

    n_comp1 = 600
    n_comp2 = 50

    num_targets = len(target_cols)
    num_aux_targets = len(aux_target_cols)
    num_all_targets = len(all_target_cols)
    hidden_size=4096

    tar_freq = np.array([np.min(list(f_table(train[target_cols].iloc[:,i]).values())) for i in range(len(target_cols))])
    tar_weight0 = np.array([np.log(i+100) for i in tar_freq])
    tar_weight0_min = dp(np.min(tar_weight0))
    tar_weight = tar_weight0_min/tar_weight0
    pos_weight = torch.tensor(tar_weight).to(DEVICE)

    tar_freq = np.array([np.min(list(f_table(train[all_target_cols].iloc[:,i]).values())) for i in range(len(all_target_cols))])
    tar_weight0 = np.array([np.log(i+100) for i in tar_freq])
    tar_weight0_min = dp(np.min(tar_weight0))
    pos_weight_all = tar_weight0_min/tar_weight0
    pos_weight_all = torch.tensor(pos_weight_all).to(DEVICE)

    class SmoothBCEwLogits(_WeightedLoss):
        def __init__(self, weight=None, reduction='mean', smoothing=0.0,pos_weight = None):
            super().__init__(weight=weight, reduction=reduction)
            self.smoothing = smoothing
            self.weight = weight
            self.reduction = reduction
            self.pos_weight = pos_weight

        @staticmethod
        def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
            assert 0 <= smoothing < 1
            with torch.no_grad():
                targets = targets * (1.0 - smoothing) + 0.5 * smoothing
            return targets

        def forward(self, inputs, targets):
            targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
                self.smoothing)
            loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight,
                                                      pos_weight = self.pos_weight)

            if  self.reduction == 'sum':
                loss = loss.sum()
            elif  self.reduction == 'mean':
                loss = loss.mean()

            return loss

    class TrainDataset:
        def __init__(self, features, targets):
            self.features = features
            self.targets = targets

        def __len__(self):
            return (self.features.shape[0])

        def __getitem__(self, idx):
            dct = {
                'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
                'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            
            }
            return dct

    class TestDataset:
        def __init__(self, features):
            self.features = features

        def __len__(self):
            return (self.features.shape[0])

        def __getitem__(self, idx):
            dct = {
                'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
            }
            return dct


    def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
        model.train()
        final_loss = 0

        for data in dataloader:
            optimizer.zero_grad()
            inputs, targets = data['x'].to(device), data['y'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            final_loss += loss.item()

        final_loss /= len(dataloader)

        return final_loss

    def valid_fn(model, loss_fn, dataloader, device):
        model.eval()
        final_loss = 0
        valid_preds = []

        for data in dataloader:
            inputs, targets = data['x'].to(device), data['y'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            final_loss += loss.item()
            valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

        final_loss /= len(dataloader)
        valid_preds = np.concatenate(valid_preds)

        return final_loss, valid_preds

    def inference_fn(model, dataloader, device):
        model.eval()
        preds = []

        for data in dataloader:
            inputs = data['x'].to(device)
            with torch.no_grad():
                outputs = model(inputs)

            preds.append(outputs.sigmoid().detach().cpu().numpy())

        preds = np.concatenate(preds)

        return preds

    class Model(nn.Module):
        def __init__(self, num_features, num_targets):
            super(Model, self).__init__()
            self.hidden_size = [1500, 1250, 1000, 750]
            self.dropout_value = [0.5, 0.35, 0.3, 0.25]

            self.batch_norm1 = nn.BatchNorm1d(num_features)
            self.dense1 = nn.Linear(num_features, self.hidden_size[0])

            self.batch_norm2 = nn.BatchNorm1d(self.hidden_size[0])
            self.dropout2 = nn.Dropout(self.dropout_value[0])
            self.dense2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])

            self.batch_norm3 = nn.BatchNorm1d(self.hidden_size[1])
            self.dropout3 = nn.Dropout(self.dropout_value[1])
            self.dense3 = nn.Linear(self.hidden_size[1], self.hidden_size[2])

            self.batch_norm4 = nn.BatchNorm1d(self.hidden_size[2])
            self.dropout4 = nn.Dropout(self.dropout_value[2])
            self.dense4 = nn.Linear(self.hidden_size[2], self.hidden_size[3])

            self.batch_norm5 = nn.BatchNorm1d(self.hidden_size[3])
            self.dropout5 = nn.Dropout(self.dropout_value[3])
            self.dense5 = nn.utils.weight_norm(nn.Linear(self.hidden_size[3], num_targets))

        def forward(self, x):
            x = self.batch_norm1(x)
            x = F.leaky_relu(self.dense1(x))

            x = self.batch_norm2(x)
            x = self.dropout2(x)
            x = F.leaky_relu(self.dense2(x))

            x = self.batch_norm3(x)
            x = self.dropout3(x)
            x = F.leaky_relu(self.dense3(x))

            x = self.batch_norm4(x)
            x = self.dropout4(x)
            x = F.leaky_relu(self.dense4(x))

            x = self.batch_norm5(x)
            x = self.dropout5(x)
            x = self.dense5(x)
            return x
        
    class FineTuneScheduler:
        def __init__(self, epochs):
            self.epochs = epochs
            self.epochs_per_step = 0
            self.frozen_layers = []

        def copy_without_top(self, model, num_features, num_targets, num_targets_new):
            self.frozen_layers = []

            model_new = Model(num_features, num_targets)
            model_new.load_state_dict(model.state_dict())

            # Freeze all weights
            for name, param in model_new.named_parameters():
                layer_index = name.split('.')[0][-1]

                if layer_index == '5':
                    continue

                param.requires_grad = False

                # Save frozen layer names
                if layer_index not in self.frozen_layers:
                    self.frozen_layers.append(layer_index)

            self.epochs_per_step = self.epochs // len(self.frozen_layers)

            # Replace the top layers with another ones
            model_new.batch_norm5 = nn.BatchNorm1d(model_new.hidden_size[3])
            model_new.dropout5 = nn.Dropout(model_new.dropout_value[3])
            model_new.dense5 = nn.utils.weight_norm(nn.Linear(model_new.hidden_size[-1], num_targets_new))
            model_new.to(DEVICE)
            return model_new

        def step(self, epoch, model):
            if len(self.frozen_layers) == 0:
                return

            if epoch % self.epochs_per_step == 0:
                last_frozen_index = self.frozen_layers[-1]

                # Unfreeze parameters of the last frozen layer
                for name, param in model.named_parameters():
                    layer_index = name.split('.')[0][-1]

                    if layer_index == last_frozen_index:
                        param.requires_grad = True

                del self.frozen_layers[-1]  # Remove the last layer as unfrozen

    def run_training(fold, seed):

        seed_everything(seed)

        trn_idx = train[train['kfold'] != fold].index
        val_idx = train[train['kfold'] == fold].index

        train_df = train[train['kfold'] != fold].reset_index(drop=True).copy()
        valid_df = train[train['kfold'] == fold].reset_index(drop=True).copy()

        x_train, y_train, y_train_all  = train_df[feature_cols], train_df[target_cols].values, train_df[all_target_cols].values
        x_valid, y_valid, y_valid_all =  valid_df[feature_cols], valid_df[target_cols].values, valid_df[all_target_cols].values
        x_test = test_[feature_cols]

        #------------ norm --------------
        col_num = list(set(feat_dic['gene'] + feat_dic['cell']) & set(feature_cols))
        col_num.sort()
        x_train[col_num],ss = norm_fit(x_train[col_num],True,'quan')
        x_valid[col_num]    = norm_tra(x_valid[col_num],ss)
        x_test[col_num]     = norm_tra(x_test[col_num],ss)

        #------------ pca --------------
        def pca_pre(tr,va,te,
                    n_comp,feat_raw,feat_new):
            pca = PCA(n_components=n_comp, random_state=42)
            tr2 = pd.DataFrame(pca.fit_transform(tr[feat_raw]),columns=feat_new)
            va2 = pd.DataFrame(pca.transform(va[feat_raw]),columns=feat_new)
            te2 = pd.DataFrame(pca.transform(te[feat_raw]),columns=feat_new)
            return(tr2,va2,te2)

        pca_feat_g = [f'pca_G-{i}' for i in range(n_comp1)]
        feat_dic['pca_g'] = pca_feat_g
        x_tr_g_pca,x_va_g_pca,x_te_g_pca = pca_pre(x_train,x_valid,x_test,
                                                   n_comp1,feat_dic['gene'],pca_feat_g)
        x_train = pd.concat([x_train,x_tr_g_pca],axis = 1)
        x_valid = pd.concat([x_valid,x_va_g_pca],axis = 1)
        x_test  = pd.concat([x_test,x_te_g_pca],axis = 1)

        pca_feat_g = [f'pca_C-{i}' for i in range(n_comp2)]
        feat_dic['pca_c'] = pca_feat_g
        x_tr_c_pca,x_va_c_pca,x_te_c_pca = pca_pre(x_train,x_valid,x_test,
                                                   n_comp2,feat_dic['cell'],pca_feat_g)
        x_train = pd.concat([x_train,x_tr_c_pca],axis = 1)
        x_valid = pd.concat([x_valid,x_va_c_pca],axis = 1)
        x_test  = pd.concat([x_test,x_te_c_pca], axis = 1)

        #------------ var --------------
        var_thresh = VarianceThreshold(0.8)
        var_thresh.fit(x_train)
        x_train = x_train.loc[:,var_thresh.variances_ > 0.8]
        x_valid = x_valid.loc[:,var_thresh.variances_ > 0.8]
        x_test  = x_test.loc[:,var_thresh.variances_  > 0.8]

        num_features = x_train.shape[1]

        x_train,x_valid,x_test = x_train.values,x_valid.values,x_test.values

        def train_model(model, tag_name, target_cols_now, fine_tune_scheduler=None):
            if tag_name == 'ALL_TARGETS':
                train_dataset = TrainDataset(x_train, y_train_all)
                valid_dataset = TrainDataset(x_valid, y_valid_all)
            else:
                train_dataset = TrainDataset(x_train, y_train)
                valid_dataset = TrainDataset(x_valid, y_valid)

            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY[tag_name])
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                      steps_per_epoch=len(trainloader),
                                                      pct_start=PCT_START,
                                                      div_factor=DIV_FACTOR[tag_name], 
                                                      max_lr=MAX_LR[tag_name],
                                                      epochs=EPOCHS)

            if tag_name == 'ALL_TARGETS':
                loss_tr = SmoothBCEwLogits(smoothing=0.001,pos_weight = pos_weight_all)
            else:
                loss_tr = SmoothBCEwLogits(smoothing=0.001,pos_weight = pos_weight)
            loss_fn = nn.BCEWithLogitsLoss()

            oof = np.zeros((len(train), len(target_cols_now)))
            best_loss = np.inf

            mod_name = f"mod31_{tag_name}_{seed}_{fold}.pth"
            for epoch in range(EPOCHS):
                if fine_tune_scheduler is not None:
                    fine_tune_scheduler.step(epoch, model)
                train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, DEVICE)
                valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
                print(f"SEED: {seed}, FOLD: {fold}, {tag_name}, EPOCH: {epoch}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}")
                if np.isnan(valid_loss):
                    break
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    oof[val_idx] = valid_preds
                    torch.save(model.state_dict(),mod_name )
            return oof

        fine_tune_scheduler = FineTuneScheduler(EPOCHS)

        tag_name = 'SCORED_ONLY'
        mod_name = mod_path3 + f"mod31_{tag_name}_{seed}_{fold}.pth"
        # Load the fine-tuned model with the best loss
        model = Model(num_features, num_targets)
        model.load_state_dict(torch.load(mod_name, map_location=torch.device('cpu')))
        model.to(DEVICE)

        #--------------------- PREDICTION---------------------
        valid_dataset = TrainDataset(x_valid, y_valid)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
        oof = np.zeros((len(train), len(target_cols)))
        oof[val_idx] = inference_fn(model, validloader, DEVICE)
        
        testdataset = TestDataset(x_test)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
        predictions = np.zeros((len(test_), num_targets))
        predictions = inference_fn(model, testloader, DEVICE)
        return oof, predictions


    def run_k_fold(NFOLDS, seed):
        oof = np.zeros((len(train), len(target_cols)))
        predictions = np.zeros((len(test), len(target_cols)))

        for fold in range(NFOLDS):
            oof_, pred_ = run_training(fold, seed)

            predictions += pred_ / NFOLDS
            oof += oof_

        return oof, predictions

    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)
    
    oof_tmp = dp(oof)
    oof_tmp = oof_tmp * len(SEED) / (SEED.index(seed)+1)
    sc_dic[seed] = np.mean([log_loss(train[target_cols].iloc[:,i],oof_tmp[:,i]) for i in range(len(target_cols))])

print(np.mean([log_loss(train[target_cols].iloc[:,i],oof[:,i]) for i in range(len(target_cols))]))

train0[target_cols] = oof
test[target_cols] = predictions

sub = sample_submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

### mod3 ###
train0_3 = train0.copy()
sub_3 = sub.copy()

pd.DataFrame(sc_dic,index=['sc']).T


# 3 models weighted average
sub_wct = sub_1.copy()
sub_wct[target_cols] = sub_1[target_cols] * 0.65 + sub_2[target_cols] * 0.1 + sub_3[target_cols] * 0.25

# final submission
sub_wct.to_csv('submission_wct.csv', index=False)


# ğŸŒŸ MoA XGBoost Pipeline (15000 Sample + Feature Engineering + CV + Loss Plot)
# Optimized for Kaggle environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from xgboost import __version__ as xgb_version
from tqdm.auto import tqdm  # For progress tracking


# For balanced multilabel CV
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Parameters
USE_PRETRAINED_MODELS = True  # Set to True if you have pre-trained models
OUTPUT_SUBMISSION_PATH = "/kaggle/working/submission_xgb.csv"  # Output path within the Kaggle environment

# Check for GPU availability on Kaggle
import subprocess
try:
    gpu_info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
    gpu_available = True
    print("GPU is available on this Kaggle kernel!")
    print(gpu_info)
except:
    gpu_available = False
    print("No GPU available, will use CPU mode.")

# 1. Load data - Kaggle paths
print("Loading datasets...")
train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")
train_targets_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
train_targets_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")
test_features = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
sample_submission = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")

# 2. Sample 15000 training records
print("Sampling 15000 records for training...")
df_sample = train_features.sample(n=15000, random_state=42).reset_index(drop=True)
ids = df_sample['sig_id']
Y = train_targets_scored[train_targets_scored['sig_id'].isin(ids)].reset_index(drop=True)

df_valid = train_features[~train_features['sig_id'].isin(ids)].reset_index(drop=True)
Y_valid = train_targets_scored[train_targets_scored['sig_id'].isin(df_valid['sig_id'])].reset_index(drop=True)

# 3. Feature engineering
def preprocess_features(df):
    df = df.drop(columns=["sig_id"])
    df = pd.get_dummies(df, columns=["cp_type", "cp_dose", "cp_time"])
    return df

print("Preprocessing features...")
X = preprocess_features(df_sample)
X_test = preprocess_features(test_features)
X_valid = preprocess_features(df_valid)

# Feature scaling
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)

# 4. Set up Cross Validation
N_SPLITS = 3
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

models = {}
histories = {}

# Create directories for model storage - Kaggle paths
model_path = "/kaggle/input/moa-all-models/weights/output_param/xgb_models/param"
# os.makedirs(model_path, exist_ok=True)
# os.makedirs(f"{model_path}/param", exist_ok=True)  # For model parameters
# os.makedirs(f"{model_path}/lossplot", exist_ok=True)  # For loss plots

# Set GPU count based on Kaggle environment (typically has 1 GPU)
num_gpus = 1 if gpu_available else 0
if gpu_available:
    print(f"ğŸš€ Starting training with {num_gpus} GPU acceleration")
else:
    print("ğŸš€ Starting training with CPU")

start_time = time.time()

# Check XGBoost version
print(f"XGBoost version: {xgb_version}")

# Calculate total tasks for progress estimation
target_columns = Y.columns[1:]  # Skip 'sig_id'
total_targets = len(target_columns)
print(f"Total targets to train: {total_targets}")

# Track training times
target_times = []

# 5. Train models or load pre-trained models
if USE_PRETRAINED_MODELS:
    print("ğŸ”„ Using pre-trained models, loading from saved files...")
    
    # Check if model files exist
    model_exists = all(os.path.exists(f"{model_path}/xgb_{target}.pkl") for target in target_columns)
    
    if not model_exists:
        print("â�Œ Error: Not all model files found. Set USE_PRETRAINED_MODELS=False to retrain")
        exit(1)
    
    # Load pre-trained models
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='Loading models'):
        model_file = f"{model_path}/xgb_{target}.pkl"
        models[target] = joblib.load(model_file)
        print(f"Model loaded: {target}")
    
    print("âœ… All models loaded successfully")
else:
    # Normal training flow
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='Overall Progress'):
        target_start = time.time()
        
        print(f"\nTraining model ({i+1}/{total_targets}): {target}")
        y_target = Y[target].values
        
        # Use train_test_split for validation split
        indices = range(len(X))
        
        # Check class distribution to ensure at least 2 samples per class
        unique_values, counts = np.unique(y_target, return_counts=True)
        min_count = counts.min() if len(counts) > 0 else 0
        
        # Skip stratified sampling if any class has too few samples
        if min_count < 2 or len(unique_values) <= 1:
            print(f"Warning: Imbalanced data for target {target}, skipping stratification")
            stratify_data = None
        else:
            stratify_data = y_target
            
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=stratify_data)
        
        X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
        y_train, y_val = y_target[train_indices], y_target[val_indices]

        # Check for class imbalance
        pos_rate = np.mean(y_train)
        print(f"Positive rate: {pos_rate:.4f}")
        
        # Adjust weight for imbalanced data
        scale_pos_weight = 1
        if pos_rate < 0.2:
            scale_pos_weight = (1 - pos_rate) / pos_rate
            print(f"Imbalanced data, adjusting weight to: {scale_pos_weight:.2f}")

        # Current GPU selection (Kaggle typically has 1 GPU)
        current_gpu = 0 if gpu_available else None
        
        try:
            # Try GPU acceleration if available
            if gpu_available:
                model = XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric='logloss',
                    random_state=42,
                    verbosity=1,
                    tree_method="gpu_hist", # Use gpu_hist for Kaggle GPU
                    scale_pos_weight=scale_pos_weight
                )
                
                print(f"Training model using GPU...")
            else:
                # CPU configuration
                model = XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric='logloss',
                    random_state=42,
                    verbosity=1,
                    tree_method="hist",
                    scale_pos_weight=scale_pos_weight
                )
                
                print("Training model using CPU...")
                
            # Fit model
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False  # Don't output loss in terminal
            )
        except Exception as e:
            print(f"Training failed, falling back to CPU: {str(e)}")
            # Fallback to CPU
            model = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=42,
                verbosity=0,
                tree_method="hist",
                device="cpu",
                scale_pos_weight=scale_pos_weight
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )

        # Create loss plot
        try:
            results = model.evals_result()
            histories[target] = results

            plt.figure(figsize=(10, 6))
            plt.plot(results["validation_0"]["logloss"], label="Train")
            plt.plot(results["validation_1"]["logloss"], label="Valid")
            plt.title(f"Logloss Curve for {target}")
            plt.xlabel("Iterations")
            plt.ylabel("Logloss")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{model_path}/lossplot/loss_{target}.png")
            plt.close()
        except Exception as e:
            print(f"Failed to plot loss curve: {str(e)}")

        # Save model
        models[target] = model
        joblib.dump(model, f"{model_path}/param/xgb_{target}.pkl")
        
        # Calculate and display progress
        target_time = time.time() - target_start
        target_times.append(target_time)
        avg_time_per_target = np.mean(target_times)
        remaining_targets = total_targets - (i + 1)
        estimated_remaining_time = avg_time_per_target * remaining_targets
        
        # Format estimated time remaining
        remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
        completion_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)
        
        print(f"Target {i+1}/{total_targets} completed! ({target})")
        print(f"Average time per target: {avg_time_per_target:.2f} seconds")
        print(f"Estimated time remaining: {remaining_time_str}")
        print(f"Estimated completion time: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Final completion message
print("\nâœ… All models training completed!")
total_time = time.time() - start_time
print(f"Total time: {datetime.timedelta(seconds=int(total_time))}")

# 6. Make predictions and create submission file
print("ğŸ”® Starting prediction...")
predictions = []

# Show prediction progress
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='Prediction progress'):
    model = models[target]  # Use in-memory models to avoid reloading
    pred = model.predict_proba(X_test)[:, 1]
    predictions.append(pred)

predictions = np.array(predictions).T
submission = sample_submission.copy()
submission.iloc[:, 1:] = predictions
submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
print(f"ğŸ�‰ Submission file created: {OUTPUT_SUBMISSION_PATH}")

predictions_V = []

# Show prediction progress
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='Validation prediction progress'):
    model = models[target]  # Use in-memory models to avoid reloading
    pred = model.predict_proba(X_valid)[:, 1]
    predictions_V.append(pred)

predictions_V = np.array(predictions_V).T

# Create validation submission with correct sig_ids
validation_submission = pd.DataFrame(columns=Y_valid.columns)
validation_submission['sig_id'] = df_valid['sig_id']
for col in target_columns:
    validation_submission[col] = predictions_V[:, list(target_columns).index(col)]

# Save validation predictions to a separate file
VALIDATION_OUTPUT_PATH = "/kaggle/working/validation_predictions_xgb.csv"
validation_submission.to_csv(VALIDATION_OUTPUT_PATH, index=False)
print(f"ğŸ�‰ é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆå·²å»ºç«‹: {VALIDATION_OUTPUT_PATH}")

# æª¢è¦–é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆçš„å‰�å¹¾è¡Œ
print("\né©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆé �è¦½:")


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
import datetime
# from cuml.svm import SVC  # GPU ç‰ˆ
# from cuml import __version__ as cuml_version
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# import cupy as cp

# /kaggle/input/lish-moa
# ================= è·¯å¾‘è¨­å®šå�€ =================
DATA_ROOT = "/kaggle/input/lish-moa/"
MYMODELS_ROOT = "/kaggle/working/"
CSV_ROOT = "/kaggle/working/"
MODEL_OUTPUT_ROOT = "/kaggle/input/moa-all-models/svm_cpu_models"
TRAIN_FEATURES_PATH = os.path.join(DATA_ROOT, "train_features.csv")
TRAIN_TARGETS_SCORED_PATH = os.path.join(DATA_ROOT, "train_targets_scored.csv")
TRAIN_TARGETS_NONSCORED_PATH = os.path.join(DATA_ROOT, "train_targets_nonscored.csv")
TEST_FEATURES_PATH = os.path.join(DATA_ROOT, "test_features.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
OUTPUT_SUBMISSION_PATH = os.path.join(CSV_ROOT, "submission_svm.csv")
VALIDATION_OUTPUT_PATH = os.path.join(CSV_ROOT, "validation_predictions_svm.csv")
# =============================================

USE_PRETRAINED_MODELS = True

train_features = pd.read_csv(TRAIN_FEATURES_PATH)
train_targets_scored = pd.read_csv(TRAIN_TARGETS_SCORED_PATH)
train_targets_nonscored = pd.read_csv(TRAIN_TARGETS_NONSCORED_PATH)
test_features = pd.read_csv(TEST_FEATURES_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

df_sample = train_features.sample(n=15000, random_state=42).reset_index(drop=True)
ids = df_sample['sig_id']
Y = train_targets_scored[train_targets_scored['sig_id'].isin(ids)].reset_index(drop=True)

df_valid = train_features[~train_features['sig_id'].isin(ids)].reset_index(drop=True)
Y_valid = train_targets_scored[train_targets_scored['sig_id'].isin(df_valid['sig_id'])].reset_index(drop=True)

def preprocess_features(df):
    df = df.drop(columns=["sig_id"])
    df = pd.get_dummies(df, columns=["cp_type", "cp_dose", "cp_time"])
    return df

X = preprocess_features(df_sample)
X_test = preprocess_features(test_features)
X_valid = preprocess_features(df_valid)

print("ğŸ§  ä½¿ç”¨ CPU é€²è¡Œç‰¹å¾µæ¨™æº–åŒ–")
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

N_SPLITS = 3
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

models = {}
histories = {}

model_path = MODEL_OUTPUT_ROOT
# os.makedirs(model_path, exist_ok=True)
# os.makedirs(f"{model_path}/param", exist_ok=True)  # å„²å­˜æ¨¡å�‹å�ƒæ•¸çš„è³‡æ–™å¤¾
# os.makedirs(f"{model_path}/lossplot", exist_ok=True)  # å„²å­˜lossåœ–çš„è³‡æ–™å¤¾

print(f"ğŸš€ é–‹å§‹è¨“ç·´ - ä½¿ç”¨ CPU")
start_time = time.time()
# print(f"cuML ç‰ˆæœ¬: {cuml_version}")

target_columns = Y.columns[1:]
total_targets = len(target_columns)
print(f"ç¸½å…±éœ€è¦�è¨“ç·´ {total_targets} å€‹ç›®æ¨™")

target_times = []

def compute_logloss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


if USE_PRETRAINED_MODELS:
    print("ğŸ”„ ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹")
    for target in tqdm(target_columns, desc='è¼‰å…¥æ¨¡å�‹'):
        model_file = f"{model_path}/param/svm_{target}.pkl"
        models[target] = joblib.load(model_file)
    print("âœ… å·²æˆ�åŠŸè¼‰å…¥æ‰€æœ‰æ¨¡å�‹")
else:
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='æ•´é«”é€²åº¦'):
        target_start = time.time()
        print(f"\nè¨“ç·´æ¨¡å�‹ ({i+1}/{total_targets}): {target}")
        y_target = Y[target].values

        indices = range(len(X))
        unique_values, counts = np.unique(y_target, return_counts=True)
        min_count = counts.min() if len(counts) > 0 else 0
        stratify_data = y_target if min_count >= 2 and len(unique_values) > 1 else None

        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=stratify_data)
        X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
        y_train, y_val = y_target[train_indices], y_target[val_indices]

        print(f"æ­£ä¾‹æ¯”ä¾‹: {np.mean(y_train):.4f}")

        model = SVC(kernel='linear', C=1.0, probability=True)
        # å�¯ç°¡åŒ–ç‚ºï¼ˆä¸�è½‰å�‹ä¹Ÿå�¯ï¼‰ï¼š
        model.fit(X_train, y_train)
        train_proba = model.predict_proba(X_train)[:, 1]
        val_proba = model.predict_proba(X_val)[:, 1]


        train_loss = compute_logloss(np.array(y_train), np.array(train_proba))
        val_loss = compute_logloss(np.array(y_val), np.array(val_proba))
        print(f"è¨“ç·´ LogLoss: {train_loss:.4f} | é©—è­‰ LogLoss: {val_loss:.4f}")

        models[target] = model
        joblib.dump(model, f"{model_path}/param/svm_{target}.pkl")

        target_time = time.time() - target_start
        target_times.append(target_time)
        avg_time_per_target = np.mean(target_times)
        estimated_remaining_time = avg_time_per_target * (total_targets - i - 1)
        print(f"é �ä¼°å®Œæˆ�æ™‚é–“: {(datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)).strftime('%Y-%m-%d %H:%M:%S')}")

print("\nâœ… æ‰€æœ‰æ¨¡å�‹è¨“ç·´å®Œæˆ�")
total_time = time.time() - start_time
print(f"ç¸½è€—æ™‚: {datetime.timedelta(seconds=int(total_time))}")

print("ğŸ”® é–‹å§‹æ¸¬è©¦é›†é �æ¸¬")
predictions = []
for target in tqdm(target_columns, desc='é �æ¸¬é€²åº¦'):
    model = models[target]
    try:
        pred = model.predict_proba(X_test)[:, 1]
    except:
        pred = np.full(len(X_test), 0.5)
    predictions.append(pred)

predictions = np.array(predictions).T
submission = sample_submission.copy()
submission[target_columns] = predictions
submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
print("ğŸ“„ æ��äº¤æª”æ¡ˆå„²å­˜æ–¼:", OUTPUT_SUBMISSION_PATH)

print("âœ… é–‹å§‹é©—è­‰é›†é �æ¸¬")
predictions_V = []
for target in tqdm(target_columns, desc='Validation prediction'):
    model = models[target]
    try:
        pred = model.predict_proba(X_valid)[:, 1]

    except:
        pred = np.full(len(X_valid), 0.5)
    predictions_V.append(pred)

predictions_V = np.array(predictions_V).T
validation_submission = pd.DataFrame(columns=Y_valid.columns)
validation_submission['sig_id'] = df_valid['sig_id']
for col in target_columns:
    validation_submission[col] = predictions_V[:, list(target_columns).index(col)]
validation_submission.to_csv(VALIDATION_OUTPUT_PATH, index=False)
print(f"ğŸ�‰ é©—è­‰é �æ¸¬å®Œæˆ�: {VALIDATION_OUTPUT_PATH}")
print(validation_submission.head())


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from tqdm.auto import tqdm  # å¼•å…¥tqdmç”¨æ–¼é€²åº¦è¿½è¹¤
import multiprocessing as mp  # ç”¨æ–¼å¤šæ ¸å¿ƒåŠ é€Ÿ

# GPU è‡ªå‹•å�µæ¸¬
try:
    import pynvml
    pynvml.nvmlInit()
    NUM_GPUS = pynvml.nvmlDeviceGetCount()
    USE_GPU = NUM_GPUS > 0
    print(f"ğŸš€ å�µæ¸¬åˆ° {NUM_GPUS} é¡† GPUï¼ŒLightGBM å°‡ä½¿ç”¨ GPU åŠ é€Ÿè¨“ç·´ï¼�")
    print(f"ğŸš€ å�µæ¸¬åˆ° {NUM_GPUS} é¡† GPUï¼ŒLightGBM å°‡è‡ªå‹•åˆ†é…� GPU è¨“ç·´ï¼�")
except Exception:
    USE_GPU = False
    NUM_GPUS = 0
    print("ğŸ’» æœªå�µæ¸¬åˆ° GPUï¼Œå°‡ä½¿ç”¨ CPU é€²è¡Œ LightGBM è¨“ç·´")

# Optional: ç”¨æ–¼ balanced multilabel CV
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# è¨­å®šå�ƒæ•¸ - æ˜¯å�¦ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹
USE_PRETRAINED_MODELS = True  # è¨­ç‚ºTrueæ™‚ï¼Œå°‡å¾�output/paramè³‡æ–™å¤¾è¼‰å…¥å·²è¨“ç·´çš„æ¨¡å�‹

# ================= è·¯å¾‘è¨­å®šå�€ =================
DATA_ROOT = "/kaggle/input/lish-moa/"
MYMODELS_ROOT = "/kaggle/working/"
CSV_ROOT = "/kaggle/working/"
MODEL_OUTPUT_ROOT = "/kaggle/input/moa-all-models/weights/output_param/lgbm_models"
TRAIN_FEATURES_PATH = os.path.join(DATA_ROOT, "train_features.csv")
TRAIN_TARGETS_SCORED_PATH = os.path.join(DATA_ROOT, "train_targets_scored.csv")
TRAIN_TARGETS_NONSCORED_PATH = os.path.join(DATA_ROOT, "train_targets_nonscored.csv")
TEST_FEATURES_PATH = os.path.join(DATA_ROOT, "test_features.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
OUTPUT_SUBMISSION_PATH = os.path.join(CSV_ROOT, "submission_lgbm.csv")
VALIDATION_OUTPUT_PATH = os.path.join(CSV_ROOT, "validation_predictions_lgbm.csv")
# =============================================

# 1. è®€å�–è³‡æ–™
train_features = pd.read_csv(TRAIN_FEATURES_PATH)
train_targets_scored = pd.read_csv(TRAIN_TARGETS_SCORED_PATH)
train_targets_nonscored = pd.read_csv(TRAIN_TARGETS_NONSCORED_PATH)
test_features = pd.read_csv(TEST_FEATURES_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

# 2. æŠ½æ¨£ 15000 ç­†è¨“ç·´è³‡æ–™
df_sample = train_features.sample(n=15000, random_state=42).reset_index(drop=True)
ids = df_sample['sig_id']
Y = train_targets_scored[train_targets_scored['sig_id'].isin(ids)].reset_index(drop=True)
df_valid = train_features[~train_features['sig_id'].isin(ids)].reset_index(drop=True)
Y_valid = train_targets_scored[train_targets_scored['sig_id'].isin(df_valid['sig_id'])].reset_index(drop=True)

# 3. ç‰¹å¾µå·¥ç¨‹
def preprocess_features(df):
    df = df.drop(columns=["sig_id"])
    df = pd.get_dummies(df, columns=["cp_type", "cp_dose", "cp_time"])
    return df

X = preprocess_features(df_sample)
X_test = preprocess_features(test_features)
X_valid = preprocess_features(df_valid)

# æ¨™æº–åŒ–ç‰¹å¾µ
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)

# 4. Cross Validation è¨­å®šï¼ˆMultilabelStratifiedKFoldï¼‰
N_SPLITS = 3
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

models = {}
histories = {}

# å»ºç«‹å®Œæ•´è·¯å¾‘çš„å„²å­˜è³‡æ–™å¤¾
model_path = MODEL_OUTPUT_ROOT
# os.makedirs(model_path, exist_ok=True)
# os.makedirs(f"{model_path}/param", exist_ok=True)  # å„²å­˜æ¨¡å�‹å�ƒæ•¸çš„è³‡æ–™å¤¾
# os.makedirs(f"{model_path}/lossplot", exist_ok=True)  # å„²å­˜lossåœ–çš„è³‡æ–™å¤¾

# è¨­ç½®å�¯ç”¨æ ¸å¿ƒæ•¸é‡�
num_cores = mp.cpu_count()
print(f"ğŸš€ é–‹å§‹è¨“ç·´ - ä½¿ç”¨ {num_cores} å€‹ CPU æ ¸å¿ƒåŠ é€Ÿ")
start_time = time.time()

# é¡¯ç¤ºLightGBMç‰ˆæœ¬
lgb_version = lgb.__version__
print(f"LightGBM ç‰ˆæœ¬: {lgb_version}")

# è¨ˆç®—ç¸½ä»»å‹™æ•¸é‡�ï¼Œç”¨æ–¼é€²åº¦ä¼°ç®—
target_columns = Y.columns[1:]
total_targets = len(target_columns)
print(f"ç¸½å…±éœ€è¦�è¨“ç·´ {total_targets} å€‹ç›®æ¨™")

# è¿½è¹¤è¨“ç·´æ™‚é–“
target_times = []

class LGBMTracker:
    """ç”¨æ–¼è¿½è¹¤LightGBMè¨“ç·´é��ç¨‹çš„é¡�åˆ¥"""
    def __init__(self):
        self.train_losses = []
        self.valid_losses = []
    
    def add_loss(self, train_loss, valid_loss):
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)

def compute_logloss(y_true, y_pred):
    """è¨ˆç®—å°�æ•¸æ��å¤±"""
    # ç¢ºä¿�é �æ¸¬å€¼åœ¨ (0, 1) ä¹‹é–“ï¼Œé�¿å…�æ•¸å€¼å•�é¡Œ
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# 5. è¨“ç·´æ¨¡å�‹æˆ–è¼‰å…¥å·²æœ‰çš„æ¨¡å�‹
if USE_PRETRAINED_MODELS:
    print("ğŸ”„ ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹ï¼Œå¾�ä¿�å­˜çš„æª”æ¡ˆä¸­è¼‰å…¥...")
    
    # æª¢æŸ¥æ˜¯å�¦å­˜åœ¨æ¨¡å�‹æª”æ¡ˆ
    model_exists = all(os.path.exists(f"{model_path}/param/lgbm_{target}.pkl") for target in target_columns)
    
    if not model_exists:
        print("â�Œ éŒ¯èª¤: æ‰¾ä¸�åˆ°æ‰€æœ‰éœ€è¦�çš„æ¨¡å�‹æª”æ¡ˆï¼Œè«‹è¨­å®š USE_PRETRAINED_MODELS=False é‡�æ–°è¨“ç·´")
        exit(1)
    
    # è¼‰å…¥å·²è¨“ç·´çš„æ¨¡å�‹
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='è¼‰å…¥æ¨¡å�‹'):
        model_file = f"{model_path}/param/lgbm_{target}.pkl"
        models[target] = joblib.load(model_file)
        print(f"å·²è¼‰å…¥æ¨¡å�‹: {target}")
    
    print("âœ… å·²æˆ�åŠŸè¼‰å…¥æ‰€æœ‰æ¨¡å�‹")
else:
    # æ­£å¸¸è¨“ç·´æµ�ç¨‹
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='æ•´é«”é€²åº¦'):
        target_start = time.time()
        
        print(f"\nè¨“ç·´æ¨¡å�‹ ({i+1}/{total_targets}): {target}")
        y_target = Y[target].values
        fold = 0
        fold_logloss = []
        
        # ä½¿ç”¨ train_test_split å‡½æ•¸é€²è¡Œåˆ†å‰²
        indices = range(len(X))
        
        # æ›´åš´æ ¼æª¢æŸ¥é¡�åˆ¥åˆ†å¸ƒï¼Œç¢ºä¿�æ¯�å€‹é¡�åˆ¥è‡³å°‘æœ‰ 2 ç­†è³‡æ–™
        unique_values, counts = np.unique(y_target, return_counts=True)
        min_count = counts.min() if len(counts) > 0 else 0
        
        # å¦‚æ�œä»»ä½•é¡�åˆ¥çš„æ¨£æœ¬æ•¸å°‘æ–¼ 2 æˆ–è€…å�ªæœ‰ä¸€å€‹é¡�åˆ¥ï¼Œå‰‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£
        if (min_count < 2 or len(unique_values) <= 1):
            print(f"è­¦å‘Š: ç›®æ¨™ {target} çš„è³‡æ–™åˆ†å¸ƒä¸�å�‡è¡¡ï¼ŒæŸ�é¡�åˆ¥æ¨£æœ¬æ•¸é��å°‘ï¼Œå°‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£")
            stratify_data = None
        else:
            stratify_data = y_target
            
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=stratify_data)
        
        X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
        y_train, y_val = y_target[train_indices], y_target[val_indices]

        # æª¢æŸ¥é¡�åˆ¥æ˜¯å�¦ä¸�å¹³è¡¡
        pos_rate = np.mean(y_train)
        print(f"æ­£ä¾‹æ¯”ä¾‹: {pos_rate:.4f}")
        
        # é‡�å°�ä¸�å¹³è¡¡è³‡æ–™èª¿æ•´æ¬Šé‡�
        scale_pos_weight = 1
        if (pos_rate < 0.2):
            scale_pos_weight = (1 - pos_rate) / pos_rate
            print(f"è³‡æ–™ä¸�å¹³è¡¡ï¼Œèª¿æ•´æ¬Šé‡�ç‚º: {scale_pos_weight:.2f}")
        
        tracker = LGBMTracker()
        eval_results = {}
        
        try:
            # å»ºç«‹ LightGBM åˆ†é¡�å™¨
            print("è¨“ç·´æ¨¡å�‹ä¸­...", end=" ")
            lgbm_params = dict(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                min_data_in_leaf=1,
                min_child_samples=1,
                min_gain_to_split=0,
                random_state=42,
                n_jobs=-1
            )
            if USE_GPU:
                lgbm_params['device'] = 'gpu'
                lgbm_params['gpu_device_id'] = i % NUM_GPUS
                print(f"[GPU {lgbm_params['gpu_device_id']}]", end=" ")
            model = lgb.LGBMClassifier(**lgbm_params)
            # fit æ™‚è½‰ float32
            model.fit(
                X_train.astype(np.float32) if USE_GPU else X_train,
                y_train,
                eval_set=[(X_train.astype(np.float32) if USE_GPU else X_train, y_train), (X_val.astype(np.float32) if USE_GPU else X_val, y_val)],
                eval_metric='logloss',
                callbacks=[lgb.log_evaluation(0)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            # ç�²å�–æ¯�æ¬¡è¿­ä»£çš„è©•ä¼°çµ�æ�œ
            if hasattr(model, 'evals_result_'):
                eval_results = model.evals_result_
            
            # è¨ˆç®—è¨“ç·´é›†å’Œé©—è­‰é›†çš„æ��å¤±
            train_proba = model.predict_proba(X_train)[:, 1]
            val_proba = model.predict_proba(X_val)[:, 1]
            
            train_loss = compute_logloss(y_train, train_proba)
            val_loss = compute_logloss(y_val, val_proba)
            
            print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            
            tracker.add_loss(train_loss, val_loss)
            
        except Exception as e:
            print(f"LightGBM è¨“ç·´å¤±æ•—: {str(e)}")
            # å˜—è©¦ä½¿ç”¨è¼ƒç°¡å–®çš„å�ƒæ•¸
            print("å˜—è©¦ä½¿ç”¨ç°¡åŒ–çš„LightGBMæ¨¡å�‹...", end=" ")
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                num_leaves=16,
                scale_pos_weight=scale_pos_weight,
                min_data_in_leaf=1,
                min_child_samples=1,
                min_gain_to_split=0,
                random_state=42,
                n_jobs=-1
            )
            # å�ªå‚³å¿…è¦�å�ƒæ•¸ï¼Œç§»é™¤ verbose/early_stopping_rounds
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='logloss',
                callbacks=[lgb.log_evaluation(0)]
            )
            
            # è¨ˆç®—è¨“ç·´é›†å’Œé©—è­‰é›†çš„æ��å¤±
            train_proba = model.predict_proba(X_train)[:, 1]
            val_proba = model.predict_proba(X_val)[:, 1]
            
            train_loss = compute_logloss(y_train, train_proba)
            val_loss = compute_logloss(y_val, val_proba)
            
            print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            
            tracker.add_loss(train_loss, val_loss)

        # # ç¹ªè£½ Loss plot
        # try:
        #     histories[target] = tracker
            
        #     plt.figure(figsize=(10, 6))
            
        #     # å˜—è©¦ç¹ªè£½è¨“ç·´é��ç¨‹ä¸­çš„æ��å¤±æ›²ç·š
        #     if eval_results and 'train' in eval_results and 'logloss' in eval_results['train']:
        #         train_curve = eval_results['train']['logloss']
        #         valid_curve = eval_results['valid']['logloss']
                
        #         plt.plot(train_curve, label="Train")
        #         plt.plot(valid_curve, label="Valid")
        #         plt.title(f"LightGBM Logloss Curve for {target}")
        #         plt.xlabel("Iterations")
        #     else:
        #         # å¦‚æ�œæ²’æœ‰è¨“ç·´é��ç¨‹çš„æ›²ç·šï¼Œå°±ç¹ªè£½æœ€çµ‚çµ�æ�œ
        #         plt.plot([train_loss], label="Train")
        #         plt.plot([val_loss], label="Valid")
        #         plt.title(f"Logloss for {target}")
        #         plt.xlabel("Model")
            
        #     plt.ylabel("Logloss")
        #     plt.legend()
        #     plt.grid()
        #     plt.tight_layout()
        #     plt.savefig(f"{model_path}/lossplot/loss_{target}.png")
        #     plt.close()
        # except Exception as e:
        #     print(f"ç„¡æ³•ç¹ªè£½æ��å¤±æ›²ç·š: {str(e)}")

        # å„²å­˜æ¨¡å�‹
        models[target] = model
        joblib.dump(model, f"{model_path}/param/lgbm_{target}.pkl")
        
        # è¨ˆç®—ä¸¦é¡¯ç¤ºé€²åº¦
        target_time = time.time() - target_start
        target_times.append(target_time)
        avg_time_per_target = np.mean(target_times)
        remaining_targets = total_targets - (i + 1)
        estimated_remaining_time = avg_time_per_target * remaining_targets
        
        # è½‰æ�›é �ä¼°å‰©é¤˜æ™‚é–“ç‚ºæ›´æ˜“è®€æ ¼å¼�
        remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
        completion_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)
        
        print(f"ç›®æ¨™ {i+1}/{total_targets} å·²å®Œæˆ�! ({target})")
        print(f"å¹³å�‡æ¯�å€‹ç›®æ¨™è¨“ç·´æ™‚é–“: {avg_time_per_target:.2f} ç§’")
        print(f"é �ä¼°å‰©é¤˜æ™‚é–“: {remaining_time_str}")
        print(f"é �ä¼°å®Œæˆ�æ™‚é–“: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")

# æœ€çµ‚å®Œæˆ�è¨Šæ�¯
print("\nâœ… æ‰€æœ‰æ¨¡å�‹è¨“ç·´å·²å®Œæˆ�!")
total_time = time.time() - start_time
print(f"ç¸½è€—æ™‚: {datetime.timedelta(seconds=int(total_time))}")

# 6. æ�¨è«–èˆ‡å»ºç«‹æ��äº¤æª”æ¡ˆ
print("ğŸ”® é–‹å§‹é€²è¡Œé �æ¸¬...")
predictions = []

# é¡¯ç¤ºé �æ¸¬é€²åº¦
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='é �æ¸¬é€²åº¦'):
    model = models[target]  # å„ªå…ˆä½¿ç”¨è¨˜æ†¶é«”ä¸­çš„æ¨¡å�‹é�¿å…�é‡�è¤‡è¼‰å…¥
    pred = model.predict_proba(X_test)[:, 1]
    predictions.append(pred)

predictions = np.array(predictions).T
submission = sample_submission.copy()
submission.iloc[:, 1:] = predictions
submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
print("ğŸ�‰ å·²ç”¢å‡º submission_lgbm.csv å�¯ç›´æ�¥ä¸Šå‚³åˆ° Kaggle")

# æ·»åŠ é©—è­‰é›†çš„é �æ¸¬
print("ğŸ”� é–‹å§‹é€²è¡Œé©—è­‰é›†é �æ¸¬...")
predictions_V = []

# Show prediction progress
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='Validation prediction progress'):
    model = models[target]  # Use in-memory models to avoid reloading
    pred = model.predict_proba(X_valid)
    # å�‹æ…‹è‡ªå‹•è™•ç�†ï¼šDataFrameã€�cupyã€�numpy
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 1].values
    elif hasattr(pred, 'get'):
        pred = pred.get()
        pred = pred[:, 1]
    else:
        pred = pred[:, 1]
    predictions_V.append(pred)

predictions_V = np.array(predictions_V).T

# Create validation submission with correct sig_ids
validation_submission = pd.DataFrame(columns=Y_valid.columns)
validation_submission['sig_id'] = df_valid['sig_id']
for col in target_columns:
    validation_submission[col] = predictions_V[:, list(target_columns).index(col)]

# Save validation predictions to a separate file
validation_submission.to_csv(VALIDATION_OUTPUT_PATH, index=False)
print(f"ğŸ�‰ LightGBM é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆå·²å»ºç«‹: {VALIDATION_OUTPUT_PATH}")

# æª¢è¦–é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆçš„å‰�å¹¾è¡Œ
# print("\né©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆé �è¦½:")
# print(validation_submission.head())

# # æ¸¬è©¦æ˜¯å�¦å�¯æˆ�åŠŸåœ¨ GPU ä¸Šè¨“ç·´
# if USE_GPU:
#     print("\nğŸ”� åŸ·è¡Œ GPU æ¸¬è©¦...")
#     try:
#         # å»ºç«‹ä¸€å€‹å°�çš„æ¸¬è©¦æ•¸æ“šé›†
#         X_test_gpu = np.array([[0,0],[1,1]])
#         y_test_gpu = np.array([0,1])
        
#         # æ¸¬è©¦ GPU è¨“ç·´
#         test_model = lgb.LGBMClassifier(device='gpu')
#         test_model.fit(X_test_gpu, y_test_gpu)
#         print("âœ… GPU æ¸¬è©¦æˆ�åŠŸï¼�æ‚¨çš„ LightGBM å·²ç¶“è¨­ç½®å¥½ä½¿ç”¨ GPU åŠ é€Ÿã€‚")
#     except Exception as e:
#         print(f"â�Œ GPU æ¸¬è©¦å¤±æ•—: {e}")
#         print("å¦‚æ�œæ‚¨ç¢ºå®šæœ‰ GPUï¼Œè«‹ç¢ºèª�å·²å®‰è£� GPU ç‰ˆæœ¬çš„ LightGBM:")
#         print("conda install -c conda-forge lightgbm cudatoolkit=11.0")
#         print("æˆ–è€…:")
#         print("pip install lightgbm --install-option=--gpu")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time
import datetime
# print(f"scikit-learn ç‰ˆæœ¬: {sklearn_version}")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import __version__ as sklearn_version
from sklearn.calibration import CalibratedClassifierCV
from tqdm.auto import tqdm  # å¼•å…¥tqdmç”¨æ–¼é€²åº¦è¿½è¹¤
import multiprocessing as mp  # ç”¨æ–¼å¤šæ ¸å¿ƒåŠ é€Ÿ

# Optional: ç”¨æ–¼ balanced multilabel CV
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# ================= è·¯å¾‘è¨­å®šå�€ =================
DATA_ROOT = "/kaggle/input/lish-moa/"
MYMODELS_ROOT = "/kaggle/working/"
CSV_ROOT = "/kaggle/working/"
MODEL_OUTPUT_ROOT = "/kaggle/input/moa-all-models/new_rf/"
TRAIN_FEATURES_PATH = os.path.join(DATA_ROOT, "train_features.csv")
TRAIN_TARGETS_SCORED_PATH = os.path.join(DATA_ROOT, "train_targets_scored.csv")
TRAIN_TARGETS_NONSCORED_PATH = os.path.join(DATA_ROOT, "train_targets_nonscored.csv")
TEST_FEATURES_PATH = os.path.join(DATA_ROOT, "test_features.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
OUTPUT_SUBMISSION_PATH = os.path.join(CSV_ROOT, "submission_rf.csv")
VALIDATION_OUTPUT_PATH = os.path.join(CSV_ROOT, "validation_predictions_rf.csv")
# =============================================
USE_PRETRAINED_MODELS = True
# 1. è®€å�–è³‡æ–™
train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")
train_targets_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
train_targets_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")
test_features = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
sample_submission = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")

# 2. æŠ½æ¨£ 15000 ç­†è¨“ç·´è³‡æ–™
df_sample = train_features.sample(n=15000, random_state=42).reset_index(drop=True)
ids = df_sample['sig_id']
Y = train_targets_scored[train_targets_scored['sig_id'].isin(ids)].reset_index(drop=True)

# 3. ç‰¹å¾µå·¥ç¨‹
def preprocess_features(df):
    df = df.drop(columns=["sig_id"])
    df = pd.get_dummies(df, columns=["cp_type", "cp_dose", "cp_time"])
    return df

X = preprocess_features(df_sample)
X_test = preprocess_features(test_features)

df_valid = train_features[~train_features['sig_id'].isin(ids)].reset_index(drop=True)
Y_valid = train_targets_scored[train_targets_scored['sig_id'].isin(df_valid['sig_id'])].reset_index(drop=True)
X_valid = preprocess_features(df_valid)

# æ¨™æº–åŒ–ç‰¹å¾µ
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)

# 4. Cross Validation è¨­å®šï¼ˆMultilabelStratifiedKFoldï¼‰
N_SPLITS = 3
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

models = {}
histories = {}

# å»ºç«‹å®Œæ•´è·¯å¾‘çš„å„²å­˜è³‡æ–™å¤¾
model_path = MODEL_OUTPUT_ROOT
# os.makedirs(model_path, exist_ok=True)
# os.makedirs(f"{model_path}/param", exist_ok=True)  # å„²å­˜æ¨¡å�‹å�ƒæ•¸çš„è³‡æ–™å¤¾
# os.makedirs(f"{model_path}/lossplot", exist_ok=True)  # å„²å­˜lossåœ–çš„è³‡æ–™å¤¾

# è¨­ç½®å�¯ç”¨æ ¸å¿ƒæ•¸é‡�
num_cores = mp.cpu_count()
print(f"ğŸš€ é–‹å§‹è¨“ç·´ - ä½¿ç”¨ {num_cores} å€‹ CPU æ ¸å¿ƒåŠ é€Ÿ")
start_time = time.time()

# æª¢æŸ¥ scikit-learn ç‰ˆæœ¬
print(f"scikit-learn ç‰ˆæœ¬: {sklearn_version}")

# è¨ˆç®—ç¸½ä»»å‹™æ•¸é‡�ï¼Œç”¨æ–¼é€²åº¦ä¼°ç®—
target_columns = Y.columns[1:]
total_targets = len(target_columns)
print(f"ç¸½å…±éœ€è¦�è¨“ç·´ {total_targets} å€‹ç›®æ¨™")

# è¿½è¹¤è¨“ç·´æ™‚é–“
target_times = []

class RFTracker:
    """ç”¨æ–¼è¿½è¹¤Random Forestè¨“ç·´é��ç¨‹çš„é¡�åˆ¥"""
    def __init__(self):
        self.train_losses = []
        self.valid_losses = []
    
    def add_loss(self, train_loss, valid_loss):
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)

def compute_logloss(y_true, y_pred):
    """è¨ˆç®—å°�æ•¸æ��å¤±"""
    # ç¢ºä¿�é �æ¸¬å€¼åœ¨ (0, 1) ä¹‹é–“ï¼Œé�¿å…�æ•¸å€¼å•�é¡Œ
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# GPU è‡ªå‹•å�µæ¸¬
try:
    import cupy as cp
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    NUM_GPUS = cp.cuda.runtime.getDeviceCount()
    USE_GPU = NUM_GPUS > 0
    print(f"ğŸš€ å�µæ¸¬åˆ° {NUM_GPUS} é¡† GPUï¼Œå°‡å„ªå…ˆä½¿ç”¨ cuML GPU RandomForestï¼�")
except Exception:
    USE_GPU = False
    NUM_GPUS = 0
    print("ğŸ’» æœªå�µæ¸¬åˆ° GPU æˆ– cuMLï¼Œå°‡ä½¿ç”¨ sklearn CPU RandomForest")

# 5. è¨“ç·´æ¨¡å�‹æˆ–è¼‰å…¥å·²æœ‰çš„æ¨¡å�‹
if USE_PRETRAINED_MODELS:
    print("ğŸ”„ ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹ï¼Œå¾�ä¿�å­˜çš„æª”æ¡ˆä¸­è¼‰å…¥...")
    
    # æª¢æŸ¥æ˜¯å�¦å­˜åœ¨æ¨¡å�‹æª”æ¡ˆ
    model_exists = all(os.path.exists(f"{model_path}/rf_{target}.pkl") for target in target_columns)
    
    if not model_exists:
        print("â�Œ éŒ¯èª¤: æ‰¾ä¸�åˆ°æ‰€æœ‰éœ€è¦�çš„æ¨¡å�‹æª”æ¡ˆï¼Œè«‹è¨­å®š USE_PRETRAINED_MODELS=False é‡�æ–°è¨“ç·´")
        exit(1)
    
    try:
        # è¼‰å…¥å·²è¨“ç·´çš„æ¨¡å�‹
        for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='è¼‰å…¥æ¨¡å�‹'):
            model_file = f"{model_path}/rf_{target}.pkl"
            try:
                models[target] = joblib.load(model_file)
                print(f"å·²è¼‰å…¥æ¨¡å�‹: {target}")
            except Exception as e:
                print(f"â�Œ è¼‰å…¥æ¨¡å�‹ {target} å¤±æ•—: {e}")
                print("âš ï¸� æª¢æŸ¥åˆ°æ¨¡å�‹æª”æ¡ˆèˆ‡ç›®å‰� scikit-learn ç‰ˆæœ¬ä¸�ç›¸å®¹ï¼Œå°‡è‡ªå‹•åˆ‡æ�›ç‚ºé‡�æ–°è¨“ç·´æ¨¡å¼�ã€‚")
                USE_PRETRAINED_MODELS = False
                break
        if USE_PRETRAINED_MODELS:
            print("âœ… å·²æˆ�åŠŸè¼‰å…¥æ‰€æœ‰æ¨¡å�‹")
    except Exception as e:
        print(f"â�Œ è¼‰å…¥æ¨¡å�‹æ™‚ç™¼ç”ŸéŒ¯èª¤: {e}")
        print("âš ï¸� æª¢æŸ¥åˆ°æ¨¡å�‹æª”æ¡ˆèˆ‡ç›®å‰� scikit-learn ç‰ˆæœ¬ä¸�ç›¸å®¹ï¼Œå°‡è‡ªå‹•åˆ‡æ�›ç‚ºé‡�æ–°è¨“ç·´æ¨¡å¼�ã€‚")
        USE_PRETRAINED_MODELS = False

if not USE_PRETRAINED_MODELS:
    # æ­£å¸¸è¨“ç·´æµ�ç¨‹
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='æ•´é«”é€²åº¦'):
        target_start = time.time()
        
        print(f"\nè¨“ç·´æ¨¡å�‹ ({i+1}/{total_targets}): {target}")
        y_target = Y[target].values
        fold = 0
        fold_logloss = []
        
        # ä½¿ç”¨ train_test_split å‡½æ•¸é€²è¡Œåˆ†å‰²
        indices = np.arange(len(X))  # sklearn 1.2 éœ€è¦� array-like

        # æ›´åš´æ ¼æª¢æŸ¥é¡�åˆ¥åˆ†å¸ƒï¼Œç¢ºä¿�æ¯�å€‹é¡�åˆ¥è‡³å°‘æœ‰ 2 ç­†è³‡æ–™
        unique_values, counts = np.unique(y_target, return_counts=True)
        min_count = counts.min() if len(counts) > 0 else 0

        # å¦‚æ�œä»»ä½•é¡�åˆ¥çš„æ¨£æœ¬æ•¸å°‘æ–¼ 2 æˆ–è€…å�ªæœ‰ä¸€å€‹é¡�åˆ¥ï¼Œå‰‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£
        if min_count < 2 or len(unique_values) <= 1:
            print(f"è­¦å‘Š: ç›®æ¨™ {target} çš„è³‡æ–™åˆ†å¸ƒä¸�å�‡è¡¡ï¼ŒæŸ�é¡�åˆ¥æ¨£æœ¬æ•¸é��å°‘ï¼Œå°‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£")
            stratify_data = None
        else:
            stratify_data = y_target

        # sklearn 1.2: indices å¿…é ˆæ˜¯ array-like
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=stratify_data)

        X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
        y_train, y_val = y_target[train_indices], y_target[val_indices]

        # æª¢æŸ¥é¡�åˆ¥æ˜¯å�¦ä¸�å¹³è¡¡
        pos_rate = np.mean(y_train)
        print(f"æ­£ä¾‹æ¯”ä¾‹: {pos_rate:.4f}")
        
        # é‡�å°�ä¸�å¹³è¡¡è³‡æ–™èª¿æ•´æ¬Šé‡�
        class_weight = None
        if pos_rate < 0.2 or pos_rate > 0.8:
            weight_ratio = (1 - pos_rate) / pos_rate if pos_rate < 0.5 else pos_rate / (1 - pos_rate)
            class_weight = {0: 1, 1: weight_ratio} if pos_rate < 0.5 else {0: weight_ratio, 1: 1}
            print(f"è³‡æ–™ä¸�å¹³è¡¡ï¼Œèª¿æ•´æ¬Šé‡�ç‚º: {weight_ratio}")
        
        tracker = RFTracker()
        
        try:
            if USE_GPU:
                gpu_device = i % NUM_GPUS
                print(f"[GPU {gpu_device}] cuML RF è¨“ç·´ä¸­...", end=" ")
                with cp.cuda.Device(gpu_device):
                    try:
                        # èª¿æ•´å�ƒæ•¸ä»¥æ��é«˜ç©©å®šæ€§
                        model = cuRF(
                            n_estimators=100,  # æ¸›å°‘æ¨¹çš„æ•¸é‡�
                            max_depth=6,       # æ¸›å°‘æ¨¹çš„æ·±åº¦
                            n_streams=1,       # å�ªç”¨ä¸€å€‹æµ�ï¼Œå¢�åŠ ç©©å®šæ€§
                            max_features=0.8,  # é™�åˆ¶ç‰¹å¾µæ•¸é‡�
                            # max_samples=0.8,  # sklearn 1.2 ä¸�æ”¯æ�´æ­¤å�ƒæ•¸ï¼Œç§»é™¤
                            random_state=42,
                            handle=None
                        )
                        # ç¢ºä¿�è³‡æ–™æ˜¯é€£çºŒçš„è¨˜æ†¶é«”å¡Š
                        X_train_gpu = X_train.values.astype(np.float32, order='C')
                        X_val_gpu = X_val.values.astype(np.float32, order='C')
                        
                        # é‡‹æ”¾ä¸€äº›è¨˜æ†¶é«”
                        cp.get_default_memory_pool().free_all_blocks()
                        
                        # è¨“ç·´æ¨¡å�‹
                        model.fit(X_train_gpu, y_train)
                        train_proba = model.predict_proba(X_train_gpu)[:, 1]
                        val_proba = model.predict_proba(X_val_gpu)[:, 1]
                        
                        if hasattr(train_proba, 'get'):
                            train_proba = train_proba.get()
                            val_proba = val_proba.get()
                    except Exception as gpu_error:
                        print(f"\nâš ï¸� GPUè¨“ç·´å¤±æ•—: {str(gpu_error)}")
                        print("å˜—è©¦ä½¿ç”¨æ›´ç°¡å–®çš„GPUè¨­ç½®...")
                        
                        # é‡‹æ”¾è¨˜æ†¶é«”
                        cp.get_default_memory_pool().free_all_blocks()
                        
                        # å˜—è©¦ä½¿ç”¨æ›´ç°¡å–®çš„è¨­ç½®
                        model = cuRF(
                            n_estimators=50,
                            max_depth=4,
                            n_streams=1,
                            max_features=0.6,
                            # max_samples=0.6,  # sklearn 1.2 ä¸�æ”¯æ�´æ­¤å�ƒæ•¸ï¼Œç§»é™¤
                            random_state=42,
                            handle=None
                        )
                        
                        # æ¸›å°‘è³‡æ–™é‡�
                        if len(X_train) > 5000:
                            X_train_sample = X_train.sample(n=5000, random_state=42)
                            y_train_sample = y_train[X_train_sample.index]
                            X_train_gpu = X_train_sample.values.astype(np.float32, order='C')
                        else:
                            X_train_gpu = X_train.values.astype(np.float32, order='C')
                            y_train_sample = y_train
                            
                        X_val_gpu = X_val.values.astype(np.float32, order='C')
                        
                        model.fit(X_train_gpu, y_train_sample)
                        train_proba = model.predict_proba(X_train_gpu)[:, 1]
                        val_proba = model.predict_proba(X_val_gpu)[:, 1]
                        
                        if hasattr(train_proba, 'get'):
                            train_proba = train_proba.get()
                            val_proba = val_proba.get()
            else:
                print("sklearn RF è¨“ç·´ä¸­...", end=" ")
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    class_weight=class_weight,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                train_proba = model.predict_proba(X_train)[:, 1]
                val_proba = model.predict_proba(X_val)[:, 1]
            train_loss = compute_logloss(y_train, train_proba)
            val_loss = compute_logloss(y_val, val_proba)
            print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            tracker.add_loss(train_loss, val_loss)
            
        except Exception as e:
            print(f"Random Forest è¨“ç·´å¤±æ•—: {str(e)}")
            # å˜—è©¦ä½¿ç”¨è¼ƒç°¡å–®çš„éš¨æ©Ÿæ£®æ�—æ¨¡å�‹
            print("å˜—è©¦ä½¿ç”¨ç°¡åŒ–çš„éš¨æ©Ÿæ£®æ�—æ¨¡å�‹...", end=" ")
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=4,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # è¨ˆç®—è¨“ç·´é›†å’Œé©—è­‰é›†çš„æ��å¤±
            train_proba = model.predict_proba(X_train)[:, 1]
            val_proba = model.predict_proba(X_val)[:, 1]
            
            train_loss = compute_logloss(y_train, train_proba)
            val_loss = compute_logloss(y_val, val_proba)
            
            print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            
            tracker.add_loss(train_loss, val_loss)

        # ç¹ªè£½ Loss plot
        try:
            histories[target] = tracker
            
            plt.figure(figsize=(10, 6))
            plt.plot([train_loss], label="Train")
            plt.plot([val_loss], label="Valid")
            plt.title(f"Logloss for {target}")
            plt.xlabel("Model")
            plt.ylabel("Logloss")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{model_path}/lossplot/loss_{target}.png")
            plt.close()
        except Exception as e:
            print(f"ç„¡æ³•ç¹ªè£½æ��å¤±æ›²ç·š: {str(e)}")

        # å„²å­˜æ¨¡å�‹
        models[target] = model
        joblib.dump(model, f"{model_path}/param/rf_{target}.pkl")
        
        # è¨ˆç®—ä¸¦é¡¯ç¤ºé€²åº¦
        target_time = time.time() - target_start
        target_times.append(target_time)
        avg_time_per_target = np.mean(target_times)
        remaining_targets = total_targets - (i + 1)
        estimated_remaining_time = avg_time_per_target * remaining_targets
        
        # è½‰æ�›é �ä¼°å‰©é¤˜æ™‚é–“ç‚ºæ›´æ˜“è®€æ ¼å¼�
        remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
        completion_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)
        
        print(f"ç›®æ¨™ {i+1}/{total_targets} å·²å®Œæˆ�! ({target})")
        print(f"å¹³å�‡æ¯�å€‹ç›®æ¨™è¨“ç·´æ™‚é–“: {avg_time_per_target:.2f} ç§’")
        print(f"é �ä¼°å‰©é¤˜æ™‚é–“: {remaining_time_str}")
        print(f"é �ä¼°å®Œæˆ�æ™‚é–“: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")

# æœ€çµ‚å®Œæˆ�è¨Šæ�¯
print("\nâœ… æ‰€æœ‰æ¨¡å�‹è¨“ç·´å·²å®Œæˆ�!")
total_time = time.time() - start_time
print(f"ç¸½è€—æ™‚: {datetime.timedelta(seconds=int(total_time))}")

# 6. æ�¨è«–èˆ‡å»ºç«‹æ��äº¤æª”æ¡ˆ
print("ğŸ”® é–‹å§‹é€²è¡Œé �æ¸¬...")
predictions = []

# é¡¯ç¤ºé �æ¸¬é€²åº¦
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='é �æ¸¬é€²åº¦'):
    model = models[target]  # å„ªå…ˆä½¿ç”¨è¨˜æ†¶é«”ä¸­çš„æ¨¡å�‹é�¿å…�é‡�è¤‡è¼‰å…¥
    
    try:
        # ä½¿ç”¨ GPU æ¨¡å�‹æ™‚éœ€è¦�ç‰¹æ®Šè™•ç�†
        if USE_GPU and hasattr(model, 'predict_proba') and 'cuml' in str(type(model)):
            # æ¸…ç�† GPU è¨˜æ†¶é«”
            cp.get_default_memory_pool().free_all_blocks()
            
            # åˆ†æ‰¹é �æ¸¬ä»¥é�¿å…�è¨˜æ†¶é«”å•�é¡Œ
            BATCH_SIZE = 5000
            all_preds = []
            
            for start_idx in range(0, len(X_test), BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, len(X_test))
                X_batch = X_test.iloc[start_idx:end_idx].values.astype(np.float32)
                
                with cp.cuda.Device(i % NUM_GPUS):  # é�¸æ“‡ GPU
                    batch_pred = model.predict_proba(X_batch)[:, 1]
                    if hasattr(batch_pred, 'get'):
                        batch_pred = batch_pred.get()
                    all_preds.append(batch_pred)
            
            # å�ˆä½µæ‰€æœ‰æ‰¹æ¬¡çš„é �æ¸¬çµ�æ�œ
            pred = np.concatenate(all_preds)
        else:
            # CPU æ¨¡å�‹ç›´æ�¥é �æ¸¬
            pred = model.predict_proba(X_test)[:, 1]
            
        predictions.append(pred)
    except Exception as e:
        print(f"\nâš ï¸� ç›®æ¨™ {target} çš„é �æ¸¬å¤±æ•—: {str(e)}")
        print("å˜—è©¦ä½¿ç”¨å‚™ç”¨æ–¹æ³•...")
        
        # ä½¿ç”¨ä¿�å®ˆçš„æ–¹æ³•å†�è©¦ä¸€æ¬¡
        try:
            if USE_GPU:
                # å˜—è©¦è½‰æ�›å›� CPU æ¨¡å�‹
                from sklearn.ensemble import RandomForestClassifier
                cpu_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1
                )
                # å¾�ç�¾æœ‰æ¨¡å�‹è¨“ç·´
                cpu_model.fit(X.values, Y[target].values)
                pred = cpu_model.predict_proba(X_test)[:, 1]
            else:
                # ä½¿ç”¨å�‡å€¼ä½œç‚ºé �æ¸¬ (æœ€å¾Œçš„é�¸é …)
                mean_val = Y[target].mean()
                pred = np.full(len(X_test), mean_val)
                
            print(f"å·²ä½¿ç”¨å‚™ç”¨æ–¹æ³•ç‚ºç›®æ¨™ {target} ç”Ÿæˆ�é �æ¸¬")
            predictions.append(pred)
        except Exception as backup_error:
            print(f"å‚™ç”¨æ–¹æ³•ä¹Ÿå¤±æ•—äº†: {str(backup_error)}")
            # ä½¿ç”¨å…¨ 0.5 é �æ¸¬
            pred = np.full(len(X_test), 0.5)
            predictions.append(pred)

predictions = np.array(predictions).T
submission = sample_submission.copy()
submission.iloc[:, 1:] = predictions
submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
print("ğŸ�‰ å·²ç”¢å‡º submission_rf.csv å�¯ç›´æ�¥ä¸Šå‚³åˆ° Kaggle")

# æ·»åŠ é©—è­‰é›†çš„é �æ¸¬
print("ğŸ”� é–‹å§‹é€²è¡Œé©—è­‰é›†é �æ¸¬...")
predictions_V = []

# Show prediction progress
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='Validation prediction progress'):
    try:
        model = models[target]  # Use in-memory models to avoid reloading
        
        # ä½¿ç”¨ GPU æ¨¡å�‹æ™‚éœ€è¦�ç‰¹æ®Šè™•ç�†
        if USE_GPU and hasattr(model, 'predict_proba') and 'cuml' in str(type(model)):
            # æ¸…ç�† GPU è¨˜æ†¶é«”
            cp.get_default_memory_pool().free_all_blocks()
            
            # åˆ†æ‰¹é �æ¸¬ä»¥é�¿å…�è¨˜æ†¶é«”å•�é¡Œ
            BATCH_SIZE = 5000
            all_preds = []
            
            for start_idx in range(0, len(X_valid), BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, len(X_valid))
                X_batch = X_valid.iloc[start_idx:end_idx].values.astype(np.float32)
                
                with cp.cuda.Device(i % NUM_GPUS):  # é�¸æ“‡ GPU
                    batch_pred = model.predict_proba(X_batch)
                    if isinstance(batch_pred, pd.DataFrame):
                        batch_pred = batch_pred.iloc[:, 1].values
                    elif hasattr(batch_pred, 'get'):
                        batch_pred = batch_pred.get()
                        batch_pred = batch_pred[:, 1]
                    else:
                        batch_pred = batch_pred[:, 1]
                    all_preds.append(batch_pred)
            
            # å�ˆä½µæ‰€æœ‰æ‰¹æ¬¡çš„é �æ¸¬çµ�æ�œ
            pred = np.concatenate(all_preds)
        else:
            # CPU æ¨¡å�‹ç›´æ�¥é �æ¸¬
            pred = model.predict_proba(X_valid)
            # å�‹æ…‹è‡ªå‹•è™•ç�†ï¼šDataFrameã€�cupyã€�numpy
            if isinstance(pred, pd.DataFrame):
                pred = pred.iloc[:, 1].values
            elif hasattr(pred, 'get'):
                pred = pred.get()
                pred = pred[:, 1]
            else:
                pred = pred[:, 1]
                
        predictions_V.append(pred)
    except Exception as e:
        print(f"\nâš ï¸� é©—è­‰é›†ç›®æ¨™ {target} çš„é �æ¸¬å¤±æ•—: {str(e)}")
        print("å˜—è©¦ä½¿ç”¨å‚™ç”¨æ–¹æ³•...")
        
        try:
            # ä½¿ç”¨å¹³å�‡å€¼ä½œç‚ºé �æ¸¬
            mean_val = Y[target].mean()
            pred = np.full(len(X_valid), mean_val)
            print(f"å·²ä½¿ç”¨å¹³å�‡å€¼ {mean_val:.4f} ä½œç‚ºç›®æ¨™ {target} çš„é �æ¸¬")
            predictions_V.append(pred)
        except Exception as backup_error:
            # ç·Šæ€¥å¾Œå‚™é�¸é …
            print(f"å‚™ç”¨æ–¹æ³•ä¹Ÿå¤±æ•—äº†ï¼š{str(backup_error)}")
            pred = np.full(len(X_valid), 0.5)
            predictions_V.append(pred)

predictions_V = np.array(predictions_V).T

# Create validation submission with correct sig_ids
validation_submission = pd.DataFrame(columns=Y_valid.columns)
validation_submission['sig_id'] = df_valid['sig_id']
for col in target_columns:
    validation_submission[col] = predictions_V[:, list(target_columns).index(col)]

# Save validation predictions to a separate file

validation_submission.to_csv(VALIDATION_OUTPUT_PATH, index=False)
print(f"ğŸ�‰ RandomForest é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆå·²å»ºç«‹: {VALIDATION_OUTPUT_PATH}")

# æª¢è¦–é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆçš„å‰�å¹¾è¡Œ
print("\né©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆé �è¦½:")
print(validation_submission.head())


import tensorflow
print(tensorflow.__version__)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tqdm.auto import tqdm  # å¼•å…¥tqdmç”¨æ–¼é€²åº¦è¿½è¹¤
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Optional: ç”¨æ–¼ balanced multilabel CV
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# è¨­å®šå�ƒæ•¸ - æ˜¯å�¦ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹
USE_PRETRAINED_MODELS = True  # è¨­ç‚ºTrueæ™‚ï¼Œå°‡å¾�output/paramè³‡æ–™å¤¾è¼‰å…¥å·²è¨“ç·´çš„æ¨¡å�‹

# ================= è·¯å¾‘è¨­å®šå�€ =================
DATA_ROOT = "/kaggle/input/lish-moa/"
MYMODELS_ROOT = "/kaggle/working/"
CSV_ROOT = "/kaggle/working/"
MODEL_OUTPUT_ROOT = "/kaggle/input/moa-all-models/weights/output_param/nn_models"
TRAIN_FEATURES_PATH = os.path.join(DATA_ROOT, "train_features.csv")
TRAIN_TARGETS_SCORED_PATH = os.path.join(DATA_ROOT, "train_targets_scored.csv")
TRAIN_TARGETS_NONSCORED_PATH = os.path.join(DATA_ROOT, "train_targets_nonscored.csv")
TEST_FEATURES_PATH = os.path.join(DATA_ROOT, "test_features.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
OUTPUT_SUBMISSION_PATH = os.path.join(CSV_ROOT, "submission_nn.csv")
VALIDATION_OUTPUT_PATH = os.path.join(CSV_ROOT, "validation_predictions_nn.csv")
# =============================================

# æª¢æŸ¥æ˜¯å�¦æœ‰å�¯ç”¨GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # è¨­å®šè¨˜æ†¶é«”å¢�é•·é™�åˆ¶
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"ğŸš€ ä½¿ç”¨GPUåŠ é€Ÿ: {len(gpus)}å€‹GPUå�¯ç”¨")
    except RuntimeError as e:
        print(f"GPUè¨­å®šå¤±æ•—: {e}")
else:
    print("âš ï¸� æ²’æœ‰å�¯ç”¨çš„GPUï¼Œå°‡ä½¿ç”¨CPUé€²è¡Œè¨“ç·´ï¼Œé€Ÿåº¦è¼ƒæ…¢")

# 1. è®€å�–è³‡æ–™
train_features = pd.read_csv(TRAIN_FEATURES_PATH)
train_targets_scored = pd.read_csv(TRAIN_TARGETS_SCORED_PATH)
train_targets_nonscored = pd.read_csv(TRAIN_TARGETS_NONSCORED_PATH)
test_features = pd.read_csv(TEST_FEATURES_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

# 2. æŠ½æ¨£ 15000 ç­†è¨“ç·´è³‡æ–™
df_sample = train_features.sample(n=15000, random_state=42).reset_index(drop=True)
ids = df_sample['sig_id']
Y = train_targets_scored[train_targets_scored['sig_id'].isin(ids)].reset_index(drop=True)
df_valid = train_features[~train_features['sig_id'].isin(ids)].reset_index(drop=True)
Y_valid = train_targets_scored[train_targets_scored['sig_id'].isin(df_valid['sig_id'])].reset_index(drop=True)

# 3. ç‰¹å¾µå·¥ç¨‹
def preprocess_features(df):
    df = df.drop(columns=["sig_id"])
    df = pd.get_dummies(df, columns=["cp_type", "cp_dose", "cp_time"])
    return df

X = preprocess_features(df_sample)
X_test = preprocess_features(test_features)
X_valid = preprocess_features(df_valid)
# æ¨™æº–åŒ–ç‰¹å¾µ
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns).astype(np.float32)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns).astype(np.float32)
X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns).astype(np.float32)

# 4. Cross Validation è¨­å®šï¼ˆMultilabelStratifiedKFoldï¼‰
N_SPLITS = 3
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

models = {}
histories = {}

# å»ºç«‹å®Œæ•´è·¯å¾‘çš„å„²å­˜è³‡æ–™å¤¾
model_path = MODEL_OUTPUT_ROOT
# os.makedirs(model_path, exist_ok=True)
# os.makedirs(f"{model_path}/param", exist_ok=True)  # å„²å­˜æ¨¡å�‹å�ƒæ•¸çš„è³‡æ–™å¤¾
# os.makedirs(f"{model_path}/lossplot", exist_ok=True)  # å„²å­˜lossåœ–çš„è³‡æ–™å¤¾

# é¡¯ç¤ºTensorFlowç‰ˆæœ¬
tf_version = tf.__version__
print(f"TensorFlow ç‰ˆæœ¬: {tf_version}")
start_time = time.time()

# è¨ˆç®—ç¸½ä»»å‹™æ•¸é‡�ï¼Œç”¨æ–¼é€²åº¦ä¼°ç®—
target_columns = Y.columns[1:]
total_targets = len(target_columns)
print(f"ç¸½å…±éœ€è¦�è¨“ç·´ {total_targets} å€‹ç›®æ¨™")

# è¿½è¹¤è¨“ç·´æ™‚é–“
target_times = []

class NNTracker:
    """ç”¨æ–¼è¿½è¹¤ç¥�ç¶“ç¶²çµ¡è¨“ç·´é��ç¨‹çš„é¡�åˆ¥"""
    def __init__(self):
        self.history = None
    
    def add_history(self, history):
        self.history = history

def compute_logloss(y_true, y_pred):
    """è¨ˆç®—å°�æ•¸æ��å¤±"""
    # ç¢ºä¿�é �æ¸¬å€¼åœ¨ (0, 1) ä¹‹é–“ï¼Œé�¿å…�æ•¸å€¼å•�é¡Œ
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def build_model(input_dim, learning_rate=0.001):
    """å»ºç«‹ç¥�ç¶“ç¶²çµ¡æ¨¡å�‹"""
    model = Sequential([
        # è¼¸å…¥å±¤
        Dense(256, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # éš±è—�å±¤1
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # éš±è—�å±¤2
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # è¼¸å‡ºå±¤ (äºŒå…ƒåˆ†é¡�)
        Dense(1, activation='sigmoid')
    ])
    
    # ç·¨è­¯æ¨¡å�‹
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 5. è¨“ç·´æ¨¡å�‹æˆ–è¼‰å…¥å·²æœ‰çš„æ¨¡å�‹
if USE_PRETRAINED_MODELS:
    print("ğŸ”„ ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹ï¼Œå¾�ä¿�å­˜çš„æª”æ¡ˆä¸­è¼‰å…¥...")
    
    # æª¢æŸ¥æ˜¯å�¦å­˜åœ¨æ¨¡å�‹æª”æ¡ˆ
    model_exists = all(os.path.exists(f"{model_path}/param/nn_{target}.h5") for target in target_columns)
    
    if not model_exists:
        print("â�Œ éŒ¯èª¤: æ‰¾ä¸�åˆ°æ‰€æœ‰éœ€è¦�çš„æ¨¡å�‹æª”æ¡ˆï¼Œè«‹è¨­å®š USE_PRETRAINED_MODELS=False é‡�æ–°è¨“ç·´")
        exit(1)
    
    # è¼‰å…¥å·²è¨“ç·´çš„æ¨¡å�‹
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='è¼‰å…¥æ¨¡å�‹'):
        model_file = f"{model_path}/param/nn_{target}.h5"
        models[target] = load_model(model_file)
        print(f"å·²è¼‰å…¥æ¨¡å�‹: {target}")
    
    print("âœ… å·²æˆ�åŠŸè¼‰å…¥æ‰€æœ‰æ¨¡å�‹")
else:
    # æ­£å¸¸è¨“ç·´æµ�ç¨‹
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='æ•´é«”é€²åº¦'):
        target_start = time.time()
        
        print(f"\nè¨“ç·´æ¨¡å�‹ ({i+1}/{total_targets}): {target}")
        y_target = Y[target].values
        fold = 0
        fold_logloss = []
        
        # ä½¿ç”¨ train_test_split å‡½æ•¸é€²è¡Œåˆ†å‰²
        indices = range(len(X))
        
        # æ›´åš´æ ¼æª¢æŸ¥é¡�åˆ¥åˆ†å¸ƒï¼Œç¢ºä¿�æ¯�å€‹é¡�åˆ¥è‡³å°‘æœ‰ 2 ç­†è³‡æ–™
        unique_values, counts = np.unique(y_target, return_counts=True)
        min_count = counts.min() if len(counts) > 0 else 0
        
        # å¦‚æ�œä»»ä½•é¡�åˆ¥çš„æ¨£æœ¬æ•¸å°‘æ–¼ 2 æˆ–è€…å�ªæœ‰ä¸€å€‹é¡�åˆ¥ï¼Œå‰‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£
        if min_count < 2 or len(unique_values) <= 1:
            print(f"è­¦å‘Š: ç›®æ¨™ {target} çš„è³‡æ–™åˆ†å¸ƒä¸�å�‡è¡¡ï¼ŒæŸ�é¡�åˆ¥æ¨£æœ¬æ•¸é��å°‘ï¼Œå°‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£")
            stratify_data = None
        else:
            stratify_data = y_target
            
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=stratify_data)
        
        X_train, X_val = X.iloc[train_indices].values, X.iloc[val_indices].values
        y_train, y_val = y_target[train_indices], y_target[val_indices]

        # æª¢æŸ¥é¡�åˆ¥æ˜¯å�¦ä¸�å¹³è¡¡
        pos_rate = np.mean(y_train)
        print(f"æ­£ä¾‹æ¯”ä¾‹: {pos_rate:.4f}")
        
        # é‡�å°�ä¸�å¹³è¡¡è³‡æ–™èª¿æ•´æ¬Šé‡�
        class_weight = None
        if pos_rate < 0.2 or pos_rate > 0.8:
            weight_ratio = (1 - pos_rate) / pos_rate if pos_rate < 0.5 else pos_rate / (1 - pos_rate)
            # ç‚ºkerasæº–å‚™é¡�åˆ¥æ¬Šé‡�
            class_weight = {0: 1, 1: weight_ratio} if pos_rate < 0.5 else {0: weight_ratio, 1: 1}
            print(f"è³‡æ–™ä¸�å¹³è¡¡ï¼Œèª¿æ•´æ¬Šé‡�ç‚º: {weight_ratio:.2f}")
        
        tracker = NNTracker()
        
        # è¨­å®šæ—©å�œå�ƒæ•¸
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # è¨­å®šå­¸ç¿’ç�‡æ¸›å°‘ç­–ç•¥
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
        
        try:
            # ä½¿ç”¨ç¥�ç¶“ç¶²çµ¡åˆ†é¡�å™¨
            print(f"è¨“ç·´æ¨¡å�‹ä¸­...", end=" ")
            # ç�²å�–ç‰¹å¾µæ•¸é‡�
            input_dim = X_train.shape[1]
            
            # å‰µå»ºæ¨¡å�‹
            model = build_model(input_dim)
            
            # è¨“ç·´æ¨¡å�‹
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                class_weight=class_weight,
                verbose=0  # ä¸�é¡¯ç¤ºè¨“ç·´é€²åº¦
            )
            
            # è¨ˆç®—è¨“ç·´é›†å’Œé©—è­‰é›†çš„æ��å¤±
            train_proba = model.predict(X_train, verbose=0).flatten()
            val_proba = model.predict(X_val, verbose=0).flatten()
            
            train_loss = compute_logloss(y_train, train_proba)
            val_loss = compute_logloss(y_val, val_proba)
            
            print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            
            tracker.add_history(history)
            
        except Exception as e:
            print(f"ç¥�ç¶“ç¶²çµ¡ è¨“ç·´å¤±æ•—: {str(e)}")
            # å˜—è©¦ä½¿ç”¨è¼ƒç°¡å–®çš„ç¥�ç¶“ç¶²çµ¡æ¨¡å�‹
            print("å˜—è©¦ä½¿ç”¨ç°¡åŒ–çš„ç¥�ç¶“ç¶²çµ¡æ¨¡å�‹...", end=" ")
            
            # å‰µå»ºç°¡å–®æ¨¡å�‹
            model = Sequential([
                Dense(64, input_dim=X_train.shape[1], activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.01),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # ç°¡å–®è¨“ç·´
            history = model.fit(
                X_train, y_train,
                epochs=30,
                batch_size=64,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                class_weight=class_weight,
                verbose=0
            )
            
            # è¨ˆç®—è¨“ç·´é›†å’Œé©—è­‰é›†çš„æ��å¤±
            train_proba = model.predict(X_train, verbose=0).flatten()
            val_proba = model.predict(X_val, verbose=0).flatten()
            
            train_loss = compute_logloss(y_train, train_proba)
            val_loss = compute_logloss(y_val, val_proba)
            
            print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            
            tracker.add_history(history)

        # ç¹ªè£½ Loss plot
        try:
            histories[target] = tracker
            
            if hasattr(tracker, 'history') and tracker.history is not None:
                history = tracker.history.history
                
                plt.figure(figsize=(12, 5))
                
                # ç¹ªè£½æ��å¤±æ›²ç·š
                plt.subplot(1, 2, 1)
                plt.plot(history['loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Validation Loss')
                plt.title(f'Loss Curve for {target}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                # ç¹ªè£½æº–ç¢ºç�‡æ›²ç·š
                plt.subplot(1, 2, 2)
                plt.plot(history['accuracy'], label='Train Accuracy')
                plt.plot(history['val_accuracy'], label='Validation Accuracy')
                plt.title('Accuracy Curve')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f"{model_path}/lossplot/loss_{target}.png")
                plt.close()
            else:
                # å¦‚æ�œæ²’æœ‰è¨“ç·´é��ç¨‹çš„æ›²ç·šï¼Œå°±ç¹ªè£½æœ€çµ‚çµ�æ�œ
                plt.figure(figsize=(10, 6))
                plt.bar(['Train Loss', 'Valid Loss'], [train_loss, val_loss])
                plt.title(f"Final Loss for {target}")
                plt.ylabel("Log Loss")
                plt.grid(axis='y')
                plt.tight_layout()
                plt.savefig(f"{model_path}/lossplot/loss_{target}.png")
                plt.close()
                
        except Exception as e:
            print(f"ç„¡æ³•ç¹ªè£½æ��å¤±æ›²ç·š: {str(e)}")

        # å„²å­˜æ¨¡å�‹
        models[target] = model
        model.save(f"{model_path}/param/nn_{target}.h5")
        
        # è¨ˆç®—ä¸¦é¡¯ç¤ºé€²åº¦
        target_time = time.time() - target_start
        target_times.append(target_time)
        avg_time_per_target = np.mean(target_times)
        remaining_targets = total_targets - (i + 1)
        estimated_remaining_time = avg_time_per_target * remaining_targets
        
        # è½‰æ�›é �ä¼°å‰©é¤˜æ™‚é–“ç‚ºæ›´æ˜“è®€æ ¼å¼�
        remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
        completion_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)
        
        print(f"ç›®æ¨™ {i+1}/{total_targets} å·²å®Œæˆ�! ({target})")
        print(f"å¹³å�‡æ¯�å€‹ç›®æ¨™è¨“ç·´æ™‚é–“: {avg_time_per_target:.2f} ç§’")
        print(f"é �ä¼°å‰©é¤˜æ™‚é–“: {remaining_time_str}")
        print(f"é �ä¼°å®Œæˆ�æ™‚é–“: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")

# æœ€çµ‚å®Œæˆ�è¨Šæ�¯
print("\nâœ… æ‰€æœ‰æ¨¡å�‹è¨“ç·´å·²å®Œæˆ�!")
total_time = time.time() - start_time
print(f"ç¸½è€—æ™‚: {datetime.timedelta(seconds=int(total_time))}")

# 6. æ�¨è«–èˆ‡å»ºç«‹æ��äº¤æª”æ¡ˆ
print("ğŸ”® é–‹å§‹é€²è¡Œé �æ¸¬...")
predictions = []

# å°‡æ¸¬è©¦æ•¸æ“šè½‰æ�›ç‚ºnumpyæ•¸çµ„
X_test_array = X_test.values

# é¡¯ç¤ºé �æ¸¬é€²åº¦
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='é �æ¸¬é€²åº¦'):
    model = models[target]  # å„ªå…ˆä½¿ç”¨è¨˜æ†¶é«”ä¸­çš„æ¨¡å�‹é�¿å…�é‡�è¤‡è¼‰å…¥
    pred = model.predict(X_test_array, verbose=0).flatten()
    predictions.append(pred)

predictions = np.array(predictions).T
submission = sample_submission.copy()
submission.iloc[:, 1:] = predictions
submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
print("ğŸ�‰ å·²ç”¢å‡º submission_nn.csv å�¯ç›´æ�¥ä¸Šå‚³åˆ° Kaggle")

# æ·»åŠ é©—è­‰é›†çš„é �æ¸¬
print("ğŸ”� é–‹å§‹é€²è¡Œé©—è­‰é›†é �æ¸¬...")
predictions_V = []

# Show prediction progress
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='Validation prediction progress'):
    model = models[target]  # Use in-memory models to avoid reloading
    pred = model.predict(X_valid.values.astype(np.float32), verbose=0).flatten()  # å¼·åˆ¶è½‰å�‹ float32
    predictions_V.append(pred)

predictions_V = np.array(predictions_V).T

# Create validation submission with correct sig_ids
validation_submission = pd.DataFrame(columns=Y_valid.columns)
validation_submission['sig_id'] = df_valid['sig_id']
for col in target_columns:
    validation_submission[col] = predictions_V[:, list(target_columns).index(col)]

# Save validation predictions to a separate file
validation_submission.to_csv(VALIDATION_OUTPUT_PATH, index=False)
print(f"ğŸ�‰ NN é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆå·²å»ºç«‹: {VALIDATION_OUTPUT_PATH}")


import pandas as pd

# è®€å�–æ¯�å€‹ submission æª”æ¡ˆ
files = [
    # "/kaggle/working/submission_cat.csv",
    "/kaggle/working/submission_lgbm.csv",
    "/kaggle/working/submission_nn.csv",
    "/kaggle/working/submission_rf.csv",
    "/kaggle/working/submission_xgb.csv",
    "/kaggle/working/submission_svm.csv"
]

# å…ˆè®€ç¬¬ä¸€å€‹ç•¶ä½œ base
df_ensemble = pd.read_csv(files[0])
df_ensemble.iloc[:, 1:] = 0  # æŠŠé �æ¸¬æ¬„è¨­ç‚º 0ï¼Œç”¨ä¾†åŠ ç¸½ç”¨

# åŠ ç¸½æ‰€æœ‰ submission çš„é �æ¸¬æ¬„ï¼ˆä¸�å�« idï¼‰
for file in files:
    df = pd.read_csv(file)
    df_ensemble.iloc[:, 1:] += df.iloc[:, 1:]

# å�–å¹³å�‡
df_ensemble.iloc[:, 1:] /= len(files)

# å„²å­˜ç‚ºæ–°çš„ submission æª”æ¡ˆ
df_ensemble.to_csv("/kaggle/working/submission_kon.csv", index=False)

print("æˆ�åŠŸè��å�ˆæˆ� submission_kon.csv ï½� ğŸ�¥âœ¨")


# # 3 models weighted average
# sub_all = df_ensemble.copy()
# sub_all[target_cols] = sub_wct[target_cols] * 0.6 + df_ensemble[target_cols] * 0.2 + sub_jj[target_cols] * 0.2

# # final submission
# sub_all.to_csv('/kaggle/working/submission.csv', index=False)


sub_1[target_cols]


sub_2[target_cols]


sub_3[target_cols]


import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F


# # å­”ç¥¥æœ‰'s path
# self_Catboost_path = './å­”ç¥¥æœ‰/Self_CatBoost'
# self_LightGBM_path = './å­”ç¥¥æœ‰/Self_LightGBM'
# self_NN_path = './å­”ç¥¥æœ‰/Self_NN'
# self_Random_Forest_path = './å­”ç¥¥æœ‰/Self_Ranom_Forest'
# self_SVM_path = './å­”ç¥¥æœ‰/Self_SVM'
# self_XGboost_path = './å­”ç¥¥æœ‰/Self_XGboost'

# # self_Top_OIMMN_path = './å­”ç¥¥æœ‰/Top_Overfit_is_my_middle_name'

# # æ±Ÿæ‰¿ç€š's path
# top_HNing_path = './æ±Ÿæ‰¿ç€š/Top_HNing'

# # è”¡ç‘‹å®¸'s path
# Leader_CNN_path = './è”¡ç‘‹å®¸/LeaderBoard_1D_CNN'
# Leader_DNN_path = './è”¡ç‘‹å®¸/LeaderBoard_DNN'
# Leader_TabNet_path = './è”¡ç‘‹å®¸/LeaderBoard_TabNet'


# read test

CatBoost_submission = pd.read_csv(self_Catboost_path + '/submission.csv') #
LGBM_submission = pd.read_csv(self_LightGBM_path + '/submission_lgbm.csv')
NN_submission = pd.read_csv(self_NN_path + '/submission_nn.csv')
RF_submission = pd.read_csv(self_Random_Forest_path + '/submission_rf.csv')
SVM_submission = pd.read_csv(self_SVM_path + '/submission_svm.csv')
XG_submission = pd.read_csv(self_XGboost_path + '/submission_xgb.csv')

HNing_submission =  pd.read_csv(top_HNing_path + '/submission_jj.csv')

Leader_CNN_submission = pd.read_csv(Leader_CNN_path + '/submission.csv') #
Leader_DNN_submission = pd.read_csv(Leader_DNN_path + '/submission.csv') #
Leader_TabNet_submission = pd.read_csv(Leader_TabNet_path + '/submission.csv') #


features = CatBoost_submission.columns
features


# Checking testing set shape
print(CatBoost_submission.shape)
print(LGBM_submission.shape)
print(NN_submission.shape)
print(RF_submission.shape)
print(SVM_submission.shape)
print(XG_submission.shape)
print(HNing_submission.shape)
print(Leader_CNN_submission.shape)
print(Leader_DNN_submission.shape)
print(Leader_TabNet_submission.shape)


key_col = 'sig_id'
def rename_columns(df, model_name):
    return df.rename(columns={col: f"{model_name}_{col}" for col in df.columns if col != key_col})

CatBoost_submission = rename_columns(CatBoost_submission, 'CatBoost')
LGBM_submission = rename_columns(LGBM_submission, 'LGBM')
NN_submission = rename_columns(NN_submission, 'NN')
RF_submission = rename_columns(RF_submission, 'RF')
SVM_submission = rename_columns(SVM_submission, 'SVM')
XG_submission = rename_columns(XG_submission, 'XG')
HNing_submission = rename_columns(HNing_submission, 'HNing')
Leader_CNN_submission = rename_columns(Leader_CNN_submission, 'CNN')
Leader_DNN_submission = rename_columns(Leader_DNN_submission, 'DNN')
Leader_TabNet_submission = rename_columns(Leader_TabNet_submission, 'TBNET')

merged_submission = CatBoost_submission.merge(LGBM_submission, on=key_col) \
                                       .merge(NN_submission, on=key_col) \
                                       .merge(RF_submission, on=key_col) \
                                       .merge(SVM_submission, on=key_col) \
                                       .merge(XG_submission, on=key_col) \
                                       .merge(HNing_submission, on=key_col) \
                                       .merge(Leader_CNN_submission, on=key_col)\
                                       .merge(Leader_DNN_submission, on=key_col)\
                                       .merge(Leader_TabNet_submission, on=key_col)



class TabNet(nn.Module):
    def __init__(self, input_dim, output_dim=207, device='cpu'):
        super(TabNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1200)
        self.att1 = nn.MultiheadAttention(embed_dim=400, num_heads=5, dropout=0.0, 
                                          bias=True, add_bias_kv=True, kdim=400, vdim=400, 
                                          batch_first=True, device=device)
        self.norm1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(1600, 800)  # 1200 + 400 -> 1600
        self.att2 = nn.MultiheadAttention(embed_dim=400, num_heads=5, dropout=0.0, 
                                          bias=True, add_bias_kv=True, kdim=400, vdim=400, 
                                          batch_first=True, device=device)
        self.norm2 = nn.LayerNorm(400)
        self.att3 = nn.MultiheadAttention(embed_dim=400, num_heads=5, dropout=0.0, 
                                    bias=True, add_bias_kv=True, kdim=400, vdim=400, 
                                    batch_first=True, device=device)
        self.fc3 = nn.Linear(400, output_dim)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))  # (batch, 1200)
        # Reshape 1200 -> (batch, 3, 400)
        x1_seq = x1.view(x1.size(0), 3, 400)
        attn1_out, _ = self.att1(x1_seq, x1_seq, x1_seq)  # (batch, 3, 400)
        attn1_pooled = attn1_out.mean(dim=1)
        attn1_pooled = self.norm1(attn1_pooled)
        
        # Concatenate: (batch, 1200) + (batch, 400) â†’ (batch, 1600)
        x2 = F.relu(self.fc2(torch.cat([x1, attn1_pooled], dim=1)))  # (batch, 800)
        x2_seq = x2.view(x2.size(0), 2, 400)
        attn2_out, _ = self.att2(x2_seq, x2_seq, x2_seq)  # (batch, 2, 400)

        attn2_pooled = attn2_out.mean(dim=1)  # (batch, 400)
        attn2_pooled = self.norm2(attn2_pooled)

        out = torch.sigmoid(self.fc3(attn2_pooled))  # (batch, output_dim)
        return out



input_dim = (6 + 1 + 3) * 206
output_dim = 206

model = TabNet(input_dim=input_dim, output_dim=output_dim, device='cpu')
model.load_state_dict(torch.load('./best_model.pth'))


model.eval()
X_test = merged_submission.drop(columns=['sig_id']).values

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# dataset = TensorDataset(X_test_tensor, y_val_tensor)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

with torch.no_grad():
    outputs = model(X_test_tensor)

pred_np = outputs.cpu().numpy()

label_columns = features.drop('sig_id')

result_df = merged_submission[['sig_id']].copy()
result_df[label_columns] = pred_np


test_features = pd.read_csv('./test_features.csv')


mask = test_features['cp_type'] == 'ctl_vehicle'

cols_to_zero = result_df.columns.difference(['sig_id'])  # æ‰¾å‡ºé�� cp_type çš„æ¬„ä½�
result_df.loc[mask, cols_to_zero] = 0


result_df.to_csv('./submission.csv')

