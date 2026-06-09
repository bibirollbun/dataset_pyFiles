import sys
print(sys.version)


!python --version


# # # TabNet
# # !pip install tabnet
# !pip install --no-index --find-links=/kaggle/input/pytorchtabnet pytorch-tabnet

!pip install pytorch_tabnet


import sys
sys.path.append('../input/iterativestratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


### General ###
import os
import copy
import tqdm
import pickle
import random
import warnings
warnings.filterwarnings("ignore")
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

from pickle import load,dump

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


train_features_full = train_features
train_targets_scored_full = train_targets_scored
train_targets_nonscored_full = train_targets_nonscored


# train_features = train_features.sample(n=15000, random_state=42)
# train_targets_scored = train_targets_scored.sample(n=15000, random_state=42)
# train_targets_nonscored = train_targets_nonscored.sample(n=15000, random_state=42)
print(len(train_features), len(train_targets_scored), len(train_targets_nonscored))


# sampled_sig_ids = train_features['sig_id'].values

# new_test_features = train_features_full[~train_features_full['sig_id'].isin(sampled_sig_ids)].reset_index(drop=True)
# new_test_targets_scored = train_targets_scored_full[~train_targets_scored_full['sig_id'].isin(sampled_sig_ids)].reset_index(drop=True)
# new_test_targets_nonscored = train_targets_nonscored_full[~train_targets_nonscored_full['sig_id'].isin(sampled_sig_ids)].reset_index(drop=True)

# print(f"New test set size: {len(new_test_features)}")


test_features = pd.read_csv('../input/lish-moa/test_features.csv')
df = pd.read_csv('../input/lish-moa/sample_submission.csv')


train_features2=train_features.copy()
test_features2=test_features.copy()
# new_test_features2 = new_test_features.copy()


GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]


qt = QuantileTransformer(n_quantiles=100,random_state=42,output_distribution='normal')
train_features[GENES+CELLS] = qt.fit_transform(train_features[GENES+CELLS])
test_features[GENES+CELLS] = qt.transform(test_features[GENES+CELLS])
# new_test_features[GENES + CELLS] = qt.transform(new_test_features[GENES + CELLS])


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


n_comp = 600  #<--Update
pca_g = PCA(n_components=n_comp, random_state=42)
data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
# data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES]), pd.DataFrame(new_test_features[GENES])])
gpca= (pca_g.fit(data[GENES]))


train2= (gpca.transform(train_features[GENES]))
test2 = (gpca.transform(test_features[GENES]))
# new_test2 = (gpca.transform(new_test_features[GENES]))


train_gpca = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
test_gpca = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])
# new_test_gpca = pd.DataFrame(new_test2, columns=[f'pca_G-{i}' for i in range(n_comp)])


# drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
training_index = train_features.index
testing_index = test_features.index
# new_test_index = new_test_features.index

train_features = pd.concat((train_features.reset_index(drop=True), train_gpca), axis=1)
test_features = pd.concat((test_features.reset_index(drop=True), test_gpca), axis=1)
# new_test_features = pd.concat((new_test_features.reset_index(drop=True), new_test_gpca), axis=1)

train_features.index = training_index
test_features.index = testing_index
# new_test_features.index = new_test_index


dump(gpca, open('gpca.pkl', 'wb'))


#CELLS
n_comp = 50  #<--Update

pca_c = PCA(n_components=n_comp, random_state=42)
data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
cpca= (pca_c.fit(data[CELLS]))
train2= (cpca.transform(train_features[CELLS]))
test2 = (cpca.transform(test_features[CELLS]))
# new_test2 = (cpca.transform(new_test_features[CELLS]))

train_cpca = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
test_cpca = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])
# new_test_cpca = pd.DataFrame(new_test2, columns=[f'pca_C-{i}' for i in range(n_comp)])

# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
train_features = pd.concat((train_features.reset_index(drop=True), train_cpca.reset_index(drop=True)), axis=1)
test_features = pd.concat((test_features.reset_index(drop=True), test_cpca.reset_index(drop=True)), axis=1)
# new_test_features = pd.concat((new_test_features.reset_index(drop=True), new_test_cpca.reset_index(drop=True)), axis=1)

train_features.index = training_index
test_features.index = testing_index
# new_test_features.index = new_test_index

dump(cpca, open('cpca.pkl', 'wb'))


train_features


# new_test_features


from sklearn.feature_selection import VarianceThreshold

c_n = [f for f in list(train_features.columns) if f not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]
mask = (train_features[c_n].var() >= 0.85).values
tmp = train_features[c_n].loc[:, mask]
train_features = pd.concat([train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].reset_index(drop=True), tmp.reset_index(drop=True)], axis=1)
tmp = test_features[c_n].loc[:, mask]
test_features = pd.concat([test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].reset_index(drop=True), tmp.reset_index(drop=True)], axis=1)

# tmp = new_test_features[c_n].loc[:, mask]
# new_test_features = pd.concat([new_test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].reset_index(drop=True), tmp.reset_index(drop=True)], axis=1)

train_features.index = training_index
test_features.index = testing_index
# new_test_features.index = new_test_index


train_features2


# new_test_features2


from sklearn.cluster import KMeans
# def fe_cluster_genes(train, test, new_test, n_clusters_g = 22, SEED = 42):
def fe_cluster_genes(train, test, n_clusters_g = 22, SEED = 42):
    
    features_g = GENES
    #features_c = CELLS
    
    # def create_cluster(train, test, new_test, features, kind = 'g', n_clusters = n_clusters_g):
    def create_cluster(train, test, features, kind = 'g', n_clusters = n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        # new_test_ = new_test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        
        kmeans_genes = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        dump(kmeans_genes, open('kmeans_genes.pkl', 'wb'))
        train[f'clusters_{kind}'] = kmeans_genes.predict(train_.values)
        test[f'clusters_{kind}'] = kmeans_genes.predict(test_.values)
        # new_test[f'clusters_{kind}'] = kmeans_genes.predict(new_test_.values)
        
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        # new_test = pd.get_dummies(new_test, columns = [f'clusters_{kind}']) # 這裡出bug
        # return train, test, new_test
        return train, test

    # train, test, new_test = create_cluster(train, test, new_test, features_g, kind = 'g', n_clusters = n_clusters_g)
    train, test = create_cluster(train, test, features_g, kind = 'g', n_clusters = n_clusters_g)
   # train, test = create_cluster(train, test, features_c, kind = 'c', n_clusters = n_clusters_c)
    return train, test

# train_features2 ,test_features2, new_test_features2 =fe_cluster_genes(train_features2,test_features2, new_test_features2)
train_features2 ,test_features2 =fe_cluster_genes(train_features2,test_features2)


train_features2


test_features2


# new_test_features2


# def fe_cluster_cells(train, test, new_test, n_clusters_c = 4, SEED = 42):
def fe_cluster_cells(train, test, n_clusters_c = 4, SEED = 42):
    
    #features_g = GENES
    features_c = CELLS
    
    # def create_cluster(train, test, new_test, features, kind = 'c', n_clusters = n_clusters_c):
    def create_cluster(train, test, features, kind = 'c', n_clusters = n_clusters_c):
        train_ = train[features].copy()
        test_ = test[features].copy()
        # new_test_ = new_test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        kmeans_cells = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        dump(kmeans_cells, open('kmeans_cells.pkl', 'wb'))
        train[f'clusters_{kind}'] = kmeans_cells.predict(train_.values)
        test[f'clusters_{kind}'] = kmeans_cells.predict(test_.values)
        # new_test[f'clusters_{kind}'] = kmeans_cells.predict(new_test_.values)
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        # new_test = pd.get_dummies(new_test, columns = [f'clusters_{kind}'])
        return train, test
    
   # train, test = create_cluster(train, test, features_g, kind = 'g', n_clusters = n_clusters_g)
    train, test = create_cluster(train, test, features_c, kind = 'c', n_clusters = n_clusters_c)
    return train, test

train_features2 ,test_features2=fe_cluster_cells(train_features2,test_features2)


train_features2


# new_test_features2


train_pca=pd.concat((train_gpca.reset_index(drop=True),train_cpca.reset_index(drop=True)),axis=1)
test_pca=pd.concat((test_gpca.reset_index(drop=True),test_cpca.reset_index(drop=True)),axis=1)
# new_test_pca=pd.concat((new_test_gpca.reset_index(drop=True),new_test_cpca.reset_index(drop=True)),axis=1)

train_features.index = training_index
test_features.index = testing_index
# new_test_features.index = new_test_index


test_gpca['pca_G-143']


def fe_cluster_pca(train, test, n_clusters=5,SEED = 42):
        data=pd.concat([train,test],axis=0)
        kmeans_pca = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        dump(kmeans_pca, open('kmeans_pca.pkl', 'wb'))
        train[f'clusters_pca'] = kmeans_pca.predict(train.values)
        test[f'clusters_pca'] = kmeans_pca.predict(test.values)
        # new_test[f'clusters_pca'] = kmeans_pca.predict(new_test.values)
        train = pd.get_dummies(train, columns = [f'clusters_pca'])
        test = pd.get_dummies(test, columns = [f'clusters_pca'])
        # new_test = pd.get_dummies(new_test, columns = [f'clusters_pca'])
        return train, test
train_cluster_pca ,test_cluster_pca = fe_cluster_pca(train_pca,test_pca)


train_cluster_pca = train_cluster_pca.iloc[:,650:]
test_cluster_pca = test_cluster_pca.iloc[:,650:]
# new_test_cluster_pca = new_test_cluster_pca.iloc[:,650:]


train_features_cluster=train_features2.iloc[:,876:]
test_features_cluster=test_features2.iloc[:,876:]
# new_test_features_cluster=new_test_features2.iloc[:,876:]



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
        df['c26_c13'] = df['c-26'] * df['c-13']
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
# new_test_features_stats=new_test_features2.iloc[:,902:]


train_features = pd.concat((train_features.reset_index(drop=True), train_features_cluster.reset_index(drop=True),train_cluster_pca.reset_index(drop=True),train_features_stats.reset_index(drop=True)), axis=1)
test_features = pd.concat((test_features.reset_index(drop=True), test_features_cluster.reset_index(drop=True),test_cluster_pca.reset_index(drop=True),test_features_stats.reset_index(drop=True)), axis=1)
# new_test_features = pd.concat((new_test_features.reset_index(drop=True), new_test_features_cluster.reset_index(drop=True),new_test_cluster_pca.reset_index(drop=True),new_test_features_stats.reset_index(drop=True)), axis=1)

train_features.index = training_index
test_features.index = testing_index
# new_test_features.index = new_test_index


train = train_features.merge(train_targets_scored, on='sig_id')
train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)
# new_test = new_test_features[new_test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

target = train[train_targets_scored.columns]


train = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)
# new_test = new_test.drop('cp_type', axis=1)


target_cols = target.drop('sig_id', axis=1).columns.values.tolist()


train = pd.get_dummies(train, columns=['cp_time','cp_dose'])
test_ = pd.get_dummies(test, columns=['cp_time','cp_dose'])
# new_test_ = pd.get_dummies(new_test, columns=['cp_time','cp_dose'])


feature_cols = [c for c in train.columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['sig_id']]


len(feature_cols)


from torch.nn.modules.loss import _WeightedLoss
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
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


MAX_EPOCH = 200

tabnet_params = dict(
    n_d = 16,
    n_a = 96,
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


class LogitsLogLoss(Metric):

    def __init__(self):
        self._name = "logits_ll"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        logits = 1 / (1 + np.exp(-y_pred))
        aux = (1 - y_true) * np.log(1 - logits + 1e-15) + y_true * np.log(logits + 1e-15)
        return np.mean(-aux)


feature_cols


import pickle
with open("/kaggle/working/op", "wb") as fp:   #Pickling
    pickle.dump(feature_cols, fp)


len(feature_cols)


# feature_cols
print('g-0' in feature_cols)
print('pca_G-142' in feature_cols)
print('pca_G-143' in feature_cols)
print('pca_G-144' in feature_cols)

print('pca_G-143' in train.columns)


type(test_[feature_cols].values)


scores_auc_all = []
test_cv_preds = []
# new_test_cv_preds = []

NB_SPLITS = 7
oof_preds = []
oof_targets = []
scores = []
scores_auc = []
# SEED = [0]
SEED = [0,1,2,3,4,5,6]
for s in SEED:
    tabnet_params['seed'] = s
    for fold_nb, (train_idx, val_idx) in enumerate(MultilabelStratifiedKFold(n_splits=NB_SPLITS, shuffle=True, random_state=s)
.split(train, target)):
        print(b_,"FOLDS: ", r_, fold_nb + 1, y_, 'seed:', tabnet_params['seed'])
        print(g_, '*' * 60, c_)
    
        # X_train, y_train = train[feature_cols].values[train_idx, :], target[target_cols].values[train_idx, :]
        # X_val, y_val = train[feature_cols].values[val_idx, :], target[target_cols].values[val_idx, :]
        X_train = train[feature_cols].astype(np.float32).values[train_idx, :]
        X_val = train[feature_cols].astype(np.float32).values[val_idx, :]
        y_train = target[target_cols].astype(np.float32).values[train_idx, :]
        y_val = target[target_cols].astype(np.float32).values[val_idx, :]
        test_features = test_[feature_cols].astype(np.float32).values
        # new_test_features = new_test_[feature_cols].astype(np.float32).values
        ### Model ###
        model = TabNetRegressor(**tabnet_params)
        
        ### Fit ###
        model.fit(
            X_train = X_train,
            y_train = y_train,
            eval_set = [(X_val, y_val)],
            eval_name = ["val"],
            eval_metric = ["logits_ll"],
            max_epochs = MAX_EPOCH,
            patience = 40,
            batch_size = 1024, 
            virtual_batch_size = 32,
            num_workers = 1,
            drop_last = False,
            loss_fn = SmoothBCEwLogits(smoothing=5e-5))
        print(y_, '-' * 60)
            
        ### Predict on validation ###
        preds_val = model.predict(X_val)
        # Apply sigmoid to the predictions
        preds = 1 / (1 + np.exp(-preds_val))
        score = np.min(model.history["val_logits_ll"])
        saving_path_name = 'TabNet_seed_'+str(tabnet_params['seed'])+'_fold_'+str(fold_nb+1)
        saved_filepath = model.save_model(saving_path_name)
        
        loaded_model =  TabNetRegressor()
        loaded_model.load_model(saved_filepath)
        
        loaded_model =  TabNetRegressor()
        loaded_model.load_model(saved_filepath)
    
        ### Save OOF for CV ###
        oof_preds.append(preds_val)
        oof_targets.append(y_val)
        scores.append(score)
    
        ### Predict on test ###
        model.load_model(saved_filepath)
        preds_test = model.predict(test_features)
        test_cv_preds.append(1 / (1 + np.exp(-preds_test)))

        # preds_new_test = model.predict(new_test_features)
        # new_test_cv_preds.append(1 / (1 + np.exp(-preds_new_test)))

oof_preds_all = np.concatenate(oof_preds)
oof_targets_all = np.concatenate(oof_targets)
test_preds_all = np.stack(test_cv_preds)
# new_test_preds_all = np.stack(new_test_cv_preds)


# test_[feature_cols].values.shape


# test_features = test_[feature_cols].astype(np.float32).values
# preds_test = model.predict(test_features)
# print('success')


aucs = []
for task_id in range(oof_preds_all.shape[1]):
    aucs.append(roc_auc_score(y_true = oof_targets_all[:, task_id],
                              y_score = oof_preds_all[:, task_id]
                             ))
print(f"{b_}Overall AUC: {r_}{np.mean(aucs)}")
print(f"{b_}Average CV: {r_}{np.mean(scores)}")


print(oof_preds_all.shape)
print(oof_targets_all.shape)
print(oof_preds_all.shape)
print(tabnet_params['seed'])


all_feat = [col for col in df.columns if col not in ["sig_id"]]
# To obtain the same lenght of test_preds_all and submission
test = pd.read_csv("../input/lish-moa/test_features.csv")
sig_id = test[test["cp_type"] != "ctl_vehicle"].sig_id.reset_index(drop = True)
tmp = pd.DataFrame(test_preds_all.mean(axis = 0), columns = all_feat)
tmp["sig_id"] = sig_id

submission = pd.merge(test[["sig_id"]], tmp, on = "sig_id", how = "left")
submission.fillna(0, inplace = True)
submission.to_csv("/kaggle/working/submission.csv", index = None)
submission.head()


print(f"{b_}submission.shape: {r_}{submission.shape}")


# temppppp = pd.read_csv('../input/lish-moa/train_features.csv')

# new_test = train_features_full[~temppppp['sig_id'].isin(sampled_sig_ids)]
# sig_id = new_test[new_test["cp_type"] != "ctl_vehicle"].sig_id.reset_index(drop = True)

# tmp = pd.DataFrame(new_test_preds_all.mean(axis = 0), columns = all_feat)
# tmp["sig_id"] = sig_id

# # pd_new_test_preds_all = pd.DataFrame(new_test_preds_all)
# submission = pd.merge(new_test[["sig_id"]], tmp, on = "sig_id", how = "left")
# submission.fillna(0, inplace = True)
# # submission.index = new_test_index
# submission.to_csv("/kaggle/working/ztabnet_HNing.csv", index = None)
# submission


# tmp


# new_test_preds_all.shape


import gc
gc.collect()




