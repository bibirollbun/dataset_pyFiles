!pip -q install ../input/pytorchtabnet/pytorch_tabnet-3.1.1-py3-none-any.whl


import os
import random
import time
import pickle
import psutil
import gc


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold

import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric

# setting some globl config

plt.style.use('ggplot')
orange_black = [
    '#fdc029', '#df861d', '#FF6347', '#aa3d01', '#a30e15', '#800000', '#171820'
]
plt.rcParams['figure.figsize'] = (16,9)
plt.rcParams["figure.facecolor"] = '#FFFACD'
plt.rcParams["axes.facecolor"] = '#FFFFE0'
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = orange_black[3]
plt.rcParams["grid.alpha"] = 0.5
plt.rcParams["grid.linestyle"] = '--'


import warnings
warnings.filterwarnings("ignore")

INFERENCE = True


%%time
features_avg = ['B_11', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_2', 
                'B_20', 'B_28', 'B_29', 'B_3', 'B_33', 'B_36', 'B_37', 'B_4', 'B_42', 
                'B_5', 'B_8', 'B_9', 'D_102', 'D_103', 'D_105', 'D_111', 'D_112', 'D_113', 
                'D_115', 'D_118', 'D_119', 'D_121', 'D_124', 'D_128', 'D_129', 'D_131', 
                'D_132', 'D_133', 'D_139', 'D_140', 'D_141', 'D_143', 'D_144', 'D_145', 
                'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 
                'D_49', 'D_50', 'D_51', 'D_52', 'D_56', 'D_58', 'D_62', 'D_70', 'D_71', 
                'D_72', 'D_74', 'D_75', 'D_79', 'D_81', 'D_83', 'D_84', 'D_88', 'D_91', 
                'P_2', 'P_3', 'R_1', 'R_10', 'R_11', 'R_13', 'R_18', 'R_19', 'R_2', 'R_26', 
                'R_27', 'R_28', 'R_3', 'S_11', 'S_12', 'S_22', 'S_23', 'S_24', 'S_26', 
                'S_27', 'S_5', 'S_7', 'S_8', ]
features_min = ['B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_19', 'B_2', 'B_20', 'B_22', 
                'B_24', 'B_27', 'B_28', 'B_29', 'B_3', 'B_33', 'B_36', 'B_4', 'B_42', 
                'B_5', 'B_9', 'D_102', 'D_103', 'D_107', 'D_109', 'D_110', 'D_111', 
                'D_112', 'D_113', 'D_115', 'D_118', 'D_119', 'D_121', 'D_122', 'D_128', 
                'D_129', 'D_132', 'D_133', 'D_139', 'D_140', 'D_141', 'D_143', 'D_144', 
                'D_145', 'D_39', 'D_41', 'D_42', 'D_45', 'D_46', 'D_48', 'D_50', 'D_51', 
                'D_53', 'D_54', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_62', 'D_70', 
                'D_71', 'D_74', 'D_75', 'D_78', 'D_79', 'D_81', 'D_83', 'D_84', 'D_86', 
                'D_88', 'D_96', 'P_2', 'P_3', 'P_4', 'R_1', 'R_11', 'R_13', 'R_17', 'R_19', 
                'R_2', 'R_27', 'R_28', 'R_4', 'R_5', 'R_8', 'S_11', 'S_12', 'S_23', 'S_25', 
                'S_3', 'S_5', 'S_7', 'S_9', ]
features_max = ['B_1', 'B_11', 'B_13', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_2', 
                'B_22', 'B_24', 'B_27', 'B_28', 'B_29', 'B_3', 'B_31', 'B_33', 'B_36', 
                'B_4', 'B_42', 'B_5', 'B_7', 'B_9', 'D_102', 'D_103', 'D_105', 'D_109', 
                'D_110', 'D_112', 'D_113', 'D_115', 'D_121', 'D_124', 'D_128', 'D_129', 
                'D_131', 'D_139', 'D_141', 'D_144', 'D_145', 'D_39', 'D_41', 'D_42', 
                'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_50', 'D_51', 'D_52', 
                'D_53', 'D_56', 'D_58', 'D_59', 'D_60', 'D_62', 'D_70', 'D_72', 'D_74', 
                'D_75', 'D_79', 'D_81', 'D_83', 'D_84', 'D_88', 'D_89', 'P_2', 'P_3', 
                'R_1', 'R_10', 'R_11', 'R_26', 'R_28', 'R_3', 'R_4', 'R_5', 'R_7', 'R_8', 
                'S_11', 'S_12', 'S_23', 'S_25', 'S_26', 'S_27', 'S_3', 'S_5', 'S_7', 'S_8', ]
features_last = ['B_1', 'B_11', 'B_12', 'B_13', 'B_14', 'B_16', 'B_18', 'B_19', 'B_2', 
                 'B_20', 'B_21', 'B_24', 'B_27', 'B_28', 'B_29', 'B_3', 'B_30', 'B_31', 
                 'B_33', 'B_36', 'B_37', 'B_38', 'B_39', 'B_4', 'B_40', 'B_42', 'B_5', 
                 'B_8', 'B_9', 'D_102', 'D_105', 'D_106', 'D_107', 'D_108', 'D_110', 
                 'D_111', 'D_112', 'D_113', 'D_114', 'D_115', 'D_116', 'D_117', 'D_118', 
                 'D_119', 'D_120', 'D_121', 'D_124', 'D_126', 'D_128', 'D_129', 'D_131', 
                 'D_132', 'D_133', 'D_137', 'D_138', 'D_139', 'D_140', 'D_141', 'D_142', 
                 'D_143', 'D_144', 'D_145', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 
                 'D_46', 'D_47', 'D_48', 'D_49', 'D_50', 'D_51', 'D_52', 'D_53', 'D_55', 
                 'D_56', 'D_59', 'D_60', 'D_62', 'D_63', 'D_64', 'D_66', 'D_68', 'D_70', 
                 'D_71', 'D_72', 'D_73', 'D_74', 'D_75', 'D_77', 'D_78', 'D_81', 'D_82', 
                 'D_83', 'D_84', 'D_88', 'D_89', 'D_91', 'D_94', 'D_96', 'P_2', 'P_3', 
                 'P_4', 'R_1', 'R_10', 'R_11', 'R_12', 'R_13', 'R_16', 'R_17', 'R_18', 
                 'R_19', 'R_25', 'R_28', 'R_3', 'R_4', 'R_5', 'R_8', 'S_11', 'S_12', 
                 'S_23', 'S_25', 'S_26', 'S_27', 'S_3', 'S_5', 'S_7', 'S_8', 'S_9', ]
features_categorical = ['B_30_last', 'B_38_last', 'D_114_last', 'D_116_last',
                        'D_117_last', 'D_120_last', 'D_126_last',
                        'D_63_last', 'D_64_last', 'D_66_last', 'D_68_last']

for i in [0, 1] if INFERENCE else [0]:
    # i == 0 -> process the train data
    # i == 1 -> process the test data
    df = pd.read_feather(['../input/amexfeather/train_data.ftr',
                          '../input/amexfeather/test_data.ftr'][i])
    cid = pd.Categorical(df.pop('customer_ID'), ordered=True)
    last = (cid != np.roll(cid, -1)) # mask for last statement of every customer
    if i == 0: # train
        target = df.loc[last, 'target']
    print('Read', i)
    gc.collect()
    df_avg = (df
              .groupby(cid)
              .mean()[features_avg]
              .rename(columns={f: f"{f}_avg" for f in features_avg})
             )
    print('Computed avg', i)
    gc.collect()
    df_max = (df
              .groupby(cid)
              .max()[features_max]
              .rename(columns={f: f"{f}_max" for f in features_max})
             )
    print('Computed max', i)
    gc.collect()
    df_min = (df
              .groupby(cid)
              .min()[features_min]
              .rename(columns={f: f"{f}_min" for f in features_min})
             )
    print('Computed min', i)
    gc.collect()
    df_last = (df.loc[last, features_last]
               .rename(columns={f: f"{f}_last" for f in features_last})
               .set_index(np.asarray(cid[last]))
              )
    df = None # we no longer need the original data
    print('Computed last', i)
    
    df_categorical = df_last[features_categorical].astype(object)
    features_not_cat = [f for f in df_last.columns if f not in features_categorical]
    if i == 0: # train
        ohe = OneHotEncoder(drop='first', sparse=False, dtype=np.float32, handle_unknown='ignore')
        ohe.fit(df_categorical)
        with open("ohe.pickle", 'wb') as f: pickle.dump(ohe, f)
    df_categorical = pd.DataFrame(ohe.transform(df_categorical).astype(np.float16),
                                  index=df_categorical.index).rename(columns=str)
    print('Computed categorical', i)
    
    df = pd.concat([df_last[features_not_cat], df_categorical, df_avg, df_min, df_max], axis=1)
    
    # Impute missing values
    df.fillna(value=0, inplace=True)
    
    del df_avg, df_max, df_min, df_last, df_categorical, cid, last, features_not_cat
    
    if i == 0: # train
        # Free the memory
        df.reset_index(drop=True, inplace=True) # frees 0.2 GByte
        df.to_feather('train_processed.ftr')
        df = None
        gc.collect()

train = pd.read_feather('train_processed.ftr')
test = df
target = target.reset_index(drop=True)
del df, ohe

print('Shapes:', train.shape, target.shape)


class CFG:
    DEBUG = False
    model = 'tabnet'
    N_folds = 5
    seed = 42
    batch_size = 512
    max_epochs = 60


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(seed = CFG.seed)


psutil.virtual_memory().percent


def amex_metric_numpy(y_true: np.array, y_pred: np.array) -> float:

    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos

    # sorting by descring prediction values
    indices = np.argsort(y_pred)[::-1]
    preds, target = y_pred[indices], y_true[indices]

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = target[four_pct_filter].sum() / n_pos

    # weighted gini coefficient
    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)


# using amex metric to evaluate tabnet
class Amex_tabnet(Metric):
    
  def __init__(self):
    self._name = 'amex_tabnet'
    self._maximize = True

  def __call__(self, y_true, y_pred):
    amex = amex_metric_numpy(y_true, y_pred[:, 1])
    return max(amex, 0.)


print('\n ', '-'*50)
print('\nTraining: ', CFG.model)
print('\n ', '-'*50)

print('\nSeed: ', CFG.seed)
print('N folds: ', CFG.N_folds)
print('train shape: ', train.shape)
print('targets shape: ', target.shape)


print('\nN features: ', len(train.columns.values.tolist()))
print('\n')

# Create out of folds array
oof_predictions = np.zeros((train.shape[0]))
test_predictions = np.zeros(test.shape[0])
feature_importances = pd.DataFrame()
feature_importances["feature"] = train.columns.tolist()
stats = pd.DataFrame()
explain_matrices = []
masks_ =[]


    
kfold = StratifiedKFold(n_splits = CFG.N_folds, shuffle=True, random_state = CFG.seed)

for fold, (train_idx, valid_idx) in enumerate(kfold.split(train, target)):

    ## DEBUG MODE
    if CFG.DEBUG == True:
        if fold > 0:
            print('\nDEBUG mode activated: Will train only one fold...\n')
            break      

    start = time.time()

    X_train, y_train = train.loc[train_idx], target.loc[train_idx]
    X_valid, y_valid = train.loc[valid_idx], target.loc[valid_idx]        
        
    model = TabNetClassifier(n_d = 32,
                             n_a = 32,
                             n_steps = 3,
                             gamma = 1.3,
                             n_independent = 2,
                             n_shared = 2,
                             momentum = 0.02,
                             clip_value = None,
                             lambda_sparse = 1e-3,
                             optimizer_fn = torch.optim.Adam,
                             optimizer_params = dict(lr = 1e-3, weight_decay=1e-3),
                             scheduler_fn = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                             scheduler_params = {'T_0':5,
                                                 'eta_min':1e-4,
                                                 'T_mult':1,
                                                 'last_epoch':-1},
                             mask_type = 'entmax',
                             seed = CFG.seed)
    
    

    ## train
    model.fit(np.array(X_train),
              np.array(y_train.values.ravel()),
              eval_set = [(np.array(X_valid), np.array(y_valid.values.ravel()))],
              max_epochs = CFG.max_epochs,
              patience = 50,
              batch_size = CFG.batch_size,
              eval_metric = ['auc', 'accuracy', Amex_tabnet]) # Last metric is used for early stopping
    
    # Saving best model
    saving_path_name = f"./fold{fold}"
    saved_filepath = model.save_model(saving_path_name)
    
    # model explanability
    explain_matrix, masks = model.explain(X_valid.values)
    explain_matrices.append(explain_matrix)
    masks_.append(masks[0])
    masks_.append(masks[1])
    
    # Inference
    oof_predictions[valid_idx] = model.predict_proba(X_valid.values)[:, 1]
    
    #if CFG
    # logodds function
    
    test_predictions += model.predict_proba(test.values)[:, 1]/5
    feature_importances[f"importance_fold{fold}+1"] = model.feature_importances_
    
    # Loss , metric tracking
    stats[f'fold{fold+1}_train_loss'] = model.history['loss']
    stats[f'fold{fold+1}_val_metric'] = model.history['val_0_amex_tabnet']


    end = time.time()
    time_delta = np.round((end - start)/60, 2)
     
    print(f'\nFold {fold+1}/{CFG.N_folds} | {time_delta:.2f} min')

    ### free memory
    del X_train, y_train
    del X_valid, y_valid
    gc.collect()

print(f'OOF score across folds: {amex_metric_numpy(target, oof_predictions.flatten())}')


for i in stats.filter(like='train', axis=1).columns.tolist():
    plt.plot(stats[i], label=str(i))
plt.title('Train loss')
plt.legend()  


for i in stats.filter(like='val', axis=1).columns.tolist():
    plt.plot(stats[i], label=str(i))
plt.title('Train RMSPE')
plt.legend() 


feature_importances['mean_importance']=feature_importances[['importance_fold0+1','importance_fold1+1']].mean(axis=1)
feature_importances.sort_values(by='mean_importance', ascending=False, inplace=True)
sns.barplot(y=feature_importances['feature'][:50],x=feature_importances['mean_importance'][:50], palette='inferno')
plt.title('Mean Feature Importance by Folds')
plt.show()


fig, axs = plt.subplots(5, 2, figsize=(16,16))
axs = axs.flatten()

k=-1    
for i, (mask, j) in enumerate(zip(masks_, axs)):
    sns.heatmap(mask[:150], ax=j)
    if i%2 == 0:
        k+=1
    j.set_title((f"Fold{k} Mask for First 150 Instances"))
plt.tight_layout()


fig, axs = plt.subplots(len(explain_matrices), 1, figsize=(20,8))
for i,matrix in enumerate(explain_matrices):
    axs[i].set_title(f'Fold{i} Explain Matrix for First 150 Instances')
    sns.heatmap(matrix[:150], ax=axs[i])
plt.tight_layout() 


sub = pd.DataFrame({'customer_ID': test.index,
                    'prediction': test_predictions})
sub.to_csv('submission_tabnet.csv', index=False)


sub

