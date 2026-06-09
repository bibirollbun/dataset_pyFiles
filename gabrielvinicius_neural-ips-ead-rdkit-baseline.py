!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl
!pip install /kaggle/input/umap-learn/umap_learn-0.5.7-py3-none-any.whl


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

import umap

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit import RDLogger
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
import joblib

from tqdm.notebook import tqdm

import warnings
# Suppress RuntimeWarnings from pandas formatting
warnings.filterwarnings("ignore", category=RuntimeWarning)


class CFG:

    #Extra Data Flag
    use_extra_data = False

    smiles_fe = 'AUTOCORR2D' #AUTO, AUTOCORR2D

    #Reducer
    reduce_fings = False
    n_components = 30
    min_dist=0.1
    
    test_size = 0.2
    seed = 42
    model_loss = 'rmse'

    # lgb_params = {
    #                 'params':{ 
    #                     'objective': 'regression',
    #                     'metric': model_loss,
    #                     'boosting_type': 'gbdt',
    #                     'learning_rate': 0.05,
    #                     'num_leaves': 127,           
    #                     'learning_rate': 0.07,       
    #                     'feature_fraction': 0.8,     
    #                     'bagging_fraction': 0.9,     
    #                     'bagging_freq': 1,           # Bag every iteration
    #                     'lambda_l1': 0.1,            # L1 regularization
    #                     'lambda_l2': 0.1,            # L2 regularization
    #                     'min_data_in_leaf': 1,
    #                     'verbosity': -1,
    #                     'early_stopping_rounds':50,
    #                     'seed': seed
    #                 },
    #                 'num_boost_round': 2000,
    #     }

    #Baseline

    lgb_params = {
                'params':{ 
                    'objective': 'regression',
                    'metric': model_loss,
                    'boosting_type': 'gbdt',
                    'learning_rate': 0.05,
                    'verbosity': -1,
                    'early_stopping_rounds':50,
                    'seed': seed
                },
                'num_boost_round': 200,
    }

    pred_strat = 'ensamble'#ensamble, desc, fings
    
    


train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
print(train_df.shape)
train_df.head()


train_df.info()


#https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion/585178
if CFG.use_extra_data:
    extra_tg_df = pd.read_csv('/kaggle/input/smiles-tg/Tg_SMILES_class_pid_polyinfo_median.csv')
    print(extra_tg_df.shape)
    display(extra_tg_df.head(3))
    
    extra_tc_df = pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv')
    print(extra_tc_df.shape)
    display(extra_tc_df.head(3))


if CFG.use_extra_data:
    # Prepare extra_tg_df
    extra_tg_clean = extra_tg_df[['SMILES', 'PID', 'Tg']].rename(columns={'PID': 'id'})
    extra_tg_clean[['FFV', 'Tc', 'Density', 'Rg']] = float('nan')
    
    # Prepare extra_tc_df  
    extra_tc_clean = extra_tc_df[['SMILES', 'TC_mean']].rename(columns={'TC_mean': 'Tc'})
    extra_tc_clean['id'] = range(len(train_df), len(train_df) + len(extra_tc_df))
    extra_tc_clean[['Tg', 'FFV', 'Density', 'Rg']] = float('nan')
    
    # Reorder columns to match train_df
    # extra_tg_clean = extra_tg_clean[['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]
    extra_tc_clean = extra_tc_clean[['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]
    
    # Combine all datasets into train_df
    train_df = pd.concat([train_df, 
                          #extra_tg_clean, 
                          extra_tc_clean
                         ], ignore_index=True)
    
    print(train_df.count())
    train_df.head()


train_df.head()


targets = ['FFV','Tc','Rg','Density','Tg']


null_percenteage = (train_df.isnull().mean()*100).round(2)
null_percenteage = null_percenteage[null_percenteage > 0].sort_values()
fig,ax = plt.subplots(figsize = (7,5))
null_percenteage.plot(kind = 'bar', rot = 0, ax=ax)

for i, (col, value) in enumerate(zip(null_percenteage.index, null_percenteage.values)):
    label_text = f"{value}%"
    ax.text(i, value + 1, label_text, 
             ha='center', va='bottom', 
             fontsize=8, weight = 600 ,rotation=0,
             )

ax.set_ylim(0,100)
    
ax.set_title('Percentage of Nulls')
plt.tight_layout()
plt.show()


fig,ax = plt.subplots(figsize = (10,10))

train_df[targets].hist(ax=ax)

plt.title('jdfnb')
plt.tight_layout()
plt.show()



null_percenteage = (train_df.isnull().mean()*100).round(2)
null_percenteage = null_percenteage[null_percenteage > 0].sort_values()
fig,ax = plt.subplots(figsize = (7,5))
null_percenteage.plot(kind = 'bar', rot = 0, ax=ax)

for i, (col, value) in enumerate(zip(null_percenteage.index, null_percenteage.values)):
    label_text = f"{value}%"
    ax.text(i, value + 1, label_text, 
             ha='center', va='bottom', 
             fontsize=8, weight = 600 ,rotation=0,
             )

ax.set_ylim(0,100)
    
ax.set_title('Percentage of Nulls')
plt.tight_layout()
plt.show()


def show_implicit_h(smiles):
    m = Chem.MolFromSmiles(smiles)
    for atom in m.GetAtoms():
        atom.SetProp('atomLabel', str(atom.GetIdx()))
    m = Chem.AddHs(m)
    return Draw.MolToImage(m, size=(300, 300))


smiles_sample = train_df['SMILES'].sample(12).reset_index(drop = True)
smile_struct_imgs = []
for smile in smiles_sample:
    smile_struct_imgs.append(show_implicit_h(smile))
    print(smile)
    print()


row, col = 0,0
fig, ax = plt.subplots(4,3, figsize = (15,15))
for i, img in enumerate(smile_struct_imgs):

    ax_id = ax[row,col]
    
    ax_id.imshow(img)
    ax_id.axis('off')
    
    row = row+ 1 if (i+1)%3 == 0 else row
    
    

    col+=1
    col = 0 if (i+1)%3 == 0  else col

    


def get_molecular_descriptors(max_autocorr=10):
    """Get molecular descriptors - either hardcoded list or auto-discovered"""

    descriptor_list_all = []
    test_mol = Chem.MolFromSmiles('CCO')

    # Collect all valid descriptors first
    for name in dir(Descriptors):
        if not name.startswith('_'):
            try:
                func = getattr(Descriptors, name)
                if callable(func):
                    result = func(test_mol)
                    if isinstance(result, (int, float)) and not np.isnan(result):
                        descriptor_list_all.append((name, func))
            except:
                pass

    print(f"ğŸ”� Total discovered descriptors before filtering: {len(descriptor_list_all)}")

    # Sort AUTOCORR2D descriptors by their numeric suffix
    autocorr_descriptors = [
        (name, func)
        for name, func in descriptor_list_all
        if name.startswith('AUTOCORR2D_')
    ]
    autocorr_descriptors.sort(key=lambda x: int(x[0].split('_')[-1]))

    # Select only the lowest-numbered ones
    limited_autocorr = autocorr_descriptors[:max_autocorr]

    # Include all other descriptors
    other_descriptors = [
        (name, func)
        for name, func in descriptor_list_all
        if not name.startswith('AUTOCORR2D_')
    ]

    # Final descriptor list
    descriptor_list = limited_autocorr + other_descriptors

    print(f"âœ… Auto-discovered {len(descriptor_list)} descriptors (limited to {max_autocorr} AUTOCORR2D):")
    names = [name for name, _ in descriptor_list]
    print("  " + ", ".join(names))

    feature_names = [name for name, _ in descriptor_list]
    return descriptor_list


MOLECULAR_DESCRIPTORS = get_molecular_descriptors(max_autocorr=10)



def process_single_smiles(smiles):
    mol_features = []
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan] * len(MOLECULAR_DESCRIPTORS)
        for name, func in MOLECULAR_DESCRIPTORS:
            try:
                value = func(mol)
                if np.isinf(value) or abs(value) > 1e10:
                    value = np.nan
                mol_features.append(value)
            except:
                mol_features.append(np.nan)
    except:
        return [np.nan] * len(MOLECULAR_DESCRIPTORS)

    return mol_features

def smiles_to_features(smiles_list, clean_descriptors=False, n_jobs=None):
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 2)

    print(f"Processing {len(smiles_list)} SMILES with {n_jobs} processes...", flush=True)

    with Pool(n_jobs) as pool:
        features = list(tqdm(pool.imap(process_single_smiles, smiles_list), total=len(smiles_list)))

    features = np.array(features, dtype=float)
    print(" âœ…", flush=True)

    molecular_descriptors = MOLECULAR_DESCRIPTORS

    if clean_descriptors:
        nan_ratio = np.isnan(features).mean(axis=0)
        dropped_mask = nan_ratio > 0.98

        if np.any(dropped_mask):
            dropped_names = [molecular_descriptors[i][0] for i, drop in enumerate(dropped_mask) if drop]
            print(f"âš ï¸� Dropping {len(dropped_names)} descriptors with >98% missing values:")
            print("   " + ", ".join(dropped_names))

            features = features[:, ~dropped_mask]
            molecular_descriptors = [d for i, d in enumerate(molecular_descriptors) if not dropped_mask[i]]

    return features, molecular_descriptors


descriptor_names = [desc[0] for desc in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

def get_mol_descriptors(smile):

    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return [None] * len(descriptor_names)
    return calculator.CalcDescriptors(mol)



%%time
if CFG.smiles_fe == 'AUTO':
    descriptors = Parallel(n_jobs=-1)(
        delayed(get_mol_descriptors)(smile) for smile in tqdm(train_df['SMILES'])
    )

    descriptors_df = pd.DataFrame(descriptors, columns =descriptor_names)

    print('Filtering Null cols (50% of nulls)')
    desc_feats_null = descriptors_df.isnull().mean()
    valid_cols = desc_feats_null[desc_feats_null<0.5].index
    print(f'Total columns = {descriptors_df.shape[1]}')
    
    descriptors_df = descriptors_df.loc[:, valid_cols]
    
    print(f'Valid columns = {descriptors_df.shape[1]}')
elif CFG.smiles_fe == 'AUTOCORR2D':
    descriptors, molecular_descriptors = smiles_to_features(train_df['SMILES'].values, 
                                                            clean_descriptors=True)
    descriptor_names = [name for name, fun in molecular_descriptors]

    descriptors_df = pd.DataFrame(descriptors, columns =descriptor_names)




print(descriptors_df.shape)
descriptors_df.head()


def get_fingerprint(smile, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smile)
    generator = GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = generator.GetFingerprint(mol)
    # fp = AllChem.GetMorganFingerprintAsBitVect(mol, 
    #                                            radius=radius, 
    #                                            nBits=nBits)
    return np.array(fp)


fings =  np.array([get_fingerprint(smile) for smile in tqdm(train_df['SMILES'].tolist())])


%%time
if CFG.reduce_fings:
    reducer = umap.UMAP(n_neighbors=15, 
                        min_dist=CFG.min_dist, 
                        n_components=CFG.n_components, 
                        random_state=CFG.seed,
                        #metric="jaccard"
                       )
    
    # desc_matrix = reducer.fit_transform(descriptors_df.values)
    fings = reducer.fit_transform(fings)
    joblib.dump(reducer, "umap_model.pkl")


fings.shape


# nbrs = NearestNeighbors(n_neighbors=20).fit(descriptors_df.values)
# for target in targets:
#     mean_values = []
#     not_null_mask = train_df[~train_df[target].isnull()].index
#     null_mask = train_df[train_df[target].isnull()].index
    
#     search_matrix = descriptors_df.values[null_mask]
    
#     distances, indices = nbrs.kneighbors(search_matrix)

    
#     for top_smiliars_ids in indices:
#         not_null_indices = [x for x in top_smiliars_ids if x not in null_mask]
#         if len(not_null_indices)>0:
#             mean_value_aux = train_df.loc[not_null_indices, target].mean()
#         else:
#             #print('Dont Have Valid values in neighbors')
#             # mean_value_aux = train_df[target].mean()
#             mean_value_aux = np.nan
            
#         mean_values.append(mean_value_aux)
#     train_df.loc[null_mask, target] =  mean_values 


# null_percenteage = (train_df.isnull().mean()*100).round(2)
# null_percenteage = null_percenteage[null_percenteage > 0].sort_values()
# fig,ax = plt.subplots(figsize = (7,5))
# null_percenteage.plot(kind = 'bar', rot = 0, ax=ax)

# for i, (col, value) in enumerate(zip(null_percenteage.index, null_percenteage.values)):
#     label_text = f"{value}%"
#     ax.text(i, value + 1, label_text, 
#              ha='center', va='bottom', 
#              fontsize=8, weight = 600 ,rotation=0,
#              )

# ax.set_ylim(0,100)
    
# ax.set_title('Percentage of Nulls')
# plt.tight_layout()
# plt.show()


def add_fold_column(df, n_splits=5, shuffle=True, random_state=CFG.seed):
    df = df.copy()
    df['fold'] = -1  # Initialize fold column

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for fold_number, (_, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, 'fold'] = fold_number

    return df


train_df = add_fold_column(train_df)


train_df['fold'].value_counts()


def train_lgbm(X_train, y_train, X_val, y_val):
    """
    Trains a LightGBM regressor and plots training vs validation loss over boosting iterations.
    
    Parameters:
        X_train (pd.DataFrame or np.array): Training features
        y_train (pd.Series or np.array): Training targets
        X_val (pd.DataFrame or np.array): Validation features
        y_val (pd.Series or np.array): Validation targets
        num_boost_round (int): Maximum number of boosting iterations
        early_stopping_rounds (int): Early stopping criteria

    Returns:
        model: Trained LGBM model
    """
    # Create dataset wrappers
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # Train the model
    eval_result = {}
    model = lgb.train(
        CFG.lgb_params['params'],
        lgb_train,
        num_boost_round= CFG.lgb_params['num_boost_round'],
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        #evals_result=evals_result,
        #early_stopping_rounds=CFG.lgb_params['early_stopping_rounds'],
        callbacks=[lgb.early_stopping(CFG.lgb_params['params']['early_stopping_rounds']),
                   lgb.log_evaluation(0),
                   lgb.record_evaluation(eval_result)],
        #verbose_eval=False
    )


    return model, eval_result


desc_models, desc_results = {},{}
fings_models, fings_results = {},{}
for target in targets:
    print(target)
    desc_models_aux, desc_results_aux = [],[]
    fings_models_aux, fings_results_aux = [],[]
    for fold in sorted(train_df['fold'].unique()):

        dev_train, dev_val = train_df[train_df['fold'] != fold], train_df[train_df['fold'] == fold]
        train_ids, valid_ids = dev_train.index, dev_val.index
        
        mask = train_df[~train_df[target].isnull()].index
        train_ids_aux = [idx for idx in train_ids if idx in mask]

        val_ids_aux = [idx for idx in valid_ids if idx in mask]
        
        desc_train = descriptors_df.loc[train_ids_aux,:].values
        fings_train = fings[train_ids_aux]
        y_train = train_df.loc[train_ids_aux, target].values
    
        desc_val= descriptors_df.loc[val_ids_aux,:].values
        fings_val = fings[val_ids_aux]
        y_val = train_df.loc[val_ids_aux, target].values
    
        print('Fold = ',fold, 'Train = ', y_train.shape, 'Valid = ',y_val.shape)
        print()
    
        desc_model, desc_evals_result =  train_lgbm(X_train = desc_train, 
                                                    y_train = y_train, 
                                                    X_val = desc_val, 
                                                    y_val = y_val)
    
        fings_model, fings_evals_result =  train_lgbm(X_train = fings_train, 
                                                y_train = y_train, 
                                                X_val = fings_val, 
                                                y_val = y_val)
        desc_models_aux.append(desc_model)
        desc_results_aux.append(desc_evals_result)

        fings_models_aux.append(fings_model)
        fings_results_aux.append(fings_evals_result)
    
    desc_models[f'{target}'] = desc_models_aux
    desc_results[f'{target}'] = desc_results_aux
    
    fings_models[f'{target}'] = fings_models_aux
    fings_results[f'{target}'] = fings_results_aux
    print(40*'--')
    


print(desc_models.keys())
print(fings_models.keys())


def got_avg_error(list_evals):
    padded = np.array([
            np.pad(arr, (0, max(arr.shape[0] for arr in list_evals) - len(arr)), constant_values=np.nan)
            for arr in list_evals
        ])

    average = np.nanmean(padded, axis=0)

    return  average


 fig, ax = plt.subplots(2,5,figsize=(15, 8))


for col,name in enumerate(desc_results.keys()):

    eval_results_desc = desc_results[name]
    eval_results_fings = fings_results[name]

    train_loss_desc=[]
    valid_loss_des=[]
    train_loss_fings=[]
    valid_loss_fings=[]

    for i in range(5):

        try:
            train_loss_desc.append(np.array(eval_results_desc[i]['train'][CFG.model_loss]))
            valid_loss_des.append(np.array(eval_results_desc[i]['valid'][CFG.model_loss]))
            train_loss_fings.append(np.array(eval_results_fings[i]['train'][CFG.model_loss]))
            valid_loss_fings.append(np.array(eval_results_fings[i]['valid'][CFG.model_loss]))
        except:
            train_loss_desc.append(np.array(eval_results_desc[i]['train']['l1']))
            valid_loss_des.append(np.array(eval_results_desc[i]['valid']['l1']))
            train_loss_fings.append(np.array(eval_results_fings[i]['train']['l1']))
            valid_loss_fings.append(np.array(eval_results_fings[i]['valid']['l1']))

    train_loss_desc =  got_avg_error(train_loss_desc)
    valid_loss_des =  got_avg_error(valid_loss_des)
    train_loss_fings =  got_avg_error(train_loss_fings)
    valid_loss_fings =  got_avg_error(valid_loss_fings)
    
    ax[0,col].plot(train_loss_desc, label='Train RMSE')
    ax[0,col].plot(valid_loss_des, label='Validation RMSE')
    ax[0,col].set_xlabel('Iteration')
    ax[0,col].set_ylabel(CFG.model_loss)
    ax[0,col].set_title(f'DESC {name}')

    ax[0,col].legend()
    ax[0,col].grid(True)

    ax[1,col].plot(train_loss_fings, label='Train RMSE')
    ax[1,col].plot(valid_loss_fings, label='Validation RMSE')
    ax[1,col].set_xlabel('Iteration')
    ax[1,col].set_ylabel(CFG.model_loss)
    ax[1,col].set_title(f'FINGS {name}')

    ax[1,col].legend()
    ax[1,col].grid(True)


plt.tight_layout()
plt.show()


oof = pd.DataFrame()


for fold in sorted(train_df['fold'].unique()):
    desc_pred, fings_pred = [], []
    for target in targets:

        dev_train, dev_val = train_df[train_df['fold'] != fold].copy(), train_df[train_df['fold'] == fold].copy()
        train_ids, valid_ids = dev_train.index, dev_val.index
        
        
        desc_train = descriptors_df.loc[train_ids,:].values
        fings_train = fings[train_ids]
    
        desc_val= descriptors_df.loc[valid_ids,:].values
        fings_val = fings[valid_ids]
        y_val = train_df.loc[valid_ids, targets].values

        desc_model = desc_models[target][fold]
        fings_model = fings_models[target][fold]

        desc_pred_propety = desc_model.predict(desc_val)
        fings_pred_propety = fings_model.predict(fings_val)



        desc_pred.append(desc_pred_propety)
        fings_pred.append(fings_pred_propety)

    desc_pred = np.column_stack(desc_pred)   
    fings_pred = np.column_stack(fings_pred)
    dev_val[[f'{col}_desc_pred' for col in targets]] = desc_pred
    dev_val[[f'{col}_fings_pred' for col in targets]] = fings_pred
    oof = pd.concat([oof, dev_val.reset_index(drop = True)])



import open_polymer_2025_metric as metric

for target in targets:

    oof[f'{target}_ensamble_pred'] = (oof[f'{target}_desc_pred'] + oof[f'{target}_fings_pred'])/2
    oof[f'{target}_fings_pred'] = oof[f'{target}_fings_pred'].fillna(metric.NULL_FOR_SUBMISSION)
    oof[f'{target}_desc_pred'] = oof[f'{target}_desc_pred'].fillna(metric.NULL_FOR_SUBMISSION)
    oof[f'{target}_ensamble_pred'] = oof[f'{target}_ensamble_pred'].fillna(metric.NULL_FOR_SUBMISSION)
    
    
    comp_score_desc = metric.scaling_error(oof[target], oof[f'{target}_desc_pred'], target)
    comp_score_fings = metric.scaling_error(oof[target], oof[f'{target}_fings_pred'], target)
    comp_score_ensam = metric.scaling_error(oof[target], oof[f'{target}_ensamble_pred'], target)

    print(f'{target} - Score:')
    print(f'Descriptors = {comp_score_desc}')
    print(f'Fings = {comp_score_fings}')
    print(f'Ensamble = {comp_score_ensam}')

oof_desc_pred = oof[['id'] + [f'{col}_desc_pred' for col in targets]]
oof_desc_pred = oof_desc_pred.rename(columns={f'{col}_desc_pred':col for col in targets})

oof_fings_pred = oof[['id'] + [f'{col}_fings_pred' for col in targets]]
oof_fings_pred = oof_fings_pred.rename(columns={f'{col}_fings_pred':col for col in targets})

oof_ensam_pred = oof[['id'] + [f'{col}_ensamble_pred' for col in targets]]
oof_ensam_pred = oof_ensam_pred.rename(columns={f'{col}_ensamble_pred':col for col in targets})

estimated_lb_score_desc = metric.score(oof[['id'] + targets], oof_desc_pred, 'id')
estimated_lb_score_fings = metric.score(oof[['id'] + targets],oof_fings_pred, 'id')
estimated_lb_score_ensam = metric.score(oof[['id'] + targets],oof_ensam_pred, 'id')

print()
print('Estimate LB Score')
print(f'Descriptors = {estimated_lb_score_desc}')
print(f'Fings = {estimated_lb_score_fings}')
print(f'Ensamble = {estimated_lb_score_ensam}')


sample_sub = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
sample_sub.head()


test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
sub_df = test_df.copy().drop(['SMILES'], axis = 1)
test_df.head()


if CFG.smiles_fe == 'AUTO':
    test_descriptors = Parallel(n_jobs=-1)(
        delayed(get_mol_descriptors)(smile) for smile in tqdm(test_df['SMILES'])
    )

    test_descriptors_df = pd.DataFrame(test_descriptors, columns =descriptor_names)

    print('Filtering Null cols (50% of nulls)')
    desc_feats_null = test_descriptors_df.isnull().mean()
    valid_cols = desc_feats_null[desc_feats_null<0.5].index
    print(f'Total columns = {test_descriptors_df.shape[1]}')
    
    test_descriptors_df = test_descriptors_df.loc[:, valid_cols]
    
    print(f'Valid columns = {descriptors_df.shape[1]}')
elif CFG.smiles_fe == 'AUTOCORR2D':
    test_descriptors = [process_single_smiles(smile) for smile in test_df['SMILES'].values]
    descriptor_names = [name for name, fun in MOLECULAR_DESCRIPTORS]

    test_descriptors_df = pd.DataFrame(test_descriptors, columns =descriptor_names)
    test_descriptors_df = test_descriptors_df.loc[:,descriptors_df.columns]


test_descriptors_df.shape, descriptors_df.shape


test_fings =  np.array([get_fingerprint(smile) for smile in tqdm(test_df['SMILES'].tolist())])
print(test_fings.shape)


test_desc_pred, test_fings_pred = [],[]
target_cols = sample_sub.columns[1:]
for fold in sorted(train_df['fold'].unique()):
    desc_pred, fings_pred = [], []
    for target in target_cols:

        desc_model = desc_models[target][fold]
        fings_model = fings_models[target][fold]

        desc_pred_propety = desc_model.predict(test_descriptors_df.values)
        fings_pred_propety = fings_model.predict(test_fings)
        
       

        desc_pred.append(desc_pred_propety)
        fings_pred.append(fings_pred_propety)


    desc_pred = np.column_stack(desc_pred)   
    fings_pred = np.column_stack(fings_pred)

    if fold == 0:
        test_desc_pred = desc_pred
        test_fings_pred = fings_pred
    else:
        test_desc_pred+= desc_pred
        test_fings_pred += fings_pred

test_desc_pred = test_desc_pred/5
test_fings_pred = test_fings_pred/5


if CFG.pred_strat == 'ensamble':
    test_pred = (test_desc_pred+test_fings_pred)/2
elif CFG.pred_strat == 'desc':
    test_pred = test_desc_pred
elif CFG.pred_strat == 'fings':
    test_pred = test_fings_pred


sub_df[target_cols] = test_pred
sub_df.head()


sub_df.to_csv('submission.csv', index=False)


train_df.head()

