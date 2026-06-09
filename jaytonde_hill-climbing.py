!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


!pip install --no-index -U --find-links=/kaggle/input/tabm-tabular-dl-library tabm==0.0.1.dev0
!pip -q install /kaggle/input/pytorchtabnet/pytorch_tabnet-4.1.0-py3-none-any.whl


!pip -q install /kaggle/input/tabpfn-v2/tabpfn-2.0.0-py3-none-any.whl


import numpy as np
import pandas as pd
from scipy.stats import rankdata 

import sys
sys.path.append('/kaggle/input/tabm-tabular-dl-library')

import os
import tabm
import math
import torch
import random
import warnings
from tqdm import tqdm
import pandas as pd
import numpy as np
import rtdl_num_embeddings
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from sklearn.model_selection import KFold
from scipy.stats import rankdata 
#from colorama import Fore, Style
from typing import Optional, Tuple
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from tabm_reference import Model, make_parameter_groups
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from pytorch_tabnet.tab_model import TabNetRegressor

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding
from tensorflow.keras.layers import Concatenate, BatchNormalization
import tensorflow.keras.backend as K
from pytorch_tabnet.tab_model import TabNetRegressor
from tabpfn import TabPFNRegressor


experiments = [
"/kaggle/input/xgboost-exp-01",
"/kaggle/input/catboost-exp-01", 
"/kaggle/input/lgbm-exp-01",
"/kaggle/input/xgboost-exp-02",
"/kaggle/input/catboost-exp-02",
"/kaggle/input/lgbm-exp-03",
"/kaggle/input/catboost-exp-03",
"/kaggle/input/tabm-exp-01",
"/kaggle/input/nn-exp-01",
"/kaggle/input/tn-exp-01",
"/kaggle/input/tf-exp-01",
"/kaggle/input/svr-exp-01",
"/kaggle/input/abd-exp-01",
"/kaggle/input/catboost-exp-04",
"/kaggle/input/lgbm-exp-04",
"/kaggle/input/tabm-exp-02",
"/kaggle/input/ds-exp-01",
"/kaggle/input/nn-exp-02",
"/kaggle/input/nn-exp-04",
"/kaggle/input/catboost-exp-05",
"/kaggle/input/xgboost-exp-05",
"/kaggle/input/lgbm-exp-05",
"/kaggle/input/tn-exp-02",
"/kaggle/input/vr-exp-01",
"/kaggle/input/tt-exp-01",
"/kaggle/input/en-exp-01",
"/kaggle/input/en-exp-02",
"/kaggle/input/nn-exp-05",
"/kaggle/input/rf-exp-05",
"/kaggle/input/mcts-exp-02",
"/kaggle/input/catboost-exp-06",
"/kaggle/input/xgboost-exp-06",
"/kaggle/input/lgbm-exp-06",
"/kaggle/input/nn-exp-06",
"/kaggle/input/xgboost-exp-07",
"/kaggle/input/ri-exp-01",
"/kaggle/input/xgboost-exp-08",
"/kaggle/input/catboost-exp-08",
"/kaggle/input/lgbm-exp-08",
"/kaggle/input/xgboost-exp-09",
"/kaggle/input/prlnn-exp-01",
"/kaggle/input/ri-exp-06",
"/kaggle/input/xgboost-exp-10",
"/kaggle/input/lasso-exp-01",
"/kaggle/input/lir-exp-01",
"/kaggle/input/svr-exp-06",
"/kaggle/input/et-exp-01",
"/kaggle/input/lasso-exp-02",
"/kaggle/input/pr-exp-06",
"/kaggle/input/tr-exp-06",
"/kaggle/input/lasso-exp-06",
"/kaggle/input/ransac-exp-01",
"/kaggle/input/cnn-exp-01",
"/kaggle/input/ts-exp-01"
]


import pandas as pd
import pandas.api.types
import numpy as np
from lifelines.utils import concordance_index

class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'predictions'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []

    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))


parquet_experiments = [
    "/kaggle/input/abd-exp-01",
    "/kaggle/input/lgbm-exp-04", 
    "/kaggle/input/catboost-exp-01", 
    "/kaggle/input/lgbm-exp-01", 
    "/kaggle/input/lgbm-exp-03", 
    "/kaggle/input/ds-exp-01",
    "/kaggle/input/nn-exp-02",
    "/kaggle/input/nn-exp-04",
    "/kaggle/input/tabm-exp-03",
    "/kaggle/input/catboost-exp-05",
    "/kaggle/input/xgboost-exp-05",
    "/kaggle/input/lgbm-exp-05",
    "/kaggle/input/tn-exp-02",
    "/kaggle/input/vr-exp-01",
    "/kaggle/input/tt-exp-01",
    "/kaggle/input/en-exp-01",
    "/kaggle/input/svr-exp-01",
    "/kaggle/input/en-exp-02",
    "/kaggle/input/nn-exp-05",
    "/kaggle/input/rf-exp-05",
    "/kaggle/input/mcts-exp-02",
    "/kaggle/input/catboost-exp-06",
    "/kaggle/input/xgboost-exp-06",
    "/kaggle/input/lgbm-exp-06",
    "/kaggle/input/nn-exp-06",
    "/kaggle/input/xgboost-exp-07",
    "/kaggle/input/ri-exp-01",
    "/kaggle/input/xgboost-exp-08",
    "/kaggle/input/catboost-exp-08",
    "/kaggle/input/lgbm-exp-08",
    "/kaggle/input/xgboost-exp-09",
    "/kaggle/input/prlnn-exp-01",
    "/kaggle/input/ri-exp-06",
    "/kaggle/input/xgboost-exp-10",
    "/kaggle/input/lasso-exp-01",
    "/kaggle/input/lir-exp-01",
    "/kaggle/input/svr-exp-06",
    "/kaggle/input/et-exp-01",
    "/kaggle/input/lasso-exp-02",
    "/kaggle/input/pr-exp-06",
    "/kaggle/input/tr-exp-06",
    "/kaggle/input/lasso-exp-06",
    "/kaggle/input/ransac-exp-01",
    "/kaggle/input/cnn-exp-01",
    "/kaggle/input/ts-exp-01"
]


def compute_metric_cindex(oof, exp_name, fast=False):    
    if fast:
        y_true = oof[["ID","efs","efs_time","race_group"]].copy()
        y_pred = oof[["ID","predictions"]].copy()
        return optimized_score(y_true.copy(), y_pred.copy(), "ID"), oof
    else:
        y_true = oof[["ID","efs","efs_time","race_group"]].copy()
        y_pred = oof[["ID","predictions"]].copy()
        return score(y_true.copy(), y_pred.copy(), "ID"), oof


pip install openpyxl


%%time

from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
import gc
from scipy.stats import rankdata
from lifelines.utils import concordance_index
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

class ParticipantVisibleError(Exception):
    pass



def fast_concordance_index(event_times, predicted_scores, event_observed):
    """Faster implementation of concordance index using NumPy operations"""
    # Convert to numpy arrays for faster operations
    event_times = np.asarray(event_times)
    predicted_scores = np.asarray(predicted_scores)
    event_observed = np.asarray(event_observed)
    
    # Only consider pairs where at least one has an event
    mask = event_observed == 1
    
    if not np.any(mask):
        return 0.0
    
    # Get indices of samples with events
    event_indices = np.where(mask)[0]
    
    # Initialize counters
    concordant = 0
    discordant = 0
    tied_risk = 0
    pairs = 0

    # For each sample with an event
    for i in event_indices:
        # Find all samples with longer survival time
        longer_survival = event_times > event_times[i]
        
        # For samples with same survival time, only consider if they didn't have an event
        same_survival = (event_times == event_times[i]) & (event_observed == 0)
        
        # Combine masks
        comparable = longer_survival | same_survival
        
        if not np.any(comparable):
            continue
        
        # Get predictions for comparable samples
        comp_scores = predicted_scores[comparable]
        current_score = predicted_scores[i]
        
        # Count concordant, discordant, and tied pairs
        concordant += np.sum(comp_scores < current_score)
        discordant += np.sum(comp_scores > current_score)
        tied_risk += np.sum(comp_scores == current_score)
        
        # Update total pairs count
        pairs += np.sum(comparable)
    
    if pairs == 0:
        return 0.0
    
    return (concordant + 0.5 * tied_risk) / pairs


def optimized_score(solution_data, submission_data, row_id_column_name=None):
    """Optimized scoring function using NumPy operations"""
    if row_id_column_name:
        # Extract necessary columns and convert to NumPy arrays
        event = solution_data['efs'].values
        time = solution_data['efs_time'].values
        race_groups = solution_data['race_group'].values
        predictions = submission_data['predictions'].values
    else:
        # Assume data is already in the right format
        event = solution_data[:, 0]  # efs
        time = solution_data[:, 1]   # efs_time
        race_groups = solution_data[:, 2]  # race_group
        predictions = submission_data  # predictions
    
    # Get unique race groups
    unique_races = np.unique(race_groups)
    c_indices = []
    
    # Calculate c-index for each race group
    for race in unique_races:
        mask = race_groups == race
        
        # Use lifelines concordance_index for correctness
        c_index_race = concordance_index(
            time[mask],
            -predictions[mask],
            event[mask]
        )
        c_indices.append(c_index_race)
    
    c_indices = np.array(c_indices)
    return float(np.mean(c_indices) - np.sqrt(np.var(c_indices)))


multiprocessing.cpu_count()


from tqdm import tqdm
import pandas as pd

# Initialize empty list to store individual prediction dataframes
dfs_to_merge = []

# Load first experiment to get the base structure
base_exp = experiments[0]
if base_exp in parquet_experiments:
    base_df = pd.read_parquet(base_exp + '/' + (base_exp.split('/')[-1]).replace('-','_') + '_oof.parquet')
else:
    base_df = pd.read_excel(base_exp + '/' + (base_exp.split('/')[-1]).replace('-','_') + '_oof.xlsx')

# Create base dataframe with ID and target variables
preds_df = base_df[["ID", "efs", "efs_time", "race_group"]].copy()

# Load and merge predictions from each experiment
for exp_name in tqdm(experiments):
    # Read the prediction file
    if exp_name in parquet_experiments:
        if "mcts-exp-02" in exp_name:
            curr_df = pd.read_parquet(exp_name + '/mcts_exp_01' + '_oof.parquet')
        elif "ts-exp-01" in exp_name:
            prev_df = pd.read_parquet("/kaggle/input/cnn-exp-01/cnn_exp_01_oof.parquet")
            curr_df = pd.read_parquet("/kaggle/input/ts-exp-01/ts_exp_01_oof.parquet")
            curr_df["ID"] = prev_df["ID"]
        else:
            curr_df = pd.read_parquet(exp_name + '/' + (exp_name.split('/')[-1]).replace('-','_') + '_oof.parquet')
    else:
        curr_df = pd.read_excel(exp_name + '/' + (exp_name.split('/')[-1]).replace('-','_') + '_oof.xlsx')
    
    # Create a temporary dataframe with ID and predictions
    temp_df = curr_df[["ID", "predictions"]].copy()
    temp_df = temp_df.rename(columns={"predictions": exp_name})
    
    # Merge with the main dataframe
    preds_df = preds_df.merge(temp_df, on="ID", how="left")


preds_df.shape, preds_df.columns 


len(experiments)


best_score    = 0
best_index    = -1
best_ensemble = 0

for k,name in enumerate(experiments):
    oof_pre = preds_df[["ID","efs","efs_time","race_group",name]]
    oof_pre = oof_pre.rename(columns={name: "predictions"})
    s, oof = compute_metric_cindex(oof_pre, name, fast=True)
    if s > best_score:
        best_score    = s
        best_index    = name
        best_ensemble = oof
        
    print(f'C-index {s} {name}') 
print()
print(f'Best single model is {best_index} with C-Index = {best_score}')
experiments.remove(best_index)
first_best_index = best_index


experiments


USE_NEGATIVE_WGT = True


indices        = [best_index]
old_best_score = best_score


# PREPARE/MOVE VARIABLES TO GPU FOR SPEED UP
best_ensemble = best_ensemble
start         = -0.50
if not USE_NEGATIVE_WGT: start = 0.01
ww            = np.arange(start,0.51,0.01) # GPU
nn            = len(ww)


# BEGIN HILL CLIMBING
models  = [best_index]
weights = []
metrics = [best_score]


models, metrics, ww


def evaluate_weight(args):
    """Function to evaluate a single weight for a model - designed for parallel execution"""
    weight, best_preds, model_oof, solution_data = args
    
    # Create potential ensemble using NumPy operations
    best_ranks = rankdata(best_preds)
    model_ranks = rankdata(model_oof)
    potential_ensemble = (1 - weight) * best_ranks + weight * model_ranks
    
    # Calculate score
    new_score = optimized_score(solution_data, potential_ensemble)
    
    return weight, new_score

def optimize_ensemble(experiments, initial_ensemble, weights_range, score_function, max_iterations=100):
    # Convert initial ensemble to NumPy arrays for faster processing
    ids = initial_ensemble["ID"].values
    initial_preds = initial_ensemble["predictions"].values
    
    # Prepare solution data as NumPy array
    solution_data = np.column_stack((
        preds_df["efs"].values,
        preds_df["efs_time"].values,
        preds_df["race_group"].values
    ))
    
    # Initialize variables
    iteration = 0
    best_score = score_function(
        solution_data,
        initial_preds,
        None
    )
    
    model_weights = {}
    remaining_experiments = experiments.copy()
    best_preds = initial_preds.copy()
    
    # Get number of CPU cores for parallel processing
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    
    while remaining_experiments and iteration < max_iterations:
        print(f"{iteration}th iteration")
        iteration += 1
        best_iteration = {
            'index': -1,
            'weight': 0,
            'score': best_score
        }
        
        # Try each remaining model
        for model_path in remaining_experiments:
            try:
                # Load model OOF predictions
                model_name = model_path.split("/")[-1].replace('-', '_')
                model_oof = preds_df[model_path].values
                
                # Prepare arguments for parallel processing
                args_list = [(weight, best_preds, model_oof, solution_data) 
                             for weight in weights_range]
                
                # Process weights in parallel
                with ProcessPoolExecutor(max_workers=num_cores) as executor:
                    futures = [executor.submit(evaluate_weight, args) for args in args_list]
                    
                    # Process results as they complete
                    for future in tqdm(as_completed(futures), total=len(futures), 
                                      desc=f"Testing weights for {model_name}"):
                        weight, new_score = future.result()
                        
                        # Update best if improved
                        if new_score > best_iteration['score']:
                            best_iteration.update({
                                'index': model_path,
                                'weight': weight,
                                'score': new_score
                            })
                
                # Clean up
                del model_oof
                gc.collect()
                
            except Exception as e:
                print(f"Error processing {model_path}: {str(e)}")
                continue
        
        # Check if we found an improvement
        if best_iteration['index'] == -1:
            print("No improvement found, stopping")
            print(best_iteration)
            break
            
        # Update ensemble with best model found
        best_score = best_iteration['score']
        model_weights[best_iteration['index']] = best_iteration['weight']
        remaining_experiments.remove(best_iteration['index'])
        
        print(
            f"Iteration {iteration}: Added {best_iteration['index']} "
            f"with weight {best_iteration['weight']:.4f}, "
            f"Score: {best_iteration['score']:.4f}"
        )
        
        # Update best ensemble for next iteration
        model_oof = preds_df[best_iteration['index']].values
        best_ranks = rankdata(best_preds)
        model_ranks = rankdata(model_oof)
        best_preds = (1 - best_iteration['weight']) * best_ranks + best_iteration['weight'] * model_ranks
        
    # Create final ensemble DataFrame for compatibility
    final_ensemble = pd.DataFrame({
        "ID": ids,
        "predictions": best_preds
    })
    
    return model_weights, best_score, final_ensemble

# Run the optimized ensemble function
model_weights, best_score, final_ensemble = optimize_ensemble(experiments, best_ensemble, ww, optimized_score)



# %%time

# from tqdm import tqdm
# import pandas as pd
# import numpy as np
# from typing import List, Dict, Tuple
# import logging
# import gc

# def optimize_ensemble(experiments, initial_ensemble, weights_range, score_function, max_iterations = 100):
    
#     # Initialize variables
#     iteration  = 0
#     best_score = score_function(
#         initial_ensemble[["ID", "efs", "efs_time", "race_group"]].copy(),
#         initial_ensemble[["ID", "predictions"]].copy(),
#         "ID"
#     )
    
#     model_weights         = {}
#     remaining_experiments = experiments.copy()
#     best_ensemble         = initial_ensemble.copy()
    
#     while remaining_experiments and iteration < max_iterations:
#         print(f"{iteration}th iteration")
#         iteration += 1
#         best_iteration = {
#             'index' : -1,
#             'weight': 0,
#             'score' : best_score
#         }
        
#         # Try each remaining model
#         for model_path in remaining_experiments:
#             try:
#                 #print(f"Iteration {iteration}: Trying model {model_path}")
                
#                 # Load model OOF predictions
#                 model_name = model_path.split("/")[-1].replace('-', '_')
#                 model_oof  = preds_df[model_path]
                
#                 # Try different weights
#                 for weight in tqdm(weights_range, desc=f"Testing weights for {model_name}"):
#                     # Create potential ensemble
#                     potential_ensemble = pd.DataFrame({
#                         "ID": best_ensemble["ID"],
#                         "predictions": (1 - weight) * rankdata(best_ensemble["predictions"]) + 
#                                      weight * rankdata(model_oof)
#                     })
                    
#                     # Evaluate new ensemble
#                     new_score = score_function(
#                         preds_df[["ID", "efs", "efs_time", "race_group"]].copy(),
#                         potential_ensemble.copy(),
#                         "ID"
#                     )
                    
#                     # Update best if improved
#                     if new_score > best_iteration['score']:
#                         best_iteration.update({
#                             'index' : model_path,
#                             'weight': weight,
#                             'score' : new_score
#                         })
                
#                 # Clean up
#                 del model_oof
#                 gc.collect()
                
#             except Exception as e:
#                 print(f"Error processing {model_path}: {str(e)}")
#                 continue
        
#         # Check if we found an improvement
#         if best_iteration['index'] == -1:
#             print("No improvement found, stopping")
#             print(best_iteration)
#             break
            
#         # Update ensemble with best model found
#         best_score                             = best_iteration['score']
#         model_weights[best_iteration['index']] = best_iteration['weight']
#         remaining_experiments.remove(best_iteration['index'])
        
#         print(
#             f"Iteration {iteration}: Added {best_iteration['index']} "
#             f"with weight {best_iteration['weight']:.4f}, "
#             f"Score: {best_iteration['score']:.4f}"
#         )
        
#         # Update best ensemble for next iteration
#         model_oof = preds_df[best_iteration['index']]
#         best_ensemble["predictions"] = (
#             (1 - best_iteration['weight']) * rankdata(best_ensemble["predictions"]) + 
#             best_iteration['weight'] * rankdata(model_oof)
#         )
        
#     return model_weights, best_score


# model_weights, best_score = optimize_ensemble(experiments, best_ensemble, ww, score)


model_weights, best_score, final_ensemble


# first_best_index, experiments


# class CFG:
#     folds = 10

# # https://www.kaggle.com/datasets/jsday96/mcts-tabm-models/data?select=TabMRegressor.py
# class TabMRegressor:
#     def __init__(
#         self,
#         arch_type: str        = 'tabm-mini',
#         backbone: dict        = {'type': 'MLP', 'n_blocks': 3, 'd_block': 512, 'dropout': 0.1},
#         d_embedding: int      = 64,  # Only used for 'tabm-mini'
#         bin_count: int        = 48,  # Only used for 'tabm-mini'
#         k: int                = 32,
#         learning_rate: float  = 1e-4,
#         weight_decay: float   = 1e-3,
#         clip_grad_norm: bool  = True,
#         max_epochs: int       = 100,
#         patience: int         = 15,
#         batch_size: int       = 32,
#         compile_model: bool   = False,
#         device: Optional[str] = 'cuda:0',
#         random_state: int     = 0,
#         verbose: bool         = True
#     ):
#         self.arch_type = arch_type
#         self.backbone = backbone
#         self.d_embedding = d_embedding
#         self.bin_count = bin_count
#         self.k = k
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.clip_grad_norm = clip_grad_norm
#         self.max_epochs = max_epochs
#         self.patience = patience
#         self.batch_size = batch_size
#         self.compile_model = compile_model
#         self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
#         self.random_state = random_state
#         self.verbose = verbose

#     def fit(
#         self,
#         X: pd.DataFrame,
#         y: np.array,
#         eval_set: Tuple[pd.DataFrame, np.array]
#     ):
#         # PREPROCESS DATA.
#         X_cat_train, X_cont_train, cat_cardinalities, y_train = self._preprocess_data(X, y, training=True)
#         X_cat_val, X_cont_val, _, y_val = self._preprocess_data(eval_set[0], eval_set[1], training=False)

#         # CREATE MODEL & TRAINING ALGO.
#         bins = rtdl_num_embeddings.compute_bins(X_cont_train, n_bins=self.bin_count) if self.arch_type == 'tabm-mini' else None
#         self.model = Model(
#             n_num_features=X_cont_train.shape[1],
#             cat_cardinalities=cat_cardinalities,
#             n_classes=None,
#             backbone=self.backbone,
#             bins=bins,
#             num_embeddings=(
#                 None
#                 if bins is None
#                 else {
#                     'type': 'PiecewiseLinearEmbeddings',
#                     'd_embedding': self.d_embedding,
#                     'activation': True,
#                     'version': 'B',
#                 }
#             ),
#             arch_type=self.arch_type,
#             k=self.k,
#         ).to(self.device)
#         optimizer = torch.optim.AdamW(make_parameter_groups(self.model), lr=self.learning_rate, weight_decay=self.weight_decay)
#         if self.compile_model:
#             self.model = torch.compile(self.model)

#         loss_fn = torch.nn.MSELoss().to(self.device)
#         # TRAIN & TEST MODEL.
#         best = {
#             'epoch': -1,
#             'eval_loss': math.inf,
#             'model_state_dict': None,
#         }
#         remaining_patience = self.patience
#         epoch_size = math.ceil(len(X) / self.batch_size)


#         for epoch in range(self.max_epochs):
#             # TRAIN.
#             optimizer.zero_grad()
#             train_losses = []
#             progress_bar = torch.randperm(len(y_train), device=self.device).split(self.batch_size)
#             progress_bar = tqdm(progress_bar, desc=f'Epoch {epoch}', total=epoch_size) if self.verbose else progress_bar
#             for batch_idx in progress_bar:
#                 self.model.train()

#                 with torch.amp.autocast(device_type='cuda', dtype = torch.bfloat16):
#                     y_pred = self.model(
#                         X_cont_train[batch_idx],
#                         X_cat_train[batch_idx],
#                     ).squeeze(-1).float()

#                 loss = loss_fn(y_pred.flatten(0, 1), y_train[batch_idx].repeat_interleave(self.k))
#                 loss.backward()
#                 if self.clip_grad_norm:
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#                 optimizer.step()

#                 train_losses.append(loss.item())


#              # EVALUATE.
#             self.model.eval()
#             val_losses = []
#             with torch.no_grad():
#                 for batch_idx in torch.arange(0, len(y_val), self.batch_size, device=self.device):
#                     y_pred = self.model(
#                         X_cont_val[batch_idx:batch_idx+self.batch_size],
#                         X_cat_val[batch_idx:batch_idx+self.batch_size],
#                     ).squeeze(-1).float()

#                     loss = loss_fn(y_pred.flatten(0, 1), y_val[batch_idx:batch_idx+self.batch_size].repeat_interleave(self.k))
#                     val_losses.append(loss.item())


#             # PRINT INFO.
#             mean_train_loss = np.mean(train_losses)
#             mean_val_loss = np.mean(val_losses)
#             if self.verbose:
#                 print(f'Epoch {epoch} | Train Loss: {mean_train_loss} | Val Loss: {mean_val_loss}')


#             # COMPARE TO BEST.
#             if mean_val_loss < best['eval_loss']:
#                 best['epoch'] = epoch
#                 best['eval_loss'] = mean_val_loss
#                 best['model_state_dict'] = self.model.state_dict()
#                 remaining_patience = self.patience
                
#                 if self.verbose:
#                     print('ðŸŒ¸ New best epoch! ðŸŒ¸')
#             else:
#                 remaining_patience -= 1

#             # EARLY STOPPING.
#             if remaining_patience == 0:
#                 break

#             # RESTORE BEST MODEL.
#             self.model.load_state_dict(best['model_state_dict'])


#     def predict(
#         self,
#         X: pd.DataFrame,
#         batch_size: Optional[int] = 8096
#     ) -> np.ndarray:
#         # PREPROCESS DATA.
#         X_cat, X_cont, _, _ = self._preprocess_data(X, y=None, training=False)

#         # PREDICT.
#         self.model.eval()
#         y_pred = []
#         with torch.no_grad():
#             for batch_idx in torch.arange(0, len(X), batch_size, device=self.device):
#                 y_pred.append(
#                     self.model(
#                         X_cont[batch_idx:batch_idx+batch_size],
#                         X_cat[batch_idx:batch_idx+batch_size],
#                     ).squeeze(-1).float().cpu().numpy()
#                 )

#         y_pred = np.concatenate(y_pred)


#         # DENORMALIZE TARGETS.
#         y_pred = y_pred * self._target_std + self._target_mean


#         # COMPUTE ENSEMBLE MEAN.
#         y_pred = np.mean(y_pred, axis=1)

#         return y_pred


#     def _preprocess_data(self, X: pd.DataFrame, y: pd.Series, training: bool):
#         # PICK NON-CONSTANT COLUMNS.
#         if training:
#             self._non_constant_columns = X.columns[X.nunique() > 1]

#         X = X[self._non_constant_columns]

#         # SEPARATE CATEGORICAL & CONTINUOUS FEATURES.
#         categorical_features = [col for col in X.columns if X[col].dtype.name == 'object']
#         X_cat = X[categorical_features].to_numpy()
#         X_cont = X.drop(columns=categorical_features).to_numpy()

#         # ENCODE CATEGORICAL FEATURES.
#         cat_cardinalities = [X[col].nunique() for col in categorical_features]

#         if training:
#             self._categorical_encoders = [
#                 OrdinalEncoder()
#                 for _ in range(X_cat.shape[1])
#             ]
#         X_cat = np.concatenate([
#             encoder.fit_transform(X_cat[:, i:i+1])
#             for i, encoder in enumerate(self._categorical_encoders)
#         ], axis=1)

#         # NORMALIZE TARGETS.
#         if training:
#             self._target_mean = y.mean()
#             self._target_std = y.std()

#             y = (y - self._target_mean) / self._target_std


#         # SCALE CONTINUOUS FEATURES.
#         if training:
#             noise = (
#                 np.random.default_rng(0)
#                 .normal(0.0, 1e-5, X_cont.shape)
#                 .astype(X_cont.dtype)
#             )
#             self._cont_feature_preprocessor = QuantileTransformer(
#                 n_quantiles=max(min(len(X) // 30, 1000), 10),
#                 output_distribution='normal',
#                 subsample=10**9,
#             ).fit(X_cont + noise)

#         X_cont = self._cont_feature_preprocessor.transform(X_cont)


#         # CONVERT TO TENSORS.
#         X_cat = torch.tensor(X_cat, dtype=torch.long, device=self.device)
#         X_cont = torch.tensor(X_cont, dtype=torch.float32, device=self.device)

#         if y is not None:
#             y = torch.tensor(y, dtype=torch.float32, device=self.device)

#         return X_cat, X_cont, cat_cardinalities, y



# def get_tabm_features(data):
#     RMV = ["ID","efs","efs_time","y","fold"]
#     FEATURES = [c for c in data.columns if not c in RMV]
    
#     RMV              = ['ID']
#     X_test           = data.drop(RMV, axis=1)
#     y_pred           = data[['ID']]
    
#     #print("X_test shape:", X_test.shape, '\n')
    
#     cat_cols         = X_test.select_dtypes(include=['object']).columns.tolist()
#     num_cols         = X_test.select_dtypes(exclude=['object']).columns.tolist()
    
#     # Preprocessing categorical
#     imputer          = SimpleImputer(strategy='constant', fill_value='NAN')
#     X_test[cat_cols] = imputer.fit_transform(X_test[cat_cols])

#     # Preprocessing numerical
#     imputer          = SimpleImputer(strategy="median")
#     X_test[num_cols] = imputer.fit_transform(X_test[num_cols])

#     return X_test,FEATURES


# def prepare_features(model_path, train, test):

#     RMV = ["ID","efs","efs_time","y"]
#     FEATURES = [c for c in train.columns if not c in RMV]
#     #print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
    

#     CATS = []
#     for c in FEATURES:
#         if train[c].dtype=="object":
#             CATS.append(c)
#             train[c] = train[c].fillna("NAN")
#             test[c]  = test[c].fillna("NAN")
#         elif "DeepTabels" in model_path or "tn" in model_path or "svr" in model_path:
#             train[c] = train[c].fillna(-1)
#             test[c]  = test[c].fillna(-1)
            
        
#     #print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")
    
#     combined = pd.concat([train,test],axis=0,ignore_index=True)
#     #print("Combined data shape:", combined.shape )
    
#     # LABEL ENCODE CATEGORICAL FEATURES
#     #print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
#     for c in FEATURES:
    
#         # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
#         if c in CATS:
#             #print(f"{c}, ",end="")
#             combined[c],_ = combined[c].factorize()
#             combined[c]  -= combined[c].min()
#             combined[c]   = combined[c].astype("int32")
#             combined[c]   = combined[c].astype("category")
            
#         # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
#         else:
#             if combined[c].dtype =="float64":
#                 combined[c]      = combined[c].astype("float32")
#             if combined[c].dtype =="int64":
#                 combined[c]      = combined[c].astype("int32")
        
#     train = combined.iloc[:len(train)].copy()
#     test  = combined.iloc[len(train):].reset_index(drop=True).copy()
                
#     return train, test, FEATURES


# def get_nn_features(train, test):
    
#     CAT_SIZE = []
#     CAT_EMB  = []
#     NUMS     = []
#     CATS     = []

#     RMV = ["ID","efs","efs_time","y","fold"]
#     FEATURES = [c for c in train.columns if not c in RMV]
    
#     for c in FEATURES:
#         if train[c].dtype=="object":
#             train[c] = train[c].fillna("NAN")
#             test[c]  = test[c].fillna("NAN")
#             CATS.append(c)
#         elif not "age" in c:
#             train[c] = train[c].astype("str")
#             test[c]  = test[c].astype("str")
#             CATS.append(c)


#     combined = pd.concat([train,test],axis=0,ignore_index=True)
#     for c in FEATURES:
#         if c in CATS:
#             # LABEL ENCODE
#             combined[c],_ = combined[c].factorize()
#             combined[c] -= combined[c].min()
#             combined[c] = combined[c].astype("int32")
#             #combined[c] = combined[c].astype("category")

#             n = combined[c].nunique()
#             mn = combined[c].min()
#             mx = combined[c].max()
#             #print(f'{c} has ({n}) unique values')
    
#             CAT_SIZE.append(mx+1) 
#             CAT_EMB.append( int(np.ceil( np.sqrt(mx+1))) ) 
#         else:
#             if combined[c].dtype=="float64":
#                 combined[c] = combined[c].astype("float32")
#             if combined[c].dtype=="int64":
#                 combined[c] = combined[c].astype("int32")
                
#             m = combined[c].mean()
#             s = combined[c].std()
#             combined[c] = (combined[c]-m)/s
#             combined[c] = combined[c].fillna(0)
            
#             NUMS.append(c)

#     train = combined.iloc[:len(train)].copy()
#     test = combined.iloc[len(train):].reset_index(drop=True).copy()

#     return test[CATS], test[NUMS]


# def get_tf_features(train, test):
#     RMV = ["ID","efs","efs_time","y","y_na","fold"]
#     FEATURES = [c for c in train.columns if not c in RMV]


#     test                             = test.replace('Not done', 'missing')
#     test                             = test.replace('Not tested', 'missing')
    
#     test['na_count']                 = test.isna().sum(axis=1)
#     test['age_karnofsky']            = test['age_at_hct'] * test['karnofsky_score']
#     test['age_comorbidity']          = test['age_at_hct'] * test['comorbidity_score']
#     test['donor_recipient_age_diff'] = abs(test['donor_age'] - test['age_at_hct'])
#     test['hla_match_ratio']          = (test['hla_high_res_8'] + test['hla_low_res_8']) / 16
#     test['age_squared']              = test['age_at_hct'] ** 2
#     test['karnofsky_squared']        = test['karnofsky_score'] ** 2
#     test['16?']                      = np.where(test['age_at_hct']<=16,1,0)
    
#     FEATURES.extend(["na_count", "age_karnofsky", "age_comorbidity", "donor_recipient_age_diff", "hla_match_ratio", "age_squared", "karnofsky_squared", "16?"])

#     CATS = []
#     for c in FEATURES:
#         if test[c].dtype=="object":
#             CATS.append(c)
#             test[c] = test[c].fillna("missing")

#     for c in FEATURES:
#         # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
#         if c in CATS:
#             #print(f"{c}, ",end="")
#             test[c],_ = test[c].factorize()
#             test[c]  -= test[c].min()
#             test[c]   = test[c].astype("int32")
#             test[c]   = test[c].astype("category")
#         else:
#             if test[c].dtype == "float64":
#                 test[c]      = test[c].astype("float32")
#             if test[c].dtype =="int64":
#                 test[c]      = test[c].astype("int32")
    
#     return test, FEATURES


# import pickle
# from tqdm import tqdm
# from sklearn.svm import SVR
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import KNNImputer

# imputer               = KNNImputer(n_neighbors=5, weights='uniform')

# scaler = StandardScaler()

# FOLDS = 10

# def inference(model_path, train, test_df):
    
#     path = model_path.split('/')[-1]
#     file = path.replace('-','_')

#     if "dt" not in model_path:
#         with open(f"/kaggle/input/{path}/{file}.pkl", 'rb') as f:
#             models = pickle.load(f)
            
#     print("All models are loaded successfully....!")

#     test_predictions = np.zeros(len(test_df))

#     for fold in tqdm(range(FOLDS)):
#         if "dt" in model_path:
#             # model = keras.models.load_model(model_path, custom_objects=custom_objects)
#             # train, test, FEATURES   = prepare_features(model_path, train, test_df)
#             # model                   = tf.keras.models.load_model(model_path+f"/model_fold_{fold}.keras", custom_objects=dt.__dict__, compile=)
#             # fold_preds              = model.predict(test.copy())
#             # fold_preds              = fold_preds.flatten()
#             pass
#         else:
#             model  = models[fold]
            
            
#         if "svr" in model_path:
#             if fold==0:
#                 train, test, FEATURES = prepare_features(model_path, train.copy(), test_df.copy())

#             # Handle missing values
#             train_imputed         = imputer.fit_transform(train[FEATURES].copy())
#             test_imputed          = imputer.transform(test[FEATURES])

#             # Convert back to DataFrame to maintain feature names
#             train_imputed         = pd.DataFrame(train_imputed, columns=FEATURES, index=train.index)
#             test_imputed          = pd.DataFrame(test_imputed, columns=FEATURES, index=test.index)
            
#             # Scale features
#             scaler.fit(train_imputed)
#             test_scaled           = scaler.transform(test_imputed)
            
#             fold_preds            = model.predict(test_scaled)

#         elif "tn" in model_path:
#             if fold == 0:
#                 train, test, FEATURES   = prepare_features(model_path, train, test_df)
                
#             fold_preds              = model.predict(test[FEATURES].values).flatten()
            
#         elif "nn" in model_path:
#             if fold == 0:
#                 X_cat, X_num      = get_nn_features(train, test_df.copy())
#             fold_preds        = model.predict([X_cat.values, X_num.values])
#             fold_preds        = fold_preds.flatten()

#         elif "tf" in model_path:
#             if fold == 0:
#                 test, FEATURES = get_tf_features(train.copy(),test_df.copy())
#                 fold_preds     = model.predict(test[FEATURES].copy())

#         elif "tabm" in model_path:
#             if fold == 0:
#                 test, FEATURES = get_tabm_features(test_df.copy())
#             fold_preds              = model.predict(test[FEATURES].copy())
            
#         else: 
#             if fold == 0:
#                 train, test, FEATURES   = prepare_features(model_path, train, test_df)
#             fold_preds              = model.predict(test[FEATURES].copy())
            
#         test_predictions += fold_preds
    
#     # Get the average predictionr
#     test_predictions /= FOLDS
        
#     return test_predictions


# train_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
# test_df  = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")


# first_best_index


# initial_test_preds = inference(first_best_index, train_df, test_df)
# initial_test_preds


# model_weights


# test_preds = []
# for model, weight in model_weights.items():
#     print(f"Using model : {model}")
#     test_df            = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
    
#     test_preds         = inference(model, train_df.copy(), test_df)
#     test_preds         = (1-weight) * rankdata(initial_test_preds) + weight * rankdata(test_preds)
#     initial_test_preds = test_preds


# train_df               = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
# test_df                = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
# test_preds             = np.zeros(len(test_df))


# preds_dict = {}
# for model, weight in model_weights.items():
#     print(f"exp name : {model}\n")
#     test_df                = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
#     test_preds             = inference(model, train_df.copy(), test_df.copy())
#     test_preds             = (1-weight) * rankdata(initial_test_preds) + weight * rankdata(test_preds)
#     initial_test_preds     = test_preds


# test_preds


# potential_ensemble                = pd.DataFrame()
# potential_ensemble["predictions"] = test_preds
# potential_ensemble["ID"]          = test_df["ID"]
# new_score                         = score(test_df[["ID","efs","efs_time","race_group"]].copy(), potential_ensemble.copy(), "ID")
# new_score


# best_score


# sub            = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
# sub.prediction = test_preds
# sub.to_csv("submission.csv",index=False)
# print("Sub shape:",sub.shape)
# sub.head()




