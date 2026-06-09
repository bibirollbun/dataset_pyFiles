import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from joblib import Parallel, delayed

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
import lightgbm as lgb

import torch
import gc


def report_gpu():
    gc.collect()
    torch.cuda.empty_cache()
report_gpu()


def getKmers(sequence, size):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]


def kmers_rna(seq_data, size, label_data=None):
    kmers, sequence_lengths = [], []
    seq_data = seq_data.copy()

    for _, row in seq_data.iterrows(): 
        target_id, seq = row.iloc[0], row.iloc[1]
        seq_length = len(seq)
        
        if label_data is not None:
            label_matches = label_data[label_data['ID'].str.startswith(f"{target_id}_")]
            if seq_length != len(label_matches):
                continue
                
        kmers.append(getKmers(sequence=seq, size=size))
        sequence_lengths.append(seq_length)

    seq_data['seq_length'] = sequence_lengths
    seq_data['seq_kmers'] = kmers

    return seq_data


def one_hot_encode_rna(seq):
    mapping = {'A': [1, 0, 0, 0],
               'U': [0, 1, 0, 0],
               'G': [0, 0, 1, 0],
               'C': [0, 0, 0, 1]}
    return [mapping.get(nt, [0, 0, 0, 0]) for nt in seq]



def fix_dataset(seq_df, labels_df=None):
    data = {}
    start_time = time.time()  # Start timing
    
    for indx, row in tqdm(seq_df.iterrows(), total=len(seq_df), desc="Processing Sequences"):
        all_encoded = []
        seq_id, seq, seq_len = row.target_id, row.sequence, row.sequence_legnth

        # Encode RNA Sequence
        encode_seq = one_hot_encode_rna(seq)
        for pos, one_hot in enumerate(encode_seq):
            all_encoded.append([seq_id, pos] + one_hot)
        # Df of encoder
        encoded_df = pd.DataFrame(all_encoded, columns=['ID', 'seq_len', 'A', 'U', 'G', 'C'])
        
        # Convert sequence to numerical values
        numerical_seq = [seq_map.get(nuc, 4) for nuc in seq]

        # Count nucleotide occurrences using pandas' value_counts
        counts = pd.Series(list(seq)).value_counts()
        A_count = counts.get('A', 0)
        C_count = counts.get('C', 0)
        G_count = counts.get('G', 0)
        U_count = counts.get('U', 0)

        # Create DataFrame efficiently
        encoded_df['RNA_seq'] = numerical_seq
        encoded_df['seq_id'] = [f"{seq_id}_{i}" for i in range(1, seq_len+1)]
        encoded_df['A_count'] = A_count
        encoded_df['C_count'] = C_count
        encoded_df['G_count'] = G_count
        encoded_df['U_count'] = U_count

               # Check for labels if they exist
        if labels_df is not None:
            seq_id_df = labels_df[labels_df['ID'].str.startswith(f"{seq_id}_")]

            if not seq_id_df.empty:
                coords = seq_id_df[['x_1', 'y_1', 'z_1']].to_numpy()
                if coords.shape[0] == seq_len:
                    # Normalize coordinates efficiently
                    mean = coords.mean(axis=0)
                    std = coords.std(axis=0) + 1e-8  # Avoid division by zero
                    coords_norm = (coords - mean) / std
                    encoded_df[['x_1', 'y_1', 'z_1']] = coords_norm
                else:
                    print(f"Warning: Mismatch for {seq_id} - coords: {coords.shape[0]}, seq_len: {seq_len}")
                    continue  
    
        data[seq_id] = encoded_df
        
    # Merge all data after loop
    merge_data = pd.concat(data.values(), ignore_index=True)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    return merge_data



def clean_data(data, pop_col, remove_col, std_cols): 
    
    print(f'BEFORE Data:{data.shape}')
    # Remove missing values
    data = data.dropna().copy()  

    # Remove and store the ID column efficiently
    id_col = data.pop(pop_col) 
    data = data.drop(columns=remove_col)
     
    # StandardScaler
    data[std_cols] = StandardScaler().fit_transform(data[std_cols])
    print(f'AFTER Data:{data.shape}')

    return data, id_col


def train_models(df_train, df_val, target_data, models):
    start_time = time.time()

    
    df_merged = pd.concat([df_train, df_val], ignore_index=True)
    print(f'merge Train Val:{df_merged.shape}')
    
    X_train = df_merged.drop(columns=target_data)
    y_train = df_merged[target_data]
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    
    def train_model(name, model):
        model.fit(X_train, y_train)
        return (name, model)
    
    trained_models = Parallel(n_jobs=-1)( delayed(train_model)(name, model) for name, model in models.items())
    
    for name, model in trained_models:
        print(f"{name} model trained")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")
    
    return trained_models


def predictions(models, unseen_target): 
    # Submission dataframe
    sample_sub = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')
    id_s, resname_seq, resid_seq = sample_sub.ID, sample_sub.resname, sample_sub.resid
    submission = pd.DataFrame({'ID':id_s,
                              'resname':resname_seq, 
                              'resid':resid_seq})

    #make the predictions 
    predictions = {}
    for (name, model), trgt_lst in zip(models, unseen_target): 
        y_pred = model.predict(clean_test_data) # lst [x,y,z]
        predictions[name] = pd.DataFrame(y_pred, columns=trgt_lst)
        print(f'{name} model predict')
    
    # Final submision Df
    final_df = submission.copy()
    for df in predictions.values():
        final_df = final_df.merge(df, left_index=True, right_index=True, how='left') 
    print('... Submision ready')
    
    return  final_df


train_deq_path  = '/kaggle/input/stanford-rna-3d-folding/train_sequences.csv'
train_label_path ='/kaggle/input/stanford-rna-3d-folding/train_labels.csv' 

val_seq_path ='/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv'
val_label_path ='/kaggle/input/stanford-rna-3d-folding/validation_labels.csv' 


test_seq_path ='/kaggle/input/stanford-rna-3d-folding/test_sequences.csv' 


used_columns = ['target_id', 'sequence']
seq_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
std_cols = ['seq_len', 'A_count', 'C_count', 'G_count', 'U_count']
remove_col, pop_col = ['ID'], 'seq_id'
target_coord = ['x_1', 'y_1', 'z_1']


unseen_target_name =[['x_1', 'y_1', 'z_1'], ['x_2', 'y_2', 'z_2'], ['x_3', 'y_3', 'z_3'],
             ['x_4', 'y_4', 'z_4'], ['x_5', 'y_5', 'z_5'] ]


# LGBMRegressor (x1, y1, z1)
lgbm = lgb.LGBMRegressor(n_estimators=200,
                              learning_rate=1,
                              max_depth=-1,
                              random_state=42)
lgbm_model = RegressorChain(lgbm)

# GBDTRegression (x2, y2, z2)
gbdtr = GradientBoostingRegressor(n_estimators=20, 
                                learning_rate=0.1, 
                                max_depth=50, 
                                random_state=42)
gbdtr_model = RegressorChain(gbdtr)

# ExtraTreeRegression (x3, y3, z3)
extree = ExtraTreesRegressor(n_estimators=10,
                                 max_depth=50, 
                                 criterion='friedman_mse',
                                 random_state=42)
extree_model = RegressorChain(extree)

# RandomForest Regression (x4, y4, z4)
rf_r = RandomForestRegressor(criterion='friedman_mse',
                                n_estimators=10, 
                                max_depth=80, 
                                bootstrap=True,
                                random_state=42)
rf_r_model = RegressorChain(rf_r)

# # RadiusNeighborsRegressor (x5, y5, z5)
# knn_r = RadiusNeighborsRegressor(radius= 1.0,
#                          weights=  'distance', #'uniform', 'distance'
#                          metric= 'minkowski', #'euclidean', manhattan, minkowski
#                          algorithm = 'auto',  #kd_tree, ball_tree
#                               p=1)
# knn_r_model = RegressorChain(knn_r)

# XGB regressor (x5, y5, z5)
xgb_r = XGBRegressor(n_estimators=80, 
                     learning_rate=1, 
                     max_depth=30, 
                     random_state=42)

xgb_r_model = RegressorChain(xgb_r)



models = {
    'LGBMRegressor': lgbm_model,
    'GradientBoostingRegressor': gbdtr_model,
    'ExtraTreesRegressor': extree_model,
    'RandomForestRegressor': rf_r_model,
    'XGBRegressor': xgb_r_model
}


#train data
train_sequences = pd.read_csv(train_deq_path, usecols=used_columns)
train_labels = pd.read_csv(train_label_path)

# Val Data
validation_sequences = pd.read_csv(val_seq_path, usecols=used_columns)
validation_labels = pd.read_csv(val_label_path)

# Test data
test_sequences = pd.read_csv(test_seq_path ,usecols=used_columns)


# rna_train_seq = kmers_rna(seq_data=train_sequences, size=3, label_data=train_labels)
# rna_val_seq = kmers_rna(seq_data=validation_sequences, size=3, label_data=validation_labels)
# rna_test_seq = kmers_rna(seq_data=test_sequences, size=3, label_data=None)


train_sequences['sequence_legnth'] = train_sequences['sequence'].str.len()
validation_sequences['sequence_legnth'] = validation_sequences['sequence'].str.len()
test_sequences['sequence_legnth'] = test_sequences['sequence'].str.len()


print('.... Train Data Procesed')
train_data  = fix_dataset(seq_df=train_sequences, labels_df=train_labels)

print('.... Validation Data Procesed')
val_data  = fix_dataset(seq_df=validation_sequences, labels_df=validation_labels)

print('.... Test Data Procesed')
test_data  = fix_dataset(seq_df=test_sequences, labels_df=None)


 print('... Clean Train Data')
clean_train_data, _ = clean_data(data = train_data, pop_col = pop_col,
                              remove_col = remove_col, std_cols =  std_cols)
print('... Clean Validation Data')
clean_val_data, _ = clean_data(data = val_data, pop_col = pop_col,
                              remove_col = remove_col, std_cols =  std_cols)

print('... Clean Test Data')
clean_test_data, unseen_id = clean_data(data = test_data, pop_col = pop_col,
                              remove_col = remove_col, std_cols =  std_cols)


trained_models = train_models(df_train=clean_train_data, 
                              df_val=clean_val_data,
                              target_data =target_coord, 
                              models=models) #  


submission = predictions(models= trained_models, 
                         unseen_target=unseen_target_name)


submission.to_csv('submission.csv', index=False)

