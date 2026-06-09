# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import os
import time
import gc
import traceback
from collections import Counter

# Data Manipulation Libraries
import numpy as np
import pandas as pd

# Machine Learning Libraries
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Visualization Libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Configuration
warnings = __import__('warnings')  # Import warnings
warnings.filterwarnings('ignore')  # Suppress warnings
np.random.seed(0)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Import the sequence and labels dataframes

train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')

#Merge the training dataframe with the labels to generate a single dataframe
train_labels['ID'] = train_labels['ID'].str.rsplit('_', n=1).str[0]
train_df  = train_labels.merge(how = 'left' , left_on = 'ID' , right_on = 'target_id' , right = train_sequences )


# Fill the Null value of coordinates with the mean coordinates of the same residue in the same structure

for col in ['x_1', 'y_1', 'z_1']:
        train_df[col] = train_df.groupby(['target_id', 'resname'])[col].transform(lambda x: x.fillna(x.mean()))


train_df


train_df.head()


#Relative Resid calculates the relative position of the residue in the sequence by dividing the 
#Residue number with the sequence length. This helps us obtain the normalized relative position of the
#reside between 0-1. Values closer to 1 denote residues closer to end of sequence and those near 0s are at the start

train_df['relative_resid'] = (train_df['resid']/train_df['sequence'].apply(len)).astype('float')


# Keep only the important columns that will be used
train_df = train_df[['sequence','x_1','y_1','z_1','resname','resid','relative_resid']] #,'chains'


train_df = train_df.dropna()


train_df


submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')
submission['ID']  = submission['ID'].str.rsplit('_' ,n =1  ).str[0]
test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
test  = test_sequences.merge(how = 'left' , left_on = 'target_id' , right_on = 'ID' , right = submission)


#Returns the relative residue id of each nucleotide, which represents the normalized position of the
#nucleotide in the sequence. Since the resid usually contains numbers 1,2,3,4... denoting positions of each
#nucleotide the relative_resid will return its relative position in the range [0-1] providing a sense of how closer 
#to beginning (near 0) or end of the sequence (near 1) the nucleotide is.

test['relative_resid'] = (test['resid']/test['sequence'].apply(len)).astype('float')


selected_cols = ['sequence','resname','ID','resid','relative_resid'] #,'chains'


test = test[selected_cols]


for col in selected_cols:
    # Handle potential NaN values before encoding, substitute them with NE
    test[col] = test[col].astype(str).fillna('NE')


test


# File paths
DATA_DIR = "/kaggle/input/stanford-rna-3d-folding/"
OUTPUT_DIR = "/kaggle/working/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Function to one hot encode a single character to a vector of 5 dimensions. Also handles unknown letters.
def one_hot_enc(char):
    end_stream = []
    if char == 'A':
        end_stream = [1, 0, 0, 0, 0]
    elif char == 'C':
        end_stream = [0, 1, 0, 0, 0]
    elif char == 'G':
        end_stream = [0, 0, 1, 0, 0]
    elif char == 'U':
        end_stream = [0, 0, 0, 1, 0]
    else:
        end_stream = [0, 0, 0, 0, 1]
    return end_stream

### Function to process the features and create processed data for training and testing
def create_test_data(dataframe, train=False):
    X_data = []; y_data = [];
    previous_nucleotide='-'; previous_seq = '-'

    '''The following loop iterates through the nuceotides of each sequence and uses both the local and global sequence
    information to create new features'''
    for index, row in dataframe.iterrows():
        
        seq = row['sequence']                     #Gets the whole sequence as a string
        nucleotide = row['resname']               #Gets the current nucleotide (A,U,G,C)
        resid = int(row['resid']);                #Gets the residue id/index.
        rel_resid = float(row['relative_resid'])  #Gets the relative residue id (given by resid/sequence_length) 
        
        '''The following condition gets the current and previous nucleotides in the sequence, 
        one hot enocode them and concates the result.Since neighboring nucleotides can be base pairs, 
        this gives an indication of potential base-pairs'''
        if seq==previous_seq:
            current_base_pair = one_hot_enc(nucleotide) + one_hot_enc(previous_nucleotide) #dataframe.iloc[index-1,dataframe.columns.get_loc('resname')]#[]
        else:
            current_base_pair = one_hot_enc(nucleotide) + one_hot_enc('-')
        previous_nucleotide = nucleotide; previous_seq = seq

        '''Gets the pairs of nucleotide in the start and end of the sequence and one_hot encodes them. The result
        are stored in variables: starting_basepair and end_basepair'''
        if(len(seq)<2):
            nucleotide_beg = seq[0]+'-'; nucleotide_end = '-'+seq[0]
        else:
            nucleotide_beg = seq[0:2]; nucleotide_end = seq[-2:]
        starting_basepair =  one_hot_enc(nucleotide_beg[0]) + one_hot_enc(nucleotide_beg[1])
        end_basepair = one_hot_enc(nucleotide_end[-2]) + one_hot_enc(nucleotide_end[-1])
        
        
        features = []
        length = len(seq)

        #Get the GC content which is essentially the sum total of the G and C nucleotides in the sequence and indicates the potentially available pairs for forming G-C base-pairs which are useful for prediction.
        gc_content = seq.count('G') + seq.count('C')
        
        #Similarly AU content, which is sum of A + U nucleotides essentially denotes the potentially available pairs that can form A-U bond/base-pair.
        au_content = seq.count('A') + seq.count('U')

        #We now add the total count of all possible two lettered pairs of numbers that are possible. 
        au_cnt = seq.count('AU'); aur_cnt = seq.count('UA')
        gc_cnt = seq.count('GC'); gcr_cnt = seq.count('CG')
        rem_counts = [length, seq.count('AC'),seq.count('CA'),seq.count('CU'),seq.count('UC'),seq.count('AG'),seq.count('GA'),seq.count('AA'),seq.count('GG'),seq.count('GU'),seq.count('UG'),seq.count('UC'),seq.count('CU'),seq.count('CC'),seq.count('UU'),seq.count('GGGA'), seq.count('GAAA'),seq.count('GAGA'),seq.count('GUGA')] #,chains,seq.count('GUGA') ,seq.count('UUCG'),seq.count('GGGA') #

        #Append all the features to X_data
        X_data.append(current_base_pair+starting_basepair+[gc_content, au_content, au_cnt, aur_cnt, gc_cnt, gcr_cnt]+rem_counts+end_basepair+[resid,rel_resid])
        
        if train:
            y_data.append([row['x_1'],row['y_1'],row['z_1']])
            
    if not train:
        return X_data
    else:
        return X_data, y_data


from sklearn.metrics import mean_squared_error, make_scorer


import optuna
## Additional code for 
#### HyperparameterOptimization
def objective2(trial):     
     params = {
         "n_estimators": trial.suggest_int("n_estimators", 100, 900),
         "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.2, log=True),
         "num_leaves": trial.suggest_int("num_leaves", 20, 300),
         "max_depth": trial.suggest_int("max_depth", 3, 25),
         "min_child_samples": trial.suggest_int("min_child_samples", 5, 150),
         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
         "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
         "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
         "random_state": 42,
         "device" : 'cpu',
         "n_jobs": -1,
     }

     # Train one regressor per coordinate
     losses = []
     #print(i)
     model = MultiOutputRegressor(xgb.XGBRegressor(verbose=-1,**params))
    
     kf = KFold(n_splits=3, shuffle=True, random_state=42)
     mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
     scores = cross_val_score(model, X_train, y_train, cv=kf, scoring=mse_scorer)
     #mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
     print(-scores)
    
     return -scores.mean()#sum(losses) / len(losses)

#study2 = optuna.create_study(direction="minimize")
#study2.optimize(objective2, n_trials=20 )

#print("Best params:", study.best_params)

'''
Best params: {'n_estimators': 430, 'max_depth': 16, 'reg_alpha': 3.143887428307033}
'''


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


test_data = create_test_data(test,train=False)


train_df


X_train, y_train = create_test_data(train_df,train=True)


len(X_train)


features=len(X_train[0])


#valx.shape


X_train = np.reshape(np.array(X_train),[-1,features])
y_train = np.reshape(np.array(y_train),[-1,3])


#trainx , valx , trainy , valy = train_test_split(X_train , y_train , random_state = 0 , test_size = 0.25)


nan_indices = np.isnan(y_train).any(axis=1)


X_train = X_train[~nan_indices]
y_train = y_train[~nan_indices]


X_test = np.reshape(np.array(test_data),[-1,features]) 


## Found after hyperparameter optimization
params={'n_estimators': 430, 'max_depth': 16, 'reg_alpha': 3.143887428307033}


import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

#Multioutput regressor predicts multiple target variables together (here x,y, and z coordinates)
multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:linear',**params))#.fit(trainx, trainy)


multioutputregressor.fit(X_train,y_train)


#multioutputregressor.estimators_[0].feature_importances_


from sklearn.metrics import mean_squared_error


#mse = mean_squared_error(trainy, multioutputregressor.predict(trainx))
mse = mean_squared_error(y_train, multioutputregressor.predict(X_train))
#mse_val = mean_squared_error(valy, multioutputregressor.predict(valx))


mse


predictions = multioutputregressor.predict(X_test)


predictions.shape


def create_submission_template_final(test_df, sample_submission_df,preds):
    """
    Creates a submission template based on test data.
    """
    # Check if sample_submission.csv is available
    if sample_submission_df is None:
        print("Sample submission file not found. Creating a new template.")
        
        # Create a new DataFrame for submission
        submission_df = pd.DataFrame()
        
        # Example code to fill the template (adjust as needed)
        ids = []
        resnames = []
        resids = []
        
        for _, row in test_df.iterrows():
            sequence = row['sequence']
            target_id = row['target_id']
        #    
            for i, nucleotide in enumerate(sequence, 1):
                ids.append(f"{target_id}_{i}")
                resnames.append(nucleotide)
                resids.append(i)
        
        submission_df['ID'] = ids
        submission_df['resname'] = resnames
        submission_df['resid'] = resids
        
        # Add coordinate columns (5 structures)
        for i in range(1, 6):
            submission_df[f'x_{i}'] = preds[:,0]
            submission_df[f'y_{i}'] = preds[:,1]
            submission_df[f'z_{i}'] = preds[:,2]
    else:
        submission_df = sample_submission_df.copy()
        print("Submission template created based on the provided example.")
    
    return submission_df


submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')


predictions.shape


submission_df = create_submission_template_final(test_sequences,None,predictions)


submission_df


def reverse_mapping(encoded):
    seq=[]
    for c in encoded:
        pos=-1
        if 1 in c:
            pos = list(c).index(1)
        if pos == 0:
            seq.append('A')
        elif pos==1:
            seq.append('C')
        elif pos==2:
            seq.append('G')
        elif pos==3:
            seq.append('U')
        elif pos==4:
            seq.append('N')
        else:
            seq.append('K')
    return seq


submission_df.to_csv('submission.csv',index=False)














































































































































































































































