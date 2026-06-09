# Directing all the calls to Pandas to NVIDIA cuDF on GPU
%load_ext cudf.pandas


# ==== General =====

import pandas as pd
import numpy as np
import copy
from itertools import combinations

# ==== Visualization =====

import matplotlib.pyplot as plt
import seaborn as sns

# ==== Stats =====

from scipy.stats import chi2_contingency
import itertools
import scipy.stats as ss

# ==== ML =====

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

import xgboost as xgb
from cuml.preprocessing import TargetEncoder

from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F


# Train dataset
train_data = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
print("Number of records: %d | Number of features: %d" % (train_data.shape[0], train_data.shape[1]))
print("\n")
train_data.set_index('id', inplace=True)
train_data.head()


original_data = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", delimiter=';')
# Map target variable to boolean
original_data['y'] = original_data.y.map({'yes':1, 'no':0})
print("Number of records: %d | Number of features: %d" % (original_data.shape[0], original_data.shape[1]))
print("\n")
original_data.head()


test_data = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
# Set y field to -1 
test_data['y'] = -1
print("Number of records: %d | Number of features: %d" % (test_data.shape[0], test_data.shape[1]))
print("\n")
test_data.set_index('id', inplace=True)
test_data.head()


combined_data = pd.concat([train_data, test_data, original_data], axis=0)
combined_data


def detect_cat_num(df):
    """ Detecting categorical and numerical features """
    # Save categorical and numerical columns
    CATS = []
    NUMS = []
    
    for col in df.columns[:-1]: # Except y column
        t = 'CAT'
        if df[col].dtype == 'object':
            CATS.append(col)
        else:
            NUMS.append(col)
            t = 'NUM'
        n = df[col].nunique() # Count of unique values
        na = df[col].isna().sum() # Total of null values
        print(f"[{t}] Feature {col} has {n} unique values | {na} null values")    
    
    print("\nCategory Features: ", CATS)
    print("Numerical Features: ", NUMS)  

    return CATS, NUMS
    
def num_2_cat_feat(df, CATS, NUMS):
    """ Duplicates and Converts numerical features to categorical, and applies label encoding to categorical features
    """
    CATS1 = []
    SIZES = {}
    for c in NUMS + CATS: # All columns
        n = c
        if c in NUMS: # If the feat is numerical, duplicated and append to new categorical feats
            n = f"{c}2"
            CATS1.append(n)
        df[n],_ = df[c].factorize() # If not numerical do label encoding
        SIZES[n] = df[n].max()+1 # Cardinality
    
        df[c] = df[c].astype('int32') 
        df[n] = df[n].astype('int32')
    
    print("New CATS:", CATS1 )
    print("Cardinality of all CATS:", SIZES )
    return df, CATS1, SIZES

def new_combination_feat(df, CATS, CATS1, SIZES):
    """ Creating new fetures based on the interaction between two existing categories """
    pairs = combinations(CATS + CATS1, 2)
    new_cols = {}
    CATS2 = []
    
    for c1, c2 in pairs:
        name = "_".join(sorted((c1, c2)))
        new_cols[name] = df[c1] * SIZES[c2] + df[c2] # ensures that each unique combination of (c1, c2) gets a unique integer.
        CATS2.append(name)
    if new_cols:
        new_df = pd.DataFrame(new_cols)         
        df = pd.concat([df, new_df], axis=1) 
    
    print(f"Created {len(CATS2)} new CAT columns")
    return df, CATS2

def frequency_feature(df, CATS, CATS1, CATS2):
    """ Create new features representing the frequency of each unique value within the column """
    CE = []
    CC = CATS+CATS1+CATS2
    
    print(f"Processing {len(CC)} columns... ",end="")
    for i,c in enumerate(CC):
        if i%10==0: print(f"{i}, ",end="")
        tmp = df.groupby(c).y.count() # Count 
        tmp = tmp.astype('int32')
        tmp.name = f"CE_{c}"
        CE.append( f"CE_{c}" )
        df = df.merge(tmp, on=c, how='left')
    print()
    return df, CE


def combined_feat_pipeline(df):
    """ Pipeline for creating new features """
    # Detecting numerical, and categorical features
    CATS, NUMS = detect_cat_num(df)
    # Get numerical -> categorical features
    df, CATS1, SIZES = num_2_cat_feat(df, CATS, NUMS)
    # Get interaction features
    df, CATS2 = new_combination_feat(df, CATS, CATS1, SIZES)
    # Get frequency-based features
    df, CE = frequency_feature(df, CATS, CATS1, CATS2)
    print("============== NEW FEATURES ADDED ==================")
    return df, NUMS, CATS, CATS1, CATS2, CE


# Get the new features, and the lists of the new features
combined_data, NUMS, CATS, CATS1, CATS2, CE = combined_feat_pipeline(combined_data)


# Split the combined data with the new features into train, test, and original data
train = combined_data.iloc[:len(train_data)]
test = combined_data.iloc[len(train_data):len(train_data)+len(test_data)]
original = combined_data.iloc[-len(original_data):]
# Delete for free space
del combined_data
print("Train shape", train.shape,"Test shape", test.shape,"Original shape", original.shape )


def age_related_feat(df):
    """ Age-related features """

    # Split age into groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 20, 40, 60, max(df['age'])], labels=['young', 'adult', 'adult-senior', 'senior'])
    # Identify whether a person is a senior
    df['is_senior'] = (df['age'] > 60).astype(int)
    # Capture square pattern
    df['age_squared'] = df['age'] ** 2

    return df

def job_socioeconomic_feat(df):
    """ Job & socioeconomics features """

    # --- Job grouping ---
    white_collar = ["admin.", "technician", "management"]
    blue_collar = ["blue-collar", "services", "housemaid"]
    unstable = ["student", "unemployed"]
    others = ["entrepreneur", "self-employed", "retired", "unknown"]

    # Map of jobs
    def map_job(job):
        if job in white_collar:
            return "white_collar"
        elif job in blue_collar:
            return "blue_collar"
        elif job in unstable:
            return "unstable"
        else:
            return "other"

    df['job_group'] = df['job'].apply(map_job) # Apply map of jobs to the dataframe

    # --- Employment stability (binary: stable vs unstable) ---
    df["job_stable"] = df["job_group"].isin(["white_collar", "blue_collar"]).astype(int)

    # --- Interaction: job × education ---
    df["job_edu_interaction"] = df["job_group"] + "_" + df["education"].astype(str)

    # --- Interaction: marital × job ---
    
    df["marital_job"] = df["marital"].astype(str) + "_" + df["job_group"]

    return df

def financial_feat(df):
    """ Financial features """

    # Split balance into groups
    df["balance_category"] = pd.cut(df["balance"], bins=[min(df['balance']), 0, np.mean(df['balance']), max(df['balance'])],
                                    labels=["low", "medium", "high"])
    df["balance_category"] = df["balance_category"].astype('object')
    # Identify whether the person has positive balance
    df["has_positive_balance"] = (df["balance"] > 0).astype(int)
    # Identify whether the person has two types of loans
    df["loan_burden"] = df[["housing", "loan"]].apply(lambda x: (x == "yes").sum(), axis=1)
    # Relationship between balance and age
    df["balance_to_age"] = df["balance"] / (df["age"] + 1)

    return df

def timing_feat(df):
    """ Timing fetures """
    # Identify day pattern
    df["is_end_of_month"] = (df["day"] >= 25).astype(int)
    # Client recently contacted < 30 days (with constraint)
    df["recently_contacted"] = ((df["pdays"] != -1) & (df["pdays"] < 30)).astype(int)
    # Rate of campaign intensity based on previous contact 
    df["campaign_intensity"] = df["campaign"] / (df["previous"] + 1)
    # Normalize the duration of the campaign
    df["log_duration"] = np.log1p(df["duration"])

    return df

def p_campaign_behaviour(df):
    """ Capture campaign behaviour """
    # Identify whether the client was contact previously without constraint 
    df["previously_contacted"] = (df["pdays"] != -1).astype(int)
    # Capture the success cases
    df["previous_success"] = (df["poutcome"] == "success").astype(int)
    # Ratio of the previous contact vs current campaign contacts
    df["prev_contact_ratio"] = df["previous"] / (df["campaign"] + 1)

    return df

def stat_feat(df):
    """ Statistical features """

    # Risk score based on the number of loans
    df["risk_score"] = (
        (df["default"] == "yes").astype(int) +
        (df["housing"] == "yes").astype(int) +
        (df["loan"] == "yes").astype(int)
    )
    # Contact efficiency based on the duration of the campaign
    df["contact_efficiency"] = df["duration"] / (df["campaign"] + 1)
    # Split the duration into groups
    df["duration_bucket"] = pd.cut(df["duration"], bins=[min(df['duration']), 1000, 2000, max(df['duration'])],
                                   labels=["short", "medium", "long"])

    df["duration_bucket"] = df["duration_bucket"].astype('object')
    return df





def feat_eng_pipe(df):
    """ Pipeline for creating new features """
    
    df_new = df.copy()
    # Get age-related features
    df_new = age_related_feat(df_new)
    # Get job socioeconomics features
    df_new = job_socioeconomic_feat(df_new)
    # Get financial features
    df_new = financial_feat(df_new)
    # Get timing features
    df_new = timing_feat(df_new)
    # Get campaign behaviour features
    df_new = p_campaign_behaviour(df_new)
    # Get stats features
    df_new = stat_feat(df_new)

    return df_new

# Combine the inicial datasets
combined_data = pd.concat([train_data, test_data, original_data], axis=0)
# New data with new features
CATS3_DF = feat_eng_pipe(combined_data[CATS+NUMS])
# Filter the dataset to retain only the new features
CATS3_DF = CATS3_DF.iloc[:, 17:]
# Name of the new features
CATS3 = list(CATS3_DF.columns)
display(CATS3_DF.iloc[:, :11].head())
print('\n')
display(CATS3_DF.iloc[:, 11:21].head())
print(f"ADDED {len(CATS3)} NEW FEATURES ")


# Create a copy of the new features dataset
encoded_data = CATS3_DF.copy()
encoded_data


# Get the category columns
category_columns = [encoded_data[obj].name for obj in encoded_data if encoded_data[obj].dtype == 'object']

# Perform a label encoder to the categories
le = LabelEncoder()
for col in category_columns:
    encoded_data.loc[:, col] = le.fit_transform(encoded_data.loc[:, col])
encoded_data


# Split the combined data with the new features into train, test, and original data
train1 = encoded_data.iloc[:len(train_data)]
train1 = pd.concat([train1, train_data['y']], axis=1)
test1 = encoded_data.iloc[len(train_data):len(train_data)+len(test_data)]
original1 = encoded_data.iloc[-len(original_data):]
original1 = pd.concat([original1, original_data['y']], axis=1)

print("Train shape", train1.shape,"Test shape", test1.shape,"Original shape", original1.shape )


FOLDS = 7 # Folds for CV
SEED = 42 # Set a seed for reproducibility

# XGB Hyperparameters
params = {
    "objective": "binary:logistic",  
    "eval_metric": "auc",           
    "learning_rate": 0.1,
    "max_depth": 0,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "seed": SEED,
    "device": "cuda",
    "grow_policy": "lossguide", 
    "max_leaves": 32,          
    "alpha": 2.0,
}


# Set K fold approach for CV
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
# Set standarization method
sc = preprocessing.MinMaxScaler()

# Out of Fold predictions
test_preds_1 = np.zeros(len(test1)) # For test dataset
oof_preds_1 = np.zeros(len(train1)) # For out of fold predictions

for n, (train_idx, val_idx) in enumerate(kf.split(train1)):
    print(f" ==== Fold number: {n} ==== ")

    ### Get the data
    # Adding the original dataset as row
    X_train = pd.concat([train1.iloc[train_idx, :], original1], axis=0, ignore_index=True)
    # Extract the y target variable from training set
    y_train = X_train.loc[:, 'y']
    # Drop the y column
    X_train = X_train.drop('y', axis=1)
    # Copy the test set
    X_test = test1.copy()
    # Get the validation data
    X_val = train1.iloc[val_idx, :].drop('y', axis=1)
    y_val = train1.loc[val_idx, 'y']
    
    ######  Scaling
    
    X_train = pd.DataFrame(
        sc.fit_transform(X_train),
        columns=X_train.columns,   # keep same column names
        index=X_train.index        # keep row indices
    )
    
    X_val = pd.DataFrame(
        sc.transform(X_val),
        columns=X_val.columns,   # keep same column names
        index=X_val.index        # keep row indices
    )

    X_test = pd.DataFrame(
        sc.transform(X_test),
        columns=X_test.columns,   # keep same column names
        index=X_test.index        # keep row indices
    )    

    #####  Converting to DMatrix format
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_eval = xgb.DMatrix(X_val, label=y_val)
    dtest  = xgb.DMatrix(X_test, enable_categorical=True)

    #####  Training the XGB model
    model_1 = xgb.train(
        params=params,
        dtrain=d_train,
        num_boost_round=10_000,
        evals=[(d_train, "train"), (d_eval, "valid")],
        early_stopping_rounds=200,
        verbose_eval=200
    )

    oof_preds_1[val_idx] = model_1.predict(d_eval, iteration_range=(0, model_1.best_iteration + 1)) # Save the oof predictions
    test_preds_1 += model_1.predict(dtest, iteration_range=(0, model_1.best_iteration + 1)) / FOLDS # Applying the mean of the test predictions


# XGB model performance based on ROC AUC score
m = roc_auc_score(train1.y, oof_preds_1)
print(f"XGB with Original Data as columns CV = {m}")


# Get all the featues
FEATURES = NUMS+CATS+CATS1+CATS2+CE
print(f"We have {len(FEATURES)} features.")


class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=256*1024):
        self.features = features
        self.target = target
        self.df = df
        self.it = 0 
        self.batch_size = batch_size
        self.batches = int( np.ceil( len(df) / self.batch_size ) )
        super().__init__()

    def reset(self):
        '''Reset the iterator'''
        self.it = 0

    def next(self, input_data):
        '''Yield next batch of data.'''
        if self.it == self.batches:
            return 0 # Return 0 when there's no more batch.
        
        a = self.it * self.batch_size
        b = min( (self.it + 1) * self.batch_size, len(self.df) )
        #dt = cudf.DataFrame(self.df.iloc[a:b])
        dt = self.df.iloc[a:b]
        input_data(data=dt[self.features], label=dt[self.target]) 
        self.it += 1
        return 1


oof_preds = np.zeros(len(train)) # For out of fold predictions
test_preds = np.zeros(len(test)) # For test dataset

# Out of Fold predictions
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {fold+1} ###")
    print("#"*25)

    #### Get the data
    # X + y train set and original set
    Xy_train = train.iloc[train_idx][ FEATURES+['y'] ].copy()
    Xy_more = original[ FEATURES+['y'] ]
    for k in range(1):
        # Add original data as new rows
        Xy_train = pd.concat([Xy_train,Xy_more],axis=0,ignore_index=True)
    # Get validation data
    X_valid = train.iloc[val_idx][FEATURES].copy()
    y_valid = train.iloc[val_idx]['y']

    # Get test data
    X_test = test[FEATURES].copy()

    ##### Perform category encoding using Target Encoder
    CC = CATS1+CATS2
    print(f"Target encoding {len(CC)} features... ",end="")
    for i,c in enumerate(CC):
        if i%10==0: print(f"{i}, ",end="")
        TE0 = TargetEncoder(n_folds=10, smooth=0, split_method='random', stat='mean')
        Xy_train[c] = TE0.fit_transform(Xy_train[c],Xy_train['y']).astype('float32')
        X_valid[c] = TE0.transform(X_valid[c]).astype('float32')
        X_test[c] = TE0.transform(X_test[c]).astype('float32')
    print()

    Xy_train[CATS] = Xy_train[CATS].astype('category')
    X_valid[CATS] = X_valid[CATS].astype('category')
    X_test[CATS] = X_test[CATS].astype('category')

    # Convert datasets to DMatrix format
    Xy_train = IterLoadForDMatrix(Xy_train, FEATURES, 'y')
    dtrain = xgb.QuantileDMatrix(Xy_train, enable_categorical=True, max_bin=256)
    dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    dtest  = xgb.DMatrix(X_test, enable_categorical=True)
    
    ##### Training XGB model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=10_000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=200,
        verbose_eval=200
    )

    ##### Prediction of oof and test predictions
    oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / FOLDS


# XGB model performance based on ROC AUC score
m = roc_auc_score(train.y, oof_preds)
print(f"XGB with Original Data as columns CV = {m}")


# Correlation between the predictions
corr = np.corrcoef(oof_preds, oof_preds_1)[0,1]
print(f"Correlation between models: {corr:.4f}")


# Get the input for the Weighted Model
X = pd.concat([pd.DataFrame(oof_preds), pd.DataFrame(oof_preds_1)], axis=1)
X.columns = [0,1]
X


# Convert pandas to torch tensors
X_tensor = torch.tensor(X[:700000].values, dtype=torch.float32)
y_tensor = torch.tensor(train_data.loc[:699999, 'y'].values, dtype=torch.float32)
# Create dataset and dataloader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


class WeightedModel(nn.Module):
    def __init__(self):
        super(WeightedModel, self).__init__()
        # Learnable weights for each OOF input
        self.weights = nn.Parameter(torch.ones(2))  # [w1, w2]
    
    def forward(self, x):
        
        # Normalize weights so they sum to 1
        w = F.softmax(self.weights, dim=0)  # ensures w1+w2=1 and w>=0
        
        # Weighted average
        combined = torch.sum(x * w, dim=1, keepdim=True)
        
        # Output is probability already (since oof are probabilities)
        return combined


# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create the model
weighted_model = WeightedModel().to(device)
# Loss function
criterion = nn.BCELoss()
# Optimizer
optimizer = torch.optim.Adam(weighted_model.parameters(), lr=0.01)
# Best loss for save the best model
best_loss = np.inf

# Training loop
for epoch in range(10):
    print(f"Epoch: {epoch}")
    for X_batch, y_batch in dataloader:
        # Prepare for update
        optimizer.zero_grad()
        # Compute predictions
        preds = weighted_model(X_batch)
        # Compute loss
        loss = criterion(preds, y_batch.unsqueeze(1))  # make y shape (batch,1)
        # Save only the best model
        if loss < best_loss:
            best_loss = loss
            torch.save(weighted_model.state_dict(), 'best_weighted_model.pth')
        # Propagate the loss
        loss.backward()
        # Update weights
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# Load best model
weighted_model.load_state_dict(torch.load('best_weighted_model.pth', weights_only=True))


# Validate the results
X_val = torch.tensor(X[700000:].values, dtype=torch.float32).to(device)
# Set the model to validation mode
with torch.no_grad():
    # Compute predictions
    preds = weighted_model(X_val).squeeze().cpu().numpy()

final_w = F.softmax(weighted_model.weights, dim=0).detach().cpu().numpy()
print("Learned weights:", final_w)
# Get weigths
w1, w2 = final_w


# Compute the final predictions
final_oof_preds = w1*oof_preds + w2*oof_preds_1


# Performance with the weighted approach
m = roc_auc_score(train.y, final_oof_preds)
print(f"XGB with Original Data as columns CV = {m}")


# Compute the final predictions
final_test_preds = w1*test_preds + w2*test_preds_1


sub = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
sub['y'] = final_test_preds
sub.to_csv("submission.csv",index=False)
print('Submission shape',sub.shape)
sub.head()

