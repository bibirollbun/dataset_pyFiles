# upgrade sklearn
!pip install -U --q scikit-learn
#!pip install -U --q skorch


#
# Libraries
#

# General
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, string, re, random, gc, pickle, math,warnings
import json, glob
from itertools import *
from datetime import date
from tqdm.keras import TqdmCallback
from tqdm import tqdm
import copy


# Sklearn
import sklearn
from sklearn.model_selection import *
from sklearn.feature_extraction import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.utils import *
from sklearn.pipeline import *
from sklearn.compose import *
from sklearn.base import BaseEstimator, ClassifierMixin


# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset


# Setting
pd.set_option('max_colwidth',None)
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

data_path = []

for dirname, _, filenames in os.walk('/kaggle/input/playground-series-s5e7'):
    for filename in filenames:
        if filename.endswith('csv'):
            data_path.append(os.path.join(dirname, filename))


print(f"~~~~ Using Sklearn {sklearn.__version__} ~~~~")


#
# Config
#

class Config:
    SEED = 937
    TRAIN_FILE = data_path[1]
    TEST_FILE = data_path[2]
    SUB_FILE = data_path[0]
    SPLIT_SIZE = 0.2
    INPUT = 9
    UNITS = [36, 18, 9]
    #UNITS = [64, 32, 16]
    #UNITS = [32, 16, 8]
    EPOCH = 3000
    BATCH_SIZE = 32
    LEARNING_RATE = 0.002
    QUANT = 0.75
    CV = 0.3

config = Config()


#
# Custom Function -- Imputation
#

def impute(df):
    """
    Impute numerical columns in a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with imputed numerical columns.
    """
    
    impute_cols = []
    
    # find & extract columns with NaN
    res = df.isna().sum()

    for i, j in zip(res,list(df.columns)):
        if i > 0:  # no. of rows with NaN > 0
            impute_cols.append(j)
        else:
            pass

    print(f"Found \" {', '.join(impute_cols)} \" columns in the dataset \n")
    
    # loop through each column & impute based on the column type 
    for c in impute_cols:
        # check if its numerical
        if df[c].dtype == int or df[c].dtype == float:
            qnt = df[c].quantile(config.QUANT)   # 75% quantile
            df[[c]] = df[[c]].fillna(value=qnt,axis=1)
        # check if its object
        else:
            df[c].fillna(df[c].mode()[0], inplace=True)
       
    print(f"** Imputation Completed **\n")
    return df


#
# Custom Class -- Encoding
#

class CustomEncoder:
    """
    Feature & Target column encoding.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with encoded columns.
    """    
    
    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns
        self.feature_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore',drop=None,dtype=np.int64)
        self.target_encoder = LabelEncoder()
        self.feature_columns = None

    def fit_transform_features(self, X: pd.DataFrame):
        # Separate categorical and numerical columns
        X_cat = X[self.categorical_columns]
        X_num = X.drop(columns=self.categorical_columns)

        # Fit and transform categorical features
        encoded_cat = self.feature_encoder.fit_transform(X_cat)
        encoded_cat_df = pd.DataFrame(
            encoded_cat,
            columns=self.feature_encoder.get_feature_names_out(self.categorical_columns),
            index=X.index
        )

        # Combine numerical and encoded categorical features
        combined_df = pd.concat([X_num, encoded_cat_df], axis=1)
        return combined_df

    def fit_transform_target(self, y: pd.Series):
        return self.target_encoder.fit_transform(y)

    def inverse_transform_target(self, y_encoded):
        return self.target_encoder.inverse_transform(y_encoded)


#
# Custom Class -- Run PyTorch Model
#

class MyPyTModel(BaseEstimator, ClassifierMixin):
    """
    Wrapper to make PyTorch model compatible with Sklearn.

    Args:
        BaseEstimator: Base estimator / model.
        Classifier: Mixin class for all classifiers in scikit-learn
    """  
        
    def __init__(self, input_size=None, hidden_sizes=None, epochs=20, lr=0.001,verbose=True):
        self.__estimator_type__ = "classifier"
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [36, 18, 9]
        self.input_size = input_size if input_size is not None else 9
        self.verbose = verbose
        self.epochs = epochs
        self.lr = lr
        self.model_ = None

    # function to fit the tensors directly
    def fit_tensor(self, X, y,X_val=None,y_val=None):
        # instantiate model
        self.model_ = BinaryClassifier(self.input_size, self.hidden_sizes)
        # loss function
        criterion = nn.BCELoss()
        # optimiser
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        # run epochs
        for epoch in tqdm(range(self.epochs), desc="Training Progress"):
            self.model_.train()
            optimizer.zero_grad()
            outputs = self.model_(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # display loss & accuracy for epoch
        if self.verbose:
            preds = (outputs >= 0.5).float()
            acc = accuracy_score(y.numpy(), preds.numpy())
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss.item():.4f} - Training Accuracy: {acc:.4f}")

        # evaluate on validation dataset
        if X_val is not None and y_val is not None:
            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(X_val)
                val_preds = (val_outputs >= 0.5).float().numpy()
                val_acc = accuracy_score(y_val, val_preds)
                return val_acc
            
        return None  # nothing to return

    # function to make predictions on test-data
    def predict(self, X_test, threshold=0.5):
        self.model_.eval()
        with torch.no_grad():
            #inputs = torch.tensor(X_test, dtype=torch.float32)
            outputs = self.model_(X_test)
            predictions = (outputs >= threshold).int().squeeze()
        return predictions.cpu().numpy()


#
# Data
#

train = pd.read_csv(config.TRAIN_FILE,index_col='id')
test = pd.read_csv(config.TEST_FILE,index_col='id')
sub = pd.read_csv(config.SUB_FILE)

# view / stats
print(f"Training size: {train.shape} | Test size: {test.shape}\n")
train.head()


#
# Preprocessing - Impuation
#

train_df = impute(train)
test_df = impute(test)

# view
train_df.head()


#
# Preprocessing - Feature Engineering & Train-Test Split
# 

# feature engineering
x = train_df.loc[:, train_df.columns != 'Personality']
y = train_df[['Personality']]

# train-val split
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=config.SPLIT_SIZE,random_state=config.SEED,stratify=y,)

# view
print(f"Train size: {x_train.shape} | Validation size: {x_val.shape}")


#
# Preprocessing - Encoding
#

# feature categorical columns
feat_cat_cols = ['Stage_fear','Drained_after_socializing']

# define encoder
encoder = CustomEncoder(categorical_columns=feat_cat_cols)

# # encode features & target (prior to split)
# x_enc = encoder.fit_transform_features(x)
# y_enc = encoder.fit_transform_target(y)

# encode features (training & validation)
x_train_enc = encoder.fit_transform_features(x_train)
x_val_enc = encoder.fit_transform_features(x_val)

# Encode target (training & validation)
y_train_enc = encoder.fit_transform_target(y_train)
y_val_enc = encoder.fit_transform_target(y_val)

# Encode (test)
test_df_enc = encoder.fit_transform_features(test_df)


#
# Preprocessing - PyTorch Tensors
#

# training tensors
x_train_tensor = torch.tensor(x_train_enc.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_enc, dtype=torch.float32).reshape(-1, 1)

# validation tensors
x_val_tensor = torch.tensor(x_val_enc.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_enc, dtype=torch.float32).reshape(-1, 1)

# test tensors
x_test_tensor = torch.tensor(test_df_enc.values, dtype=torch.float32)

# view
print(f"Training feature tensor size: {x_train_tensor.size()} | Training target tensor size: {y_train_tensor.size()}")
print(f"Testing feature tensor size: {x_test_tensor.size()}")


#
# Build PyTorch Model - Binary Classification
#

# reassign input size
Config.INPUT = x_train_tensor.size()[1]
hidden_sizes = config.UNITS

# # classifier class
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.output = nn.Linear(hidden_sizes[2], 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.output(x))
        return x


# # summary
summary(BinaryClassifier(input_size=Config.INPUT, hidden_sizes=hidden_sizes))


#
# Model Training & Evaluation (base)
#

# base model
base_clf_model = MyPyTModel(input_size=x_train_tensor.shape[1], hidden_sizes=config.UNITS, epochs=config.EPOCH, lr=config.LEARNING_RATE)

# fit model
base_acc = '{:.3%}'.format( base_clf_model.fit_tensor(x_train_tensor, y_train_tensor, X_val=x_val_tensor, y_val=y_val_tensor ) )
print(f"\nBase Accuracy on Validation Data: {(base_acc)}")


#
# Submission File
#

# prediction on test-data
y_test_pred = base_clf_model.predict(x_test_tensor)

# label prediction on test-data
y_test_labels = encoder.inverse_transform_target(y_test_pred)

# submission columns
sub_cols = list(sub.columns)

# create submission dataframe
submission = pd.DataFrame(list(zip(list(test_df_enc.index), list(y_test_labels))),
                           columns=sub_cols)

# export to csv
submission.to_csv('submission.csv', index=False)

# view
submission.head()

