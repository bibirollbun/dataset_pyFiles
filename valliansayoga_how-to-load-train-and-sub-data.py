import numpy as np
import pandas as pd
import os
import warnings


warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_path = "/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/transfer_plate.csv"
train = pd.read_csv(train_path)
train.head()


target = train.iloc[:, -3:].copy()
target = target.dropna() # There are NaNs to accomodate the rows of N samples * 2 arrays
target


X = train.iloc[:, :-4].copy()
column_names = ["sample_id"] + [i for i in range(X.shape[1]-1)]

X.columns = column_names
X.head()


X.sample_id = X.sample_id.ffill()
X.sample_id.head()


X.sample_id = X.sample_id.str.strip()
X.sample_id.head()


col_to_cast = X.select_dtypes(object).columns[1:]
    
for col in col_to_cast:
    X[col] = X[col].str.replace("[\[\]]", "", regex=True).astype("int64")

X.select_dtypes(object).head()  # Should only resulting in sample_id name


def load_train():
    df = pd.read_csv("/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/transfer_plate.csv")
    X = df.iloc[:, :-4]
    y = df.iloc[:, -3:].dropna()

    X.columns = ["sample_id"] + [i for i in range(X.shape[1]-1)]
    X.sample_id = X.sample_id.ffill().str.strip()

    col_to_cast = X.select_dtypes(object).columns[1:]
    
    for col in col_to_cast:
        X[col] = X[col].str.replace("[\[\]]", "", regex=True).astype("int64")
        
    return X, y

X, y = load_train()
X.head()


y.head()


# Transforming into an array shaped (96, 2, 2048)
# 96 samples; 2 measurement arrays; 2048 data points

X_reshaped = X.drop("sample_id", axis=1).values.reshape(-1, 2, 2048)
X_reshaped


# Mean input
X_mean = X_reshaped.mean(axis=1)  # Axis 1 to do mean on each set of array
print("X_mean:", X_mean, sep="\n")
print("X_mean.shape:", X_mean.shape, sep="\n") # (96, 2048); 96 samples; 2048 mean points of 2 measurements


# Two inputs 
# You can train a NN with 2 input heads or you can train 2 separate other model and average the prediction values

X0 = X_reshaped[:, 0, :]  # Selecting the first measurement
X1 = X_reshaped[:, 1, :]  # Selecting the second measurement
print("X0:", X0, sep="\n")
print("X0.shape:", X0.shape, sep="\n") # (96, 2048); 96 samples; 2048 individual points of a measurement
print("------------------------------------------")
print("X1:", X1, sep="\n")
print("X1.shape:", X1.shape, sep="\n") # (96, 2048); 96 samples; 2048 individual points of a measurement


def load_submission():
    df = pd.read_csv("/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/96_samples.csv", header=None)
    df.columns = ["sample_id"] + [i for i in range(df.shape[1]-1)]
    col_to_cast = df.select_dtypes(object).columns[1:]
    for col in col_to_cast:
        df[col] = df[col].str.replace("[\[\]]", "", regex=True).astype("int64")
    df.sample_id = df.sample_id.ffill().str.strip().str.replace("sample", "").astype("int8")
    return df

sub = load_submission()
sub.head()


sub_reshaped = sub.drop("sample_id", axis=1).values.reshape(-1, 2, 2048)
sub_reshaped


# Mean input
sub_mean = sub_reshaped.mean(axis=1)  # Axis 1 to do mean on each set of array
print("sub_mean:", sub_mean, sep="\n")
print("sub_mean.shape:", sub_mean.shape, sep="\n") # (96, 2048); 96 samples; 2048 mean points of 2 measurements


# Two inputs 
# You can train a NN with 2 input heads or you can train 2 separate other model and average the prediction values

sub_reshaped0 = sub_reshaped[:, 0, :]  # Selecting the first measurement
sub_reshaped1 = sub_reshaped[:, 1, :]  # Selecting the second measurement
print("sub_reshaped0:", sub_reshaped0, sep="\n")
print("sub_reshaped0.shape:", sub_reshaped0.shape, sep="\n") # (96, 2048); 96 samples; 2048 individual points of a measurement
print("------------------------------------------")
print("sub_reshaped1:", sub_reshaped1, sep="\n")
print("sub_reshaped1.shape:", sub_reshaped1.shape, sep="\n") # (96, 2048); 96 samples; 2048 individual points of a measurement

