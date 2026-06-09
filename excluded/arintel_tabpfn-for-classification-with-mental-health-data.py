pip install tabpfn


# Load required packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from tabpfn import TabPFNClassifier
#from tabpfn import TabPFNRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gc

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#Load train and test datasets
train_path = "/kaggle/input/playground-series-s4e11/train.csv"
test_path = "/kaggle/input/playground-series-s4e11/test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)


#Concatenate test and train datasets for feature transformation
df = pd.concat([df_train, df_test], ignore_index=True)


# Replace missing values of category variables
for col in df.select_dtypes(exclude=[np.number]).columns:
    df[col] = df[col].fillna("Missing")
    df[col] = df[col].fillna("Missing")


df


#Set target column
target_column = 'Depression'

#Split back into test and train data
X_train = df.dropna(subset=[target_column]).drop(columns=[target_column])
y_train=df[target_column].dropna()
X_test = df[df[target_column].isna()].drop(columns=[target_column])


# Initialize model
model = TabPFNClassifier(device='cuda') #activate GPU!
model.fit(X_train[:10000], y_train[:10000]) #tabpfn is limited to 10000 rows and 500 features :(


# Inference is memory intensive and large workloads must be processed in batches
def batch_predict(model, X_test, batch_size=1000):
    predictions = []
    for i in range(0, len(X_test), batch_size):
        print(i)
        torch.cuda.empty_cache()
        batch = X_test[i:i + batch_size]
        batch_predictions = model.predict(batch)
        predictions.extend(batch_predictions)
        gc.collect()
    return np.array(predictions)


# Make predictions
predictions = batch_predict(model, X_test, batch_size=25000)


output_csv="submission.csv"

results_df = pd.DataFrame({"id": X_test["id"].values, "Depression": predictions})
results_df.to_csv(output_csv, index=False)
print(f"Submission saved under {output_csv}")

