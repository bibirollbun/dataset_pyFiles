# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from fastai.tabular.all import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# Identifying paths
base_dir = "/kaggle/input/playground-series-s5e5/"
train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")

# Load datasets
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)


df_train.info()


df_train.describe()


df_test.info()


df_test.describe()


# Creating list of categorical and numerical columns 
cat_columns = []
num_columns = []
target = "Calories"
for column in df_train.columns:
   
    if df_train[column].dtype == "object":
        cat_columns.append(column)
    else:
        num_columns.append(column)

num_columns.remove(target)
print("List of categorical columns: ", cat_columns)
print("List of numerical columns: ", num_columns)


tr = TabularPandas(
    df_train, # Training dataset
    cat_names=cat_columns, # Names of categorical columns
    cont_names=num_columns, # Names of columns with continous values (numeric)
    y_names=target, # Name for feature that we are predicting
    splits=RandomSplitter(seed=42)(range_of(df_train)), # Splitting our dataset into validation sets
    procs=[Categorify] # Transformaitons that will be done on dataset (In this example will make our categoircal columns numerical)
)


dls = tr.dataloaders(bs=64) # Creating dataloader with batch size of 64


def rmsle(preds, targets):
    preds = torch.clamp(preds, min=0)  # prevent log of negative
    return torch.sqrt(F.mse_loss(torch.log1p(preds), torch.log1p(targets)))

rmsle_metric = AccumMetric(rmsle, flatten=False)


model = tabular_learner(
    dls, # Dataloader
    layers=[20, 10], # NN with 2 hidden layers with 20 and 10 units
    metrics=[rmsle_metric], 
    loss_func=F.mse_loss, # Function that will be used as loss
)

model.fit_one_cycle(10) # Training model with 10 epochs


model.recorder.plot_loss()


# Create test TabularPandas using training pipeline
to_test = TabularPandas(df_test, 
                        procs=[Categorify],                  # same transformations
                        cat_names=tr.cat_names, 
                        cont_names=tr.cont_names, 
                        y_names=None, 
                        splits=None)




test_dl = model.dls.test_dl(to_test.items)  # Pass raw dataframe, not TabularPandas


preds, _ = model.get_preds(dl=test_dl)
df_test['Calories'] = preds.squeeze().numpy()


df_test


submission_df = df_test[["id", "Calories"]]
submission_df.to_csv("submission.csv")

