!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


!pip install mordred --no-index --find-links=file:///kaggle/input/mordred-1-2-0-py3-none-any/


import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# RDKit
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')  # RDKitの冗長なログを抑制

# Mordred
from mordred import Calculator, descriptors

# PyTorch
import torch

# TabPFN
from tabpfn import TabPFNRegressor
from tabpfn.model_loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model


tg=pd.read_csv('/kaggle/input/modred-dataset/desc_tg.csv')
tc=pd.read_csv('/kaggle/input/modred-dataset/desc_tc.csv')
rg=pd.read_csv('/kaggle/input/modred-dataset/desc_rg.csv')
ffv=pd.read_csv('/kaggle/input/modred-dataset/desc_ffv.csv')
density=pd.read_csv('/kaggle/input/modred-dataset/desc_de.csv')
test=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
ID=test['id']

for i in (tg,tc,rg,ffv,density):
     i.drop(columns=[col for col in i.columns if i[col].nunique() == 1],axis=1,inplace=True)

# Remove columns with object or category dtype
tg = tg.select_dtypes(exclude=['object', 'category'])
rg = rg.select_dtypes(exclude=['object', 'category'])
ffv = ffv.select_dtypes(exclude=['object', 'category'])
tc = tc.select_dtypes(exclude=['object', 'category'])
density  = density.select_dtypes(exclude=['object', 'category'])


# Make test ds
mols_test = [Chem.MolFromSmiles(s) for s in test.SMILES]
# Initialize the Mordred Calculator
calc = Calculator(descriptors, ignore_3D=True) # ignore_3D=True for 2D descriptors
desc_test = calc.pandas(mols_test)


desc_test.drop(columns=[col for col in desc_test.columns if desc_test[col].nunique() == 1],axis=1,inplace=True)
desc_test = desc_test.select_dtypes(exclude=['object', 'category'])
desc_test.dropna(axis=1, how='all', inplace=True)


def batch_predict(model, X, batch_size=512):
    preds = []
    for i in range(0, len(X), batch_size):
        preds.append(model.predict(X[i:i+batch_size]))
    return np.concatenate(preds)

def model(train_d,test_d,model,target,submission=False):
    # We divide the data into training and validation sets for model evaluation
    train_cols = set(train_d.columns) - {target}
    test_cols = set(test_d.columns)
    # Intersect the feature columns
    common_cols = list(train_cols & test_cols)
    X=train_d[common_cols].copy()
    y=train_d[target].copy()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

    Model=model(
        model_path="/kaggle/input/tabpfn-v2-regressor/pytorch/default/1/tabpfn-v2-regression.ckpt",
        ignore_pretraining_limits=True,
        inference_config={
            "SUBSAMPLE_SAMPLES": 5000
        },
    )
    if submission==False:
        Model.fit(X_train,y_train)
        print("Fitting Done!")
        print("Saving model...")
        save_fitted_tabpfn_model(Model, Path(f"trained_reg_{target}.tabpfn_fit"))
        y_pred = batch_predict(Model, X_test, batch_size=256)
        del Model
        torch.cuda.empty_cache()
        return mean_absolute_error(y_pred,y_test)         # We assess our model performance using MAE metric

    
    if submission==True:
        Model.fit(X,y)
        print("Fitting Done!")
        print("Saving model...")
        save_fitted_tabpfn_model(Model, Path(f"trained_reg_{target}.tabpfn_fit"))
        submission=batch_predict(Model, test_d[common_cols].copy(), batch_size=256)
        del Model
        torch.cuda.empty_cache()
        return submission


sub = {'id': ID}
for target, data in zip(['Tg', 'FFV', 'Tc', 'Density', 'Rg'], [tg, ffv, tc, density, rg]):
    preds = model(data, desc_test, TabPFNRegressor, target, submission=True)
    sub[target] = preds
    del preds
    torch.cuda.empty_cache()


pfn_sub_df=pd.DataFrame(sub)
pfn_sub_df=pfn_sub_df.set_index('id')


print(pfn_sub_df)


pfn_sub_df.to_csv("submission.csv")

