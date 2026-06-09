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


# TabPFN extension package
!pip install -q --no-index --find-links /kaggle/input/tabpfn-v2-install/tabpfn-extensions/dist tabpfn-extensions
!mkdir -p /usr/local/lib/python3.10/dist-packages/tabpfn_extensions/hpo/hpo_models
!cp /kaggle/input/tabpfn-v2-install/tabpfn-v2-*.ckpt /usr/local/lib/python3.10/dist-packages/tabpfn_extensions/hpo/hpo_models
# TabPFN package
!pip install -q --no-index --find-links /kaggle/input/tabpfn-v2-install tabpfn
!mkdir -p /root/.cache/tabpfn/
!cp /kaggle/input/tabpfn-v2-install/tabpfn-v2-*.ckpt /root/.cache/tabpfn/


import numpy as np
import pandas as pd
#from tabpfn import TabPFNClassifier
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
    AutoTabPFNClassifier,
    AutoTabPFNRegressor,
)
classifier = AutoTabPFNClassifier(max_time=60*60*4, ges_scoring_string="auroc", device="cuda")
#classifier = TabPFNClassifier(n_estimators=64)



# from https://github.com/emanuele/kaggle_pbr/blob/master/load_data.py
import numpy as np

def read_data(file_name):
    """This function is adapted from:
    https://github.com/benhamner/BioResponse/blob/master/Benchmarks/csv_io.py
    """
    f = open(file_name)
    # skip header
    f.readline()
    samples = []
    for line in f:
        line = line.strip().split(",")
        sample = [float(x) for x in line]
        samples.append(sample)
    return samples


def load(base_path):
    """Conveninence function to load all data as numpy arrays.
    """
    train = read_data(f"{base_path}/train.csv")
    y_train = np.array([x[0] for x in train])
    X_train = np.array([x[1:] for x in train])
    X_test = np.array(read_data(f"{base_path}/test.csv"))
    return X_train, y_train, X_test

X_train, y_train, X_test = load(base_path="/kaggle/input/predict-who-is-more-influential-in-a-social-network")


classifier.fit(X_train, y_train)


y_pred = classifier.predict_proba(X_test)[:, 1]


df_submission = pd.read_csv("/kaggle/input/predict-who-is-more-influential-in-a-social-network/sample_predictions.csv")


df_submission["Choice"] = y_pred
df_submission.to_csv("submission.csv", index=False)




