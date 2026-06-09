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


import os
import numpy as np
import pandas as pd
import pickle

# Configuration
DATA_DIR = "/kaggle/input/openpolymer-new-dataset"
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Load test embeddings and sample submission
test_embeddings = np.load(os.path.join(DATA_DIR, "test_embeddings.npy"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

# Helper functions
def load_models(model_type, target):
    path = os.path.join(DATA_DIR, f"models_{model_type}_{target}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_ensemble(models, X):
    preds = np.zeros((len(models), X.shape[0]))
    for i, model in enumerate(models):
        preds[i] = model.predict(X)
    return preds.mean(axis=0)

# Generate predictions
predictions = {}
for target in TARGETS:
    cat_models = load_models("cat", target)
    lgb_models = load_models("lgb", target)
    pred_cat = predict_ensemble(cat_models, test_embeddings)
    pred_lgb = predict_ensemble(lgb_models, test_embeddings)
    predictions[target] = (pred_cat + pred_lgb) / 2

# Prepare submission
submission = sample_sub.copy()
for target in TARGETS:
    submission[target] = predictions[target]

# Ensure id column is string and strip spaces if any
submission['id'] = submission['id'].astype(str).str.strip()

# Round predictions
submission[TARGETS] = submission[TARGETS].round(6)

# Save submission file
submission.to_csv("submission.csv", index=False, encoding='utf-8')
print("submission.csv created successfully!")
print(submission.head())


