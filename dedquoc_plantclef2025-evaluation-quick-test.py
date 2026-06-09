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


import pandas as pd
from sklearn.metrics import f1_score

# Example ground truth and predictions
ground_truth = pd.DataFrame({
    'quadrat_id': ['CBN-Pla-B1-20130724', 'CBN-PdlC-A1-20130807'],
    'species_ids': [[1395806], [1351284, 1494911, 1381367, 1396535, 1412857, 1295807]]
})

predictions = pd.DataFrame({
    'quadrat_id': ['CBN-Pla-B1-20130724', 'CBN-PdlC-A1-20130807'],
    'species_ids': [[1395806], [1351284, 1494911, 1381367, 1396535, 1412857]]
})

# Function to compute F1 score for a single quadrat
def compute_f1_score(ground_truth_species, predicted_species):
    tp = len(set(ground_truth_species) & set(predicted_species))
    fp = len(set(predicted_species) - set(ground_truth_species))
    fn = len(set(ground_truth_species) - set(predicted_species))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

# Compute F1 scores for each quadrat
f1_scores = []
for i in range(len(ground_truth)):
    ground_truth_species = ground_truth.iloc[i]['species_ids']
    predicted_species = predictions.iloc[i]['species_ids']
    f1 = compute_f1_score(ground_truth_species, predicted_species)
    f1_scores.append(f1)

# Compute the macro-averaged F1 score per sample
macro_averaged_f1_per_sample = sum(f1_scores) / len(f1_scores)

print(f"Macro-averaged F1 score per sample: {macro_averaged_f1_per_sample}")

