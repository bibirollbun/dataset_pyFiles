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


import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

# Auto-detect target column (last column that's not 'id')
target_col = [col for col in train.columns if col != 'id'][-1]
print(f"Using target column: {target_col}")
y_true = train[target_col].values

# Load all OOF and test predictions
oof_files = sorted(glob("/kaggle/input/s5e10-oof-predictions/*oof*.npy"))
test_files = sorted(glob("/kaggle/input/s5e10-oof-predictions/*test*.npy"))

print(f"Found {len(oof_files)} OOF files and {len(test_files)} test files")

oof_preds = [np.load(f) for f in oof_files]
test_preds = [np.load(f) for f in test_files]

# Simple Average Ensemble
oof_avg = np.mean(oof_preds, axis=0)
test_avg = np.mean(test_preds, axis=0)

# Calculate RMSE CV score
rmse_cv = np.sqrt(mean_squared_error(y_true, oof_avg))
print(f"Simple Average RMSE CV: {rmse_cv:.6f}")

# Save OOF and test predictions
np.save('oof_predictions.npy', oof_avg)
np.save('test_predictions.npy', test_avg)
print("✓ OOF and test predictions saved!")

# Create submission
submission = pd.DataFrame({'id': test['id'], target_col: test_avg})
submission.to_csv('submission.csv', index=False)
print("✓ Submission saved!")


sub1= pd.read_csv("/kaggle/input/road-accident-risk-blend/submission.csv")
sub2= pd.read_csv("/kaggle/input/roadsol/submission.csv")
sub3= pd.read_csv("/kaggle/input/ps-s5e10-lightgbm-cb-ensemble/submission.csv")




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# DataFrame de predicciones
preds = pd.DataFrame({
    'm1': sub1['accident_risk'],
    'm2': sub2['accident_risk'],
    'm3': sub3['accident_risk'],
    'm4': submission['accident_risk']
})

# Correlación entre predicciones
corr_pred = preds.corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr_pred, annot=True, vmin=-1, vmax=1, cmap="coolwarm")
plt.title("Correlación entre predicciones de modelos")
plt.show()





# Aseguramos tener todas las predicciones alineadas
preds = pd.DataFrame({
    'm1': sub1['accident_risk'],
    'm2': sub2['accident_risk'],
    'm3': sub3['accident_risk'],
    'm4': submission['accident_risk']
})

# Promedio simple
pred_final_mean = preds.mean(axis=1)

# Export
submission_mean = pd.DataFrame({
    'id': sub1['id'],
    'accident_risk': pred_final_mean
})

submission_mean.to_csv("submission.csv", index=False)
print("Submission guardada: submission_ensemble_mean.csv")


