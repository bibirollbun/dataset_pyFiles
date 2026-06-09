pip install autogluon


import pandas as pd
import numpy as np
import os
import warnings
from autogluon.tabular import TabularPredictor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
original = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


train = train.drop("id", axis=1)
test = test.drop("id", axis=1)
original = original.drop("id", axis=1)
train = pd.concat([train, original], ignore_index=True)
train = train.drop_duplicates()


train.head()


train.info()


cat_cols = ['Soil Type', 'Crop Type']
le = LabelEncoder()
for col in cat_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


train.head()


fer_label_enc = LabelEncoder()
train["Fertilizer Name"] = fer_label_enc.fit_transform(train["Fertilizer Name"])


X = train.drop(columns=["Fertilizer Name"])
y = train["Fertilizer Name"]
X_test = test


predictor = TabularPredictor(label='Fertilizer Name', eval_metric='log_loss').fit(
    train_data=train,
    presets='best_quality',
    ag_args_fit={'num_gpus': 1},
    verbosity=4  # Max verbosity for debugging
)


predictor.leaderboard(silent=True)


pred_prob = predictor.predict_proba(test)  

top_3_indices = np.argsort(-pred_prob.values, axis=1)[:, :3]

# Map indices to original label names
top_3_labels = fer_label_enc.inverse_transform(top_3_indices.ravel()).reshape(top_3_indices.shape)

# Join top-3 labels as a space-separated string for submission
submission['Fertilizer Name'] = [' '.join(row) for row in top_3_labels.astype(str)]


submission.to_csv('submission.csv', index=False)
print("✅ Final submission saved successfully!")


print("\nSubmission Preview:")
print(submission.head())

