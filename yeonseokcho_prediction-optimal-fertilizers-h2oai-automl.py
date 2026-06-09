import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# Data Loading

train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

print(train.shape, test.shape, sample_submission.shape)
train.head()


sample_submission.head()


# generate new features

train['NPK_sum'] = train['Nitrogen'] + train['Phosphorous'] + train['Potassium']

train['N_P_ratio'] = train['Nitrogen'] / (train['Phosphorous'] + 1)
train['N_K_ratio'] = train['Nitrogen'] / (train['Potassium'] + 1)
train['P_K_ratio'] = train['Phosphorous'] / (train['Potassium'] + 1)

train['Soil_Crop'] = train['Soil Type'] + "_" + train['Crop Type']

test['NPK_sum'] = test['Nitrogen'] + test['Phosphorous'] + test['Potassium']

test['N_P_ratio'] = test['Nitrogen'] / (test['Phosphorous'] + 1)
test['N_K_ratio'] = test['Nitrogen'] / (test['Potassium'] + 1)
test['P_K_ratio'] = test['Phosphorous'] / (test['Potassium'] + 1)

test['Soil_Crop'] = test['Soil Type'] + "_" + test['Crop Type']

train.shape, test.shape,


print("Target value counts:")
print(train['Fertilizer Name'].value_counts())

plt.figure(figsize=(6,2))
sns.countplot(y='Fertilizer Name', data=train, order=train['Fertilizer Name'].value_counts().index)
plt.title("Class Distribution of Target (Fertilizer Name)")
plt.show()


# label encoding for Target
from sklearn.preprocessing import LabelEncoder

le_fert = LabelEncoder()
train['Fertilizer Name'] = le_fert.fit_transform(train['Fertilizer Name'])

all_data = pd.concat([train.drop('Fertilizer Name', axis=1), test], axis=0, ignore_index=True)

# One-Hot Encoding for Soil Type, Crop Type
all_data = pd.get_dummies(all_data, columns=['Soil Type', 'Crop Type'])

train_features = all_data.iloc[:len(train), :]
test_features = all_data.iloc[len(train):, :]

features = train_features.columns.tolist()
target = 'Fertilizer Name'

print("Sample after One-Hot Encoding:")
print(train_features.shape)
train_features.head()


print("Target sample:")
train[target].head()


import h2o
from h2o.automl import H2OAutoML

h2o.init()

train_h2o = h2o.H2OFrame(pd.concat([train_features, train[target]], axis=1))

# Set target as factor for classification
train_h2o[target] = train_h2o[target].asfactor()

test_h2o = h2o.H2OFrame(test_features)

features = [col for col in train_h2o.columns if col != target and col != 'id']


print(features)
print(train_h2o.columns)


aml = H2OAutoML(
    max_models=None, #default=None
    max_runtime_secs=3600, #default=3600 (1hr)
    nfolds=5, # default=5
    seed=42,
    sort_metric="mean_per_class_error"
)
aml.train(x=features, y=target, training_frame=train_h2o)


# Install libraries
!pip install polars pyarrow


lb = aml.leaderboard.as_data_frame(use_multi_thread=True)
print(lb.head())

plt.figure(figsize=(6,2))
sns.barplot(x='mean_per_class_error', y='model_id', data=lb.head(10), palette="viridis")
plt.title("Top 10 AutoML Models (mean_per_class_error)")
plt.xlabel("Mean Per Class Error")
plt.ylabel("Model ID")
plt.show()


from sklearn.model_selection import train_test_split

X = train_features
y = train[target]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert validation set to H2OFrame
val_df = X_val.copy()
val_df[target] = y_val
val_h2o = h2o.H2OFrame(val_df)
val_h2o[target] = val_h2o[target].asfactor()

# Predict on validation set
val_pred = aml.leader.predict(val_h2o).as_data_frame(use_multi_thread=True)
val_true = y_val.values

# Top-3 prediction extraction
val_pred_proba = val_pred.iloc[:,1:].values  # skip first column (predicted label)
val_pred_top3_idx = np.argsort(-val_pred_proba, axis=1)[:,:3]
val_pred_top3_label = le_fert.inverse_transform(val_pred_top3_idx.ravel()).reshape(val_pred_top3_idx.shape)
val_true_label = le_fert.inverse_transform(val_true)

# Show some prediction examples
print("\n[Sample Top-3 Predictions vs True Label]")
for i in range(10):
    print(f"Predicted Top-3: {val_pred_top3_label[i]}, True: {val_true_label[i]}")

# Top-3 accuracy calculation
top3_correct = [val_true_label[i] in val_pred_top3_label[i] for i in range(len(val_true_label))]
top3_acc = np.mean(top3_correct)
print(f"\nTop-3 Validation Accuracy: {top3_acc:.4f}")


# Prediction Probability Visualization

plt.figure(figsize=(6,2))
sns.histplot(np.max(val_pred_proba, axis=1), bins=30, kde=True)
plt.title("Histogram of Max Predicted Probability (Validation Set)")
plt.xlabel("Max Predicted Probability")
plt.ylabel("Frequency")
plt.show()


# Test Set Prediction and Submission

test_pred = aml.leader.predict(test_h2o).as_data_frame(use_multi_thread=True)
test_pred_proba = test_pred.iloc[:,1:].values
test_pred_top3_idx = np.argsort(-test_pred_proba, axis=1)[:,:3]
test_pred_top3_label = le_fert.inverse_transform(test_pred_top3_idx.ravel()).reshape(test_pred_top3_idx.shape)

submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(row) for row in test_pred_top3_label]
})
submission.to_csv('submission.csv', index=False)
print(submission.head())


# Summary Visualization: 

top1_pred_idx = np.argmax(test_pred_proba, axis=1)
top1_pred_label = le_fert.inverse_transform(top1_pred_idx)
plt.figure(figsize=(6,2))
sns.countplot(y=top1_pred_label, order=pd.Series(top1_pred_label).value_counts().index)
plt.title("Top-1 Predicted Fertilizer Distribution (Test Set)")
plt.xlabel("Count")
plt.ylabel("Fertilizer Name")
plt.show()

