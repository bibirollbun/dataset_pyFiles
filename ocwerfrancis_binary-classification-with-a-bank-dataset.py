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


import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (12, 8)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
train_df.head()


test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

test_df.head()


submission_df = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

submission_df


train_df.shape, test_df.shape


train_df.isnull().sum()


train_df.dtypes


train_df.info()


test_df.info()


train_df = train_df.copy()
test_df = test_df.copy()


train_df.head()


train_df.hist(bins = 25,figsize=(20,10))


train_df.columns


train_df.y.value_counts()


train_df.default.value_counts()


train_df.poutcome.value_counts()


train_df.poutcome.dtype


def month_to_number(df):
    # Month mapping dictionary
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    # Apply the mapping
    df['month_number'] = df['month'].str.lower().map(month_map)


month_to_number(train_df)

month_to_number(test_df)


train_df.head()


from sklearn.model_selection import train_test_split


train_df,val_df = train_test_split(train_df, test_size=0.25, random_state=42)


len(train_df), len(val_df)


train_df.shape, val_df.shape, test_df.shape


val_df.head()


train_df.columns


input_cols = ['age', 'job', 'marital', 'education', 'default', 'balance',
       'housing', 'loan', 'contact', 'day', 'duration', 'campaign',
       'pdays', 'previous', 'poutcome', 'month_number']

target_col = 'y'


train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

# Validation dataset inputs and target

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()

test_inputs = test_df[input_cols].copy()


numeric_cols = list(var for var in train_inputs.columns if train_inputs[var].dtype != 'O')
numeric_cols


categorical_cols = list(var for var in train_inputs.columns if train_inputs[var].dtype == 'O')

categorical_cols


train_inputs.isnull().sum()


train_targets.isnull().sum()


from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler().fit(train_inputs[numeric_cols])


train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


categorical_cols


from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))


train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


X_train


# build the lightgbm model
import lightgbm as lgb


model = lgb.LGBMClassifier()


model.fit(X_train, train_targets)


train_pred = model.predict(X_train)
train_pred


train_targets


val_pred = model.predict(X_val)

val_pred


val_targets


# view accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(val_pred, val_targets)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(val_targets, val_pred)))


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(train_targets, train_pred)))


# print the scores on training and test set

print('Training set score: {:.4f}'.format(model.score(X_train, train_targets)))

print('Validation set score: {:.4f}'.format(model.score(X_val, val_targets)))


# Get probability predictions instead of class predictions
val_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of class 1

# Calculate ROC-AUC
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(val_targets, val_pred_proba)
print(f"LightGBM Model ROC-AUC score: {auc_score:.4f}")



# Check class distribution
print("\nðŸ“Š Class Distribution Analysis:")
print(f"Training set - Class 0: {np.sum(train_targets == 0):,} | Class 1: {np.sum(train_targets == 1):,}")
print(f"Validation set - Class 0: {np.sum(val_targets == 0):,} | Class 1: {np.sum(val_targets == 1):,}")

# Check if we have imbalance
class_ratio = np.sum(val_targets == 0) / np.sum(val_targets == 1)
print(f"Class ratio (0:1): {class_ratio:.2f}:1")




# Let's see the prediction distribution to understand the model better
plt.figure(figsize=(12, 4))

# Plot 1: Prediction distribution
plt.subplot(1, 2, 1)
plt.hist(val_pred_proba, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predictions')

# Plot 2: ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(val_targets, val_pred_proba)
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (AUC = {auc_score:.4f})')
plt.tight_layout()
plt.show()


# build the lightgbm model
import lightgbm as lgb

final_model =lgb.LGBMClassifier( metric='auc',n_estimators=2000,reg_lambda=0.2, scale_pos_weight=7.23,learning_rate=0.02,max_bin=255,reg_alpha=0.1, num_leaves=127,random_state=42,n_jobs=-1)

final_model.fit(X_train, train_targets,eval_set=[(X_val, val_targets)],eval_metric='auc')


# print the scores on training and test set

print('Training set score: {:.4f}'.format(final_model.score(X_train, train_targets)))

print('Validation set score: {:.4f}'.format(final_model.score(X_val, val_targets)))


# Get probability predictions instead of class predictions
val_pred_proba = final_model.predict_proba(X_val)[:, 1]  # Probability of class 1

# Calculate ROC-AUC
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(val_targets, val_pred_proba)
print(f"LightGBM Model ROC-AUC score: {auc_score:.4f}")



# Use your best model (0.9675 AUC) to make predictions
test_predictions = final_model.predict_proba(X_test)[:, 1]

print("Test predictions generated!")
print(f"Number of test predictions: {len(test_predictions)}")
print(f"Predictions range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
print(f"Mean prediction: {test_predictions.mean():.4f}")


test_predictions[:3]


# Verify the first few predictions match what we see
print("First 10 test predictions:")
for i, pred in enumerate(test_predictions[:10]):
    print(f"  {i+1:2d}: {pred:.4f} ({pred*100:5.2f}%)")


# Check the distribution of test predictions
plt.figure(figsize=(10, 6))
plt.hist(test_predictions, bins=50, alpha=0.7, edgecolor='black', color='purple')
plt.xlabel('Predicted Probability of Filing Claim')
plt.ylabel('Frequency')
plt.title('Distribution of Test Predictions')
plt.grid(True, alpha=0.3)
plt.show()

# Detailed statistics
print("\nTest Predictions Analysis:")
print(f"Min probability: {test_predictions.min():.4f}")
print(f"Max probability: {test_predictions.max():.4f}")
print(f"Mean probability: {test_predictions.mean():.4f}")
print(f"Std probability: {test_predictions.std():.4f}")
print(f"Percentage > 0.5: {(test_predictions > 0.5).mean():.2%}")
print(f"Percentage > 0.7: {(test_predictions > 0.7).mean():.2%}")


# Create submission with the exact format from the sample
submission_df = pd.DataFrame({
    'id': range(750000, 1000000),  # IDs from 750000 to 999999
    'y': test_predictions          # Column name should be 'y' according to sample
})

print("Submission created with correct format!")
print(f"Columns: {submission_df.columns.tolist()}")
print(f"First few rows:")
print(submission_df.head())


# Save the submission file
final_filename = 'submission.csv'
submission_df.to_csv(final_filename, index=False)
print(f"Submission saved as: {final_filename}")

# Also save a backup with your AUC score
backup_filename = f'submission_auc_{auc_score:.4f}.csv'
submission_df.to_csv(backup_filename, index=False)
print(f"Backup saved as: {backup_filename}")




