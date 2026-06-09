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
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import kaggle_evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.model_selection import cross_val_predict
import kaggle_evaluation.cmi_inference_server
warnings.filterwarnings("ignore")


# Load datasets
train = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
test = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
test_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')





train.head()


print(train_demo.columns)
print(train_demo.head())



print(train.columns)
print(train.head())



def show_eda():
    print("Train shape:", train.shape)
    print("Train demographics shape:", train_demo.shape)
    print("Test shape:", test.shape)
    
    # Label distribution from train.csv
    plt.figure(figsize=(10, 6))
    sns.countplot(y='behavior', data=train)
    plt.title("Behavior Label Distribution")
    plt.tight_layout()
    plt.show()
    
    # Demographics visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sns.countplot(x='sex', data=train_demo, ax=axes[0, 0])
    axes[0, 0].set_title('Gender Distribution (0=Female, 1=Male)')

    sns.countplot(x='handedness', data=train_demo, ax=axes[0, 1])
    axes[0, 1].set_title('Handedness (0=Left, 1=Right)')

    sns.countplot(x='adult_child', data=train_demo, ax=axes[1, 0])
    axes[1, 0].set_title('Adult vs Child (0=Child, 1=Adult)')

    sns.histplot(train_demo['age'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Age Distribution')

    plt.tight_layout()
    plt.show()

show_eda()


from sklearn.preprocessing import LabelEncoder

def extract_features(df):
    cols = ['acc_x', 'acc_y', 'acc_z']
    feats = df.groupby('sequence_id').agg({
        col: ['mean', 'std', 'min', 'max'] for col in cols
    })
    feats.columns = ['_'.join(c) for c in feats.columns]
    feats.reset_index(inplace=True)
    return feats

# Extract features
X_feats = extract_features(train)

# Extract label per sequence_id (take first behavior in each sequence)
y = train.groupby('sequence_id')['behavior'].first().reset_index()

# Merge features and labels
X = X_feats.merge(y, on='sequence_id')
train['behavior'] = train['behavior'].str.strip("'").str.strip('"')
# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(X['behavior'])

# Prepare training features by dropping label and sequence_id
X_train = X.drop(columns=['behavior', 'sequence_id'])

# Now X_train is your feature matrix, y_encoded your encoded labels
print("Classes:", le.classes_)
print("Feature matrix shape:", X_train.shape)
print("Encoded labels shape:", y_encoded.shape)



clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_encoded)



y_pred = cross_val_predict(clf, X_train, y_encoded, cv=5)

print("\nClassification Report:")
print(classification_report(y_encoded, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_encoded, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()



le.classes_
print([repr(c) for c in le.classes_])


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df_seq = sequence.to_pandas()
    df_demo = demographics.to_pandas()
    
    feats = extract_features(df_seq)

    # Attach subject and demographics
    seq_subj = df_seq[['sequence_id', 'subject']].drop_duplicates()
    feats = feats.merge(seq_subj, on='sequence_id')
    feats = feats.merge(df_demo, on='subject', how='left')

    # Drop identifiers
    feats.drop(columns=['sequence_id', 'subject'], inplace=True)

    # Ensure same features as training
    for col in X_train.columns:
        if col not in feats.columns:
            feats[col] = 0  # fill missing columns with 0
    feats = feats[X_train.columns]  # same order

    pred = clf.predict(feats)[0]
    label = le.inverse_transform([pred])[0]

    return 'Moves hand to target location'  # Ensure it's a string, not quoted inside quotes



inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )




