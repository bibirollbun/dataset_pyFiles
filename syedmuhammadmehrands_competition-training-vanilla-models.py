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


from sklearn.preprocessing import LabelEncoder


import pandas as pd
df_train = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')


df_demographics = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')




df_train.head()


df_demographics.head(20)


df_test = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
df_test.head()


df_train['gesture'].value_counts()


df_merged = df_train.merge(df_demographics, on='subject')
df_merged.head()


df_merged.columns


df_merged.info()


import pandas as pd

pd.set_option('display.max_rows', 400)


missing_data = df_merged.isnull().sum()
missing_data


df_merged.shape


pd.set_option('display.max_rows', 400)


print(df_merged.columns.tolist())


columns_to_drop = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5', 'tof_1_v0', 'tof_1_v1', 'tof_1_v2', 'tof_1_v3', 'tof_1_v4', 'tof_1_v5', 'tof_1_v6', 'tof_1_v7', 'tof_1_v8', 'tof_1_v9', 'tof_1_v10', 'tof_1_v11', 'tof_1_v12', 'tof_1_v13', 'tof_1_v14', 'tof_1_v15', 'tof_1_v16', 'tof_1_v17', 'tof_1_v18', 'tof_1_v19', 'tof_1_v20', 'tof_1_v21', 'tof_1_v22', 'tof_1_v23', 'tof_1_v24', 'tof_1_v25', 'tof_1_v26', 'tof_1_v27', 'tof_1_v28', 'tof_1_v29', 'tof_1_v30', 'tof_1_v31', 'tof_1_v32', 'tof_1_v33', 'tof_1_v34', 'tof_1_v35', 'tof_1_v36', 'tof_1_v37', 'tof_1_v38', 'tof_1_v39', 'tof_1_v40', 'tof_1_v41', 'tof_1_v42', 'tof_1_v43', 'tof_1_v44', 'tof_1_v45', 'tof_1_v46', 'tof_1_v47', 'tof_1_v48', 'tof_1_v49', 'tof_1_v50', 'tof_1_v51', 'tof_1_v52', 'tof_1_v53', 'tof_1_v54', 'tof_1_v55', 'tof_1_v56', 'tof_1_v57', 'tof_1_v58', 'tof_1_v59', 'tof_1_v60', 'tof_1_v61', 'tof_1_v62', 'tof_1_v63', 'tof_2_v0', 'tof_2_v1', 'tof_2_v2', 'tof_2_v3', 'tof_2_v4', 'tof_2_v5', 'tof_2_v6', 'tof_2_v7', 'tof_2_v8', 'tof_2_v9', 'tof_2_v10', 'tof_2_v11', 'tof_2_v12', 'tof_2_v13', 'tof_2_v14', 'tof_2_v15', 'tof_2_v16', 'tof_2_v17', 'tof_2_v18', 'tof_2_v19', 'tof_2_v20', 'tof_2_v21', 'tof_2_v22', 'tof_2_v23', 'tof_2_v24', 'tof_2_v25', 'tof_2_v26', 'tof_2_v27', 'tof_2_v28', 'tof_2_v29', 'tof_2_v30', 'tof_2_v31', 'tof_2_v32', 'tof_2_v33', 'tof_2_v34', 'tof_2_v35', 'tof_2_v36', 'tof_2_v37', 'tof_2_v38', 'tof_2_v39', 'tof_2_v40', 'tof_2_v41', 'tof_2_v42', 'tof_2_v43', 'tof_2_v44', 'tof_2_v45', 'tof_2_v46', 'tof_2_v47', 'tof_2_v48', 'tof_2_v49', 'tof_2_v50', 'tof_2_v51', 'tof_2_v52', 'tof_2_v53', 'tof_2_v54', 'tof_2_v55', 'tof_2_v56', 'tof_2_v57', 'tof_2_v58', 'tof_2_v59', 'tof_2_v60', 'tof_2_v61', 'tof_2_v62', 'tof_2_v63', 'tof_3_v0', 'tof_3_v1', 'tof_3_v2', 'tof_3_v3', 'tof_3_v4', 'tof_3_v5', 'tof_3_v6', 'tof_3_v7', 'tof_3_v8', 'tof_3_v9', 'tof_3_v10', 'tof_3_v11', 'tof_3_v12', 'tof_3_v13', 'tof_3_v14', 'tof_3_v15', 'tof_3_v16', 'tof_3_v17', 'tof_3_v18', 'tof_3_v19', 'tof_3_v20', 'tof_3_v21', 'tof_3_v22', 'tof_3_v23', 'tof_3_v24', 'tof_3_v25', 'tof_3_v26', 'tof_3_v27', 'tof_3_v28', 'tof_3_v29', 'tof_3_v30', 'tof_3_v31', 'tof_3_v32', 'tof_3_v33', 'tof_3_v34', 'tof_3_v35', 'tof_3_v36', 'tof_3_v37', 'tof_3_v38', 'tof_3_v39', 'tof_3_v40', 'tof_3_v41', 'tof_3_v42', 'tof_3_v43', 'tof_3_v44', 'tof_3_v45', 'tof_3_v46', 'tof_3_v47', 'tof_3_v48', 'tof_3_v49', 'tof_3_v50', 'tof_3_v51', 'tof_3_v52', 'tof_3_v53', 'tof_3_v54', 'tof_3_v55', 'tof_3_v56', 'tof_3_v57', 'tof_3_v58', 'tof_3_v59', 'tof_3_v60', 'tof_3_v61', 'tof_3_v62', 'tof_3_v63', 'tof_4_v0', 'tof_4_v1', 'tof_4_v2', 'tof_4_v3', 'tof_4_v4', 'tof_4_v5', 'tof_4_v6', 'tof_4_v7', 'tof_4_v8', 'tof_4_v9', 'tof_4_v10', 'tof_4_v11', 'tof_4_v12', 'tof_4_v13', 'tof_4_v14', 'tof_4_v15', 'tof_4_v16', 'tof_4_v17', 'tof_4_v18', 'tof_4_v19', 'tof_4_v20', 'tof_4_v21', 'tof_4_v22', 'tof_4_v23', 'tof_4_v24', 'tof_4_v25', 'tof_4_v26', 'tof_4_v27', 'tof_4_v28', 'tof_4_v29', 'tof_4_v30', 'tof_4_v31', 'tof_4_v32', 'tof_4_v33', 'tof_4_v34', 'tof_4_v35', 'tof_4_v36', 'tof_4_v37', 'tof_4_v38', 'tof_4_v39', 'tof_4_v40', 'tof_4_v41', 'tof_4_v42', 'tof_4_v43', 'tof_4_v44', 'tof_4_v45', 'tof_4_v46', 'tof_4_v47', 'tof_4_v48', 'tof_4_v49', 'tof_4_v50', 'tof_4_v51', 'tof_4_v52', 'tof_4_v53', 'tof_4_v54', 'tof_4_v55', 'tof_4_v56', 'tof_4_v57', 'tof_4_v58', 'tof_4_v59', 'tof_4_v60', 'tof_4_v61', 'tof_4_v62', 'tof_4_v63', 'tof_5_v0', 'tof_5_v1', 'tof_5_v2', 'tof_5_v3', 'tof_5_v4', 'tof_5_v5', 'tof_5_v6', 'tof_5_v7', 'tof_5_v8', 'tof_5_v9', 'tof_5_v10', 'tof_5_v11', 'tof_5_v12', 'tof_5_v13', 'tof_5_v14', 'tof_5_v15', 'tof_5_v16', 'tof_5_v17', 'tof_5_v18', 'tof_5_v19', 'tof_5_v20', 'tof_5_v21', 'tof_5_v22', 'tof_5_v23', 'tof_5_v24', 'tof_5_v25', 'tof_5_v26', 'tof_5_v27', 'tof_5_v28', 'tof_5_v29', 'tof_5_v30', 'tof_5_v31', 'tof_5_v32', 'tof_5_v33', 'tof_5_v34', 'tof_5_v35', 'tof_5_v36', 'tof_5_v37', 'tof_5_v38', 'tof_5_v39', 'tof_5_v40', 'tof_5_v41', 'tof_5_v42', 'tof_5_v43', 'tof_5_v44', 'tof_5_v45', 'tof_5_v46', 'tof_5_v47', 'tof_5_v48', 'tof_5_v49', 'tof_5_v50', 'tof_5_v51', 'tof_5_v52', 'tof_5_v53', 'tof_5_v54', 'tof_5_v55', 'tof_5_v56', 'tof_5_v57', 'tof_5_v58', 'tof_5_v59', 'tof_5_v60', 'tof_5_v61', 'tof_5_v62', 'tof_5_v63']
print(columns_to_drop)


df_imu = df_merged.drop(columns=columns_to_drop, axis=1)
df_imu.shape


missing_data_imu = df_imu.isnull().sum()
missing_data_imu


msk = df_imu['rot_w'].isna()
temp = df_imu[msk]


temp['sequence_id'].value_counts()


df_imu['sequence_id'].unique().shape


sequences_to_remove = temp['sequence_id'].unique().tolist()



df_imu = df_imu[~df_imu['sequence_id'].isin(sequences_to_remove)].reset_index(drop=True)



missing_data = df_imu.isnull().sum()
missing_data


df_imu.dtypes


le = LabelEncoder()
df_imu['gesture_encoded'] = le.fit_transform(df_imu['gesture'])
df_imu[['gesture_encoded', 'gesture']].value_counts()


df_imu[df_imu['sequence_id']=='SEQ_000008']


df_imu.dtypes


df_new = df_imu.copy()
df_new.columns








from sklearn.preprocessing import OneHotEncoder





df_imu.head()


cat_col = df_imu.select_dtypes(include=['object'])
cat_col.columns


# !pip install numpy==1.24.3 --force-reinstall



import seaborn as sns
import matplotlib.pyplot as plt

for col in ['orientation', 'behavior', 'phase']:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df_imu, x=col, hue='gesture', order=df_imu[col].value_counts().index)
    plt.title(f'Gesture distribution over {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


num_col = df_imu.select_dtypes(include=['number'])
num_col.columns


for col in ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x',
       'rot_y', 'rot_z', 'adult_child', 'age', 'sex', 'handedness',
       'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_imu, x='gesture', y=col)
    plt.title(f'{col} distribution per gesture')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


df_imu.groupby('sequence_id')['orientation'].nunique().value_counts()



df_imu[num_col.columns].corr()


sequence_labels = df_imu[['sequence_id', 'gesture_encoded']].drop_duplicates()

sequence_labels


from sklearn.model_selection import StratifiedShuffleSplit

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Step 3: Generate train/val sequence IDs
for train_idx, val_idx in strat_split.split(sequence_labels['sequence_id'], sequence_labels['gesture_encoded']):
    train_ids = sequence_labels.iloc[train_idx]
    val_ids = sequence_labels.iloc[val_idx]


train_seq_ids = train_ids['sequence_id'].values
val_seq_ids = val_ids['sequence_id'].values

df_train_imu = df_imu[df_imu['sequence_id'].isin(train_seq_ids)]
df_val_imu = df_imu[df_imu['sequence_id'].isin(val_seq_ids)]


print(df_train_imu.shape)
print(df_val_imu.shape)


def aggregate_features(df):
    agg = df.groupby('sequence_id').agg({
        'acc_x': ['mean', 'std'],
        'acc_y': ['mean', 'std'],
        'acc_z': ['mean', 'std'],
        'rot_w': ['mean', 'std'],
        'rot_x': ['mean', 'std'],
        'rot_y': ['mean', 'std'],
        'rot_z': ['mean', 'std'],
        # Add other sensor features here
    }).reset_index()
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns]
    return agg




X_train = aggregate_features(df_train_imu)
X_val = aggregate_features(df_val_imu)


X_train.head()


seq_id_train = df_train_imu[['sequence_id', 'gesture_encoded']].drop_duplicates()
seq_id_val = df_val_imu[['sequence_id', 'gesture_encoded']].drop_duplicates()

df_training = X_train.merge(seq_id_train, on='sequence_id', how='left')
df_validation = X_val.merge(seq_id_val, on='sequence_id', how='left')
df_training.head()


def frequency_encode_categoricals(df, categorical_cols):
    freq_features = []
    for col in categorical_cols:
        counts = pd.crosstab(df['sequence_id'], df[col], normalize='index')
        counts.columns = [f'{col}_freq_{c}' for c in counts.columns]
        freq_features.append(counts)
    return pd.concat(freq_features, axis=1).reset_index()


cat_train = frequency_encode_categoricals(df_train_imu, ['behavior', 'phase'])
cat_val = frequency_encode_categoricals(df_val_imu, ['behavior', 'phase'])

cat_train.head()


df_train_final = df_training.merge(cat_train, on='sequence_id', how='left')
df_val_final = df_validation.merge(cat_val, on='sequence_id', how='left')

df_train_final.head()


sequence_ids_train = df_train_final['sequence_id'].copy()
sequence_ids_val = df_val_final['sequence_id'].copy()


X_train = df_train_final.drop(['gesture_encoded', 'sequence_id'], axis=1)
y_train = df_train_final['gesture_encoded']

X_val = df_val_final.drop(['gesture_encoded', 'sequence_id'], axis=1)
y_val = df_val_final['gesture_encoded']

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

X_train.head()



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# !pip install scikit-learn==1.5.0



# !pip install -U scikit-learn imbalanced-learn



from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score



gesture_to_type = df_imu.drop_duplicates('gesture')[['gesture', 'sequence_type']].set_index('gesture')['sequence_type'].to_dict()
gesture_to_type


def evaluate_bfrb_f1(y_true_encoded, y_pred_encoded, label_encoder, target_gestures):
    """
    Evaluates Binary F1 and Macro F1 as per the BFRB competition rules.
    
    Parameters:
    - y_true_encoded: array-like of encoded true gesture labels
    - y_pred_encoded: array-like of encoded predicted gesture labels
    - label_encoder: fitted sklearn.preprocessing.LabelEncoder
    - target_gestures: list of gesture names considered as target gestures (BFRB)
    
    Returns:
    - A dictionary with binary_f1, macro_f1, and final_score
    """
    # Decode labels back to gesture names
    y_true = label_encoder.inverse_transform(y_true_encoded)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    # Convert to binary: 'target' vs 'non_target'
    y_true_binary = ['target' if gesture in target_gestures else 'non_target' for gesture in y_true]
    y_pred_binary = ['target' if gesture in target_gestures else 'non_target' for gesture in y_pred]

    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary', pos_label='target')

    # Collapse all non-target gestures into 'non_target' for macro F1
    y_true_collapsed = [gesture if gesture in target_gestures else 'non_target' for gesture in y_true]
    y_pred_collapsed = [gesture if gesture in target_gestures else 'non_target' for gesture in y_pred]

    macro_f1 = f1_score(y_true_collapsed, y_pred_collapsed, average='macro')

    final_score = (binary_f1 + macro_f1) / 2

    return {
        'binary_f1': binary_f1,
        'macro_f1': macro_f1,
        'final_score': final_score
    }    


temp = df_imu[df_imu['sequence_type']=='Target']
temp['gesture'].unique()


target_gesture_names = ['Cheek - pinch skin', 'Forehead - pull hairline', 'Neck - scratch',
       'Neck - pinch skin', 'Eyelash - pull hair', 'Eyebrow - pull hair',
       'Forehead - scratch', 'Above ear - pull hair']


gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=10, subsample=0.5)
gbc.fit(X_train, y_train)
# y_pred_gbc = gbc.predict(X_val_scaled)

# scores_gbc = evaluate_bfrb_f1(y_val, y_pred_gbc, le, target_gesture_names)

# print(f"Binary F1: {scores_gbc['binary_f1']:.4f}")
# print(f"Macro F1: {scores_gbc['macro_f1']:.4f}")
# print(f"Final Score: {scores_gbc['final_score']:.4f}")


model_rf = RandomForestClassifier(n_estimators=600, max_features='sqrt', min_samples_split=2, class_weight='balanced_subsample', max_depth=15, bootstrap=True)
model_rf.fit(X_train, y_train)
# y_pred_rf = model_rf.predict(X_val_scaled)
# scores_rf = evaluate_bfrb_f1(y_val, y_pred_rf, le, target_gesture_names)

# print(f"Binary F1: {scores_rf['binary_f1']:.4f}")
# print(f"Macro F1: {scores_rf['macro_f1']:.4f}")
# print(f"Final Score: {scores_rf['final_score']:.4f}")

# train_pred = model_rf.predict(X_train_resampled)
# f1_score(y_train_resampled, train_pred, average='macro')


import polars as pl
import joblib  # or pickle
import os

joblib.dump(gbc, 'gbc.joblib')
#joblib.dump(model_rf, 'rf.joblib')



joblib.dump(le, "label_encoder.joblib")
joblib.dump(model_rf, 'rf.joblib')


