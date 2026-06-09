#1 - Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


#2 - Data Loading
train = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
test = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
test_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)


#3 - Feature Engineering
# Filter only 'Performs gesture'
train_filtered = train[train['behavior'] == 'Performs gesture']

# Aggregate statistical features per sequence
def extract_features(df):
    features = []
    for seq_id, seq_df in df.groupby('sequence_id'):
        feats = {'sequence_id': seq_id}
        for col in ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z', 
                    'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']:
            feats[f'{col}_mean'] = seq_df[col].mean()
            feats[f'{col}_std'] = seq_df[col].std()
            feats[f'{col}_min'] = seq_df[col].min()
            feats[f'{col}_max'] = seq_df[col].max()
        # TOF PRO features
        for sensor_num in range(1, 6):
            sensor_prefix = f"tof_{sensor_num}_"
            sensor_cols = [col for col in seq_df.columns if col.startswith(sensor_prefix)]
            feats[f"{sensor_prefix}mean_pixel"] = seq_df[sensor_cols].mean().mean()
            feats[f"{sensor_prefix}std_pixel"] = seq_df[sensor_cols].std().mean()
            feats[f"{sensor_prefix}neg1_pct"] = seq_df[sensor_cols].eq(-1).mean().mean()
        # Add gesture label
        feats['gesture'] = seq_df['gesture'].iloc[0]
        features.append(feats)
    return pd.DataFrame(features)

final_df_pro = extract_features(train_filtered)


#4 - LightGBM Model

# Prepare data
X = final_df_pro.drop(columns=['sequence_id', 'gesture'])
y = final_df_pro['gesture']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model
lgb_clf_pro = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(le.classes_),
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

# Import early_stopping callback
from lightgbm import early_stopping

# Train
lgb_clf_pro.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss',
    callbacks=[early_stopping(stopping_rounds=50)]
)

# Evaluate
y_pred = lgb_clf_pro.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(y_val, y_pred, average='macro')

print(f"LightGBM (TOF PRO) Accuracy: {accuracy:.4f}")
print(f"LightGBM (TOF PRO) Macro F1 Score: {macro_f1:.4f}")


#5 - GesturePredictor Class

class GesturePredictor:
    def __init__(self):
        self.model = lgb_clf_pro
        self.le = le
        self.features_to_use = X_train.columns.tolist()

    def predict(self, sequence_df: pd.DataFrame) -> str:
        feats = {}
        for col in ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z', 
                    'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']:
            feats[f'{col}_mean'] = sequence_df[col].mean()
            feats[f'{col}_std'] = sequence_df[col].std()
            feats[f'{col}_min'] = sequence_df[col].min()
            feats[f'{col}_max'] = sequence_df[col].max()
        for sensor_num in range(1, 6):
            sensor_prefix = f"tof_{sensor_num}_"
            sensor_cols = [col for col in sequence_df.columns if col.startswith(sensor_prefix)]
            feats[f"{sensor_prefix}mean_pixel"] = sequence_df[sensor_cols].mean().mean()
            feats[f"{sensor_prefix}std_pixel"] = sequence_df[sensor_cols].std().mean()
            feats[f"{sensor_prefix}neg1_pct"] = sequence_df[sensor_cols].eq(-1).mean().mean()
        feats_df = pd.DataFrame([feats])
        feats_df = feats_df[self.features_to_use]
        pred_probs = self.model.predict_proba(feats_df)
        pred_idx = pred_probs.argmax(axis=1)[0]
        pred_label = self.le.inverse_transform([pred_idx])[0]
        return pred_label


# Save model and LabelEncoder
import joblib

joblib.dump(lgb_clf_pro, '/kaggle/working/lgb_clf_pro.pkl')
joblib.dump(le, '/kaggle/working/label_encoder.pkl')

print("âœ… Model and LabelEncoder saved!")



# #6 - Kaggle Submission

# Generate submission.parquet
submission = pd.DataFrame({
    "sequence_id": test['sequence_id'].unique(),
    "gesture": [
        GesturePredictor().predict(test[test['sequence_id'] == seq_id])
        for seq_id in test['sequence_id'].unique()
    ]
})

print(submission.head())

# Save as required by the competition
submission.to_parquet('/kaggle/working/submission.parquet', index=False)
print("âœ… submission.parquet saved!")

