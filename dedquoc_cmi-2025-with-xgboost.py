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


%%time
import os

import pandas as pd
import polars as pl
import xgboost as xgb
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats   
import warnings
warnings.filterwarnings('ignore')
import kaggle_evaluation.cmi_inference_server


train_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
demo_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
test_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv"
test_demo_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"

# Global model state
model_ges = None
le_ges = None
feature_columns = None
_model_trained = False

# Paths (for local testing)
TRAIN_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
DEMO_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"


%%time

# Load training and test data
train = pd.read_csv(train_path)
train_demo = pd.read_csv(demo_path)
test = pd.read_csv(test_path)
test_demo = pd.read_csv(test_demo_path)

# Confirm loaded successfully
print(train.shape, train_demo.shape)
print(test.shape, test_demo.shape)


test.columns


test.head(10)


test_demo.columns


test_demo.head(10)


%%time
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))
sns.countplot(data=train, x='gesture', order=train['gesture'].value_counts().index)
plt.xticks(rotation=90)
plt.title("Distribution of Gesture Labels")
plt.show()


%%time
# Behavior
plt.figure(figsize=(10, 4))
sns.countplot(data=train, y='behavior', order=train['behavior'].value_counts().index)
plt.title('Behavior Distribution')
plt.show()

# Gesture
plt.figure(figsize=(10, 6))
sns.countplot(data=train, y='gesture', order=train['gesture'].value_counts().index)
plt.title('Gesture Distribution')
plt.show()


%%time
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
train['gesture_label'] = label_encoder.fit_transform(train['gesture'])
gesture_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label mapping:\n", gesture_mapping)


%%time
train = train.merge(train_demo, on='subject', how='left')
test = test.merge(test_demo, on='subject', how='left')


print(train.columns.tolist())


%%time
drop_cols = ['row_id', 'sequence_id', 'gesture', 'subject']
existing_drop_cols = [col for col in drop_cols if col in train.columns]

X = train.drop(columns=existing_drop_cols + ['gesture_label'])
y = train['gesture_label']

existing_test_drop_cols = [col for col in drop_cols if col in test.columns]
X_test = test.drop(columns=existing_test_drop_cols)


%%time
from sklearn.model_selection import train_test_split

# Drop non-numeric and label columns
features_to_drop = ['row_id', 'sequence_id', 'gesture', 'subject']
X = train.drop(columns=features_to_drop, errors='ignore')
y = train['gesture']

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_valid, y_train, y_valid = train_test_split(
    X.select_dtypes(include=['int', 'float', 'bool']),
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


%%time
from xgboost import XGBClassifier

# Define the model with GPU support
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False,
    tree_method='auto', # Specify the 
    device ='cuda',
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=2,  # or 1 if you want to be gentler on CPU,
    #enable_categorical=True
)

# Fit the model
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=15,
    verbose=True
)


%%time
import pandas as pd
import numpy as np

# Load or use already trained model
# model = ... (your trained model)

# Preprocessing steps you applied during training
def preprocess(sample_df):
    # Example: drop unused columns, encode categorical variables, fill missing values
    X = sample_df.copy()

    # Drop columns not used during training
    drop_cols = ['row_id', 'sequence_id']  # adjust based on your training
    X = X.drop(columns=[col for col in drop_cols if col in X.columns], errors='ignore')

    # Encode categoricals if needed (must match training)
    # For example: X['subject'] = subject_encoder.transform(X['subject'])

    # Fill missing values or use other preprocessing
    X = X.fillna(0)

    return X

# Final prediction function

def predict(sample: pd.DataFrame, sample_metadata: pd.DataFrame) -> np.ndarray:
    # Combine or use just sample if you don't need metadata
    X = preprocess(sample)  # your preprocessing function
    pred = model.predict(X)
    return np.array(pred)



%%time
from sklearn.metrics import classification_report, confusion_matrix

y_val_pred = model.predict(X_valid)
print(confusion_matrix(y_valid, y_val_pred))
print(classification_report(y_valid, y_val_pred))


import joblib
joblib.dump(model, "model.pkl")   


%%time
def extract_features_polars(sequence: pl.DataFrame) -> pl.DataFrame:
    """
    Extract features from one sequence (filtered to gesture phase)
    Returns a 1-row Polars DataFrame
    """
    feats = {}

    # Filter to gesture phase using behavior or phase
    if 'behavior' in sequence.columns:
        valid_rows = sequence.filter(pl.col('behavior') == 'Active')
        if len(valid_rows) == 0:
            valid_rows = sequence.filter(pl.col('behavior').str.to_lowercase() == 'active')
    elif 'phase' in sequence.columns:
        valid_rows = sequence.filter(pl.col('phase') == 2)
    else:
        valid_rows = sequence  # fallback

    if len(valid_rows) == 0:
        valid_rows = sequence

    df = valid_rows.to_pandas()

    # IMU: acc_x, acc_y, acc_z
    for ax in 'xyz':
        col = f'acc_{ax}'
        if col in df:
            x = df[col].dropna().values
            if len(x) == 0: x = [0]
            feats[f'{col}_mean'] = np.mean(x)
            feats[f'{col}_std'] = np.std(x)
            feats[f'{col}_max'] = np.max(x)
            feats[f'{col}_min'] = np.min(x)
            feats[f'{col}_kurt'] = stats.kurtosis(x) if len(x) > 2 else 0
            feats[f'{col}_skew'] = stats.skew(x) if len(x) > 2 else 0
            feats[f'{col}_amp'] = np.max(x) - np.min(x)
            feats[f'{col}_jerk'] = np.std(np.diff(x)) if len(x) > 1 else 0

    # Rotation
    for ax in 'wxyz':
        col = f'rot_{ax}'
        if col in df:
            x = df[col].dropna().values
            if len(x) == 0: x = [0]
            feats[f'{col}_mean'] = np.mean(x)
            feats[f'{col}_std'] = np.std(x)

    # Thermopiles
    for i in range(1, 6):
        col = f'thm_{i}'
        if col in df:
            x = df[col].dropna().values
            feats[f'thm_{i}_mean'] = np.mean(x) if len(x) > 0 else 0
            feats[f'thm_{i}_std'] = np.std(x) if len(x) > 0 else 0

    # ToF
    for i in range(1, 6):
        tof_cols = [f'tof_{i}_v{j}' for j in range(64) if f'tof_{i}_v{j}' in df.columns]
        if tof_cols:
            vals = df[tof_cols].values.flatten()
            valid = vals[(vals != -1) & (~np.isnan(vals))]
            feats[f'tof_{i}_mean'] = np.mean(valid) if len(valid) > 0 else 0
            feats[f'tof_{i}_std'] = np.std(valid) if len(valid) > 0 else 0

    return pl.DataFrame([feats])  


%%time
def train_model_on_first_call(train_path: str, demo_path: str):
    global model_ges, le_ges, feature_columns, _model_trained

    if _model_trained:
        return

    print("ğŸ”� Starting model training on first call...")

    # Load data
    try:
        train = pl.read_csv(train_path)
        demo = pl.read_csv(demo_path)
    except Exception as e:
        print(f"â�Œ Failed to load data: {e}")
        return

    print(f"Total train rows: {len(train)}")
    if 'behavior' in train.columns:
        print("Unique behavior values:", train['behavior'].unique().to_list())
    if 'phase' in train.columns:
        print("Unique phase values:", train['phase'].unique().to_list())

    # Filter to gesture phase
    if 'behavior' in train.columns:
        train_gesture = train.filter(
            (pl.col('behavior') == 'Active') | 
            (pl.col('behavior').str.to_lowercase() == 'active')
        )
    elif 'phase' in train.columns:
        train_gesture = train.filter(pl.col('phase') == 2)
    else:
        print("â�Œ No behavior or phase column â€” using all data as fallback")
        train_gesture = train

    print(f"Rows after gesture filtering: {len(train_gesture)}")

    if len(train_gesture) == 0:
        print("â�Œ No gesture-phase data found. Cannot train.")
        # Fallback: predict constant
        le_ges = LabelEncoder()
        le_ges.fit(['non_target'])
        model_ges = lambda x: 0
        feature_columns = ['acc_x_mean']
        _model_trained = True
        return

    # Merge with demographics
    train_gesture = train_gesture.join(demo, on='subject', how='left')

    # Get unique sequence IDs
    sequence_ids = train_gesture['sequence_id'].unique().to_list()
    print(f"Number of sequences to process: {len(sequence_ids)}")

    feature_list = []
    labels = []

    for seq_id in sequence_ids:
        seq_data = train_gesture.filter(pl.col('sequence_id') == seq_id)
        if len(seq_data) == 0:
            continue

        feats = extract_features_polars(seq_data)
        if len(feats) == 0 or len(feats.columns) == 0:
            continue
        feature_list.append(feats)

        # Extract label
        gesture_series = seq_data['gesture'].drop_nulls()
        if len(gesture_series) > 0:
            labels.append(gesture_series[0])
        else:
            labels.append('non_target')

    # Check if we have any features
    if len(feature_list) == 0:
        print("â�Œ No features extracted. Using fallback model.")
        le_ges = LabelEncoder()
        le_ges.fit(['non_target'])
        model_ges = lambda x: 0
        feature_columns = ['dummy_feature']
        _model_trained = True
        return

    # Combine all features
    X = pl.concat(feature_list).fill_null(0).fill_nan(0)
    y = pl.Series("gesture", labels).drop_nulls()

    if len(X) == 0 or len(y) == 0:
        print("â�Œ No valid labels or features. Using fallback.")
        le_ges = LabelEncoder()
        le_ges.fit(['non_target'])
        model_ges = lambda x: 0
        feature_columns = ['dummy']
        _model_trained = True
        return

    # Align X and y
    min_len = min(len(X), len(y))
    X_pd = X[:min_len].to_pandas()
    y_pd = y[:min_len].to_list()

    # Encode
    le_ges = LabelEncoder()
    y_enc = le_ges.fit_transform(y_pd)

    # Train XGBoost
    print(f"âœ… Training on {len(X_pd)} sequences with {len(le_ges.classes_)} classes")
    model_ges = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        random_state=42,
        eval_metric='mlogloss'
    )
    model_ges.fit(X_pd, y_enc)

    feature_columns = X_pd.columns.tolist()
    _model_trained = True
    print("ğŸ�‰ Model training complete!")   


%%time
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    global model_ges, le_ges, feature_columns, _model_trained

    # Train model on first call if not already trained
    if not _model_trained:
        print("ğŸ�¯ First prediction: training model...")
        train_model_on_first_call(TRAIN_PATH, DEMO_PATH)

    # Extract features from current sequence
    try:
        feats = extract_features_polars(sequence)
    except Exception as e:
        print(f"â�Œ Feature extraction failed: {e}")
        feats = pl.DataFrame([{}])  # empty fallback

    # Merge with demographics
    try:
        subject = sequence['subject'].unique()[0]
        demo_row = demographics.filter(pl.col('subject') == subject)
        if len(demo_row) == 0:
            # Create empty demo row with zeros
            demo_data = {
                'subject': [subject],
                'adult_child': [0],
                'age': [30],
                'sex': [0],
                'handedness': [1],
                'height_cm': [170],
                'shoulder_to_wrist_cm': [60],
                'elbow_to_wrist_cm': [35]
            }
            demo_row = pl.DataFrame(demo_data)
        feats = feats.join(demo_row, on='subject', how='left')
    except Exception as e:
        print(f"âš ï¸� Demographics merge failed: {e}")
        # Add subject columns as zero
        for col in ['adult_child', 'age', 'sex', 'handedness', 
                   'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']:
            if col not in feats.columns:
                feats = feats.with_columns(pl.lit(0).cast(pl.Float64).alias(col))

    # Fill missing values
    feats = feats.fill_null(0).fill_nan(0)

    # Ensure all expected features are present
    for col in feature_columns:
        if col not in feats.columns:
            feats = feats.with_columns(pl.lit(0).cast(pl.Float64).alias(col))
    
    # Select and sort columns to match training
    feats = feats.select(sorted(feature_columns))

    # Convert to pandas for prediction
    try:
        X = feats.to_pandas()
        pred_class = model_ges.predict(X)[0]
        predicted_gesture = le_ges.inverse_transform([pred_class])[0]
    except Exception as e:
        print(f"â�Œ Prediction failed: {e}")
        predicted_gesture = 'Text on phone'  # safe fallback

    return predicted_gesture   


%%time
import kaggle_evaluation.cmi_inference_server

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

