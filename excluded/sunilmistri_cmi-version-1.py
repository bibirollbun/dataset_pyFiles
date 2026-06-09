# Base on the Notebook 
# https://www.kaggle.com/code/jiaoyouzhang/cmi-2025-only-lightgbm


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import polars as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb
import os
import warnings
warnings.filterwarnings('ignore')

# Import evaluation API
import kaggle_evaluation.cmi_inference_server





# Load data
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test_demographics_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")
train_demographics_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# Filter to target sequences only (BFRB gestures)
train_df = train_df.loc[train_df['sequence_type'] == 'Target'].reset_index(drop=True)
print(f"Target sequences shape: {train_df.shape}")



# Analyze gesture distribution
print("\nGesture distribution:")
print(train_df['gesture'].value_counts())


# Define sensor columns
sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
all_sensor_cols = [col for col in train_df.columns if any(s in col for s in ['acc_', 'rot_', 'thm_', 'tof_'])]
print(f"\nTotal sensor columns: {len(all_sensor_cols)}")


le = LabelEncoder()
train_df['target'] = le.fit_transform(train_df['gesture'])
print(f"\nEncoded gestures: {len(le.classes_)} classes")
print("Classes:", le.classes_)






def create_features(df, demographic_df):

    # Create summary dataframe
    df=df.copy()
    summary = (df.groupby(['subject','sequence_id'])[sensor_cols].agg(['mean', 'std', 'min', 'max']))
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    #print( summary.shape)
    
    df_1 = summary.merge(demographic_df, on="subject", how="left")
    #print(df_1.shape)
    return df_1   


train_features_df = create_features(train_df, train_demographics_df)
train_info_df = train_df[['subject', 'sequence_id', 'target']].drop_duplicates()
#print(train_info_df.shape)
train_input_df= train_features_df.merge(train_info_df , on=['subject', 'sequence_id'], how='left')
#print(train_input_df.shape)


test_features_df = create_features(test_df, test_demographics_df)


# Prepare features and target
feature_cols = [col for col in train_input_df.columns 
                if col not in ['sequence_id', 'target', 'gesture', 'subject']]
X = train_input_df[feature_cols].fillna(-1)
y = train_input_df['target']

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")


def competition_metric(y_true, y_pred, le_instance, all_original_gestures):
    """
    Competition metric calculation
    """
    bfrb_gestures = [g for g in all_original_gestures if g in le_instance.classes_]
    
    # Binary F1: All are Target in this filtered dataset
    y_true_binary = np.ones_like(y_true, dtype=int)
    y_pred_binary = np.ones_like(y_pred, dtype=int)
    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary', pos_label=1, zero_division=0)
    
    # Macro F1: specific gesture classification
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    final_score = (binary_f1 + macro_f1) / 2
    return final_score, binary_f1, macro_f1


# Cross-validation setup
print("\nSetting up cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
models = []

all_original_gestures_in_train = train_df['gesture'].unique()

callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=100)]

# LightGBM model with cross-validation
print("\nTraining LightGBM models with cross-validation...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"\nFold {fold + 1}/5")
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # LightGBM model with GPU acceleration
    model = lgb.LGBMClassifier(
        objective='multiclass',
        n_estimators= 1000,
        learning_rate= 0.08,
        max_depth= 15,
        reg_alpha= 0.8,
        lambda_l2= 4.0,  
        num_leaves=31, 
        min_child_samples= 32,
        colsample_bytree= 0.3,
        subsample= 0.5,
        subsample_freq=0,
        cat_smooth=30.0,
        is_unbalance=True,
        max_bin=127,
        verbose=-1,  
        metric='multi_logloss',   
        #device='gpu',  
    )
    
    # Train model with verbose output
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],  
        eval_metric='multi_logloss',  
        callbacks=callbacks
    )
    
    # Predict
    y_pred_fold = model.predict(X_val_fold)
    
    # Calculate score
    score, binary_f1, macro_f1 = competition_metric(
        y_val_fold, y_pred_fold, le, all_original_gestures_in_train
    )
    
    cv_scores.append(score)
    models.append(model)
    
    print(f"Fold {fold + 1} - Competition Score: {score:.4f} (Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f})")

print(f"\nCross-validation results:")
print(f"Mean CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
print(f"Individual fold scores: {cv_scores}")

# Train final model on all data with GPU acceleration
print("\nTraining final model on all training data...")





# Prediction function for submission
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Prediction function for Kaggle evaluation
    """
    try:
        # Convert to pandas
        sequence_pd = sequence.to_pandas()
        demographics_pd = demographics.to_pandas()
        
        # Create features for this single sequence
        seq_features = create_features(sequence_pd, demographics_pd)

        # Prepare feature vector - ensure all expected features are present
        X_inference = seq_features[feature_cols].fillna(-1)
        
        # Predict using ensemble of CV models
        predictions = []
        for model in models:
            pred = model.predict(X_inference)
            # Ensure we get a scalar value
            if isinstance(pred, np.ndarray):
                pred = pred[0]
            predictions.append(int(pred))
        
        # Use majority vote or most confident prediction
        predicted_label_id = max(set(predictions), key=predictions.count)
        
        # Convert back to gesture string
        predicted_gesture_str = le.inverse_transform([predicted_label_id])[0]
        
        return predicted_gesture_str
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Return a default gesture if prediction fails
        return le.classes_[0]




# Test the prediction function
print("\nTesting prediction function...")
sample_sequence = train_df[train_df['sequence_id'] == train_df['sequence_id'].iloc[0]]
sample_demographics = train_demographics_df[train_demographics_df['subject'] == sample_sequence['subject'].iloc[0]]

sample_seq_pl = pl.from_pandas(sample_sequence)
sample_demo_pl = pl.from_pandas(sample_demographics)

test_prediction = predict(sample_seq_pl, sample_demo_pl)
actual_gesture = sample_sequence['gesture'].iloc[0]
print(f"Test prediction: {test_prediction}")
print(f"Actual gesture: {actual_gesture}")
print(f"Match: {test_prediction == actual_gesture}")


test_sequence =  pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test_demographics=  pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")

test_prediction = predict(test_sequence, test_demographics)
print(f"Test prediction: {test_prediction}")


# Setup inference server
print("\nSetting up inference server...")
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

