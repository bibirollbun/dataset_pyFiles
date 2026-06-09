import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import polars as pl
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Download the form and saved weights
print("--- Loading Saved Model and Artifacts ---")
final_lstm_model = load_model('/kaggle/input/cmi-ptmodels/CMI_PTModels_ver9/CMI_PTModels_ver9/trained_model.h5')
scaler = joblib.load('/kaggle/input/cmi-ptmodels/CMI_PTModels_ver9/CMI_PTModels_ver9/scaler.pkl')
label_encoders_multi_output = joblib.load('/kaggle/input/cmi-ptmodels/CMI_PTModels_ver9/CMI_PTModels_ver9/label_encoders.pkl')
global_mean_imputer_values = joblib.load('/kaggle/input/cmi-ptmodels/CMI_PTModels_ver9/CMI_PTModels_ver9/global_mean_imputer_values.pkl')
global_mode_imputer_values = joblib.load('/kaggle/input/cmi-ptmodels/CMI_PTModels_ver9/CMI_PTModels_ver9/global_mode_imputer_values.pkl')
imu_features = joblib.load('/kaggle/input/cmi-ptmodels/CMI_PTModels_ver9/CMI_PTModels_ver9/imu_features.pkl')
thm_features = joblib.load('/kaggle/input/cmi-ptmodels/CMI_PTModels_ver9/CMI_PTModels_ver9/thm_features.pkl')
tof_features = joblib.load('/kaggle/input/cmi-ptmodels/CMI_PTModels_ver9/CMI_PTModels_ver9/tof_features.pkl')
categorical_cols_for_ohe_seq = joblib.load('/kaggle/input/cmi-ptmodels/CMI_PTModels_ver9/CMI_PTModels_ver9/categorical_cols_for_ohe_seq.pkl')
numeric_feature_cols_after_ohe = joblib.load('/kaggle/input/cmi-ptmodels/CMI_PTModels_ver9/CMI_PTModels_ver9/numeric_feature_cols_after_ohe.pkl')

# Definition of global variables
sensor_cols_prefix = ['acc_', 'rot_', 'thm_', 'tof_']
max_sequence_length = 210
target_cols_multi_output = ['sequence_type', 'phase', 'gesture', 'orientation', 'behavior']
window_size = 5

# 6. Define prediction function for API - IMPORTANT: Apply Feature Engineering here too!
# ----------------------------------------------------------------------------------------------------
print("\n---6. Defining Prediction Function for API ---")

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    current_sequence_df = sequence.to_pandas()
    current_demographics_df = demographics.to_pandas()

    for col in current_sequence_df.select_dtypes(include=['float64']).columns:
        current_sequence_df[col] = current_sequence_df[col].astype(np.float32)
    for col in current_sequence_df.select_dtypes(include=['int64']).columns:
        if 'id' not in col and 'counter' not in col:
            current_sequence_df[col] = current_sequence_df[col].astype(np.int16)
        else:
            current_sequence_df[col] = current_sequence_df[col].astype(np.int32)
    if 'subject' in current_sequence_df.columns and 'subject' in current_demographics_df.columns:
        current_test_batch = current_sequence_df.merge(current_demographics_df, on='subject', how='left')
    else:
        current_test_batch = current_sequence_df.copy()
        for col in current_demographics_df.columns:
            if col != 'subject':
                current_test_batch[col] = current_demographics_df[col].iloc[0]

    sensor_columns_to_process_in_predict = [col for col in current_test_batch.columns if any(s in col for s in sensor_cols_prefix)]
    for col in sensor_columns_to_process_in_predict:
        if col in current_test_batch.columns:
            current_test_batch[col] = current_test_batch[col].replace(-1.0, np.nan).astype(np.float32)
    for col in current_test_batch.columns:
        if current_test_batch[col].isnull().any():
            if col in global_mode_imputer_values:
                current_test_batch[col] = current_test_batch[col].fillna(global_mode_imputer_values[col])
            elif col in global_mean_imputer_values:
                current_test_batch[col] = current_test_batch[col].fillna(global_mean_imputer_values[col]).astype(np.float32)
    
    # --- START FEATURE ENGINEERING FOR PREDICTION (Optimized & Enhanced) ---
    temp_features_df_predict = current_test_batch.copy()
    
    acc_rot_base_cols_predict = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    acc_rot_base_cols_predict = [col for col in acc_rot_base_cols_predict if col in temp_features_df_predict.columns]

    # Calculate first-order features
    temp_features_df_predict['acc_magnitude'] = np.sqrt(
        temp_features_df_predict['acc_x']**2 +
        temp_features_df_predict['acc_y']**2 +
        temp_features_df_predict['acc_z']**2).astype(np.float32)
    temp_features_df_predict['rot_magnitude'] = np.sqrt(
        temp_features_df_predict['rot_w']**2 +
        temp_features_df_predict['rot_x']**2 +
        temp_features_df_predict['rot_y']**2 +
        temp_features_df_predict['rot_z']**2).astype(np.float32)
    temp_features_df_predict['thm_1_diff'] = temp_features_df_predict.groupby('sequence_id')['thm_1'].diff().fillna(0).astype(np.float32)

    for col in acc_rot_base_cols_predict:
        temp_features_df_predict[f'{col}_diff_1'] = temp_features_df_predict.groupby('sequence_id')[col].diff(1).fillna(0).astype(np.float32)
        temp_features_df_predict[f'{col}_diff_2'] = temp_features_df_predict.groupby('sequence_id')[col].diff(2).fillna(0).astype(np.float32)
        
    for col in acc_rot_base_cols_predict:
        if col in temp_features_df_predict.columns:
            grouped_predict = temp_features_df_predict.groupby('sequence_id')
            temp_features_df_predict[f'{col}_roll_mean'] = grouped_predict[col].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean()).astype(np.float32)
            temp_features_df_predict[f'{col}_roll_std'] = grouped_predict[col].transform(lambda x: x.rolling(window=window_size, min_periods=1).std()).fillna(0).astype(np.float32)
            temp_features_df_predict[f'{col}_roll_min'] = grouped_predict[col].transform(lambda x: x.rolling(window=window_size, min_periods=1).min()).astype(np.float32)
            temp_features_df_predict[f'{col}_roll_max'] = grouped_predict[col].transform(lambda x: x.rolling(window=window_size, min_periods=1).max()).astype(np.float32)
            temp_features_df_predict[f'{col}_roll_median'] = grouped_predict[col].transform(lambda x: x.rolling(window=window_size, min_periods=1).median()).astype(np.float32)
            temp_features_df_predict[f'{col}_roll_skew'] = grouped_predict[col].transform(lambda x: x.rolling(window=window_size, min_periods=1).skew()).fillna(0).astype(np.float32)
            temp_features_df_predict[f'{col}_roll_kurt'] = grouped_predict[col].transform(lambda x: x.rolling(window=window_size, min_periods=1).kurt()).fillna(0).astype(np.float32)
            temp_features_df_predict[f'{col}_zero_cross_rate'] = grouped_predict[col].transform(lambda x: x.rolling(window=window_size, min_periods=1).apply(lambda y: np.sum(np.diff(np.sign(y)) != 0), raw=True)).fillna(0).astype(np.float32)
            temp_features_df_predict[f'{col}_diff1_roll_mean'] = grouped_predict[f'{col}_diff_1'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean()).astype(np.float32)
    
    # Feature Interactions
    if 'acc_x' in temp_features_df_predict.columns and 'acc_y' in temp_features_df_predict.columns:
        temp_features_df_predict['acc_x_acc_y_prod'] = (temp_features_df_predict['acc_x'] * temp_features_df_predict['acc_y']).astype(np.float32)
        temp_features_df_predict['acc_x_y_diff'] = (temp_features_df_predict['acc_x'] - temp_features_df_predict['acc_y']).astype(np.float32)
    if 'acc_x' in temp_features_df_predict.columns and 'acc_z' in temp_features_df_predict.columns:
        temp_features_df_predict['acc_x_acc_z_prod'] = (temp_features_df_predict['acc_x'] * temp_features_df_predict['acc_z']).astype(np.float32)
        temp_features_df_predict['acc_x_z_diff'] = (temp_features_df_predict['acc_x'] - temp_features_df_predict['acc_z']).astype(np.float32)
    if 'acc_y' in temp_features_df_predict.columns and 'acc_z' in temp_features_df_predict.columns:
        temp_features_df_predict['acc_y_acc_z_prod'] = (temp_features_df_predict['acc_y'] * temp_features_df_predict['acc_z']).astype(np.float32)
        temp_features_df_predict['acc_y_z_diff'] = (temp_features_df_predict['acc_y'] - temp_features_df_predict['acc_z']).astype(np.float32)
    if 'rot_x' in temp_features_df_predict.columns and 'rot_y' in temp_features_df_predict.columns:
        temp_features_df_predict['rot_x_rot_y_prod'] = (temp_features_df_predict['rot_x'] * temp_features_df_predict['rot_y']).astype(np.float32)
        temp_features_df_predict['rot_x_y_diff'] = (temp_features_df_predict['rot_x'] - temp_features_df_predict['rot_y']).astype(np.float32)
    if 'rot_x' in temp_features_df_predict.columns and 'rot_z' in temp_features_df_predict.columns:
        temp_features_df_predict['rot_x_rot_z_prod'] = (temp_features_df_predict['rot_x'] * temp_features_df_predict['rot_z']).astype(np.float32)
        temp_features_df_predict['rot_x_z_diff'] = (temp_features_df_predict['rot_x'] - temp_features_df_predict['rot_z']).astype(np.float32)
    if 'rot_y' in temp_features_df_predict.columns and 'rot_z' in temp_features_df_predict.columns:
        temp_features_df_predict['rot_y_rot_z_prod'] = (temp_features_df_predict['rot_y'] * temp_features_df_predict['rot_z']).astype(np.float32)
        temp_features_df_predict['rot_y_z_diff'] = (temp_features_df_predict['rot_y'] - temp_features_df_predict['rot_z']).astype(np.float32)
        
    # ToF Global Statistics
    for i in range(1, 6):
        tof_cols_i = [f'tof_{i}_v{j}' for j in range(64)]
        existing_tof_cols_i = [col for col in tof_cols_i if col in temp_features_df_predict.columns]
        if existing_tof_cols_i:
            temp_features_df_predict[f'tof_{i}_mean'] = temp_features_df_predict[existing_tof_cols_i].mean(axis=1).astype(np.float32)
            temp_features_df_predict[f'tof_{i}_std'] = temp_features_df_predict[existing_tof_cols_i].std(axis=1).fillna(0).astype(np.float32)

    # FFT Features
    fft_cols_to_process_predict = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']
    fft_cols_to_process_predict = [col for col in fft_cols_to_process_predict if col in current_test_batch.columns]
    fft_features_predict_list = []
    grouped_df_fft_predict = current_test_batch.groupby('sequence_id')
    for seq_id, group in grouped_df_fft_predict:
        fft_row = {'sequence_id': seq_id}
        for col in fft_cols_to_process_predict:
            signal = group[col].values
            if len(signal) < max_sequence_length:
                signal = np.pad(signal, (0, max_sequence_length - len(signal)), 'constant')
            elif len(signal) > max_sequence_length:
                signal = signal[:max_sequence_length]
            yf = np.fft.fft(signal)
            magnitude = np.abs(yf[1:max_sequence_length//2])
            if len(magnitude) > 0:
                fft_row[f'{col}_fft_mean_mag'] = np.mean(magnitude).astype(np.float32)
                fft_row[f'{col}_fft_std_mag'] = np.std(magnitude).astype(np.float32)
                fft_row[f'{col}_fft_max_mag'] = np.max(magnitude).astype(np.float32)
                fft_row[f'{col}_fft_peak_freq_idx'] = np.argmax(magnitude).astype(np.float32)
                fft_row[f'{col}_fft_total_energy'] = np.sum(magnitude**2).astype(np.float32)
                fft_row[f'{col}_fft_low_freq_energy'] = np.sum(magnitude[:len(magnitude)//4]**2).astype(np.float32)
                fft_row[f'{col}_fft_high_freq_energy'] = np.sum(magnitude[len(magnitude)//2:]**2).astype(np.float32)
            else:
                fft_row.update({f'{col}_fft_mean_mag': 0.0, f'{col}_fft_std_mag': 0.0, f'{col}_fft_max_mag': 0.0,
                                f'{col}_fft_peak_freq_idx': 0.0, f'{col}_fft_total_energy': 0.0,
                                f'{col}_fft_low_freq_energy': 0.0, f'{col}_fft_high_freq_energy': 0.0})
        fft_features_predict_list.append(fft_row)
    fft_df_predict = pd.DataFrame(fft_features_predict_list)
    temp_features_df_predict = pd.merge(temp_features_df_predict, fft_df_predict, on='sequence_id', how='left')
    
    # --- END FEATURE ENGINEERING FOR PREDICTION (Optimized & Enhanced) ---
    
    for col in categorical_cols_for_ohe_seq:
        temp_dummies = pd.get_dummies(temp_features_df_predict[col], prefix=col)
        expected_dummy_cols = [c for c in numeric_feature_cols_after_ohe if c.startswith(f"{col}_")]
        for expected_col in expected_dummy_cols:
            if expected_col not in temp_dummies.columns:
                temp_dummies[expected_col] = 0.0
        temp_dummies = temp_dummies[expected_dummy_cols]
        temp_features_df_predict = pd.concat([temp_features_df_predict.drop(columns=[col]), temp_dummies], axis=1)

    # Ensure all columns exist before scaling and re-ordering
    for col in numeric_feature_cols_after_ohe:
        if col not in temp_features_df_predict.columns:
            temp_features_df_predict[col] = 0.0
            
    # Scaling
    temp_features_df_predict[numeric_feature_cols_after_ohe] = scaler.transform(temp_features_df_predict[numeric_feature_cols_after_ohe]).astype(np.float32)
    
    def prepare_input_matrix(df, feature_list, max_seq_len):
        current_matrix = df[feature_list].values.astype(np.float32)
        if current_matrix.shape[0] < max_seq_len:
            padding_needed = max_seq_len - current_matrix.shape[0]
            padding_matrix = np.zeros((padding_needed, current_matrix.shape[1]), dtype=np.float32)
            current_matrix = np.vstack((current_matrix, padding_matrix))
        elif current_matrix.shape[0] > max_seq_len:
            current_matrix = current_matrix[:max_seq_len, :]
        return np.expand_dims(current_matrix, axis=0)

    imu_input_tensor = prepare_input_matrix(temp_features_df_predict, imu_features, max_sequence_length)
    thm_input_tensor = prepare_input_matrix(temp_features_df_predict, thm_features, max_sequence_length)
    tof_input_tensor = prepare_input_matrix(temp_features_df_predict, tof_features, max_sequence_length)

    predictions = final_lstm_model.predict([imu_input_tensor, thm_input_tensor, tof_input_tensor])
    gesture_preds = predictions[target_cols_multi_output.index('gesture')]
    predicted_gesture_index = np.argmax(gesture_preds, axis=1)[0]
    predicted_gesture_label = label_encoders_multi_output['gesture'].inverse_transform([predicted_gesture_index])[0]
    return predicted_gesture_label

print("Prediction function for API defined.")

# 7. Setup and Run Server
# ----------------------------------------------------------------------------------------------------
print("\n---7. Setting up and Running Server ---")
import kaggle_evaluation.cmi_inference_server
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print("Running server in official evaluation mode...")
    inference_server.serve()
else:
    print("Running server locally for testing.")
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )




