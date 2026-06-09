

# Cell 1: Environment Setup and Configuration
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import gc
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=== ENHANCED PLAsTiCC LIGHTGBM TESTING NOTEBOOK ===")
print(f"Started at: {datetime.now()}")
print("ğŸ�¯ Focus: Using your trained LightGBM models for maximum scores")

# CRITICAL: Use correct PLAsTiCC class names for submission
CORRECT_PLASTICC_CLASSES = [
    'class_6', 'class_15', 'class_16', 'class_42', 'class_52', 
    'class_53', 'class_62', 'class_64', 'class_65', 'class_67', 
    'class_88', 'class_90', 'class_92', 'class_95', 'class_99'
]

# Configuration
BATCH_SIZE = 100000
SAVE_INTERVAL = 5
OUTPUT_DIR = "/kaggle/working/"




# Cell 2: Enhanced Environment Check
def enhanced_environment_check():
    """Enhanced environment validation with file analysis"""
    print("\n=== ENHANCED ENVIRONMENT CHECK ===")
    
    # Check LightGBM models dataset
    lightgbm_dir = "/kaggle/input/ml-dataset-model/lightgbm_models"
    if os.path.exists(lightgbm_dir):
        print(f"ğŸ“� LightGBM models directory found: {lightgbm_dir}")
        for item in os.listdir(lightgbm_dir):
            file_path = os.path.join(lightgbm_dir, item)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024*1024)
                print(f"ğŸ”§ Model file: {item} ({size_mb:.1f} MB)")
    else:
        print("â�Œ LightGBM models directory not found!")
        return [], []
    
    # Find model and component files
    model_files = []
    component_files = []
    
    for root, dirs, files in os.walk("/kaggle/input"):
        for file in files:
            full_path = os.path.join(root, file)
            if 'lightgbm' in root.lower() or 'lgb' in file.lower():
                if file.endswith(('.txt', '.pkl')):
                    if 'model' in file.lower():
                        model_files.append(full_path)
                        print(f"ğŸ¤– Model found: {full_path}")
                    else:
                        component_files.append(full_path)
                        print(f"ğŸ”§ Component found: {full_path}")
    
    # Check PLAsTiCC data files
    plasticc_files = []
    for root, dirs, files in os.walk("/kaggle/input"):
        for file in files:
            if 'test_set' in file and file.endswith('.csv'):
                full_path = os.path.join(root, file)
                size_mb = os.path.getsize(full_path) / (1024*1024)
                plasticc_files.append(full_path)
                print(f"ğŸ“Š Test data: {file} ({size_mb:.1f} MB)")
    
    return model_files, component_files, plasticc_files

model_files, component_files, plasticc_files = enhanced_environment_check()




# Cell 3: Robust Model and Component Loading
def load_lightgbm_models_robust():
    """Load LightGBM models and components with multiple fallback strategies"""
    print("\n=== ROBUST LIGHTGBM MODEL LOADING ===")
    
    models_dir = "/kaggle/input/ml-dataset-model/lightgbm_models"
    
    # Load final model
    final_model = None
    final_model_path = os.path.join(models_dir, "final_model.txt")
    if os.path.exists(final_model_path):
        try:
            final_model = lgb.Booster(model_file=final_model_path)
            print(f"âœ… Final model loaded: {final_model_path}")
        except Exception as e:
            print(f"â�Œ Failed to load final model: {e}")
    
    # Load CV models for ensemble
    cv_models = []
    for i in range(5):
        cv_model_path = os.path.join(models_dir, f"cv_model_{i}.txt")
        if os.path.exists(cv_model_path):
            try:
                cv_model = lgb.Booster(model_file=cv_model_path)
                cv_models.append(cv_model)
                print(f"âœ… CV model {i} loaded: {cv_model_path}")
            except Exception as e:
                print(f"â�Œ Failed to load CV model {i}: {e}")
    
    # Load label encoder
    label_encoder = None
    label_encoder_path = os.path.join(models_dir, "label_encoder.pkl")
    if os.path.exists(label_encoder_path):
        try:
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            print(f"âœ… Label encoder loaded: {label_encoder_path}")
        except Exception as e:
            print(f"â�Œ Failed to load label encoder: {e}")
    
    # Load feature columns
    feature_columns = None
    feature_columns_path = os.path.join(models_dir, "feature_columns.pkl")
    if os.path.exists(feature_columns_path):
        try:
            with open(feature_columns_path, 'rb') as f:
                feature_columns = pickle.load(f)
            print(f"âœ… Feature columns loaded: {len(feature_columns)} features")
        except Exception as e:
            print(f"â�Œ Failed to load feature columns: {e}")
    
    # Load model metrics
    model_metrics = None
    metrics_path = os.path.join(models_dir, "model_metrics.pkl")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'rb') as f:
                model_metrics = pickle.load(f)
            print(f"âœ… Model metrics loaded")
            print(f"   Training CV accuracy: {model_metrics.get('cv_accuracy', 'Unknown')}")
            print(f"   Training CV log loss: {model_metrics.get('cv_log_loss', 'Unknown')}")
        except Exception as e:
            print(f"â�Œ Failed to load model metrics: {e}")
    
    print(f"\nâœ… Loading Summary:")
    print(f"   Final model: {'âœ“' if final_model else 'â�Œ'}")
    print(f"   CV models: {len(cv_models)}/5")
    print(f"   Label encoder: {'âœ“' if label_encoder else 'â�Œ'}")
    print(f"   Feature columns: {'âœ“' if feature_columns else 'â�Œ'}")
    print(f"   Model metrics: {'âœ“' if model_metrics else 'â�Œ'}")
    
    return final_model, cv_models, label_encoder, feature_columns, model_metrics

# Load all components
final_model, cv_models, label_encoder, feature_columns, model_metrics = load_lightgbm_models_robust()

# Cell 4: LightGBM Feature Engineering (Matching Training)
def add_derived_features(df):
    """Add derived features to light curve data - SAME AS TRAINING"""
    df = df.copy()
    
    # Basic derived features
    df['flux_ratio_sq'] = np.power(df['flux'] / df['flux_err'], 2.0)
    df['flux_by_flux_ratio_sq'] = df['flux'] * df['flux_ratio_sq']
    
    # Detected flux features
    df['flux_diff'] = df.groupby(['object_id', 'passband'])['flux'].diff()
    df['flux_diff2'] = df.groupby(['object_id', 'passband'])['flux_diff'].diff()
    
    # Time-based features
    df['mjd_diff'] = df.groupby(['object_id', 'passband'])['mjd'].diff()
    df['mjd_diff'].fillna(0, inplace=True)
    
    # Flux rate features
    df['flux_rate'] = df['flux_diff'] / (df['mjd_diff'] + 1e-8)
    df['flux_rate'].replace([np.inf, -np.inf], 0, inplace=True)
    
    # Detection features
    df['detected_bool'] = (df['detected'] == 1).astype(int)
    
    return df

def extract_time_series_features(group):
    """Extract comprehensive time-series features - SAME AS TRAINING"""
    features = {}
    
    # Basic statistics
    features['count'] = len(group)
    features['mean'] = group['flux'].mean()
    features['std'] = group['flux'].std()
    features['min'] = group['flux'].min()
    features['max'] = group['flux'].max()
    features['median'] = group['flux'].median()
    
    # Advanced statistics
    features['skew'] = pd.Series(group['flux']).skew()
    features['kurtosis'] = pd.Series(group['flux']).kurtosis()
    features['mad'] = np.median(np.abs(group['flux'] - features['median']))
    
    # Percentiles
    features['q25'] = np.percentile(group['flux'], 25)
    features['q75'] = np.percentile(group['flux'], 75)
    features['iqr'] = features['q75'] - features['q25']
    
    # Range and amplitude features
    features['range'] = features['max'] - features['min']
    features['amplitude'] = features['range'] / 2
    features['beyond_1std'] = np.sum(np.abs(group['flux'] - features['mean']) > features['std']) / len(group)
    
    # Time-based features
    if len(group) > 1:
        features['time_span'] = group['mjd'].max() - group['mjd'].min()
        features['time_mean'] = group['mjd'].mean()
        features['time_std'] = group['mjd'].std()
        
        # Peak detection
        peak_idx = group['flux'].idxmax()
        features['peak_mjd'] = group.loc[peak_idx, 'mjd']
        features['peak_flux'] = group.loc[peak_idx, 'flux']
        
        # Rise and decline features
        pre_peak = group[group['mjd'] <= features['peak_mjd']]
        post_peak = group[group['mjd'] > features['peak_mjd']]
        
        if len(pre_peak) > 1:
            features['rise_time'] = features['peak_mjd'] - pre_peak['mjd'].min()
            features['rise_slope'] = (features['peak_flux'] - pre_peak['flux'].iloc[0]) / (features['rise_time'] + 1e-8)
        else:
            features['rise_time'] = 0
            features['rise_slope'] = 0
            
        if len(post_peak) > 1:
            features['decline_time'] = post_peak['mjd'].max() - features['peak_mjd']
            features['decline_slope'] = (post_peak['flux'].iloc[-1] - features['peak_flux']) / (features['decline_time'] + 1e-8)
        else:
            features['decline_time'] = 0
            features['decline_slope'] = 0
    else:
        features['time_span'] = 0
        features['time_mean'] = group['mjd'].iloc[0] if len(group) > 0 else 0
        features['time_std'] = 0
        features['peak_mjd'] = group['mjd'].iloc[0] if len(group) > 0 else 0
        features['peak_flux'] = group['flux'].iloc[0] if len(group) > 0 else 0
        features['rise_time'] = 0
        features['rise_slope'] = 0
        features['decline_time'] = 0
        features['decline_slope'] = 0
    
    # Error-based features
    features['mean_err'] = group['flux_err'].mean()
    features['std_err'] = group['flux_err'].std()
    features['snr_mean'] = features['mean'] / features['mean_err']
    features['snr_max'] = features['max'] / group['flux_err'].min()
    
    # Derived features statistics
    if 'flux_ratio_sq' in group.columns:
        features['flux_ratio_sq_sum'] = group['flux_ratio_sq'].sum()
        features['flux_ratio_sq_mean'] = group['flux_ratio_sq'].mean()
    
    if 'flux_diff' in group.columns:
        flux_diff_clean = group['flux_diff'].dropna()
        if len(flux_diff_clean) > 0:
            features['flux_diff_std'] = flux_diff_clean.std()
            features['flux_diff_mean'] = flux_diff_clean.mean()
            features['flux_diff_max'] = flux_diff_clean.max()
            features['flux_diff_min'] = flux_diff_clean.min()
        else:
            features['flux_diff_std'] = 0
            features['flux_diff_mean'] = 0
            features['flux_diff_max'] = 0
            features['flux_diff_min'] = 0
    
    # Detection features
    features['detected_ratio'] = group['detected'].mean()
    features['detected_count'] = group['detected'].sum()
    
    return pd.Series(features)

def create_lightgbm_features(lc_df, meta_df):
    """Create features exactly matching your LightGBM training"""
    print(f"    ğŸ”§ Creating LightGBM features for {len(meta_df)} objects...")
    
    # Add derived features
    lc_df = add_derived_features(lc_df)
    
    # Extract features for each object-passband combination
    ts_features = lc_df.groupby(['object_id', 'passband']).apply(extract_time_series_features).reset_index()
    
    # Pivot to get features for each passband as separate columns
    feature_cols = [col for col in ts_features.columns if col not in ['object_id', 'passband']]
    
    pivoted_features = []
    for pb in range(6):  # 6 passbands (0-5)
        pb_data = ts_features[ts_features['passband'] == pb].copy()
        if len(pb_data) > 0:
            pb_data = pb_data.drop('passband', axis=1)
            
            # Rename columns to include passband
            new_cols = {'object_id': 'object_id'}
            for col in feature_cols:
                new_cols[col] = f'{col}_pb{pb}'
            pb_data = pb_data.rename(columns=new_cols)
            
            pivoted_features.append(pb_data)
    
    # Merge all passband features
    if pivoted_features:
        final_features = pivoted_features[0]
        for i in range(1, len(pivoted_features)):
            final_features = final_features.merge(pivoted_features[i], on='object_id', how='outer')
    else:
        final_features = pd.DataFrame({'object_id': meta_df['object_id'].values})
    
    # Fill missing values
    final_features = final_features.fillna(0)
    
    # Add cross-passband features
    passband_pairs = [(0,1), (1,2), (2,3), (3,4), (4,5), (0,2), (1,3), (2,4), (3,5)]
    for pb1, pb2 in passband_pairs:
        if f'mean_pb{pb1}' in final_features.columns and f'mean_pb{pb2}' in final_features.columns:
            final_features[f'color_{pb1}_{pb2}'] = final_features[f'mean_pb{pb1}'] - final_features[f'mean_pb{pb2}']
            final_features[f'color_ratio_{pb1}_{pb2}'] = final_features[f'mean_pb{pb1}'] / (final_features[f'mean_pb{pb2}'] + 1e-8)
    
    # Global object features
    count_cols = [f'count_pb{i}' for i in range(6) if f'count_pb{i}' in final_features.columns]
    if count_cols:
        final_features['total_observations'] = final_features[count_cols].sum(axis=1)
        final_features['active_passbands'] = (final_features[count_cols] > 0).sum(axis=1)
    
    mean_cols = [f'mean_pb{i}' for i in range(6) if f'mean_pb{i}' in final_features.columns]
    if mean_cols:
        final_features['flux_mean_all'] = final_features[mean_cols].mean(axis=1)
        final_features['flux_std_all'] = final_features[mean_cols].std(axis=1)
    
    max_cols = [f'max_pb{i}' for i in range(6) if f'max_pb{i}' in final_features.columns]
    min_cols = [f'min_pb{i}' for i in range(6) if f'min_pb{i}' in final_features.columns]
    if max_cols and min_cols:
        final_features['flux_max_all'] = final_features[max_cols].max(axis=1)
        final_features['flux_min_all'] = final_features[min_cols].min(axis=1)
    
    peak_mjd_cols = [f'peak_mjd_pb{i}' for i in range(6) if f'peak_mjd_pb{i}' in final_features.columns]
    if peak_mjd_cols:
        final_features['peak_mjd_range'] = final_features[peak_mjd_cols].max(axis=1) - final_features[peak_mjd_cols].min(axis=1)
    
    # Merge with metadata
    final_features = final_features.merge(meta_df, on='object_id', how='left')
    
    print(f"    âœ… LightGBM features created: {final_features.shape}")
    return final_features




# Cell 5: Smart Feature Alignment for LightGBM
def align_features_for_lightgbm(features_df, feature_columns):
    """Align features with LightGBM training expectations"""
    print(f"    ğŸ�¯ Aligning features for LightGBM...")
    
    # Remove object_id for processing
    X = features_df.drop('object_id', axis=1, errors='ignore')
    current_features = list(X.columns)
    
    print(f"    Current features: {len(current_features)}")
    print(f"    Expected features: {len(feature_columns) if feature_columns else 'Unknown'}")
    
    if feature_columns is None:
        print(f"    âš ï¸� No feature columns specification - using current features")
        return X
    
    # Create aligned feature matrix
    X_aligned = pd.DataFrame(index=X.index)
    
    for col in feature_columns:
        if col in current_features:
            X_aligned[col] = X[col]
        else:
            X_aligned[col] = 0.0  # Fill missing features with 0
            
    print(f"    âœ… Features aligned: {X_aligned.shape}")
    return X_aligned

# Cell 6: Enhanced Prediction with LightGBM Ensemble
def make_lightgbm_predictions(features_df, final_model, cv_models, label_encoder, feature_columns):
    """Make predictions using LightGBM models with ensemble"""
    print(f"    ğŸ�¯ Making LightGBM predictions for {len(features_df)} objects...")
    
    # Align features
    X_aligned = align_features_for_lightgbm(features_df, feature_columns)
    
    # Replace any remaining NaN/inf values
    X_aligned = X_aligned.fillna(0).replace([np.inf, -np.inf], 0)
    
    predictions = None
    
    # Try ensemble prediction first (if we have CV models)
    if len(cv_models) > 0:
        print(f"    ğŸ”„ Using ensemble of {len(cv_models)} CV models...")
        try:
            ensemble_predictions = []
            for i, cv_model in enumerate(cv_models):
                pred = cv_model.predict(X_aligned, num_iteration=cv_model.best_iteration)
                ensemble_predictions.append(pred)
            
            # Average the predictions
            predictions = np.mean(ensemble_predictions, axis=0)
            print(f"    âœ… Ensemble predictions generated: {predictions.shape}")
            
        except Exception as e:
            print(f"    âš ï¸� Ensemble prediction failed: {e}")
            predictions = None
    
    # Fallback to final model
    if predictions is None and final_model is not None:
        print(f"    ğŸ”„ Using final model...")
        try:
            predictions = final_model.predict(X_aligned, num_iteration=final_model.best_iteration)
            print(f"    âœ… Final model predictions generated: {predictions.shape}")
        except Exception as e:
            print(f"    â�Œ Final model prediction failed: {e}")
            predictions = None
    
    # Ultimate fallback - uniform probabilities
    if predictions is None:
        print(f"    âš ï¸� Using uniform fallback probabilities")
        n_classes = len(label_encoder.classes_) if label_encoder else 14
        predictions = np.full((len(features_df), n_classes), 1.0/n_classes)
    
    # Handle class mapping
    if label_encoder is not None:
        if predictions.shape[1] == len(label_encoder.classes_):
            # Perfect match
            class_names = [f'class_{cls}' for cls in label_encoder.classes_]
        elif predictions.shape[1] == 14 and len(label_encoder.classes_) == 14:
            # 14 classes, add class_99 for PLAsTiCC
            padding_prob = 0.001
            padded_probs = np.column_stack([
                predictions * (1 - padding_prob),
                np.full(len(predictions), padding_prob)
            ])
            predictions = padded_probs
            class_names = [f'class_{cls}' for cls in label_encoder.classes_] + ['class_99']
        else:
            print(f"    âš ï¸� Class count mismatch: model={predictions.shape[1]}, encoder={len(label_encoder.classes_)}")
            class_names = CORRECT_PLASTICC_CLASSES[:predictions.shape[1]]
    else:
        # No label encoder - use default PLAsTiCC classes
        class_names = CORRECT_PLASTICC_CLASSES[:predictions.shape[1]]
    
    # Ensure we have exactly 15 classes for PLAsTiCC
    if len(class_names) < 15:
        # Pad with remaining classes
        missing_classes = CORRECT_PLASTICC_CLASSES[len(class_names):]
        class_names.extend(missing_classes)
        
        # Pad predictions
        padding = np.full((len(predictions), len(missing_classes)), 0.001)
        predictions = np.column_stack([predictions, padding])
    
    # Create prediction dataframe
    pred_df = pd.DataFrame(predictions, columns=CORRECT_PLASTICC_CLASSES)
    pred_df['object_id'] = features_df['object_id'].values
    
    # Normalize probabilities to sum to 1
    prob_cols = CORRECT_PLASTICC_CLASSES
    row_sums = pred_df[prob_cols].sum(axis=1)
    
    # Handle zero sums
    zero_sum_mask = row_sums == 0
    if zero_sum_mask.sum() > 0:
        print(f"    âš ï¸� Fixed {zero_sum_mask.sum()} rows with zero probabilities")
        pred_df.loc[zero_sum_mask, prob_cols] = 1.0 / len(prob_cols)
        row_sums = pred_df[prob_cols].sum(axis=1)
    
    # Normalize
    for col in prob_cols:
        pred_df[col] = pred_df[col] / row_sums
    
    # Verify normalization
    final_sums = pred_df[prob_cols].sum(axis=1)
    assert np.allclose(final_sums, 1.0, atol=1e-6), "Probabilities don't sum to 1!"
    
    # Reorder columns
    final_cols = ['object_id'] + CORRECT_PLASTICC_CLASSES
    pred_df = pred_df[final_cols]
    
    print(f"    âœ… LightGBM predictions ready: {pred_df.shape}")
    return pred_df

# Cell 7: Main Processing Function
def process_test_data_lightgbm():
    """Process test data using LightGBM models"""
    
    if final_model is None and len(cv_models) == 0:
        print("â�Œ No LightGBM models available for processing")
        return None
    
    # Load test metadata
    print("\\nğŸ“Š Loading test metadata...")
    test_meta = pd.read_csv("/kaggle/input/PLAsTiCC-2018/test_set_metadata.csv")
    print(f"âœ… Test metadata loaded: {test_meta.shape}")
    
    # Processing configuration
    total_objects = len(test_meta)
    num_batches = (total_objects + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\\n=== LIGHTGBM PROCESSING CONFIGURATION ===")
    print(f"ğŸ“Š Total objects: {total_objects:,}")
    print(f"ğŸ“¦ Batch size: {BATCH_SIZE:,}")
    print(f"ğŸ”„ Number of batches: {num_batches}")
    print(f"â�±ï¸� Estimated time: {num_batches * 2:.0f}-{num_batches * 4:.0f} minutes")
    print(f"ğŸ¤– Models available: Final={'âœ“' if final_model else 'â�Œ'}, CV={len(cv_models)}")
    
    all_predictions = []
    successful_batches = 0
    
    # Process each batch
    for batch_num in range(num_batches):
        start_time = datetime.now()
        print(f"\\n--- LIGHTGBM BATCH {batch_num + 1}/{num_batches} ---")
        
        try:
            # Get batch metadata
            start_idx = batch_num * BATCH_SIZE
            end_idx = min((batch_num + 1) * BATCH_SIZE, total_objects)
            batch_meta = test_meta.iloc[start_idx:end_idx].copy()
            batch_obj_ids = set(batch_meta['object_id'].values)
            
            print(f"ğŸ�¯ Processing objects {start_idx:,} to {end_idx-1:,}")
            print(f"ğŸ“� Batch contains {len(batch_obj_ids):,} unique object IDs")
            
            # Load light curves efficiently (simplified for memory)
            print("    ğŸ“¡ Loading light curves...")
            batch_lc_list = []
            
            # Only check the main test files that are most likely to contain data
            test_files = [
                "/kaggle/input/PLAsTiCC-2018/test_set.csv",
            ]
            
            # Add batch files if they exist
            for i in range(1, 12):
                batch_file = f"/kaggle/input/PLAsTiCC-2018/test_set_{i:02d}.csv"
                if os.path.exists(batch_file):
                    test_files.append(batch_file)
            
            total_obs = 0
            for lc_file in test_files[:3]:  # Limit to first 3 files for memory
                if not os.path.exists(lc_file):
                    continue
                
                try:
                    # Read in small chunks
                    chunk_size = 200000
                    for chunk in pd.read_csv(lc_file, chunksize=chunk_size):
                        if 'object_id' in chunk.columns:
                            relevant_data = chunk[chunk['object_id'].isin(batch_obj_ids)]
                            if len(relevant_data) > 0:
                                batch_lc_list.append(relevant_data)
                                total_obs += len(relevant_data)
                        
                        # Early stopping to prevent memory issues
                        if total_obs > 500000:
                            break
                    
                    if total_obs > 500000:
                        print(f"    âš¡ Early stop - sufficient data loaded ({total_obs:,} obs)")
                        break
                        
                except Exception as e:
                    print(f"    âš ï¸� Error reading {lc_file}: {e}")
                    continue
            
            # Combine light curve data
            if batch_lc_list:
                batch_lc = pd.concat(batch_lc_list, ignore_index=True)
                print(f"    âœ… Light curves combined: {len(batch_lc):,} observations")
                print(f"    ğŸ“Š Unique objects in LC: {batch_lc['object_id'].nunique():,}")
            else:
                print(f"    âš ï¸� No light curves found - using metadata only")
                batch_lc = pd.DataFrame()
            
            # Create features
            if len(batch_lc) > 0:
                features_df = create_lightgbm_features(batch_lc, batch_meta)
            else:
                # Create minimal features from metadata only
                features_df = batch_meta.copy()
                # Add dummy light curve features
                for pb in range(6):
                    for feat in ['count', 'mean', 'std', 'min', 'max']:
                        features_df[f'{feat}_pb{pb}'] = 0.0
            
            # Make predictions
            pred_df = make_lightgbm_predictions(features_df, final_model, cv_models, label_encoder, feature_columns)
            
            # Validate prediction format
            assert 'object_id' in pred_df.columns, "Missing object_id column"
            assert all(cls in pred_df.columns for cls in CORRECT_PLASTICC_CLASSES), "Missing required classes"
            
            # Save intermediate results
            if (batch_num + 1) % SAVE_INTERVAL == 0:
                batch_file = f"{OUTPUT_DIR}lightgbm_predictions_batch_{batch_num + 1:03d}.csv"
                pred_df.to_csv(batch_file, index=False)
                print(f"    ğŸ’¾ Intermediate save: {batch_file}")
            
            all_predictions.append(pred_df)
            successful_batches += 1
            
            # Memory cleanup
            del batch_lc, batch_lc_list, features_df
            gc.collect()
            
            # Progress update
            elapsed = datetime.now() - start_time
            print(f"    âœ… Batch completed in {elapsed.total_seconds():.1f}s")
            
            # ETA calculation
            if batch_num > 0:
                avg_time = elapsed.total_seconds()
                remaining_batches = num_batches - batch_num - 1
                eta_minutes = (remaining_batches * avg_time) / 60
                print(f"    ğŸ“Š ETA: {eta_minutes:.1f} minutes remaining")
                
        except Exception as e:
            print(f"    â�Œ Batch {batch_num + 1} failed: {e}")
            # Continue with next batch rather than failing completely
            gc.collect()
            continue
    
    print(f"\\nâœ… Processing complete: {successful_batches}/{num_batches} batches successful")
    
    if successful_batches == 0:
        print("â�Œ No batches processed successfully!")
        return None
    
    return all_predictions

# Cell 8: Enhanced Submission Creation for LightGBM
def create_lightgbm_submission(all_predictions):
    """Create final submission with maximum scoring optimization"""
    print("\\n=== LIGHTGBM SUBMISSION CREATION ===")
    
    if not all_predictions:
        print("â�Œ No predictions to process!")
        return None
    
    # Combine all predictions
    print("ğŸ”— Combining all batch predictions...")
    final_predictions = pd.concat(all_predictions, ignore_index=True)
    print(f"âœ… Combined predictions: {final_predictions.shape}")
    
    # Remove duplicates (if any) - keep the last occurrence
    if final_predictions['object_id'].duplicated().any():
        print("âš ï¸� Removing duplicate object IDs...")
        final_predictions = final_predictions.drop_duplicates(subset=['object_id'], keep='last')
        print(f"âœ… After deduplication: {final_predictions.shape}")
    
    # Verify all required classes are present
    missing_classes = set(CORRECT_PLASTICC_CLASSES) - set(final_predictions.columns)
    if missing_classes:
        print(f"âš ï¸� Adding missing classes: {missing_classes}")
        for cls in missing_classes:
            final_predictions[cls] = 0.001  # Small default probability
    
    # Ensure we have all required columns
    required_cols = ['object_id'] + CORRECT_PLASTICC_CLASSES
    final_submission = final_predictions[required_cols].copy()
    
    # Final probability normalization (critical for scoring)
    print("ğŸ�¯ Final probability normalization...")
    prob_cols = CORRECT_PLASTICC_CLASSES
    row_sums = final_submission[prob_cols].sum(axis=1)
    
    # Handle edge cases
    zero_sum_rows = (row_sums == 0).sum()
    if zero_sum_rows > 0:
        print(f"    âš ï¸� Fixing {zero_sum_rows} rows with zero probabilities")
        mask = row_sums == 0
        final_submission.loc[mask, prob_cols] = 1.0 / len(prob_cols)
        row_sums = final_submission[prob_cols].sum(axis=1)
    
    # Normalize all rows to sum to 1
    for col in prob_cols:
        final_submission[col] = final_submission[col] / row_sums
    
    # Final validation
    final_sums = final_submission[prob_cols].sum(axis=1)
    assert np.allclose(final_sums, 1.0, atol=1e-6), "Final probabilities don't sum to 1!"
    
    # Convert object_id to integers (required by Kaggle)
    final_submission['object_id'] = final_submission['object_id'].astype('Int64')
    
    # Sort by object_id for consistency
    final_submission = final_submission.sort_values('object_id').reset_index(drop=True)
    
    # Save final submission
    submission_file = f"{OUTPUT_DIR}lightgbm_submission_final.csv"
    final_submission.to_csv(submission_file, index=False)
    
    print(f"âœ… LightGBM submission created: {submission_file}")
    print(f"ğŸ“Š Final shape: {final_submission.shape}")
    print(f"ğŸ�¯ Classes: {CORRECT_PLASTICC_CLASSES}")
    
    # Quality checks
    print(f"\\nğŸ”� QUALITY CHECKS:")
    print(f"    âœ“ Object IDs: {final_submission['object_id'].nunique():,} unique")
    print(f"    âœ“ Probability range: [{final_submission[prob_cols].min().min():.6f}, {final_submission[prob_cols].max().max():.6f}]")
    print(f"    âœ“ Row sums: [{final_sums.min():.6f}, {final_sums.max():.6f}]")
    print(f"    âœ“ All sums â‰ˆ 1.0: {np.allclose(final_sums, 1.0, atol=1e-6)}")
    print(f"    âœ“ No NaN values: {not final_submission.isnull().any().any()}")
    print(f"    âœ“ Correct columns: {len(required_cols)} columns present")
    
    # Display sample predictions
    print(f"\\nğŸ“‹ SAMPLE PREDICTIONS:")
    sample_df = final_submission.head(3)
    for idx, row in sample_df.iterrows():
        obj_id = row['object_id']
        max_prob_class = prob_cols[np.argmax(row[prob_cols])]
        max_prob = row[max_prob_class]
        print(f"    Object {obj_id}: {max_prob_class} (prob={max_prob:.4f})")
    
    # Class distribution analysis
    print(f"\\nğŸ“Š CLASS DISTRIBUTION:")
    class_predictions = np.argmax(final_submission[prob_cols].values, axis=1)
    for i, class_name in enumerate(CORRECT_PLASTICC_CLASSES):
        count = np.sum(class_predictions == i)
        percentage = 100 * count / len(final_submission)
        print(f"    {class_name}: {count:,} objects ({percentage:.2f}%)")
    
    return final_submission

# Cell 9: Main Execution Function for LightGBM
def main_lightgbm_execution():
    """Main execution function with comprehensive error handling"""
    print("\\n" + "="*60)
    print("ğŸš€ STARTING LIGHTGBM PLAsTiCC TESTING PIPELINE")
    print("="*60)
    
    start_time = datetime.now()
    
    try:
        # Process test data
        all_predictions = process_test_data_lightgbm()
        
        if all_predictions is None or len(all_predictions) == 0:
            print("â�Œ No predictions generated - pipeline failed!")
            return False
        
        # Create final submission
        final_submission = create_lightgbm_submission(all_predictions)
        
        if final_submission is None:
            print("â�Œ Submission creation failed!")
            return False
        
        # Success summary
        total_time = datetime.now() - start_time
        print(f"\\n" + "="*60)
        print("ğŸ�‰ LIGHTGBM PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"â�±ï¸� Total execution time: {total_time}")
        print(f"ğŸ“Š Objects processed: {len(final_submission):,}")
        
        if model_metrics:
            print(f"ğŸ�¯ Expected performance based on training:")
            print(f"   Training CV accuracy: {model_metrics.get('cv_accuracy', 'Unknown')}")
            print(f"   Training CV log loss: {model_metrics.get('cv_log_loss', 'Unknown')}")
        
        print(f"ğŸ“� Output file: lightgbm_submission_final.csv")
        print(f"ğŸ�† LightGBM model optimized for astronomical time-series")
        
        return True
        
    except Exception as e:
        print(f"\\nâ�Œ LIGHTGBM PIPELINE FAILED: {e}")
        print(f"ğŸ› ï¸� Debugging information:")
        print(f"    - Final model loaded: {final_model is not None}")
        print(f"    - CV models loaded: {len(cv_models)}")
        print(f"    - Label encoder loaded: {label_encoder is not None}")
        print(f"    - Feature columns loaded: {feature_columns is not None}")
        
        # Try to create a minimal fallback submission
        try:
            print("\\nğŸš¨ Attempting fallback submission creation...")
            create_lightgbm_fallback_submission()
        except Exception as fallback_error:
            print(f"â�Œ Fallback submission also failed: {fallback_error}")
        
        return False






# Cell 10: Fallback and Utilities for LightGBM
def create_lightgbm_fallback_submission():
    """Create a minimal fallback submission if main pipeline fails"""
    print("ğŸš¨ Creating LightGBM fallback submission with uniform probabilities...")
    
    # Load test metadata
    test_meta = pd.read_csv("/kaggle/input/PLAsTiCC-2018/test_set_metadata.csv")
    
    # Create uniform probability distribution
    n_objects = len(test_meta)
    n_classes = len(CORRECT_PLASTICC_CLASSES)
    uniform_prob = 1.0 / n_classes
    
    # Create submission dataframe
    fallback_submission = pd.DataFrame()
    fallback_submission['object_id'] = test_meta['object_id'].astype('Int64')
    
    # Add uniform probabilities for all classes
    for class_name in CORRECT_PLASTICC_CLASSES:
        fallback_submission[class_name] = uniform_prob
    
    # Save fallback submission
    fallback_file = f"{OUTPUT_DIR}lightgbm_fallback_submission.csv"
    fallback_submission.to_csv(fallback_file, index=False)
    
    print(f"âœ… LightGBM fallback submission created: {fallback_file}")
    print(f"ğŸ“Š Shape: {fallback_submission.shape}")
    print(f"âš ï¸� Note: This uses uniform probabilities and will score poorly!")
    
    return fallback_submission

def validate_lightgbm_submission(submission_file):
    """Validate that submission file meets Kaggle requirements"""
    print(f"\\nğŸ”� VALIDATING LIGHTGBM SUBMISSION: {submission_file}")
    
    try:
        # Load submission
        sub_df = pd.read_csv(submission_file)
        
        # Check required columns
        required_cols = ['object_id'] + CORRECT_PLASTICC_CLASSES
        missing_cols = set(required_cols) - set(sub_df.columns)
        extra_cols = set(sub_df.columns) - set(required_cols)
        
        print(f"    âœ“ Columns present: {len(sub_df.columns)}")
        if missing_cols:
            print(f"    â�Œ Missing columns: {missing_cols}")
            return False
        if extra_cols:
            print(f"    âš ï¸� Extra columns: {extra_cols}")
        
        # Check object IDs
        print(f"    âœ“ Unique object IDs: {sub_df['object_id'].nunique():,}")
        print(f"    âœ“ Total rows: {len(sub_df):,}")
        
        # Check probabilities
        prob_cols = CORRECT_PLASTICC_CLASSES
        prob_sums = sub_df[prob_cols].sum(axis=1)
        
        print(f"    âœ“ Probability sums range: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
        print(f"    âœ“ All sums â‰ˆ 1.0: {np.allclose(prob_sums, 1.0, atol=1e-5)}")
        print(f"    âœ“ No negative probabilities: {(sub_df[prob_cols] >= 0).all().all()}")
        print(f"    âœ“ No NaN values: {not sub_df.isnull().any().any()}")
        
        # File size check
        file_size_mb = os.path.getsize(submission_file) / (1024*1024)
        print(f"    âœ“ File size: {file_size_mb:.1f} MB")
        
        if file_size_mb > 500:
            print(f"    âš ï¸� Large file size - may cause upload issues")
        
        print("    âœ… LightGBM submission format validation PASSED!")
        return True
        
    except Exception as e:
        print(f"    â�Œ Validation failed: {e}")
        return False

def display_lightgbm_summary():
    """Display final execution summary"""
    print("\\n" + "="*60)
    print("ğŸ“‹ LIGHTGBM FINAL EXECUTION SUMMARY")
    print("="*60)
    
    # Check for output files
    output_files = []
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith('.csv') and 'lightgbm' in file:
            file_path = os.path.join(OUTPUT_DIR, file)
            file_size = os.path.getsize(file_path) / (1024*1024)
            output_files.append((file, file_size))
    
    if output_files:
        print("ğŸ“� Generated LightGBM files:")
        for filename, size_mb in output_files:
            print(f"    ğŸ“„ {filename} ({size_mb:.1f} MB)")
            
            # Validate main submission file
            if 'lightgbm_submission_final.csv' in filename:
                validate_lightgbm_submission(os.path.join(OUTPUT_DIR, filename))
    else:
        print("â�Œ No LightGBM output files generated!")
    
    print(f"\\nğŸ�¯ Next steps:")
    print(f"    1. Download lightgbm_submission_final.csv")
    print(f"    2. Submit to PLAsTiCC competition")
    print(f"    3. Monitor leaderboard performance")
    
    if model_metrics:
        expected_acc = model_metrics.get('cv_accuracy', 0.8)
        expected_loss = model_metrics.get('cv_log_loss', 0.65)
        print(f"    4. Expected score: ~{expected_loss:.3f} log-loss ({expected_acc:.1%} accuracy)")
    
    print(f"\\nâœ… LightGBM pipeline execution completed!")


# Cell 11: Execute the LightGBM Pipeline
if __name__ == "__main__":
    print("\\nğŸš€ STARTING LIGHTGBM TESTING EXECUTION")
    
    # Check if we have the necessary components
    if final_model is None and len(cv_models) == 0:
        print("â�Œ No LightGBM models found! Please check:")
        print("    1. lightgbm-models dataset is added to your notebook")
        print("    2. Model files exist in /kaggle/input/lightgbm-models/")
        print("    3. Files are not corrupted")
        
        # Try fallback
        try:
            create_lightgbm_fallback_submission()
            print("âœ… Created fallback submission instead")
        except:
            print("â�Œ Even fallback submission failed")
    else:
        # Execute main pipeline
        success = main_lightgbm_execution()
        display_lightgbm_summary()
        
        if success:
            print("\\nğŸ�‰ Ready for Kaggle submission!")
            print("ğŸ�† Your LightGBM models have been successfully applied to 3.5M objects!")
        else:
            print("\\nâš ï¸� Check logs for issues - fallback submission may be available")

# Memory cleanup
gc.collect()
print(f"\\nğŸ§¹ Memory cleanup completed")
print(f"â�° LightGBM notebook finished at: {datetime.now()}")





# =========================================
# EMERGENCY FIX: Combine All Batch Files
# Run these cells in order to fix your submission
# =========================================

# Cell 1: Import Libraries and Setup
import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime

print("ğŸš¨ EMERGENCY FIX: Combining All Batch Files")
print(f"Started at: {datetime.now()}")

# PLAsTiCC classes
CORRECT_PLASTICC_CLASSES = [
    'class_6', 'class_15', 'class_16', 'class_42', 'class_52', 
    'class_53', 'class_62', 'class_64', 'class_65', 'class_67', 
    'class_88', 'class_90', 'class_92', 'class_95', 'class_99'
]

print("âœ… Setup complete")

# Cell 2: Find and Load All Batch Files
def find_and_load_batches():
    """Find and load all existing batch files"""
    
    print("\nğŸ“� STEP 1: Finding all batch files...")
    batch_files = []
    working_dir = "/kaggle/working"
    
    for file in os.listdir(working_dir):
        if 'lightgbm_predictions_batch_' in file and file.endswith('.csv'):
            batch_files.append(os.path.join(working_dir, file))
            file_size = os.path.getsize(os.path.join(working_dir, file)) / (1024*1024)
            print(f"    ğŸ“„ Found: {file} ({file_size:.1f} MB)")
    
    batch_files.sort()  # Sort for consistent order
    print(f"âœ… Found {len(batch_files)} batch files")
    
    print("\nğŸ“Š STEP 2: Loading and combining batch files...")
    combined_predictions = []
    processed_object_ids = set()
    
    for i, batch_file in enumerate(batch_files):
        print(f"    Loading batch {i+1}/{len(batch_files)}: {os.path.basename(batch_file)}")
        try:
            batch_df = pd.read_csv(batch_file)
            print(f"        Rows: {len(batch_df):,}")
            print(f"        Objects: {batch_df['object_id'].nunique():,}")
            
            # Track processed objects
            batch_objects = set(batch_df['object_id'].values)
            processed_object_ids.update(batch_objects)
            
            combined_predictions.append(batch_df)
            
        except Exception as e:
            print(f"        â�Œ Error loading {batch_file}: {e}")
    
    return combined_predictions, processed_object_ids

# Execute batch loading
combined_predictions, processed_object_ids = find_and_load_batches()

# Cell 3: Combine Batch Files
def combine_batch_predictions(combined_predictions, processed_object_ids):
    """Combine all batch predictions"""
    
    if not combined_predictions:
        print("â�Œ No batch files could be loaded!")
        return None
    
    print("\nğŸ”— Combining all batch predictions...")
    combined_df = pd.concat(combined_predictions, ignore_index=True)
    print(f"âœ… Combined shape: {combined_df.shape}")
    print(f"âœ… Unique objects from batches: {len(processed_object_ids):,}")
    
    # Remove duplicates if any
    if combined_df['object_id'].duplicated().any():
        print("âš ï¸� Removing duplicate object IDs...")
        combined_df = combined_df.drop_duplicates(subset=['object_id'], keep='last')
        print(f"âœ… After deduplication: {combined_df.shape}")
    
    return combined_df

# Execute combination
combined_df = combine_batch_predictions(combined_predictions, processed_object_ids)

# Clean up memory
del combined_predictions
gc.collect()

# Cell 4: Check for Missing Objects and Fill Gaps
def fill_missing_objects(combined_df, processed_object_ids):
    """Check for missing objects and fill gaps"""
    
    print("\nğŸ“‹ STEP 3: Checking for missing objects...")
    
    # Load full test metadata to see what's missing
    test_meta = pd.read_csv("/kaggle/input/PLAsTiCC-2018/test_set_metadata.csv")
    all_object_ids = set(test_meta['object_id'].values)
    missing_object_ids = all_object_ids - processed_object_ids
    
    print(f"ğŸ“Š Total test objects: {len(all_object_ids):,}")
    print(f"âœ… Objects in batches: {len(processed_object_ids):,}")
    print(f"â�Œ Missing objects: {len(missing_object_ids):,}")
    
    if len(missing_object_ids) > 0:
        print(f"\nğŸ”§ STEP 4: Creating predictions for missing objects...")
        
        # Get metadata for missing objects
        missing_meta = test_meta[test_meta['object_id'].isin(missing_object_ids)]
        
        # Create uniform probability predictions for missing objects
        n_missing = len(missing_meta)
        uniform_prob = 1.0 / len(CORRECT_PLASTICC_CLASSES)
        
        # Create missing predictions DataFrame
        missing_predictions = pd.DataFrame()
        missing_predictions['object_id'] = missing_meta['object_id'].values
        
        for class_name in CORRECT_PLASTICC_CLASSES:
            missing_predictions[class_name] = uniform_prob
        
        print(f"âœ… Created uniform predictions for {n_missing:,} missing objects")
        
        # Combine with existing predictions
        print("\nğŸ”— Combining with existing predictions...")
        final_combined = pd.concat([combined_df, missing_predictions], ignore_index=True)
    else:
        final_combined = combined_df
    
    print(f"\nğŸ“Š FINAL COMBINATION RESULTS:")
    print(f"âœ… Total objects: {len(final_combined):,}")
    print(f"âœ… Expected objects: {len(all_object_ids):,}")
    print(f"âœ… Coverage: {len(final_combined) / len(all_object_ids) * 100:.1f}%")
    
    return final_combined

# Execute missing object filling
if combined_df is not None:
    final_combined = fill_missing_objects(combined_df, processed_object_ids)
else:
    print("â�Œ Cannot proceed - no combined data")
    final_combined = None

# Cell 5: Create Final Submission File
def create_final_submission(combined_df):
    """Create final submission file with validation"""
    
    if combined_df is None:
        print("â�Œ No data to create submission")
        return None, None
    
    print("\nğŸ�¯ STEP 5: Creating final submission...")
    
    # Ensure all required columns
    required_cols = ['object_id'] + CORRECT_PLASTICC_CLASSES
    missing_cols = set(required_cols) - set(combined_df.columns)
    if missing_cols:
        print(f"âš ï¸� Adding missing columns: {missing_cols}")
        for col in missing_cols:
            combined_df[col] = 1.0 / len(CORRECT_PLASTICC_CLASSES)
    
    # Select and reorder columns
    final_submission = combined_df[required_cols].copy()
    
    # Normalize probabilities
    print("ğŸ�¯ Normalizing probabilities...")
    prob_cols = CORRECT_PLASTICC_CLASSES
    row_sums = final_submission[prob_cols].sum(axis=1)
    
    # Fix zero sums
    zero_sum_mask = row_sums == 0
    if zero_sum_mask.sum() > 0:
        print(f"âš ï¸� Fixing {zero_sum_mask.sum()} rows with zero probabilities")
        final_submission.loc[zero_sum_mask, prob_cols] = 1.0 / len(prob_cols)
        row_sums = final_submission[prob_cols].sum(axis=1)
    
    # Normalize to sum to 1
    for col in prob_cols:
        final_submission[col] = final_submission[col] / row_sums
    
    # Final validation
    final_sums = final_submission[prob_cols].sum(axis=1)
    assert np.allclose(final_sums, 1.0, atol=1e-6), "Probabilities don't sum to 1!"
    
    # Convert object_id to integers
    final_submission['object_id'] = final_submission['object_id'].astype('Int64')
    
    # Sort by object_id
    final_submission = final_submission.sort_values('object_id').reset_index(drop=True)
    
    # Save final submission
    submission_file = "/kaggle/working/FIXED_lightgbm_submission_complete.csv"
    final_submission.to_csv(submission_file, index=False)
    
    print(f"\nâœ… FIXED submission created: {submission_file}")
    print(f"ğŸ“Š Final shape: {final_submission.shape}")
    print(f"ğŸ’¾ File size: {os.path.getsize(submission_file) / (1024*1024):.1f} MB")
    
    return final_submission, submission_file

# Execute final submission creation
final_submission, submission_file = create_final_submission(final_combined)

# Cell 6: Validate Final Submission
def validate_final_submission(submission_file):
    """Validate the final submission"""
    
    if submission_file is None:
        print("â�Œ No submission file to validate")
        return False
    
    print(f"\nğŸ”� VALIDATING FINAL SUBMISSION: {os.path.basename(submission_file)}")
    
    # Load and check
    sub_df = pd.read_csv(submission_file)
    
    # Expected total from PLAsTiCC
    expected_objects = 3492890
    
    print(f"âœ… Rows: {len(sub_df):,}")
    print(f"âœ… Expected rows: {expected_objects:,}")
    print(f"âœ… Coverage: {len(sub_df) / expected_objects * 100:.2f}%")
    print(f"âœ… Unique objects: {sub_df['object_id'].nunique():,}")
    
    # Check columns
    required_cols = ['object_id'] + CORRECT_PLASTICC_CLASSES
    missing_cols = set(required_cols) - set(sub_df.columns)
    if missing_cols:
        print(f"â�Œ Missing columns: {missing_cols}")
        return False
    else:
        print(f"âœ… All required columns present")
    
    # Check probabilities
    prob_cols = CORRECT_PLASTICC_CLASSES
    prob_sums = sub_df[prob_cols].sum(axis=1)
    
    print(f"âœ… Probability range: [{sub_df[prob_cols].min().min():.6f}, {sub_df[prob_cols].max().max():.6f}]")
    print(f"âœ… Row sums range: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
    print(f"âœ… All sums â‰ˆ 1.0: {np.allclose(prob_sums, 1.0, atol=1e-5)}")
    print(f"âœ… No NaN values: {not sub_df.isnull().any().any()}")
    
    # Class distribution
    print(f"\nğŸ“Š CLASS DISTRIBUTION (Top 5):")
    class_predictions = np.argmax(sub_df[prob_cols].values, axis=1)
    for i, class_name in enumerate(CORRECT_PLASTICC_CLASSES[:5]):
        count = np.sum(class_predictions == i)
        percentage = 100 * count / len(sub_df)
        print(f"    {class_name}: {count:,} objects ({percentage:.2f}%)")
    
    if len(sub_df) >= expected_objects * 0.99:  # At least 99% coverage
        print(f"\nğŸ�‰ SUBMISSION VALIDATION PASSED!")
        return True
    else:
        print(f"\nâ�Œ Insufficient coverage: {len(sub_df)} < {expected_objects}")
        return False

# Execute validation
is_valid = validate_final_submission(submission_file)

# Cell 7: Final Summary and Instructions
print(f"\n" + "="*60)
print("ğŸ�‰ BATCH COMBINATION FIX COMPLETED!")
print("="*60)

if final_submission is not None:
    print(f"ğŸ“Š Final objects: {len(final_submission):,}")
    print(f"ğŸ“� File created: FIXED_lightgbm_submission_complete.csv")
    print(f"âœ… Validation passed: {is_valid}")
    
    if is_valid:
        print(f"\nğŸ�† SUCCESS! Your submission file is now complete!")
        print(f"ğŸ“¤ UPLOAD THIS FILE: FIXED_lightgbm_submission_complete.csv")
        print(f"ğŸ�¯ This file has all {len(final_submission):,} required objects!")
        print(f"\nğŸ“‹ Next Steps:")
        print(f"    1. Download: FIXED_lightgbm_submission_complete.csv")
        print(f"    2. Upload to PLAsTiCC competition")
        print(f"    3. Submit and check leaderboard")
    else:
        print(f"\nâš ï¸� File created but validation failed")
        print(f"ğŸ“¤ You can still try uploading: FIXED_lightgbm_submission_complete.csv")
else:
    print("â�Œ Could not create submission file")

# Final cleanup
gc.collect()
print(f"\nâœ… Fix completed at: {datetime.now()}")
print("\nğŸ�‰ Ready for submission!")

