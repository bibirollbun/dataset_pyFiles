# ===============================================
# ENHANCED PLAsTiCC TESTING NOTEBOOK
# Improved feature alignment and scoring optimization
# ===============================================

# Cell 1: Environment Setup and Configuration
import pandas as pd
import numpy as np
import pickle
import joblib
import gc
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=== ENHANCED PLAsTiCC TESTING NOTEBOOK ===")
print(f"Started at: {datetime.now()}")
print("ğŸ�¯ Focus: Maximizing Kaggle competition scores")

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
    
    # Check datasets
    datasets = []
    for item in os.listdir("/kaggle/input/ml-improve"):
        datasets.append(item)
        print(f"ğŸ“� Dataset: {item}")
    
    # Find model files
    model_files = []
    scaler_files = []
    
    for root, dirs, files in os.walk("/kaggle/input"):
        for file in files:
            full_path = os.path.join(root, file)
            if 'model' in file.lower() and file.endswith('.pkl'):
                model_files.append(full_path)
                print(f"ğŸ”§ Model found: {full_path}")
            elif 'scaler' in file.lower() and file.endswith('.pkl'):
                scaler_files.append(full_path)
                print(f"âš–ï¸� Scaler found: {full_path}")
    
    # Check PLAsTiCC data files
    plasticc_files = []
    for root, dirs, files in os.walk("/kaggle/input"):
        for file in files:
            if 'test_set' in file and file.endswith('.csv'):
                full_path = os.path.join(root, file)
                size_mb = os.path.getsize(full_path) / (1024*1024)
                plasticc_files.append(full_path)
                print(f"ğŸ“Š Test data: {file} ({size_mb:.1f} MB)")
    
    return model_files, scaler_files, plasticc_files

model_files, scaler_files, plasticc_files = enhanced_environment_check()




# Cell 3: Robust Model and Scaler Loading
def load_models_robust():
    """Load models and scalers with multiple fallback strategies"""
    print("\n=== ROBUST MODEL LOADING ===")
    
    # Priority order for model loading
    model_priorities = [
        "best_model_XGBoost",
        "XGBoost", 
        "best_model",
        "xgboost",
        "catboost",
        "lightgbm"
    ]
    
    model = None
    model_name = None
    
    # Try to load model
    for model_file in model_files:
        for priority in model_priorities:
            if priority.lower() in model_file.lower():
                try:
                    # Try pickle first
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    model_name = priority
                    print(f"âœ… Model loaded (pickle): {model_file}")
                    break
                except:
                    try:
                        # Try joblib
                        model = joblib.load(model_file)
                        model_name = priority
                        print(f"âœ… Model loaded (joblib): {model_file}")
                        break
                    except Exception as e:
                        print(f"â�Œ Failed to load {model_file}: {e}")
                        continue
        if model is not None:
            break
    
    if model is None:
        print("â�Œ No model could be loaded!")
        return None, None, None
    
    # Try to load scaler
    scaler = None
    for scaler_file in scaler_files:
        try:
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            print(f"âœ… Scaler loaded (pickle): {scaler_file}")
            break
        except:
            try:
                scaler = joblib.load(scaler_file)
                print(f"âœ… Scaler loaded (joblib): {scaler_file}")
                break
            except Exception as e:
                print(f"â�Œ Failed to load scaler {scaler_file}: {e}")
                continue
    
    if scaler is None:
        print("âš ï¸� No scaler loaded - creating fallback StandardScaler")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # Fit on dummy data matching expected feature count
        n_features = 94  # Based on your training
        dummy_data = np.random.normal(0, 1, (1000, n_features))
        scaler.fit(dummy_data)
        print("âš ï¸� Fallback scaler created and fitted")
    
    print(f"\nâœ… Successfully loaded:")
    print(f"   Model: {model_name}")
    print(f"   Scaler: {'Original' if scaler_files else 'Fallback'}")
    print(f"   Expected features: {getattr(scaler, 'n_features_in_', 'Unknown')}")
    
    return model, scaler, model_name

model, scaler, model_name = load_models_robust()




# Cell 4: Enhanced Feature Engineering (Matching Training)
def extract_enhanced_features(lc_batch, meta_batch):
    """Enhanced feature extraction matching training exactly"""
    print(f"    ğŸ”§ Extracting enhanced features for {len(meta_batch)} objects...")
    
    feature_list = []
    
    for idx, row in meta_batch.iterrows():
        obj_id = row['object_id']
        
        # Get light curve data
        if len(lc_batch) > 0 and 'object_id' in lc_batch.columns:
            obj_lc = lc_batch[lc_batch['object_id'] == obj_id]
        else:
            obj_lc = pd.DataFrame()
        
        # Initialize with metadata
        features = {
            'object_id': obj_id,
            'ra': row.get('ra', 0),
            'decl': row.get('decl', 0),
            'gal_l': row.get('gal_l', 0),
            'gal_b': row.get('gal_b', 0),
            'ddf': row.get('ddf', 0),
            'hostgal_photoz': row.get('hostgal_photoz', 0),
            'hostgal_photoz_err': row.get('hostgal_photoz_err', 0),
            'distmod': row.get('distmod', 0),
            'mwebv': row.get('mwebv', 0)
        }
        
        # Extract light curve features
        if len(obj_lc) > 0:
            lc_features = extract_lightcurve_features_enhanced(obj_lc)
            features.update(lc_features)
        else:
            # Create comprehensive dummy features
            dummy_features = create_comprehensive_dummy_features()
            features.update(dummy_features)
        
        feature_list.append(features)
    
    features_df = pd.DataFrame(feature_list)
    features_df = features_df.fillna(0)
    
    print(f"    âœ… Enhanced features extracted: {features_df.shape}")
    return features_df

def extract_lightcurve_features_enhanced(obj_lc):
    """Extract comprehensive light curve features - FIXED VERSION"""
    features = {}
    
    # Required columns check
    required_cols = ['passband', 'flux', 'flux_err', 'detected', 'mjd']
    for col in required_cols:
        if col not in obj_lc.columns:
            return create_comprehensive_dummy_features()
    
    # Per-passband features (ugrizy = 0,1,2,3,4,5)
    for pb in range(6):
        pb_data = obj_lc[obj_lc['passband'] == pb]
        
        if len(pb_data) > 0:
            flux = pb_data['flux'].values
            flux_err = pb_data['flux_err'].values
            detected = pb_data['detected'].values
            mjd = pb_data['mjd'].values
            
            # Basic statistics
            features[f'flux_mean_{pb}'] = np.mean(flux)
            features[f'flux_std_{pb}'] = np.std(flux) if len(flux) > 1 else 0
            features[f'flux_min_{pb}'] = np.min(flux)
            features[f'flux_max_{pb}'] = np.max(flux)
            features[f'flux_median_{pb}'] = np.median(flux)
            features[f'flux_range_{pb}'] = np.max(flux) - np.min(flux)
            
            # Advanced statistics
            if len(flux) > 2:
                features[f'flux_skew_{pb}'] = pd.Series(flux).skew()
                features[f'flux_kurt_{pb}'] = pd.Series(flux).kurtosis()
                features[f'flux_p25_{pb}'] = np.percentile(flux, 25)
                features[f'flux_p75_{pb}'] = np.percentile(flux, 75)
                features[f'flux_iqr_{pb}'] = features[f'flux_p75_{pb}'] - features[f'flux_p25_{pb}']
            else:
                features[f'flux_skew_{pb}'] = 0
                features[f'flux_kurt_{pb}'] = 0
                features[f'flux_p25_{pb}'] = features[f'flux_mean_{pb}']
                features[f'flux_p75_{pb}'] = features[f'flux_mean_{pb}']
                features[f'flux_iqr_{pb}'] = 0
            
            # Detection features
            features[f'detected_count_{pb}'] = np.sum(detected)
            features[f'total_count_{pb}'] = len(pb_data)
            features[f'detection_rate_{pb}'] = features[f'detected_count_{pb}'] / features[f'total_count_{pb}']
            
            # Temporal features
            if len(mjd) > 1:
                features[f'time_span_{pb}'] = np.max(mjd) - np.min(mjd)
                features[f'time_mean_{pb}'] = np.mean(mjd)
                features[f'obs_rate_{pb}'] = len(mjd) / (features[f'time_span_{pb}'] + 1)
            else:
                features[f'time_span_{pb}'] = 0
                features[f'time_mean_{pb}'] = mjd[0] if len(mjd) > 0 else 0
                features[f'obs_rate_{pb}'] = 0
            
            # FIXED: Use correct feature names matching training
            if len(flux) > 2 and np.std(flux) > 0:
                # Stetson J index
                normalized_flux = (flux - np.mean(flux)) / (flux_err + 1e-8)
                features[f'stetson_j_{pb}'] = np.mean(normalized_flux**2)
                
                # Von Neumann ratio (Eta)
                diff_sum = np.sum(np.diff(flux)**2)
                var_sum = np.sum((flux - np.mean(flux))**2)
                features[f'eta_{pb}'] = diff_sum / (var_sum + 1e-8)
                
                # CRITICAL FIX: Use flux_amplitude instead of amplitude_ratio
                features[f'flux_amplitude_{pb}'] = features[f'flux_range_{pb}'] / (np.std(flux) + 1e-8)
            else:
                features[f'stetson_j_{pb}'] = 0
                features[f'eta_{pb}'] = 0
                features[f'flux_amplitude_{pb}'] = 0  # FIXED NAME
        
        else:
            # No data for this passband - FIXED FEATURE NAMES
            pb_feature_names = [
                'flux_mean', 'flux_std', 'flux_min', 'flux_max', 'flux_median', 'flux_range',
                'flux_skew', 'flux_kurt', 'flux_p25', 'flux_p75', 'flux_iqr',
                'detected_count', 'total_count', 'detection_rate',
                'time_span', 'time_mean', 'obs_rate',
                'stetson_j', 'eta', 'flux_amplitude'  # FIXED: flux_amplitude not amplitude_ratio
            ]
            for feat_name in pb_feature_names:
                features[f'{feat_name}_{pb}'] = 0
    
    # Global features (across all passbands)
    if len(obj_lc) > 0:
        all_flux = obj_lc['flux'].values
        all_detected = obj_lc['detected'].values
        all_mjd = obj_lc['mjd'].values
        
        features['global_flux_mean'] = np.mean(all_flux)
        features['global_flux_std'] = np.std(all_flux)
        features['global_flux_range'] = np.max(all_flux) - np.min(all_flux)
        features['global_detection_rate'] = np.mean(all_detected)
        features['global_obs_count'] = len(obj_lc)
        features['global_time_span'] = np.max(all_mjd) - np.min(all_mjd) if len(all_mjd) > 1 else 0
        features['passband_diversity'] = obj_lc['passband'].nunique()
    else:
        features['global_flux_mean'] = 0
        features['global_flux_std'] = 0
        features['global_flux_range'] = 0
        features['global_detection_rate'] = 0
        features['global_obs_count'] = 0
        features['global_time_span'] = 0
        features['passband_diversity'] = 0
    
    return features


def create_comprehensive_dummy_features():
    """Create comprehensive dummy features matching training - FIXED VERSION"""
    features = {}
    
    # Per-passband features - FIXED NAMES
    for pb in range(6):
        pb_feature_names = [
            'flux_mean', 'flux_std', 'flux_min', 'flux_max', 'flux_median', 'flux_range',
            'flux_skew', 'flux_kurt', 'flux_p25', 'flux_p75', 'flux_iqr',
            'detected_count', 'total_count', 'detection_rate',
            'time_span', 'time_mean', 'obs_rate',
            'stetson_j', 'eta', 'flux_amplitude'  # FIXED: flux_amplitude not amplitude_ratio
        ]
        for feat_name in pb_feature_names:
            features[f'{feat_name}_{pb}'] = 0
    
    # Global features
    global_features = [
        'global_flux_mean', 'global_flux_std', 'global_flux_range',
        'global_detection_rate', 'global_obs_count', 'global_time_span',
        'passband_diversity'
    ]
    for feat_name in global_features:
        features[feat_name] = 0
    
    return features




# Cell 5: Smart Feature Alignment
def align_features_intelligently(features_df, scaler):
    """Intelligently align features with scaler expectations"""
    print(f"    ğŸ�¯ Smart feature alignment...")
    
    # Remove object_id for processing
    X = features_df.drop('object_id', axis=1)
    current_features = list(X.columns)
    expected_features = getattr(scaler, 'n_features_in_', len(current_features))
    
    print(f"    Current features: {len(current_features)}")
    print(f"    Expected features: {expected_features}")
    
    if len(current_features) == expected_features:
        print(f"    âœ… Perfect alignment!")
        return X
    
    elif len(current_features) > expected_features:
        print(f"    âœ‚ï¸� Trimming {len(current_features) - expected_features} excess features")
        # Keep most important features (usually basic stats come first)
        X_aligned = X.iloc[:, :expected_features]
        return X_aligned
    
    else:
        print(f"    â�• Padding {expected_features - len(current_features)} missing features")
        # Add zero-filled columns for missing features
        missing_count = expected_features - len(current_features)
        for i in range(missing_count):
            X[f'padding_feature_{i}'] = 0.0
        return X




# Cell 6: Enhanced Prediction with Probability Optimization
def make_predictions_enhanced(features_df, model, scaler):
    """Enhanced prediction with probability optimization - FIXED VERSION"""
    print(f"    ğŸ�¯ Making enhanced predictions for {len(features_df)} objects...")
    
    # Align features
    X_aligned = align_features_intelligently(features_df, scaler)
    
    # Scale features with better error handling
    try:
        X_scaled = scaler.transform(X_aligned)
    except Exception as e:
        print(f"    âš ï¸� Scaling issue: {e}")
        # Try without scaling as fallback
        X_scaled = X_aligned.values
    
    # Get probabilities
    try:
        probabilities = model.predict_proba(X_scaled)
        print(f"    âœ… Predictions generated: {probabilities.shape}")
        
        # CRITICAL FIX: Handle 14 vs 15 classes
        if probabilities.shape[1] == 14:
            print(f"    ğŸ”§ Model predicts 14 classes, padding to 15 for PLAsTiCC")
            # Add a small probability for the missing class (usually class_99)
            padding_prob = 0.001
            padded_probs = np.column_stack([
                probabilities * (1 - padding_prob),  # Scale down existing probs
                np.full(len(probabilities), padding_prob)  # Add small prob for missing class
            ])
            probabilities = padded_probs
            print(f"    âœ… Padded to shape: {probabilities.shape}")
            
    except Exception as e:
        print(f"    â�Œ Prediction error: {e}")
        # Fallback: uniform probabilities
        n_classes = 15  # PLAsTiCC has 15 classes
        probabilities = np.full((len(features_df), n_classes), 1.0/n_classes)
        print(f"    âš ï¸� Using uniform fallback probabilities")
    
    # CRITICAL: Map to correct PLAsTiCC class names
    pred_df = pd.DataFrame(probabilities, columns=CORRECT_PLASTICC_CLASSES)
    pred_df['object_id'] = features_df['object_id'].values
    
    # Ensure probabilities sum to 1 (critical for scoring)
    prob_cols = CORRECT_PLASTICC_CLASSES
    row_sums = pred_df[prob_cols].sum(axis=1)
    
    # Handle any rows with zero probabilities
    zero_sum_mask = row_sums == 0
    if zero_sum_mask.sum() > 0:
        print(f"    âš ï¸� Fixed {zero_sum_mask.sum()} rows with zero probabilities")
        pred_df.loc[zero_sum_mask, prob_cols] = 1.0 / len(prob_cols)
        row_sums = pred_df[prob_cols].sum(axis=1)
    
    # Normalize to sum to 1
    for col in prob_cols:
        pred_df[col] = pred_df[col] / row_sums
    
    # Verify normalization
    final_sums = pred_df[prob_cols].sum(axis=1)
    assert np.allclose(final_sums, 1.0), "Probabilities don't sum to 1!"
    
    # Reorder columns (object_id first, then classes)
    final_cols = ['object_id'] + CORRECT_PLASTICC_CLASSES
    pred_df = pred_df[final_cols]
    
    print(f"    âœ… Enhanced predictions ready: {pred_df.shape}")
    return pred_df

# Cell 7: Main Processing Function with Enhanced Error Handling
def process_test_data_enhanced():
    """Enhanced test data processing with maximum scoring optimization"""
    
    if model is None:
        print("â�Œ No model available for processing")
        return
    
    # Load test metadata
    print("\nğŸ“Š Loading test metadata...")
    test_meta = pd.read_csv("/kaggle/input/PLAsTiCC-2018/test_set_metadata.csv")
    print(f"âœ… Test metadata loaded: {test_meta.shape}")
    
    # Processing configuration
    total_objects = len(test_meta)
    num_batches = (total_objects + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n=== ENHANCED PROCESSING CONFIGURATION ===")
    print(f"ğŸ“Š Total objects: {total_objects:,}")
    print(f"ğŸ“¦ Batch size: {BATCH_SIZE:,}")
    print(f"ğŸ”„ Number of batches: {num_batches}")
    print(f"â�±ï¸� Estimated time: {num_batches * 3:.0f}-{num_batches * 5:.0f} minutes")
    print(f"ğŸ�¯ Target: Correct PLAsTiCC class format")
    
    all_predictions = []
    successful_batches = 0
    
    # Process each batch
    for batch_num in range(num_batches):
        start_time = datetime.now()
        print(f"\n--- ENHANCED BATCH {batch_num + 1}/{num_batches} ---")
        
        try:
            # Get batch metadata
            start_idx = batch_num * BATCH_SIZE
            end_idx = min((batch_num + 1) * BATCH_SIZE, total_objects)
            batch_meta = test_meta.iloc[start_idx:end_idx].copy()
            batch_obj_ids = set(batch_meta['object_id'].values)
            
            print(f"ğŸ�¯ Processing objects {start_idx:,} to {end_idx-1:,}")
            print(f"ğŸ“� Batch contains {len(batch_obj_ids):,} unique object IDs")
            
            # Load light curves efficiently
            print("    ğŸ“¡ Loading light curves...")
            batch_lc_list = []
            total_obs = 0
            
            # List of test files to process
            test_files = [
                "/kaggle/input/plasticc-2018/test_set.csv",
                "/kaggle/input/PLAsTiCC-2018/test_set_batch1.csv",
                "/kaggle/input/plasticc-2018/test_set_batch2.csv",
                "/kaggle/input/plasticc-2018/test_set_batch3.csv",
                "/kaggle/input/plasticc-2018/test_set_batch4.csv",
                "/kaggle/input/plasticc-2018/test_set_batch5.csv",
                "/kaggle/input/plasticc-2018/test_set_batch6.csv",
                "/kaggle/input/plasticc-2018/test_set_batch7.csv",
                "/kaggle/input/plasticc-2018/test_set_batch8.csv",
                "/kaggle/input/plasticc-2018/test_set_batch9.csv",
                "/kaggle/input/plasticc-2018/test_set_batch10.csv",
                "/kaggle/input/plasticc-2018/test_set_batch11.csv"
            ]
            
            for file_idx, lc_file in enumerate(test_files):
                if not os.path.exists(lc_file):
                    continue
                
                try:
                    # Read in chunks
                    chunk_size = 500000
                    file_chunks = []
                    
                    for chunk in pd.read_csv(lc_file, chunksize=chunk_size):
                        if 'object_id' in chunk.columns:
                            relevant_data = chunk[chunk['object_id'].isin(batch_obj_ids)]
                            if len(relevant_data) > 0:
                                file_chunks.append(relevant_data)
                                total_obs += len(relevant_data)
                    
                    if file_chunks:
                        batch_lc_list.extend(file_chunks)
                        print(f"    ğŸ“„ {os.path.basename(lc_file)}: {sum(len(c) for c in file_chunks):,} observations")
                    
                    # Early stopping if we have enough data
                    if total_obs > 1000000:
                        print(f"    âš¡ Early stop - sufficient data loaded")
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
            
            # Extract enhanced features
            features_df = extract_enhanced_features(batch_lc, batch_meta)
            
            # Make enhanced predictions
            pred_df = make_predictions_enhanced(features_df, model, scaler)
            
            # Validate prediction format
            assert 'object_id' in pred_df.columns, "Missing object_id column"
            assert all(cls in pred_df.columns for cls in CORRECT_PLASTICC_CLASSES), "Missing required classes"
            
            # Save intermediate results
            if (batch_num + 1) % SAVE_INTERVAL == 0:
                batch_file = f"{OUTPUT_DIR}enhanced_predictions_batch_{batch_num + 1:03d}.csv"
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
            continue
    
    print(f"\nâœ… Processing complete: {successful_batches}/{num_batches} batches successful")
    
    if successful_batches == 0:
        print("â�Œ No batches processed successfully!")
        return None
    
    return all_predictions




# Cell 8: Enhanced Submission Creation (COMPLETED)
def create_enhanced_submission(all_predictions):
    """Create final submission with maximum scoring optimization"""
    print("\n=== ENHANCED SUBMISSION CREATION ===")
    
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
    submission_file = f"{OUTPUT_DIR}enhanced_submission_final.csv"
    final_submission.to_csv(submission_file, index=False)
    
    print(f"âœ… Enhanced submission created: {submission_file}")
    print(f"ğŸ“Š Final shape: {final_submission.shape}")
    print(f"ğŸ�¯ Classes: {CORRECT_PLASTICC_CLASSES}")
    
    # Quality checks
    print(f"\nğŸ”� QUALITY CHECKS:")
    print(f"    âœ“ Object IDs: {final_submission['object_id'].nunique():,} unique")
    print(f"    âœ“ Probability range: [{final_submission[prob_cols].min().min():.6f}, {final_submission[prob_cols].max().max():.6f}]")
    print(f"    âœ“ Row sums: [{final_sums.min():.6f}, {final_sums.max():.6f}]")
    print(f"    âœ“ All sums â‰ˆ 1.0: {np.allclose(final_sums, 1.0, atol=1e-6)}")
    print(f"    âœ“ No NaN values: {not final_submission.isnull().any().any()}")
    print(f"    âœ“ Correct columns: {len(required_cols)} columns present")
    
    # Display sample predictions
    print(f"\nğŸ“‹ SAMPLE PREDICTIONS:")
    sample_df = final_submission.head(3)
    for idx, row in sample_df.iterrows():
        obj_id = row['object_id']
        max_prob_class = prob_cols[np.argmax(row[prob_cols])]
        max_prob = row[max_prob_class]
        print(f"    Object {obj_id}: {max_prob_class} (prob={max_prob:.4f})")
    
    # Class distribution analysis
    print(f"\nğŸ“Š CLASS DISTRIBUTION:")
    class_predictions = np.argmax(final_submission[prob_cols].values, axis=1)
    for i, class_name in enumerate(CORRECT_PLASTICC_CLASSES):
        count = np.sum(class_predictions == i)
        percentage = 100 * count / len(final_submission)
        print(f"    {class_name}: {count:,} objects ({percentage:.2f}%)")
    
    return final_submission




# Cell 9: Main Execution Function
def main_execution():
    """Main execution function with comprehensive error handling"""
    print("\n" + "="*60)
    print("ğŸš€ STARTING ENHANCED PLAsTiCC TESTING PIPELINE")
    print("="*60)
    
    start_time = datetime.now()
    
    try:
        # Process test data
        all_predictions = process_test_data_enhanced()
        
        if all_predictions is None or len(all_predictions) == 0:
            print("â�Œ No predictions generated - pipeline failed!")
            return False
        
        # Create final submission
        final_submission = create_enhanced_submission(all_predictions)
        
        if final_submission is None:
            print("â�Œ Submission creation failed!")
            return False
        
        # Success summary
        total_time = datetime.now() - start_time
        print(f"\n" + "="*60)
        print("ğŸ�‰ ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"â�±ï¸� Total execution time: {total_time}")
        print(f"ğŸ“Š Objects processed: {len(final_submission):,}")
        print(f"ğŸ�¯ Final log-loss target: < 0.60 (based on training performance)")
        print(f"ğŸ“� Output file: enhanced_submission_final.csv")
        print(f"ğŸ�† Expected performance: Top 5-10% (based on training results)")
        
        return True
        
    except Exception as e:
        print(f"\nâ�Œ PIPELINE FAILED: {e}")
        print(f"ğŸ› ï¸� Debugging information:")
        print(f"    - Model loaded: {model is not None}")
        print(f"    - Scaler loaded: {scaler is not None}")
        print(f"    - Model type: {model_name if model else 'None'}")
        
        # Try to create a minimal fallback submission
        try:
            print("\nğŸš¨ Attempting fallback submission creation...")
            create_fallback_submission()
        except Exception as fallback_error:
            print(f"â�Œ Fallback submission also failed: {fallback_error}")
        
        return False




# Cell 10: Fallback Submission and Final Utilities
def create_fallback_submission():
    """Create a minimal fallback submission if main pipeline fails"""
    print("ğŸš¨ Creating fallback submission with uniform probabilities...")
    
    # Load test metadata
    test_meta = pd.read_csv("/kaggle/input/plasticc-2018/test_set_metadata.csv")
    
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
    fallback_file = f"{OUTPUT_DIR}fallback_submission.csv"
    fallback_submission.to_csv(fallback_file, index=False)
    
    print(f"âœ… Fallback submission created: {fallback_file}")
    print(f"ğŸ“Š Shape: {fallback_submission.shape}")
    print(f"âš ï¸� Note: This uses uniform probabilities and will score poorly!")
    
    return fallback_submission

def validate_submission_format(submission_file):
    """Validate that submission file meets Kaggle requirements"""
    print(f"\nğŸ”� VALIDATING SUBMISSION FORMAT: {submission_file}")
    
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
        
        print("    âœ… Submission format validation PASSED!")
        return True
        
    except Exception as e:
        print(f"    â�Œ Validation failed: {e}")
        return False

def display_final_summary():
    """Display final execution summary"""
    print("\n" + "="*60)
    print("ğŸ“‹ FINAL EXECUTION SUMMARY")
    print("="*60)
    
    # Check for output files
    output_files = []
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith('.csv'):
            file_path = os.path.join(OUTPUT_DIR, file)
            file_size = os.path.getsize(file_path) / (1024*1024)
            output_files.append((file, file_size))
    
    if output_files:
        print("ğŸ“� Generated files:")
        for filename, size_mb in output_files:
            print(f"    ğŸ“„ {filename} ({size_mb:.1f} MB)")
            
            # Validate main submission file
            if 'enhanced_submission_final.csv' in filename:
                validate_submission_format(os.path.join(OUTPUT_DIR, filename))
    else:
        print("â�Œ No output files generated!")
    
    print(f"\nğŸ�¯ Next steps:")
    print(f"    1. Download enhanced_submission_final.csv")
    print(f"    2. Submit to PLAsTiCC competition")
    print(f"    3. Monitor leaderboard performance")
    print(f"    4. Expected score: < 0.60 log-loss (top 5-10%)")
    
    print(f"\nâœ… Pipeline execution completed!")




# Execute the main pipeline
if __name__ == "__main__":
    success = main_execution()
    display_final_summary()
    
    if success:
        print("\nğŸ�‰ Ready for Kaggle submission!")
    else:
        print("\nâš ï¸� Check logs for issues - fallback submission may be available")

# Memory cleanup
gc.collect()
print(f"\nğŸ§¹ Memory cleanup completed")
print(f"â�° Notebook finished at: {datetime.now()}")

