import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder 
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier 
from category_encoders import TargetEncoder
import pickle


#Custom MAP@3
def mapk(actual, predicted, k=3):
    score = 0.0
    for a, p in zip(actual, predicted):
        try:
            # Check if 'a' is in the top k predictions 'p'
            score += 1.0 / (p[:k].index(a) + 1)
        except ValueError:
            # 'a' is not in the top k predictions
            continue
    return score / len(actual)


 # Feature Pipeline (Modified for more advanced FE)
class FertilizerPipeline:
    def __init__(self):
        self.label_encoders = {}
        self.fitted = False
        self.one_hot_cols = ['Soil Type', 'Crop Type', 
                             'Temparature_bin', 'Humidity_bin', 'Moisture_bin', 
                             'Nitrogen_bin', 'Potassium_bin', 'Phosphorous_bin']
        self.target_encode_cols = ['Soil_Crop_Combo'] 
        self.ohe = None 
        # Store group-based statistics (e.g., for nutrient deviations)
        # This will now store dictionaries of Series: {'crop_means': {'Nitrogen': Series, 'Potassium': Series, ...}}
        self.group_stats = {'crop_means': {}, 'soil_means': {}, 'soil_crop_means': {}} 

    def _feature_engineering(self, df):
        """Applies various feature engineering steps.
        This runs AFTER initial data loading and numerical binning.
        """
        df = df.copy() 
        
        # Original features
        df['Total_Nutrients'] = df['Nitrogen'] + df['Potassium'] + df['Phosphorous']
        df['NPK_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + df['Potassium'] + 1e-5)
        df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
        df['K_P_ratio'] = df['Potassium'] / (df['Phosphorous'] + 1e-5)

        df['Moisture_Temp'] = df['Moisture'] * df['Temparature']
        df['Humidity_Temp'] = df['Humidity'] / (df['Temparature'] + 1e-5)
        df['Temp_Hum_Interaction'] = df['Temparature'] * df['Humidity']

        df['Heavy_Nitrogen'] = (df['Nitrogen'] > 35).astype(int)
        
        df['Nitrogen_Ratio'] = df['Nitrogen'] / (df['Total_Nutrients'] + 1e-5)
        df['Potassium_Ratio'] = df['Potassium'] / (df['Total_Nutrients'] + 1e-5)
        df['Phosphorous_Ratio'] = df['Phosphorous'] / (df['Total_Nutrients'] + 1e-5)
        
        df['Soil_Crop_Combo'] = df['Soil Type'] + '_' + df['Crop Type']

        # --- NEW & IMPROVED FEATURE ENGINEERING ---
        # 1. Nutrient Deviation from Crop Type / Soil Type / Soil_Crop_Combo Mean
        nutrient_cols = ['Nitrogen', 'Potassium', 'Phosphorous']
        
        if self.fitted: # Use pre-fitted stats for transformation (test/val data)
            for col in nutrient_cols:
                global_mean_col = df[col].mean() # Fallback global mean
                
                # Retrieve the specific Series for mapping
                crop_means_series = self.group_stats['crop_means'].get(col)
                soil_means_series = self.group_stats['soil_means'].get(col)
                soil_crop_means_series = self.group_stats['soil_crop_means'].get(col)
                
                # Apply map and fillna with global mean if category is unseen
                # Check if the series exists before mapping to prevent error if a nutrient column wasn't in training
                df[f'{col}_Dev_Crop'] = df[col] - (df['Crop Type'].map(crop_means_series).fillna(global_mean_col) if crop_means_series is not None else global_mean_col)
                df[f'{col}_Dev_Soil'] = df[col] - (df['Soil Type'].map(soil_means_series).fillna(global_mean_col) if soil_means_series is not None else global_mean_col)
                df[f'{col}_Dev_Soil_Crop'] = df[col] - (df['Soil_Crop_Combo'].map(soil_crop_means_series).fillna(global_mean_col) if soil_crop_means_series is not None else global_mean_col)
        else: # During initial fitting (on combined train+original data in dummy pipeline)
            # Calculate group means and store them as Pandas Series for direct mapping later
            for col in nutrient_cols:
                self.group_stats['crop_means'][col] = df.groupby('Crop Type')[col].mean()
                self.group_stats['soil_means'][col] = df.groupby('Soil Type')[col].mean()
                self.group_stats['soil_crop_means'][col] = df.groupby('Soil_Crop_Combo')[col].mean()
                
                # Apply transforms for the current DataFrame
                df[f'{col}_Dev_Crop'] = df[col] - df.groupby('Crop Type')[col].transform('mean')
                df[f'{col}_Dev_Soil'] = df[col] - df.groupby('Soil Type')[col].transform('mean')
                df[f'{col}_Dev_Soil_Crop'] = df[col] - df.groupby('Soil_Crop_Combo')[col].transform('mean')
        
        # 2. More interaction features
        df['N_Humidity_Interaction'] = df['Nitrogen'] * df['Humidity']
        df['P_Moisture_Interaction'] = df['Phosphorous'] * df['Moisture']
        df['K_Temp_Interaction'] = df['Potassium'] * df['Temparature']
        
        # 3. Nutrient Balance Score
        # Using squared differences to emphasize larger deviations
        df['Nutrient_Balance_Score'] = np.sqrt((df['Nitrogen_Ratio'] - df['Phosphorous_Ratio'])**2 + \
                                               (df['Phosphorous_Ratio'] - df['Potassium_Ratio'])**2 + \
                                               (df['Potassium_Ratio'] - df['Nitrogen_Ratio'])**2)
                                      
        # 4. Squared terms for key numerical features (selective)
        df['Temparature_sq'] = df['Temparature']**2
        df['Moisture_sq'] = df['Moisture']**2
        df['Humidity_sq'] = df['Humidity']**2
        
        return df

    def fit_transform_train(self, df, y=None):
        # This function will now also handle the initial fitting of group_stats
        # _feature_engineering is called, which handles populating group_stats when self.fitted is False
        df_fe = self._feature_engineering(df) 

        if y is not None:
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y)
            self.label_encoders['Fertilizer Name'] = le_target
        else:
            y_encoded = None
        
        self.fitted = True
        return df_fe, y_encoded

    def transform_data(self, df):
        if not self.fitted:
            raise Exception("Pipeline must be fitted first by calling fit_transform_train.")
        
        df_fe = self._feature_engineering(df) # This will use stored group_stats
        
        # Ensure one-hot columns are consistent type for transformation
        for col in self.one_hot_cols:
             if col in df_fe.columns:
                df_fe[col] = df_fe[col].astype(str)
        
        return df_fe

    def inverse_transform_target(self, y_encoded):
        if 'Fertilizer Name' not in self.label_encoders:
            raise Exception("Target LabelEncoder not fitted. Cannot inverse transform.")
        return self.label_encoders['Fertilizer Name'].inverse_transform(y_encoded)

# Function to bin numerical features into quantiles
def quantile_bin_encode(df, cols, q=5, labels=['very low', 'low', 'medium', 'high', 'very high']):
    df_transformed = df.copy()
    
    for col in cols:
        try:
            binned = pd.qcut(df_transformed[col], q=q, labels=labels, duplicates='drop')
        except ValueError:
            binned = pd.cut(df_transformed[col], bins=q, labels=labels, include_lowest=True)

        unique_binned_labels = binned.cat.categories
        actual_labels = [label for label in labels if label in unique_binned_labels]
        # Use existing labels list to ensure consistent mapping for OHE
        label_map = {label: idx for idx, label in enumerate(labels) if label in unique_binned_labels}
        
        df_transformed[f"{col}_bin"] = binned.map(label_map).astype('int64')
        
    return df_transformed


print("Starting pipeline execution...")

# --- 1. Load Data ---
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")

# --- 2. Initial Feature Engineering (Binning) ---
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.drop('id') 

train_binned = quantile_bin_encode(train, numerical_cols)
test_binned = quantile_bin_encode(test, numerical_cols)
original_binned = quantile_bin_encode(original, numerical_cols)


# --- 3. Initialize Pipeline and Prepare Dataframes for CV ---
pipeline = FertilizerPipeline()


# Prepare full training data for fitting the pipeline's internal group_stats
# (This includes both the competition train data and the external original data)


full_train_data_for_fe = pd.concat([train_binned.drop(columns=['Fertilizer Name', 'id']), original_binned.drop(columns=['Fertilizer Name'])], ignore_index=True)

# Call _feature_engineering on a dummy pipeline instance to pre-calculate group_stats
# We do this once on the full augmented training data before CV starts.


dummy_pipeline_for_stats = FertilizerPipeline()

# The call below will populate dummy_pipeline_for_stats.group_stats

dummy_pipeline_for_stats._feature_engineering(full_train_data_for_fe.copy()) 
pipeline.group_stats = dummy_pipeline_for_stats.group_stats # Copy these fitted stats to the main pipeline

# Now, apply _feature_engineering to X_train_base and X_test_base_original
# using the *fitted* group_stats (pipeline.fitted will be False initially for X_train_base's fit_transform)
# The first call to fit_transform_train will set pipeline.fitted = True


X_train_base_temp, y_encoded = pipeline.fit_transform_train(train_binned.drop(columns=['Fertilizer Name', 'id']), train_binned['Fertilizer Name'])


# Then, re-run _feature_engineering on X_train_base to apply the newly fitted group_stats
# This ensures that the original train data features also reflect the overall means


X_train_base = pipeline._feature_engineering(train_binned.drop(columns=['Fertilizer Name', 'id']))

X_test_base_original = pipeline.transform_data(test_binned.drop(columns=['id']))

original_y_encoded = pipeline.label_encoders['Fertilizer Name'].transform(original_binned['Fertilizer Name'])



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

num_classes = len(np.unique(y_encoded))
xgb_preds = np.zeros((X_test_base_original.shape[0], num_classes))

f1_scores = []
map3_scores = []

print(f"Number of classes: {num_classes}")
print("Starting K-fold cross-validation with a single XGBoost Classifier and advanced FE...")

# Define the columns that need OHE *after* binning
ohe_feature_cols = pipeline.one_hot_cols
# Columns to exclude from final feature set for XGBoost (these will be replaced by OHE features)
cols_to_drop_after_encoding = ohe_feature_cols + pipeline.target_encode_cols


for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_base, y_encoded)):
    print(f"\n--- Fold {fold + 1} ---")
    
    # Ensure to get fresh copies for each fold to prevent unintended modifications
    X_train_fold_base = X_train_base.iloc[train_idx].copy()
    X_val_fold = X_train_base.iloc[val_idx].copy()
    y_train_fold_base = y_encoded[train_idx]
    y_val_fold = y_encoded[val_idx] 

    # --- Data Augmentation ---
    # original_features_aligned will use the pipeline's fitted group_stats
    original_features_aligned = pipeline.transform_data(original_binned.drop(columns=['Fertilizer Name']))
    
    X_train_fold = pd.concat([X_train_fold_base, original_features_aligned], ignore_index=True)
    y_train_fold = np.concatenate([y_train_fold_base, original_y_encoded])

    X_test_fold_transformed = X_test_base_original.copy() # Uses pipeline's fitted group_stats

    # Apply One-Hot Encoding
    # Ensure columns are string type before OHE
    for col in ohe_feature_cols:
        if col in X_train_fold.columns:
            X_train_fold[col] = X_train_fold[col].astype(str)
            X_val_fold[col] = X_val_fold[col].astype(str)
            X_test_fold_transformed[col] = X_test_fold_transformed[col].astype(str)
    
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # Fit OHE on the augmented training data (X_train_fold)
    ohe.fit(X_train_fold[ohe_feature_cols])

    X_train_ohe = pd.DataFrame(ohe.transform(X_train_fold[ohe_feature_cols]),
                               columns=ohe.get_feature_names_out(ohe_feature_cols),
                               index=X_train_fold.index)
    X_val_ohe = pd.DataFrame(ohe.transform(X_val_fold[ohe_feature_cols]),
                             columns=ohe.get_feature_names_out(ohe_feature_cols),
                             index=X_val_fold.index)
    X_test_ohe = pd.DataFrame(ohe.transform(X_test_fold_transformed[ohe_feature_cols]),
                              columns=ohe.get_feature_names_out(ohe_feature_cols),
                              index=X_test_fold_transformed.index)

    # Apply Target Encoding
    # Fit TargetEncoder on the current fold's training data for specified columns
    for col in pipeline.target_encode_cols: 
        te = TargetEncoder(cols=[col], handle_unknown='value', handle_missing='value', smoothing=5.0) # Added smoothing
        
        X_train_fold[col] = te.fit_transform(X_train_fold[col], y_train_fold)
        X_val_fold[col] = te.transform(X_val_fold[col])
        X_test_fold_transformed[col] = te.transform(X_test_fold_transformed[col])
    
    # Drop original OHE and target encoded columns from the feature sets and concatenate OHE features
    # Ensure all numerical features are included, including the new FE ones.
    # We need to filter out columns that are *not* the OHE or TE cols.
    non_encoded_numerical_cols = [col for col in X_train_fold.columns if col not in cols_to_drop_after_encoding]

    X_train_processed = pd.concat([X_train_fold[non_encoded_numerical_cols], X_train_ohe], axis=1)
    X_val_processed = pd.concat([X_val_fold[non_encoded_numerical_cols], X_val_ohe], axis=1)
    X_test_processed = pd.concat([X_test_fold_transformed[non_encoded_numerical_cols], X_test_ohe], axis=1)

    # Sanitize column names for XGBoost (replace special characters)
    X_train_processed.columns = X_train_processed.columns.str.replace(r'[\[\]<>]', '_', regex=True)
    X_val_processed.columns = X_val_processed.columns.str.replace(r'[\[\]<>]', '_', regex=True)
    X_test_processed.columns = X_test_processed.columns.str.replace(r'[\[\]<>]', '_', regex=True)


    # --- Model Initialization and Training (Only XGBoost) ---
    xgb = XGBClassifier(objective='multi:softprob',
                        use_label_encoder=False, # Deprecated but harmless
                        eval_metric='mlogloss',  # Metric for early stopping
                        tree_method='hist',      # Use histogram-based tree for speed/memory
                        random_state=42,
                        n_estimators=10000,          # Increased n_estimators further
                        learning_rate=0.015,         # Reduced learning rate
                        max_depth=12,                # Increased max_depth to capture more complexity
                        subsample=0.8,               # Adjusted subsample
                        colsample_bytree=0.8,        # Adjusted colsample_bytree
                        min_child_weight=1,          # Added min_child_weight (controls overfitting)
                        gamma=0.05,                  # Added gamma (min loss reduction to split)
                        reg_alpha=0.1,               # L1 regularization
                        n_jobs=-1)

    print("Training XGBoost model...")
    xgb.fit(X_train_processed, y_train_fold,
            eval_set=[(X_val_processed, y_val_fold)], 
            early_stopping_rounds=200,           # Increased early stopping rounds
            verbose=200)                         # Print progress every 200 rounds

    print("Generating predictions...")
    y_val_pred_xgb = xgb.predict_proba(X_val_processed) 
    
    y_val_ensemble_probs = y_val_pred_xgb 
    y_val_ensemble_classes = np.argmax(y_val_ensemble_probs, axis=1)

    f1_macro = f1_score(y_val_fold, y_val_ensemble_classes, average='macro') 
    f1_scores.append(f1_macro)
    
    top3_preds = np.argsort(y_val_ensemble_probs, axis=1)[:, ::-1][:, :3]
    map3 = mapk(y_val_fold.tolist(), top3_preds.tolist(), k=3) 
    map3_scores.append(map3)

    print(f"F1 (macro): {f1_macro:.4f} | MAP@3: {map3:.4f}")

    xgb_preds += xgb.predict_proba(X_test_processed) / skf.n_splits

print("\nK-fold cross-validation complete.")

# --- Final CV Results ---
print("\n***** Final CV Results *****")
print(f"Avg F1: {np.mean(f1_scores):.4f}")
print(f"Avg MAP@3: {np.mean(map3_scores):.4f}")

# --- Final Prediction for Submission ---
final_preds = xgb_preds 

top_3_preds_indices = np.argsort(final_preds, axis=1)[:, ::-1][:, :3]

top_3_labels = [
    [pipeline.inverse_transform_target([i])[0] for i in row]
    for row in top_3_preds_indices
]

submission = pd.DataFrame({
    "ID": test["id"],
    "Fertilizer Name": [" ".join(row) for row in top_3_labels]
})

submission.to_csv("submission_single_xgboost_advanced_fe_tuned_v3.csv", index=False)
print("Single XGBoost model submission with advanced FE and more aggressive tuning saved as submission_single_xgboost_advanced_fe_tuned_v3.csv!")




