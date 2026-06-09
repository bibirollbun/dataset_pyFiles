# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import pandas as pd
# pd.options.mode.copy_on_write = True
# from tqdm.auto import tqdm
# tqdm.pandas()
# import numpy as np
# import random
# import xgboost as xgb
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from typing import List
# import warnings
# warnings.simplefilter('ignore')

# def fast_map_k(actual: List[List], predicted: List[List], k: int = 3) -> float:
#     total_score = 0.0
    
#     for true_items, pred_items in zip(actual, predicted):
#         if not true_items:
#             continue
            
#         pred_items = pred_items[:k]
#         true_set = set(true_items)
        
#         # Create boolean mask for hits
#         hits = np.array([item in true_set for item in pred_items])
        
#         if not hits.any():
#             continue
        
#         # Calculate cumulative hits and positions
#         cumulative_hits = np.cumsum(hits)
#         positions = np.arange(1, len(pred_items) + 1)
        
#         # Calculate precision at each hit position
#         precisions = cumulative_hits[hits] / positions[hits]
#         score = np.sum(precisions) / min(len(true_items), k)
#         total_score += score
    
#     return total_score / len(actual)

# # Load data
# df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
# df_orginal = pd.read_csv('/kaggle/input/fertilizerprediction/fertilizer-prediction.csv')
# df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# df_train.drop(columns=['id'], inplace=True)
# df_test.drop(columns=['id'], inplace=True)

# # make categorical features from numericals
# for col in df_test.select_dtypes(include=['number']).columns:
#     df_train[f"cat_{col}"] = df_train[col].astype(str)
#     df_orginal[f"cat_{col}"] = df_orginal[col].astype(str)
#     df_test[f"cat_{col}"] = df_test[col].astype(str)

# # add a const features
# df_train['const'] = 1
# df_orginal['const'] = 1
# df_test['const'] = 1

# # Convert categorical columns to 'category' dtype
# for col in df_test.select_dtypes(include=['object']).columns:
#     df_train[col] = df_train[col].astype('category')
#     df_orginal[col] = df_orginal[col].astype('category')
#     df_test[col] = df_test[col].astype('category')

# # Get Target column
# target = df_train.pop('Fertilizer Name')
# target_org = df_orginal.pop('Fertilizer Name')

# # Encode target labels
# le = LabelEncoder()
# target = le.fit_transform(target)
# target_org = le.transform(target_org)

# # Ensemble configuration
# FOLDS = 5
# SEED = 42
# np.random.seed(SEED)
# random.seed(SEED)

# # Initialize storage for ensemble predictions
# all_oof_predictions = []
# all_test_predictions = []
# model_names = []

# # Model 1: XGBoost
# print("=" * 50)
# print("Training XGBoost Model")
# print("=" * 50)

# skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
# oof_xgb = np.zeros((len(df_train), np.unique(target).shape[0]))
# pred_test_xgb = np.zeros((len(df_test), np.unique(target).shape[0]))
# final_score_xgb = 0

# params_xgb = {
#     'objective': 'multi:softprob',
#     'num_class': 7,
#     'max_depth': 7,
#     'learning_rate': 0.01,
#     'n_estimators': 100_000,
#     'reg_alpha': 7,
#     'reg_lambda': 5.3,
#     'gamma': 0.3,
#     'max_delta_step': 4,
#     'subsample': 0.86,
#     'colsample_bytree': 0.4,
#     'min_child_weight': 5,
#     'random_state': SEED,
#     'eval_metric': 'mlogloss',
#     'enable_categorical': True,
#     'device': "cuda"
# }

# for i, (indx_train, indx_valid) in enumerate(skf.split(df_train, target)):
#     print(f"XGBoost Fold {i+1}")

#     X_train, y_train = df_train.iloc[indx_train], target[indx_train]
#     X_valid, y_valid = df_train.iloc[indx_valid], target[indx_valid]
#     X_test = df_test.copy()

#     X_train = pd.concat([X_train, df_orginal], axis=0)
#     y_train = np.concatenate([y_train, target_org], axis=0)

#     dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
#     dval = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
#     dtest = xgb.DMatrix(X_test, enable_categorical=True)

#     model = xgb.train(
#         params_xgb, 
#         dtrain, 
#         num_boost_round=100_000, 
#         evals=[(dtrain, 'train'), (dval, 'validation')], 
#         early_stopping_rounds=50, 
#         verbose_eval=5000
#     )

#     oof_xgb[indx_valid] = model.predict(dval)
#     pred_test_xgb += model.predict(dtest)

#     top_preds = np.argsort(oof_xgb[indx_valid], axis=1)[:, -3:][:, ::-1]  
#     score = fast_map_k([[label] for label in y_valid], top_preds)
#     final_score_xgb += score
#     print(f"XGBoost Fold {i+1} Score: {score:.5f}")

# pred_test_xgb /= FOLDS
# final_score_xgb /= FOLDS
# print(f"XGBoost Overall Score: {final_score_xgb:.5f}")

# all_oof_predictions.append(oof_xgb)
# all_test_predictions.append(pred_test_xgb)
# model_names.append('xgb')

# # Model 2: Random Forest
# print("=" * 50)
# print("Training Random Forest Model")
# print("=" * 50)

# oof_rf = np.zeros((len(df_train), np.unique(target).shape[0]))
# pred_test_rf = np.zeros((len(df_test), np.unique(target).shape[0]))
# final_score_rf = 0

# rf_params = {
#     'n_estimators': 200,
#     'max_depth': 15,
#     'min_samples_split': 5,
#     'min_samples_leaf': 2,
#     'random_state': SEED,
#     'n_jobs': -1
# }

# for i, (indx_train, indx_valid) in enumerate(skf.split(df_train, target)):
#     print(f"Random Forest Fold {i+1}")

#     X_train, y_train = df_train.iloc[indx_train], target[indx_train]
#     X_valid, y_valid = df_train.iloc[indx_valid], target[indx_valid]
#     X_test = df_test.copy()

#     X_train = pd.concat([X_train, df_orginal], axis=0)
#     y_train = np.concatenate([y_train, target_org], axis=0)

#     # Convert categorical columns to numeric for RF
#     X_train_rf = X_train.copy()
#     X_valid_rf = X_valid.copy()
#     X_test_rf = X_test.copy()
    
#     for col in X_train_rf.select_dtypes(include=['category']).columns:
#         X_train_rf[col] = X_train_rf[col].cat.codes
#         X_valid_rf[col] = X_valid_rf[col].cat.codes
#         X_test_rf[col] = X_test_rf[col].cat.codes

#     model = RandomForestClassifier(**rf_params)
#     model.fit(X_train_rf, y_train)

#     oof_rf[indx_valid] = model.predict_proba(X_valid_rf)
#     pred_test_rf += model.predict_proba(X_test_rf)

#     top_preds = np.argsort(oof_rf[indx_valid], axis=1)[:, -3:][:, ::-1]  
#     score = fast_map_k([[label] for label in y_valid], top_preds)
#     final_score_rf += score
#     print(f"Random Forest Fold {i+1} Score: {score:.5f}")

# pred_test_rf /= FOLDS
# final_score_rf /= FOLDS
# print(f"Random Forest Overall Score: {final_score_rf:.5f}")

# all_oof_predictions.append(oof_rf)
# all_test_predictions.append(pred_test_rf)
# model_names.append('rf')

# # Model 3: Logistic Regression
# print("=" * 50)
# print("Training Logistic Regression Model")
# print("=" * 50)

# oof_lr = np.zeros((len(df_train), np.unique(target).shape[0]))
# pred_test_lr = np.zeros((len(df_test), np.unique(target).shape[0]))
# final_score_lr = 0

# lr_params = {
#     'C': 1.0,
#     'max_iter': 1000,
#     'random_state': SEED,
#     'n_jobs': -1
# }

# for i, (indx_train, indx_valid) in enumerate(skf.split(df_train, target)):
#     print(f"Logistic Regression Fold {i+1}")

#     X_train, y_train = df_train.iloc[indx_train], target[indx_train]
#     X_valid, y_valid = df_train.iloc[indx_valid], target[indx_valid]
#     X_test = df_test.copy()

#     X_train = pd.concat([X_train, df_orginal], axis=0)
#     y_train = np.concatenate([y_train, target_org], axis=0)

#     # Convert categorical columns to numeric for LR
#     X_train_lr = X_train.copy()
#     X_valid_lr = X_valid.copy()
#     X_test_lr = X_test.copy()
    
#     for col in X_train_lr.select_dtypes(include=['category']).columns:
#         X_train_lr[col] = X_train_lr[col].cat.codes
#         X_valid_lr[col] = X_valid_lr[col].cat.codes
#         X_test_lr[col] = X_test_lr[col].cat.codes

#     model = LogisticRegression(**lr_params)
#     model.fit(X_train_lr, y_train)

#     oof_lr[indx_valid] = model.predict_proba(X_valid_lr)
#     pred_test_lr += model.predict_proba(X_test_lr)

#     top_preds = np.argsort(oof_lr[indx_valid], axis=1)[:, -3:][:, ::-1]  
#     score = fast_map_k([[label] for label in y_valid], top_preds)
#     final_score_lr += score
#     print(f"Logistic Regression Fold {i+1} Score: {score:.5f}")

# pred_test_lr /= FOLDS
# final_score_lr /= FOLDS
# print(f"Logistic Regression Overall Score: {final_score_lr:.5f}")

# all_oof_predictions.append(oof_lr)
# all_test_predictions.append(pred_test_lr)
# model_names.append('lr')

# # Ensemble: Stacking with Logistic Regression
# print("=" * 50)
# print("Training Stacking Ensemble")
# print("=" * 50)

# # Create meta-features for stacking
# meta_features_train = np.column_stack(all_oof_predictions)
# meta_features_test = np.column_stack(all_test_predictions)

# # Train meta-learner
# meta_learner = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
# meta_learner.fit(meta_features_train, target)

# # Get final ensemble predictions
# ensemble_pred_test = meta_learner.predict_proba(meta_features_test)

# # Calculate ensemble score on validation data
# ensemble_oof = meta_learner.predict_proba(meta_features_train)
# ensemble_score = 0

# for i, (indx_train, indx_valid) in enumerate(skf.split(df_train, target)):
#     top_preds = np.argsort(ensemble_oof[indx_valid], axis=1)[:, -3:][:, ::-1]  
#     score = fast_map_k([[label] for label in target[indx_valid]], top_preds)
#     ensemble_score += score

# ensemble_score /= FOLDS
# print(f"Ensemble Score: {ensemble_score:.5f}")

# # Print individual model scores
# print("\n" + "=" * 50)
# print("MODEL COMPARISON")
# print("=" * 50)
# for i, name in enumerate(model_names):
#     print(f"{name.upper()}: {[final_score_xgb, final_score_rf, final_score_lr][i]:.5f}")
# print(f"ENSEMBLE: {ensemble_score:.5f}")

# # Generate final predictions
# top_preds = np.argsort(ensemble_pred_test, axis=1)[:,-3:][:,::-1]
# top_labels = le.inverse_transform(top_preds.ravel()).reshape(top_preds.shape)

# df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
# df_sub['Fertilizer Name'] = [' '.join(row) for row in top_labels]

# df_sub.to_csv('submission.csv', index=False)
# print("\nSubmission file saved as 'submission.csv'")


# import numpy as np
# import pandas as pd
# import os
# from sklearn.preprocessing import LabelEncoder
# import cupy as cp
# import gc

# # --- Configuration ---
# # Prefix for the submission file name
# sub_name = 'hill_climb_ensemble'
# # Number of distinct fertilizer classes (used for reshaping predictions)
# N_CLASSES = 7

# # --- File Paths ---
# # Base directories where OOF and prediction files are located.
# # These paths are typical for Kaggle notebooks.
# base_paths = [
#     "/kaggle/input/predicting-optimal-fertilizers",
#     "/kaggle/input/predicting-fertilizer-name-stacking-ensemble/results",
#     "/kaggle/input/predicting-optimal-fertilizers-stacking/results",
#     "/kaggle/input/k/jaxa623/xgboost-repeatedstratifiedkfold/"
# ]

# # Specific OOF/PRED files mentioned and added separately in the original notebook.
# # These are treated as additional models in the ensemble.
# custom_oof_path_ps5e6 = "/kaggle/input/ps5e6-oof-scores/oof_ensemble.npy"
# custom_pred_path_ps5e6 = "/kaggle/input/ps5e6-oof-scores/pred.npy"
# # Assign unique logical names to these custom files for consistent mapping
# # These names are used as the 'logical_name' when explicitly adding these models.
# # The `get_model_base_name_for_pairing` function does not apply to these explicitly loaded files.
# LOGICAL_NAME_PS5E6_OOF = "ps5e6_oof_ensemble"
# LOGICAL_NAME_PS5E6_PRED = "ps5e6_pred_output"


# # --- Data Loading ---
# print("--- Starting Data Loading ---")

# # Load the training data to get true labels and initialize the LabelEncoder
# try:
#     train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
#     le = LabelEncoder()
#     # Fit LabelEncoder on the 'Fertilizer Name' from training data
#     encoded_labels = le.fit_transform(train_df['Fertilizer Name'])
#     # Store true labels (encoded) for OOF evaluation
#     true_labels = encoded_labels
# except FileNotFoundError:
#     print("Error: train.csv not found. Please ensure the dataset is correctly mounted.")
#     exit() # Exit if essential file is missing

# # Dictionaries to store full paths of OOF and prediction files,
# # mapped by a consistent base name for each model.
# # This part is for identifying *pairs* of OOF/PRED files from a single model.
# paired_oof_filepaths = {}
# paired_pred_filepaths = {}

# # Helper function to extract a consistent base name for a model from its filename.
# # This is crucial for matching OOF and prediction files that belong to the same model.
# def get_model_base_name_for_pairing(filename):
#     # Removes common suffixes like '_oof', '_test', '_pred', '_train' to find a common base
#     name = filename.lower().replace('.npy', '')
#     if name.endswith('_oof'):
#         return name.replace('_oof', '')
#     elif name.endswith('_test'):
#         return name.replace('_test', '')
#     elif name.endswith('_pred'):
#         return name.replace('_pred', '')
#     elif name.endswith('_train'): # Handle cases like xgb_repeat_train_oof.npy
#         # For 'xgb_repeat_train_oof', this would return 'xgb_repeat_train'.
#         # If a corresponding 'xgb_repeat_test_oof' exists, its base would also be 'xgb_repeat_test_oof'.
#         # This requires careful review of actual file naming if it's meant to be just 'xgb_repeat'.
#         # For now, it will strip just '_train'.
#         return name.replace('_train', '')
#     return name # Fallback if no specific suffix, might not pair correctly

# # Walk through all specified base paths to find .npy files
# for path in base_paths:
#     if not os.path.exists(path):
#         print(f"Warning: Path '{path}' does not exist. Skipping.")
#         continue
#     for root, _, files_in_dir in os.walk(path):
#         for file in files_in_dir:
#             filepath = os.path.join(root, file)
#             if not filepath.endswith('.npy'):
#                 continue

#             base_name = get_model_base_name_for_pairing(file)

#             # Assign to maps based on naming convention
#             if 'oof' in file.lower():
#                 paired_oof_filepaths[base_name] = filepath
#             elif 'test' in file.lower() or 'pred' in file.lower():
#                 paired_pred_filepaths[base_name] = filepath

# # Now, find models that have both an OOF and a PRED file based on the base_name
# paired_model_bases = sorted(list(set(paired_oof_filepaths.keys()) & set(paired_pred_filepaths.keys())))

# # Lists to store loaded OOF and prediction data, and their corresponding model names.
# x_train_oof_list = []
# x_test_pred_list = []
# oof_model_names = [] # Stores the logical names of models in the order they are loaded

# # Load data for explicitly paired models found through convention
# print(f"-> Loading generically paired models from base paths...")
# for model_base_name in paired_model_bases:
#     oof_filepath = paired_oof_filepaths[model_base_name]
#     pred_filepath = paired_pred_filepaths[model_base_name]

#     print(f"   - Loading paired model: '{model_base_name}'")
#     try:
#         oof_data = np.load(oof_filepath)
#         pred_data = np.load(pred_filepath)
#     except FileNotFoundError:
#         print(f"Warning: Missing NPY file for paired model '{model_base_name}'. Skipping.")
#         continue # Skip to next model if files are missing

#     # Ensure data is in (N_SAMPLES, N_CLASSES) shape before flattening
#     if oof_data.ndim == 1:
#         # Assuming 1D arrays are already N_SAMPLES * N_CLASSES
#         pass
#     elif oof_data.ndim == 2 and oof_data.shape[1] == N_CLASSES:
#         # Already (N_SAMPLES, N_CLASSES), just flatten
#         pass
#     else:
#         print(f"Warning: OOF data shape for '{model_base_name}' is {oof_data.shape}. Reshaping to (-1, {N_CLASSES}).")
#         oof_data = oof_data.reshape(-1, N_CLASSES)

#     if pred_data.ndim == 1:
#         pass
#     elif pred_data.ndim == 2 and pred_data.shape[1] == N_CLASSES:
#         pass
#     else:
#         print(f"Warning: PRED data shape for '{model_base_name}' is {pred_data.shape}. Reshaping to (-1, {N_CLASSES}).")
#         pred_data = pred_data.reshape(-1, N_CLASSES)

#     x_train_oof_list.append(oof_data.flatten())
#     x_test_pred_list.append(pred_data.flatten())
#     oof_model_names.append(model_base_name)

# # --- Explicitly load and add the specific OOF/PRED files from ps5e6-oof-scores ---
# # These are treated as independent models that are simply added to the ensemble candidates.
# print("\n-> Loading specific custom models from '/kaggle/input/ps5e6-oof-scores/'...")
# custom_files_to_load_explicitly = [
#     (custom_oof_path_ps5e6, LOGICAL_NAME_PS5E6_OOF),
#     (custom_pred_path_ps5e6, LOGICAL_NAME_PS5E6_PRED)
# ]

# for file_path, logical_name in custom_files_to_load_explicitly:
#     print(f"   - Loading custom model: '{logical_name}' from '{file_path}'")
#     try:
#         data = np.load(file_path)
#     except FileNotFoundError:
#         print(f"Warning: Missing custom NPY file for '{logical_name}'. Skipping.")
#         continue

#     # Reshape and append to the appropriate list
#     if data.ndim == 1:
#         pass
#     elif data.ndim == 2 and data.shape[1] == N_CLASSES:
#         pass
#     else:
#         print(f"Warning: Custom data shape for '{logical_name}' is {data.shape}. Reshaping to (-1, {N_CLASSES}).")
#         data = data.reshape(-1, N_CLASSES)

#     # Determine if it's an OOF or PRED and append to the correct list
#     # Assuming 'oof' in logical_name for OOF, else it's a PRED.
#     # This might need refinement if 'pred' in logical_name is also an OOF, etc.
#     if 'oof' in logical_name.lower():
#         x_train_oof_list.append(data.flatten())
#     else:
#         x_test_pred_list.append(data.flatten())
#     oof_model_names.append(logical_name)


# # --- Check if any models were loaded ---
# if not x_train_oof_list:
#     raise ValueError("Error: No OOF/PRED arrays were successfully loaded. Please check file paths, naming conventions, and existence.")
# # If x_train_oof_list is populated, but x_test_pred_list is not (or vice-versa), this would also cause issues.
# # A more robust check might compare lengths or ensure a balanced set is added.
# if len(x_train_oof_list) != len(x_test_pred_list):
#     print("Warning: Mismatch in number of OOF and PRED files loaded. This might indicate an issue with file pairing/loading.")
#     # For now, we proceed but this could lead to errors later if the shapes don't align.


# # Stack the loaded 1D arrays. Transpose to make each column a model's predictions.
# # x_train will have shape (num_samples * N_CLASSES, num_models)
# x_train = np.stack(x_train_oof_list).T
# x_test = np.stack(x_test_pred_list).T

# print("\n--- Data Loading Summary ---")
# print(f"Combined OOF (x_train) shape: {x_train.shape}")
# print(f"Combined PRED (x_test) shape: {x_test.shape}")
# print(f"Order of models loaded: {oof_model_names}")
# if x_train.shape[1] != len(oof_model_names) or x_test.shape[1] != len(oof_model_names):
#     print("CRITICAL WARNING: Mismatch in number of loaded models and data columns. Review data loading logic.")


# # --- Evaluation Functions (using CuPy for GPU acceleration) ---
# def map3_score_single(actual_labels, predicted_flat_array, n_classes):
#     """
#     Calculates Mean Average Precision @ 3 (MAP@3) for a single model's predictions.
#     Leverages CuPy for GPU acceleration.

#     Args:
#         actual_labels (cp.ndarray or np.ndarray): True labels (1D array).
#         predicted_flat_array (cp.ndarray or np.ndarray): Flattened predicted probabilities (1D array).
#         n_classes (int): Number of classes.

#     Returns:
#         float: The calculated MAP@3 score.
#     """
#     actual_cp = cp.asarray(actual_labels)
#     predicted_flat_cp = cp.asarray(predicted_flat_array)

#     n_samples = actual_cp.shape[0]
#     # Reshape flattened predictions to (n_samples, n_classes) for easy processing
#     pred_cp = predicted_flat_cp.reshape(n_samples, n_classes)

#     # Get the indices of the top 3 predicted classes for each sample
#     # argsort with negative values gives indices in descending order
#     top3_indices = cp.argsort(-pred_cp, axis=1)[:, :3]

#     # Check if the actual label is present in the top 3 predictions
#     # `actual_cp[:, cp.newaxis]` broadcasts actual labels to match top3_indices's shape
#     matches = (top3_indices == actual_cp[:, cp.newaxis])

#     # Define weights for MAP@3: 1 for 1st, 0.5 for 2nd, 1/3 for 3rd position
#     weights_map3 = cp.array([1.0, 0.5, 1.0/3], dtype=cp.float32)
#     # `weights_map3[cp.newaxis, :]` broadcasts weights to match matches's shape
#     scores_per_sample = cp.sum(matches * weights_map3[cp.newaxis, :], axis=1)

#     # Calculate the mean MAP@3 score across all samples and convert to CPU float
#     score = cp.mean(scores_per_sample).item()

#     # Clean up CuPy memory
#     del actual_cp, predicted_flat_cp, pred_cp, top3_indices, matches, weights_map3, scores_per_sample
#     gc.collect()
#     return score

# def multiple_map3_scores(actual_labels, predicted_flat_matrix, n_classes):
#     """
#     Calculates Mean Average Precision @ 3 (MAP@3) for predictions from multiple models
#     (e.g., for different weight combinations in hill climbing).
#     Leverages CuPy for GPU acceleration.

#     Args:
#         actual_labels (cp.ndarray): True labels (1D array).
#         predicted_flat_matrix (cp.ndarray): Matrix of flattened predicted probabilities,
#                                             shape (num_samples * n_classes, num_models_or_combinations).
#         n_classes (int): Number of classes.

#     Returns:
#         cp.ndarray: A 1D CuPy array of MAP@3 scores, one for each model/combination.
#     """
#     actual_cp = cp.asarray(actual_labels)
#     predicted_flat_cp = cp.asarray(predicted_flat_matrix)

#     n_samples = actual_cp.shape[0]
#     num_models_or_combinations = predicted_flat_cp.shape[1]

#     weights_map3 = cp.array([1.0, 0.5, 1.0/3], dtype=cp.float32)
#     scores_all_models = cp.empty(num_models_or_combinations, dtype=cp.float32)

#     for i in range(num_models_or_combinations):
#         # Reshape current model's flattened predictions to (n_samples, n_classes)
#         pred_cp = predicted_flat_cp[:, i].reshape(n_samples, n_classes)

#         # Get top 3 predicted classes
#         top3_indices = cp.argsort(-pred_cp, axis=1)[:, :3]

#         # Check for matches with actual labels
#         matches = (top3_indices == actual_cp[:, cp.newaxis])

#         # Calculate weighted sum for each sample
#         scores_per_sample = cp.sum(matches * weights_map3[cp.newaxis, :], axis=1)

#         # Calculate mean MAP@3 for this model/combination
#         scores_all_models[i] = cp.mean(scores_per_sample)

#         # Clean up CuPy memory for this iteration
#         del pred_cp, top3_indices, matches, scores_per_sample
#         gc.collect()

#     del actual_cp, predicted_flat_cp, weights_map3
#     gc.collect()
#     return scores_all_models


# # --- Single Model Evaluation ---
# print("\n--- Evaluating Single Models MAP@3 Scores ---")
# best_score_single_model = 0
# best_model_index = -1

# # Iterate through each loaded OOF model and evaluate its MAP@3 score
# for k, name in enumerate(oof_model_names):
#     s = map3_score_single(true_labels, x_train[:, k], N_CLASSES)
#     if s > best_score_single_model:
#         best_score_single_model = s
#         best_model_index = k
#     print(f"MAP@3 {s:0.5f} - '{name}'")

# print(f"\nBest single model is '{oof_model_names[best_model_index]}' with MAP@3 = {best_score_single_model:0.5f}")


# # --- Hill Climbing Ensemble ---
# print("\n--- Starting Hill Climbing Ensemble Optimization ---")

# # Configuration for hill climbing
# USE_NEGATIVE_WGT = True  # Allows negative weights (can help for diverse models)
# MAX_MODELS = 1000        # Maximum number of unique models to include in the ensemble
# TOL = 1e-5               # Tolerance for score improvement to stop the climbing process

# # Initialize the ensemble with the best performing single model
# # 'models_in_ensemble_history' stores the index of models chosen at each step
# models_in_ensemble_history = [best_model_index]
# # 'ensemble_weights_applied_history' stores the specific weight (w) applied at each step
# ensemble_weights_applied_history = []
# # 'ensemble_map3_scores_history' stores the MAP@3 score of the ensemble after each step
# ensemble_map3_scores_history = [best_score_single_model]

# # The current best ensemble's predictions (on OOF data), stored on GPU for speed
# current_best_ensemble_predictions_gpu = cp.array(x_train[:, best_model_index])

# print(f"0: Initial ensemble MAP@3 {best_score_single_model:0.5f} (from '{oof_model_names[best_model_index]}')")

# # Define the range of weights to try for combining current ensemble with a new model
# start_weight = -0.50 if USE_NEGATIVE_WGT else 0.02
# weights_to_explore = cp.arange(start_weight, 0.51, 0.02)
# num_weights_to_explore = len(weights_to_explore)

# # Move true labels and x_train data to GPU once for faster access within the loop
# true_labels_gpu = cp.array(true_labels)
# x_train_gpu = cp.array(x_train) # All OOF predictions on GPU

# # Hill Climbing Loop
# for iteration in range(2_000_000): # High iteration limit, but typically stops early by tolerance
#     iteration_best_map3 = 0.0
#     iteration_best_model_idx = -1
#     iteration_best_weight_value = 0.0
#     # Stores the predictions of the *potential* new best ensemble (if found)
#     potential_next_ensemble_predictions_gpu = None

#     # Iterate through all available base models to find the best one to add
#     for candidate_model_idx, _ in enumerate(oof_model_names):
#         candidate_model_predictions_gpu = x_train_gpu[:, candidate_model_idx]

#         # Create combinations: current_ensemble * (1-w) + candidate_model * w
#         # cp.newaxis is used for broadcasting (N_SAMPLES*N_CLASSES, 1) to (N_SAMPLES*N_CLASSES, N_WEIGHTS_TO_EXPLORE)
#         combined_m1 = cp.repeat(current_best_ensemble_predictions_gpu[:, cp.newaxis], num_weights_to_explore, axis=1) * (1 - weights_to_explore)
#         combined_m2 = cp.repeat(candidate_model_predictions_gpu[:, cp.newaxis], num_weights_to_explore, axis=1) * weights_to_explore
#         all_candidate_combinations = combined_m1 + combined_m2

#         # Evaluate MAP@3 for all combinations with this candidate model
#         map3_scores_for_this_candidate = multiple_map3_scores(true_labels_gpu, all_candidate_combinations, N_CLASSES)
#         current_candidate_max_map3 = cp.max(map3_scores_for_this_candidate).item() # Get max score and move to CPU

#         # If this candidate model, with its best weight, improves the ensemble score
#         if current_candidate_max_map3 > iteration_best_map3:
#             iteration_best_map3 = current_candidate_max_map3
#             iteration_best_model_idx = candidate_model_idx
#             # Find the weight that yielded this best score
#             best_weight_idx_for_candidate = cp.argmax(map3_scores_for_this_candidate).item()
#             iteration_best_weight_value = weights_to_explore[best_weight_idx_for_candidate].item()
#             # Store the actual predictions of this new potential best ensemble
#             potential_next_ensemble_predictions_gpu = all_candidate_combinations[:, best_weight_idx_for_candidate]

#         # Clear CuPy memory for intermediate arrays
#         del candidate_model_predictions_gpu, combined_m1, combined_m2, all_candidate_combinations, map3_scores_for_this_candidate
#         gc.collect()

#     # --- Stopping Criteria ---
#     # Append the best found model index to the ensemble history
#     models_in_ensemble_history.append(iteration_best_model_idx)

#     # Check for improvement against the previous best ensemble score
#     score_improvement = iteration_best_map3 - ensemble_map3_scores_history[-1]

#     # If improvement is below tolerance, stop climbing
#     if score_improvement < TOL:
#         print(f"-> Improvement ({score_improvement:0.7f}) below tolerance {TOL}. Stopping hill climbing.")
#         models_in_ensemble_history.pop() # Remove the last added model as it didn't significantly improve
#         break

#     # Check if we exceeded the maximum allowed unique models
#     unique_models_count = len(np.unique(models_in_ensemble_history))
#     if unique_models_count > MAX_MODELS:
#         print(f"-> Reached {MAX_MODELS} unique models. Stopping hill climbing.")
#         models_in_ensemble_history.pop() # Revert last addition
#         break

#     # --- Record New Best Result and Update Current Ensemble ---
#     print(f"{iteration + 1}: New best MAP@3 {iteration_best_map3:0.7f} (adding '{oof_model_names[iteration_best_model_idx]}' with weight {iteration_best_weight_value:0.3f})")
#     ensemble_weights_applied_history.append(iteration_best_weight_value)
#     ensemble_map3_scores_history.append(iteration_best_map3)
#     current_best_ensemble_predictions_gpu = potential_next_ensemble_predictions_gpu


# # --- Calculate Final Ensemble Weights ---
# print("\n--- Calculating Final Ensemble Model Weights ---")

# # This calculates the effective contribution of each individual model to the final ensemble.
# # It reconstructs the sequential weighting applied during the hill-climbing process.
# final_model_effective_contributions = {}

# # The very first model in the history has a base contribution of 1.0
# first_model_name = oof_model_names[models_in_ensemble_history[0]]
# final_model_effective_contributions[first_model_name] = 1.0

# # Iterate through the history of added models and their weights
# for i in range(len(ensemble_weights_applied_history)):
#     added_model_idx = models_in_ensemble_history[i + 1] # Index of the model added in this step
#     weight_of_added_model = ensemble_weights_applied_history[i] # Weight (w) it was added with

#     # The contribution of ALL previously existing models in the ensemble
#     # is scaled by (1 - weight_of_added_model).
#     for model_name_in_ensemble in list(final_model_effective_contributions.keys()): # Use list to avoid RuntimeError during dict modification
#         final_model_effective_contributions[model_name_in_ensemble] *= (1 - weight_of_added_model)

#     # The newly added model's contribution is its direct weight
#     model_name_to_add = oof_model_names[added_model_idx]
#     if model_name_to_add not in final_model_effective_contributions:
#         final_model_effective_contributions[model_name_to_add] = 0.0
#     final_model_effective_contributions[model_name_to_add] += weight_of_added_model

# # Convert the dictionary of effective contributions to a pandas DataFrame for display and use.
# df_final_weights = pd.DataFrame(list(final_model_effective_contributions.items()),
#                                  columns=['model', 'weight'])
# df_final_weights = df_final_weights.sort_values('weight', ascending=False).reset_index(drop=True)

# print(df_final_weights)

# # Sanity check: the sum of weights should be close to 1.0
# print(f"\nSanity Check: Sum of final ensemble weights = {df_final_weights.weight.sum():0.3f}")


# # --- Final Ensemble Evaluation on OOF Data ---
# print("\n--- Evaluating Final Ensemble on OOF Data ---")

# # Move x_train data from GPU back to CPU for easier NumPy operations if needed,
# # though direct CuPy operations can also be used if entire pipeline is GPU-accelerated.
# x_train_cpu = x_train_gpu.get()

# # Initialize an array for the final ensemble OOF predictions
# final_ensemble_oof_predictions = np.zeros_like(x_train_cpu[:, 0], dtype=np.float32)

# # Create a mapping from model name to its column index in x_train/x_test
# model_name_to_column_map = {name: idx for idx, name in enumerate(oof_model_names)}

# # Apply the calculated final weights to create the ensemble OOF predictions
# for _, row in df_final_weights.iterrows():
#     model_name = row['model']
#     weight = row['weight']
#     col_idx = model_name_to_column_map[model_name]
#     final_ensemble_oof_predictions += x_train_cpu[:, col_idx] * weight

# # Evaluate the final ensemble's MAP@3 score
# final_ensemble_map3_score = map3_score_single(true_labels, cp.asarray(final_ensemble_oof_predictions), N_CLASSES)
# print(f"Overall Hill Climbing Ensemble MAP@3 = {final_ensemble_map3_score:0.7f}")

# # Optional: Save the combined OOF predictions to a CSV file (uncomment if needed)
# # pd.DataFrame({sub_name + '_oof': final_ensemble_oof_predictions.flatten()}).to_csv(f'{sub_name}_oof.csv', index=False)


# # --- Generate Submission File ---
# print("\n--- Generating Submission File ---")

# # Initialize an array for the final ensemble test predictions
# final_ensemble_test_predictions = np.zeros_like(x_test[:, 0], dtype=np.float32)

# # Apply the same final weights to create the ensemble test predictions
# for _, row in df_final_weights.iterrows():
#     model_name = row['model']
#     weight = row['weight']
#     col_idx = model_name_to_column_map[model_name]
#     final_ensemble_test_predictions += x_test[:, col_idx] * weight

# # Reshape the final ensemble predictions to (num_test_samples, N_CLASSES)
# # The initial flatten() and stack() operations mean the data is
# # (num_samples * N_CLASSES, num_models).
# # To get (num_test_samples, N_CLASSES), we need to reshape.
# # x_test_cpu.shape[0] is (250000 * 7) if test data has 250000 samples.
# num_test_samples = x_test.shape[0] // N_CLASSES
# final_ensemble_test_predictions_reshaped = final_ensemble_test_predictions.reshape(num_test_samples, N_CLASSES)

# # Get the top 3 predicted class indices for each test sample
# # argsort with negative values gives indices in descending order.
# # [:, -3:] slices the last 3 (top 3) elements.
# # [:, ::-1] reverses the order to be highest probability first.
# top3_predicted_indices = np.argsort(final_ensemble_test_predictions_reshaped, axis=1)[:, -3:][:, ::-1]

# # Convert the top 3 indices back to original Fertilizer Name categories
# # `flatten()` makes it 1D, `inverse_transform` converts indices to labels,
# # then `reshape` back to (num_test_samples, 3).
# top3_predicted_categories = le.inverse_transform(top3_predicted_indices.flatten()).reshape(-1, 3)

# # Load the sample submission file
# try:
#     submission_df = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
# except FileNotFoundError:
#     print("Error: sample_submission.csv not found. Cannot create submission file.")
#     exit()

# # Populate the 'Fertilizer Name' column with space-separated top 3 categories
# submission_df['Fertilizer Name'] = [' '.join(row) for row in top3_predicted_categories]

# # Define the output submission filename
# submission_filename = f'{sub_name}_submission.csv'
# # Save the submission file
# submission_df.to_csv(submission_filename, index=False)

# print(f"Submission file '{submission_filename}' created successfully. Happy Learning!")



# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import pandas as pd
# pd.options.mode.copy_on_write = True
# from tqdm.auto import tqdm
# tqdm.pandas()
# import numpy as np
# import random
# import xgboost as xgb
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from typing import List
# import warnings
# warnings.simplefilter('ignore')

# def fast_map_k(actual: List[List], predicted: List[List], k: int = 3) -> float:
#     """
#     Calculates the Mean Average Precision at K (MAP@K).

#     Args:
#         actual (List[List]): A list of lists, where each inner list contains the true labels.
#         predicted (List[List]): A list of lists, where each inner list contains the predicted labels.
#         k (int): The number of top predictions to consider.

#     Returns:
#         float: The MAP@K score.
#     """
#     total_score = 0.0
    
#     for true_items, pred_items in zip(actual, predicted):
#         if not true_items:
#             continue
            
#         pred_items = pred_items[:k]
#         true_set = set(true_items)
        
#         # Create boolean mask for hits
#         hits = np.array([item in true_set for item in pred_items])
        
#         if not hits.any():
#             continue
        
#         # Calculate cumulative hits and positions
#         cumulative_hits = np.cumsum(hits)
#         positions = np.arange(1, len(pred_items) + 1)
        
#         # Calculate precision at each hit position
#         precisions = cumulative_hits[hits] / positions[hits]
#         score = np.sum(precisions) / min(len(true_items), k)
#         total_score += score
    
#     return total_score / len(actual)

# # Load data
# df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
# df_orginal = pd.read_csv('/kaggle/input/fertilizerprediction/fertilizer-prediction.csv')
# df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# df_train.drop(columns=['id'], inplace=True)
# df_test.drop(columns=['id'], inplace=True)

# # make categorical features from numericals
# for col in df_test.select_dtypes(include=['number']).columns:
#     df_train[f"cat_{col}"] = df_train[col].astype(str)
#     df_orginal[f"cat_{col}"] = df_orginal[col].astype(str)
#     df_test[f"cat_{col}"] = df_test[col].astype(str)

# # add a const feature
# df_train['const'] = 1
# df_orginal['const'] = 1
# df_test['const'] = 1

# # Convert categorical columns to 'category' dtype
# # This is crucial for XGBoost's enable_categorical=True and for consistent preprocessing
# for col in df_test.select_dtypes(include=['object']).columns:
#     df_train[col] = df_train[col].astype('category')
#     df_orginal[col] = df_orginal[col].astype('category')
#     df_test[col] = df_test[col].astype('category')

# # Get Target column
# target = df_train.pop('Fertilizer Name')
# target_org = df_orginal.pop('Fertilizer Name')

# # Encode target labels
# le = LabelEncoder()
# target = le.fit_transform(target)
# target_org = le.transform(target_org) # Use the same encoder for original dataset target

# # Ensemble configuration
# FOLDS = 5
# SEED = 42
# np.random.seed(SEED)
# random.seed(SEED)

# # Initialize storage for ensemble predictions
# all_oof_predictions = []
# all_test_predictions = []
# model_names = []

# # Get list of categorical and numerical features for preprocessing pipelines
# # Exclude 'const' if it's not a true categorical or numerical feature, or treat it as numerical
# numerical_features = df_train.select_dtypes(include=np.number).columns.tolist()
# categorical_features = df_train.select_dtypes(include='category').columns.tolist()

# # Define preprocessor for models that need one-hot encoding and scaling (RF, LR)
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numerical_features),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#     ],
#     remainder='passthrough' # Keep other columns (like 'const' if not explicitly handled)
# )

# # Model 1: XGBoost
# print("=" * 50)
# print("Training XGBoost Model")
# print("=" * 50)
# skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
# oof_xgb = np.zeros((len(df_train), np.unique(target).shape[0]))
# pred_test_xgb = np.zeros((len(df_test), np.unique(target).shape[0]))
# final_score_xgb = 0

# params_xgb = {
#     'objective': 'multi:softprob',
#     'num_class': 7,
#     'max_depth': 7,
#     'learning_rate': 0.01,
#     'n_estimators': 100_000,
#     'reg_alpha': 7,
#     'reg_lambda': 5.3,
#     'gamma': 0.3,
#     'max_delta_step': 4,
#     'subsample': 0.86,
#     'colsample_bytree': 0.4,
#     'min_child_weight': 5,
#     'random_state': SEED,
#     'eval_metric': 'mlogloss',
#     'enable_categorical': True, # XGBoost can handle categorical features directly
#     'device': "cuda" # Ensure CUDA is properly installed and configured for this to work
# }

# for i, (indx_train, indx_valid) in enumerate(skf.split(df_train, target)):
#     print(f"XGBoost Fold {i+1}")

#     X_train_fold, y_train_fold = df_train.iloc[indx_train], target[indx_train]
#     X_valid_fold, y_valid_fold = df_train.iloc[indx_valid], target[indx_valid]
#     X_test_fold = df_test.copy()

#     # Concatenate original dataset to training data for this fold
#     X_train_combined = pd.concat([X_train_fold, df_orginal], axis=0)
#     y_train_combined = np.concatenate([y_train_fold, target_org], axis=0)

#     dtrain = xgb.DMatrix(X_train_combined, label=y_train_combined, enable_categorical=True)
#     dval = xgb.DMatrix(X_valid_fold, label=y_valid_fold, enable_categorical=True)
#     dtest = xgb.DMatrix(X_test_fold, enable_categorical=True)

#     model = xgb.train(
#         params_xgb, 
#         dtrain, 
#         num_boost_round=100_000, 
#         evals=[(dtrain, 'train'), (dval, 'validation')], 
#         early_stopping_rounds=50, 
#         verbose_eval=5000
#     )

#     oof_xgb[indx_valid] = model.predict(dval)
#     pred_test_xgb += model.predict(dtest)

#     top_preds = np.argsort(oof_xgb[indx_valid], axis=1)[:, -3:][:, ::-1]  
#     score = fast_map_k([[label] for label in y_valid_fold], top_preds)
#     final_score_xgb += score
#     print(f"XGBoost Fold {i+1} Score: {score:.5f}")

# pred_test_xgb /= FOLDS
# final_score_xgb /= FOLDS
# print(f"XGBoost Overall Score: {final_score_xgb:.5f}")
# all_oof_predictions.append(oof_xgb)
# all_test_predictions.append(pred_test_xgb)
# model_names.append('xgb')

# # Model 2: Random Forest
# print("=" * 50)
# print("Training Random Forest Model")
# print("=" * 50)
# oof_rf = np.zeros((len(df_train), np.unique(target).shape[0]))
# pred_test_rf = np.zeros((len(df_test), np.unique(target).shape[0]))
# final_score_rf = 0

# rf_params = {
#     'n_estimators': 200,
#     'max_depth': 15,
#     'min_samples_split': 5,
#     'min_samples_leaf': 2,
#     'random_state': SEED,
#     'n_jobs': -1
# }
# # Potential Improvement: Hyperparameter tuning for RandomForest:
# # from sklearn.model_selection import GridSearchCV
# # param_grid_rf = {
# #     'n_estimators': [100, 200, 300],
# #     'max_depth': [10, 15, 20],
# #     'min_samples_split': [2, 5, 10]
# # }
# # grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=SEED),
# #                               param_grid=param_grid_rf, cv=5, n_jobs=-1, scoring='accuracy')

# for i, (indx_train, indx_valid) in enumerate(skf.split(df_train, target)):
#     print(f"Random Forest Fold {i+1}")

#     X_train_fold, y_train_fold = df_train.iloc[indx_train], target[indx_train]
#     X_valid_fold, y_valid_fold = df_train.iloc[indx_valid], target[indx_valid]
#     X_test_fold = df_test.copy()

#     # Concatenate original dataset to training data for this fold
#     X_train_combined = pd.concat([X_train_fold, df_orginal], axis=0)
#     y_train_combined = np.concatenate([y_train_fold, target_org], axis=0)
    
#     # Create a pipeline with preprocessing and RandomForestClassifier
#     rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                                   ('classifier', RandomForestClassifier(**rf_params))])

#     rf_pipeline.fit(X_train_combined, y_train_combined)

#     oof_rf[indx_valid] = rf_pipeline.predict_proba(X_valid_fold)
#     pred_test_rf += rf_pipeline.predict_proba(X_test_fold)

#     top_preds = np.argsort(oof_rf[indx_valid], axis=1)[:, -3:][:, ::-1]  
#     score = fast_map_k([[label] for label in y_valid_fold], top_preds)
#     final_score_rf += score
#     print(f"Random Forest Fold {i+1} Score: {score:.5f}")

# pred_test_rf /= FOLDS
# final_score_rf /= FOLDS
# print(f"Random Forest Overall Score: {final_score_rf:.5f}")
# all_oof_predictions.append(oof_rf)
# all_test_predictions.append(pred_test_rf)
# model_names.append('rf')

# # Model 3: Logistic Regression
# print("=" * 50)
# print("Training Logistic Regression Model")
# print("=" * 50)
# oof_lr = np.zeros((len(df_train), np.unique(target).shape[0]))
# pred_test_lr = np.zeros((len(df_test), np.unique(target).shape[0]))
# final_score_lr = 0

# lr_params = {
#     'C': 0.1, # Adjusted C for potential better performance with scaling
#     'max_iter': 1000,
#     'random_state': SEED,
#     'n_jobs': -1,
#     'solver': 'saga' # 'saga' is good for large datasets and handles L1/L2 penalties
# }
# # Potential Improvement: Hyperparameter tuning for LogisticRegression:
# # from sklearn.model_selection import RandomizedSearchCV
# # param_dist_lr = {
# #     'classifier__C': np.logspace(-3, 2, 6),
# #     'classifier__solver': ['lbfgs', 'liblinear', 'saga']
# # }
# # rand_search_lr = RandomizedSearchCV(estimator=Pipeline(steps=[('preprocessor', preprocessor),
# #                                                              ('classifier', LogisticRegression(random_state=SEED))]),
# #                                    param_distributions=param_dist_lr, n_iter=10, cv=5, n_jobs=-1, scoring='accuracy')

# for i, (indx_train, indx_valid) in enumerate(skf.split(df_train, target)):
#     print(f"Logistic Regression Fold {i+1}")

#     X_train_fold, y_train_fold = df_train.iloc[indx_train], target[indx_train]
#     X_valid_fold, y_valid_fold = df_train.iloc[indx_valid], target[indx_valid]
#     X_test_fold = df_test.copy()

#     # Concatenate original dataset to training data for this fold
#     X_train_combined = pd.concat([X_train_fold, df_orginal], axis=0)
#     y_train_combined = np.concatenate([y_train_fold, target_org], axis=0)
    
#     # Create a pipeline with preprocessing and LogisticRegression
#     lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                                   ('classifier', LogisticRegression(**lr_params))])

#     lr_pipeline.fit(X_train_combined, y_train_combined)

#     oof_lr[indx_valid] = lr_pipeline.predict_proba(X_valid_fold)
#     pred_test_lr += lr_pipeline.predict_proba(X_test_fold)

#     top_preds = np.argsort(oof_lr[indx_valid], axis=1)[:, -3:][:, ::-1]  
#     score = fast_map_k([[label] for label in y_valid_fold], top_preds)
#     final_score_lr += score
#     print(f"Logistic Regression Fold {i+1} Score: {score:.5f}")

# pred_test_lr /= FOLDS
# final_score_lr /= FOLDS
# print(f"Logistic Regression Overall Score: {final_score_lr:.5f}")
# all_oof_predictions.append(oof_lr)
# all_test_predictions.append(pred_test_lr)
# model_names.append('lr')

# # Ensemble: Stacking with Logistic Regression
# print("=" * 50)
# print("Training Stacking Ensemble")
# print("=" * 50)

# # Create meta-features for stacking
# meta_features_train = np.column_stack(all_oof_predictions)
# meta_features_test = np.column_stack(all_test_predictions)

# # Train meta-learner
# # Using a slightly regularized Logistic Regression for the meta-learner
# meta_learner = LogisticRegression(C=0.01, max_iter=1000, random_state=SEED, solver='lbfgs') 
# meta_learner.fit(meta_features_train, target)

# # Get final ensemble predictions
# ensemble_pred_test = meta_learner.predict_proba(meta_features_test)

# # Calculate ensemble score on validation data
# ensemble_oof = meta_learner.predict_proba(meta_features_train)
# ensemble_score = 0
# for i, (indx_train, indx_valid) in enumerate(skf.split(df_train, target)):
#     # Calculate score only on the current fold's validation set
#     top_preds = np.argsort(ensemble_oof[indx_valid], axis=1)[:, -3:][:, ::-1]  
#     score = fast_map_k([[label] for label in target[indx_valid]], top_preds)
#     ensemble_score += score

# ensemble_score /= FOLDS
# print(f"Ensemble Score: {ensemble_score:.5f}")

# # Print individual model scores
# print("\n" + "=" * 50)
# print("MODEL COMPARISON")
# print("=" * 50)
# for i, name in enumerate(model_names):
#     # This assumes the order in model_names corresponds to the order of scores
#     # A dictionary would be more robust for mapping names to scores
#     if name == 'xgb':
#         print(f"{name.upper()}: {final_score_xgb:.5f}")
#     elif name == 'rf':
#         print(f"{name.upper()}: {final_score_rf:.5f}")
#     elif name == 'lr':
#         print(f"{name.upper()}: {final_score_lr:.5f}")
# print(f"ENSEMBLE: {ensemble_score:.5f}")

# # Generate final predictions
# top_preds = np.argsort(ensemble_pred_test, axis=1)[:,-3:][:,::-1]
# top_labels = le.inverse_transform(top_preds.ravel()).reshape(top_preds.shape)

# df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
# df_sub['Fertilizer Name'] = [' '.join(row) for row in top_labels]
# df_sub.to_csv('submission.csv', index=False)

# print("\nSubmission file saved as 'submission.csv'")


print("fina run!!")


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
pd.options.mode.copy_on_write = True
from tqdm.auto import tqdm
tqdm.pandas()
import numpy as np
import random
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List
import warnings
import category_encoders as ce # For Target Encoding

warnings.simplefilter('ignore')

def fast_map_k(actual: List[List], predicted: List[List], k: int = 3) -> float:
    """
    Calculates the Mean Average Precision at K (MAP@K).

    Args:
        actual (List[List]): A list of lists, where each inner list contains the true labels.
        predicted (List[List]): A list of lists, where each inner list contains the predicted labels.
        k (int): The number of top predictions to consider.

    Returns:
        float: The MAP@K score.
    """
    total_score = 0.0
    
    for true_items, pred_items in zip(actual, predicted):
        if not true_items:
            continue
            
        pred_items = pred_items[:k]
        true_set = set(true_items)
        
        # Create boolean mask for hits
        hits = np.array([item in true_set for item in pred_items])
        
        if not hits.any():
            continue
        
        # Calculate cumulative hits and positions
        cumulative_hits = np.cumsum(hits)
        positions = np.arange(1, len(pred_items) + 1)
        
        # Calculate precision at each hit position
        precisions = cumulative_hits[hits] / positions[hits]
        score = np.sum(precisions) / min(len(true_items), k)
        total_score += score
    
    return total_score / len(actual)

# Custom Scorer for MAP@3
map3_scorer = make_scorer(fast_map_k, greater_is_better=True)

# Load data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_orginal = pd.read_csv('/kaggle/input/fertilizerprediction/fertilizer-prediction.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)

# --- Feature Engineering ---

# 1. Base categorical features from numericals
# These will be treated as categorical features in subsequent steps
numerical_cols_original = df_test.select_dtypes(include=['number']).columns.tolist()
for col in numerical_cols_original:
    df_train[f"cat_{col}"] = df_train[col].astype(str)
    df_orginal[f"cat_{col}"] = df_orginal[col].astype(str)
    df_test[f"cat_{col}"] = df_test[col].astype(str)

# 2. Add a constant feature (can sometimes help models)
df_train['const'] = 1
df_orginal['const'] = 1
df_test['const'] = 1

# 3. NEW: Interaction Features
# Combine 'Crop Type' and 'Soil Type' as a new categorical feature
df_train['Crop_Soil_Interaction'] = df_train['Crop Type'].astype(str) + '_' + df_train['Soil Type'].astype(str)
df_orginal['Crop_Soil_Interaction'] = df_orginal['Crop Type'].astype(str) + '_' + df_orginal['Soil Type'].astype(str)
df_test['Crop_Soil_Interaction'] = df_test['Crop Type'].astype(str) + '_' + df_test['Soil Type'].astype(str)

# Example numerical interaction/ratio features
df_train['N_P_Ratio'] = df_train['Nitrogen'] / (df_train['Phosphorus'] + 1e-6) # Add small epsilon to avoid division by zero
df_orginal['N_P_Ratio'] = df_orginal['Nitrogen'] / (df_orginal['Phosphorus'] + 1e-6)
df_test['N_P_Ratio'] = df_test['Nitrogen'] / (df_test['Phosphorus'] + 1e-6)

df_train['Temp_Moist_Diff'] = df_train['Temperature'] - df_train['Moisture']
df_orginal['Temp_Moist_Diff'] = df_orginal['Temperature'] - df_orginal['Moisture']
df_test['Temp_Moist_Diff'] = df_test['Temperature'] - df_test['Moisture']

df_train['Soil_Moist_Interaction'] = df_train['Soil Type'].astype(str) + '_' + df_train['Moisture'].astype(str)
df_orginal['Soil_Moist_Interaction'] = df_orginal['Soil Type'].astype(str) + '_' + df_orginal['Moisture'].astype(str)
df_test['Soil_Moist_Interaction'] = df_test['Soil Type'].astype(str) + '_' + df_test['Moisture'].astype(str)


# 4. Convert all object columns (including newly created interaction features) to 'category' dtype
# This is crucial for XGBoost's enable_categorical=True and for consistent preprocessing
for col in df_train.select_dtypes(include=['object']).columns: # Check df_train as it will contain all new cols
    df_train[col] = df_train[col].astype('category')
    df_orginal[col] = df_orginal[col].astype('category')
    df_test[col] = df_test[col].astype('category')


# Get Target column
target = df_train.pop('Fertilizer Name')
target_org = df_orginal.pop('Fertilizer Name')

# Encode target labels
le = LabelEncoder()
target = le.fit_transform(target)
target_org = le.transform(target_org) # Use the same encoder for original dataset target

# Ensemble configuration
FOLDS = 5
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Initialize storage for ensemble predictions
all_oof_predictions = []
all_test_predictions = []
model_names = []

# --- Define Feature Types for Preprocessing ---
# Numerical features (original and new numerical ones)
numerical_features = [col for col in df_train.select_dtypes(include=np.number).columns.tolist() if col != 'const']

# Categorical features for One-Hot Encoding (low cardinality, or if target encoding isn't applied)
# Example: If 'Crop Type' is high cardinality but also a direct predictor,
# you might include it in target_encode_features instead.
one_hot_features = [col for col in df_train.select_dtypes(include='category').columns.tolist() if col not in ['Crop Type', 'Soil Type', 'cat_Humidity', 'cat_Moisture', 'Crop_Soil_Interaction', 'Soil_Moist_Interaction']]

# Categorical features for Target Encoding (often high cardinality ones)
# This list needs careful selection based on your data and EDA
target_encode_features = ['Crop Type', 'Soil Type', 'cat_Humidity', 'cat_Moisture', 'Crop_Soil_Interaction', 'Soil_Moist_Interaction']

# Remaining features (like 'const') will be passed through
# df_train.columns will get updated as features are added, so recalculate.
all_features = numerical_features + one_hot_features + target_encode_features + ['const'] # Ensure 'const' is included

# --- Preprocessor for RF and LR (includes Target Encoding and Scaling/OHE) ---
# Target Encoding must be done within the CV loop to prevent leakage
# For the ColumnTransformer outside the loop, we define the structure
# and fit/transform it within each fold.
# We'll create a new preprocessor for each fold for target encoding.

# Model 1: XGBoost
print("=" * 50)
print("Training XGBoost Model")
print("=" * 50)
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
oof_xgb = np.zeros((len(df_train), np.unique(target).shape[0]))
pred_test_xgb = np.zeros((len(df_test), np.unique(target).shape[0]))
final_score_xgb = 0

params_xgb = {
    'objective': 'multi:softprob',
    'num_class': 7,
    'max_depth': 7,
    'learning_rate': 0.01,
    'n_estimators': 100_000,
    'reg_alpha': 7,
    'reg_lambda': 5.3,
    'gamma': 0.3,
    'max_delta_step': 4,
    'subsample': 0.86,
    'colsample_bytree': 0.4,
    'min_child_weight': 5,
    'random_state': SEED,
    'eval_metric': 'mlogloss',
    'enable_categorical': True, # XGBoost can handle 'category' dtype directly
    'device': "cuda",
    'tree_method': 'hist', 
    'early_stopping_rounds': 50 
}

for i, (indx_train, indx_valid) in enumerate(skf.split(df_train, target)):
    print(f"XGBoost Fold {i+1}")

    X_train_fold, y_train_fold = df_train.iloc[indx_train], target[indx_train]
    X_valid_fold, y_valid_fold = df_train.iloc[indx_valid], target[indx_valid]
    X_test_fold = df_test.copy()

    # Concatenate original dataset to training data for this fold
    X_train_combined = pd.concat([X_train_fold, df_orginal], axis=0)
    y_train_combined = np.concatenate([y_train_fold, target_org], axis=0)

    # Convert all object columns to 'category' in combined set if they were not before
    for col in X_train_combined.select_dtypes(include=['object']).columns:
        X_train_combined[col] = X_train_combined[col].astype('category')
    
    for col in X_valid_fold.select_dtypes(include=['object']).columns:
        X_valid_fold[col] = X_valid_fold[col].astype('category')
        
    for col in X_test_fold.select_dtypes(include=['object']).columns:
        X_test_fold[col] = X_test_fold[col].astype('category')

    dtrain = xgb.DMatrix(X_train_combined, label=y_train_combined, enable_categorical=True)
    dval = xgb.DMatrix(X_valid_fold, label=y_valid_fold, enable_categorical=True)
    dtest = xgb.DMatrix(X_test_fold, enable_categorical=True)

    evals_result = {}
    bst = xgb.train(
        params_xgb, 
        dtrain, 
        num_boost_round=params_xgb['n_estimators'], 
        evals=[(dtrain, 'train'), (dval, 'validation')], 
        early_stopping_rounds=params_xgb['early_stopping_rounds'], 
        verbose_eval=5000,
        callbacks=[xgb.callback.record_evaluation(evals_result)]
    )
    
    best_iteration = bst.best_iteration
    
    oof_xgb[indx_valid] = bst.predict(dval, iteration_range=(0, best_iteration + 1))
    pred_test_xgb += bst.predict(dtest, iteration_range=(0, best_iteration + 1))

    top_preds = np.argsort(oof_xgb[indx_valid], axis=1)[:, -3:][:, ::-1]  
    score = fast_map_k([[label] for label in y_valid_fold], top_preds)
    final_score_xgb += score
    print(f"XGBoost Fold {i+1} Score: {score:.5f}")

pred_test_xgb /= FOLDS
final_score_xgb /= FOLDS
print(f"XGBoost Overall Score: {final_score_xgb:.5f}")
all_oof_predictions.append(oof_xgb)
all_test_predictions.append(pred_test_xgb)
model_names.append('xgb')


# --- Model 2: Random Forest ---
print("=" * 50)
print("Training Random Forest Model")
print("=" * 50)
oof_rf = np.zeros((len(df_train), np.unique(target).shape[0]))
pred_test_rf = np.zeros((len(df_test), np.unique(target).shape[0]))
final_score_rf = 0

rf_params = {
    'n_estimators': 300,
    'max_depth': 12,
    'min_samples_split': 5,
    'min_samples_leaf': 3,
    'random_state': SEED,
    'n_jobs': -1,
    'class_weight': 'balanced'
}

for i, (indx_train, indx_valid) in enumerate(skf.split(df_train, target)):
    print(f"Random Forest Fold {i+1}")

    X_train_fold, y_train_fold = df_train.iloc[indx_train], target[indx_train]
    X_valid_fold, y_valid_fold = df_train.iloc[indx_valid], target[indx_valid]
    X_test_fold = df_test.copy()

    X_train_combined = pd.concat([X_train_fold, df_orginal], axis=0)
    y_train_combined = np.concatenate([y_train_fold, target_org], axis=0)

    # --- IMPORTANT: Create and fit preprocessor for EACH FOLD to prevent leakage ---
    fold_preprocessor = ColumnTransformer(
        transformers=[
            ('target_encode', ce.TargetEncoder(cols=target_encode_features, smoothing=0.2, min_samples_leaf=1), target_encode_features), # Tune smoothing
            ('num', StandardScaler(), numerical_features),
            ('cat_onehot', OneHotEncoder(handle_unknown='ignore'), one_hot_features)
        ],
        remainder='passthrough' # 'const' feature
    )
    
    # Fit the preprocessor only on the training data of the current fold
    # and transform all splits.
    X_train_processed = fold_preprocessor.fit_transform(X_train_combined, y_train_combined)
    X_valid_processed = fold_preprocessor.transform(X_valid_fold)
    X_test_processed = fold_preprocessor.transform(X_test_fold)

    model = RandomForestClassifier(**rf_params)
    model.fit(X_train_processed, y_train_combined)

    oof_rf[indx_valid] = model.predict_proba(X_valid_processed)
    pred_test_rf += model.predict_proba(X_test_processed)

    top_preds = np.argsort(oof_rf[indx_valid], axis=1)[:, -3:][:, ::-1]  
    score = fast_map_k([[label] for label in y_valid_fold], top_preds)
    final_score_rf += score
    print(f"Random Forest Fold {i+1} Score: {score:.5f}")

pred_test_rf /= FOLDS
final_score_rf /= FOLDS
print(f"Random Forest Overall Score: {final_score_rf:.5f}")
all_oof_predictions.append(oof_rf)
all_test_predictions.append(pred_test_rf)
model_names.append('rf')


# --- Model 3: Logistic Regression ---
print("=" * 50)
print("Training Logistic Regression Model")
print("=" * 50)
oof_lr = np.zeros((len(df_train), np.unique(target).shape[0]))
pred_test_lr = np.zeros((len(df_test), np.unique(target).shape[0]))
final_score_lr = 0

lr_params = {
    'C': 0.05,
    'max_iter': 2000,
    'random_state': SEED,
    'n_jobs': -1,
    'solver': 'saga',
    'penalty': 'l2',
    'class_weight': 'balanced'
}

for i, (indx_train, indx_valid) in enumerate(skf.split(df_train, target)):
    print(f"Logistic Regression Fold {i+1}")

    X_train_fold, y_train_fold = df_train.iloc[indx_train], target[indx_train]
    X_valid_fold, y_valid_fold = df_train.iloc[indx_valid], target[indx_valid]
    X_test_fold = df_test.copy()

    X_train_combined = pd.concat([X_train_fold, df_orginal], axis=0)
    y_train_combined = np.concatenate([y_train_fold, target_org], axis=0)

    # --- IMPORTANT: Create and fit preprocessor for EACH FOLD to prevent leakage ---
    fold_preprocessor = ColumnTransformer(
        transformers=[
            ('target_encode', ce.TargetEncoder(cols=target_encode_features, smoothing=0.2, min_samples_leaf=1), target_encode_features),
            ('num', StandardScaler(), numerical_features),
            ('cat_onehot', OneHotEncoder(handle_unknown='ignore'), one_hot_features)
        ],
        remainder='passthrough'
    )

    X_train_processed = fold_preprocessor.fit_transform(X_train_combined, y_train_combined)
    X_valid_processed = fold_preprocessor.transform(X_valid_fold)
    X_test_processed = fold_preprocessor.transform(X_test_fold)
    
    model = LogisticRegression(**lr_params)
    model.fit(X_train_processed, y_train_combined)

    oof_lr[indx_valid] = model.predict_proba(X_valid_processed)
    pred_test_lr += model.predict_proba(X_test_processed)

    top_preds = np.argsort(oof_lr[indx_valid], axis=1)[:, -3:][:, ::-1]  
    score = fast_map_k([[label] for label in y_valid_fold], top_preds)
    final_score_lr += score
    print(f"Logistic Regression Fold {i+1} Score: {score:.5f}")

pred_test_lr /= FOLDS
final_score_lr /= FOLDS
print(f"Logistic Regression Overall Score: {final_score_lr:.5f}")
all_oof_predictions.append(oof_lr)
all_test_predictions.append(pred_test_lr)
model_names.append('lr')


# --- Ensemble: Stacking with Logistic Regression ---
print("=" * 50)
print("Training Stacking Ensemble")
print("=" * 50)

meta_features_train = np.column_stack(all_oof_predictions)
meta_features_test = np.column_stack(all_test_predictions)

meta_learner_params = {
    'C': 0.005,
    'max_iter': 2000,
    'random_state': SEED,
    'solver': 'lbfgs'
}
meta_learner = LogisticRegression(**meta_learner_params)
meta_learner.fit(meta_features_train, target)

ensemble_pred_test = meta_learner.predict_proba(meta_features_test)

ensemble_oof = meta_learner.predict_proba(meta_features_train)
ensemble_score = 0
for i, (indx_train, indx_valid) in enumerate(skf.split(df_train, target)):
    top_preds = np.argsort(ensemble_oof[indx_valid], axis=1)[:, -3:][:, ::-1]  
    score = fast_map_k([[label] for label in target[indx_valid]], top_preds)
    ensemble_score += score

ensemble_score /= FOLDS
print(f"Ensemble Score: {ensemble_score:.5f}")

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
for i, name in enumerate(model_names):
    if name == 'xgb':
        print(f"{name.upper()}: {final_score_xgb:.5f}")
    elif name == 'rf':
        print(f"{name.upper()}: {final_score_rf:.5f}")
    elif name == 'lr':
        print(f"{name.upper()}: {final_score_lr:.5f}")
print(f"ENSEMBLE: {ensemble_score:.5f}")

# Generate final predictions
top_preds = np.argsort(ensemble_pred_test, axis=1)[:,-3:][:,::-1]
top_labels = le.inverse_transform(top_preds.ravel()).reshape(top_preds.shape)

df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
df_sub['Fertilizer Name'] = [' '.join(row) for row in top_labels]
df_sub.to_csv('submission.csv', index=False)

print("\nSubmission file saved as 'submission.csv'")

