import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')



def create_submission_stacking(path_to_ds, file_short_names, target_col='y', 
                              validation_file=None, n_folds=5, random_state=42):
    
    
    print(f"Loading {len(file_short_names)} submission files...")
    
    # Load all submission files
    submissions = []
    for i, file_name in enumerate(file_short_names):
        file_path = f"{path_to_ds}{file_name}.csv"
        df = pd.read_csv(file_path)
        print(f"Loaded {file_name}: {df.shape}")
        submissions.append(df)
    
    # Create base dataframe with IDs
    base_df = submissions[0][['id']].copy()
    
    # Add predictions from each submission as features
    feature_names = []
    for i, (df, file_name) in enumerate(zip(submissions, file_short_names)):
        feature_name = f"model_{i+1}_{file_name[:7]}"  # Use first 7 chars of filename
        base_df[feature_name] = df[target_col]
        feature_names.append(feature_name)
    
    print(f"Created feature matrix: {base_df.shape}")
    print("Features:", feature_names)
    
    # Prepare feature matrix
    X = base_df[feature_names].values
    
    if validation_file is not None:
        # If we have validation data, do proper stacking
        print(f"Loading validation file: {validation_file}")
        val_df = pd.read_csv(validation_file)
        
        # Merge validation labels with our predictions
        merged_df = base_df.merge(val_df[['id', target_col]], on='id', how='inner')
        y_true = merged_df[target_col].values
        X_matched = merged_df[feature_names].values
        
        print(f"Matched {len(y_true)} samples with true labels")
        
        # Perform cross-validation to create meta-features
        meta_features = np.zeros((X_matched.shape[0], len(feature_names)))
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_matched)):
            print(f"Processing fold {fold + 1}/{n_folds}")
            
            X_train_fold, X_val_fold = X_matched[train_idx], X_matched[val_idx]
            y_train_fold = y_true[train_idx]
            
            # Train Random Forest on base predictions
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state + fold,
                n_jobs=-1
            )
            
            rf.fit(X_train_fold, y_train_fold)
            
            # Predict on validation fold
            val_pred = rf.predict(X_val_fold)
            meta_features[val_idx, :] = X_val_fold  # Keep original features for now
        
        # Train final meta-model on all data
        final_rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        final_rf.fit(X_matched, y_true)
        
        # Evaluate on validation set
        val_predictions = final_rf.predict(X_matched)
        val_mse = mean_squared_error(y_true, val_predictions)
        val_rmse = np.sqrt(val_mse)
        
        print(f"\nValidation RMSE: {val_rmse:.6f}")
        
        # Make final predictions on all data
        final_predictions = final_rf.predict(X)
        
    else:
        # No validation data - use ensemble of the predictions
        print("No validation file provided. Using weighted ensemble approach...")
        
        # Create Random Forest to learn relationships between predictions
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Use weighted average as pseudo-target (similar to your original approach)
        # Weights based on assumed model quality (higher score = higher weight)
        scores = [float(name.replace('0.', '0.')) for name in file_short_names if name.replace('0.', '0.').replace('.', '').isdigit()]
        
        if len(scores) == len(file_short_names):
            # Normalize scores to weights
            weights = np.array(scores) / sum(scores)
            print("Using score-based weights:", weights)
        else:
            # Use your original weights
            original_weights = [0.19, 0.17, 0.15, 0.13, 0.10, 0.09, 0.08, 0.05, 0.04]
            weights = np.array(original_weights[:len(file_short_names)])
            weights = weights / weights.sum()  # Normalize
            print("Using original weights:", weights)
        
        # Create pseudo-target
        y_pseudo = np.average(X, axis=1, weights=weights)
        
        # Add some noise to create variation for the RF to learn from
        np.random.seed(random_state)
        noise_factor = 0.001  # Small noise to help RF learn
        y_pseudo += np.random.normal(0, noise_factor * np.std(y_pseudo), len(y_pseudo))
        
        # Train Random Forest
        rf.fit(X, y_pseudo)
        
        # Generate final predictions
        final_predictions = rf.predict(X)
        
        final_rf = rf
    
    # Print feature importances
    print("\nRandom Forest Feature Importances:")
    importances = final_rf.feature_importances_
    for feature, importance in zip(feature_names, importances):
        print(f"  {feature}: {importance:.4f}")
    
    # Create final submission
    final_submission = base_df[['id']].copy()
    final_submission[target_col] = final_predictions
    
    print(f"\nFinal predictions - Min: {final_predictions.min():.6f}, "
          f"Max: {final_predictions.max():.6f}, Mean: {final_predictions.mean():.6f}")
    
    return final_submission, final_rf, base_df



# Usage with your data
if __name__ == "__main__":
    # Your original parameters
    path_to_ds = '/kaggle/input/2-august-2025-ps-s5e8/submission '
    file_short_names = [
        '0.97526', '0.97450', '0.97441', '0.97353', '0.97302', 
        '0.97298', '0.97245', '0.97118', '0.97109'
    ]
    
    # Create stacking ensemble
    print("=" * 60)
    print("BASIC STACKING ENSEMBLE")
    print("=" * 60)
    
    stacked_submission, rf_model, feature_df = create_submission_stacking(
        path_to_ds=path_to_ds,
        file_short_names=file_short_names,
        target_col='y',
        random_state=42
    )
    
    # Save basic stacking result
    stacked_submission.to_csv('stacking_ensemble_rf.csv', index=False)
    print("\nSaved basic stacking result to 'stacking_ensemble_rf.csv'")

     # Display first few rows
    print("\nFirst 10 rows of basic stacking:")
    print(stacked_submission.head(10))


# def advanced_stacking_ensemble(path_to_ds, file_short_names, target_col='y'):
#     """
#     Advanced stacking with multiple Random Forest configurations
#     """
#     print("Creating Advanced Stacking Ensemble...")
    
#     # Load submissions
#     submissions = []
#     for file_name in file_short_names:
#         df = pd.read_csv(f"{path_to_ds}{file_name}.csv")
#         submissions.append(df)
    
#     # Create feature matrix
#     base_df = submissions[0][['id']].copy()
#     feature_names = []
    
#     for i, (df, file_name) in enumerate(zip(submissions, file_short_names)):
#         feature_name = f"model_{i+1}"
#         base_df[feature_name] = df[target_col]
#         feature_names.append(feature_name)
    
#     X = base_df[feature_names].values
    
#     # Create multiple Random Forest models with different configurations
#     rf_configs = [
#         {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 5},
#         {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 3},
#         {'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 7},
#         {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 4},
#     ]
    
#     # Use weighted average as target
#     scores = [0.97526, 0.97450, 0.97441, 0.97353, 0.97302, 0.97298, 0.97245, 0.97118, 0.97109]
#     weights = np.array(scores[:len(file_short_names)])
#     weights = weights / weights.sum()
    
#     y_target = np.average(X, axis=1, weights=weights)
    
#     # Train multiple Random Forests and average their predictions
#     predictions_list = []
    
#     for i, config in enumerate(rf_configs):
#         print(f"Training RF configuration {i+1}/{len(rf_configs)}")
        
#         rf = RandomForestRegressor(
#             random_state=42 + i,
#             n_jobs=-1,
#             **config
#         )
        
#         rf.fit(X, y_target)
#         pred = rf.predict(X)
#         predictions_list.append(pred)
        
#         print(f"  Config {i+1} - Min: {pred.min():.6f}, Max: {pred.max():.6f}")
    
#     # Average predictions from all RF models
#     final_predictions = np.mean(predictions_list, axis=0)
    
#     # Create submission
#     final_submission = base_df[['id']].copy()
#     final_submission[target_col] = final_predictions
    
#     print(f"\nFinal ensemble predictions - Min: {final_predictions.min():.6f}, "
#           f"Max: {final_predictions.max():.6f}, Mean: {final_predictions.mean():.6f}")
    
#     return final_submission



# # Usage with your data
# if __name__ == "__main__":
#     # Your original parameters
#     path_to_ds = '/kaggle/input/2-august-2025-ps-s5e8/submission '
#     file_short_names = [
#         '0.97526', '0.97450', '0.97441', '0.97353', '0.97302', 
#         '0.97298', '0.97245', '0.97118', '0.97109'
#     ]

    
#     print("\n" + "=" * 60)
#     print("ADVANCED STACKING ENSEMBLE")
#     print("=" * 60)
    
#     # Create advanced ensemble
#     advanced_submission = advanced_stacking_ensemble(
#         path_to_ds=path_to_ds,
#         file_short_names=file_short_names,
#         target_col='y'
#     )
    
#     # Save advanced result
#     advanced_submission.to_csv('advanced_stacking_rf.csv', index=False)
#     print("\nSaved advanced stacking result to 'advanced_stacking_rf.csv'")
    
#     print("\nFirst 10 rows of advanced stacking:")
#     print(advanced_submission.head(10))

