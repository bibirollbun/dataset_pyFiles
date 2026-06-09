import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import os
import pickle
pd.set_option('display.max_columns', 500)

train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv',index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv',index_col='id')


import joblib 

label_encoder = joblib.load('/kaggle/input/p05e06-xgb2-for-fertilizers/le.joblib')


train_df['Fertilizer Name'] = label_encoder.transform(train_df['Fertilizer Name'])


with open('/kaggle/input/weighted-ensemble-through-hc-fertlilizer/oof_predictions.pkl','rb') as file:
    oof_predictions = pickle.load(file)


with open('/kaggle/input/weighted-ensemble-through-hc-fertlilizer/test_predictions.pkl','rb') as file:
    test_predictions = pickle.load(file)


with open('/kaggle/input/xgb7-for-fertilizer/model_predictions.pkl','rb') as file:
    predictions = pickle.load(file)

oof_predictions['xgb'] = predictions['oof_avg']
test_predictions['xgb'] = predictions['pred_test']


with open('/kaggle/input/lgb5-for-fertilizer/oof_predictions.pkl','rb') as file:
    oof_predictions_goss = pickle.load(file)


with open('/kaggle/input/lgb5-for-fertilizer/test_predictions.pkl','rb') as file:
    test_predictions_goss = pickle.load(file)


with open('/kaggle/input/autogluon3-for-fertilizer/WeightedEnsemble_L2_PSEUDO_oof_pred_probs_0.359742.pkl','rb') as file:
    oof_predictions_ag = pickle.load(file)


with open('/kaggle/input/autogluon3-for-fertilizer/WeightedEnsemble_L2_PSEUDO_test_pred_probs_0.359742.pkl','rb') as file:
    test_predictions_ag = pickle.load(file)

oof_predictions['ag'] = oof_predictions_ag
test_predictions['ag'] = test_predictions_ag


oof_predictions['lgb-gosss'] = oof_predictions_goss['lgb_goss']
test_predictions['lgb-gosss'] = test_predictions_goss['lgb_goss']


import numpy as np
# from sklearn.linear_model import Ridge
from cuml import Ridge
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def map_at_k(y_true, y_pred_proba, k=3):
    """
    Calculate Mean Average Precision at K (MAP@K)
    
    Args:
        y_true: True labels (1D array/series of class indices)
        y_pred_proba: Predicted probabilities (2D array: samples x classes)
        k: Number of top predictions to consider
    
    Returns:
        MAP@K score
    """
    # Convert to numpy array and reset index if pandas Series
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_true, 'reset_index'):
        y_true = y_true.reset_index(drop=True)
    
    y_true = np.array(y_true)
    n_samples = len(y_true)
    map_sum = 0.0
    
    for i in range(n_samples):
        # Get top k predictions (indices sorted by probability)
        top_k_indices = np.argsort(y_pred_proba[i])[::-1][:k]
        
        # Calculate average precision for this sample
        true_label = y_true[i]
        ap = 0.0
        relevant_count = 0
        
        for j in range(k):
            if top_k_indices[j] == true_label:
                relevant_count += 1
                ap += relevant_count / (j + 1)
        
        map_sum += ap
    
    return map_sum / n_samples

class RidgeEnsemble:
    def __init__(self, alpha=1.0, normalize_features=True):
        self.alpha = alpha
        self.normalize_features = normalize_features
        self.ridge_models = {}
        self.scalers = {}
        self.n_classes = None
        
    def fit(self, predictions_dict, y_true):
        """
        Fit Ridge regression for each class
        
        Args:
            predictions_dict: Dict of model predictions {'model_name': predictions_array}
            y_true: True labels (1D array of class indices)
        """
        # Stack predictions from all models
        model_names = list(predictions_dict.keys())
        stacked_preds = np.hstack([predictions_dict[name] for name in model_names])
        
        self.n_classes = stacked_preds.shape[1]
        self.model_names = model_names
        
        # Train a Ridge model for each class
        for class_idx in range(self.n_classes):
            # Create binary target for current class
            y_binary = (y_true == class_idx).astype(int)
            
            # Normalize features if requested
            if self.normalize_features:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(stacked_preds)
                self.scalers[class_idx] = scaler
            else:
                X_scaled = stacked_preds
            
            # Fit Ridge regression
            ridge = Ridge(alpha=self.alpha, fit_intercept=True, normalize=True)
            ridge.fit(X_scaled, y_binary)
            self.ridge_models[class_idx] = ridge
    
    def predict_proba(self, predictions_dict):
        """
        Generate ensemble predictions
        
        Args:
            predictions_dict: Dict of model predictions for test data
            
        Returns:
            Ensemble probabilities (samples x classes)
        """
        # Stack predictions
        stacked_preds = np.hstack([predictions_dict[name] for name in self.model_names])
        n_samples = stacked_preds.shape[0]
        ensemble_probs = np.zeros((n_samples, self.n_classes))
        
        # Get predictions from each Ridge model
        for class_idx in range(self.n_classes):
            if self.normalize_features:
                X_scaled = self.scalers[class_idx].transform(stacked_preds)
            else:
                X_scaled = stacked_preds
            
            # Predict probabilities for this class
            class_probs = self.ridge_models[class_idx].predict(X_scaled)
            # Clip to [0, 1] range
            ensemble_probs[:, class_idx] = np.clip(class_probs, 0, 1)
        
        # Normalize probabilities to sum to 1
        row_sums = ensemble_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        ensemble_probs = ensemble_probs / row_sums
        
        return ensemble_probs

def evaluate_ridge_ensemble_kfold(oof_predictions, test_predictions, y_train, 
                                  alpha_values=[0.1, 1.0, 10.0], n_splits=5, random_state=42):
    """
    Evaluate Ridge ensemble using K-Fold cross validation
    
    Args:
        oof_predictions: Dict of out-of-fold predictions {'model_name': predictions_array}
        test_predictions: Dict of test predictions {'model_name': predictions_array}
        y_train: Training labels
        alpha_values: List of alpha values to test
        n_splits: Number of K-fold splits
        random_state: Random state for reproducibility
    
    Returns:
        best_ensemble: Best trained ensemble model
        test_probs: Test predictions from best ensemble
        best_map3: Best MAP@3 score
        le: Label encoder used
        cv_results: Cross-validation results for each alpha
    """
    # Convert pandas Series to numpy array and encode labels
    if hasattr(y_train, 'values'):
        y_train_array = y_train.values
    else:
        y_train_array = np.array(y_train)
    
    # Create label encoder for string labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_array)
    
    # Initialize K-Fold cross validator
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    best_map3 = 0
    best_alpha = None
    best_ensemble = None
    cv_results = {}
    
    print(f"Performing {n_splits}-Fold Cross Validation for Ridge ensemble...")
    print(f"Total samples: {len(y_train_encoded)}")
    print(f"Number of classes: {len(np.unique(y_train_encoded))}")
    
    for alpha in alpha_values:
        print(f"\nTesting alpha = {alpha}")
        fold_scores = []
        
        # Perform K-fold cross validation
        for fold, (train_idx, val_idx) in enumerate(kfold.split(y_train_encoded, y_train_encoded)):
            print(f"  Fold {fold + 1}/{n_splits}...", end=" ")
            
            # Split data for current fold
            fold_train_preds = {name: preds[train_idx] for name, preds in oof_predictions.items()}
            fold_val_preds = {name: preds[val_idx] for name, preds in oof_predictions.items()}
            fold_y_train = y_train_encoded[train_idx]
            fold_y_val = y_train_encoded[val_idx]
            
            # Train ensemble on fold training data
            fold_ensemble = RidgeEnsemble(alpha=alpha)
            fold_ensemble.fit(fold_train_preds, fold_y_train)
            
            # Predict on fold validation data
            fold_val_probs = fold_ensemble.predict_proba(fold_val_preds)
            
            # Calculate MAP@3 for this fold
            fold_map3 = map_at_k(fold_y_val, fold_val_probs, k=3)
            fold_scores.append(fold_map3)
            print(f"MAP@3: {fold_map3:.4f}")
        
        # Calculate mean and std of MAP@3 across folds
        mean_map3 = np.mean(fold_scores)
        std_map3 = np.std(fold_scores)
        cv_results[alpha] = {
            'mean_map3': mean_map3,
            'std_map3': std_map3,
            'fold_scores': fold_scores
        }
        
        print(f"  Mean MAP@3: {mean_map3:.4f} (+/- {std_map3:.4f})")
        
        # Update best parameters
        if mean_map3 > best_map3:
            best_map3 = mean_map3
            best_alpha = alpha
    
    print(f"\n" + "="*50)
    print(f"CROSS VALIDATION RESULTS:")
    print(f"="*50)
    for alpha, results in cv_results.items():
        print(f"Alpha {alpha:>6}: {results['mean_map3']:.4f} (+/- {results['std_map3']:.4f})")
    
    print(f"\nBest alpha: {best_alpha}")
    print(f"Best CV MAP@3: {best_map3:.4f} (+/- {cv_results[best_alpha]['std_map3']:.4f})")
    
    # Train final ensemble on all training data using best alpha
    print(f"\nTraining final ensemble with alpha={best_alpha} on full training data...")
    best_ensemble = RidgeEnsemble(alpha=best_alpha)
    best_ensemble.fit(oof_predictions, y_train_encoded)
    
    # Generate final test predictions
    test_ensemble_probs = best_ensemble.predict_proba(test_predictions)
    
    print(f"Final ensemble trained successfully!")
    print(f"Test predictions shape: {test_ensemble_probs.shape}")
    
    return best_ensemble, test_ensemble_probs, best_map3, le, cv_results

def print_detailed_cv_results(cv_results):
    """Print detailed cross-validation results"""
    print(f"\n" + "="*60)
    print(f"DETAILED CROSS VALIDATION RESULTS")
    print(f"="*60)
    
    for alpha, results in cv_results.items():
        print(f"\nAlpha = {alpha}")
        print(f"Fold scores: {[f'{score:.4f}' for score in results['fold_scores']]}")
        print(f"Mean: {results['mean_map3']:.4f}")
        print(f"Std:  {results['std_map3']:.4f}")
        print(f"Min:  {min(results['fold_scores']):.4f}")
        print(f"Max:  {max(results['fold_scores']):.4f}")

# Example usage
if __name__ == "__main__":
    # Assuming you have your data loaded
    # oof_predictions, test_predictions, train_df should be defined
    
    # Run the K-fold ensemble evaluation
    ensemble, test_probs, best_map3, le, cv_results = evaluate_ridge_ensemble_kfold(
        oof_predictions=oof_predictions,
        test_predictions=test_predictions, 
        y_train=train_df['Fertilizer Name'],  
        alpha_values=[0.01, 0.1, 1.0, 10.0, 100.0],
        n_splits=5,
        random_state=42
    )
    
    # Print detailed results
    print_detailed_cv_results(cv_results)
    
    print(f"\n" + "="*50)
    print(f"FINAL RESULTS")
    print(f"="*50)
    print(f"Best CV MAP@3 score: {best_map3:.4f}")
    print(f"Test predictions shape: {test_probs.shape}")
    print(f"Number of classes: {len(le.classes_)}")
    print(f"Classes: {le.classes_}")


# import numpy as np
# from cuml import LogisticRegression
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import warnings
# warnings.filterwarnings('ignore')

# def map_at_k(y_true, y_pred_proba, k=3):
#     """
#     Calculate Mean Average Precision at K (MAP@K)
    
#     Args:
#         y_true: True labels (1D array/series of class indices)
#         y_pred_proba: Predicted probabilities (2D array: samples x classes)
#         k: Number of top predictions to consider
    
#     Returns:
#         MAP@K score
#     """
#     # Convert to numpy array and reset index if pandas Series
#     if hasattr(y_true, 'values'):
#         y_true = y_true.values
#     if hasattr(y_true, 'reset_index'):
#         y_true = y_true.reset_index(drop=True)
    
#     y_true = np.array(y_true)
#     n_samples = len(y_true)
#     map_sum = 0.0
    
#     for i in range(n_samples):
#         # Get top k predictions (indices sorted by probability)
#         top_k_indices = np.argsort(y_pred_proba[i])[::-1][:k]
        
#         # Calculate average precision for this sample
#         true_label = y_true[i]
#         ap = 0.0
#         relevant_count = 0
        
#         for j in range(k):
#             if top_k_indices[j] == true_label:
#                 relevant_count += 1
#                 ap += relevant_count / (j + 1)
        
#         map_sum += ap
    
#     return map_sum / n_samples

# class LogisticEnsemble:
#     def __init__(self, C=1.0, normalize_features=True, solver='qn', max_iter=1000):
#         self.C = C  # Inverse of regularization (higher C = less regularization)
#         self.normalize_features = normalize_features
#         self.solver = solver
#         self.max_iter = max_iter
#         self.logistic_model = None
#         self.scaler = None
#         self.n_classes = None
        
#     def fit(self, predictions_dict, y_true):
#         """
#         Fit Logistic Regression for multiclass classification
        
#         Args:
#             predictions_dict: Dict of model predictions {'model_name': predictions_array}
#             y_true: True labels (1D array of class indices)
#         """
#         # Stack predictions from all models
#         model_names = list(predictions_dict.keys())
#         stacked_preds = np.hstack([predictions_dict[name] for name in model_names])
        
#         self.n_classes = stacked_preds.shape[1]
#         self.model_names = model_names
        
#         # Normalize features if requested
#         if self.normalize_features:
#             self.scaler = StandardScaler()
#             X_scaled = self.scaler.fit_transform(stacked_preds)
#         else:
#             X_scaled = stacked_preds
        
#         # Fit Logistic Regression (handles multiclass automatically)
#         self.logistic_model = LogisticRegression(**{
#                 'solver': 'qn',
#                 'penalty': 'l1',
#                 'C': self.C, 
#                 'tol': 0.00011987826963558092, 
#                 'fit_intercept': True, 
#                 'class_weight': None,
#                 'max_iter': 10_000,
#                 'random_state': 42
#             }) 
#         self.logistic_model.fit(X_scaled, y_true)
    
#     def predict_proba(self, predictions_dict):
#         """
#         Generate ensemble predictions
        
#         Args:
#             predictions_dict: Dict of model predictions for test data
            
#         Returns:
#             Ensemble probabilities (samples x classes)
#         """
#         # Stack predictions
#         stacked_preds = np.hstack([predictions_dict[name] for name in self.model_names])
        
#         # Scale features if normalization was used
#         if self.normalize_features:
#             X_scaled = self.scaler.transform(stacked_preds)
#         else:
#             X_scaled = stacked_preds
        
#         # Get probability predictions
#         ensemble_probs = self.logistic_model.predict_proba(X_scaled)
        
#         return ensemble_probs

# def evaluate_logistic_ensemble_kfold(oof_predictions, test_predictions, y_train, 
#                                    C_values=[0.01, 0.1, 1.0, 10.0, 100.0],
#                                    n_folds=5, random_state=42):
#     """
#     Evaluate logistic ensemble using k-fold cross-validation
    
#     Args:
#         oof_predictions: Dict of out-of-fold predictions from base models
#         test_predictions: Dict of test predictions from base models
#         y_train: Training labels
#         C_values: List of C values to test
#         n_folds: Number of folds for cross-validation
#         random_state: Random state for reproducibility
    
#     Returns:
#         best_ensemble: Trained ensemble with best C value
#         test_probs: Test predictions from best ensemble
#         best_map3: Best MAP@3 score achieved
#         le: Label encoder used
#         cv_results: Dictionary with detailed CV results
#     """
#     # Convert pandas Series to numpy array and encode labels
#     if hasattr(y_train, 'values'):
#         y_train_array = y_train.values
#     else:
#         y_train_array = np.array(y_train)
    
#     # Create label encoder for string labels
#     le = LabelEncoder()
#     y_train_encoded = le.fit_transform(y_train_array)
    
#     # Initialize k-fold cross-validation
#     kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
#     # Store results for each C value
#     cv_results = {}
#     best_map3 = 0
#     best_C = None
#     best_ensemble = None
    
#     print(f"Running {n_folds}-fold cross-validation for hyperparameter tuning...")
#     print("="*60)
    
#     for C in C_values:
#         print(f"\nTesting C = {C}")
#         fold_scores = []
        
#         # Perform k-fold CV for this C value
#         for fold, (train_idx, val_idx) in enumerate(kfold.split(y_train_encoded, y_train_encoded)):
#             # Split predictions for this fold
#             fold_train_preds = {name: preds[train_idx] for name, preds in oof_predictions.items()}
#             fold_val_preds = {name: preds[val_idx] for name, preds in oof_predictions.items()}
            
#             y_fold_train = y_train_encoded[train_idx]
#             y_fold_val = y_train_encoded[val_idx]
            
#             # Train ensemble on fold training data
#             ensemble = LogisticEnsemble(C=C)
#             ensemble.fit(fold_train_preds, y_fold_train)
            
#             # Predict on fold validation data
#             val_probs = ensemble.predict_proba(fold_val_preds)
            
#             # Calculate MAP@3 for this fold
#             fold_map3 = map_at_k(y_fold_val, val_probs, k=3)
#             fold_scores.append(fold_map3)
            
#             print(f"  Fold {fold+1}: MAP@3 = {fold_map3:.4f}")
        
#         # Calculate mean and std across folds
#         mean_map3 = np.mean(fold_scores)
#         std_map3 = np.std(fold_scores)
        
#         print(f"  Mean MAP@3: {mean_map3:.4f} (+/- {std_map3:.4f})")
        
#         # Store results
#         cv_results[C] = {
#             'fold_scores': fold_scores,
#             'mean_score': mean_map3,
#             'std_score': std_map3
#         }
        
#         # Update best parameters
#         if mean_map3 > best_map3:
#             best_map3 = mean_map3
#             best_C = C
    
#     print("\n" + "="*60)
#     print("Cross-Validation Results Summary:")
#     print("="*60)
#     for C, results in cv_results.items():
#         print(f"C = {C:6.2f}: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
    
#     print(f"\nBest C: {best_C}, Best CV MAP@3: {best_map3:.4f}")
    
#     # Train final ensemble on full training data with best C
#     print(f"\nTraining final ensemble with C = {best_C} on full training data...")
#     best_ensemble = LogisticEnsemble(C=best_C)
#     best_ensemble.fit(oof_predictions, y_train_encoded)
    
#     # Generate final test predictions
#     test_ensemble_probs = best_ensemble.predict_proba(test_predictions)
    
#     return best_ensemble, test_ensemble_probs, best_map3, le, cv_results

# # Alternative function for nested cross-validation (more robust but slower)
# def evaluate_logistic_ensemble_nested_cv(oof_predictions, test_predictions, y_train,
#                                         C_values=[0.01, 0.1, 1.0, 10.0, 100.0],
#                                         outer_folds=5, inner_folds=3, random_state=42):
#     """
#     Evaluate logistic ensemble using nested cross-validation
#     This provides an unbiased estimate of model performance
#     """
#     # Convert and encode labels
#     if hasattr(y_train, 'values'):
#         y_train_array = y_train.values
#     else:
#         y_train_array = np.array(y_train)
    
#     le = LabelEncoder()
#     y_train_encoded = le.fit_transform(y_train_array)
    
#     # Outer CV for performance estimation
#     outer_kfold = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
#     outer_scores = []
#     best_C_per_fold = []
    
#     print(f"Running nested {outer_folds}x{inner_folds} cross-validation...")
#     print("="*70)
    
#     for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_kfold.split(y_train_encoded, y_train_encoded)):
#         print(f"\nOuter Fold {outer_fold + 1}/{outer_folds}")
#         print("-" * 40)
        
#         # Split data for outer fold
#         outer_train_preds = {name: preds[outer_train_idx] for name, preds in oof_predictions.items()}
#         outer_test_preds = {name: preds[outer_test_idx] for name, preds in oof_predictions.items()}
#         y_outer_train = y_train_encoded[outer_train_idx]
#         y_outer_test = y_train_encoded[outer_test_idx]
        
#         # Inner CV for hyperparameter tuning
#         inner_kfold = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
#         best_inner_score = 0
#         best_inner_C = None
        
#         for C in C_values:
#             inner_scores = []
            
#             for inner_train_idx, inner_val_idx in inner_kfold.split(y_outer_train, y_outer_train):
#                 # Further split for inner CV
#                 inner_train_preds = {name: preds[inner_train_idx] for name, preds in outer_train_preds.items()}
#                 inner_val_preds = {name: preds[inner_val_idx] for name, preds in outer_train_preds.items()}
#                 y_inner_train = y_outer_train[inner_train_idx]
#                 y_inner_val = y_outer_train[inner_val_idx]
                
#                 # Train and evaluate
#                 ensemble = LogisticEnsemble(C=C)
#                 ensemble.fit(inner_train_preds, y_inner_train)
#                 val_probs = ensemble.predict_proba(inner_val_preds)
#                 inner_score = map_at_k(y_inner_val, val_probs, k=3)
#                 inner_scores.append(inner_score)
            
#             mean_inner_score = np.mean(inner_scores)
#             if mean_inner_score > best_inner_score:
#                 best_inner_score = mean_inner_score
#                 best_inner_C = C
        
#         best_C_per_fold.append(best_inner_C)
#         print(f"  Best C for this fold: {best_inner_C} (CV score: {best_inner_score:.4f})")
        
#         # Train final model for this outer fold with best C
#         final_ensemble = LogisticEnsemble(C=best_inner_C)
#         final_ensemble.fit(outer_train_preds, y_outer_train)
#         outer_test_probs = final_ensemble.predict_proba(outer_test_preds)
#         outer_score = map_at_k(y_outer_test, outer_test_probs, k=3)
#         outer_scores.append(outer_score)
        
#         print(f"  Outer fold score: {outer_score:.4f}")
    
#     # Final results
#     mean_outer_score = np.mean(outer_scores)
#     std_outer_score = np.std(outer_scores)
#     most_common_C = max(set(best_C_per_fold), key=best_C_per_fold.count)
    
#     print("\n" + "="*70)
#     print("Nested Cross-Validation Results:")
#     print("="*70)
#     print(f"Estimated model performance: {mean_outer_score:.4f} (+/- {std_outer_score:.4f})")
#     print(f"Best C values per fold: {best_C_per_fold}")
#     print(f"Most common best C: {most_common_C}")
    
#     # Train final model on full data
#     print(f"\nTraining final model with C = {most_common_C}")
#     final_ensemble = LogisticEnsemble(C=most_common_C)
#     final_ensemble.fit(oof_predictions, y_train_encoded)
#     test_probs = final_ensemble.predict_proba(test_predictions)
    
#     return final_ensemble, test_probs, mean_outer_score, le, {
#         'outer_scores': outer_scores,
#         'best_C_per_fold': best_C_per_fold,
#         'most_common_C': most_common_C
#     }

# ensemble, test_probs, map3_score, le, cv_results = evaluate_logistic_ensemble_kfold(
#         oof_predictions=oof_predictions,
#         test_predictions=test_predictions, 
#         y_train=train_df['Fertilizer Name'],
#         C_values=[0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
#         n_folds=5
# )
    
# print(f"\nFinal ensemble MAP@3 score: {map3_score:.4f}")
# print(f"Test predictions shape: {test_probs.shape}")
    
# # Uncomment for nested CV (more robust but slower)
# # ensemble_nested, test_probs_nested, nested_score, le_nested, nested_results = evaluate_logistic_ensemble_nested_cv(
# #         oof_predictions=oof_predictions,
# #         test_predictions=test_predictions, 
# #         y_train=train_df['Fertilizer Name'],
# #         C_values=[0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
# #         outer_folds=5,
# #         inner_folds=3
# # )

# # print(f"\nFinal ensemble MAP@3 score: {nested_score}")


top3_indices = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
final_predictions = [
    " ".join(str(name) for name in label_encoder.inverse_transform(indices)) 
    for indices in top3_indices
]


sub = pd.DataFrame({'id':test_df.index,
                   'Fertilizer Name':final_predictions})
sub.to_csv('submission.csv',index=False)
sub

