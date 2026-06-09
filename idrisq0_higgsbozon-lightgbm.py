# Import necessary libraries
import numpy as np # For linear algebra and numerical operations
import pandas as pd # For data processing and reading CSV files
import matplotlib.pyplot as plt # For basic visualizations
import seaborn as sns # For more appealing visualizations
import lightgbm as lgb # Our model: LightGBM
from sklearn.model_selection import StratifiedKFold # For cross-validation
from sklearn.metrics import roc_auc_score # AUC score for evaluation

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("Libraries imported successfully!")


def ams(y_true, y_pred_labels, weights):
    """
    Calculates the Approximate Median Significance (AMS) score.
    The official evaluation metric for the Higgs Boson ML Challenge.
    
    y_true: array of true labels (1 for signal, 0 for background)
    y_pred_labels: array of predicted labels (1 for signal, 0 for background)
    weights: array of event weights
    """
    # Ensure inputs are numpy arrays for boolean indexing
    y_true = np.array(y_true)
    y_pred_labels = np.array(y_pred_labels)
    weights = np.array(weights)
    
    # Calculate the sum of weights for true positives (s) and false positives (b)
    s = weights[(y_true == 1) & (y_pred_labels == 1)].sum()
    b = weights[(y_true == 0) & (y_pred_labels == 1)].sum()
    
    # The competition's regularization term
    b_reg = 10.0
    
    # The AMS formula
    radicand = 2 * ((s + b + b_reg) * np.log(1 + s / (b + b_reg)) - s)
    
    # Handle cases where the radicand is negative
    if radicand < 0:
        return 0.0
    else:
        return np.sqrt(radicand)


# Define the path to the data files
data_path = '/kaggle/input/higgs-boson/'

# Load the training and test data directly from the zip files
train_df = pd.read_csv(data_path + 'training.zip')
test_df = pd.read_csv(data_path + 'test.zip')

# Display the first 5 rows of the training data
print("Training Data Head:")
display(train_df.head())

# Display the first 5 rows of the test data
print("\nTest Data Head:")
display(test_df.head())

print("\nData loaded successfully from .zip files!")


# --- Check the dimensions of the data ---
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# --- Get a concise summary of the training data ---
print("\nTraining Data Info:")
train_df.info()

# --- Generate descriptive statistics for the training data ---
print("\nTraining Data Description:")
display(train_df.describe())

# --- Check the distribution of the target variable 'Label' ---
print("\nTarget Label Distribution:")
print(train_df['Label'].value_counts())


# Replace the placeholder -999.0 with NumPy's standard NaN
train_df.replace(-999.0, np.nan, inplace=True)
test_df.replace(-999.0, np.nan, inplace=True)

# Calculate and display the number of missing values in the training data
print("Columns with Missing Values (Train Data):\n")
missing_values_train = train_df.isnull().sum()

# Filter to show only columns that have missing values
missing_values_train = missing_values_train[missing_values_train > 0]

# Sort them by the number of missing values, descending
print(missing_values_train.sort_values(ascending=False))


# 4.5 - Feature Engineering

def delta_phi(phi1, phi2):
    """
    Calculates the cyclical difference between two angles (phi).
    The result is always between -pi and +pi.
    """
    dphi = phi1 - phi2
    dphi[dphi > np.pi] -= 2 * np.pi
    dphi[dphi < -np.pi] += 2 * np.pi
    return dphi

def create_physics_features(df):
    """
    Engineers new physics-inspired features from the original data.
    """
    print(f"Original number of features: {df.shape[1]}")
    new_df = df.copy()

    # --- Transverse mass between lepton and Missing Transverse Energy (MET) ---
    # This is a very powerful feature for W boson decay channels.
    new_df['DER_mass_transverse_met_lep'] = np.sqrt(
        2 * new_df['PRI_lep_pt'] * new_df['PRI_met'] * (1 - np.cos(delta_phi(new_df['PRI_lep_phi'], new_df['PRI_met_phi'])))
    )

    # --- Angular separations ---
    # Delta R is a common measure of separation in particle physics
    new_df['DER_delta_R_jet_lep'] = np.sqrt(
        (new_df['PRI_jet_leading_eta'] - new_df['PRI_lep_eta'])**2 +
        delta_phi(new_df['PRI_jet_leading_phi'], new_df['PRI_lep_phi'])**2
    )

    # --- Ratios of momenta ---
    # These can help distinguish between different physical processes.
    new_df['DER_pt_ratio_lep_jet'] = new_df['PRI_lep_pt'] / new_df['PRI_jet_leading_pt']
    new_df['DER_pt_ratio_lep_met'] = new_df['PRI_lep_pt'] / (new_df['PRI_met'] + 1e-6) # Add epsilon to avoid division by zero

    # --- Log transformations for skewed features ---
    # This helps the model handle features with very large values and long tails.
    for col in ['PRI_tau_pt', 'PRI_lep_pt', 'PRI_met', 'PRI_jet_leading_pt', 'PRI_jet_subleading_pt']:
        if col in new_df.columns:
            new_df[f'LOG_{col}'] = np.log1p(new_df[col].fillna(0)) # FillNa before log to avoid errors

    # --- Combined momentum ---
    # A simplified vector sum of pt for key objects
    new_df['DER_pt_sum_lep_jet_met'] = new_df['PRI_lep_pt'] + new_df['PRI_jet_leading_pt'] + new_df['PRI_met']
    
    print(f"New number of features: {new_df.shape[1]}")
    return new_df


# Apply the feature engineering function to both dataframes
train_df = create_physics_features(train_df)
test_df = create_physics_features(test_df)

print("\nFeature engineering complete!")


# --- 1. Define the columns that are not features ---
# We will drop 'EventId' and 'Label'. 'Weight' will be stored separately.
features = [col for col in train_df.columns if col not in ['EventId', 'Weight', 'Label']]

# --- 2. Create the feature matrix X ---
X = train_df[features]

# --- 3. Create the target vector y and the weights vector ---
y = train_df['Label'].map({'s': 1, 'b': 0})
weights = train_df['Weight'] # Store the weights in their own variable

# --- 4. Prepare the test set features ---
X_test = test_df[features]

# --- Verification Step ---
print(f"Features shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")
print(f"Weights shape: {weights.shape}") # Check weights shape
print(f"Test features shape (X_test): {X_test.shape}")

# Display the first few rows of the final feature matrix
print("\nFirst 5 rows of our feature matrix (X):")
display(X.head())

# Display the first few values of our target vector (y)
print("\nFirst 5 values of our target vector (y):")
display(y.head())


# NEW SECTION: 6.1 - Hyperparameter Tuning with Optuna
import optuna

# This is the objective function that Optuna will run on each trial.
# It trains a model with a given set of parameters and returns its performance (AUC score).
def objective(trial):
    # We tell Optuna which parameters to try and in what ranges.
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 2000,
        'verbosity': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    
    # We use Cross-Validation to get a robust score for each set of parameters.
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Using 3 folds for speed during optimization
    cv_scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        weights_train = weights.iloc[train_idx]
        
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        weights_val = weights.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        
        early_stopping_callback = lgb.early_stopping(100, verbose=False)
        
        model.fit(X_train, y_train, 
                  sample_weight=weights_train,
                  eval_set=[(X_val, y_val)],
                  eval_sample_weight=[weights_val],
                  callbacks=[early_stopping_callback])
        
        val_preds = model.predict_proba(X_val)[:, 1]
        cv_scores.append(roc_auc_score(y_val, val_preds, sample_weight=weights_val))
        
    # Return the mean score from the folds
    return np.mean(cv_scores)

# Start the optimization study
# direction='maximize' -> try to maximize the value returned by the objective function (AUC)
study = optuna.create_study(direction='maximize')

# n_trials: how many different parameter combinations to test.
# This can be increased for better results if you have time.
study.optimize(objective, n_trials=30) # Run 30 trials. You can increase this to 50-100.

# Get the best parameters found by the study
best_params = study.best_params

# Add the fixed parameters back into our dictionary of best parameters
best_params['objective'] = 'binary'
best_params['metric'] = 'auc'
best_params['n_estimators'] = 2000
best_params['verbosity'] = -1
best_params['n_jobs'] = -1
best_params['seed'] = 42
best_params['boosting_type'] = 'gbdt'

print("\n✅ Optimization complete!")
print("Best AUC score found:", study.best_value)
print("Best parameters found:", best_params)

# IMPORTANT: We will now use the `best_params` variable, not the old `params` variable.

# --- Setup for the final cross-validation run ---
N_SPLITS = 5 # Let's increase the number of folds back to 5 for the final training
kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Initialize containers to store results
scores = []
test_predictions = []
feature_importances = pd.DataFrame(index=X.columns)

print("\nSetup complete! Best hyperparameters are ready for the final training.")


# This is the main training loop.
best_thresholds = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f"===== Fold {fold+1} =====")
    
    # ... (the data splitting part remains the same) ...
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    weights_train = weights.iloc[train_idx] 
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    weights_val = weights.iloc[val_idx] 
    
    # ... (model initialization and early stopping remain the same) ...
    model = lgb.LGBMClassifier(**best_params)
    early_stopping_callback = lgb.early_stopping(100, verbose=False)
    
    model.fit(X_train, y_train, 
              sample_weight=weights_train,
              eval_set=[(X_val, y_val)],
              eval_sample_weight=[weights_val],
              callbacks=[early_stopping_callback])
              
    # --- START OF MODIFIED BLOCK ---
    
    # 5. Make predictions on validation set
    val_preds_proba = model.predict_proba(X_val)[:, 1]
    
    # Store test predictions for this fold
    test_fold_preds = model.predict_proba(X_test)[:, 1]
    test_predictions.append(test_fold_preds)
    
    # 6. Calculate weighted AUC score for the fold
    fold_auc = roc_auc_score(y_val, val_preds_proba, sample_weight=weights_val)
    scores.append(fold_auc)
    
    # 7. Find the best threshold to maximize AMS for this fold
    best_ams_fold = 0
    best_threshold_fold = 0
    # Iterate through a range of possible thresholds
    for threshold in np.arange(0.1, 0.9, 0.01):
        # Convert probabilities to labels using the current threshold
        val_pred_labels = (val_preds_proba > threshold).astype(int)
        # Calculate AMS
        current_ams = ams(y_val, val_pred_labels, weights_val)
        if current_ams > best_ams_fold:
            best_ams_fold = current_ams
            best_threshold_fold = threshold
            
    best_thresholds.append(best_threshold_fold)
    
    # Store the feature importances
    feature_importances[f'fold_{fold+1}'] = model.feature_importances_
    
    # Print the final scores for this fold
    print(f"Fold {fold+1} Weighted AUC: {fold_auc:.5f} | Best AMS: {best_ams_fold:.4f} at threshold {best_threshold_fold:.2f}")

    # --- END OF MODIFIED BLOCK ---

print("\n Training complete! All folds have been processed.")


# --- 1. Calculate and display the final Cross-Validation Score ---
print(f"Mean CV AUC Score: {np.mean(scores):.5f} ± {np.std(scores):.5f}")


# --- 2. Average the test set predictions from all folds ---
# This creates our final set of predictions for the submission file.
final_test_preds = np.mean(test_predictions, axis=0)
print(f"Shape of our final test predictions: {final_test_preds.shape}")


# --- 3. Visualize Feature Importance ---
# First, calculate the mean importance across all folds
feature_importances['mean'] = feature_importances.mean(axis=1)

# Then, sort the features by their mean importance
feature_importances.sort_values(by='mean', ascending=False, inplace=True)

# Now, create the plot
plt.figure(figsize=(12, 10))
sns.barplot(x='mean', 
            y=feature_importances.index, 
            data=feature_importances,
            palette='viridis')

plt.title('LightGBM Feature Importance (Averaged over 5 Folds)')
plt.xlabel('Average Importance Score')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()


# NEW SECTION: 7 - Create the Submission File

print("Creating submission file...")

# --- 1. Calculate the optimal threshold ---
# We use the average of the best thresholds found during cross-validation.
optimal_threshold = np.mean(best_thresholds)
print(f"Using optimal threshold of {optimal_threshold:.4f} (averaged over {N_SPLITS} folds)")

# --- 2. Create the submission DataFrame ---
submission_df = pd.DataFrame({'EventId': test_df['EventId']})

# --- 3. Determine the RankOrder ---
# The rank is based on the descending order of signal probabilities.
# Using pandas .rank() is a direct and efficient way to do this.
submission_df['RankOrder'] = pd.Series(final_test_preds).rank(ascending=False).astype(int)

# --- 4. Determine the Class using the optimal threshold ---
submission_df['Class'] = ['s' if prob > optimal_threshold else 'b' for prob in final_test_preds]

# --- 5. Save the file ---
submission_df.to_csv('submission_final.csv', index=False)

print("\n✅ Submission file created successfully!")
print("Here are the first 5 rows of your submission file:")
display(submission_df.head())

