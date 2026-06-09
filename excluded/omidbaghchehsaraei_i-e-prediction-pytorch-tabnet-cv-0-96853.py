!pip install pytorch-tabnet


import torch
import optuna
import warnings
import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Load Data ---
TRAIN_PATH = "/kaggle/input/playground-series-s5e7/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e7/test.csv"
SAMPLE_SUBMISSION_PATH = "/kaggle/input/playground-series-s5e7/sample_submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Sample Submission shape: {sample_submission_df.shape}")

print("\nTrain Data Head:")
print(train_df.head()) 

# --- Preprocessing ---

# Define features (X) and target (y) 
features = [col for col in train_df.columns if col not in ['id', 'Personality']]
X = train_df[features]
y = train_df['Personality']
X_test = test_df[features] 

print(f"\nFeatures selected: {features}")

numerical_features = [
    'Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
    'Friends_circle_size', 'Post_frequency',
]
categorical_features = ['Stage_fear', 'Drained_after_socializing']

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' 
)

# Map 'Extrovert' to 0 and 'Introvert' to 1 for consistent log_loss and accuracy calculation
y_binary_true_for_metrics = y.map({'Extrovert': 0, 'Introvert': 1})

# Determine device for TabNet (GPU if available, else CPU)
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device_name}")

# --- Optuna Objective Function for Hyperparameter Tuning ---
def objective(trial):
    # TabNet Hyperparameters to tune
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])

    tabnet_params = {
        'n_d': trial.suggest_int('n_d', 8, 64, step=8),
        'n_a': trial.suggest_int('n_a', 8, 64, step=8),
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0, step=0.2),
        'n_independent': trial.suggest_int('n_independent', 1, 5),
        'n_shared': trial.suggest_int('n_shared', 1, 5),
        'mask_type': trial.suggest_categorical('mask_type', ['sparsemax', 'entmax']),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True),
        'clip_value': trial.suggest_float('clip_value', 0.5, 2.0),
        'optimizer_fn': optim.Adam,
        'optimizer_params': dict(lr=trial.suggest_float('lr', 1e-3, 1e-2, log=True)),
        'scheduler_fn': ReduceLROnPlateau,
        'scheduler_params': dict(
            mode='min',
            factor=trial.suggest_float('factor', 0.1, 0.5),
            patience=trial.suggest_int('patience_scheduler', 5, 15),
            min_lr=1e-6,
            verbose=False
        ),
        'verbose': 0,
        'seed': 42,
        'device_name': device_name
    }

    FOLDS = 5
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    fold_logloss_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        y_val_fold_binary = y_binary_true_for_metrics.iloc[val_idx]

        # Preprocess data first to get numerical numpy arrays for TabNet
        X_train_processed = preprocessor.fit_transform(X_train_fold)
        X_val_processed = preprocessor.transform(X_val_fold)

        y_train_processed = y_binary_true_for_metrics.iloc[train_idx].values.astype(int)
        y_val_processed = y_binary_true_for_metrics.iloc[val_idx].values.astype(int)

        # Initialize TabNetClassifier
        model = TabNetClassifier(**tabnet_params)
        
        try:
            model.fit(
                X_train=X_train_processed, y_train=y_train_processed,
                eval_set=[(X_val_processed, y_val_processed)],
                eval_metric=['logloss'],
                patience=trial.suggest_int('patience_early_stopping', 20, 50),
                max_epochs=50,
                batch_size=batch_size,
                virtual_batch_size=batch_size // 2,
                drop_last=False 
            )
            
            val_preds_proba = model.predict_proba(X_val_processed)
            current_logloss = log_loss(y_val_fold_binary, val_preds_proba[:, 1])
            fold_logloss_scores.append(current_logloss)
        except Exception as e:
            print(f"Trial {trial.number} encountered error during TabNet training: {e}")
            return float('inf') 

    return np.mean(fold_logloss_scores)

# --- Run Optuna Study ---
print("\nStarting Optuna Hyperparameter Optimization for TabNet...") 
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True) 

print("\nOptuna optimization finished.")
print(f"Best trial number: {study.best_trial.number}")
print(f"Best LogLoss (CV): {study.best_value:.5f}")
print("Best hyperparameters:")
best_tabnet_params = study.best_params

# Reconstruct TabNet params for final model
final_tabnet_params_for_constructor = {
    'n_d': best_tabnet_params['n_d'],
    'n_a': best_tabnet_params['n_a'],
    'n_steps': best_tabnet_params['n_steps'],
    'gamma': best_tabnet_params['gamma'],
    'n_independent': best_tabnet_params['n_independent'],
    'n_shared': best_tabnet_params['n_shared'],
    'mask_type': best_tabnet_params['mask_type'],
    'lambda_sparse': best_tabnet_params['lambda_sparse'],
    'clip_value': best_tabnet_params['clip_value'],
    'optimizer_fn': optim.Adam,
    'optimizer_params': dict(lr=best_tabnet_params['lr']),
    'scheduler_fn': ReduceLROnPlateau,
    'scheduler_params': dict(
        mode='min',
        factor=best_tabnet_params['factor'],
        patience=best_tabnet_params['patience_scheduler'],
        min_lr=1e-6,
        verbose=False
    ),
    'verbose': 0,
    'seed': 42,
    'device_name': device_name
}

# Extract batch_size and patience_early_stopping specifically for the .fit() method
final_batch_size = best_tabnet_params['batch_size']
final_patience_early_stopping = best_tabnet_params['patience_early_stopping']


for key, value in best_tabnet_params.items():
    print(f"  {key}: {value}")


# --- Final Model Training with Best Parameters ---
print("\nTraining final TabNet model with best hyperparameters...") 

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds_proba = np.zeros(len(X))
test_preds_proba_agg = np.zeros((len(X_test), 2)) 
fold_logloss_scores = []
fold_accuracy_scores = []

X_processed_full = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)
y_processed_full = y_binary_true_for_metrics.values.astype(int)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed_full, y_processed_full)):
    print(f"\n{'#'*15} Fold {fold+1}/{FOLDS} {'#'*15}")
    
    X_train_fold, X_val_fold = X_processed_full[train_idx], X_processed_full[val_idx]
    y_train_fold, y_val_fold = y_processed_full[train_idx], y_processed_full[val_idx] 

    # Initialize TabNetClassifier with constructor parameters
    model = TabNetClassifier(**final_tabnet_params_for_constructor)
    
    print(f"  Training TabNet for Fold {fold+1}...")
    
    model.fit(
        X_train=X_train_fold, y_train=y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        eval_metric=['logloss'],
        patience=final_patience_early_stopping,
        max_epochs=50, 
        batch_size=final_batch_size,
        virtual_batch_size=final_batch_size // 2,
        drop_last=False
    )

    val_preds_proba = model.predict_proba(X_val_fold) 
    oof_preds_proba[val_idx] = val_preds_proba[:, 1] 

    current_logloss = log_loss(y_val_fold, val_preds_proba[:, 1]) 
    fold_logloss_scores.append(current_logloss)

    val_preds_class = model.predict(X_val_fold)
    current_accuracy = accuracy_score(y_val_fold, val_preds_class)
    fold_accuracy_scores.append(current_accuracy)

    print(f"  ✅ Fold {fold+1} LogLoss: {current_logloss:.5f}, Accuracy: {current_accuracy:.5f}")

    test_preds_proba_agg += model.predict_proba(X_test_processed) / FOLDS

print(f"\nAverage LogLoss across {FOLDS} folds: {np.mean(fold_logloss_scores):.5f}")
print(f"Average Accuracy across {FOLDS} folds: {np.mean(fold_accuracy_scores):.5f}")

# --- Final OOF Metrics Calculation ---
final_oof_logloss = log_loss(y_binary_true_for_metrics, oof_preds_proba)
print(f"\nFinal Out-Of-Fold LogLoss: {final_oof_logloss:.5f}")

final_oof_predicted_classes_binary = (oof_preds_proba >= 0.5).astype(int)
final_oof_accuracy = accuracy_score(y_binary_true_for_metrics, final_oof_predicted_classes_binary)
print(f"Final Out-Of-Fold Accuracy: {final_oof_accuracy:.5f}")


# --- Save OOF and Test Predictions for Ensembling ---

oof_df = pd.DataFrame({'id': train_df['id'], 'tabnet_oof_proba_I': oof_preds_proba})
oof_df.to_csv('oof_predictions_tabnet.csv', index=False)
print(f"\nOOF predictions saved to oof_predictions_tabnet.csv")
print("OOF Predictions Head:")
print(oof_df.head())

test_preds_df = pd.DataFrame({
    'id': test_df['id'],
    'tabnet_test_proba_E': test_preds_proba_agg[:, 0],
    'tabnet_test_proba_I': test_preds_proba_agg[:, 1]
})
test_preds_df.to_csv('test_predictions_tabnet.csv', index=False)
print(f"\nTest predictions saved to test_predictions_tabnet.csv")
print("Test Predictions Head:")
print(test_preds_df.head())


# --- Create Submission File ---

predicted_classes_test_binary = (test_preds_proba_agg[:, 1] >= 0.5).astype(int) 

submission_labels_map = {0: 'Extrovert', 1: 'Introvert'}
predicted_classes_test_string = np.vectorize(submission_labels_map.get)(predicted_classes_test_binary)

final_submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': predicted_classes_test_string}) 

submission_filename = 'submission.csv'
final_submission_df.to_csv(submission_filename, index=False)

print(f"\nSubmission file '{submission_filename}' created successfully.")
print("Submission Head:")
print(final_submission_df.head())

