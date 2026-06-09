!pip install scikit-learn==1.5.2 koolbox


import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations and array handling

import warnings  # For controlling warning messages
import optuna  # For hyperparameter optimization

from sklearn.linear_model import LogisticRegression  # For logistic regression modeling
from sklearn.model_selection import StratifiedKFold  # For stratified k-fold cross-validation
from sklearn.metrics import accuracy_score  # For evaluating model accuracy
from sklearn.impute import SimpleImputer  # For handling missing data

warnings.filterwarnings('ignore')  # Suppress warning messages


class CFG:
    # File paths
    train_path = '/kaggle/input/playground-series-s5e7/train.csv'
    test_path = '/kaggle/input/playground-series-s5e7/test.csv'
    sample_sub_path = '/kaggle/input/playground-series-s5e7/sample_submission.csv'
    original_path = "/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_dataset.csv"
    
    # Model configuration
    target = 'Personality'
    n_folds = 5
    seed = 42
    cv = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    metric = accuracy_score
    n_optuna_trials = 500


# Load data
train = pd.read_csv(CFG.train_path, index_col='id')
test = pd.read_csv(CFG.test_path, index_col='id')

# Load and prepare original data
original = pd.read_csv(CFG.original_path)
original = original.rename(columns={'Personality': 'match_p'})
original = original.drop_duplicates([
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency'
])

# Merge with original data
train = train.merge(original, how='left')
test = test.merge(original, how='left')

# Map target to numeric (0: Extrovert, 1: Introvert)
train[CFG.target] = train[CFG.target].map({"Extrovert": 0, "Introvert": 1})

# Drop match_p column (99% missing, not useful)
train = train.drop('match_p', axis=1)
test = test.drop('match_p', axis=1)

# Handle categorical columns
cat_cols = ["Stage_fear", "Drained_after_socializing"]
for col in cat_cols:
    train[col] = train[col].fillna("missing").astype("category").cat.codes
    test[col] = test[col].fillna("missing").astype("category").cat.codes

# Identify numeric columns with missing values
numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 
                'Going_outside', 'Friends_circle_size', 'Post_frequency']

# Impute missing values in numeric columns
imputer = SimpleImputer(strategy='median')
train[numeric_cols] = imputer.fit_transform(train[numeric_cols])
test[numeric_cols] = imputer.transform(test[numeric_cols])

# Prepare features and target
X = train.drop(CFG.target, axis=1)
y = train[CFG.target]
X_test = test

print(f"Training features shape: {X.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Target distribution: {y.value_counts()}")


class Trainer:
    """
    Sets up the trainer with a model, cross-validation strategy, evaluation metric, and other configuration options.

    This allows the trainer to be flexible and reusable for different models, metrics, and tasks.
    """
    def __init__(self, model, cv, metric, metric_precision=6, 
                 metric_threshold=0.5, use_early_stopping=False, 
                 verbose=False, task="binary"):
        # Store the model, cross-validation splitter, and metric function
        self.model = model
        self.cv = cv
        self.metric = metric
        self.metric_precision = metric_precision  # Number of decimal places for metric output
        self.metric_threshold = metric_threshold  # Threshold for converting probabilities to class labels
        self.verbose = verbose  # Whether to print progress
        self.task = task  # Task type (e.g., "binary" classification)
        self.fold_scores = []  # List to store scores for each fold
        self.models = []  # List to store trained models for each fold

    def fit(self, X, y):
        # Reset fold scores and models before training
        self.fold_scores = []
        self.models = []
        
        # Loop over each fold provided by the cross-validation splitter
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            # Split data into training and validation sets for this fold
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone the model to ensure a fresh model for each fold
            fold_model = self.model.__class__(**self.model.get_params())
            # Train the model on the training data
            fold_model.fit(X_train, y_train)
            # Save the trained model
            self.models.append(fold_model)
            
            # Predict probabilities for the validation set
            y_pred_proba = fold_model.predict_proba(X_val)[:, 1]
            # Convert probabilities to binary predictions using the threshold
            y_pred = (y_pred_proba >= self.metric_threshold).astype(int)
            
            # Calculate the evaluation metric for this fold
            score = self.metric(y_val, y_pred)
            # Store the score
            self.fold_scores.append(score)
            
            # Optionally print the score for this fold
            if self.verbose:
                print(f"Fold {fold + 1} Score: {score:.{self.metric_precision}f}")
                
    def predict(self, X_test):
        # Initialize an array to accumulate predictions from each fold model
        predictions = np.zeros(len(X_test))
        # Sum the predicted probabilities from each fold model
        for model in self.models:
            predictions += model.predict_proba(X_test)[:, 1]
        # Average the predictions across all folds
        return predictions / len(self.models)


import logging
import sys
from contextlib import redirect_stdout, redirect_stderr
import io

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Also suppress the underlying logger
logging.getLogger("optuna").setLevel(logging.WARNING)

def objective(trial):
    """Objective function for Optuna optimization"""
    try:
        # Define solver and penalty combinations
        solver_penalty_options = [
            ('liblinear', 'l1'),
            ('liblinear', 'l2'),
            ('lbfgs', 'l2'),
            ('lbfgs', None),
            ('newton-cg', 'l2'),
            ('newton-cg', None),
            ('newton-cholesky', 'l2'),
            ('newton-cholesky', None)
        ]
        
        # Sample hyperparameters
        solver, penalty = trial.suggest_categorical('solver_penalty', solver_penalty_options)
        
        params = {
            'random_state': CFG.seed,
            'max_iter': 1000,
            'C': trial.suggest_float('C', 0.01, 10.0, log=True),
            'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            'solver': solver,
            'penalty': penalty
        }
        
        # Classification threshold
        threshold = trial.suggest_float('threshold', 0.3, 0.7)
        
        # Train and evaluate
        trainer = Trainer(
            LogisticRegression(**params),
            cv=CFG.cv,
            metric=CFG.metric,
            metric_precision=6,
            metric_threshold=threshold,
            verbose=False,
            task="binary",
        )
        
        trainer.fit(X, y)
        return np.mean(trainer.fold_scores)
        
    except Exception as e:
        return 0.0

# Run optimization silently
print("Starting hyperparameter optimization (this may take a few minutes)...")

# Reduce number of trials for faster execution
n_trials = min(CFG.n_optuna_trials, 200)

# Create sampler
sampler = optuna.samplers.TPESampler(
    seed=CFG.seed, 
    multivariate=True, 
    n_startup_trials=n_trials // 10
)

# Create study without verbosity
study = optuna.create_study(
    direction='maximize', 
    sampler=sampler
)

# Suppress all output during optimization
with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
    try:
        study.optimize(
            objective, 
            n_trials=n_trials, 
            n_jobs=4,
            show_progress_bar=False,  # Disable progress bar
            gc_after_trial=True
        )
    except KeyboardInterrupt:
        pass

# Extract best parameters
if len(study.trials) > 0:
    best_params = study.best_params
    print(f"\n✓ Optimization completed successfully!")
    print(f"  - Trials completed: {len(study.trials)}")
    print(f"  - Best CV score: {study.best_value:.6f}")
    print(f"\nBest parameters found:")
    for key, value in best_params.items():
        if key == 'solver_penalty':
            print(f"  - {key}: solver={value[0]}, penalty={value[1]}")
        else:
            print(f"  - {key}: {value}")
else:
    # Fallback parameters
    print("Optimization failed. Using default parameters.")
    best_params = {
        'solver_penalty': ('lbfgs', None),
        'C': 1.0,
        'tol': 1e-4,
        'fit_intercept': True,
        'class_weight': None,
        'threshold': 0.5
    }


# Extract and prepare best parameters
solver, penalty = best_params['solver_penalty']
best_threshold = best_params['threshold']

lr_params = {
    'random_state': CFG.seed,
    'max_iter': 1000,
    'C': best_params['C'],
    'tol': best_params['tol'],
    'fit_intercept': best_params['fit_intercept'],
    'class_weight': best_params['class_weight'],
    'solver': solver,
    'penalty': penalty
}

print(f'Best threshold: {best_threshold:.3f}')

# Train final model
lr_trainer = Trainer(
    LogisticRegression(**lr_params),
    cv=CFG.cv,
    metric=CFG.metric,
    metric_threshold=best_threshold,
    metric_precision=6,
    verbose=True,
    task="binary",
)

lr_trainer.fit(X, y)

# Get predictions
lr_test_pred_probs = lr_trainer.predict(X_test)

print(f"\nMean CV Score: {np.mean(lr_trainer.fold_scores):.6f}")
print(f"Std CV Score: {np.std(lr_trainer.fold_scores):.6f}")


def save_submission(name, test_pred_probs, score, threshold=0.5):
    """Save submission file"""
    sub = pd.read_csv(CFG.sample_sub_path)
    
    # Apply threshold and map to labels
    sub[CFG.target] = (test_pred_probs > threshold).astype(int)
    sub[CFG.target] = sub[CFG.target].map({0: "Extrovert", 1: "Introvert"})
    
    # Save file
    filename = f'submission_{name}_{score:.6f}.csv'
    sub.to_csv(filename, index=False)
    print(f"Submission saved: {filename}")
    
    # Display sample
    print("\nSubmission preview:")
    print(sub.head())
    print(f"\nPrediction distribution:")
    print(sub[CFG.target].value_counts())
    
    return sub

# Save final submission
final_score = np.mean(lr_trainer.fold_scores)
submission = save_submission(
    'logistic_regression', 
    lr_test_pred_probs, 
    final_score, 
    best_threshold
)




