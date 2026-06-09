!pip install itables wandb==0.23.1 optuna==4.6.0 -q
!pip install "protobuf<6.0.0" # hopefully this will fix --> AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'


# Global configs
CFG = {
    "COMPETITION"     : "s5e12",
    "KAGGLE_DATA_PATH": "/kaggle/input/playground-series-s5e12/",
    "EXTRA_TRAIN_PATH": "",
    "TARGET"          : "diagnosed_diabetes",
    "METRIC"          : "roc_auc_score",
    "FOLDS"           : 5,
    "SEED"            : 42,
    "TASK_TYPE"       : "classification",
    "MODEL_TYPE"      : "linear",

    "SMOKE_TEST"      : False,
    "REPORT_TO_WANDB" : False,
    "USE_ACCELERATOR" : True, 
    "MIXED_PRECISION" : True,
    "SEARCH_WEIGHTS"  : True,
    "OPTUNA_N_TRIALS" : 50,   # as of now, only implemented for weights
}

# Load W&B secret
if CFG["REPORT_TO_WANDB"]:
    try:
        import os
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        CFG["WANDB_API_KEY"] = WANDB_API_KEY
        print("âœ… Secrets loaded.")

    except Exception as e:
        WANDB_API_KEY = None
        CFG["WANDB_API_KEY"] = None
        print(f"Exception: {e}")
        print("â„¹ï¸� No Kaggle secrets found (using default public mode).")

# Enable cuml accelerator | https://docs.rapids.ai/api/cuml/stable/cuml-accel/
if CFG["USE_ACCELERATOR"]:
    try:
        import cuml.accel
        cuml.accel.install()
        print("âš¡ GPU Acceleration Enabled")
    except (ImportError, RuntimeError, Exception) as e:
        print(f"â„¹ï¸� cuML Acceleration not available: {e}. Using standard CPU paths.")


import os
import time
import random
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
from itables import init_notebook_mode, show
init_notebook_mode(all_interactive=False,connected=True)

# Sklearn Preprocessing
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute          import SimpleImputer
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline

# The Models
from sklearn.linear_model    import LogisticRegression, SGDClassifier
from sklearn.ensemble        import StackingClassifier

# Hyperparameter search
import optuna

# Deep Learning
import tensorflow as tf
from tensorflow       import keras
from tensorflow.keras import layers, callbacks, optimizers, mixed_precision

# DNN Strategy Initialization
try:
    tpu  = None # Placeholder for TPU if ever needed
    gpus = tf.config.list_physical_devices('GPU')

    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"âœ… Multi-GPU detected: Running on {len(gpus)} GPUs.")
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy()
        print("âœ… Single GPU detected.")
    else:
        strategy = tf.distribute.get_strategy()
        print("â„¹ï¸� No GPU detected, using CPU.")

except Exception as e:
        strategy = tf.distribute.get_strategy()
        print(f"âš ï¸� Strategy init failed: {e}. Falling back to default.")    

# Mixed Precision Setup (This makes training faster and uses less memory on T4 GPUs) 
if CFG["MIXED_PRECISION"]:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        try:
            mixed_precision.set_global_policy('mixed_float16')
            print("âš¡ Mixed precision enabled (float16).")
        except Exception as e:
            print(f"âš ï¸� Mixed precision failed: {e}")
    else:
        print("â„¹ï¸� Mixed precision skipped: No GPU detected. Using float32 for CPU stability.")

# Metric
from sklearn.metrics import roc_auc_score

# Wandb
import wandb
from wandb.integration.keras import WandbCallback

# Sets the seed for reproducibility in numpy, random, torch CPU, and CUDA.
np.random.seed(CFG["SEED"])
random.seed(CFG["SEED"])
tf.random.set_seed(CFG["SEED"])


# HELPER FUNCTIONS
# -----------------------------

def iqr_outlier_capping(train, valid=None, test=None, columns=None):
    """
    Applies IQR-based outlier capping to specified columns of one, two, or three DataFrames.

    Parameters:
        train (pd.DataFrame): The training DataFrame used to calculate IQR thresholds.
        valid (pd.DataFrame, optional): The validation DataFrame to cap using train thresholds.
        test (pd.DataFrame, optional): The test DataFrame to cap using train thresholds.
        columns (list, optional): List of column names to apply capping to. If None, applies to all numerical columns.

    Returns:
        tuple: A tuple containing:
            - train_capped (pd.DataFrame): Capped training DataFrame.
            - valid_capped (pd.DataFrame or None): Capped validation DataFrame (if provided).
            - test_capped (pd.DataFrame or None): Capped test DataFrame (if provided).

    Note: Make sure there are no nans
    """
    train_capped = train.copy() # Avoid modifying the original DataFrame
    valid_capped = valid.copy() if valid is not None else None
    test_capped = test.copy() if test is not None else None

    if columns is None:
        columns = train.select_dtypes(include='number').columns.tolist()  # All numerical columns

    # Calculate IQR-based thresholds from the training set
    # w/ .dropna() to handle cols with nans: required by np.percentile
    for col in columns:
        Q1 = np.percentile(train[col].dropna(), 25)
        Q3 = np.percentile(train[col].dropna(), 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Show Values
        # print(f'Columns {col}: \tLower Bound is: {lower_bound:.2f} \tUpper Bound is: {upper_bound:.2f}')

        # Cap outliers in the training set
        train_capped[col] = np.clip(train_capped[col], lower_bound, upper_bound)

        # If validation set is provided, cap using training set thresholds
        if valid is not None:
            valid_capped[col] = np.clip(valid[col], lower_bound, upper_bound)

        # If test set is provided, cap using training set thresholds
        if test is not None:
            test_capped[col] = np.clip(test[col], lower_bound, upper_bound)

    return train_capped, valid_capped, test_capped

    # EXAMPLE USE: TRAIN_capped, _, TEST_capped = iqr_outlier_capping(TRAIN_DF.dropna(), None, TEST_DF, columns=TRAIN_DF.select_dtypes('number').columns.difference([target]))

# Analysis of all NUMERIC features

# ============================================================
# Function to create and display plots for a single numerical variable
def numeric_univariate_plots(
    train, test, target
    # extra
    ):

    # Select columns
    focus_cols = train.select_dtypes(np.number).columns.difference([target])

    # Merge data for visualization (without modifying original DataFrames)
    train_temp = train[focus_cols].copy()
    test_temp = test[focus_cols].copy()
    # extra_temp = extra[focus_cols].copy()
    train_temp["Dataset"] = "Train"
    test_temp["Dataset"] = "Test"
    # extra_temp["Dataset"] = "Extra"
    combined_data = pd.concat([
        train_temp,
        test_temp,
        # extra_temp
        ])

    # Start loop
    for col in focus_cols:

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        annot_kws = {'xy': (0.03, 0.75), 'xycoords': 'axes fraction', 'fontsize': 10}

        # Box plot
        sns.boxplot(data=combined_data, x=col, y="Dataset", palette=CFG.PALETTE_1, ax=axes[0])
        axes[0].set_xlabel(col)
        axes[0].set_title(f"Box Plot of {col}")

        # Histogram
        if combined_data[col].nunique() > 15:
            sns.histplot(data=combined_data, x=col, hue='Dataset', palette=CFG.PALETTE_1, bins=50,
                         stat='density', common_norm=False, multiple='dodge', kde=False)
            axes[1].set_xlabel(col)
            axes[1].set_ylabel("Frequency")
            axes[1].set_title(f"Histogram of {col} [Train, Test]")
            # axes[1].set_title(f"Histogram of {col} [Train, Test, Extra]")
            # axes[1].legend()
            axes[1].annotate(f"Skewness (TRAIN): {train[col].skew():.2f}\nKurtosis (TRAIN): {train[col].kurt():.2f}",
                             xy=annot_kws['xy'], xycoords=annot_kws['xycoords'], fontsize=annot_kws['fontsize'])
        else:
            sns.countplot(data=combined_data, x=col, hue='Dataset', palette=CFG.PALETTE_1)
            axes[1].set_xlabel(col)
            axes[1].set_ylabel("Count")
            axes[1].set_title(f"Histogram of {col} [Train, Test]")
            # axes[1].set_title(f"Countplot of {col} [Train, Test, Extra]")
            # axes[1].legend()
            axes[1].annotate(f"Skewness (TRAIN): {train[col].skew():.2f}\nKurtosis (TRAIN): {train[col].kurt():.2f}",
                             xy=annot_kws['xy'], xycoords=annot_kws['xycoords'], fontsize=annot_kws['fontsize'])
        # Adjust spacing and show
        plt.tight_layout()
        plt.show()

# ============================================================
def print_with_sep(text,sep="=",n=30):
  print("\n")
  print(sep*n)
  print('\t',text)
  print(sep*n)

# ============================================================
def print_dataset_overview(datasets):

    # Check shapes
    print_with_sep("Shapes")
    for name, df in datasets.items():
      print(f"{name} shape: {df.shape}")

    # Check duplicates
    print_with_sep("Duplicates")
    for name, df in datasets.items():
      print(f"{name} duplicates: {df.duplicated().sum()}")

    # Check nans
    print_with_sep("NaNs")
    for name, df in datasets.items():
      print(f"{name} NaNs: {df.isnull().sum().sum()}")

    # Check col difference
    print_with_sep("Columns not in test")
    for name, df in datasets.items():
        print(set(datasets["train"].columns).difference(set(datasets["test"].columns)))
    
    # Check descriptive stats
    print_with_sep("Descriptive Statistics")
    for name, df in datasets.items():
        print(f"{name} Description:")
        percentage_missing = df.isnull().sum()/df.shape[0]; percentage_missing.name = '% Missing'
        data_types = df.dtypes; data_types.name = 'd_type'

        display(
            pd.concat([
                df.describe(include='all').T,
                percentage_missing,
                data_types],
                      axis=1).replace(np.nan,'-').style.background_gradient(cmap='Blues'))
        print("\n")
        
        break # [TEMPORARY] added to only show train_df description
        


# Load Data

if CFG["SMOKE_TEST"]:
    X_TRAIN = pd.read_csv(CFG["KAGGLE_DATA_PATH"]+"train.csv").sample(frac=0.01, random_state=CFG["SEED"])
    X_TEST  = pd.read_csv(CFG["KAGGLE_DATA_PATH"]+"test.csv").sample(frac=0.01, random_state=CFG["SEED"])
    y       = X_TRAIN.pop(CFG["TARGET"])
else:
    X_TRAIN = pd.read_csv(CFG["KAGGLE_DATA_PATH"]+"train.csv") 
    X_TEST  = pd.read_csv(CFG["KAGGLE_DATA_PATH"]+"test.csv")  
    y       = X_TRAIN.pop(CFG["TARGET"])


datasets = {
    "train": X_TRAIN,
    "test" : X_TEST,
}

# Drop ID if present (it's noise)
if 'id' in X_TRAIN.columns:
    X_TRAIN = X_TRAIN.drop('id', axis=1)
    test_ids = X_TEST['id'] # Save for submission
    X_TEST = X_TEST.drop('id', axis=1)


# Check ordinal vars first
ordinal_vars = ["education_level", "income_level"]
for var in ordinal_vars:
    print(f"Unique values in {var}: {X_TRAIN[var].unique()}")


# Create maps for custom ordinal encoding
ORDINAL_MAPS = {
    "education_level": {
        "No formal": 0,
        "Highschool": 1,
        "Graduate": 2,
        "Postgraduate": 3
    },
    "income_level": {
        "Low": 0,
        "Lower-Middle": 1,
        "Middle": 2,
        "Upper-Middle": 3,
        "High": 4
    }
}

# Apply multiple different maps to specific columns
for col, mapping in ORDINAL_MAPS.items():
    if col in X_TRAIN.columns:
        X_TRAIN[col] = X_TRAIN[col].map(mapping)
        X_TEST[col]  = X_TEST[col].map(mapping)


# Define Feature Groups
BOOLEAN_FEATURES = ["family_history_diabetes",
                    "hypertension_history",
                    "cardiovascular_history"]
X_TRAIN[BOOLEAN_FEATURES]  = X_TRAIN[BOOLEAN_FEATURES].astype(bool)
X_TEST[BOOLEAN_FEATURES]   = X_TEST[BOOLEAN_FEATURES].astype(bool)
NUMERIC_FEATURES           = X_TRAIN.select_dtypes(include=np.number).columns
CATEGORICAL_FEATURES       = X_TRAIN.select_dtypes(exclude=np.number).columns

# Inspect
print("Train data:\n"); show(X_TRAIN.head())
print("Test data:\n");  show(X_TEST.head())
print_dataset_overview(datasets)


# Preprocessing for numerical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler' , StandardScaler()) # Crucial for Linear/NN models
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot' , OneHotEncoder(handle_unknown='ignore'))
])

# Set preprocessor for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ], 
    remainder='passthrough',
)


# Logistic regression args
LR_CONFIG = {
    'solver'      :'liblinear',
    'C'           : 1.0,
    'random_state': CFG["SEED"]
}

# ElasticNet args
ELASTIC_CONFIG = {
    'solver'      :'saga', 
    'penalty'     :'elasticnet', 
    'l1_ratio'    : 0.5, # 50% Lasso, 50% Ridge
    'C'           : 0.5,  # Stronger regularization
    'max_iter'    : 2000,
    'random_state': CFG["SEED"]
}

# DNN args
DNN_CONFIG = {
    "learning_rate": 1e-4, # specified IF adam is not used (see build_dnn function)
    "epochs"       : 50, 
    "batch_size"   : 256*strategy.num_replicas_in_sync if not CFG["MIXED_PRECISION"] else 256*2*strategy.num_replicas_in_sync,
}

# Mapping to access the correct config
CONFIG_MAP = {
    "lr"     : LR_CONFIG,
    "elastic": ELASTIC_CONFIG,
    "dnn"    : DNN_CONFIG
}


# 1. Standard Logistic Regression
lr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier'  , LogisticRegression(**LR_CONFIG))
])

# 2. ElasticNet (via SGDClassifier for speed or LogReg with saga solver)
# We use 'saga' solver because it supports ElasticNet
elastic_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier'  , LogisticRegression(**ELASTIC_CONFIG))
])


def build_dnn(input_shape):
    # Everything inside this 'scope' is mirrored across GPUs
    
    with strategy.scope():
        
        USE_ADAM_OPTIM = True
        if USE_ADAM_OPTIM:
            optimizer = "adam"
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate = DNN_CONFIG["learning_rate"],
                clipnorm      = 1.0
            )
        
        model = keras.Sequential([
            # Input Layer
            layers.Input(shape=(input_shape,)),
            
            # Hidden Layer 1: Wide enough to capture patterns
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(), # Stabilizes learning
            layers.Dropout(0.3),         # Prevents overfitting
            
            # Hidden Layer 2
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output Layer: Sigmoid for Binary Classification (0 to 1 probability)
            # https://stackoverflow.com/questions/66367065/enabling-mix-precision-in-tensor-flow-model-training-decreases-the-speed-instead
            layers.Dense(1, activation='sigmoid', dtype='float32')
        ])
        
        model.compile(
            optimizer = optimizer,
            loss      = 'binary_crossentropy',
            metrics   = ['AUC']
        )
        return model

# We need to preprocess the data manually for the NN first because
# Keras doesn't plug directly into sklearn's stacking classifier easily without wrappers.
# For the Pareto principle, we will stack manually later.


if CFG["REPORT_TO_WANDB"]:
    if CFG["WANDB_API_KEY"]:
        # Direct login using the API key (non-interactive)
        wandb.login(key=CFG["WANDB_API_KEY"]) #relogin=True
        print("âœ… W&B logged in successfully.")
    else:
        print("âš ï¸� WANDB_API_KEY not found, running in offline mode.")
        wandb.login(mode="offline")


# Define estimators group with best hyperparams (if available)
estimators = ['lr', 'elastic', 'dnn']

# Initialize dictionaries to keep predictions from each model
OOF_PREDS   = dict()
TEST_PREDS  = dict()
FOLD_SCORES = dict()

# Start model loop
for estimator in estimators:
    print("="*50,f"Fitting {estimator}","="*50)

    if CFG["REPORT_TO_WANDB"]:
        # 1. Initialize a new WandB Run for the current estimator
        run = wandb.init(
            project = "Kaggle-Experiment-Tracking",                # Give your project a name
            group   = f"{CFG['COMPETITION']}-{CFG['MODEL_TYPE']}", # Group related runs (optional)
            name    = f"{estimator}_KFold_{CFG['FOLDS']}",         # Unique name for this run
            config  = {**CFG, **CONFIG_MAP[estimator]}             # Log all configuration
        )

        # Access the combined config through wandb.config if needed
        # current_config = wandb.config 
    
    skf = StratifiedKFold(n_splits=CFG["FOLDS"], shuffle=True, random_state=CFG["SEED"])

    # Define empty oof variables to fill
    oof_preds   = np.zeros(shape = len(X_TRAIN))
    test_preds  = np.zeros(shape = len(X_TEST))
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_TRAIN, y)):
        print(f"\n{'#'*10} Fold {fold+1}/{CFG['FOLDS']} {'#'*10}")

        # Define splits
        X_train, X_valid, X_test_capped = iqr_outlier_capping(
            X_TRAIN.iloc[train_idx], 
            X_TRAIN.iloc[valid_idx], X_TEST.copy(), 
            columns = X_TRAIN[NUMERIC_FEATURES]
        )
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        start = time.time()
            
        # --- 1. Train Logistic Regression ---
        if estimator == "lr":
            model = lr_model
            model.fit(X_train, y_train)
    
        # --- 2. Train ElasticNet ---
        if estimator == "elastic":
            model = elastic_model
            model.fit(X_train, y_train)
            
        # --- 3. Train Neural Network ---
        # Note: if using the Keras-WandB callback, there's no need for manual logging
        if estimator == "dnn":
        
            # We must fit the preprocessor first for NN
            preprocessor.fit(X_train)
            X_train_proc   = preprocessor.transform(X_train)
            X_valid_proc   = preprocessor.transform(X_valid)
            X_test_proc    = preprocessor.transform(X_test_capped)
            
            # Get shape for input layer
            input_shape = X_train_proc.shape[1]
    
            model = build_dnn(
                input_shape=input_shape, 
                # optimizer=DNN_CONFIG["optimizer"]
            )
            
            # Early Stopping callback
            es = callbacks.EarlyStopping(
                monitor              = 'val_AUC', 
                patience             = 5, 
                mode                 = 'max',
                restore_best_weights = True
            )
            
            model.fit(
                X_train_proc, y_train,
                validation_data = (X_valid_proc, y_valid),
                epochs          = DNN_CONFIG["epochs"],
                batch_size      = DNN_CONFIG["batch_size"],
                callbacks       = [es, WandbCallback(save_graph=False, save_model=False)] if CFG["REPORT_TO_WANDB"] else [es],
                verbose         = 0
            )
        
        # Get predictions and Predict OOF and test
        # For LR/Elastic, the model (Pipeline) expects the raw (capped) data:
        if estimator in ["lr", "elastic"]:
            preds                = model.predict_proba(X_valid)[:,1]
            oof_preds[valid_idx] = model.predict_proba(X_valid)[:,1] # Pipeline runs preprocessor.transform(x_valid)
            test_preds           += model.predict_proba(X_test_capped)[:,1]    # Pipeline runs preprocessor.transform(x_test_capped)
            fold_score           = roc_auc_score(y_valid, preds)               # Calculate AUC score
        
        # For DNN, the predictions were already calculated in the DNN block (but are currently commented out)
        if estimator == "dnn":
            preds                = model.predict(X_valid_proc).flatten()
            oof_preds[valid_idx] = model.predict(X_valid_proc).flatten()
            test_preds           += model.predict(X_test_proc).flatten() 
            fold_score           = roc_auc_score(y_valid, preds)

        print(f" Fold {fold+1}: AUC Score: {fold_score:.5f}")    
        fold_scores.append(fold_score)
        end = time.time()
        print(f"Fold {fold+1} finished in {end - start:.2f} seconds")

        if CFG["REPORT_TO_WANDB"]:
            # ğŸ“� WANDB LOGGING: Log score and time for the current fold
            wandb.log({
                f"{estimator}_fold_{fold+1}_AUC": fold_score,
                f"fold_{fold+1}_runtime_sec": end - start
            })
    
    mean_valid_score = np.mean(fold_scores); 
    print(f"Mean AUC: {mean_valid_score:.3f}")

    if CFG["REPORT_TO_WANDB"]:
        # ğŸ“� WANDB LOGGING: Log the final (mean) metrics
        wandb.log({
            f"{estimator}_mean_AUC": mean_valid_score,
            "final_mean_AUC": mean_valid_score # Log the metric for comparison
        })
    
    # Optionally: Save OOF/Test predictions as a WandB Artifact
    # data_table = wandb.Table(data=list(zip(X_TRAIN.index.tolist(), oof_preds)), columns = ["id", "oof_prediction"])
    # wandb.log({"OOF_Predictions": data_table})

    # Save model weights/pipeline
    # artifact = wandb.Artifact(f"{estimator}_model", type="model")
    # artifact.add_file(f'model_{estimator}.pkl') # Assume you save the model here
    # run.log_artifact(artifact)

    if CFG["REPORT_TO_WANDB"]:
        # ğŸ“� End the run (stop logging)
        run.finish() 
        # wandb.finish()

    test_predictions = test_preds / CFG["FOLDS"]

    # Save OOF and test predictions + fold scores by model
    OOF_PREDS[estimator]   = oof_preds
    TEST_PREDS[estimator]  = test_predictions
    FOLD_SCORES[estimator] = fold_scores



# Save predictions for later use (ensamble notebook)
pd.DataFrame(OOF_PREDS).to_csv("s5e12_oof_preds.csv")
pd.DataFrame(TEST_PREDS).to_csv("s5e12_test_preds.csv")

# Check Individual Performance
scores_df = pd.DataFrame(FOLD_SCORES)

# Boxplots
plt.figure(figsize=(12,6))
sns.boxplot(data=scores_df, palette="mako", orient='h')

# Add titles and labels
plt.title('Score Distribution Across Models', fontsize=16)
plt.xlabel('ROC-AUC', fontsize=14)
plt.ylabel('Models', fontsize=14)
plt.tight_layout()
plt.show()


def objective(trial):

    # Sample model weights and normalize
    w1 = trial.suggest_float("w1", 0, 1)
    w2 = trial.suggest_float("w2", 0, 1)
    w3 = 1 - (w1 + w2)                   # Constraint: weights must sum to 1

    # Skip invalid combinations
    if w3 < 0 or w3 > 1:
        raise optuna.exceptions.TrialPruned()

    # Weighted ensemble of out-of-fold probabilities
    ensemble_preds = (
        w1 * OOF_PREDS["lr"]      +
        w2 * OOF_PREDS["elastic"] +
        w3 * OOF_PREDS["dnn"] 
    )
    
    score = roc_auc_score(y, ensemble_preds)
    return score


if CFG["SEARCH_WEIGHTS"]:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=CFG["SEED"]))
    study.optimize(objective, n_trials = CFG["OPTUNA_N_TRIALS"]*10, timeout=3600)


# Print best params (weights)
print(study.best_params)


# Define best weights and threshold
w1 = study.best_params["w1"]
w2 = study.best_params["w2"]
w3 = 1 - (w2+w1)

# Weighted ensemble of model predictions with weights
ensemble_preds = (
    w1 * TEST_PREDS["lr"]      +
    w2 * TEST_PREDS["elastic"] +
    w3 * TEST_PREDS["dnn"] 
)

# Prepare submission df
submission_df = pd.DataFrame({
    "id": test_ids,
    "y" : ensemble_preds
})

# Display the first rows (sanity check)
display(submission_df.head())

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

