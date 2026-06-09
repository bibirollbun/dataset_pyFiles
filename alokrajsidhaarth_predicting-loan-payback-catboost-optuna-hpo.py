# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

## Import the models
import catboost
from catboost import CatBoostClassifier, Pool

## Import statistical analysis tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score

## Import for HPO
import optuna

## Import misc
import warnings
import time

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

print ('Train data size : ',df_train.shape)
print ('Test data size : ',df_test.shape)


df_train.head()


print(df_train.isnull().sum())
print(df_test.isnull().sum())


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List

# Set a clean style for the plots
sns.set_style("whitegrid")

def distribution_print(df: pd.DataFrame, max_categories: int = 30, figsize: tuple = (10, 5)):
    """
    Generates appropriate distribution plots for numerical and categorical columns
    in a Pandas DataFrame.

    - Numerical columns: Histograms and Box Plots (for high-cardinality numeric and floats).
    - Categorical columns: Count Plots (for objects, bools, and low-cardinality integers/floats).

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        max_categories (int): Maximum number of unique values a column can have
                              to be treated as categorical (default is 15).
        figsize (tuple): Base size for the generated plots (width, height).
    """
    print(f"--- Analyzing DataFrame with {len(df.columns)} Columns and {len(df)} Rows ---")

    numerical_cols: List[str] = []
    categorical_cols: List[str] = []

    # 1. Identify Column Types (Refined Logic for all Numeric Types)
    for col in df.columns:
        n_unique = df[col].nunique()

        if df[col].dtype in ['int64', 'float64']:
            # *** CRITICAL FIX: Check cardinality for BOTH int and float types ***
            if n_unique > max_categories:
                # High cardinality numeric (continuous)
                numerical_cols.append(col)
            else:
                # Low cardinality numeric (discrete, treated as categorical)
                categorical_cols.append(col)

        elif df[col].dtype in ['object', 'category', 'bool']:
            # Treat objects/categories/bools as categorical only if cardinality is reasonable
            if n_unique <= max_categories:
                categorical_cols.append(col)
            else:
                print(f"Skipping categorical column '{col}' ({n_unique} unique values) due to high cardinality.")
                
    print(f"\nFound {len(numerical_cols)} Numerical Columns and {len(categorical_cols)} Categorical Columns (<= {max_categories} unique values).")
    print ('numerical cols :',numerical_cols)
    print ('categorical cols :',categorical_cols)
    # 2. Plot Numerical Distributions (Continuous or High-Card.)
    if numerical_cols:
        print("\n--- Generating Numerical Distributions (Histograms & Box Plots) ---")
        for i, col in enumerate(numerical_cols):
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            fig.suptitle(f'Distribution of: {col}', fontsize=14, fontweight='bold')

            # Histogram (Density Plot)
            sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color='skyblue', bins=30)
            axes[0].set_title('Histogram with KDE')
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Frequency')

            # Box Plot
            sns.boxplot(x=df[col].dropna(), ax=axes[1], color='lightcoral', orient='h')
            axes[1].set_title('Box Plot (Outliers)')
            axes[1].set_xlabel(col)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
            plt.show()

    # 3. Plot Categorical Distributions (Discrete or Low-Card.)
    if categorical_cols:
        print("\n--- Generating Categorical Distributions (Count Plots) ---")
        for i, col in enumerate(categorical_cols):
            plt.figure(figsize=figsize)
            plt.title(f'Count Distribution of: {col}', fontsize=14, fontweight='bold')

            # Count Plot (Bar Chart)
            # Use 'order' to sort by count descending and plot horizontally
            order = df[col].value_counts().index
            
            # The 'y' argument handles both integer and string categorical data
            sns.countplot(y=df[col].astype(str), data=df, order=order.astype(str), palette='viridis')

            plt.xlabel('Count')
            plt.ylabel(col)
            plt.tight_layout()
            plt.show()

distribution_print(df_train.iloc[:,1:])


## Going to use Boosted trees for prediction.
## ususally these are robust to outliers


def preprocess (df: pd.DataFrame, CATEGORICAL_FEATURES = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']):

    df['debt'] = df['annual_income']*df['debt_to_income_ratio']

    # 1. Calculate the mean interest rate for each loan purpose
    mean_rate_by_purpose = df.groupby('loan_purpose')['interest_rate'].mean()
    
    # 2. Create the new composite feature by mapping the calculated means back to the original DataFrame
    df['purpose_mean_interest'] = df['loan_purpose'].map(mean_rate_by_purpose)
    
    # 3. Create a ratio/difference feature
    # This captures how far the specific loan's interest rate is from the average for that purpose
    df['rate_vs_purpose_mean_ratio'] = df['interest_rate'] / df['purpose_mean_interest']
    df['rate_vs_purpose_mean_diff'] = df['interest_rate'] - df['purpose_mean_interest']

    ## change cols of CATEGORICAL_FEATURES to oject type
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype('object')
    return df


df_train = preprocess(df_train)
df_test = preprocess(df_test)


df_train.head()


# NUMERICAL_FEATURES = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate', 'debt', 'rate_vs_purpose_mean_ratio', 'rate_vs_purpose_mean_diff']
# CATEGORICAL_FEATURES = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

NUMERICAL_FEATURES = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate', 'debt', 'rate_vs_purpose_mean_ratio']
CATEGORICAL_FEATURES = ['marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def compare_plots_num_grid(df, NUMERICAL_FEATURES):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import math

    num_features = len(NUMERICAL_FEATURES)

    # 1. Determine the grid size (e.g., 2 columns, dynamic rows)
    ncols = 2 # Number of columns in the grid
    nrows = math.ceil(num_features / ncols) # Calculate required rows
    
    # Adjust overall figure size based on the number of plots
    fig_width = 5 * ncols
    fig_height = 4 * nrows

    # 2. Create ONE figure and all necessary subplots (axes)
    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(fig_width, fig_height), 
        tight_layout=True # Automatically adjusts subplot params for tight layout
    )
    
    # Flatten the axes array if it's 2D for easy iteration
    axes = axes.flatten() 
    
    # 3. Loop through features and plot on the respective subplot
    for i, num_col in enumerate(NUMERICAL_FEATURES):
        sns.violinplot(
            x='loan_paid_back',
            y=num_col,
            data=df,
            ax=axes[i], # CRITICAL: Direct the plot to a specific subplot
            palette='Set2'
        )
        axes[i].set_title(f'{num_col} by Loan Status', fontsize=10)
        axes[i].set_xlabel("Loan Paid Back (0/1)", fontsize=8) # Set x-label only once

    # 4. Remove any unused subplots if the total number of features is odd
    if num_features < len(axes):
        for j in range(num_features, len(axes)):
            fig.delaxes(axes[j])

    # 5. Display the single figure containing all subplots
    plt.suptitle("Numerical Feature Distributions by Target Variable", y=1.02, fontsize=14, fontweight='bold')
    plt.show()

# Example Call:
# compare_plots_num_grid(df, ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate', 'debt'])


compare_plots_num_grid(df_train,NUMERICAL_FEATURES)


def compare_plots_cat_grid(df, CATEGORICAL_FEATURES):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import math

    num_features = len(CATEGORICAL_FEATURES)
    
    # 1. Determine the grid size (e.g., 2 columns, dynamic rows)
    ncols = 2 # Number of columns in the grid
    # Ensure at least one row if there are features
    nrows = math.ceil(num_features / ncols) if num_features > 0 else 0 
    
    if nrows == 0:
        print("No categorical features provided to plot.")
        return

    # Adjust overall figure size based on the number of plots
    fig_width = 8 * ncols  # Increased width for categorical labels
    fig_height = 6 * nrows
    
    # 2. Create ONE figure and all necessary subplots (axes)
    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(fig_width, fig_height), 
        tight_layout=True # Automatically adjusts subplot parameters
    )
    
    # Flatten the axes array for easy iteration (works even if nrows=1)
    axes = axes.flatten() 
    
    # 3. Loop through features and plot on the respective subplot
    for i, cat_col in enumerate(CATEGORICAL_FEATURES):
        sns.barplot( # Use barplot directly since we are using subplots (ax)
            x=cat_col, 
            y='loan_paid_back',
            data=df,
            ax=axes[i], # CRITICAL: Direct the plot to a specific subplot
            palette='viridis',
            ci=None # Recommended for clean rate comparison
        )
        axes[i].set_title(f'Rate by {cat_col}', fontsize=12)
        axes[i].set_ylabel("Loan Paid Back Rate (Mean of Target)", fontsize=10)
        axes[i].tick_params(axis='x', rotation=45) # Rotate x-labels

    # 4. Remove any unused subplots if the total number of features is odd
    if num_features < len(axes):
        for j in range(num_features, len(axes)):
            fig.delaxes(axes[j])

    # 5. Display the single figure containing all subplots
    plt.suptitle("Categorical Feature Impact on Loan Paid Back Rate", y=1.02, fontsize=16, fontweight='bold')
    plt.show()

# Example Call: 
# CATEGORICAL_FEATURES = ['loan_purpose', 'employment_type', 'home_ownership']
# compare_plots_cat_grid(df, CATEGORICAL_FEATURES)


compare_plots_cat_grid(df_train,CATEGORICAL_FEATURES)


# Suppress CatBoost warnings for clean output and Optuna warnings
warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- GLOBAL CONSTANTS ---
ID_COL = 'id'
TARGET_COL = 'loan_paid_back'
N_TRIALS = 75 # Number of Optuna trials (kept low for quick execution)

def objective(trial, X_train_hpo, y_train_hpo, X_val, y_val, cat_features):
    """Optuna objective function to MAXIMIZE AUC-ROC (by minimizing negative AUC)."""
    
    params = {
        # Fixed for the objective
        'iterations': 1000,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC', # Set CatBoost metric to AUC
        'random_seed': 42,
        'verbose': 0,
        'early_stopping_rounds': 50,
        'task_type': 'GPU', # ENABLED GPU
        'bootstrap_type': 'Bernoulli',
        'class_weights' : {0.0: 3.0, 1.0: 1.0},
        # Parameters to be optimized by Optuna (Bayesian search space)
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.15, log=True),
        'depth': trial.suggest_int('depth', 5, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 100.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0),
        'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 1, 50),
        
    }

    model = CatBoostClassifier(**params)
    
    model.fit(
        X_train_hpo, y_train_hpo,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        verbose=False # Suppress progress during HPO
    )

    # Predict probabilities on validation set (P(Target=1))
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate ROC-AUC score
    roc_auc = roc_auc_score(y_val, y_val_pred_proba)
    
    # Optuna minimizes the objective, so we return negative AUC to maximize AUC
    return roc_auc


def model_train(df: pd.DataFrame) -> CatBoostClassifier:
    """
    Splits data using stratified sampling, trains the CatBoost model, 
    and performs Hyperparameter Optimization using Optuna, optimizing for AUC.

    Args:
        df: The full pandas DataFrame for training.

    Returns:
        CatBoostClassifier: The final trained model object.
    """
    print("--- 1. Data Splitting (Stratified 3-way Split) ---")
    
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]

    # Stratified Split 1: Split into Training (80%) and Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Stratified Split 2: Split Training into HPO Train (60%) and Validation (20%)
    X_train_hpo, X_val, y_train_hpo, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    print(f"Total Samples: {len(df)}")
    print(f"  HPO Train Samples: {len(X_train_hpo)}")
    print(f"  Validation Samples: {len(X_val)}")
    print(f"  Test Samples (Final Holdout): {len(X_test)}")
    print("-" * 30)

    # --- 2. Hyperparameter Optimization (Optuna) ---
    print(f"--- 2. Hyperparameter Optimization (Optuna: {N_TRIALS} Trials, Metric: AUC) ---")
    
    # Create Optuna study to find parameters that maximize AUC
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    
    print("Starting Optuna search to maximize AUC...")
    study.optimize(
        lambda trial: objective(trial, X_train_hpo, y_train_hpo, X_val, y_val, CATEGORICAL_FEATURES), 
        n_trials=N_TRIALS, 
        show_progress_bar=True
    )
    
    best_params = study.best_params
    # Note: study.best_value is the minimized value (negative AUC), so we negate it for display
    print(f"Optuna Search Complete. Best AUC: {study.best_value:.4f}")
    print(f"Best Hyperparameters: {best_params}")
    print("-" * 30)

    # --- 3. Final Model Training ---
    print("--- 3. Final CatBoost Model Training (Using Best Parameters) ---")
    
    # Prepare final model parameters
    final_model_params = {
        'iterations': 1000, 
        'loss_function': 'Logloss', 
        'eval_metric': 'AUC', # Monitor AUC during training
        'random_seed': 42,
        'verbose': 0,
        'task_type': 'GPU', # ENABLED GPU FOR FINAL MODEL
        'bootstrap_type': 'Bernoulli',
        'class_weights' : {0.0: 3.0, 1.0: 1.0},
        **best_params # Unpack the optimized parameters
    }
    
    final_model = CatBoostClassifier(**final_model_params)
    
    # Train on the full original training set (X_train, y_train) using best params
    print("Starting final model training on the full training set (X_train)...")
    final_model.fit(
        X_train, y_train,
        cat_features=CATEGORICAL_FEATURES,
        verbose=False
    )
    print("Training complete.")

    # --- 4. Final Evaluation on UNSEEN Test Set ---
    # Predict probabilities for AUC calculation
    y_pred_proba_test = final_model.predict_proba(X_test)[:, 1]
    final_auc = roc_auc_score(y_test, y_pred_proba_test)
    
    # Predict class labels for Classification Report
    y_pred_test = final_model.predict(X_test)
    
    print("-" * 30)
    print("Model Performance on Unseen Test Set (Final Evaluation):")
    print(f"Final Test AUC-ROC Score: {final_auc:.4f}") # <<< Final Metric
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test))
    print("-" * 30)
    
    return final_model


def predict(model: CatBoostClassifier, df_test: pd.DataFrame) -> pd.Series:
    """
    Scores a new dataset using the trained model and returns the probability of default (Class 0).

    Args:
        model: The trained CatBoostClassifier object.
        df_test: A pandas DataFrame containing new data to score.

    Returns:
        pd.Series: A Series containing the predicted probability of default (P(Target=0)).
    """
    print("--- 5. Prediction on New df_test Data ---")
    
    X_new = df_test[ALL_FEATURES]

    # Predict probabilities for both classes (0 and 1)
    # The output is an array where column 0 is P(0) (Default) and column 1 is P(1) (Paid Back)
    probabilities = model.predict_proba(X_new)
    
    # Probability of Default is P(Class 0)
    prob_default = pd.Series(
        probabilities[:, 0], 
        index=df_test.id, 
        name='predicted_prob_default'
    )
    # Probability of Default is P(Class 0)
    prob_ndq = pd.Series(
        probabilities[:, 1], 
        index=df_test.id, 
        name='predicted_prob_ndq'
    )
    print(f"Successfully calculated probabilities for {len(df_test)} new samples.")
    
    return prob_ndq


start_time = time.time()
trained_model = model_train(df_train)
end_time = time.time()

print ('Time taken :',end_time - start_time)

## cpu for 1 optuna trial: Time taken : 1159.896008491516
## gpu for 1 optuna trial: ~300 sec


## save the trained model
print (trained_model)
trained_model.save_model("catboost_model.cbm")



probabilities_ndq = predict(trained_model, df_test)
print(probabilities_ndq)


submission_df = probabilities_ndq.reset_index()
submission_df = submission_df.rename(columns={'index': 'id','predicted_prob_ndq': 'loan_paid_back'})
print("\n--- Example Final Submission Data (First 5 Rows) ---")
print(submission_df.head())

submission_df.to_csv('submission_catboost_optimised2.csv', index=False)





model = trained_model


# Get the feature importance values
importance_values = model.get_feature_importance()

# Get the feature names used during training
feature_names = model.feature_names_

# Check if feature names are available (they usually are after loading a model)
if not feature_names:
    print("❌ Feature names could not be retrieved from the loaded model.")
else:
    # Combine names and importance into a pandas DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values(by='Importance', ascending=False)

    ## Print the results
    print("✨ Top 10 Feature Importances:")
    print("-" * 30)
    print(feature_importance_df.head(10).to_markdown(index=False))
    print()
    # Optional: Visualize the importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
    plt.title('CatBoost Feature Importance')
    plt.show()
    




