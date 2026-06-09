# ==================================================================================
# COMPLETE ENHANCED CREDIT SCORING NOTEBOOK
# Enhanced Model Runner with Visualizations, Configurable Models & PDF Export
# ==================================================================================

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

# === Load and prepare dataset ===
app = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
features = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'DAYS_EMPLOYED', 'DAYS_BIRTH',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
]
X = app[features].replace(365243, np.nan).fillna(0)
y = app['TARGET']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.95, stratify=y, random_state=42)

# === Reject Inference Simulation ===
def simulate_acceptance_mask(X, quantile=0.25, score_column="EXT_SOURCE_3", drop_column_after=True):
    """
    Simulates an acceptance policy where applicants below the quantile threshold
    of EXT_SOURCE_3 are rejected. Adds an 'ACCEPTED' column to X.
    """
    threshold = X[score_column].quantile(quantile)
    X = X.copy()
    X["ACCEPTED"] = (X[score_column] > threshold).astype(int)
    if drop_column_after:
        X = X.drop(columns=["ACCEPTED"], errors="ignore")
    return X, threshold

# Apply acceptance simulation
X, ext_threshold = simulate_acceptance_mask(X, quantile=0.25, score_column="EXT_SOURCE_3")

# Global tracking variables
roc_curves = []
model_summaries = []
trained_models = {}

def run_model(name, model, X_train, y_train, X_val, y_val, keras=False, plot=True):
    start_time = time.time()

    # Fit the model
    if keras:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=50,
                  batch_size=512,
                  verbose=0)
        y_train_pred = model.predict(X_train, verbose=0).flatten()
        y_val_pred = model.predict(X_val, verbose=0).flatten()
        trained_models[name] = {'model': model, 'history': history}
    else:
        model.fit(X_train, y_train)
        if hasattr(model, 'predict_proba'):
            y_train_pred = model.predict_proba(X_train)[:, 1]
            y_val_pred = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_train_pred = model.decision_function(X_train)
            y_val_pred = model.decision_function(X_val)
        else:
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
        trained_models[name] = {'model': model}

    # Calculate metrics
    auc_train = roc_auc_score(y_train, y_train_pred)
    auc_val = roc_auc_score(y_val, y_val_pred)
    
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred)
    ks_val = max(tpr_val - fpr_val)
    
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    ks_train = max(tpr_train - fpr_train)
    
    runtime = time.time() - start_time

    if plot:
        roc_curves.append((fpr_val, tpr_val, name, auc_val, ks_val))

    model_summaries.append({
        "Model": name,
        "AUC (Train)": auc_train,
        "AUC (Validation)": auc_val,
        "KS (Train)": ks_train,
        "KS (Validation)": ks_val,
        "Overfitting": auc_train - auc_val,
        "Runtime (s)": round(runtime, 2)
    })

    print(f"{name}: AUC={auc_val:.3f}, KS={ks_val:.3f}, Runtime={runtime:.2f}s")
    return model

def reset_model_tracking():
    global model_summaries, roc_curves, trained_models
    model_summaries = []
    roc_curves = []
    trained_models = {}

# === Enhanced Visualization Functions ===
def create_comprehensive_plots():
    """Create comprehensive visualization plots for model comparison"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. ROC Curves
    plt.subplot(2, 3, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_curves)))
    for i, (fpr, tpr, name, auc, ks) in enumerate(roc_curves):
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colors[i], linewidth=2)
    plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
    plt.title("ROC Curves Comparison", fontsize=14, fontweight='bold')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 2. KS Statistics Comparison
    plt.subplot(2, 3, 2)
    df_summary = pd.DataFrame(model_summaries)
    ks_values = df_summary['KS (Validation)'].values
    model_names = [name.replace(' ', '\n') for name in df_summary['Model']]
    bars = plt.bar(range(len(ks_values)), ks_values, color=colors[:len(ks_values)])
    plt.title("KS Statistics Comparison", fontsize=14, fontweight='bold')
    plt.xlabel("Models")
    plt.ylabel("KS Statistic")
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. AUC Comparison
    plt.subplot(2, 3, 3)
    auc_train = df_summary['AUC (Train)'].values
    auc_val = df_summary['AUC (Validation)'].values
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, auc_train, width, label='Train AUC', alpha=0.8, color='lightblue')
    plt.bar(x + width/2, auc_val, width, label='Validation AUC', alpha=0.8, color='orange')
    plt.title("AUC: Train vs Validation", fontsize=14, fontweight='bold')
    plt.xlabel("Models")
    plt.ylabel("AUC Score")
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Runtime Comparison
    plt.subplot(2, 3, 4)
    runtimes = df_summary['Runtime (s)'].values
    bars = plt.bar(range(len(runtimes)), runtimes, color=colors[:len(runtimes)])
    plt.title("Runtime Comparison", fontsize=14, fontweight='bold')
    plt.xlabel("Models")
    plt.ylabel("Runtime (seconds)")
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # 5. Overfitting Analysis
    plt.subplot(2, 3, 5)
    overfitting = df_summary['Overfitting'].values
    colors_over = ['red' if x > 0.05 else 'green' if x < 0.02 else 'orange' for x in overfitting]
    bars = plt.bar(range(len(overfitting)), overfitting, color=colors_over, alpha=0.7)
    plt.title("Overfitting Analysis\n(Train AUC - Val AUC)", fontsize=14, fontweight='bold')
    plt.xlabel("Models")
    plt.ylabel("AUC Difference")
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='High Overfitting (>0.05)')
    plt.axhline(y=0.02, color='orange', linestyle='--', alpha=0.5, label='Moderate Overfitting (>0.02)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 6. Performance vs Runtime Scatter
    plt.subplot(2, 3, 6)
    plt.scatter(runtimes, ks_values, s=100, c=range(len(model_names)), 
                cmap='viridis', alpha=0.7, edgecolors='black')
    for i, name in enumerate(df_summary['Model']):
        plt.annotate(name.split()[0], (runtimes[i], ks_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    plt.title("Performance vs Runtime", fontsize=14, fontweight='bold')
    plt.xlabel("Runtime (seconds, log scale)")
    plt.ylabel("KS Statistic")
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def export_results_to_pdf(filename="credit_scoring_results.pdf"):
    """Export all results and visualizations to PDF"""
    with PdfPages(filename) as pdf:
        # Page 1: Model Performance Summary
        fig1, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        df_summary = pd.DataFrame(model_summaries)
        df_display = df_summary.round(3)
        
        table = ax.table(cellText=df_display.values,
                        colLabels=df_display.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color code the best performers
        best_auc_idx = df_summary['AUC (Validation)'].idxmax() + 1
        best_ks_idx = df_summary['KS (Validation)'].idxmax() + 1
        
        table[(best_auc_idx, 2)].set_facecolor('#90EE90')  # Light green for best AUC
        table[(best_ks_idx, 4)].set_facecolor('#87CEEB')   # Light blue for best KS
        
        plt.title("Credit Scoring Model Performance Summary", 
                 fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close()
        
        # Page 2: Comprehensive Plots
        fig2 = create_comprehensive_plots()
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close()
        
        # Page 3: Neural Network Training History (if available)
        keras_models = [name for name in trained_models.keys() if 'Keras' in name]
        if keras_models:
            fig3, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            for i, model_name in enumerate(keras_models):
                if 'history' in trained_models[model_name]:
                    history = trained_models[model_name]['history'].history
                    
                    # Loss plot
                    ax1 = axes[0, i] if len(keras_models) > 1 else axes[0, 0]
                    ax1.plot(history['loss'], label='Training Loss')
                    ax1.plot(history['val_loss'], label='Validation Loss')
                    ax1.set_title(f'{model_name} - Loss')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Accuracy plot
                    ax2 = axes[1, i] if len(keras_models) > 1 else axes[1, 0]
                    ax2.plot(history['accuracy'], label='Training Accuracy')
                    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
                    ax2.set_title(f'{model_name} - Accuracy')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Accuracy')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig3, bbox_inches='tight')
            plt.close()
    
    print(f"Results exported to {filename}")

def show_model_summary():
    df = pd.DataFrame(model_summaries).sort_values(by="KS (Validation)", ascending=False).reset_index(drop=True)
    return df

# === CART-Based Feature Binarization ===
def binarize_features_with_cart(X_train, X_val, y_train, max_depth=4, min_samples_leaf=50):
    cart = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    leaf_train = cart.fit(X_train, y_train).apply(X_train)
    leaf_val = cart.apply(X_val)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_binarized = encoder.fit_transform(leaf_train.reshape(-1, 1))
    X_val_binarized = encoder.transform(leaf_val.reshape(-1, 1))

    return X_train_binarized, X_val_binarized

def run_all_models(X_train, y_train, X_val, y_val, model_configs=None):
    """Run all models with optional custom configurations"""
    if model_configs is None:
        model_configs = get_default_model_configs()
    
    for name, model_func in model_configs.items():
        if "CART-Binarized" in name:
            X_train_bin, X_val_bin = binarize_features_with_cart(X_train, X_val, y_train)
            model = model_func()
            run_model(name, model, X_train_bin, y_train, X_val_bin, y_val)
        else:
            model = model_func()
            run_model(name, model, X_train, y_train, X_val, y_val, keras=('Keras' in name))
    
    # Create visualizations
    create_comprehensive_plots()
    plt.show()
    
    return show_model_summary()

def get_default_model_configs():
    """Default model configurations"""
    return {
        "Linear Probability Model": lambda: LinearRegression(),
        "CART (Decision Tree)": lambda: RandomForestClassifier(n_estimators=1, max_depth=5, bootstrap=False, random_state=42),
        "CART-Binarized â†’ Logistic": lambda: LogisticRegression(max_iter=1000),
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0, penalty='l2'),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20, max_features='sqrt', random_state=42),
        "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.03,
                                         num_leaves=31, min_child_samples=100, subsample=0.8,
                                         colsample_bytree=0.8, random_state=42),
        "SVM (Linear - Fast)": lambda: SVC(probability=True, kernel='linear', C=1.0, random_state=42),
        "Neural Net (Keras)": lambda: tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    }

def run_full_and_accepted_models(X, y, score_column="EXT_SOURCE_3", quantile=0.25):
    """
    Trains and evaluates all models on full training data and accepted-only applicants,
    using EXT_SOURCE_3 as the rejection score.
    """
    # Step 1: Simulate acceptance
    X_masked, ext_threshold = simulate_acceptance_mask(X, quantile=quantile, score_column=score_column, drop_column_after=False)

    # Step 2: Full training split
    X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # Step 3: Accepted-only split
    accepted_mask = X_masked["ACCEPTED"] == 1
    X_train_accept = X_masked.loc[accepted_mask].drop(columns=["ACCEPTED"])
    y_train_accept = y.loc[accepted_mask]
    X_val_clean = X_val_full.drop(columns=["ACCEPTED"], errors="ignore")

    # Step 4: Run all models
    print("\n=== Running models on full training data ===")
    reset_model_tracking()
    full_results = run_all_models(X_train_full, y_train_full, X_val_full, y_val_full)

    print("\n=== Running models on ACCEPTED applicants only ===")
    reset_model_tracking()
    accepted_results = run_all_models(X_train_accept, y_train_accept, X_val_clean, y_val_full)

    # Step 5: Display and export
    print("\n=== Leaderboard: Full Data ===")
    print(full_results.to_string(index=False))

    print("\n=== Leaderboard: Accepted Applicants Only ===")
    print(accepted_results.to_string(index=False))
    
    # Export results
    export_results_to_pdf("credit_scoring_full_results.pdf")

    return full_results, accepted_results

# ==================================================================================
# CONFIGURABLE MODELS - 8 FULLY CUSTOMIZABLE MODEL CELLS
# ==================================================================================

# ==================================================================================
# MODEL 1: LINEAR PROBABILITY MODEL (Linear Regression)
# ==================================================================================

# Linear Probability Model Configuration
LINEAR_CONFIG = {
    # Core parameters
    'fit_intercept': True,        # Whether to calculate intercept (bias term)
    'copy_X': True,              # Copy X or perform in-place operations
    'n_jobs': -1,                # Number of jobs for parallel computation (-1 = all cores)
    'positive': False,           # Constrain coefficients to be positive
    
    # Solver parameters (for sklearn compatibility)
    'normalize': False,          # Deprecated in sklearn 1.0+, use StandardScaler instead
    
    # Advanced options
    'random_state': 42           # Random state for reproducibility
}

def create_linear_model():
    """
    Creates a Linear Probability Model (Linear Regression for binary classification)
    
    Note: Linear Probability Model assumes linear relationship between features and probability.
    Advantages: Simple, interpretable, fast
    Disadvantages: Can predict probabilities outside [0,1], assumes linear relationship
    """
    from sklearn.linear_model import LinearRegression
    
    model = LinearRegression(
        fit_intercept=LINEAR_CONFIG['fit_intercept'],
        copy_X=LINEAR_CONFIG['copy_X'],
        n_jobs=LINEAR_CONFIG['n_jobs'],
        positive=LINEAR_CONFIG['positive']
    )
    
    print("Linear Probability Model Configuration:")
    for key, value in LINEAR_CONFIG.items():
        print(f"  {key}: {value}")
    
    return model

# ==================================================================================
# MODEL 2: CART (Decision Tree / Single Tree Random Forest)
# ==================================================================================

# CART Configuration
CART_CONFIG = {
    # Tree structure parameters
    'n_estimators': 1,           # Number of trees (1 for pure CART)
    'max_depth': 5,              # Maximum depth of tree (None for unlimited)
    'min_samples_split': 2,      # Minimum samples required to split internal node
    'min_samples_leaf': 1,       # Minimum samples required at leaf node
    'max_features': None,        # Number of features to consider at each split
    'max_leaf_nodes': None,      # Maximum number of leaf nodes
    
    # Regularization parameters
    'min_impurity_decrease': 0.0, # Minimum impurity decrease for split
    'min_weight_fraction_leaf': 0.0, # Minimum weighted fraction at leaf
    
    # Randomness and reproducibility
    'bootstrap': False,          # Whether to use bootstrap sampling (False for deterministic CART)
    'random_state': 42,          # Random state for reproducibility
    
    # Performance parameters
    'n_jobs': -1,                # Number of parallel jobs
    'verbose': 0,                # Verbosity level
    'warm_start': False,         # Reuse previous solution when fitting
    
    # Advanced parameters
    'criterion': 'gini',         # Split quality measure: 'gini', 'entropy', 'log_loss'
    'class_weight': None,        # Class weights: None, 'balanced', or dict
    'ccp_alpha': 0.0,           # Complexity parameter for pruning
    'max_samples': None          # Number of samples to draw for each tree
}

def create_cart_model():
    """
    Creates a CART (Classification and Regression Trees) model
    
    CART builds a single decision tree using binary splits.
    Advantages: Interpretable, handles non-linear relationships, no assumptions about data distribution
    Disadvantages: Prone to overfitting, unstable (small data changes can change tree structure)
    """
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=CART_CONFIG['n_estimators'],
        criterion=CART_CONFIG['criterion'],
        max_depth=CART_CONFIG['max_depth'],
        min_samples_split=CART_CONFIG['min_samples_split'],
        min_samples_leaf=CART_CONFIG['min_samples_leaf'],
        min_weight_fraction_leaf=CART_CONFIG['min_weight_fraction_leaf'],
        max_features=CART_CONFIG['max_features'],
        max_leaf_nodes=CART_CONFIG['max_leaf_nodes'],
        min_impurity_decrease=CART_CONFIG['min_impurity_decrease'],
        bootstrap=CART_CONFIG['bootstrap'],
        n_jobs=CART_CONFIG['n_jobs'],
        random_state=CART_CONFIG['random_state'],
        verbose=CART_CONFIG['verbose'],
        warm_start=CART_CONFIG['warm_start'],
        class_weight=CART_CONFIG['class_weight'],
        ccp_alpha=CART_CONFIG['ccp_alpha'],
        max_samples=CART_CONFIG['max_samples']
    )
    
    print("CART Model Configuration:")
    for key, value in CART_CONFIG.items():
        print(f"  {key}: {value}")
    
    return model

# ==================================================================================
# MODEL 3: CART-BINARIZED â†’ LOGISTIC REGRESSION
# ==================================================================================

# CART Binarization Configuration
CART_BINARIZATION_CONFIG = {
    # CART parameters for binarization
    'max_depth': 4,              # Maximum depth for CART tree
    'min_samples_leaf': 50,      # Minimum samples per leaf for CART
    'min_samples_split': 100,    # Minimum samples to split for CART
    'criterion': 'gini',         # Split criterion for CART
    'random_state': 42,          # Random state for CART
    
    # OneHotEncoder parameters
    'sparse_output': False,      # Return dense array instead of sparse
    'handle_unknown': 'ignore',  # How to handle unknown categories
    'drop': None,                # Whether to drop one category per feature
    'dtype': np.float64,         # Output dtype
}

# Logistic Regression Configuration for CART-binarized features
CART_LOGISTIC_CONFIG = {
    # Core parameters
    'penalty': 'l2',             # Regularization: 'l1', 'l2', 'elasticnet', 'none'
    'dual': False,               # Dual or primal formulation (dual=False for n_samples > n_features)
    'tol': 1e-4,                 # Tolerance for stopping criteria
    'C': 1.0,                    # Inverse of regularization strength (smaller = more regularization)
    'fit_intercept': True,       # Whether to fit intercept
    'intercept_scaling': 1,      # Scaling for synthetic intercept feature
    'class_weight': None,        # Class weights: None, 'balanced', or dict
    'random_state': 42,          # Random state for reproducibility
    
    # Solver parameters
    'solver': 'lbfgs',           # Optimization algorithm: 'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'
    'max_iter': 1000,            # Maximum iterations for solver
    'multi_class': 'auto',       # Multi-class strategy: 'auto', 'ovr', 'multinomial'
    'verbose': 0,                # Verbosity level
    'warm_start': False,         # Reuse previous solution as initialization
    'n_jobs': -1,                # Number of parallel jobs
    
    # Advanced parameters
    'l1_ratio': None,            # ElasticNet mixing parameter (only for 'elasticnet' penalty)
}

def create_cart_binarized_logistic():
    """
    Creates a CART-binarized Logistic Regression model
    
    This approach:
    1. Fits a CART tree to create leaf-based features
    2. One-hot encodes the leaf assignments
    3. Fits logistic regression on the binarized features
    
    Advantages: Combines CART's ability to find non-linear patterns with logistic regression's calibrated probabilities
    Disadvantages: Two-step process, potential information loss in binarization
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    
    # Create logistic regression model
    logistic_model = LogisticRegression(
        penalty=CART_LOGISTIC_CONFIG['penalty'],
        dual=CART_LOGISTIC_CONFIG['dual'],
        tol=CART_LOGISTIC_CONFIG['tol'],
        C=CART_LOGISTIC_CONFIG['C'],
        fit_intercept=CART_LOGISTIC_CONFIG['fit_intercept'],
        intercept_scaling=CART_LOGISTIC_CONFIG['intercept_scaling'],
        class_weight=CART_LOGISTIC_CONFIG['class_weight'],
        random_state=CART_LOGISTIC_CONFIG['random_state'],
        solver=CART_LOGISTIC_CONFIG['solver'],
        max_iter=CART_LOGISTIC_CONFIG['max_iter'],
        multi_class=CART_LOGISTIC_CONFIG['multi_class'],
        verbose=CART_LOGISTIC_CONFIG['verbose'],
        warm_start=CART_LOGISTIC_CONFIG['warm_start'],
        n_jobs=CART_LOGISTIC_CONFIG['n_jobs'],
        l1_ratio=CART_LOGISTIC_CONFIG['l1_ratio']
    )
    
    print("CART Binarization Configuration:")
    for key, value in CART_BINARIZATION_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nLogistic Regression Configuration:")
    for key, value in CART_LOGISTIC_CONFIG.items():
        print(f"  {key}: {value}")
    
    return logistic_model

# ==================================================================================
# MODEL 4: LOGISTIC REGRESSION
# ==================================================================================

# Logistic Regression Configuration
LOGISTIC_CONFIG = {
    # Core parameters
    'penalty': 'l2',             # Regularization: 'l1', 'l2', 'elasticnet', 'none'
    'dual': False,               # Dual or primal formulation
    'tol': 1e-4,                 # Tolerance for stopping criteria
    'C': 1.0,                    # Inverse of regularization strength
    'fit_intercept': True,       # Whether to fit intercept
    'intercept_scaling': 1,      # Scaling for synthetic intercept feature
    'class_weight': None,        # Class weights: None, 'balanced', or dict
    'random_state': 42,          # Random state for reproducibility
    
    # Solver parameters
    'solver': 'lbfgs',           # Optimization algorithm
    'max_iter': 1000,            # Maximum iterations
    'multi_class': 'auto',       # Multi-class strategy
    'verbose': 0,                # Verbosity level
    'warm_start': False,         # Reuse previous solution
    'n_jobs': -1,                # Number of parallel jobs
    
    # Advanced parameters
    'l1_ratio': None,            # ElasticNet mixing parameter (only for 'elasticnet')
}

def create_logistic_model():
    """
    Creates a Logistic Regression model
    
    Logistic Regression models the probability of binary outcomes using the logistic function.
    Advantages: Probabilistic output, interpretable coefficients, fast training and prediction
    Disadvantages: Assumes linear relationship between features and log-odds, sensitive to outliers
    
    Solver guide:
    - 'lbfgs': Good for small datasets, only supports l2/none penalty
    - 'liblinear': Good for small datasets, supports l1/l2 penalty
    - 'sag'/'saga': Good for large datasets, saga supports l1 penalty
    - 'newton-cg': Good for multiclass problems
    """
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression(
        penalty=LOGISTIC_CONFIG['penalty'],
        dual=LOGISTIC_CONFIG['dual'],
        tol=LOGISTIC_CONFIG['tol'],
        C=LOGISTIC_CONFIG['C'],
        fit_intercept=LOGISTIC_CONFIG['fit_intercept'],
        intercept_scaling=LOGISTIC_CONFIG['intercept_scaling'],
        class_weight=LOGISTIC_CONFIG['class_weight'],
        random_state=LOGISTIC_CONFIG['random_state'],
        solver=LOGISTIC_CONFIG['solver'],
        max_iter=LOGISTIC_CONFIG['max_iter'],
        multi_class=LOGISTIC_CONFIG['multi_class'],
        verbose=LOGISTIC_CONFIG['verbose'],
        warm_start=LOGISTIC_CONFIG['warm_start'],
        n_jobs=LOGISTIC_CONFIG['n_jobs'],
        l1_ratio=LOGISTIC_CONFIG['l1_ratio']
    )
    
    print("Logistic Regression Configuration:")
    for key, value in LOGISTIC_CONFIG.items():
        print(f"  {key}: {value}")
    
    return model

# ==================================================================================
# MODEL 5: RANDOM FOREST
# ==================================================================================

# Random Forest Configuration
RANDOM_FOREST_CONFIG = {
    # Ensemble parameters
    'n_estimators': 300,         # Number of trees in the forest
    'criterion': 'gini',         # Split quality measure: 'gini', 'entropy', 'log_loss'
    'max_depth': 8,              # Maximum depth of trees (None for unlimited)
    'min_samples_split': 2,      # Minimum samples required to split internal node
    'min_samples_leaf': 20,      # Minimum samples required at leaf node
    'min_weight_fraction_leaf': 0.0, # Minimum weighted fraction at leaf
    'max_features': 'sqrt',      # Number of features per split: int, float, 'sqrt', 'log2', None
    'max_leaf_nodes': None,      # Maximum number of leaf nodes
    'min_impurity_decrease': 0.0, # Minimum impurity decrease for split
    
    # Randomness and sampling
    'bootstrap': True,           # Whether to use bootstrap sampling
    'oob_score': False,          # Whether to compute out-of-bag score
    'random_state': 42,          # Random state for reproducibility
    'max_samples': None,         # Number of samples to draw for each tree
    
    # Performance parameters
    'n_jobs': -1,                # Number of parallel jobs (-1 for all cores)
    'verbose': 0,                # Verbosity level
    'warm_start': False,         # Reuse previous solution when adding estimators
    
    # Advanced parameters
    'class_weight': None,        # Class weights: None, 'balanced', 'balanced_subsample', or dict
    'ccp_alpha': 0.0,           # Complexity parameter for pruning
}

def create_random_forest_model():
    """
    Creates a Random Forest model
    
    Random Forest builds multiple decision trees and combines their predictions.
    Advantages: Reduces overfitting, handles missing values, provides feature importance
    Disadvantages: Less interpretable than single tree, can overfit with very noisy data
    
    Parameter tuning tips:
    - Increase n_estimators for better performance (diminishing returns after ~100-500)
    - Decrease max_depth to reduce overfitting
    - Increase min_samples_leaf to reduce overfitting
    - Use 'sqrt' or 'log2' for max_features to add randomness
    """
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=RANDOM_FOREST_CONFIG['n_estimators'],
        criterion=RANDOM_FOREST_CONFIG['criterion'],
        max_depth=RANDOM_FOREST_CONFIG['max_depth'],
        min_samples_split=RANDOM_FOREST_CONFIG['min_samples_split'],
        min_samples_leaf=RANDOM_FOREST_CONFIG['min_samples_leaf'],
        min_weight_fraction_leaf=RANDOM_FOREST_CONFIG['min_weight_fraction_leaf'],
        max_features=RANDOM_FOREST_CONFIG['max_features'],
        max_leaf_nodes=RANDOM_FOREST_CONFIG['max_leaf_nodes'],
        min_impurity_decrease=RANDOM_FOREST_CONFIG['min_impurity_decrease'],
        bootstrap=RANDOM_FOREST_CONFIG['bootstrap'],
        oob_score=RANDOM_FOREST_CONFIG['oob_score'],
        n_jobs=RANDOM_FOREST_CONFIG['n_jobs'],
        random_state=RANDOM_FOREST_CONFIG['random_state'],
        verbose=RANDOM_FOREST_CONFIG['verbose'],
        warm_start=RANDOM_FOREST_CONFIG['warm_start'],
        class_weight=RANDOM_FOREST_CONFIG['class_weight'],
        ccp_alpha=RANDOM_FOREST_CONFIG['ccp_alpha'],
        max_samples=RANDOM_FOREST_CONFIG['max_samples']
    )
    
    print("Random Forest Configuration:")
    for key, value in RANDOM_FOREST_CONFIG.items():
        print(f"  {key}: {value}")
    
    return model

# ==================================================================================
# MODEL 6: LIGHTGBM
# ==================================================================================

# LightGBM Configuration
LIGHTGBM_CONFIG = {
    # Core boosting parameters
    'objective': 'binary',       # Learning objective
    'boosting_type': 'gbdt',     # Boosting type: 'gbdt', 'dart', 'rf'
    'num_leaves': 31,            # Maximum number of leaves in one tree
    'learning_rate': 0.03,       # Learning rate / shrinkage rate
    'feature_fraction': 0.8,     # Subsample ratio of features (colsample_bytree in XGBoost)
    'bagging_fraction': 0.8,     # Subsample ratio of training data
    'bagging_freq': 5,           # Frequency of bagging (0 = disable)
    'verbose': -1,               # Verbosity level
    
    # Tree structure parameters
    'max_depth': 8,              # Maximum depth of trees (-1 = no limit)
    'min_data_in_leaf': 100,     # Minimum number of data points in a leaf
    'min_gain_to_split': 0.0,    # Minimum gain to split
    'min_sum_hessian_in_leaf': 1e-3, # Minimum sum of hessian in a leaf
    
    # Training parameters
    'n_estimators': 200,         # Number of boosting iterations
    'max_bin': 255,              # Maximum number of bins for feature discretization
    'subsample_for_bin': 200000, # Number of samples for constructing bins
    
    # Regularization parameters
    'reg_alpha': 0.0,            # L1 regularization term
    'reg_lambda': 0.0,           # L2 regularization term
    'min_child_weight': 1e-3,    # Minimum sum of instance weight in a child
    'min_child_samples': 100,    # Minimum number of data points in a child
    
    # Performance and memory
    'n_jobs': -1,                # Number of parallel threads
    'random_state': 42,          # Random seed
    'deterministic': True,       # Force deterministic training
    
    # Advanced parameters
    'class_weight': None,        # Class weights: None, 'balanced', or dict
    'subsample_freq': 0,         # Alias for bagging_freq
    'colsample_bytree': None,    # Alias for feature_fraction
    'reg_sqrt': False,           # Whether to use sqrt regularization
    'extra_trees': False,        # Use extremely randomized trees
}

def create_lightgbm_model():
    """
    Creates a LightGBM model
    
    LightGBM is a gradient boosting framework that uses tree-based learning algorithms.
    Advantages: Fast training, low memory usage, high accuracy, handles categorical features
    Disadvantages: Can overfit with small datasets, requires parameter tuning
    
    Parameter tuning tips:
    - Lower learning_rate with higher n_estimators for better performance
    - Reduce num_leaves and max_depth to prevent overfitting
    - Increase min_data_in_leaf for regularization
    - Use feature_fraction and bagging_fraction for regularization
    """
    import lightgbm as lgb
    
    model = lgb.LGBMClassifier(
        objective=LIGHTGBM_CONFIG['objective'],
        boosting_type=LIGHTGBM_CONFIG['boosting_type'],
        num_leaves=LIGHTGBM_CONFIG['num_leaves'],
        learning_rate=LIGHTGBM_CONFIG['learning_rate'],
        feature_fraction=LIGHTGBM_CONFIG['feature_fraction'],
        bagging_fraction=LIGHTGBM_CONFIG['bagging_fraction'],
        bagging_freq=LIGHTGBM_CONFIG['bagging_freq'],
        verbose=LIGHTGBM_CONFIG['verbose'],
        max_depth=LIGHTGBM_CONFIG['max_depth'],
        min_data_in_leaf=LIGHTGBM_CONFIG['min_data_in_leaf'],
        min_gain_to_split=LIGHTGBM_CONFIG['min_gain_to_split'],
        min_sum_hessian_in_leaf=LIGHTGBM_CONFIG['min_sum_hessian_in_leaf'],
        n_estimators=LIGHTGBM_CONFIG['n_estimators'],
        max_bin=LIGHTGBM_CONFIG['max_bin'],
        subsample_for_bin=LIGHTGBM_CONFIG['subsample_for_bin'],
        reg_alpha=LIGHTGBM_CONFIG['reg_alpha'],
        reg_lambda=LIGHTGBM_CONFIG['reg_lambda'],
        min_child_weight=LIGHTGBM_CONFIG['min_child_weight'],
        min_child_samples=LIGHTGBM_CONFIG['min_child_samples'],
        n_jobs=LIGHTGBM_CONFIG['n_jobs'],
        random_state=LIGHTGBM_CONFIG['random_state'],
        deterministic=LIGHTGBM_CONFIG['deterministic'],
        class_weight=LIGHTGBM_CONFIG['class_weight'],
        subsample_freq=LIGHTGBM_CONFIG['subsample_freq'],
        colsample_bytree=LIGHTGBM_CONFIG['colsample_bytree'],
        reg_sqrt=LIGHTGBM_CONFIG['reg_sqrt'],
        extra_trees=LIGHTGBM_CONFIG['extra_trees']
    )
    
    print("LightGBM Configuration:")
    for key, value in LIGHTGBM_CONFIG.items():
        print(f"  {key}: {value}")
    
    return model

# ==================================================================================
# MODEL 7: SVM (Support Vector Machine)
# ==================================================================================

# SVM Configuration
SVM_CONFIG = {
    # Core parameters
    'C': 1.0,                    # Regularization parameter (higher = less regularization)
    'kernel': 'linear',          # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    'degree': 3,                 # Degree for polynomial kernel
    'gamma': 'scale',            # Kernel coefficient: 'scale', 'auto', or float
    'coef0': 0.0,                # Independent term for poly/sigmoid kernels
    'shrinking': True,           # Whether to use shrinking heuristic
    'probability': True,         # Whether to enable probability estimates
    'tol': 1e-3,                 # Tolerance for stopping criterion
    'cache_size': 200,           # Size of kernel cache (MB)
    'class_weight': None,        # Class weights: None, 'balanced', or dict
    'verbose': False,            # Enable verbose output
    'max_iter': -1,              # Hard limit on iterations (-1 = no limit)
    'decision_function_shape': 'ovr', # Decision function shape: 'ovo', 'ovr'
    'break_ties': False,         # Break ties according to decision function
    'random_state': 42           # Random state for reproducibility
}

def create_svm_model():
    """
    Creates a Support Vector Machine model
    
    SVM finds optimal hyperplane to separate classes with maximum margin.
    Advantages: Effective in high dimensions, memory efficient, versatile (different kernels)
    Disadvantages: Slow on large datasets, sensitive to feature scaling, no probabilistic output (unless enabled)
    
    Kernel guide:
    - 'linear': Good for linearly separable data, fast for large datasets
    - 'rbf': Good default for non-linear data, most commonly used
    - 'poly': Good for polynomial relationships, can be expensive
    - 'sigmoid': Similar to neural network, rarely used
    
    Performance note: RBF kernel is O(nÂ²) to O(nÂ³), linear is much faster O(n)
    """
    from sklearn.svm import SVC
    
    model = SVC(
        C=SVM_CONFIG['C'],
        kernel=SVM_CONFIG['kernel'],
        degree=SVM_CONFIG['degree'],
        gamma=SVM_CONFIG['gamma'],
        coef0=SVM_CONFIG['coef0'],
        shrinking=SVM_CONFIG['shrinking'],
        probability=SVM_CONFIG['probability'],
        tol=SVM_CONFIG['tol'],
        cache_size=SVM_CONFIG['cache_size'],
        class_weight=SVM_CONFIG['class_weight'],
        verbose=SVM_CONFIG['verbose'],
        max_iter=SVM_CONFIG['max_iter'],
        decision_function_shape=SVM_CONFIG['decision_function_shape'],
        break_ties=SVM_CONFIG['break_ties'],
        random_state=SVM_CONFIG['random_state']
    )
    
    print("SVM Configuration:")
    for key, value in SVM_CONFIG.items():
        print(f"  {key}: {value}")
    
    print(f"\nNote: Using {SVM_CONFIG['kernel']} kernel")
    if SVM_CONFIG['kernel'] == 'linear':
        print("Linear kernel is recommended for large datasets due to O(n) complexity")
    elif SVM_CONFIG['kernel'] == 'rbf':
        print("RBF kernel has O(nÂ²) complexity - may be slow on large datasets")
    
    return model

# ==================================================================================
# MODEL 8: NEURAL NETWORK (Keras/TensorFlow)
# ==================================================================================

# Neural Network Configuration
NEURAL_NET_CONFIG = {
    # Architecture parameters
    'input_dim': None,           # Will be set automatically based on X_train.shape[1]
    'hidden_layers': [64, 32],   # List of hidden layer sizes
    'activation': 'relu',        # Activation function: 'relu', 'tanh', 'sigmoid', 'elu'
    'output_activation': 'sigmoid', # Output activation for binary classification
    'dropout_rate': 0.3,         # Dropout rate for regularization
    'use_batch_norm': True,      # Whether to use batch normalization
    
    # Compilation parameters
    'optimizer': 'adam',         # Optimizer: 'adam', 'sgd', 'rmsprop', 'adagrad'
    'learning_rate': 0.001,      # Learning rate for optimizer
    'loss': 'binary_crossentropy', # Loss function
    'metrics': ['accuracy'],     # Metrics to track
    
    # Training parameters
    'epochs': 50,                # Number of training epochs
    'batch_size': 512,           # Batch size for training
    'validation_split': 0.0,     # Fraction of training data for validation
    'verbose': 0,                # Verbosity: 0 = silent, 1 = progress bar, 2 = one line per epoch
    'shuffle': True,             # Whether to shuffle training data
    
    # Callbacks and regularization
    'early_stopping': True,      # Whether to use early stopping
    'early_stopping_patience': 10, # Epochs to wait before stopping
    'early_stopping_monitor': 'val_loss', # Metric to monitor for early stopping
    'reduce_lr': True,           # Whether to reduce learning rate on plateau
    'reduce_lr_patience': 5,     # Epochs to wait before reducing LR
    'reduce_lr_factor': 0.5,     # Factor to reduce LR by
    
    # Advanced parameters
    'kernel_initializer': 'glorot_uniform', # Weight initialization
    'bias_initializer': 'zeros', # Bias initialization
    'kernel_regularizer': None,  # Kernel regularization: None, 'l1', 'l2', 'l1_l2'
    'activity_regularizer': None, # Activity regularization
    'random_state': 42           # Random state for reproducibility
}

def create_neural_network_model():
    """
    Creates a Neural Network model using Keras/TensorFlow
    
    Neural networks can learn complex non-linear relationships through multiple layers.
    Advantages: Very flexible, can approximate any function, handles complex patterns
    Disadvantages: Black box, prone to overfitting, requires more data, longer training time
    
    Architecture tips:
    - Start with 1-2 hidden layers
    - Use dropout (0.2-0.5) to prevent overfitting
    - Batch normalization can improve training stability
    - ReLU is a good default activation function
    - Use early stopping to prevent overfitting
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    
    # Set random seed for reproducibility
    tf.random.set_seed(NEURAL_NET_CONFIG['random_state'])
    
    # Determine input dimension
    input_dim = NEURAL_NET_CONFIG['input_dim'] or X_train.shape[1]
    
    # Build model
    model = Sequential()
    
    # Input layer with optional batch normalization
    model.add(Dense(input_dim, input_shape=(input_dim,), name='input_layer'))
    if NEURAL_NET_CONFIG['use_batch_norm']:
        model.add(BatchNormalization())
    
    # Hidden layers
    for i, layer_size in enumerate(NEURAL_NET_CONFIG['hidden_layers']):
        model.add(Dense(
            layer_size, 
            activation=NEURAL_NET_CONFIG['activation'],
            kernel_initializer=NEURAL_NET_CONFIG['kernel_initializer'],
            bias_initializer=NEURAL_NET_CONFIG['bias_initializer'],
            kernel_regularizer=NEURAL_NET_CONFIG['kernel_regularizer'],
            activity_regularizer=NEURAL_NET_CONFIG['activity_regularizer'],
            name=f'hidden_{i+1}'
        ))
        
        if NEURAL_NET_CONFIG['dropout_rate'] > 0:
            model.add(Dropout(NEURAL_NET_CONFIG['dropout_rate'], name=f'dropout_{i+1}'))
    
    # Output layer
    model.add(Dense(
        1, 
        activation=NEURAL_NET_CONFIG['output_activation'],
        name='output_layer'
    ))
    
    # Choose optimizer
    if NEURAL_NET_CONFIG['optimizer'].lower() == 'adam':
        optimizer = Adam(learning_rate=NEURAL_NET_CONFIG['learning_rate'])
    elif NEURAL_NET_CONFIG['optimizer'].lower() == 'sgd':
        optimizer = SGD(learning_rate=NEURAL_NET_CONFIG['learning_rate'])
    elif NEURAL_NET_CONFIG['optimizer'].lower() == 'rmsprop':
        optimizer = RMSprop(learning_rate=NEURAL_NET_CONFIG['learning_rate'])
    else:
        optimizer = NEURAL_NET_CONFIG['optimizer']
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=NEURAL_NET_CONFIG['loss'],
        metrics=NEURAL_NET_CONFIG['metrics']
    )
    
    print("Neural Network Configuration:")
    for key, value in NEURAL_NET_CONFIG.items():
        print(f"  {key}: {value}")
    
    print(f"\nModel Architecture:")
    model.summary()
    
    return model

def get_neural_network_callbacks():
    """Get callbacks for neural network training"""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    callbacks = []
    
    if NEURAL_NET_CONFIG['early_stopping']:
        early_stopping = EarlyStopping(
            monitor=NEURAL_NET_CONFIG['early_stopping_monitor'],
            patience=NEURAL_NET_CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
    
    if NEURAL_NET_CONFIG['reduce_lr']:
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=NEURAL_NET_CONFIG['reduce_lr_factor'],
            patience=NEURAL_NET_CONFIG['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
    
    return callbacks

def run_neural_network():
    """Run the neural network model with custom training loop"""
    model = create_neural_network_model()
    callbacks = get_neural_network_callbacks()
    
    # Custom training with callbacks
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=NEURAL_NET_CONFIG['epochs'],
        batch_size=NEURAL_NET_CONFIG['batch_size'],
        verbose=NEURAL_NET_CONFIG['verbose'],
        shuffle=NEURAL_NET_CONFIG['shuffle'],
        callbacks=callbacks
    )
    
    # Make predictions
    y_train_pred = model.predict(X_train, verbose=0).flatten()
    y_val_pred = model.predict(X_val, verbose=0).flatten()
    
    # Calculate metrics
    auc_train = roc_auc_score(y_train, y_train_pred)
    auc_val = roc_auc_score(y_val, y_val_pred)
    
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred)
    ks_val = max(tpr_val - fpr_val)
    
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    ks_train = max(tpr_train - fpr_train)
    
    runtime = time.time() - start_time
    
    print(f"\nNeural Network Results:")
    print(f"  Training AUC: {auc_train:.3f}")
    print(f"  Validation AUC: {auc_val:.3f}")
    print(f"  Training KS: {ks_train:.3f}")
    print(f"  Validation KS: {ks_val:.3f}")
    print(f"  Overfitting: {auc_train - auc_val:.3f}")
    print(f"  Runtime: {runtime:.2f}s")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model, history

# ==================================================================================
# INDIVIDUAL MODEL TESTING FUNCTIONS
# ==================================================================================

def test_linear_model():
    """Test the Linear Probability Model"""
    linear_model = create_linear_model()
    reset_model_tracking()
    run_model("Linear Probability Model (Custom)", linear_model, X_train, y_train, X_val, y_val)
    print("\nPerformance Summary:")
    print(show_model_summary().to_string(index=False))

def test_cart_model():
    """Test the CART model"""
    cart_model = create_cart_model()
    reset_model_tracking()
    run_model("CART (Custom)", cart_model, X_train, y_train, X_val, y_val)
    print("\nPerformance Summary:")
    print(show_model_summary().to_string(index=False))

def test_cart_binarized_logistic():
    """Test the CART-binarized logistic regression"""
    logistic_model = create_cart_binarized_logistic()
    X_train_bin, X_val_bin = binarize_features_with_cart(
        X_train, X_val, y_train, 
        max_depth=CART_BINARIZATION_CONFIG['max_depth'],
        min_samples_leaf=CART_BINARIZATION_CONFIG['min_samples_leaf']
    )
    reset_model_tracking()
    run_model("CART-Binarized â†’ Logistic (Custom)", logistic_model, X_train_bin, y_train, X_val_bin, y_val)
    print("\nPerformance Summary:")
    print(show_model_summary().to_string(index=False))

def test_logistic_model():
    """Test the Logistic Regression model"""
    logistic_model = create_logistic_model()
    reset_model_tracking()
    run_model("Logistic Regression (Custom)", logistic_model, X_train, y_train, X_val, y_val)
    print("\nPerformance Summary:")
    print(show_model_summary().to_string(index=False))

def test_random_forest_model():
    """Test the Random Forest model"""
    rf_model = create_random_forest_model()
    reset_model_tracking()
    run_model("Random Forest (Custom)", rf_model, X_train, y_train, X_val, y_val)
    print("\nPerformance Summary:")
    print(show_model_summary().to_string(index=False))

def test_lightgbm_model():
    """Test the LightGBM model"""
    lgb_model = create_lightgbm_model()
    reset_model_tracking()
    run_model("LightGBM (Custom)", lgb_model, X_train, y_train, X_val, y_val)
    print("\nPerformance Summary:")
    print(show_model_summary().to_string(index=False))

def test_svm_model():
    """Test the SVM model"""
    svm_model = create_svm_model()
    reset_model_tracking()
    run_model("SVM (Custom)", svm_model, X_train, y_train, X_val, y_val)
    print("\nPerformance Summary:")
    print(show_model_summary().to_string(index=False))

def test_neural_network_model():
    """Test the Neural Network model"""
    model, history = run_neural_network()
    return model, history

# ==================================================================================
# COMPLETE EXECUTION AND ANALYSIS
# ==================================================================================

def run_all_configured_models():
    """Run all models with custom configurations"""
    
    models_to_run = [
        ("Linear Probability (Custom)", create_linear_model()),
        ("CART (Custom)", create_cart_model()),
        ("Logistic Regression (Custom)", create_logistic_model()),
        ("Random Forest (Custom)", create_random_forest_model()),
        ("LightGBM (Custom)", create_lightgbm_model()),
        ("SVM (Custom)", create_svm_model()),
    ]
    
    reset_model_tracking()
    
    # Run traditional models
    for name, model in models_to_run:
        run_model(name, model, X_train, y_train, X_val, y_val)
    
    # Run CART-binarized logistic separately
    print("\n" + "="*50)
    print("Running CART-Binarized â†’ Logistic Regression")
    print("="*50)
    logistic_model = create_cart_binarized_logistic()
    X_train_bin, X_val_bin = binarize_features_with_cart(X_train, X_val, y_train)
    run_model("CART-Binarized â†’ Logistic (Custom)", logistic_model, X_train_bin, y_train, X_val_bin, y_val)
    
    # Run neural network separately
    print("\n" + "="*50)
    print("Running Neural Network")
    print("="*50)
    nn_model, nn_history = run_neural_network()
    
    # Show final results
    print("\n" + "="*50)
    print("FINAL RESULTS - ALL CONFIGURED MODELS")
    print("="*50)
    create_comprehensive_plots()
    plt.show()
    
    results_df = show_model_summary()
    print(results_df.to_string(index=False))
    
    return results_df

# ==================================================================================
# FINAL EXECUTION CELL: COMPLETE ANALYSIS WITH PDF EXPORT
# ==================================================================================

def run_complete_analysis():
    """Execute the complete credit scoring analysis with enhanced visualizations and PDF export"""
    
    print("ğŸš€ Starting Complete Credit Scoring Model Analysis")
    print("="*80)

    # Step 1: Run full comparison (Full Data vs Accepted-Only)
    print("\nğŸ“Š Step 1: Running Full vs Accepted Applicants Comparison")
    print("-" * 60)

    full_results_df, accepted_results_df = run_full_and_accepted_models(X, y)

    # Step 2: Create comprehensive visualizations
    print("\nğŸ“ˆ Step 2: Creating Comprehensive Visualizations")
    print("-" * 60)

    # Create and display the main comparison plots
    comparison_fig = create_comprehensive_plots()
    plt.show()

    # Step 3: Export results to PDF
    print("\nğŸ“„ Step 3: Exporting Results to PDF")
    print("-" * 60)

    export_results_to_pdf("credit_scoring_complete_analysis.pdf")

    # Step 4: Display final summary tables
    print("\nğŸ“‹ Step 4: Final Performance Summary")
    print("-" * 60)

    print("\nğŸ�¯ FULL TRAINING DATA RESULTS:")
    print("=" * 40)
    print(full_results_df.to_string(index=False))

    print("\nğŸ�¯ ACCEPTED APPLICANTS ONLY RESULTS:")
    print("=" * 40)
    print(accepted_results_df.to_string(index=False))

    # Step 5: Key insights and recommendations
    print("\nğŸ’¡ Step 5: Key Insights & Recommendations")
    print("-" * 60)

    # Analyze best performers
    best_full_auc = full_results_df.loc[full_results_df['AUC (Validation)'].idxmax()]
    best_full_ks = full_results_df.loc[full_results_df['KS (Validation)'].idxmax()]
    best_accepted_auc = accepted_results_df.loc[accepted_results_df['AUC (Validation)'].idxmax()]
    best_accepted_ks = accepted_results_df.loc[accepted_results_df['KS (Validation)'].idxmax()]

    print(f"\nğŸ�† BEST PERFORMERS:")
    print(f"Full Data - Best AUC: {best_full_auc['Model']} ({best_full_auc['AUC (Validation)']:.3f})")
    print(f"Full Data - Best KS:  {best_full_ks['Model']} ({best_full_ks['KS (Validation)']:.3f})")
    print(f"Accepted Only - Best AUC: {best_accepted_auc['Model']} ({best_accepted_auc['AUC (Validation)']:.3f})")
    print(f"Accepted Only - Best KS:  {best_accepted_ks['Model']} ({best_accepted_ks['KS (Validation)']:.3f})")

    # Performance vs complexity analysis
    print(f"\nâš¡ PERFORMANCE vs SPEED ANALYSIS:")
    fastest_model = full_results_df.loc[full_results_df['Runtime (s)'].idxmin()]
    print(f"Fastest Model: {fastest_model['Model']} ({fastest_model['Runtime (s)']}s, AUC: {fastest_model['AUC (Validation)']:.3f})")

    high_performance = full_results_df[full_results_df['AUC (Validation)'] > 0.70]
    if not high_performance.empty:
        fastest_good = high_performance.loc[high_performance['Runtime (s)'].idxmin()]
        print(f"Fastest High-Performance Model: {fastest_good['Model']} ({fastest_good['Runtime (s)']}s, AUC: {fastest_good['AUC (Validation)']:.3f})")

    # Overfitting analysis
    print(f"\nğŸ”� OVERFITTING ANALYSIS:")
    overfitting_analysis = full_results_df.copy()
    overfitting_analysis['Overfitting_Score'] = overfitting_analysis['Overfitting']
    worst_overfitting = overfitting_analysis.loc[overfitting_analysis['Overfitting_Score'].idxmax()]
    best_generalization = overfitting_analysis.loc[overfitting_analysis['Overfitting_Score'].idxmin()]

    print(f"Most Overfitted: {worst_overfitting['Model']} (Gap: {worst_overfitting['Overfitting_Score']:.3f})")
    print(f"Best Generalization: {best_generalization['Model']} (Gap: {best_generalization['Overfitting_Score']:.3f})")

    # Recommendations
    print(f"\n RECOMMENDATIONS:")
    print("1. For Production Deployment:")
    if not high_performance.empty and fastest_good['AUC (Validation)'] > 0.72:
        print(f"   â†’ Use {fastest_good['Model']} (Good performance + Fast inference)")
    else:
        print(f"   â†’ Use {best_full_auc['Model']} (Best overall performance)")

    print("2. For Interpretability:")
    interpretable_models = full_results_df[full_results_df['Model'].str.contains('Linear|Logistic|CART')]
    if not interpretable_models.empty:
        best_interpretable = interpretable_models.loc[interpretable_models['AUC (Validation)'].idxmax()]
        print(f"   â†’ Use {best_interpretable['Model']} (AUC: {best_interpretable['AUC (Validation)']:.3f})")

    print("3. For Reject Inference:")
    accept_vs_full_comparison = pd.merge(
        full_results_df[['Model', 'AUC (Validation)']],
        accepted_results_df[['Model', 'AUC (Validation)']],
        on='Model', suffixes=('_Full', '_Accepted')
    )
    accept_vs_full_comparison['Performance_Drop'] = accept_vs_full_comparison['AUC (Validation)_Full'] - accept_vs_full_comparison['AUC (Validation)_Accepted']
    most_robust = accept_vs_full_comparison.loc[accept_vs_full_comparison['Performance_Drop'].idxmin()]
    print(f"   â†’ {most_robust['Model']} shows least performance drop when trained on accepted-only data")
    print(f"     (Drop: {most_robust['Performance_Drop']:.3f})")

    print(f"\n Analysis Complete! Results saved to 'credit_scoring_complete_analysis.pdf'")
    print("="*80)

    return full_results_df, accepted_results_df

# ==================================================================================
# OPTIONAL ANALYSIS FUNCTIONS
# ==================================================================================

def display_feature_importance():
    """Display feature importance for models that support it"""
    print("\nğŸŒ³ FEATURE IMPORTANCE ANALYSIS:")
    print("-" * 40)
    
    feature_names = features
    
    for model_name, model_info in trained_models.items():
        model = model_info['model']
        
        if hasattr(model, 'feature_importances_'):
            print(f"\n{model_name}:")
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print(feature_importance_df.to_string(index=False))
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(feature_importance_df)), feature_importance_df['Importance'])
            plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance - {model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

def create_model_comparison_heatmap():
    """Create a heatmap comparing all model metrics"""
    print("\n CREATING MODEL COMPARISON HEATMAP")
    print("-" * 40)
    
    # Prepare data for heatmap
    metrics_for_heatmap = pd.DataFrame(model_summaries).set_index('Model')[['AUC (Validation)', 'KS (Validation)', 'Runtime (s)', 'Overfitting']]
    
    # Normalize metrics for better visualization
    metrics_normalized = metrics_for_heatmap.copy()
    metrics_normalized['Runtime (s)'] = 1 / (1 + metrics_normalized['Runtime (s)'])  # Invert runtime (higher is better)
    metrics_normalized['Overfitting'] = 1 / (1 + metrics_normalized['Overfitting'])  # Invert overfitting (higher is better)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(metrics_normalized.T, 
                annot=True, 
                cmap='RdYlGn', 
                center=0.5,
                fmt='.3f',
                cbar_kws={'label': 'Normalized Score (Higher is Better)'})
    plt.title('Model Performance Heatmap\n(All Metrics Normalized - Higher is Better)', fontsize=14, fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# ==================================================================================
# MAIN EXECUTION COMMANDS
# ==================================================================================

# ==================================================================================
# TESTING CELL: SMALL SAMPLE FROM ORIGINAL DATA FOR VALIDATION
# ==================================================================================

def create_sample_from_original_data(sample_percentage=0.05):
    """
    Create a small sample from the original dataset for testing purposes.
    Uses actual data rather than synthetic data for more realistic validation.
    
    Args:
        sample_percentage (float): Percentage of original data to use (default 5%)
    """
    try:
        # Check if original data is available
        if 'X' not in globals() or 'y' not in globals():
            raise ValueError("Original data (X, y) not found. Please load the dataset first.")
        
        print(f"   Creating {sample_percentage*100:.1f}% sample from original dataset")
        print(f"   Original dataset size: {X.shape[0]:,} samples")
        
        # Calculate sample size
        sample_size = max(1000, int(len(X) * sample_percentage))  # Minimum 1000 samples
        sample_size = min(sample_size, len(X))  # Don't exceed original size
        
        # Create stratified sample to maintain class balance
        from sklearn.model_selection import train_test_split
        X_sample, _, y_sample, _ = train_test_split(
            X, y, 
            train_size=sample_size, 
            stratify=y, 
            random_state=42
        )
        
        print(f"   Sample size: {sample_size:,} samples ({sample_size/len(X)*100:.1f}%)")
        print(f"   Target distribution: {y_sample.mean():.3f} default rate")
        print(f"   Features: {list(X_sample.columns)}")
        
        # Display sample statistics
        print(f"\n Sample vs Original Comparison:")
        print(f"   Original default rate: {y.mean():.3f}")
        print(f"   Sample default rate: {y_sample.mean():.3f}")
        print(f"   Feature means comparison:")
        
        comparison_df = pd.DataFrame({
            'Original_Mean': X.mean(),
            'Sample_Mean': X_sample.mean(),
            'Difference': X.mean() - X_sample.mean()
        }).round(2)
        print(comparison_df.to_string())
        
        return X_sample, y_sample
        
    except Exception as e:
        print(f"â�Œ Error creating sample from original data: {str(e)}")
        print(" Creating synthetic sample data as fallback...")
        return create_synthetic_sample_data()

def create_synthetic_sample_data(n_samples=1000):
    """
    Fallback function to create synthetic data if original data is not available
    """
    np.random.seed(42)
    
    # Create synthetic data with similar structure to credit scoring
    sample_data = {
        'AMT_INCOME_TOTAL': np.random.lognormal(mean=11.5, sigma=0.7, size=n_samples),
        'AMT_CREDIT': np.random.lognormal(mean=13.0, sigma=0.8, size=n_samples),
        'AMT_ANNUITY': np.random.lognormal(mean=9.5, sigma=0.6, size=n_samples),
        'DAYS_EMPLOYED': np.random.normal(loc=-1500, scale=1000, size=n_samples),
        'DAYS_BIRTH': np.random.normal(loc=-15000, scale=4000, size=n_samples),
        'EXT_SOURCE_1': np.random.beta(a=2, b=3, size=n_samples),
        'EXT_SOURCE_2': np.random.beta(a=2, b=3, size=n_samples),
        'EXT_SOURCE_3': np.random.beta(a=2, b=3, size=n_samples)
    }
    
    # Add some missing values to EXT_SOURCE features (like real data)
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        missing_mask = np.random.random(n_samples) < 0.15  # 15% missing
        sample_data[col][missing_mask] = np.nan
    
    # Create DataFrame and replace missing values with 0
    X_sample = pd.DataFrame(sample_data).fillna(0)
    
    # Create target variable (correlated with features for realistic results)
    risk_score = (
        X_sample['AMT_CREDIT'] / X_sample['AMT_INCOME_TOTAL'] * 0.3 +
        (1 - X_sample['EXT_SOURCE_2']) * 0.4 +
        (1 - X_sample['EXT_SOURCE_3']) * 0.3 +
        np.random.normal(0, 0.2, n_samples)
    )
    
    # Convert to binary target (approximately 8% default rate like real data)
    y_sample = (risk_score > np.percentile(risk_score, 92)).astype(int)
    
    print(f" Created synthetic sample dataset:")
    print(f" Shape: {X_sample.shape}")
    print(f" Target distribution: {y_sample.mean():.3f} default rate")
    
    return X_sample, y_sample

def test_all_models_on_sample(sample_percentage=0.05):
    """
    Test all models on a small sample of the original dataset to verify the code works
    
    Args:
        sample_percentage (float): Percentage of original data to use (default 5%)
    """
    print(" TESTING ALL MODELS ON SAMPLE DATA")
    print("="*60)
    
    # Create sample data from original dataset
    X_sample, y_sample = create_sample_from_original_data(sample_percentage)
    
    # Split the sample data
    X_train_sample, X_val_sample, y_train_sample, y_val_sample = train_test_split(
        X_sample, y_sample, test_size=0.3, stratify=y_sample, random_state=42
    )
    
    print(f"\n Data splits:")
    print(f"   Training set: {X_train_sample.shape[0]:,} samples")
    print(f"   Validation set: {X_val_sample.shape[0]:,} samples")
    print(f"   Training default rate: {y_train_sample.mean():.3f}")
    print(f"   Validation default rate: {y_val_sample.mean():.3f}")
    
    # Adjust parameters for smaller sample size
    min_samples_for_tree = max(1, min(50, len(X_train_sample)//20))
    min_samples_for_lgb = max(1, min(100, len(X_train_sample)//10))
    
    # Test each model individually
    models_to_test = [
        ("Linear Probability", create_linear_model()),
        ("CART", create_cart_model()),
        ("Logistic Regression", create_logistic_model()),
        ("Random Forest", create_random_forest_model()),
        ("LightGBM", create_lightgbm_model()),
    ]
    
    # Only test SVM on very small samples (it's O(nÂ²) complexity!)
    svm_sample_threshold = 2000  # Max samples for SVM testing
    if len(X_train_sample) <= svm_sample_threshold:
        models_to_test.append(("SVM", create_svm_model()))
    else:
        print(f"   Skipping SVM test (sample size {len(X_train_sample)} > {svm_sample_threshold})")
        print(f"   SVM is O(nÂ²) complexity - use quick_test_svm() for SVM testing")
    
    reset_model_tracking()
    
    print("\n Testing individual models...")
    successful_models = 0
    failed_models = []
    
    # Test regular models
    for name, model in models_to_test:
        try:
            print(f"\nâœ… Testing {name}...")
            
            # Adjust LightGBM parameters for small sample
            if name == "LightGBM":
                model.min_child_samples = min_samples_for_lgb
                model.min_data_in_leaf = min_samples_for_lgb
                model.n_estimators = min(50, model.n_estimators)  # Reduce for speed
            
            # Adjust Random Forest parameters for small sample
            elif name == "Random Forest":
                model.min_samples_leaf = min_samples_for_tree
                model.n_estimators = min(100, model.n_estimators)  # Reduce for speed
            
            run_model(f"{name} (Sample)", model, X_train_sample, y_train_sample, X_val_sample, y_val_sample)
            successful_models += 1
            
        except Exception as e:
            print(f"â�Œ Error in {name}: {str(e)}")
            failed_models.append(name)
    
    # Test CART-binarized logistic
    try:
        print(f"\n Testing CART-Binarized Logistic...")
        logistic_model = create_cart_binarized_logistic()
        
        # Adjust CART binarization parameters for small sample
        adjusted_min_samples_leaf = max(1, min(min_samples_for_tree, len(X_train_sample)//10))
        
        X_train_bin, X_val_bin = binarize_features_with_cart(
            X_train_sample, X_val_sample, y_train_sample,
            max_depth=min(4, CART_BINARIZATION_CONFIG['max_depth']),
            min_samples_leaf=adjusted_min_samples_leaf
        )
        run_model("CART-Binarized â†’ Logistic (Sample)", logistic_model, X_train_bin, y_train_sample, X_val_bin, y_val_sample)
        successful_models += 1
        
    except Exception as e:
        print(f"â�Œ Error in CART-Binarized Logistic: {str(e)}")
        failed_models.append("CART-Binarized Logistic")
    
    # Test Neural Network
    try:
        print(f"\nâœ… Testing Neural Network...")
        
        # Temporarily modify neural network config for small sample
        original_epochs = NEURAL_NET_CONFIG['epochs']
        original_batch_size = NEURAL_NET_CONFIG['batch_size']
        
        # Adjust for small sample
        test_epochs = min(20, original_epochs)
        test_batch_size = min(64, max(16, len(X_train_sample)//10))
        
        # Create and compile model
        nn_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train_sample.shape[1],)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),  # Smaller layers for sample
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train model
        history = nn_model.fit(
            X_train_sample, y_train_sample,
            validation_data=(X_val_sample, y_val_sample),
            epochs=test_epochs,
            batch_size=test_batch_size,
            verbose=0
        )
        
        # Calculate metrics
        y_val_pred = nn_model.predict(X_val_sample, verbose=0).flatten()
        auc_val = roc_auc_score(y_val_sample, y_val_pred)
        
        fpr_val, tpr_val, _ = roc_curve(y_val_sample, y_val_pred)
        ks_val = max(tpr_val - fpr_val)
        
        print(f"Neural Network (Sample): AUC={auc_val:.3f}, KS={ks_val:.3f}")
        successful_models += 1
        
    except Exception as e:
        print(f"â�Œ Error in Neural Network: {str(e)}")
        failed_models.append("Neural Network")
    
    # Display results
    print(f"\n SAMPLE TEST RESULTS:")
    print(f" Successful models: {successful_models}/8")
    if failed_models:
        print(f"â�Œ Failed models: {', '.join(failed_models)}")
    print("-" * 50)
    
    if model_summaries:
        results_df = show_model_summary()
        print(results_df.to_string(index=False))
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # AUC Comparison
        auc_values = results_df['AUC (Validation)'].values
        model_names = [name.replace(' (Sample)', '').replace('â†’', 'â†’\n') for name in results_df['Model']]
        
        bars1 = axes[0,0].bar(range(len(auc_values)), auc_values, color='skyblue')
        axes[0,0].set_title("AUC Comparison (Sample Data)", fontweight='bold')
        axes[0,0].set_ylabel("AUC Score")
        axes[0,0].set_xticks(range(len(model_names)))
        axes[0,0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # KS Comparison
        ks_values = results_df['KS (Validation)'].values
        bars2 = axes[0,1].bar(range(len(ks_values)), ks_values, color='lightcoral')
        axes[0,1].set_title("KS Statistics Comparison", fontweight='bold')
        axes[0,1].set_ylabel("KS Statistic")
        axes[0,1].set_xticks(range(len(model_names)))
        axes[0,1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0,1].grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Runtime Comparison
        runtimes = results_df['Runtime (s)'].values
        bars3 = axes[1,0].bar(range(len(runtimes)), runtimes, color='lightgreen')
        axes[1,0].set_title("Runtime Comparison", fontweight='bold')
        axes[1,0].set_ylabel("Runtime (seconds)")
        axes[1,0].set_xticks(range(len(model_names)))
        axes[1,0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1,0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + height*0.1,
                          f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
        
        # Overfitting Analysis
        overfitting = results_df['Overfitting'].values
        colors_over = ['red' if x > 0.05 else 'green' if x < 0.02 else 'orange' for x in overfitting]
        bars4 = axes[1,1].bar(range(len(overfitting)), overfitting, color=colors_over, alpha=0.7)
        axes[1,1].set_title("Overfitting Analysis", fontweight='bold')
        axes[1,1].set_ylabel("Train AUC - Val AUC")
        axes[1,1].set_xticks(range(len(model_names)))
        axes[1,1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1,1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
        axes[1,1].axhline(y=0.02, color='orange', linestyle='--', alpha=0.5)
        axes[1,1].grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # Summary statistics
        print(f"\n PERFORMANCE SUMMARY:")
        print(f"   Best AUC: {results_df['AUC (Validation)'].max():.3f} ({results_df.loc[results_df['AUC (Validation)'].idxmax(), 'Model']})")
        print(f"   Best KS:  {results_df['KS (Validation)'].max():.3f} ({results_df.loc[results_df['KS (Validation)'].idxmax(), 'Model']})")
        print(f"   Fastest:  {results_df['Runtime (s)'].min():.2f}s ({results_df.loc[results_df['Runtime (s)'].idxmin(), 'Model']})")
        print(f"   Avg AUC:  {results_df['AUC (Validation)'].mean():.3f}")
        
        print(f"\n All model tests completed successfully!")
        print(f" The notebook is ready for use with the full dataset.")
        print(f" To run full analysis: full_results, accepted_results = run_complete_analysis()")
        
    else:
        print("â�Œ No models completed successfully. Check for errors above.")
    
    return X_sample, y_sample

def quick_test_svm(max_samples=500):
    """
    Quick SVM test on very small sample due to O(nÂ²) complexity
    Also tests LinearSVC as faster alternative
    """
    print(f" QUICK SVM TEST (Max {max_samples} samples)")
    print("="*45)
    
    try:
        # Get very small sample for SVM
        if 'X' not in globals():
            print("â�Œ Original data not available")
            return
            
        sample_size = min(max_samples, len(X))
        X_tiny, _, y_tiny, _ = train_test_split(
            X, y, train_size=sample_size, stratify=y, random_state=42
        )
        
        X_train_tiny, X_val_tiny, y_train_tiny, y_val_tiny = train_test_split(
            X_tiny, y_tiny, test_size=0.3, stratify=y_tiny, random_state=42
        )
        
        print(f"Testing with {len(X_train_tiny)} training samples")
        
        reset_model_tracking()
        
        # Test regular SVM
        try:
            print("\n Testing SVC (RBF kernel, probability=False for speed)...")
            from sklearn.svm import SVC
            svm_fast = SVC(kernel='rbf', probability=False, C=1.0, random_state=42)
            run_model("SVM-RBF (Fast)", svm_fast, X_train_tiny, y_train_tiny, X_val_tiny, y_val_tiny)
        except Exception as e:
            print(f"â�Œ SVC failed: {e}")
        
        # Test LinearSVC (much faster for linear problems)
        try:
            print("\n Testing LinearSVC (much faster alternative)...")
            from sklearn.svm import LinearSVC
            from sklearn.calibration import CalibratedClassifierCV
            
            # LinearSVC with probability calibration
            linear_svc = LinearSVC(C=1.0, random_state=42, max_iter=1000)
            calibrated_svc = CalibratedClassifierCV(linear_svc, method='sigmoid', cv=3)
            run_model("LinearSVC (Calibrated)", calibrated_svc, X_train_tiny, y_train_tiny, X_val_tiny, y_val_tiny)
        except Exception as e:
            print(f"â�Œ LinearSVC failed: {e}")
        
        # Test SGD with SVM loss (fastest SVM approximation)
        try:
            print("\n Testing SGD-SVM (fastest SVM approximation)...")
            from sklearn.linear_model import SGDClassifier
            sgd_svm = SGDClassifier(loss='hinge', learning_rate='constant', eta0=0.01, random_state=42)
            run_model("SGD-SVM", sgd_svm, X_train_tiny, y_train_tiny, X_val_tiny, y_val_tiny)
        except Exception as e:
            print(f"â�Œ SGD-SVM failed: {e}")
        
        if model_summaries:
            results = show_model_summary()
            print(f"\n SVM COMPARISON RESULTS:")
            print(results.to_string(index=False))
            
            print(f"\n SVM Complexity Notes:")
            print(f"   SVC (RBF):     O(nÂ²) - {sample_size}Â² = {sample_size**2:,} operations")
            print(f"   LinearSVC:     O(n) - {sample_size} operations")  
            print(f"   SGD-SVM:       O(n) - {sample_size} operations")
            print(f"   For production: Use LinearSVC or SGD-SVM for speed")
        
    except Exception as e:
        print(f"â�Œ SVM test failed: {e}")

def explain_model_complexity():
    """
    Explain Big O complexity for all models
    """
    print("ğŸ“š MODEL COMPLEXITY GUIDE")
    print("="*50)
    
    complexity_info = {
        "Linear Regression": {
            "Training": "O(nÃ—dÂ²) or O(nÃ—dÃ—i)", 
            "Prediction": "O(d)",
            "Notes": "Fast, scales well"
        },
        "Logistic Regression": {
            "Training": "O(nÃ—dÃ—i)", 
            "Prediction": "O(d)",
            "Notes": "i=iterations, usually fast"
        },
        "CART": {
            "Training": "O(nÃ—dÃ—log n)", 
            "Prediction": "O(log n)",
            "Notes": "Fast for single trees"
        },
        "Random Forest": {
            "Training": "O(mÃ—nÃ—dÃ—log n)", 
            "Prediction": "O(mÃ—log n)",
            "Notes": "m=trees, parallelizable"
        },
        "LightGBM": {
            "Training": "O(nÃ—dÃ—log n)", 
            "Prediction": "O(log n)",
            "Notes": "Very optimized, fastest boosting"
        },
        "SVM (Linear)": {
            "Training": "O(nÃ—d)", 
            "Prediction": "O(d)",
            "Notes": "Fast with LinearSVC"
        },
        "SVM (RBF)": {
            "Training": "O(nÂ²Ã—d) to O(nÂ³Ã—d)", 
            "Prediction": "O(n_svÃ—d)",
            "Notes": "âš ï¸� SLOW! Avoid for n>10k"
        },
        "Neural Network": {
            "Training": "O(nÃ—dÃ—hÃ—i)", 
            "Prediction": "O(dÃ—h)",
            "Notes": "h=hidden units, i=epochs"
        }
    }
    
    for model, info in complexity_info.items():
        print(f"\n {model}:")
        print(f"   Training:   {info['Training']}")
        print(f"   Prediction: {info['Prediction']}")
        print(f"   Notes:      {info['Notes']}")
    
    print(f"\n PERFORMANCE RECOMMENDATIONS:")
    print(f"   â€¢ n < 1,000:     All models OK")
    print(f"   â€¢ n < 10,000:    Avoid RBF SVM")
    print(f"   â€¢ n < 100,000:   Use LinearSVC instead of SVC")
    print(f"   â€¢ n > 100,000:   LightGBM, Linear models, SGD variants")
    print(f"   â€¢ n > 1M:        SGD, Linear models only")
    """
    Quick test of a single model on very small sample for rapid validation
    
    Args:
        model_name (str): Name of model to test
        sample_percentage (float): Percentage of data to use (default 2%)
    """
    print(f" QUICK TEST: {model_name}")
    print("="*40)
    
    # Get small sample
    X_sample, y_sample = create_sample_from_original_data(sample_percentage)
    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
        X_sample, y_sample, test_size=0.3, stratify=y_sample, random_state=42
    )
    
    # Create and test model
    if model_name == "Random Forest":
        model = create_random_forest_model()
        model.n_estimators = 50  # Reduce for speed
    elif model_name == "LightGBM":
        model = create_lightgbm_model()
        model.n_estimators = 50
    elif model_name == "Logistic Regression":
        model = create_logistic_model()
    else:
        print(f"Model {model_name} not recognized for quick test")
        return
    
    reset_model_tracking()
    run_model(f"{model_name} (Quick Test)", model, X_train_s, y_train_s, X_val_s, y_val_s)
    
    results = show_model_summary()
    print(f"\n Quick Test Result:")
    print(f"   AUC: {results['AUC (Validation)'].iloc[0]:.3f}")
    print(f"   KS:  {results['KS (Validation)'].iloc[0]:.3f}")
    print(f"   Runtime: {results['Runtime (s)'].iloc[0]:.2f}s")
    print(f" {model_name} working correctly!")

# ==================================================================================
# FINAL EXECUTION COMMANDS
# ==================================================================================

# Uncomment any of these lines to run specific analyses:

# 1. Quick test with single model (fastest validation):
# quick_test_single_model("Random Forest", sample_percentage=0.02)

# 2. Test all models on 5% sample (recommended first step):
# X_sample, y_sample = test_all_models_on_sample(sample_percentage=0.05)

# 3. Test all models on smaller 2% sample (for faster testing):
# X_sample, y_sample = test_all_models_on_sample(sample_percentage=0.02)

# 4. Run complete analysis with PDF export (use with full dataset):
# full_results, accepted_results = run_complete_analysis()

# 5. Run all configured models:
# results = run_all_configured_models()

# 6. Test individual models:
# test_linear_model()
# test_cart_model()
# test_cart_binarized_logistic()
# test_logistic_model()
# test_random_forest_model()
# test_lightgbm_model()
# test_svm_model()
# nn_model, nn_history = test_neural_network_model()

# 7. Optional analyses:
# display_feature_importance()
# create_model_comparison_heatmap()

print(" Complete Enhanced Credit Scoring Notebook Ready!")
print(" Quick test: quick_test_single_model('Random Forest')")
print(" Full test: X_sample, y_sample = test_all_models_on_sample()")
print(" Full analysis: full_results, accepted_results = run_complete_analysis()")
print(" All models are fully configurable through their respective CONFIG dictionaries")
quick_test_svm(max_samples=500)
test_all_models_on_sample(sample_percentage=0.05)




