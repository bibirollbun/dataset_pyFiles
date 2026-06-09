# === Imports ===

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import lightgbm as lgb

# TensorFlow/Keras
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")


def load_data(sample_fraction=1.0, random_state=42):
    """
    Load the Home Credit dataset with optional sampling.
    
    Parameters
    ----------
    sample_fraction : float
        Fraction of data to use (0.0-1.0). Use smaller values for faster iteration.
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable (1 = default, 0 = no default)
    """
    print(f"Loading data (sample_fraction={sample_fraction})...")
    
    # Adjust path as needed for your environment
    app = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
    
    if sample_fraction < 1.0:
        app = app.sample(frac=sample_fraction, random_state=random_state)
        print(f"Sampled to {len(app):,} records")
    else:
        print(f"Full dataset: {len(app):,} records")
    
    # Select key features for credit scoring
    features = [
        'AMT_INCOME_TOTAL',  # Income
        'AMT_CREDIT',        # Credit amount
        'AMT_ANNUITY',       # Annuity amount
        'DAYS_EMPLOYED',     # Employment duration
        'DAYS_BIRTH',        # Age (in days, negative)
        'EXT_SOURCE_1',      # External score 1
        'EXT_SOURCE_2',      # External score 2
        'EXT_SOURCE_3'       # External score 3
    ]
    
    X = app[features].copy()
    
    # Clean: replace anomalous employment value, fill NAs
    X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)
    X = X.fillna(0)
    
    y = app['TARGET']
    
    print(f"Default rate: {y.mean():.2%}")
    
    return X, y


def simulate_acceptance(X, y, score_column='EXT_SOURCE_3', reject_quantile=0.25):
    """
    Simulate an acceptance policy: reject the bottom quantile based on a score.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    score_column : str
        Column to use as the existing score
    reject_quantile : float
        Bottom fraction to reject (e.g., 0.25 = reject bottom 25%)
    
    Returns
    -------
    X_accepted, y_accepted : Features and target for accepted applicants only
    threshold : The score threshold used
    """
    threshold = X[score_column].quantile(reject_quantile)
    accepted_mask = X[score_column] > threshold
    
    X_accepted = X.loc[accepted_mask].copy()
    y_accepted = y.loc[accepted_mask].copy()
    
    print(f"Acceptance threshold ({score_column}): {threshold:.4f}")
    print(f"Accepted: {len(X_accepted):,} ({accepted_mask.mean():.1%})")
    print(f"Default rate (accepted only): {y_accepted.mean():.2%}")
    
    return X_accepted, y_accepted, threshold


def get_models(n_features=8, include_slow=True):
    """
    Return a dictionary of models to evaluate.
    
    Parameters
    ----------
    n_features : int
        Number of input features (needed for Keras model)
    include_slow : bool
        Whether to include slower models (Random Forest, LightGBM, Neural Net)
    
    Returns
    -------
    dict : {model_name: model_instance}
    """
    models = {
        'Linear Probability Model': LinearRegression(),
        
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            solver='lbfgs', 
            C=1.0
        ),
        
        'CART (Decision Tree)': DecisionTreeClassifier(
            max_depth=5, 
            min_samples_leaf=50,
            random_state=42
        ),
        
        'LinearSVC': LinearSVC(
            dual=True, 
            C=1.0, 
            max_iter=2000, 
            random_state=42
        ),
        
        'SGD (Hinge Loss)': SGDClassifier(
            loss='hinge', 
            penalty='l2',
            alpha=0.0001, 
            max_iter=1000,
            random_state=42
        ),
    }
    
    if include_slow:
        models['Random Forest'] = RandomForestClassifier(
            n_estimators=100, 
            max_depth=8,
            min_samples_leaf=20, 
            n_jobs=-1,
            random_state=42
        )
        
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=100, 
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=100,
            n_jobs=-1,
            random_state=42,
            verbosity=-1
        )
        
        models['Neural Net (Keras)'] = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
    return models


def binarise_with_cart(X_train, X_val, y_train, max_depth=4, min_samples_leaf=50):
    """
    Fit a CART tree and return one-hot encoded leaf assignments.
    
    Returns
    -------
    X_train_bin, X_val_bin : np.ndarray
        One-hot encoded leaf assignments
    """
    cart = DecisionTreeClassifier(
        max_depth=max_depth, 
        min_samples_leaf=min_samples_leaf, 
        random_state=42
    )
    cart.fit(X_train, y_train)
    
    leaf_train = cart.apply(X_train).reshape(-1, 1)
    leaf_val = cart.apply(X_val).reshape(-1, 1)
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_bin = encoder.fit_transform(leaf_train)
    X_val_bin = encoder.transform(leaf_val)
    
    print(f"CART produced {X_train_bin.shape[1]} leaf nodes")
    
    return X_train_bin, X_val_bin


class ModelBenchmark:
    """
    Runs and tracks model evaluations.
    """
    
    def __init__(self):
        self.results = []
        self.roc_data = []
    
    def reset(self):
        self.results = []
        self.roc_data = []
    
    def evaluate(self, name, model, X_train, y_train, X_val, y_val, is_keras=False):
        """
        Train and evaluate a single model.
        """
        start = time.time()
        
        if is_keras:
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,
                batch_size=1024,
                verbose=0,
                callbacks=[early_stop]
            )
            y_pred = model.predict(X_val, verbose=0).flatten()
        else:
            model.fit(X_train, y_train)
            
            # Get probability scores
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_val)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_pred = model.decision_function(X_val)
            else:
                y_pred = model.predict(X_val)
        
        runtime = time.time() - start
        
        # Metrics
        auc = roc_auc_score(y_val, y_pred)
        fpr, tpr, _ = roc_curve(y_val, y_pred)
        ks = np.max(tpr - fpr)
        
        self.results.append({
            'Model': name,
            'AUC': auc,
            'KS': ks,
            'Runtime (s)': round(runtime, 2)
        })
        
        self.roc_data.append((name, fpr, tpr, auc))
        
        print(f"{name}: AUC={auc:.4f}, KS={ks:.4f}, Time={runtime:.1f}s")
        
        return model
    
    def get_results(self):
        """Return results as DataFrame, sorted by KS."""
        df = pd.DataFrame(self.results)
        return df.sort_values('KS', ascending=False).reset_index(drop=True)
    
    def plot_roc(self, title='ROC Curves'):
        """Plot ROC curves for all evaluated models."""
        plt.figure(figsize=(10, 6))
        colours = plt.cm.tab10.colors
        
        for i, (name, fpr, tpr, auc) in enumerate(self.roc_data):
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", 
                     color=colours[i % len(colours)], linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def run_benchmark(sample_fraction=0.1, include_slow=True, reject_quantile=0.25):
    """
    Run the full benchmark: train on full data and accepted-only, compare results.
    
    Parameters
    ----------
    sample_fraction : float
        Fraction of data to use (0.01-1.0)
    include_slow : bool
        Include slower models (RF, LightGBM, Neural Net)
    reject_quantile : float
        Bottom fraction to simulate as rejected (for reject inference)
    
    Returns
    -------
    full_results, accepted_results, comparison : DataFrames
    """
    # Load data
    X, y = load_data(sample_fraction=sample_fraction)
    n_features = X.shape[1]
    
    # Train/val split (full population)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"\nTrain: {len(X_train):,}, Validation: {len(X_val):,}")
    
    # Simulate acceptance (on training data only)
    print(f"\n--- Simulating acceptance policy ---")
    X_train_acc, y_train_acc, threshold = simulate_acceptance(
        X_train, y_train, reject_quantile=reject_quantile
    )
    
    # Scale features (important for SVM, Neural Net)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_train_acc_scaled = scaler.transform(X_train_acc)
    
    # ========== FULL DATA ==========
    print("\n" + "="*60)
    print("TRAINING ON FULL DATA")
    print("="*60)
    
    bench_full = ModelBenchmark()
    models = get_models(n_features=n_features, include_slow=include_slow)
    
    for name, model in models.items():
        is_keras = 'Keras' in name
        bench_full.evaluate(name, model, X_train_scaled, y_train, X_val_scaled, y_val, is_keras)
    
    # CART-Binarised → Logistic
    X_train_bin, X_val_bin = binarise_with_cart(X_train_scaled, X_val_scaled, y_train)
    lr_bin = LogisticRegression(max_iter=1000)
    bench_full.evaluate('CART-Binarised → Logistic', lr_bin, X_train_bin, y_train, X_val_bin, y_val)
    
    full_results = bench_full.get_results()
    bench_full.plot_roc('ROC Curves - Full Training Data')
    
    # ========== ACCEPTED ONLY ==========
    print("\n" + "="*60)
    print("TRAINING ON ACCEPTED APPLICANTS ONLY (Reject Inference Demo)")
    print("="*60)
    
    bench_acc = ModelBenchmark()
    models_acc = get_models(n_features=n_features, include_slow=include_slow)
    
    for name, model in models_acc.items():
        is_keras = 'Keras' in name
        bench_acc.evaluate(name, model, X_train_acc_scaled, y_train_acc, X_val_scaled, y_val, is_keras)
    
    # CART-Binarised → Logistic (accepted only)
    X_train_acc_bin, X_val_bin2 = binarise_with_cart(X_train_acc_scaled, X_val_scaled, y_train_acc)
    lr_bin2 = LogisticRegression(max_iter=1000)
    bench_acc.evaluate('CART-Binarised → Logistic', lr_bin2, X_train_acc_bin, y_train_acc, X_val_bin2, y_val)
    
    accepted_results = bench_acc.get_results()
    bench_acc.plot_roc('ROC Curves - Accepted Applicants Only')
    
    # ========== COMPARISON ==========
    print("\n" + "="*60)
    print("COMPARISON: FULL vs ACCEPTED-ONLY")
    print("="*60)
    
    comparison = pd.merge(
        full_results.rename(columns={'AUC': 'AUC (Full)', 'KS': 'KS (Full)'}),
        accepted_results.rename(columns={'AUC': 'AUC (Accepted)', 'KS': 'KS (Accepted)'}),
        on='Model',
        suffixes=('', '_acc')
    )
    
    comparison['AUC Drop'] = comparison['AUC (Full)'] - comparison['AUC (Accepted)']
    comparison['KS Drop'] = comparison['KS (Full)'] - comparison['KS (Accepted)']
    comparison = comparison.sort_values('KS (Full)', ascending=False)
    
    display(comparison[['Model', 'AUC (Full)', 'AUC (Accepted)', 'AUC Drop', 
                        'KS (Full)', 'KS (Accepted)', 'KS Drop']])
    
    return full_results, accepted_results, comparison


# Quick test (1% data, fast models only)
full_results, accepted_results, comparison = run_benchmark(
    sample_fraction=0.05,
    include_slow=True,
    reject_quantile=0.25
)


# Display final leaderboards
print("=== Full Data Leaderboard ===")
display(full_results)

print("\n=== Accepted-Only Leaderboard ===")
display(accepted_results)

