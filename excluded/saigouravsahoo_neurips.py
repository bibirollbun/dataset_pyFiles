# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install pandas numpy scikit-learn xgboost matplotlib seaborn rdkit plotly

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

# Checking RDKit availability
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
    RDKIT_AVAILABLE = True
    print("RDKit available - using molecular descriptors")
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not installed - using basic SMILES features only")

def calculate_wmae_weights(y_true, target_columns):
    """
    Calculate wMAE weights according to contest specification:
    wi = (1/ri) * (K * (1/ni) / sum(1/nj for j in 1..K))
    """
    K = len(target_columns)
    weights = {}

    # Calculate ni (number of available values) and ri (range) for each property
    n_values = {}
    ranges = {}

    for col in target_columns:
        # Count non-null values
        valid_mask = ~y_true[col].isnull()
        n_values[col] = valid_mask.sum()

        # Calculate range (max - min) for non-null values
        if n_values[col] > 1:
            ranges[col] = y_true[col][valid_mask].max() - y_true[col][valid_mask].min()
        else:
            ranges[col] = 1.0  # Avoid division by zero

        # Ensure range is not zero
        if ranges[col] == 0:
            ranges[col] = 1.0

    # Calculate normalization factor: sum(1/nj for j in 1..K)
    normalization_factor = sum(1/n_values[col] for col in target_columns)

    # Calculate weights
    for col in target_columns:
        # wi = (1/ri) * (K * (1/ni) / normalization_factor)
        weights[col] = (1/ranges[col]) * (K * (1/n_values[col]) / normalization_factor)

    print(f"\nwMAE Weight Calculation:")
    print(f"{'Property':<12} {'Count (ni)':<12} {'Range (ri)':<15} {'Weight (wi)':<12}")
    print("-" * 55)
    for col in target_columns:
        print(f"{col:<12} {n_values[col]:<12} {ranges[col]:<15.4f} {weights[col]:<12.6f}")

    return weights

def calculate_wmae(y_true, y_pred, weights, target_columns):
    """
    Calculate weighted Mean Absolute Error according to contest specification
    """
    total_weighted_error = 0
    total_samples = 0

    for col in target_columns:
        # Only consider samples where true value is not null
        valid_mask = ~y_true[col].isnull()
        if valid_mask.sum() > 0:
            mae = np.mean(np.abs(y_pred[col][valid_mask] - y_true[col][valid_mask]))
            weighted_mae = weights[col] * mae
            total_weighted_error += weighted_mae * valid_mask.sum()
            total_samples += valid_mask.sum()

    if total_samples > 0:
        wmae = total_weighted_error / total_samples
    else:
        wmae = float('inf')

    return wmae

def wmae_scorer(y_true, y_pred, weights, target_columns):
    """Custom scorer for cross-validation"""
    if isinstance(y_pred, np.ndarray):
        y_pred_df = pd.DataFrame(y_pred, columns=target_columns)
    else:
        y_pred_df = y_pred

    if isinstance(y_true, np.ndarray):
        y_true_df = pd.DataFrame(y_true, columns=target_columns)
    else:
        y_true_df = y_true

    return -calculate_wmae(y_true_df, y_pred_df, weights, target_columns)  # Negative for sklearn (higher is better)

def load_and_inspect_data():
    """Load data and provide basic inspection"""
    print("Loading data...")
    train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
    test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
    target_columns = ['FFV', 'Tg', 'Tc', 'Density', 'Rg']

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print("\nMissing values in training targets:")
    for col in target_columns:
        missing = train[col].isnull().sum()
        print(f"  {col}: {missing}/{len(train)} ({missing/len(train)*100:.1f}%)")

    return train, test, target_columns

def enhanced_smiles_features(smiles_series):
    """Extract comprehensive features from SMILES strings"""
    features = []
    for s in smiles_series:
        feat = {
            # Basic counts
            'smiles_length': len(s),
            'num_C': s.count('C'),
            'num_O': s.count('O'),
            'num_N': s.count('N'),
            'num_S': s.count('S'),
            'num_P': s.count('P'),
            'num_F': s.count('F'),
            'num_Cl': s.count('Cl'),
            'num_Br': s.count('Br'),

            # Bond features
            'num_single_bonds': s.count('-'),
            'num_double_bonds': s.count('='),
            'num_triple_bonds': s.count('#'),
            'num_aromatic_bonds': s.count(':'),

            # Structural features
            'num_rings': s.count('(') + s.count('['),
            'num_branches': s.count('('),
            'num_cycles': s.count('1') + s.count('2') + s.count('3') + s.count('4') + s.count('5'),

            # Polymer-specific features
            'has_star': int('*' in s),
            'num_stars': s.count('*'),
            'has_dot': int('.' in s),
            'num_components': s.count('.') + 1,

            # Complexity measures
            'unique_chars': len(set(s)),
            'char_entropy': len(set(s)) / len(s) if len(s) > 0 else 0,
        }
        features.append(feat)
    return pd.DataFrame(features)

def enhanced_molecular_descriptors(smiles_series):
    """Calculate comprehensive molecular descriptors using RDKit"""
    if not RDKIT_AVAILABLE:
        return pd.DataFrame()

    descriptors = []
    failed_count = 0

    # Define descriptor keys once
    descriptor_keys = [
        'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
        'TPSA', 'LabuteASA', 'NumAromaticRings', 'NumSaturatedRings',
        'NumAliphaticRings', 'RingCount', 'NumHeteroatoms', 'HeavyAtomCount',
        'NumRadicalElectrons', 'BertzCT', 'BalabanJ', 'Chi0v', 'Chi1v',
        'MaxPartialCharge', 'MinPartialCharge', 'FpDensityMorgan1',
        'FpDensityMorgan2', 'FractionCsp3'
    ]

    for s in smiles_series:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol:
                desc = {
                    'MolWt': Descriptors.MolWt(mol),
                    'LogP': Descriptors.MolLogP(mol),
                    'NumHDonors': Descriptors.NumHDonors(mol),
                    'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                    'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'LabuteASA': Descriptors.LabuteASA(mol),
                    'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                    'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
                    'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
                    'RingCount': Descriptors.RingCount(mol),
                    'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
                    'HeavyAtomCount': mol.GetNumHeavyAtoms(),
                    'NumRadicalElectrons': Descriptors.NumRadicalElectrons(mol),
                    'BertzCT': Descriptors.BertzCT(mol),
                    'BalabanJ': Descriptors.BalabanJ(mol),
                    'Chi0v': Descriptors.Chi0v(mol),
                    'Chi1v': Descriptors.Chi1v(mol),
                    'MaxPartialCharge': Descriptors.MaxPartialCharge(mol),
                    'MinPartialCharge': Descriptors.MinPartialCharge(mol),
                    'FpDensityMorgan1': Descriptors.FpDensityMorgan1(mol),
                    'FpDensityMorgan2': Descriptors.FpDensityMorgan2(mol),
                }

                # Handle FractionCsp3 which might not be available
                try:
                    desc['FractionCsp3'] = Descriptors.FractionCsp3(mol)
                except AttributeError:
                    desc['FractionCsp3'] = 0
            else:
                desc = dict.fromkeys(descriptor_keys, 0)
                failed_count += 1

        except Exception:
            desc = dict.fromkeys(descriptor_keys, 0)
            failed_count += 1

        descriptors.append(desc)

    if failed_count > 0:
        print(f"Failed to process {failed_count}/{len(smiles_series)} SMILES")

    return pd.DataFrame(descriptors)

def prepare_features(train, test):
    """Prepare enhanced features from SMILES with proper alignment"""
    print("\nPreparing features...")

    # Combine all SMILES to ensure consistent feature extraction
    all_smiles = pd.concat([train['SMILES'], test['SMILES']], ignore_index=True)

    # Extract features from all SMILES together
    all_smiles_features = enhanced_smiles_features(all_smiles)
    if RDKIT_AVAILABLE:
        all_desc_features = enhanced_molecular_descriptors(all_smiles)
        all_features = pd.concat([all_smiles_features, all_desc_features], axis=1)
    else:
        all_features = all_smiles_features

    # Split back into train and test
    train_size = len(train)
    X_train = all_features.iloc[:train_size].copy()
    X_test = all_features.iloc[train_size:].copy().reset_index(drop=True)

    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Test feature matrix shape: {X_test.shape}")

    # Verify shapes match
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"Feature mismatch: train has {X_train.shape[1]} features, test has {X_test.shape[1]}")

    return X_train, X_test

def clean_and_impute_data(X_train, train, target_columns):
    """Clean data and perform iterative imputation"""
    print("\nCleaning and imputing data...")

    # Combine features and targets
    combined_data = pd.concat([X_train, train[target_columns]], axis=1)

    # Handle infinities and extreme values
    if np.isinf(combined_data.values).any():
        print("Replacing infinities with NaN...")
        combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    max_val = combined_data.max().max()
    if max_val > 1e10:
        print(f"Warning: extremely large value detected ({max_val}). Capping to 1e10.")
        combined_data = combined_data.clip(upper=1e10)

    # Iterative imputation with optimized parameters
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(
            n_estimators=50,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),
        max_iter=5,
        initial_strategy='mean',
        random_state=42,
        verbose=1
    )

    print("Running iterative imputation...")
    imputed_data = imputer.fit_transform(combined_data)
    print(f"Total NaNs after imputation: {np.isnan(imputed_data).sum()}")

    # Split back into features and targets
    X_train_imputed = pd.DataFrame(imputed_data[:, :X_train.shape[1]], columns=X_train.columns)
    y_train_imputed = pd.DataFrame(imputed_data[:, X_train.shape[1]:], columns=target_columns)

    return X_train_imputed, y_train_imputed

def scale_features(X_train_imputed, X_test):
    """Scale features using RobustScaler"""
    print("\nScaling features...")

    # Verify column alignment before scaling
    if list(X_train_imputed.columns) != list(X_test.columns):
        print("Realigning test columns...")
        X_test = X_test[X_train_imputed.columns]

    # Handle any NaNs in test data before scaling
    if X_test.isnull().any().any():
        print("Handling NaNs in test data...")
        for col in X_test.columns:
            if X_test[col].isnull().any():
                # Fill with training data mean for that column
                X_test[col].fillna(X_train_imputed[col].mean(), inplace=True)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test)

    # Final check for NaNs
    if np.isnan(X_test_scaled).any():
        print(f"Found {np.isnan(X_test_scaled).sum()} NaNs in scaled test data — replacing with zeros")
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

    print(f"Training data shape after scaling: {X_train_scaled.shape}")
    print(f"Test data shape after scaling: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled

def train_ensemble_models_with_wmae(X_train_scaled, y_train_imputed, wmae_weights, target_columns):
    """Train ensemble of models with wMAE-based cross-validation"""
    print("\nTraining ensemble models with wMAE evaluation...")

    models = {
        'RandomForest': MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        ),
        'XGBoost': MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        ),
        'GradientBoosting': MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        )
    }

    trained_models = {}
    cv_scores = {}
    cv_wmae_scores = {}

    # Cross-validation setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train_imputed)
        trained_models[name] = model

        # Cross-validation with wMAE
        wmae_scores = []
        r2_scores = []

        for train_idx, val_idx in cv.split(X_train_scaled):
            X_train_cv, X_val_cv = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_train_cv, y_val_cv = y_train_imputed.iloc[train_idx], y_train_imputed.iloc[val_idx]

            # Train model on CV fold
            fold_model = type(model)(model.estimator)
            fold_model.fit(X_train_cv, y_train_cv)

            # Predict on validation fold
            y_pred_cv = fold_model.predict(X_val_cv)
            y_pred_cv_df = pd.DataFrame(y_pred_cv, columns=target_columns, index=y_val_cv.index)

            # Calculate wMAE
            wmae_score = calculate_wmae(y_val_cv, y_pred_cv_df, wmae_weights, target_columns)
            wmae_scores.append(wmae_score)

            # Calculate R2 for comparison
            r2_score_fold = r2_score(y_val_cv, y_pred_cv, multioutput='uniform_average')
            r2_scores.append(r2_score_fold)

        cv_wmae_scores[name] = np.mean(wmae_scores)
        cv_scores[name] = np.mean(r2_scores)

        print(f"  CV wMAE: {np.mean(wmae_scores):.6f} (±{np.std(wmae_scores):.6f})")
        print(f"  CV R²: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")

    return trained_models, cv_scores, cv_wmae_scores

def make_ensemble_predictions_with_wmae(trained_models, cv_wmae_scores, X_test_scaled):
    """Make ensemble predictions using wMAE-based weights"""
    print("\nMaking predictions with wMAE-based ensemble...")

    # Calculate weights based on inverse wMAE scores (lower wMAE is better)
    weights = {}
    total_weight = 0

    # Use inverse wMAE for weights (add small epsilon to avoid division by zero)
    epsilon = 1e-8
    for name in trained_models.keys():
        weights[name] = 1 / (cv_wmae_scores[name] + epsilon)
        total_weight += weights[name]

    # Normalize weights
    if total_weight > 0:
        for name in weights:
            weights[name] /= total_weight
    else:
        weights = {name: 1/len(trained_models) for name in trained_models.keys()}

    print("Ensemble weights (based on wMAE):")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.3f} (wMAE: {cv_wmae_scores[name]:.6f})")

    # Get predictions from all models
    all_predictions = {}
    for name, model in trained_models.items():
        pred = model.predict(X_test_scaled)
        all_predictions[name] = pred
        print(f"  {name} prediction shape: {pred.shape}")

    # Calculate ensemble prediction
    ensemble_prediction = np.zeros_like(all_predictions[list(trained_models.keys())[0]])
    for name, pred in all_predictions.items():
        ensemble_prediction += weights[name] * pred

    return ensemble_prediction, all_predictions, weights

def create_submission(ensemble_prediction, test, target_columns, y_train_imputed):
    """Create and save submission file with post-processing"""
    print("\nCreating submission...")

    # Create submission DataFrame
    test_predictions_df = pd.DataFrame(ensemble_prediction, columns=target_columns)
    submission = pd.DataFrame({'id': test['id']})
    submission = pd.concat([submission, test_predictions_df], axis=1)

    # Post-processing: clip to reasonable ranges
    print("Applying post-processing...")
    for col in target_columns:
        q1, q99 = np.percentile(y_train_imputed[col], [1, 99])
        submission[col] = np.clip(submission[col], q1, q99)
        print(f"  {col}: clipped to [{q1:.3f}, {q99:.3f}]")

    return submission

def visualize_data_analysis(train, target_columns, wmae_weights):
    """Create comprehensive data analysis visualizations"""
    print("\nCreating data analysis visualizations...")

    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (15, 10)

    # 1. Missing data pattern
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Polymer Property Dataset Analysis', fontsize=16, fontweight='bold')

    # Missing data heatmap
    missing_data = train[target_columns].isnull()
    sns.heatmap(missing_data, ax=axes[0,0], cbar=True, cmap='RdYlBu_r')
    axes[0,0].set_title('Missing Data Pattern')
    axes[0,0].set_xlabel('Properties')

    # Missing data counts
    missing_counts = train[target_columns].isnull().sum()
    axes[0,1].bar(missing_counts.index, missing_counts.values, color='skyblue')
    axes[0,1].set_title('Missing Value Counts')
    axes[0,1].set_xlabel('Properties')
    axes[0,1].set_ylabel('Count')
    axes[0,1].tick_params(axis='x', rotation=45)

    # wMAE weights visualization
    weight_data = pd.Series(wmae_weights)
    axes[0,2].bar(weight_data.index, weight_data.values, color='lightcoral')
    axes[0,2].set_title('wMAE Weights by Property')
    axes[0,2].set_xlabel('Properties')
    axes[0,2].set_ylabel('Weight')
    axes[0,2].tick_params(axis='x', rotation=45)

    # Distribution plots for each property
    for i, col in enumerate(target_columns[:3]):
        if i < 3:
            train[col].dropna().hist(bins=30, ax=axes[1,i], alpha=0.7, color=f'C{i}')
            axes[1,i].set_title(f'{col} Distribution')
            axes[1,i].set_xlabel(col)
            axes[1,i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Property correlation matrix
    plt.figure(figsize=(10, 8))
    corr_matrix = train[target_columns].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Property Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_model_performance(trained_models, cv_scores, cv_wmae_scores, ensemble_weights):
    """Visualize model performance metrics"""
    print("\nCreating model performance visualizations...")

    # Model performance comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # CV R² scores
    model_names = list(cv_scores.keys())
    r2_values = list(cv_scores.values())

    axes[0].bar(model_names, r2_values, color=['skyblue', 'lightgreen', 'coral'])
    axes[0].set_title('Cross-Validation R² Scores')
    axes[0].set_ylabel('R² Score')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(r2_values):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    # CV wMAE scores (lower is better)
    wmae_values = list(cv_wmae_scores.values())
    axes[1].bar(model_names, wmae_values, color=['skyblue', 'lightgreen', 'coral'])
    axes[1].set_title('Cross-Validation wMAE Scores')
    axes[1].set_ylabel('wMAE Score (lower is better)')
    for i, v in enumerate(wmae_values):
        axes[1].text(i, v + max(wmae_values)*0.01, f'{v:.4f}', ha='center', va='bottom')

    # Ensemble weights
    weight_values = list(ensemble_weights.values())
    axes[2].bar(model_names, weight_values, color=['skyblue', 'lightgreen', 'coral'])
    axes[2].set_title('Ensemble Weights (wMAE-based)')
    axes[2].set_ylabel('Weight')
    axes[2].set_ylim(0, 1)
    for i, v in enumerate(weight_values):
        axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_feature_importance(trained_models, feature_names, top_n=15):
    """Visualize feature importance from RandomForest model"""
    if 'RandomForest' not in trained_models:
        return

    print(f"\nVisualizing top {top_n} feature importances...")

    rf_model = trained_models['RandomForest']

    # Get average feature importance across all targets
    importances = np.mean([estimator.feature_importances_ for estimator in rf_model.estimators_], axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance, y='feature', x='importance', palette='viridis')
    plt.title(f'Top {top_n} Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    return feature_importance

def create_interactive_dashboard(train, target_columns, wmae_weights, cv_wmae_scores, ensemble_weights):
    """Create an interactive dashboard using Plotly"""
    print("\nCreating interactive dashboard...")

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Property Distributions', 'wMAE Weights', 'Model Performance', 'Property Correlations'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # 1. Property distributions (box plots)
    for i, col in enumerate(target_columns):
        fig.add_trace(
            go.Box(y=train[col].dropna(), name=col, boxpoints='outliers'),
            row=1, col=1
        )

    # 2. wMAE weights
    fig.add_trace(
        go.Bar(x=list(wmae_weights.keys()), y=list(wmae_weights.values()),
               name='wMAE Weights', marker_color='lightcoral'),
        row=1, col=2
    )

    # 3. Model performance (wMAE scores)
    fig.add_trace(
        go.Bar(x=list(cv_wmae_scores.keys()), y=list(cv_wmae_scores.values()),
               name='CV wMAE', marker_color='lightblue'),
        row=2, col=1
    )

    # 4. Ensemble weights
    fig.add_trace(
        go.Bar(x=list(ensemble_weights.keys()), y=list(ensemble_weights.values()),
               name='Ensemble Weights', marker_color='lightgreen'),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        height=800,
        title_text="Polymer Property Prediction Dashboard",
        showlegend=False
    )

    # Update y-axes titles
    fig.update_yaxes(title_text="Property Values", row=1, col=1)
    fig.update_yaxes(title_text="Weight", row=1, col=2)
    fig.update_yaxes(title_text="wMAE Score", row=2, col=1)
    fig.update_yaxes(title_text="Ensemble Weight", row=2, col=2)

    # Save interactive plot
    fig.write_html("polymer_dashboard.html")
    fig.show()
    print("Interactive dashboard saved as 'polymer_dashboard.html'")

def analyze_prediction_quality(y_true, y_pred, target_columns, wmae_weights):
    """Analyze prediction quality with detailed metrics"""
    print("\nDetailed Prediction Quality Analysis:")
    print("=" * 60)

    # Calculate metrics for each property
    metrics_data = []
    for col in target_columns:
        valid_mask = ~y_true[col].isnull()
        if valid_mask.sum() > 0:
            y_true_col = y_true[col][valid_mask]
            y_pred_col = y_pred[col][valid_mask]

            mae = mean_absolute_error(y_true_col, y_pred_col)
            mse = mean_squared_error(y_true_col, y_pred_col)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_col, y_pred_col)

            # Weighted MAE for this property
            weighted_mae = wmae_weights[col] * mae

            metrics_data.append({
                'Property': col,
                'Count': valid_mask.sum(),
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'wMAE_Weight': wmae_weights[col],
                'Weighted_MAE': weighted_mae
            })

            print(f"{col}:")
            print(f"  Samples: {valid_mask.sum()}")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  wMAE Weight: {wmae_weights[col]:.6f}")
            print(f"  Weighted MAE: {weighted_mae:.6f}")
            print()

    # Calculate overall wMAE
    overall_wmae = calculate_wmae(y_true, y_pred, wmae_weights, target_columns)
    print(f"Overall wMAE: {overall_wmae:.6f}")

    return pd.DataFrame(metrics_data)

def visualize_predictions_vs_actual(y_true, y_pred, target_columns):
    """Create prediction vs actual plots"""
    print("\nCreating prediction vs actual visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for i, col in enumerate(target_columns):
        if i < len(axes):
            valid_mask = ~y_true[col].isnull()
            if valid_mask.sum() > 0:
                y_true_col = y_true[col][valid_mask]
                y_pred_col = y_pred[col][valid_mask]

                # Scatter plot
                axes[i].scatter(y_true_col, y_pred_col, alpha=0.6, s=30)

                # Perfect prediction line
                min_val = min(y_true_col.min(), y_pred_col.min())
                max_val = max(y_true_col.max(), y_pred_col.max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

                # Calculate R²
                r2 = r2_score(y_true_col, y_pred_col)
                axes[i].set_title(f'{col} (R² = {r2:.3f})')
                axes[i].set_xlabel('Actual')
                axes[i].set_ylabel('Predicted')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

    # Remove empty subplot
    if len(target_columns) < len(axes):
        fig.delaxes(axes[-1])

    plt.suptitle('Predicted vs Actual Values', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_report(metrics_df, cv_wmae_scores, ensemble_weights, feature_importance_df):
    """Create a comprehensive performance report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL PERFORMANCE REPORT")
    print("="*80)

    print("\n1. MODEL PERFORMANCE SUMMARY:")
    print("-" * 40)
    for model, score in cv_wmae_scores.items():
        weight = ensemble_weights[model]
        print(f"{model:<15} | wMAE: {score:.6f} | Weight: {weight:.3f}")

    print(f"\nBest single model: {min(cv_wmae_scores, key=cv_wmae_scores.get)}")
    print(f"Best wMAE score: {min(cv_wmae_scores.values()):.6f}")

    print("\n2. PROPERTY-WISE PERFORMANCE:")
    print("-" * 40)
    print(metrics_df.to_string(index=False, float_format='%.4f'))

    print(f"\n3. TOP IMPORTANT FEATURES:")
    print("-" * 40)
    if feature_importance_df is not None:
        print(feature_importance_df.head(10).to_string(index=False, float_format='%.4f'))

    print("\n4. KEY INSIGHTS:")
    print("-" * 40)

    # Find property with highest weight
    max_weight_prop = metrics_df.loc[metrics_df['wMAE_Weight'].idxmax(), 'Property']
    max_weight_val = metrics_df.loc[metrics_df['wMAE_Weight'].idxmax(), 'wMAE_Weight']

    # Find property with best R²
    best_r2_prop = metrics_df.loc[metrics_df['R²'].idxmax(), 'Property']
    best_r2_val = metrics_df.loc[metrics_df['R²'].idxmax(), 'R²']

    print(f"• Highest weighted property: {max_weight_prop} (weight: {max_weight_val:.6f})")
    print(f"• Best predicted property: {best_r2_prop} (R²: {best_r2_val:.3f})")
    print(f"• Ensemble uses {len([w for w in ensemble_weights.values() if w > 0.1])} main models")

    # Data availability insights
    total_samples = metrics_df['Count'].sum()
    avg_availability = metrics_df['Count'].mean()
    print(f"• Average data availability: {avg_availability:.0f} samples per property")

    print("\n" + "="*80)

def main():
    """Main execution function with enhanced wMAE integration"""
    print("Enhanced Polymer Property Prediction with wMAE")
    print("="*60)

    # Load data
    train, test, target_columns = load_and_inspect_data()

    # Calculate wMAE weights using original training data
    print("\nCalculating wMAE weights...")
    wmae_weights = calculate_wmae_weights(train, target_columns)

    # Prepare features
    X_train, X_test = prepare_features(train, test)

    # Clean and impute data
    X_train_imputed, y_train_imputed = clean_and_impute_data(X_train, train, target_columns)

    # Create visualizations for data analysis
    visualize_data_analysis(train, target_columns, wmae_weights)

    # Save imputed training dataset
    imputed_train_data = pd.concat([
        train[['id', 'SMILES']].reset_index(drop=True),
        X_train_imputed.reset_index(drop=True),
        y_train_imputed.reset_index(drop=True)
    ], axis=1)

    imputed_train_data.to_csv('imputed_train_data.csv', index=False)
    print("\nImputed training dataset saved as 'imputed_train_data.csv'")

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train_imputed, X_test)

    # Train models with wMAE evaluation
    trained_models, cv_scores, cv_wmae_scores = train_ensemble_models_with_wmae(
        X_train_scaled, y_train_imputed, wmae_weights, target_columns
    )

    # Make predictions with wMAE-based ensemble
    ensemble_prediction, all_predictions, ensemble_weights = make_ensemble_predictions_with_wmae(
        trained_models, cv_wmae_scores, X_test_scaled
    )

    # Create submission
    submission = create_submission(ensemble_prediction, test, target_columns, y_train_imputed)

    # Visualize model performance
    visualize_model_performance(trained_models, cv_scores, cv_wmae_scores, ensemble_weights)

    # Feature importance analysis
    feature_importance_df = visualize_feature_importance(trained_models, X_train_imputed.columns)

    # Validation analysis (using cross-validation predictions)
    print("\nPerforming validation analysis...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    val_predictions = []
    val_actuals = []

    for train_idx, val_idx in cv.split(X_train_scaled):
        X_val_cv = X_train_scaled[val_idx]
        y_val_cv = y_train_imputed.iloc[val_idx]

        # Use the best single model for validation visualization
        best_model_name = min(cv_wmae_scores, key=cv_wmae_scores.get)
        best_model = trained_models[best_model_name]

        y_pred_val = best_model.predict(X_val_cv)
        y_pred_val_df = pd.DataFrame(y_pred_val, columns=target_columns, index=y_val_cv.index)

        val_predictions.append(y_pred_val_df)
        val_actuals.append(y_val_cv)
        break  # Just use first fold for visualization

    if val_predictions:
        val_pred_combined = val_predictions[0]
        val_actual_combined = val_actuals[0]

        # Analyze prediction quality
        metrics_df = analyze_prediction_quality(
            val_actual_combined, val_pred_combined, target_columns, wmae_weights
        )

        # Visualize predictions vs actual
        visualize_predictions_vs_actual(val_actual_combined, val_pred_combined, target_columns)

        # Create interactive dashboard
        create_interactive_dashboard(train, target_columns, wmae_weights, cv_wmae_scores, ensemble_weights)

        # Generate comprehensive report
        create_comprehensive_report(metrics_df, cv_wmae_scores, ensemble_weights, feature_importance_df)

    # Save files
    submission.to_csv('submission_wmae_optimized.csv', index=False)
    print("\n wMAE-optimized submission saved as 'submission_wmae_optimized.csv'")

    # Save individual model predictions
    for name, pred in all_predictions.items():
        pred_df = pd.DataFrame({'id': test['id']})
        pred_df = pd.concat([pred_df, pd.DataFrame(pred, columns=target_columns)], axis=1)
        pred_df.to_csv(f'predictions_{name.lower()}_wmae.csv', index=False)

    print("Individual model predictions saved with wMAE suffix")

    # Save model performance metrics
    performance_summary = pd.DataFrame({
        'Model': list(cv_wmae_scores.keys()),
        'CV_wMAE': list(cv_wmae_scores.values()),
        'CV_R2': [cv_scores[model] for model in cv_wmae_scores.keys()],
        'Ensemble_Weight': [ensemble_weights[model] for model in cv_wmae_scores.keys()]
    })
    performance_summary.to_csv('model_performance_summary.csv', index=False)
    print(" Model performance summary saved as 'model_performance_summary.csv'")

    # Display final results
    print("\n FINAL RESULTS:")
    print("-" * 30)
    print(f"Best single model wMAE: {min(cv_wmae_scores.values()):.6f}")
    print(f"Ensemble composition: {len([w for w in ensemble_weights.values() if w > 0.1])} main models")
    print("\nSubmission preview:")
    print(submission.head())


# Execute main function
if __name__ == "__main__":
    main()


# Load the existing submission file
submission = pd.read_csv('submission_wmae_optimized.csv')

# Define the desired column order
desired_columns = ['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Reorder the columns
submission_reordered = submission[desired_columns].copy()

# Format IDs in scientific notation (2 decimal places)
submission_reordered['id'] = submission_reordered['id'].apply(lambda x: f"{x:.2E}")

# Save the reordered submission
submission_reordered.to_csv('submission.csv', index=False)

# Show the column order
print(f"\nColumn order: {list(submission_reordered.columns)}")

# Show data types and shape
print(f"Shape: {submission_reordered.shape}")
print(f"Data types:\n{submission_reordered.dtypes}")

# Check for any missing values
print(f"\nMissing values:\n{submission_reordered.isnull().sum()}")

print("\nFinal submission saved as 'submission.csv'")

