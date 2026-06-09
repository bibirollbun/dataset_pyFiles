import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from IPython.display import display
import optuna
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
%matplotlib inline
import os
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_slice
from optuna.visualization import plot_contour, plot_param_importances
from plotly.subplots import make_subplots
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures
import warnings
from tabulate import tabulate
import plotly.graph_objects as go
warnings.filterwarnings('ignore')
import shap
shap.initjs()

print("âœ… Packages loaded!")


def load_data():
    train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col='id')
    test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col='id')
    extra = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
    return train, test, extra

train_df, test_df, extra_df = load_data()

print("ğŸ“Š Data Shapes:")
display(f"Train: {train_df.shape}", f"Test: {test_df.shape}", f"Extra: {extra_df.shape}")
print("\nğŸ”� First 3 rows of training data:")
display(train_df.head(3))


# ======================
# Feature Engineering
# ======================
def feature_engineering(df):
    df['date'] = pd.to_datetime(df['date']) if 'date' in df.columns else pd.date_range(start='2015-01-01', periods=len(df), freq='D')
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['date'].dt.weekday >= 5).astype(int)
    
    # ------------------- Day & Seasonal Features -------------------
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # ------------------- Pressure Features -------------------
    df['pressure_rolling_mean'] = df['pressure'].rolling(window=7, min_periods=1).mean()
    df['pressure_rolling_std'] = df['pressure'].rolling(window=7, min_periods=1).std()
    df['pressure_diff'] = df['pressure'].diff()

    # ------------------- Temperature Features -------------------
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['temp_ewm'] = df['temparature'].ewm(span=10, adjust=False).mean()
    df['temp_change'] = df['temparature'].diff()
    df['temp_humidity_interaction'] = df['temparature'] + (0.2 * df['humidity'])

    # ------------------- Dewpoint & Humidity Features -------------------
    df['dewpoint_depression'] = df['temparature'] - df['dewpoint']
    df['rh_approx'] = 100 - (5 * df['dewpoint_depression'])

    # Saturation Vapor Pressure (SVP) - Tetens' Equation
    df['svp'] = 6.1078 * np.exp((17.27 * df['temparature']) / (df['temparature'] + 237.3))

    # Absolute Humidity (AH) in g/mÂ³
    df['abs_humidity'] = (6.112 * np.exp((17.67 * df['temparature']) / (df['temparature'] + 243.5)) * df['humidity'] * 2.1674) / (273.15 + df['temparature'])

    # ------------------- Cloud & Sunshine Features -------------------
    df['cloud_category'] = pd.cut(df['cloud'], bins=[0, 20, 50, 80, 100], labels=[0, 1, 2, 3])
    df['cloud_category'] = df['cloud_category'].astype(float)
    df['sky_opacity'] = df['cloud'] / 100
    df['sunshine_pct'] = df['sunshine'] / 24
    df['cloud_sun_ratio'] = df['cloud'] / (df['sunshine'] + 1e-6)
    df['interaction'] = df['cloud'] + df['sunshine'] + df['humidity']

    # ------------------- Wind Features -------------------
    df['winddir_sin'] = np.sin(np.radians(df['winddirection']))
    df['winddir_cos'] = np.cos(np.radians(df['winddirection']))

    # ------------------- Polynomial Features -------------------
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = ['temparature', 'humidity', 'pressure', 'windspeed', 'cloud']
    df_poly = pd.DataFrame(poly.fit_transform(df[poly_features]), columns=poly.get_feature_names_out(poly_features))
    df_poly = df_poly.add_prefix("poly_")
    df = df.join(df_poly)

    return df


print("âœ¨ New Features Created!")


# ======================
# Preprocessing Pipeline
# ======================
def create_pipeline(numeric_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    
    return preprocessor




# ======================
# Optuna Objective Function
# ======================
def objective(trial, X, y, preprocessor, model_name):
    solver_penalty_combinations = [
        ('newton-cg', 'l2'),
        ('lbfgs', 'l2'),
        ('liblinear', 'l1'),
        ('liblinear', 'l2'),
        ('sag', 'l2'),
        ('saga', 'elasticnet'),
        ('saga', 'l1'),
        ('saga', 'l2'),
        ('saga', 'none')
    ]
    
    solver, penalty = trial.suggest_categorical('solver_penalty', solver_penalty_combinations)
    
    params = {
        'C': trial.suggest_float('C', 1e-5, 100.0, log=True),
        'solver': solver,
        'penalty': penalty,
        'max_iter': trial.suggest_int('max_iter', 100, 5000),
        'class_weight': trial.suggest_categorical('class_weight', [None]),
        'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'warm_start': trial.suggest_categorical('warm_start', [True, False]),
        'intercept_scaling': trial.suggest_float('intercept_scaling', 0.1, 10.0) if solver == 'liblinear' else 1.0,
        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0) if penalty == 'elasticnet' else None
    }
    
    
    
    # Clean up None values
    params = {k: v for k, v in params.items() if v is not None}
    
    try:
        model = LogisticRegression(**params)
    except ValueError as e:
        return 0.5

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    try:
        score = cross_val_score(pipeline, X, y, cv=TimeSeriesSplit(n_splits=5), 
                              scoring='roc_auc', n_jobs=-1).mean()
    except:
        score = 0.5
        
    return score

def tune_hyperparameters(X, y, preprocessor, model_name):
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X, y, preprocessor, model_name), 
                 n_trials=500, show_progress_bar=True)
    
    if len(study.trials) == 0:
        raise ValueError(f"No completed trials for {model_name}. Check parameter constraints.")
        
    return study.best_params, study







# ======================
# Model Training & Evaluation
# ======================
def train_and_evaluate(X, y, preprocessor):
    model_configs = {
        'LogisticRegression': (LogisticRegression, {'random_state': 42})
    }
    
    results = {}
    best_models = {}
    best_params = {}
    studies = {}

    for model_name, (model_class, base_params) in model_configs.items():
        print(f"\n{'='*40}")
        print(f"Tuning {model_name}")
        print(f"{'='*40}")
        
        try:
            params, study = tune_hyperparameters(X, y, preprocessor, model_name)
        except ValueError as e:
            print(f"Skipping {model_name}: {str(e)}")
            continue
            
        best_params[model_name] = params
        studies[model_name] = study

        # Process parameters
        if 'solver_penalty' in params:
            solver, penalty = params.pop('solver_penalty')
            params['solver'] = solver
            params['penalty'] = penalty

        # Train final model
        final_model = model_class(**{**base_params, **params})
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', final_model)
        ])
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        all_y_true = []
        all_y_pred = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                pipeline.fit(X_train, y_train)
                val_preds = pipeline.predict_proba(X_val)[:, 1]
                
                # Store for aggregate metrics
                all_y_true.extend(y_val.tolist())
                all_y_pred.extend(val_preds.tolist())
                cv_scores.append(roc_auc_score(y_val, val_preds))
                print(f"Fold {fold+1}: {cv_scores[-1]:.4f}")
                
            except Exception as e:
                cv_scores.append(0.5)
                print(f"Fold {fold+1}: Failed, using default score 0.5")

        # Aggregate metrics
        mean_score = np.mean(cv_scores)
        results[model_name] = mean_score
        best_models[model_name] = pipeline
        
        
        print(f"{model_name} Mean ROC-AUC: {mean_score:.4f}")
        #print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    return results, best_models, best_params, studies


# ======================
# Enhanced Visualization Functions
# ======================

def visualize_optuna_studies(studies):
    """Visualize Optuna studies with matplotlib for persistent plots"""
    for model_name, study in studies.items():
        print(f"\n{'#'*40}")
        print(f"Optuna Visualizations for {model_name}")
        print(f"{'#'*40}")
        
        plt.figure(figsize=(15, 10))
        
        # Optimization History
        plt.subplot(2, 2, 1)
        history_df = study.trials_dataframe()
        plt.plot(history_df.number, history_df.value, marker='o', linestyle='--', color='teal')
        plt.title(f'{model_name} Optimization History', fontsize=12)
        plt.xlabel('Trial Number', fontsize=10)
        plt.ylabel('ROC-AUC Score', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Parameter Importances
        plt.subplot(2, 2, 2)
        importances = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title(f'{model_name} Parameter Importances', fontsize=12)
        
        # Slice Plot
        plt.subplot(2, 2, 3)
        slice_plot = optuna.visualization.matplotlib.plot_slice(study)
        plt.title(f'{model_name} Slice Plot', fontsize=12)
        
        # Contour Plot
        plt.subplot(2, 2, 4)
        try:
            contour_plot = optuna.visualization.matplotlib.plot_contour(study)
            plt.title(f'{model_name} Contour Plot', fontsize=12)
        except:
            plt.text(0.5, 0.5, 'No Contour Available', ha='center')
        
        plt.tight_layout()
        plt.savefig(f'optuna_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_confusion_matrix(y_true, y_pred_proba, threshold=0.5):
    """Displays confusion matrix without returning figure"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Rain', 'Rain'], 
                yticklabels=['No Rain', 'Rain'])
    
    plt.title(f'Confusion Matrix (Threshold={threshold:.2f})\nAUC: {roc_auc_score(y_true, y_pred_proba):.4f}', pad=20)
    plt.xlabel('Predicted Label', labelpad=15)
    plt.ylabel('True Label', labelpad=15)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j+0.5, i+0.3, f"{cm_percent[i,j]*100:.1f}%", 
                     ha='center', va='center', color='black')
    
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    """Returns optimal threshold only"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', 
                label=f'Optimal Threshold: {optimal_threshold:.2f}')
    
    plt.title('Receiver Operating Characteristic (ROC)', pad=20)
    plt.xlabel('False Positive Rate', labelpad=15)
    plt.ylabel('True Positive Rate', labelpad=15)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_threshold  

def plot_logistic_predictions(y_true, y_pred_proba, optimal_threshold):
    """Enhanced prediction distribution plot"""
    plt.figure(figsize=(15, 6))
    
    # Create histogram bins
    bins = np.linspace(0, 1, 50)
    plt.hist(y_pred_proba[y_true == 0], bins=bins, alpha=0.7, 
             color='skyblue', label='No Rain Days')
    plt.hist(y_pred_proba[y_true == 1], bins=bins, alpha=0.7, 
             color='salmon', label='Rain Days')
    
    # Threshold line
    plt.axvline(optimal_threshold, color='green', linestyle='--', 
                label=f'Optimal Threshold ({optimal_threshold:.2f})')
    
    plt.title('Prediction Distribution with Optimal Threshold', pad=20)
    plt.xlabel('Predicted Probability', labelpad=15)
    plt.ylabel('Frequency', labelpad=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add density curve
    sns.kdeplot(y_pred_proba[y_true == 0], color='blue', lw=2)
    sns.kdeplot(y_pred_proba[y_true == 1], color='red', lw=2)
    
    plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_coefficient_importance(model, feature_names, top_n=20):
    """Plot logistic regression coefficients"""
    if isinstance(model, Pipeline):
        coefficients = model.named_steps['classifier'].coef_[0]
    else:
        coefficients = model.coef_[0]
        
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values('Coefficient', key=abs, ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Coefficient', y='Feature', data=importance, palette='viridis')
    plt.title(f'Top {top_n} Feature Coefficients (Logistic Regression)')
    plt.xlabel('Coefficient Magnitude')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('logreg_feature_importance.png', dpi=300)
    plt.show()

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def plot_shap_importance(pipeline, X, sample_size=500):
    """Proper SHAP visualization for logistic regression"""
    try:
        print("ğŸ”� Calculating SHAP Values...")
        
        # Extract preprocessing steps and model
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['classifier']
        
        # Transform data through preprocessing
        X_processed = preprocessor.transform(X)
        
        # Handle feature names correctly
        numeric_transformer = preprocessor.named_transformers_['num']
        
        if hasattr(numeric_transformer, 'get_feature_names_out'):
            numeric_features = numeric_transformer.get_feature_names_out()
        else:
            numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        
        # Convert transformed data to DataFrame
        X_processed = pd.DataFrame(X_processed, columns=numeric_features)

        # Sample data for faster computation
        sample_idx = np.random.choice(X_processed.shape[0], size=min(sample_size, len(X_processed)), replace=False)
        X_sample = X_processed.iloc[sample_idx]

        # Create SHAP explainer
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

        # Plot summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=numeric_features, plot_type='bar', show=False)
        plt.title("SHAP Feature Importance (Logistic Regression)", fontsize=14)
        plt.tight_layout()
        plt.savefig('shap_feature_importance.png', dpi=300)
        plt.show()

        return explainer

    except Exception as e:
        print(f"â�Œ SHAP Error: {str(e)}")
        return None




# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    # Load and preprocess data
    train, test, extra = load_data()
    
    # Process extra data
    extra.columns = extra.columns.str.replace(' ', '')
    extra['rainfall'] = extra['rainfall'].map({'no': 0, 'yes': 1})
    extra = extra.dropna().reset_index(drop=True)
    
    # Merge datasets
    train['source'] = 'main'
    extra['source'] = 'extra'
    test['source'] = 'test'
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Extra shape: {extra.shape}")
    
    full_data = pd.concat([train, extra, test], axis=0).reset_index(drop=True)
    
    # Feature engineering
    print("\nğŸ”§ Applying Feature Engineering...")
    full_data = feature_engineering(full_data)
    
    
    # Split data
    train_data = full_data[full_data['source'] != 'test']  # Exclude test data
    test_data = full_data[full_data['source'] == 'test']  # Test data remains untouched
    
    # Further split the training data into train and validation sets
    train_data, val_data = train_test_split(
        train_data, 
        test_size=0.1,  # 10% of the training data will be used for validation
        random_state=42,  # Set a random seed for reproducibility
        stratify=train_data['rainfall']  # Ensure stratified split based on the target variable
    )
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Define features
    numeric_features = [
        'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
        'humidity', 'cloud', 'sunshine', 'windspeed', 'day_of_year',
        'month', 'is_weekend', 'day_sin', 'day_cos', 'pressure_rolling_mean',
        'pressure_rolling_std', 'pressure_diff', 'temp_range', 'temp_ewm',
        'temp_change', 'temp_humidity_interaction', 'dewpoint_depression',
        'rh_approx', 'svp', 'abs_humidity', 'sky_opacity', 'sunshine_pct',
        'cloud_sun_ratio', 'interaction', 'winddir_sin', 'winddir_cos',
    ] + [col for col in full_data.columns if col.startswith('poly_')]
    
    # Prepare features and target
    X_train = train_data[numeric_features]
    y_train = train_data['rainfall']
    
    X_val = val_data[numeric_features]
    y_val = val_data['rainfall']
    
    X_test = test_data[numeric_features]
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Validation features shape: {X_val.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    # Create preprocessing pipeline
    preprocessor = create_pipeline(numeric_features)
    
    # Train models
    results, models, best_params, studies = train_and_evaluate(X_train, y_train, preprocessor)
    
    # Print best hyperparameters
    print("\n\n" + "="*60)
    print("Best Hyperparameters")
    print("="*60)
    for model_name, params in best_params.items():
        print(f"\n{model_name}:")
        if 'solver_penalty' in params:
            solver, penalty = params.pop('solver_penalty')
            params['solver'] = solver
            params['penalty'] = penalty
        for param, value in params.items():
            if isinstance(value, float):
                print(f"  {param:20} = {value:.6f}")
            else:
                print(f"  {param:20} = {value}")
    print("="*60)
    
    # Generate predictions and visualize
    if models:
        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]
        
        # Plot coefficients
        print("\nğŸ”� Plotting Logistic Regression Coefficients...")
        preprocessor = best_model.named_steps['preprocessor']
        numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out().tolist()
        plot_coefficient_importance(best_model, numeric_features)
        
        # Plot SHAP values
        print("\nğŸ”� Calculating SHAP Values...")
        explainer = plot_shap_importance(best_model, X_train)  # Passing X_train as a DataFrame
        
        best_score = -1
        best_model_name = ''
        
        for model_name, model in models.items():
            print(f"\nGenerating predictions with {model_name}")
            try:
                # Get validation predictions
                val_probs = model.predict_proba(X_val)[:, 1]
                
                # Calculate optimal threshold
                print("\nCalculating Optimal Threshold...")
                optimal_threshold = plot_roc_curve(y_val, val_probs)  # Now single return value
                
                # Generate test predictions
                test_probs = model.predict_proba(X_test)[:, 1]
                
                # Visualize predictions
                print("\nVisualizing Results...")
                plot_confusion_matrix(y_val, val_probs, optimal_threshold)
                plot_logistic_predictions(y_val, val_probs, optimal_threshold)
                
                # Save predictions
                submission = pd.DataFrame({'id': range(2190, 2190 + len(test_probs)), 'rainfall': test_probs})
                submission.to_csv(f'submission_{model_name}.csv', index=False)
                print(f"Saved submission_{model_name}.csv")
                
                if results[model_name] > best_score:
                    best_score = results[model_name]
                    best_model_name = model_name
                    best_probs = test_probs
            except Exception as e:
                print(f"Error generating predictions for {model_name}: {str(e)}")

        # Save best model
        if best_model_name:
            joblib.dump(models[best_model_name], 'best_model.pkl')
            print(f"\nSaved best model ({best_model_name}) as best_model.pkl")
    else:
        print("\nNo valid models trained. Check error messages above.")


from sklearn.model_selection import cross_val_score, StratifiedKFold
# Importing the necessary libraries for the models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    # Load and preprocess data
    train, test, extra = load_data()
    
    # Process extra data
    extra.columns = extra.columns.str.replace(' ', '')
    extra['rainfall'] = extra['rainfall'].map({'no': 0, 'yes': 1})
    extra = extra.dropna().reset_index(drop=True)
    
    # Merge datasets
    train['source'] = 'main'
    extra['source'] = 'extra'
    test['source'] = 'test'
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Extra shape: {extra.shape}")
    
    full_data = pd.concat([train, extra, test], axis=0).reset_index(drop=True)
    
    # Feature engineering
    print("\nğŸ”§ Applying Feature Engineering...")
    full_data = feature_engineering(full_data)
    
    # Split data
    train_data = full_data[full_data['source'] != 'test']  # Exclude test data
    test_data = full_data[full_data['source'] == 'test']  # Test data remains untouched
    
    # Prepare features and target
    numeric_features = [
        'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
        'humidity', 'cloud', 'sunshine', 'windspeed', 'day_of_year',
        'month', 'is_weekend', 'day_sin', 'day_cos', 'pressure_rolling_mean',
        'pressure_rolling_std', 'pressure_diff', 'temp_range', 'temp_ewm',
        'temp_change', 'temp_humidity_interaction', 'dewpoint_depression',
        'rh_approx', 'svp', 'abs_humidity', 'sky_opacity', 'sunshine_pct',
        'cloud_sun_ratio', 'interaction', 'winddir_sin', 'winddir_cos',
    ] + [col for col in full_data.columns if col.startswith('poly_')]
    
    X_train = train_data[numeric_features]
    y_train = train_data['rainfall']
    
    X_test = test_data[numeric_features]
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    # Create preprocessing pipeline
    preprocessor = create_pipeline(numeric_features)
    
    # Define models with default parameters
    models = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'XGBClassifier': XGBClassifier(random_state=42, eval_metric='logloss'),
        'LGBMClassifier': LGBMClassifier(random_state=42),
    }
    
    # Train and evaluate models using cross-validation
    results = {}
    for model_name, model in models.items():
        print(f"\nTraining {model_name} with cross-validation...")
        try:
            # Create a pipeline with preprocessor and model
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Define cross-validation strategy
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold stratified CV
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                pipeline, 
                X_train, 
                y_train, 
                cv=cv, 
                scoring='roc_auc',  # Use ROC-AUC as the evaluation metric
                n_jobs=-1  # Use all available CPU cores
            )
            
            # Store the mean and standard deviation of the CV scores
            results[model_name] = {
                'mean_roc_auc': cv_scores.mean(),
                'std_roc_auc': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"{model_name} cross-validation ROC-AUC scores: {cv_scores}")
            print(f"{model_name} mean ROC-AUC: {cv_scores.mean():.4f} (Â±{cv_scores.std():.4f})")
            
            # Train the model on the full training data
            pipeline.fit(X_train, y_train)
            
            # Save the trained model
            joblib.dump(pipeline, f'{model_name}_model.pkl')
            print(f"Saved {model_name}_model.pkl")
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
    
    # Print model results
    print("\n\n" + "="*60)
    print("Cross-Validation Results")
    print("="*60)
    for model_name, scores in results.items():
        print(f"{model_name}:")
        print(f"  Mean ROC-AUC: {scores['mean_roc_auc']:.4f}")
        print(f"  Std ROC-AUC: {scores['std_roc_auc']:.4f}")
        print(f"  CV Scores: {scores['cv_scores']}")
    print("="*60)
    
    # Generate predictions with the best model
    if results:
        best_model_name = max(results, key=lambda x: results[x]['mean_roc_auc'])
        print(f"\nBest model: {best_model_name} (Mean ROC-AUC = {results[best_model_name]['mean_roc_auc']:.4f})")
        
        # Load the best model
        best_model = joblib.load(f'{best_model_name}_model.pkl')
        
        # Generate test predictions
        test_probs = best_model.predict_proba(X_test)[:, 1]
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'id': range(2190, 2190 + len(test_probs)),  # Start id from 2190 and increment by 1
            'rainfall': test_probs  # Predicted probabilities
        })
        
        # Save submission file
        submission.to_csv('submissionwithout_Tunning.csv', index=False)
        print("Submission file saved as 'submission.csv'")
        
        # Print the first few rows of the submission file for verification
        print("\nSubmission file preview:")
        print(submission.head())
    else:
        print("\nNo valid models trained. Check error messages above.")


pd.read_csv("/kaggle/working/submission_LogisticRegression.csv")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from IPython.display import display
import optuna
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
%matplotlib inline
import os
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_slice
from optuna.visualization import plot_contour, plot_param_importances
from plotly.subplots import make_subplots
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures
import warnings
from tabulate import tabulate
import plotly.graph_objects as go
warnings.filterwarnings('ignore')
import shap
shap.initjs()

print("âœ… Packages loaded!")

def load_data():
    train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col='id')
    test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col='id')
    extra = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
    return train, test, extra

train_df, test_df, extra_df = load_data()

print("ğŸ“Š Data Shapes:")
display(f"Train: {train_df.shape}", f"Test: {test_df.shape}", f"Extra: {extra_df.shape}")
print("\nğŸ”� First 3 rows of training data:")
display(train_df.head(3))

# ======================
# Enhanced Feature Engineering
# ======================
def engineer_features(df):
    """Create comprehensive meteorological features"""
    enhanced_df = df.copy()
    
    # Date features
    enhanced_df['date'] = pd.to_datetime(enhanced_df['date']) if 'date' in enhanced_df.columns else pd.date_range(start='2015-01-01', periods=len(enhanced_df), freq='D')
    enhanced_df['day_of_year'] = enhanced_df['date'].dt.dayofyear
    enhanced_df['month'] = enhanced_df['date'].dt.month
    enhanced_df['is_weekend'] = (enhanced_df['date'].dt.weekday >= 5).astype(int)
    
    # Cyclical date features
    enhanced_df['day_sin'] = np.sin(2 * np.pi * enhanced_df['day_of_year'] / 365)
    enhanced_df['day_cos'] = np.cos(2 * np.pi * enhanced_df['day_of_year'] / 365)
    
    # Temperature features
    enhanced_df['temp_range'] = enhanced_df['maxtemp'] - enhanced_df['mintemp']
    enhanced_df['temp_ewm'] = enhanced_df['temparature'].ewm(span=10, adjust=False).mean()
    enhanced_df['temp_change'] = enhanced_df['temparature'].diff()
    enhanced_df['temp_humidity_interaction'] = enhanced_df['temparature'] + (0.2 * enhanced_df['humidity'])
    
    # Dew point calculations
    enhanced_df['dewpoint_depression'] = enhanced_df['temparature'] - enhanced_df['dewpoint']
    enhanced_df['rh_approx'] = 100 - (5 * enhanced_df['dewpoint_depression'])
    
    # Advanced humidity calculations
    enhanced_df['svp'] = 6.1078 * np.exp((17.27 * enhanced_df['temparature']) / (enhanced_df['temparature'] + 237.3))
    enhanced_df['abs_humidity'] = (6.112 * np.exp((17.67 * enhanced_df['temparature']) / (enhanced_df['temparature'] + 243.5)) * enhanced_df['humidity'] * 2.1674) / (273.15 + enhanced_df['temparature'])
    
    # Cloud/sun features
    enhanced_df['cloud_category'] = pd.cut(enhanced_df['cloud'], bins=[0, 20, 50, 80, 100], labels=[0, 1, 2, 3]).astype(float)
    enhanced_df['sky_opacity'] = enhanced_df['cloud'] / 100
    enhanced_df['sunshine_pct'] = enhanced_df['sunshine'] / 24
    enhanced_df['cloud_sun_ratio'] = enhanced_df['cloud'] / (enhanced_df['sunshine'] + 1e-6)
    
    # Wind features
    enhanced_df['winddir_sin'] = np.sin(np.radians(enhanced_df['winddirection']))
    enhanced_df['winddir_cos'] = np.cos(np.radians(enhanced_df['winddirection']))
    
    # Pressure dynamics
    enhanced_df['pressure_diff'] = enhanced_df['pressure'].diff().fillna(0)
    enhanced_df['pressure_acceleration'] = enhanced_df['pressure_diff'].diff().fillna(0)
    
    # Enhanced interactions
    enhanced_df['wind_humidity_factor'] = enhanced_df['windspeed'] * (enhanced_df['humidity'] / 100)
    enhanced_df['temp_humidity_index'] = (0.8 * enhanced_df['temparature']) + ((enhanced_df['humidity'] / 100) * (enhanced_df['temparature'] - 14.3)) + 46.4
    
    # Rolling features
    for window in [3, 7, 14]:
        for col in ['temparature', 'pressure', 'humidity', 'cloud', 'windspeed']:
            enhanced_df[f'{col}_rolling_{window}d'] = enhanced_df[col].rolling(window=window, min_periods=1).mean()
            if window in [7, 14]:
                enhanced_df[f'{col}_std_{window}d'] = enhanced_df[col].rolling(window=window, min_periods=4).std().fillna(0)
    
    # Trend features
    for col in ['temparature', 'pressure', 'humidity']:
        enhanced_df[f'{col}_trend_3d'] = enhanced_df[col].diff(3).fillna(0)
    
    # Extreme value indicators
    for col in ['temparature', 'humidity', 'pressure']:
        q_high = enhanced_df[col].quantile(0.95)
        q_low = enhanced_df[col].quantile(0.05)
        enhanced_df[f'extreme_{col}'] = ((enhanced_df[col] > q_high) | (enhanced_df[col] < q_low)).astype(int)
    
    # Polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = ['temparature', 'humidity', 'pressure', 'windspeed', 'cloud']
    df_poly = pd.DataFrame(poly.fit_transform(enhanced_df[poly_features]), columns=poly.get_feature_names_out(poly_features))
    df_poly = df_poly.add_prefix("poly_")
    enhanced_df = pd.concat([enhanced_df, df_poly], axis=1)
    
    return enhanced_df

print("âœ¨ Advanced Features Created!")

# ======================
# Preprocessing Pipeline
# ======================
def create_pipeline(numeric_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    
    return preprocessor


# ======================
# Optuna Objective Function
# ======================
def objective(trial, X, y, preprocessor, model_name):
    solver_penalty_combinations = [
        ('newton-cg', 'l2'),
        ('lbfgs', 'l2'),
        ('liblinear', 'l1'),
        ('liblinear', 'l2'),
        ('sag', 'l2'),
        ('saga', 'elasticnet'),
        ('saga', 'l1'),
        ('saga', 'l2'),
        ('saga', 'none')
    ]
    
    solver, penalty = trial.suggest_categorical('solver_penalty', solver_penalty_combinations)
    
    params = {
        'C': trial.suggest_float('C', 1e-5, 100.0, log=True),
        'solver': solver,
        'penalty': penalty,
        'max_iter': trial.suggest_int('max_iter', 100, 5000),
        'class_weight': trial.suggest_categorical('class_weight', [None]),
        'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'warm_start': trial.suggest_categorical('warm_start', [True, False]),
        'intercept_scaling': trial.suggest_float('intercept_scaling', 0.1, 10.0) if solver == 'liblinear' else 1.0,
        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0) if penalty == 'elasticnet' else None
    }
    
    
    
    # Clean up None values
    params = {k: v for k, v in params.items() if v is not None}
    
    try:
        model = LogisticRegression(**params)
    except ValueError as e:
        return 0.5

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    try:
        score = cross_val_score(pipeline, X, y, cv=TimeSeriesSplit(n_splits=5), 
                              scoring='roc_auc', n_jobs=-1).mean()
    except:
        score = 0.5
        
    return score

def tune_hyperparameters(X, y, preprocessor, model_name):
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X, y, preprocessor, model_name), 
                 n_trials=500, show_progress_bar=True)
    
    if len(study.trials) == 0:
        raise ValueError(f"No completed trials for {model_name}. Check parameter constraints.")
        
    return study.best_params, study




# ======================
# Model Training & Evaluation
# ======================
def train_and_evaluate(X, y, preprocessor):
    model_configs = {
        'LogisticRegression': (LogisticRegression, {'random_state': 42})
    }
    
    results = {}
    best_models = {}
    best_params = {}
    studies = {}

    for model_name, (model_class, base_params) in model_configs.items():
        print(f"\n{'='*40}")
        print(f"Tuning {model_name}")
        print(f"{'='*40}")
        
        try:
            params, study = tune_hyperparameters(X, y, preprocessor, model_name)
        except ValueError as e:
            print(f"Skipping {model_name}: {str(e)}")
            continue
            
        best_params[model_name] = params
        studies[model_name] = study

        # Process parameters
        if 'solver_penalty' in params:
            solver, penalty = params.pop('solver_penalty')
            params['solver'] = solver
            params['penalty'] = penalty

        # Train final model
        final_model = model_class(**{**base_params, **params})
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', final_model)
        ])
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        all_y_true = []
        all_y_pred = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                pipeline.fit(X_train, y_train)
                val_preds = pipeline.predict_proba(X_val)[:, 1]
                
                # Store for aggregate metrics
                all_y_true.extend(y_val.tolist())
                all_y_pred.extend(val_preds.tolist())
                cv_scores.append(roc_auc_score(y_val, val_preds))
                print(f"Fold {fold+1}: {cv_scores[-1]:.4f}")
                
            except Exception as e:
                cv_scores.append(0.5)
                print(f"Fold {fold+1}: Failed, using default score 0.5")

        # Aggregate metrics
        mean_score = np.mean(cv_scores)
        results[model_name] = mean_score
        best_models[model_name] = pipeline
        
        
        print(f"{model_name} Mean ROC-AUC: {mean_score:.4f}")
        #print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    return results, best_models, best_params, studies


# ======================
# Enhanced Visualization Functions
# ======================

def visualize_optuna_studies(studies):
    """Visualize Optuna studies with matplotlib for persistent plots"""
    for model_name, study in studies.items():
        print(f"\n{'#'*40}")
        print(f"Optuna Visualizations for {model_name}")
        print(f"{'#'*40}")
        
        plt.figure(figsize=(15, 10))
        
        # Optimization History
        plt.subplot(2, 2, 1)
        history_df = study.trials_dataframe()
        plt.plot(history_df.number, history_df.value, marker='o', linestyle='--', color='teal')
        plt.title(f'{model_name} Optimization History', fontsize=12)
        plt.xlabel('Trial Number', fontsize=10)
        plt.ylabel('ROC-AUC Score', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Parameter Importances
        plt.subplot(2, 2, 2)
        importances = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title(f'{model_name} Parameter Importances', fontsize=12)
        
        # Slice Plot
        plt.subplot(2, 2, 3)
        slice_plot = optuna.visualization.matplotlib.plot_slice(study)
        plt.title(f'{model_name} Slice Plot', fontsize=12)
        
        # Contour Plot
        plt.subplot(2, 2, 4)
        try:
            contour_plot = optuna.visualization.matplotlib.plot_contour(study)
            plt.title(f'{model_name} Contour Plot', fontsize=12)
        except:
            plt.text(0.5, 0.5, 'No Contour Available', ha='center')
        
        plt.tight_layout()
        plt.savefig(f'optuna_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_confusion_matrix(y_true, y_pred_proba, threshold=0.5):
    """Displays confusion matrix without returning figure"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Rain', 'Rain'], 
                yticklabels=['No Rain', 'Rain'])
    
    plt.title(f'Confusion Matrix (Threshold={threshold:.2f})\nAUC: {roc_auc_score(y_true, y_pred_proba):.4f}', pad=20)
    plt.xlabel('Predicted Label', labelpad=15)
    plt.ylabel('True Label', labelpad=15)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j+0.5, i+0.3, f"{cm_percent[i,j]*100:.1f}%", 
                     ha='center', va='center', color='black')
    
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    """Returns optimal threshold only"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', 
                label=f'Optimal Threshold: {optimal_threshold:.2f}')
    
    plt.title('Receiver Operating Characteristic (ROC)', pad=20)
    plt.xlabel('False Positive Rate', labelpad=15)
    plt.ylabel('True Positive Rate', labelpad=15)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_threshold  

def plot_logistic_predictions(y_true, y_pred_proba, optimal_threshold):
    """Enhanced prediction distribution plot"""
    plt.figure(figsize=(15, 6))
    
    # Create histogram bins
    bins = np.linspace(0, 1, 50)
    plt.hist(y_pred_proba[y_true == 0], bins=bins, alpha=0.7, 
             color='skyblue', label='No Rain Days')
    plt.hist(y_pred_proba[y_true == 1], bins=bins, alpha=0.7, 
             color='salmon', label='Rain Days')
    
    # Threshold line
    plt.axvline(optimal_threshold, color='green', linestyle='--', 
                label=f'Optimal Threshold ({optimal_threshold:.2f})')
    
    plt.title('Prediction Distribution with Optimal Threshold', pad=20)
    plt.xlabel('Predicted Probability', labelpad=15)
    plt.ylabel('Frequency', labelpad=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add density curve
    sns.kdeplot(y_pred_proba[y_true == 0], color='blue', lw=2)
    sns.kdeplot(y_pred_proba[y_true == 1], color='red', lw=2)
    
    plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_coefficient_importance(model, feature_names, top_n=20):
    """Plot logistic regression coefficients"""
    if isinstance(model, Pipeline):
        coefficients = model.named_steps['classifier'].coef_[0]
    else:
        coefficients = model.coef_[0]
        
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values('Coefficient', key=abs, ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Coefficient', y='Feature', data=importance, palette='viridis')
    plt.title(f'Top {top_n} Feature Coefficients (Logistic Regression)')
    plt.xlabel('Coefficient Magnitude')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('logreg_feature_importance.png', dpi=300)
    plt.show()



def plot_shap_importance(pipeline, X, sample_size=500):
    """Proper SHAP visualization for logistic regression"""
    try:
        print("ğŸ”� Calculating SHAP Values...")
        
        # Extract preprocessing steps and model
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['classifier']
        
        # Transform data through preprocessing
        X_processed = preprocessor.transform(X)
        
        # Handle feature names correctly
        numeric_transformer = preprocessor.named_transformers_['num']
        
        if hasattr(numeric_transformer, 'get_feature_names_out'):
            numeric_features = numeric_transformer.get_feature_names_out()
        else:
            numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        
        # Convert transformed data to DataFrame
        X_processed = pd.DataFrame(X_processed, columns=numeric_features)

        # Sample data for faster computation
        sample_idx = np.random.choice(X_processed.shape[0], size=min(sample_size, len(X_processed)), replace=False)
        X_sample = X_processed.iloc[sample_idx]

        # Create SHAP explainer
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

        # Plot summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=numeric_features, plot_type='bar', show=False)
        plt.title("SHAP Feature Importance (Logistic Regression)", fontsize=14)
        plt.tight_layout()
        plt.savefig('shap_feature_importance.png', dpi=300)
        plt.show()

        return explainer

    except Exception as e:
        print(f"â�Œ SHAP Error: {str(e)}")
        return None




from sklearn.model_selection import cross_val_score, StratifiedKFold
# Importing the necessary libraries for the models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier





# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    # Load and preprocess data
    train, test, extra = load_data()
    
    # Process extra data
    extra.columns = extra.columns.str.replace(' ', '')
    extra['rainfall'] = extra['rainfall'].map({'no': 0, 'yes': 1})
    extra = extra.dropna().reset_index(drop=True)
    
    # Merge datasets
    train['source'] = 'main'
    extra['source'] = 'extra'
    test['source'] = 'test'
    
    full_data = pd.concat([train, extra, test], axis=0).reset_index(drop=True)
    
    # Feature engineering
    print("\nğŸ”§ Applying Advanced Feature Engineering...")
    full_data = engineer_features(full_data)
    
    # Split data
    train_data = full_data[full_data['source'] != 'test']
    test_data = full_data[full_data['source'] == 'test']
    
    # Train/validation split
    train_data, val_data = train_test_split(
        train_data, 
        test_size=0.1,
        random_state=42,
        stratify=train_data['rainfall']
    )
    
    # Define features (updated with new features)
    numeric_features = [
        'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
        'humidity', 'cloud', 'sunshine', 'windspeed', 'day_of_year',
        'month', 'is_weekend', 'day_sin', 'day_cos', 'temp_range',
        'dewpoint_depression', 'rh_approx', 'svp', 'abs_humidity',
        'sky_opacity', 'sunshine_pct', 'cloud_sun_ratio', 'winddir_sin',
        'winddir_cos', 'pressure_diff', 'pressure_acceleration',
        'wind_humidity_factor', 'temp_humidity_index'
    ] + [col for col in full_data.columns if col.startswith(('poly_', '_rolling_', '_trend_', 'extreme_'))]
    
    # Prepare features and target
    X_train = train_data[numeric_features]
    y_train = train_data['rainfall']
    X_val = val_data[numeric_features]
    y_val = val_data['rainfall']
    X_test = test_data[numeric_features]
    
    # Create preprocessing pipeline
    preprocessor = create_pipeline(numeric_features)
    
    # Define models with default parameters
    models = {
        'LogisticRegression': LogisticRegression(random_state=42),
        
    }
    
    # Train and evaluate models using cross-validation
    results = {}
    for model_name, model in models.items():
        print(f"\nTraining {model_name} with cross-validation...")
        try:
            # Create a pipeline with preprocessor and model
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Define cross-validation strategy
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold stratified CV
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                pipeline, 
                X_train, 
                y_train, 
                cv=cv, 
                scoring='roc_auc',  # Use ROC-AUC as the evaluation metric
                n_jobs=-1  # Use all available CPU cores
            )
            
            # Store the mean and standard deviation of the CV scores
            results[model_name] = {
                'mean_roc_auc': cv_scores.mean(),
                'std_roc_auc': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"{model_name} cross-validation ROC-AUC scores: {cv_scores}")
            print(f"{model_name} mean ROC-AUC: {cv_scores.mean():.4f} (Â±{cv_scores.std():.4f})")
            
            # Train the model on the full training data
            pipeline.fit(X_train, y_train)
            
            # Save the trained model
            joblib.dump(pipeline, f'NewFeature_without_Tunning{model_name}_model.pkl')
            print(f"Saved {model_name}_model.pkl")
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
    
    # Print model results
    print("\n\n" + "="*60)
    print("Cross-Validation Results")
    print("="*60)
    for model_name, scores in results.items():
        print(f"{model_name}:")
        print(f"  Mean ROC-AUC: {scores['mean_roc_auc']:.4f}")
        print(f"  Std ROC-AUC: {scores['std_roc_auc']:.4f}")
        print(f"  CV Scores: {scores['cv_scores']}")
    print("="*60)
    
    # Generate predictions with the best model
    if results:
        best_model_name = max(results, key=lambda x: results[x]['mean_roc_auc'])
        print(f"\nBest model: {best_model_name} (Mean ROC-AUC = {results[best_model_name]['mean_roc_auc']:.4f})")
        
        # Load the best model
        best_model = joblib.load(f'NewFeature_without_Tunning{best_model_name}_model.pkl')
        
        
        # Generate test predictions
        test_probs = best_model.predict_proba(X_test)[:, 1]
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'id': range(2190, 2190 + len(test_probs)),  # Start id from 2190 and increment by 1
            'rainfall': test_probs  # Predicted probabilities
        })
        
        # Save submission file
        submission.to_csv('New_Fetaure_submissionwithout_Tunning.csv', index=False)
        print("Submission file saved as 'submission.csv'")
        
        # Print the first few rows of the submission file for verification
        print("\nSubmission file preview:")
        print(submission.head())
    else:
        print("\nNo valid models trained. Check error messages above.")


