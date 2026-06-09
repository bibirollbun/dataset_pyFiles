!pip install textstat


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #, StratifiedKFold
from sklearn.linear_model import Ridge
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer
import textstat  # pip install textstat
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Load better embedding model
model = SentenceTransformer('all-mpnet-base-v2')

# Load data
df_test = pd.read_csv('/kaggle/input/internal-selection-bdc-2025-its/df_test.csv')
df_train = pd.read_csv('/kaggle/input/internal-selection-bdc-2025-its/df_train.csv')

# Drop missing values
#df_train.dropna(subset=['task_achievement', 'coherence_and_cohesion', 
#                       'lexical_resource', 'grammatical_range'], inplace=True)
df_train.dropna(subset=['task_achievement'], inplace=True)
df_train.dropna(subset=['coherence_and_cohesion'], inplace=True)
df_train.dropna(subset=['lexical_resource'], inplace=True)
df_train.dropna(subset=['grammatical_range'], inplace=True)

print(f"Training data shape: {df_train.shape}")

def calculate_mean_columnwise_rmse(y_true, y_pred):
    """
    Calculate Mean Columnwise RMSE
    For each column/target, calculate RMSE, then take the mean across all columns
    """
    if isinstance(y_true, pd.DataFrame):
        # If DataFrame, calculate RMSE for each column
        column_rmses = []
        for col in y_true.columns:
            rmse = np.sqrt(mean_squared_error(y_true[col], y_pred[col]))
            column_rmses.append(rmse)
        return np.mean(column_rmses), column_rmses
    else:
        # If single column/array
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse, [rmse]

def calculate_columnwise_rmse(y_true, y_pred, column_names=None):
    """
    Calculate RMSE for each column individually
    Returns dictionary of column_name -> RMSE
    """
    if isinstance(y_true, pd.DataFrame):
        column_rmses = {}
        for col in y_true.columns:
            rmse = np.sqrt(mean_squared_error(y_true[col], y_pred[col]))
            column_rmses[col] = rmse
    else:
        # Handle case where we have arrays and column names
        if column_names is None:
            column_names = [f'target_{i}' for i in range(y_true.shape[1] if len(y_true.shape) > 1 else 1)]
        
        column_rmses = {}
        if len(y_true.shape) > 1:  # Multiple columns
            for i, col_name in enumerate(column_names):
                rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
                column_rmses[col_name] = rmse
        else:  # Single column
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            column_rmses[column_names[0]] = rmse
    
    return column_rmses

def minimal_clean_text(text):
    """Minimal cleaning - preserve grammar structure"""
    if not isinstance(text, str):
        return ""
    # Only basic cleaning - keep stop words and punctuation
    return text.strip()

def calculate_advanced_features(df):
    """Calculate comprehensive writing quality features"""
    
    features_list = []
    
    for idx, row in df.iterrows():
        essay = str(row['essay'])
        prompt = str(row['prompt'])
        
        # Basic length features
        essay_length = len(essay)
        essay_word_count = len(essay.split())
        prompt_length = len(prompt.split())
        essay_prompt_ratio = essay_word_count / max(prompt_length, 1)
        
        # Vocabulary diversity
        words = word_tokenize(essay.lower())
        unique_words = len(set(words))
        ttr = unique_words / max(len(words), 1)
        
        # Readability scores (crucial for writing assessment)
        flesch_ease = textstat.flesch_reading_ease(essay)
        flesch_kincaid = textstat.flesch_kincaid_grade(essay)
        gunning_fog = textstat.gunning_fog(essay)
        coleman_liau = textstat.coleman_liau_index(essay)
        automated_readability = textstat.automated_readability_index(essay)
        
        # Sentence-level features
        sentences = sent_tokenize(essay)
        sentence_count = len(sentences)
        avg_sentence_length = essay_word_count / max(sentence_count, 1)
        
        # Syllable and complexity
        syllable_count = textstat.syllable_count(essay)
        avg_syllables_per_word = syllable_count / max(essay_word_count, 1)
        difficult_words = textstat.difficult_words(essay)
        
        # Paragraph structure
        paragraph_count = len([p for p in essay.split('\n') if p.strip()])
        
        features = [
            essay_length, essay_word_count, essay_prompt_ratio,
            unique_words, ttr, flesch_ease, flesch_kincaid,
            gunning_fog, coleman_liau, automated_readability,
            sentence_count, avg_sentence_length, syllable_count,
            avg_syllables_per_word, difficult_words, paragraph_count
        ]
        features_list.append(features)
    
    feature_names = [
        'essay_length', 'essay_word_count', 'essay_prompt_ratio',
        'unique_words', 'ttr', 'flesch_ease', 'flesch_kincaid',
        'gunning_fog', 'coleman_liau', 'automated_readability',
        'sentence_count', 'avg_sentence_length', 'syllable_count',
        'avg_syllables_per_word', 'difficult_words', 'paragraph_count'
    ]
    
    return pd.DataFrame(features_list, columns=feature_names, index=df.index)

# Calculate advanced features
print("Calculating advanced writing features...")
train_features = calculate_advanced_features(df_train)
test_features = calculate_advanced_features(df_test)

# Add features to dataframes
for col in train_features.columns:
    df_train[col] = train_features[col]
    df_test[col] = test_features[col]

print("Advanced features calculated!")
print(train_features.head())

# Create full text with minimal cleaning
df_train['essay_minimal'] = df_train['essay'].apply(minimal_clean_text)
df_test['essay_minimal'] = df_test['essay'].apply(minimal_clean_text)

df_train['full_text'] = df_train['essay_minimal']
df_test['full_text'] = df_test['essay_minimal']

# Generate sentence embeddings
print("Generating sentence embeddings...")
train_embeddings = model.encode(df_train['full_text'].tolist(), 
                               batch_size=32, show_progress_bar=True)
test_embeddings = model.encode(df_test['full_text'].tolist(), 
                              batch_size=32, show_progress_bar=True)

print(f"Embedding shape: {train_embeddings.shape}")

# Prepare features
feature_columns = train_features.columns.tolist()
train_additional_features = df_train[feature_columns].values
test_additional_features = df_test[feature_columns].values

# Scale additional features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_additional_features_scaled = scaler.fit_transform(train_additional_features)
test_additional_features_scaled = scaler.transform(test_additional_features)

# Combine embeddings with additional features (BASE features for all models)
X_train_base = np.concatenate([train_embeddings, train_additional_features_scaled], axis=1)
X_test_base = np.concatenate([test_embeddings, test_additional_features_scaled], axis=1)

print(f"Base feature shape: {X_train_base.shape}")
print(f"Embeddings: {train_embeddings.shape[1]} dims")
print(f"Additional features: {train_additional_features_scaled.shape[1]} dims")

# Prepare targets in INTERDEPENDENT ORDER
target_order = ['lexical_resource', 'grammatical_range', 'task_achievement', 'coherence_and_cohesion']
y_train = df_train[target_order]

# Calculate target statistics for reference
target_stats = {}
for target_col in target_order:
    target_range = np.max(y_train[target_col]) - np.min(y_train[target_col])
    target_std = np.std(y_train[target_col])
    target_mean = np.mean(y_train[target_col])
    target_stats[target_col] = {
        'range': target_range,
        'std': target_std,
        'mean': target_mean,
        'min': np.min(y_train[target_col]),
        'max': np.max(y_train[target_col])
    }
    print(f"{target_col} - Range: {target_range:.4f}, Std: {target_std:.4f}, Mean: {target_mean:.4f}")


# Split data for parameter tuning
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_base, y_train, test_size=0.2, random_state=42
)

# =================================
# INTERDEPENDENT PREDICTION CLASS
# =================================

class InterdependentXGBPredictor:
    """
    XGBoost predictor that uses interdependent predictions where each target
    uses the predictions from previous targets as additional features.
    """
    
    def __init__(self, target_order, base_params):
        """
        Initialize the interdependent predictor.
        
        Args:
            target_order (list): Order in which targets should be predicted
            base_params (dict): Base XGBoost parameters
        """
        self.target_order = target_order
        self.base_params = base_params
        self.models = {}
        self.scalers = {}  # For scaling interdependent features
        
    def fit(self, X_base, y):
        """
        Fit models in interdependent order.
        
        Args:
            X_base (array): Base features (embeddings + handcrafted features)
            y (DataFrame): Target values with columns in target_order
        """
        
        # Store predictions for interdependence
        train_predictions = pd.DataFrame(index=range(len(X_base)), columns=self.target_order)
        
        for i, target_col in enumerate(self.target_order):
            print(f"Training model {i+1}/{len(self.target_order)}: {target_col}")
            
            # Prepare features for current target
            if i == 0:
                # First model: only base features
                X_current = X_base
            else:
                # Subsequent models: base features + predictions from previous targets
                prev_predictions = train_predictions.iloc[:, :i].values
                
                # Scale previous predictions
                scaler = StandardScaler()
                prev_predictions_scaled = scaler.fit_transform(prev_predictions)
                self.scalers[target_col] = scaler
                
                # Combine base features with scaled previous predictions
                X_current = np.concatenate([X_base, prev_predictions_scaled], axis=1)
            
            # Train model for current target
            model = XGBRegressor(**self.base_params)
            model.fit(X_current, y[target_col])
            self.models[target_col] = model
            
            # Get predictions for interdependence (use out-of-fold to avoid overfitting)
            # For simplicity, we'll use the same data (in practice, you might want CV)
            train_predictions[target_col] = model.predict(X_current)
            
            print(f"  Features used: {X_current.shape[1]} ({X_base.shape[1]} base + {X_current.shape[1] - X_base.shape[1]} interdependent)")
    
    def predict(self, X_base):
        """
        Make interdependent predictions.
        
        Args:
            X_base (array): Base features
            
        Returns:
            pd.DataFrame: Predictions for all targets
        """
        predictions = pd.DataFrame(index=range(len(X_base)), columns=self.target_order)
        
        for i, target_col in enumerate(self.target_order):
            # Prepare features for current target
            if i == 0:
                # First model: only base features
                X_current = X_base
            else:
                # Subsequent models: base features + predictions from previous targets
                prev_predictions = predictions.iloc[:, :i].values
                
                # Scale previous predictions using fitted scaler
                prev_predictions_scaled = self.scalers[target_col].transform(prev_predictions)
                
                # Combine base features with scaled previous predictions
                X_current = np.concatenate([X_base, prev_predictions_scaled], axis=1)
            
            # Make prediction
            predictions[target_col] = self.models[target_col].predict(X_current)
        
        return predictions

# =================================
# PARAMETER TUNING SECTION
# =================================

print("\n=== PARAMETER TUNING FOR INTERDEPENDENT MODELS ===")

# Define different parameter configurations to test
param_configs = [
    # Ultra-light model
    {
        'name': 'Ultra_Light',
        'n_estimators': 50,
        'learning_rate': 0.15,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'gpu_hist',
        'gpu_id': 0
    },
    # Compact model
    {
        'name': 'Compact',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,
        'reg_alpha': 0.05,
        'reg_lambda': 0.5,
        'tree_method': 'gpu_hist',
        'gpu_id': 0
    },
    # Light model
    {
        'name': 'Light',
        'n_estimators': 150,
        'learning_rate': 0.08,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'gpu_hist',
        'gpu_id': 0
    },
    # Efficient model
    {
        'name': 'Efficient',
        'n_estimators': 250,
        'learning_rate': 0.06,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'gpu_hist',
        'gpu_id': 0
    },
]

# Test each parameter configuration
config_results = {}

for config in param_configs:
    print(f"\nTesting configuration: {config['name']}")
    
    # Base parameters
    base_params = {
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Combine with current config
    current_params = {**base_params, **{k: v for k, v in config.items() if k != 'name'}}
    
    # Create and fit interdependent predictor
    predictor = InterdependentXGBPredictor(target_order, current_params)
    predictor.fit(X_train_split, y_train_split)
    
    # Make predictions on validation set
    val_predictions = predictor.predict(X_val_split)
    
    # Calculate columnwise RMSE for each target
    config_rmse = calculate_columnwise_rmse(y_val_split, val_predictions)
    
    # Calculate Mean Columnwise RMSE
    mean_columnwise_rmse = np.mean(list(config_rmse.values()))
    
    config_results[config['name']] = {
        'mean_columnwise_rmse': mean_columnwise_rmse,
        'individual_rmse': config_rmse,
        'params': current_params,
        'predictions': val_predictions
    }
    
    print(f"Mean Columnwise RMSE: {mean_columnwise_rmse:.4f}")
    for target_col in target_order:
        print(f"  {target_col}: {config_rmse[target_col]:.4f}")

# Find best configuration
best_config_name = min(config_results.keys(), key=lambda x: config_results[x]['mean_columnwise_rmse'])
best_config = config_results[best_config_name]

print(f"\n=== BEST CONFIGURATION ===")
print(f"Best configuration: {best_config_name}")
print(f"Best Mean Columnwise RMSE: {best_config['mean_columnwise_rmse']:.4f}")
print("\nBest parameters:")
for param, value in best_config['params'].items():
    if param not in ['objective', 'random_state', 'n_jobs']:
        print(f"  {param}: {value}")

print("\nIndividual RMSE scores (interdependent order):")
for target_col in target_order:
    print(f"  {target_col}: {best_config['individual_rmse'][target_col]:.4f}")

# =================================
# TRAIN FINAL INTERDEPENDENT MODEL
# =================================

print(f"\n=== TRAINING FINAL INTERDEPENDENT MODEL ===")

# Use the best parameters for final training
best_params = best_config['params']

# Train final interdependent predictor on full training data
final_predictor = InterdependentXGBPredictor(target_order, best_params)
final_predictor.fit(X_train_base, y_train)

# Make predictions on test set
y_pred_test = final_predictor.predict(X_test_base)

# Make final predictions on validation set for reporting
final_val_predictions = final_predictor.predict(X_val_split)
final_rmse_scores = calculate_columnwise_rmse(y_val_split, final_val_predictions)
final_mean_columnwise_rmse = np.mean(list(final_rmse_scores.values()))

print("\n=== FINAL RESULTS ===")
print(f"Best configuration used: {best_config_name}")
print(f"Interdependent prediction order: {' -> '.join(target_order)}")
print(f"Mean Columnwise RMSE on validation set: {final_mean_columnwise_rmse:.4f}")

print("\nFinal RMSE scores on validation set (interdependent order):")
for target_col in target_order:
    print(f"  {target_col}: {final_rmse_scores[target_col]:.4f}")

print("\nInterdependent model feature usage:")
for i, target_col in enumerate(target_order):
    base_features = X_train_base.shape[1]
    if i == 0:
        total_features = base_features
        interdep_features = 0
    else:
        interdep_features = i
        total_features = base_features + interdep_features
    print(f"  {target_col}: {total_features} features ({base_features} base + {interdep_features} interdependent)")

print(f"\nFinal test predictions shape: {y_pred_test.shape}")
print("Test predictions preview:")
print(y_pred_test.head())

# Show configuration comparison
print(f"\n=== CONFIGURATION COMPARISON (Sorted by Mean Columnwise RMSE) ===")
print("Configuration           | Mean Columnwise RMSE")
print("-" * 50)
for config_name, results in sorted(config_results.items(), key=lambda x: x[1]['mean_columnwise_rmse']):
    print(f"{config_name:<22} | {results['mean_columnwise_rmse']:.4f}")

# Detailed breakdown for top configurations
print(f"\n=== DETAILED RMSE BREAKDOWN FOR TOP 3 CONFIGURATIONS ===")
top_3_configs = sorted(config_results.items(), key=lambda x: x[1]['mean_columnwise_rmse'])[:3]

for i, (config_name, results) in enumerate(top_3_configs, 1):
    print(f"\n{i}. {config_name} (Mean Columnwise RMSE: {results['mean_columnwise_rmse']:.4f})")
    print("   Interdependent order performance:")
    for j, target_col in enumerate(target_order):
        rmse = results['individual_rmse'][target_col]
        interdep_info = f"(+{j} interdep. features)" if j > 0 else "(base features only)"
        print(f"   {j+1}. {target_col}: {rmse:.4f} {interdep_info}")

# Additional evaluation metrics for context
print(f"\n=== ADDITIONAL EVALUATION CONTEXT ===")
print("Target variable statistics (training set):")
for target_col in target_order:
    stats = target_stats[target_col]
    final_rmse = final_rmse_scores[target_col]
    rmse_to_std_ratio = final_rmse / stats['std']
    rmse_to_range_ratio = final_rmse / stats['range']
    print(f"  {target_col}:")
    print(f"    RMSE: {final_rmse:.4f}")
    print(f"    RMSE/Std: {rmse_to_std_ratio:.4f} (lower is better)")
    print(f"    RMSE/Range: {rmse_to_range_ratio:.4f} (lower is better)")

# Save predictions with correct column order
y_pred_test_final = y_pred_test[['task_achievement', 'coherence_and_cohesion', 'lexical_resource', 'grammatical_range']]
print(f"\nFinal predictions with standard column order:")
print(y_pred_test_final.head())

# Optional: Save predictions
y_pred_test_final.to_csv('interdependent_predictions.csv', index=False)

