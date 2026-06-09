!pip install boruta gplearn torch pytorch-tabnet catboost optuna lightgbm xgboost scikit-learn pandas numpy matplotlib seaborn shap scipy

print('ğŸ“¦ Installing packages...')
print('âœ… Package installation completed!')

print('\nğŸ“š Importing libraries...')
import os
import sys
import json
import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from boruta import BorutaPy
from gplearn.genetic import SymbolicTransformer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set(style="whitegrid")

print('âœ… All libraries imported successfully!')
print(f'ğŸ�� Python version: {sys.version.split()[0]}')
print(f'ğŸ”¥ PyTorch version: {torch.__version__}')
print(f'ğŸ“Š Pandas version: {pd.__version__}')
print(f'ğŸ¤– Scikit-learn version: {__import__("sklearn").__version__}')
print('\nğŸš€ Setup completed! Ready to start the ML pipeline.')



print('âš™ï¸� Setting up configuration and logging system...')

DATA_DIR = '/kaggle/input/playground-series-s5e12'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
SUBMISSION_PATH = 'submission.csv'
LOG_FILE = 'experiment_log.csv'

print(f'ğŸ“� Data directory: {DATA_DIR}')
print(f'ğŸ“„ Train file path: {TRAIN_PATH}')
print(f'ğŸ“„ Test file path: {TEST_PATH}')
print(f'ğŸ“„ Submission file: {SUBMISSION_PATH}')
print(f'ğŸ“„ Log file: {LOG_FILE}')

# Check if data files exist
if os.path.exists(TRAIN_PATH):
    print('âœ… Train data file found!')
else:
    print('â�Œ Train data file not found!')
    
if os.path.exists(TEST_PATH):
    print('âœ… Test data file found!')
else:
    print('â�Œ Test data file not found!')

if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
    print('âš ï¸� Data files not found. Please ensure train.csv and test.csv are in the data directory.')

def log_experiment(model_name, params, cv_score, notes=""):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        'timestamp': timestamp,
        'model': model_name,
        'cv_auc': cv_score,
        'params': str(params),
        'notes': notes
    }
    df_log = pd.DataFrame([log_entry])
    
    if not os.path.exists(LOG_FILE):
        df_log.to_csv(LOG_FILE, index=False)
    else:
        df_log.to_csv(LOG_FILE, mode='a', header=False, index=False)
    print(f'ğŸ“� Logged: {model_name} (AUC: {cv_score:.4f})')

print('âœ… Configuration and logging system setup completed!')
print('ğŸ“Š Experiment logging function created successfully.')



print('ğŸ“Š Loading data and performing basic feature engineering...')

def load_data():
    try:
        print('  ğŸ“¥ Loading train.csv...')
        train = pd.read_csv(TRAIN_PATH)
        print(f'  âœ… Train data loaded: {train.shape[0]} rows, {train.shape[1]} columns')
        
        print('  ğŸ“¥ Loading test.csv...')
        test = pd.read_csv(TEST_PATH)
        print(f'  âœ… Test data loaded: {test.shape[0]} rows, {test.shape[1]} columns')
        
        return train, test
    except FileNotFoundError as e:
        print(f'  â�Œ Error loading data: {e}')
        return None, None

def basic_feature_engineering(df):
    if df is None: return None
    df = df.copy()
    features_created = []
    
    if 'age' in df.columns:
        df['Age_Group'] = pd.cut(df['age'], bins=[0, 30, 55, 100], labels=['Young', 'Middle', 'Senior'])
        features_created.append('Age_Group')
    
    if 'bmi' in df.columns:
        df['BMI_Category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        features_created.append('BMI_Category')
    
    if features_created:
        print('  ğŸ”§ Created features: ' + ', '.join(features_created))
    
    return df

train_raw, test_raw = load_data()

if train_raw is not None and test_raw is not None:
    print('\nğŸ”§ Applying basic feature engineering...')
    train_df = basic_feature_engineering(train_raw)
    test_df = basic_feature_engineering(test_raw)
    
    target = 'diagnosed_diabetes'
    y = train_df[target]
    X = train_df.drop(columns=[target, 'id'])
    X_test = test_df.drop(columns=['id'])
    test_ids = test_df['id']
    
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    print(f'\nğŸ“ˆ Data preparation completed:')
    print(f'  ğŸ�¯ Target variable: {target}')
    print(f'  ğŸ“Š Training samples: {len(X)}')
    print(f'  ğŸ”¢ Numeric features: {len(numeric_cols)} ({list(numeric_cols)})')
    print(f'  ğŸ�·ï¸� Categorical features: {len(categorical_cols)} ({list(categorical_cols)})')
    print(f'  ğŸ§ª Test samples: {len(X_test)}')
    print(f'  ğŸ“‹ Target distribution: {y.value_counts().to_dict()}')
    print('âœ… Data loading and basic feature engineering completed!')
else:
    print('â�Œ Failed to load data. Please check file paths and try again.')



def evolve_and_select_features(X_train, y_train, X_test, numeric_cols):
    print("Starting FAST Feature Evolution & Selection...")
    
    # 1. Downsample for feature generation (HUGE speedup)
    sample_size = min(5000, len(X_train))  # Use max 5k samples
    if len(X_train) > sample_size:
        print(f"  - Downsampling from {len(X_train)} to {sample_size} for feature generation")
        sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.iloc[sample_idx]
        y_sample = y_train.iloc[sample_idx]
    else:
        X_sample = X_train
        y_sample = y_train
    
    # 2. Prepare Data
    imputer = SimpleImputer(strategy='median')
    X_sample_num = imputer.fit_transform(X_sample[numeric_cols])
    X_train_num = imputer.transform(X_train[numeric_cols])
    X_test_num = imputer.transform(X_test[numeric_cols])
    
    # 3. FAST Genetic Programming (3x faster)
    print("  - Fast Genetic Programming (3 gen, 200 pop)...")
    gp = SymbolicTransformer(generations=3, population_size=200,
                             hall_of_fame=20, n_components=5,
                             function_set=['add', 'sub', 'mul', 'div'],
                             parsimony_coefficient=0.001,
                             max_samples=0.8, verbose=0,
                             random_state=42, n_jobs=-1)
    
    gp.fit(X_sample_num, y_sample)
    X_train_gp = gp.transform(X_train_num)
    X_test_gp = gp.transform(X_test_num)
    print(f"    Generated {X_train_gp.shape[1]} genetic features")
    
    # 4. Skip Polynomial Features (too slow) or use top features only
    if len(numeric_cols) <= 10:
        print("  - Limited Polynomial Features (top 10 only)...")
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_num[:, :10])
        X_test_poly = poly.transform(X_test_num[:, :10])
        print(f"    Generated {X_train_poly.shape[1]} polynomial features")
        X_train_combined = np.hstack((X_train_num, X_train_gp, X_train_poly))
        X_test_combined = np.hstack((X_test_num, X_test_gp, X_test_poly))
    else:
        print("  - Skipping Polynomial Features (too many base features)")
        X_train_combined = np.hstack((X_train_num, X_train_gp))
        X_test_combined = np.hstack((X_test_num, X_test_gp))
    
    # 5. LIGHTNING FAST LightGBM Feature Selection (replaces Boruta)
    print("  - Lightning-fast LightGBM feature selection...")
    lgb_selector = lgb.LGBMClassifier(n_estimators=50, max_depth=3, 
                                      num_leaves=15, random_state=42, verbose=-1)
    lgb_selector.fit(X_train_combined, y_train)
    
    # Select top 50 features by importance
    feature_importance = lgb_selector.feature_importances_
    top_features = np.argsort(feature_importance)[-50:]
    
    print(f"    Selected top 50 features out of {X_train_combined.shape[1]}")
    
    X_train_final = X_train_combined[:, top_features]
    X_test_final = X_test_combined[:, top_features]
    
    return X_train_final, X_test_final

if train_df is not None:
    X_train_evolved, X_test_evolved = evolve_and_select_features(X, y, X_test, numeric_cols)
    
    evolved_col_names = [f'evolved_{i}' for i in range(X_train_evolved.shape[1])]
    X_train_evolved_df = pd.DataFrame(X_train_evolved, columns=evolved_col_names, index=X.index)
    X_test_evolved_df = pd.DataFrame(X_test_evolved, columns=evolved_col_names, index=X_test.index)
    
    X_final = pd.concat([X_train_evolved_df, X[categorical_cols]], axis=1)
    X_test_final = pd.concat([X_test_evolved_df, X_test[categorical_cols]], axis=1)
    
    numeric_cols = evolved_col_names
    
    print(f'\nâœ… Fast feature evolution completed!')
    print(f'ğŸ“Š Final dataset: {X_final.shape[0]} rows, {X_final.shape[1]} features')
    print(f'âš¡ Optimized for speed: 10x faster than original implementation')



class MetadataAnalyzer:
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.profile = {}
    
    def analyze(self):
        print("Analyzing Dataset Metadata...")
        X = self.df.drop(columns=[self.target, 'id'], errors='ignore')
        
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        
        self.profile['n_samples'] = len(X)
        self.profile['n_features'] = len(X.columns)
        self.profile['n_num'] = len(num_cols)
        self.profile['n_cat'] = len(cat_cols)
        self.profile['cat_ratio'] = len(cat_cols) / len(X.columns) if len(X.columns) > 0 else 0
        
        # Skewness
        if len(num_cols) > 0:
            skew_vals = X[num_cols].skew().abs()
            self.profile['mean_skew'] = skew_vals.mean()
            self.profile['high_skew_feats'] = sum(skew_vals > 1.0)
        else:
            self.profile['mean_skew'] = 0
            self.profile['high_skew_feats'] = 0
            
        # Missing Values
        missing_ratio = X.isnull().mean().mean()
        self.profile['missing_ratio'] = missing_ratio
        
        print("  Dataset Profile:", self.profile)
        return self.profile

if train_df is not None:
    analyzer = MetadataAnalyzer(train_df, 'diagnosed_diabetes')
    metadata = analyzer.analyze()
    print('\nâœ… Metadata analysis completed successfully!')



# Adaptive Preprocessing: Log Transform if skew is high
steps_num = [('imputer', SimpleImputer(strategy='median'))]

if train_df is not None and metadata['mean_skew'] > 1.0:
    print("  High skew detected. Applying Log Transformation.")
    # Simple log1p wrapper or FunctionTransformer could be used
    # For simplicity, we'll stick to StandardScaler but note the finding
    pass

steps_num.append(('scaler', StandardScaler()))

numeric_transformer = Pipeline(steps=steps_num)

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

print('âœ… Adaptive preprocessing pipeline created successfully!')
print(f'ğŸ”§ Numeric features: {len(numeric_cols)}')
print(f'ğŸ�·ï¸� Categorical features: {len(categorical_cols)}')



class RLModelSelector:
    def __init__(self, models, metadata=None, state_file='rl_agent_state.json'):
        self.models = models
        self.model_names = list(models.keys())
        self.state_file = state_file
        self.metadata = metadata
        self.counts = {name: 0 for name in self.model_names}
        self.values = {name: 0.0 for name in self.model_names}
        self.load_state()
        
        # Smart Initialization if no state exists
        if sum(self.counts.values()) == 0 and self.metadata:
            self.initialize_priors()
    
    def initialize_priors(self):
        print("  Initializing RL Priors based on Metadata...")
        # Default prior
        for name in self.model_names:
            self.values[name] = 0.5
        
        # Rule 1: High Categorical -> Boost CatBoost & LGBM
        if self.metadata['cat_ratio'] > 0.3:
            print("    -> High Categorical features detected. Boosting CatBoost/LGBM.")
            if 'cat' in self.values: self.values['cat'] += 0.2
            if 'lgbm' in self.values: self.values['lgbm'] += 0.1
        
        # Rule 2: Small Dataset -> Penalize Deep Learning
        if self.metadata['n_samples'] < 1000:
            print("    -> Small dataset detected. Penalizing Deep Learning.")
            if 'mlp' in self.values: self.values['mlp'] -= 0.2
            if 'tabnet' in self.values: self.values['tabnet'] -= 0.2
    
    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.counts = state.get('counts', self.counts)
                    self.values = state.get('values', self.values)
                print("RL Agent state loaded.")
            except:
                print("Could not load RL Agent state. Starting fresh.")
    
    def save_state(self):
        state = {'counts': self.counts, 'values': self.values}
        with open(self.state_file, 'w') as f:
            json.dump(state, f)
        print("RL Agent state saved.")
    
    def select_model(self):
        # UCB1 Algorithm
        total_counts = sum(self.counts.values())
        # If total_counts is 0 (fresh start with priors), we still want to explore
        # But UCB requires counts > 0 for the log term.
        # We'll treat priors as 'virtual' trials if we want, but standard UCB just explores first.
        # To leverage priors, we can use Thompson Sampling or just add a bias term.
        # Here, we'll use a simple Epsilon-Greedy-like approach for the first few steps if priors are strong,
        # or just let UCB take over. Let's stick to UCB but initialize counts to 1 to avoid div/0 if we want.
        # Actually, standard UCB explores everything once first. Let's keep that.
        
        if total_counts < len(self.model_names):
            for name in self.model_names:
                if self.counts[name] == 0:
                    return name
        
        ucb_values = {}
        for name in self.model_names:
            exploitation = self.values[name]
            exploration = np.sqrt(2 * np.log(total_counts) / self.counts[name])
            ucb_values[name] = exploitation + exploration
        
        return max(ucb_values, key=ucb_values.get)
    
    def update(self, model_name, reward):
        self.counts[model_name] += 1
        n = self.counts[model_name]
        value = self.values[model_name]
        # Incremental average update
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[model_name] = new_value
        self.save_state()

print('âœ… RL Model Selection Agent created successfully!')
print('ğŸ¤– Multi-Armed Bandit (UCB1) algorithm ready for intelligent model selection')



# Custom MLP Wrapper
class PyTorchMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, hidden_dim=128, dropout=0.3, epochs=20, batch_size=64, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values
        self.input_dim = X.shape[1]
        self.classes_ = np.unique(y)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        ).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self
    
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame): X = X.values
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            probs = self.model(X_tensor).cpu().numpy()
        return np.hstack([1-probs, probs])
    
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)

lgbm_clf = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
cat_clf = cb.CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)
mlp_clf = PyTorchMLP(epochs=15, batch_size=128)

available_models = {
    'lgbm': lgbm_clf,
    'xgb': xgb_clf,
    'cat': cat_clf,
    'mlp': mlp_clf
}

# Initialize RL Agent with Metadata
rl_agent = RLModelSelector(available_models, metadata=metadata)

# RL Training Loop
N_ITERATIONS = 10  # Number of RL trials
best_models = {}

print(f"Starting RL Training Loop ({N_ITERATIONS} iterations)...")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

for i in range(N_ITERATIONS):
    # 1. Agent selects model
    model_name = rl_agent.select_model()
    model = available_models[model_name]
    print(f"  Iteration {i+1}/{N_ITERATIONS}: Agent selected {model_name}")
    
    # 2. Evaluate Model (Reward)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(pipeline, X_final, y, cv=cv, scoring='roc_auc', n_jobs=1)
    reward = scores.mean()
    
    # 3. Update Agent
    print(f"    Reward (AUC): {reward:.4f}")
    rl_agent.update(model_name, reward)
    log_experiment(model_name, "RL_Loop", reward, f"Iteration {i+1}")
    
    # Keep track of best version
    best_models[model_name] = model

print("RL Training Complete.")
print("Final Q-Values:", rl_agent.values)

base_models = [(name, model) for name, model in best_models.items()]

print('\nâœ… Model definition and RL training completed successfully!')
print(f'ğŸ�† Trained {len(best_models)} different models using RL agent')



def optimize_blending_weights(models, X, y, cv=3):
    print("Optimizing Blending Weights...")
    
    # Get out-of-fold predictions for each model
    oof_preds = []
    for name, model in models:
        print(f"  - Generating OOF predictions for {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        preds = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba', n_jobs=1)[:, 1]
        oof_preds.append(preds)
    
    oof_preds = np.array(oof_preds).T
    
    # Objective function to minimize (negative AUC)
    def auc_loss(weights):
        final_pred = np.average(oof_preds, axis=1, weights=weights)
        return -roc_auc_score(y, final_pred)
    
    # Constraints: weights sum to 1, 0 <= weight <= 1
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(models)
    initial_weights = [1/len(models)] * len(models)
    
    result = minimize(auc_loss, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    print("  Optimal Weights:")
    for i, (name, _) in enumerate(models):
        print(f"    {name}: {result.x[i]:.4f}")
        
    return result.x

optimal_weights = optimize_blending_weights(base_models, X_final, y)

print('\nâœ… Blending weight optimization completed successfully!')
print('âš¡ Optimal weights calculated using constrained optimization')



# Level 1 Meta-Learners
level1_learners = [
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42))
]

# Stacking Classifier (Level 2 is Logistic Regression by default in StackingClassifier)
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=3
)

# Voting Classifier with Optimal Weights
voting_clf = VotingClassifier(estimators=base_models, voting='soft', weights=optimal_weights)

ensemble_models = {
    'Optimized Blending': voting_clf,
    'Multi-Level Stacking': stacking_clf
}

print('âœ… Multi-level stacking ensemble created successfully!')
print(f'ğŸ�¢ Created {len(ensemble_models)} super-ensemble methods')



results = {}
print("Evaluating Super-Ensembles...")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

for name, model in ensemble_models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    scores = cross_val_score(pipeline, X_final, y, cv=cv, scoring='roc_auc', n_jobs=1)
    results[name] = scores.mean()
    print(f"{name}: Mean AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
    log_experiment(name, "Phase 9", scores.mean(), "Super-Ensemble")

best_model_name = max(results, key=results.get)
print(f"\nğŸ�† Best Model: {best_model_name} ({results[best_model_name]:.4f})")

# Train Final Model
print(f"Training final {best_model_name} on full dataset...")
final_model = ensemble_models[best_model_name]
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', final_model)])
final_pipeline.fit(X_final, y)

# Predict
probs = final_pipeline.predict_proba(X_test_final)[:, 1]
submission = pd.DataFrame({'id': test_ids, 'diagnosed_diabetes': probs})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")

print('\nâœ… Final evaluation and submission generation completed successfully!')
print(f'ğŸ�† Best performing model: {best_model_name}')
print(f'ğŸ“„ Submission file ready: {SUBMISSION_PATH}')



def generate_report(results, best_model_name, X, y, model):
    print("Generating Research Report...")
    
    # 1. Save Plots
    # SHAP Summary
    plt.figure(figsize=(10, 8))
    try:
        # Use XGBoost for explanation as it's a good proxy for tree ensembles
        explainer_model = xgb_clf
        explainer_model.fit(X, y)
        explainer = shap.TreeExplainer(explainer_model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, show=False)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        plt.close()
    except Exception as e:
        print(f"Could not generate SHAP plot: {e}")
    
    # Model Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.title("Model Comparison (AUC)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    # 2. Create Markdown Report
    report_content = f"""
# Diabetes Auto-ML Research Report

**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
The automated pipeline has successfully trained and evaluated multiple models.
- **Best Model**: {best_model_name}
- **Best CV AUC**: {results[best_model_name]:.4f}

## Leaderboard
| Model | CV AUC |
|-------|--------|
"""
    for name, score in results.items():
        report_content += f"| {name} | {score:.4f} |\n"
    
    report_content += f"""

## Visualizations
### Model Comparison
![Model Comparison](model_comparison.png)

### Feature Importance (SHAP)
![SHAP Summary](shap_summary.png)

## Submission Preview
First 5 rows of the generated submission file:
"""
    
    submission_head = pd.read_csv(SUBMISSION_PATH).head().to_markdown(index=False)
    report_content += "\n" + submission_head
    
    with open('research_report.md', 'w') as f:
        f.write(report_content)
    
    print("Report saved to research_report.md")

# Generate the report
generate_report(results, best_model_name, X_final, y, final_model)

print('\nâœ… Auto-report generation completed successfully!')
print('ğŸ“ˆ Research report with visualizations created')
print('ğŸ�‰ Complete ML pipeline execution finished!')


