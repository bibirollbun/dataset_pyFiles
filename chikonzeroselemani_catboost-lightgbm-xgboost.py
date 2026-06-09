from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path
from sklearn import preprocessing
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.base import clone
import joblib 
import warnings
warnings.filterwarnings("ignore")


N_FOLDS = 5
ALPHA = 0.10  # 90% coverage
HILL_CLIMB_STEPS = 100
MODEL_NAMES = ["LightGBM", "CatBoost", "XGBoost"]
INITIAL_WEIGHTS = [0.4, 0.3, 0.3]
N_JOBS = -1  # Use all available cores


train_path="/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv"
test_path="/kaggle/input/prediction-interval-competition-ii-house-price/test.csv"

train_df=pd.read_csv(train_path)
test = pd.read_csv(test_path)

train_df


train_df = train_df.drop(columns=["submarket","subdivision","sale_nbr"])
test = test.drop(columns=["submarket","subdivision","sale_nbr"])
train_df


def preprocess(df):
    # Convert date and extract features
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['sale_year'] = df['sale_date'].dt.year
    df['sale_month'] = df['sale_date'].dt.month
    
    # Feature engineering
    df['age'] = df['sale_year'] - df['year_built']
    df['renovated'] = np.where(df['year_reno'] > 0, 1, 0)
    df['years_since_reno'] = np.where(df['renovated'], df['sale_year'] - df['year_reno'], 0)
    df['total_baths'] = df['bath_full'] + 0.75*df['bath_3qtr'] + 0.5*df['bath_half']
    df['total_value'] = df['land_val'] + df['imp_val']
    df['living_area'] = df['sqft'] + df['sqft_fbsmt']
    

    
    # Encode categoricals (simplified example)
    cat_cols = ['sale_warning', 'join_status', 'city', 'zoning']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    
    return df.drop(columns=['sale_date', 'id'])

# Load and preprocess data
train_df = preprocess(train_df)
X_test = preprocess(test)


X = train_df.drop(columns=['sale_price'])
y = train_df['sale_price']

# Ordinal encoding.
cat_cols = ['sale_warning', 'join_status', 'city', 'zoning']
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])


def winkler_score(y_true, lower, upper, alpha=ALPHA):
    """Vectorized Mean Winkler Interval Score calculation"""
    width = upper - lower
    below = np.maximum(lower - y_true, 0)
    above = np.maximum(y_true - upper, 0)
    return width + (2/alpha) * (below + above)
    

def train_model(model, X_train, y_train, X_val):
    """Train model and predict on validation set"""
    model.fit(X_train, y_train)
    if isinstance(model, cb.CatBoostRegressor):
        preds = model.predict(X_val)
        means = preds[:, 0]
        stds = np.sqrt(preds[:, 1])
        lower = means - 1.645 * stds
        upper = means + 1.645 * stds
    else:
        lower = model["lower"].predict(X_val)
        upper = model["upper"].predict(X_val)
    return lower, upper


models = {
        "LightGBM": {
            "lower": lgb.LGBMRegressor(
                objective="quantile", 
                alpha=0.05,
                device="gpu",
                n_estimators=1500,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                subsample_freq=1,
                random_state=42
            ),
            "upper": lgb.LGBMRegressor(
                objective="quantile", 
                alpha=0.95,
                device="gpu",
                n_estimators=1500,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                subsample_freq=1,
                random_state=42
            )
        },
        "CatBoost": cb.CatBoostRegressor(
            loss_function="RMSEWithUncertainty",
            task_type="GPU",
            iterations=1500,
            learning_rate=0.05,
            depth=8,
            verbose=0,
            random_seed=42
        ),
        "XGBoost": {
            "lower": xgb.XGBRegressor(
                objective="reg:quantileerror", 
                quantile_alpha=0.05,
                tree_method="gpu_hist",
                n_estimators=1500,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                random_state=42
            ),
            "upper": xgb.XGBRegressor(
                objective="reg:quantileerror", 
                quantile_alpha=0.95,
                tree_method="gpu_hist",
                n_estimators=1500,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                random_state=42
            )
        }
    }
     
   


# Initialize storage for OOF predictions
oof_lowers = {model: np.zeros(len(X)) for model in MODEL_NAMES}
oof_uppers = {model: np.zeros(len(X)) for model in MODEL_NAMES}
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
print("Starting Cross-Validation...")
for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X, y)), total=N_FOLDS, desc="Folds"):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train and predict for each model
    for model_name in MODEL_NAMES:
        if model_name == "CatBoost":
            model = clone(models["CatBoost"])
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            means = preds[:, 0]
            stds = np.sqrt(preds[:, 1])
            oof_lowers[model_name][val_idx] = means - 1.645 * stds
            oof_uppers[model_name][val_idx] = means + 1.645 * stds
                
        else:
                # Train lower quantile model
            lower_model = clone(models[model_name]["lower"])
            lower_model.fit(X_train, y_train)
            lower_pred = lower_model.predict(X_val)
                
                # Train upper quantile model
            upper_model = clone(models[model_name]["upper"])
            upper_model.fit(X_train, y_train)
            upper_pred = upper_model.predict(X_val)
                
            oof_lowers[model_name][val_idx] = lower_pred
            oof_uppers[model_name][val_idx] = upper_pred
    
    # Calculate MWIS for each model
model_scores = {}
print("\nModel Performance Evaluation:")
for model_name in MODEL_NAMES:
        # Ensure valid intervals
    lower = np.minimum(oof_lowers[model_name], oof_uppers[model_name])
    upper = np.maximum(oof_lowers[model_name], oof_uppers[model_name])
        
        # Calculate MWIS
    wis = winkler_score(y, lower, upper)
    mwis = np.mean(wis)
    model_scores[model_name] = mwis
        
        # Calculate coverage
    coverage = np.mean((y >= lower) & (y <= upper)) * 100
    print(f"{model_name}: MWIS = {mwis:.2f}, Coverage = {coverage:.2f}%")
    
    # Hill Climbing Optimization
print("\nStarting Hill Climbing Optimization...")
current_weights = np.array(INITIAL_WEIGHTS)
best_weights = current_weights.copy()
best_score = float('inf')
    
    # Calculate initial combined score
combined_lower = sum(w * oof_lowers[model] for w, model in zip(current_weights, MODEL_NAMES))
combined_upper = sum(w * oof_uppers[model] for w, model in zip(current_weights, MODEL_NAMES))
current_score = np.mean(winkler_score(y, combined_lower, combined_upper))
    
print(f"Initial MWIS: {current_score:.4f}")
    
    # Optimization loop
for step in tqdm(range(HILL_CLIMB_STEPS), desc="Hill Climbing"):
    improved = False
    candidate_weights = current_weights.copy()
        
        # Generate candidate weights
    perturbation = np.random.dirichlet([5] * len(MODEL_NAMES)) - 0.5/len(MODEL_NAMES)
    candidate_weights = candidate_weights + 0.1 * perturbation
    candidate_weights = np.maximum(candidate_weights, 0)
    candidate_weights /= candidate_weights.sum()
        
        # Calculate combined predictions
    combined_lower = sum(w * oof_lowers[model] for w, model in zip(candidate_weights, MODEL_NAMES))
    combined_upper = sum(w * oof_uppers[model] for w, model in zip(candidate_weights, MODEL_NAMES))
        
        # Calculate MWIS
    candidate_score = np.mean(winkler_score(y, combined_lower, combined_upper))
        
        # Update if improvement
    if candidate_score < best_score:
        best_score = candidate_score
        best_weights = candidate_weights.copy()
        current_weights = candidate_weights.copy()
        improved = True
        print(f"Step {step}: New best MWIS = {best_score:.4f}, Weights = {best_weights}")
    


# Final model training
print("\nTraining Final Models...")
test_preds = {"lower": {}, "upper": {}}
    
for model_name in tqdm(MODEL_NAMES, desc="Models"):
    if model_name == "CatBoost":
        model = clone(models["CatBoost"])
        model.fit(X, y)
        preds = model.predict(X_test)
        means = preds[:, 0]
        stds = np.sqrt(preds[:, 1])
        test_preds["lower"][model_name] = means - 1.645 * stds
        test_preds["upper"][model_name] = means + 1.645 * stds
            
    else:
            # Train lower quantile model
        lower_model = clone(models[model_name]["lower"])
        lower_model.fit(X, y)
        test_preds["lower"][model_name] = lower_model.predict(X_test)
            
            # Train upper quantile model
        upper_model = clone(models[model_name]["upper"])
        upper_model.fit(X, y)
        test_preds["upper"][model_name] = upper_model.predict(X_test)
    
    # Combine test predictions
final_lower = sum(best_weights[i] * test_preds["lower"][model] 
                     for i, model in enumerate(MODEL_NAMES))
final_upper = sum(best_weights[i] * test_preds["upper"][model] 
                     for i, model in enumerate(MODEL_NAMES))
    
    # Ensure valid intervals
final_lower, final_upper = np.minimum(final_lower, final_upper), np.maximum(final_lower, final_upper)
final_lower = np.maximum(final_lower, 0)  # Ensure non-negative prices
    



    # Create submission
test_ids = test["id"]
submission = pd.DataFrame({
        "id": test_ids,
        "pi_lower": final_lower,
        "pi_upper": final_upper
    })
submission.to_csv("submission.csv", index=False)
print("Submission saved successfully")

submission = pd.read_csv("submission.csv")
submission.head(5)
    


from rich.console import Console
from rich.table import Table

console = Console()

# Header
console.rule("[bold cyan]FINAL PERFORMANCE REPORT")

# Optimized Weights Table
weights_table = Table(title="Optimized Weights", title_style="bold green")
weights_table.add_column("Model", justify="left", style="cyan")
weights_table.add_column("Weight", justify="right", style="magenta")

for model, weight in zip(MODEL_NAMES, best_weights):
    weights_table.add_row(model, f"{weight:.4f}")

console.print(weights_table)

# Model Performance Table
scores_table = Table(title="Model MWIS Scores", title_style="bold green")
scores_table.add_column("Model", justify="left", style="cyan")
scores_table.add_column("MWIS Score", justify="right", style="yellow")

for model, score in model_scores.items():
    scores_table.add_row(model, f"{score:.4f}")

console.print(scores_table)

# Ensemble Score Highlighted
ensemble_table = Table(title="Ensemble MWIS Score", title_style="bold red")
ensemble_table.add_column("Model", style="bold", justify="left")
ensemble_table.add_column("MWIS Score", style="bold yellow", justify="right")
ensemble_table.add_row("Ensemble", f"{best_score:.4f}")

console.print(ensemble_table)


