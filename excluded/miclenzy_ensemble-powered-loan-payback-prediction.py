!pip install nbformat
!pip install optuna


# Main libraries for data manipulation
import pandas as pd
import numpy as np

# Visualization libraries
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
import optuna
from sklearn.ensemble import VotingClassifier

# preprocessing and encoding
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("Imports complete")


lpdf_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
lpdf_test  = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
lpdf_sam = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


lpdf_train.info()


lpdf_train.describe()


lpdf_train.drop(columns=['id'], inplace=True)
lpdf_test.drop(columns=['id'], inplace=True)


lpdf_train.head()


lpdf_test.head()


lpdf_train.isnull().sum()


lpdf_test.isnull().sum()


lpdf_train.education_level.unique()


def prepare_data(train_df, test_df, target, ranked_mappings=None, onehot_cols=None):
    """
    Preprocess train and test DataFrames:
      - Combines train and test for consistent feature engineering.
      - Engineers new ratio, polynomial, and interaction features.
      - Rank-encode ordered columns using provided mappings.
      - One-hot encode nominal columns.
      - Leaves all numeric columns (original + engineered) as-is.
    Returns: X_train, y_train, X_test, y_test (or None), preprocessor
    """

    onehot_cols = onehot_cols or []
    ranked_mappings = ranked_mappings or {}

    # --- 1. Split train features and target ---
    y_train = train_df[target]
    
    # Store train length to split later
    train_len = len(train_df)
    
    # Combine train (without target) and test for consistent processing
    # Drop 'id' columns if they exist
    train_cols_to_drop = [col for col in [target, 'id'] if col in train_df.columns]
    test_cols_to_drop = [col for col in ['id'] if col in test_df.columns]
    
    combined_df = pd.concat([train_df.drop(columns=train_cols_to_drop), 
                             test_df.drop(columns=test_cols_to_drop)], 
                            ignore_index=True)

    # --- 2. Feature Engineering on combined_df ---
    
    # A. Create numeric versions of ranked columns for interactions
    if "education_level" in ranked_mappings:
        edu_map = {level: i for i, level in enumerate(ranked_mappings["education_level"])}
        combined_df['education_level_num'] = combined_df['education_level'].map(edu_map)

    if "grade_subgrade" in ranked_mappings:
        grade_map = {level: i for i, level in enumerate(ranked_mappings["grade_subgrade"])}
        combined_df['grade_subgrade_num'] = combined_df['grade_subgrade'].map(grade_map)

    # B. Create Ratios, Polynomials, and Interactions
    epsilon = 1e-6 
    
    # Ratio features
    if 'loan_amount' in combined_df.columns and 'annual_income' in combined_df.columns:
        combined_df['loan_to_income_ratio'] = combined_df['loan_amount'] / (combined_df['annual_income'] + epsilon)

    if 'loan_amount' in combined_df.columns and 'loan_term_years' in combined_df.columns:
        combined_df['payment_proxy'] = combined_df['loan_amount'] / (combined_df['loan_term_years'] + epsilon)

    if 'debt_to_income_ratio' in combined_df.columns and 'annual_income' in combined_df.columns:
         combined_df['debt_amount_proxy'] = combined_df['debt_to_income_ratio'] * combined_df['annual_income']

    # Polynomial features
    if 'interest_rate' in combined_df.columns:
        combined_df['interest_rate_sq'] = combined_df['interest_rate']**2
    if 'credit_score' in combined_df.columns:
        combined_df['credit_score_sq'] = combined_df['credit_score']**2

    # Interaction features
    base_numeric_cols = [
        'annual_income', 'loan_amount', 'credit_score', 'loan_term_years', 
        'interest_rate', 'debt_to_income_ratio'
    ]
    
    for num_col in base_numeric_cols:
        if num_col in combined_df.columns:
            if 'grade_subgrade_num' in combined_df.columns:
                combined_df[f'{num_col}_x_grade'] = combined_df[num_col] * (combined_df['grade_subgrade_num'] + 1)
            if 'education_level_num' in combined_df.columns:
                combined_df[f'{num_col}_per_education'] = combined_df[num_col] / (combined_df['education_level_num'] + 1)

    # --- 3. Split back into X_train and X_test ---
    X_train = combined_df.iloc[:train_len]
    X_test = combined_df.iloc[train_len:]
    y_test = None # We don't have y_test for the submission

    # --- 4. Preprocessing (ColumnTransformer) ---
    
    # Identify ranked columns (original string versions)
    ranked_cols = list(ranked_mappings.keys())

    # Identify all numeric columns (original + engineered)
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ranked_cols + onehot_cols]

    # Build the column transformer
    transformers = []

    # A. OrdinalEncoder for original ranked (string) columns
    for col, order in ranked_mappings.items():
        transformers.append(
            (f"ranked_{col}", OrdinalEncoder(categories=[order], handle_unknown='use_encoded_value', unknown_value=-1), [col])
        )

    # B. OneHotEncoder for original nominal (string) columns
    if onehot_cols:
        transformers.append(
            ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore'), onehot_cols)
        )

    # C. Passthrough for all numeric columns (original + engineered)
    if numeric_cols:
        transformers.append(("numeric", "passthrough", numeric_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    # --- 5. Fit and Transform ---
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # --- 6. Build Feature Names ---
    feature_names = []
    
    # A. Get names from OrdinalEncoder
    for col, _ in ranked_mappings.items():
        feature_names.append(col) 

    # B. Get names from OneHotEncoder
    if onehot_cols:
        try:
            onehot_names = preprocessor.named_transformers_["onehot"].get_feature_names_out(onehot_cols).tolist()
            feature_names += onehot_names
        except (AttributeError, KeyError):
             print("Could not get one-hot feature names automatically.")
             
    # C. Get names from Passthrough (numeric)
    if numeric_cols:
        feature_names += numeric_cols

    # Convert processed arrays back to DataFrames
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)

    return X_train_processed, y_train, X_test_processed, y_test, preprocessor

ranked_mappings = {
    "education_level": ["Other", "High School", "Bachelor's", "Master's", "PhD"],
    "grade_subgrade": [
        "A1", "A2", "A3", "A4", "A5",
        "B1", "B2", "B3", "B4", "B5",
        "C1", "C2", "C3", "C4", "C5",
        "D1", "D2", "D3", "D4", "D5",
        "E1", "E2", "E3", "E4", "E5",
        "F1", "F2", "F3", "F4", "F5"
    ]
}

X_train, y_train, X_test, y_test, preprocessor = prepare_data(
    train_df=lpdf_train,
    test_df=lpdf_test,
    target="loan_paid_back",
    ranked_mappings=ranked_mappings,
    onehot_cols=["gender", "loan_purpose", "employment_status"]
)

print("Encoding Complete")


models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, device='gpu'),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42, task_type='GPU'),
    
}


model_scores = []

def add_model_report(model_name, y_true, y_pred, y_proba=None, storage=model_scores):
    """
    Takes a model's predictions and optionally probabilities, computes key metrics,
    and appends them to the storage list.
    """
    report = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }
    
    if y_proba is not None:
        report["ROC AUC"] = roc_auc_score(y_true, y_proba)
    else:
        report["ROC AUC"] = None

    storage.append(report)

print("Scores Created")


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store predictions for the real test set
test_predictions = {name: np.zeros(len(X_test)) for name in models.keys()}

# Loop over models
for name, model_instance in tqdm(models.items(), desc="Training models"):
    scale = name in ["Logistic Regression", "SVM"]
    
    # Arrays to collect out-of-fold (OOF) predictions and true labels
    oof_preds = np.zeros(len(X_train))
    oof_true = np.zeros(len(X_train))
    
    # Stratified K-Fold CV
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Build pipeline
        pipe = Pipeline([
            ("scaler", StandardScaler()) if scale else ("noop", "passthrough"),
            ("model", model_instance)
        ])
        
        # Fit on training fold
        pipe.fit(X_tr, y_tr)
        
        # Predict on validation fold
        if hasattr(pipe, "predict_proba"):
            try:
                y_val_pred = pipe.predict_proba(X_val)[:, 1]
            except:
                y_val_pred = pipe.predict(X_val)
        else:
            y_val_pred = pipe.predict(X_val)
        
        # Save OOF predictions for metrics
        oof_preds[val_idx] = y_val_pred
        oof_true[val_idx] = y_val
        
        # Predict on real test set and accumulate
        if hasattr(pipe, "predict_proba"):
            try:
                y_test_pred = pipe.predict_proba(X_test)[:, 1]
            except:
                y_test_pred = pipe.predict(X_test)
        else:
            y_test_pred = pipe.predict(X_test)
        
        test_predictions[name] += y_test_pred / n_splits  # average across folds
    
    # Compute metrics using OOF predictions
    # If classifier outputs probabilities, threshold at 0.5
    if oof_preds.ndim == 1 or oof_preds.shape[1] == 1:
        y_oof_labels = (oof_preds >= 0.5).astype(int)
    else:
        y_oof_labels = oof_preds  # already labels
    
    add_model_report(name, oof_true, y_oof_labels, y_proba=oof_preds)



ensemble_models = {
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42, task_type='GPU'),
    "LightGBM": LGBMClassifier(random_state=42, device='gpu')
}

ensemble = VotingClassifier(estimators=list(ensemble_models.items()), voting='soft')
ensemble.fit(X_train, y_train)

# Predict on test set
ensemble_pred_proba = ensemble.predict_proba(X_test)[:, 1]
ensemble_pred_class = ensemble.predict(X_test)

# Add to scores
add_model_report("Ensemble (CatBoost + LightGBM)", y_train, (ensemble.predict_proba(X_train)[:, 1] >= 0.5).astype(int), y_proba=ensemble.predict_proba(X_train)[:, 1])

print("Ensemble model trained and evaluated.")


df_scores = pd.DataFrame(model_scores).sort_values(by='F1 Score', ascending=False)
display(df_scores)


# Melt the DataFrame to long format for Plotly
df_long = df_scores.melt(id_vars="Model", var_name="Metric", value_name="Score")

# Create grouped bar chart
fig = px.bar(
    df_long,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    text="Score",
    title="Model Performance Comparison",
    height=650
)

fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
fig.update_layout(
    yaxis=dict(range=[0, 1]),
    font=dict(
        family="Arial, sans-serif",
        size=14,                   
        color="RebeccaPurple"      
    ),
    plot_bgcolor='lightgray',     
    paper_bgcolor='lightblue',     
    margin=dict(l=50, r=50, t=50, b=50),
    hovermode="x unified",   
    legend=dict(
        orientation="h",     
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
fig.show()


import joblib


X_train_final = X_train.copy()
y_train_final = y_train.copy()

def objective(trial):
    lgbm_params = {
        'objective': 'binary',
        'metric': 'roc_auc',
        'n_estimators': trial.suggest_int('lgbm_n_estimators', 500, 1500),
        'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 60),
        'max_depth': trial.suggest_int('lgbm_max_depth', 5, 10),
        'reg_alpha': trial.suggest_float('lgbm_reg_alpha', 0.01, 5.0),
        'reg_lambda': trial.suggest_float('lgbm_reg_lambda', 0.01, 5.0),
        'colsample_bytree': trial.suggest_float('lgbm_colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('lgbm_subsample', 0.5, 1.0),
        'random_state': 42,
        'device': 'cpu'  # Use CPU to avoid memory issues
    }
    
    # CatBoost parameters
    catboost_params = {
        'iterations': trial.suggest_int('cat_iterations', 500, 1500),
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('cat_depth', 5, 8),
        'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 0.1, 5.0),
        'random_strength': trial.suggest_float('cat_random_strength', 0.1, 1.0),
        'border_count': trial.suggest_int('cat_border_count', 32, 200),
        'random_state': 42,
        'verbose': 0,
        'task_type': 'CPU'  # Use CPU to avoid memory issues
    }
    
    weight_lgbm = trial.suggest_float('weight_lgbm', 0.1, 1.0)
    weight_cat = trial.suggest_float('weight_cat', 0.1, 1.0)

    clf1 = LGBMClassifier(**lgbm_params)
    clf2 = CatBoostClassifier(**catboost_params)
    
    ensemble = VotingClassifier(
        estimators=[('lgbm', clf1), ('cat', clf2)],
        voting='soft',
        weights=[weight_lgbm, weight_cat]
    )
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    score = cross_val_score(ensemble, X_train_final, y_train_final, cv=skf, scoring='roc_auc', n_jobs=-1).mean()
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)  # Reduced from 20 to 10 trials

print("\n--- Optuna Study Complete ---")
print("Best trial:")
trial = study.best_trial

print(f"  Value (ROC AUC): {trial.value}")
print("  Best Hyperparameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Train final model with best parameters
best_params = trial.params

lgbm_params_final = {
    'objective': 'binary',
    'metric': 'roc_auc',
    'n_estimators': best_params['lgbm_n_estimators'],
    'learning_rate': best_params['lgbm_learning_rate'],
    'num_leaves': best_params['lgbm_num_leaves'],
    'max_depth': best_params['lgbm_max_depth'],
    'reg_alpha': best_params['lgbm_reg_alpha'],
    'reg_lambda': best_params['lgbm_reg_lambda'],
    'colsample_bytree': best_params['lgbm_colsample_bytree'],
    'subsample': best_params['lgbm_subsample'],
    'random_state': 42,
    'device': 'gpu'
}

catboost_params_final = {
    'iterations': best_params['cat_iterations'],
    'learning_rate': best_params['cat_learning_rate'],
    'depth': best_params['cat_depth'],
    'l2_leaf_reg': best_params['cat_l2_leaf_reg'],
    'random_strength': best_params['cat_random_strength'],
    'border_count': best_params['cat_border_count'],
    'random_state': 42,
    'verbose': 0,
    'task_type': 'GPU'
}

clf1_final = LGBMClassifier(**lgbm_params_final)
clf2_final = CatBoostClassifier(**catboost_params_final)

final_ensemble = VotingClassifier(
    estimators=[('lgbm', clf1_final), ('cat', clf2_final)],
    voting='soft',
    weights=[best_params['weight_lgbm'], best_params['weight_cat']]
)

# Train on full training data
final_ensemble.fit(X_train_final, y_train_final)

# Save the Optuna-optimized model
joblib.dump(final_ensemble, "ensemble.joblib")
print("\nOptuna-optimized ensemble model saved to ensemble.joblib")


finalModel = joblib.load('ensemble.joblib')


y_class = finalModel.predict(X_test)
y_proba = finalModel.predict_proba(X_test)[:, 1]


submission_df = lpdf_sam.copy()
submission_df['loan_paid_back'] = y_proba
submission_df.to_csv('submission.csv', index=False)

print('Complete')

