import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, average_precision_score, 
    precision_recall_curve, confusion_matrix, classification_report
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import math
import optuna
import warnings
warnings.filterwarnings('ignore')


%%time
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


train.head()


train.info()


test.info()


train.duplicated().sum()
test.duplicated().sum()


train.describe()


train.describe(include = 'object')


train_numeric = train.select_dtypes(include=['number'])

# Calculate correlation
corrtrain_matrix = train_numeric.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corrtrain_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap Correlation")


# --- Target Distribution: Diagnosed Diabetes ---
plt.figure(figsize=(8,5))
sns.histplot(train["diagnosed_diabetes"], bins=40, kde=True, color="skyblue")
plt.title("Distribution of Diagnosed Diabetes (Target)", fontsize=14)
plt.xlabel("Diagnosed Diabetes")
plt.ylabel("Count")
plt.show()



exclude_cols = ["diagnosed_diabetes", "id"]

num_cols = train.select_dtypes(include=["int64", "float64"]).columns
num_cols = [c for c in num_cols if c not in exclude_cols]

n_cols = 2
n_rows = math.ceil(len(num_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

for i, col in enumerate(num_cols):
    r, c = divmod(i, n_cols)
    ax = axes[r, c] if n_rows > 1 else axes[c]   # handle single row
    sns.histplot(train[col], bins=40, kde=True, color="orange", ax=ax)
    ax.set_title(f"Distribution of {col}", fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")

for j in range(len(num_cols), n_rows * n_cols):
    r, c = divmod(j, n_cols)
    fig.delaxes(axes[r, c] if n_rows > 1 else axes[c])

plt.tight_layout()
plt.show()


# --- Numerical Features vs Target (Scatter Plot) ---
exclude_cols = ["diagnosed_diabetes", "id"]
target_col = "diagnosed_diabetes"

num_cols = train.select_dtypes(include=["int64", "float64"]).columns
num_cols = [c for c in num_cols if c not in exclude_cols]

n_cols = 2
n_rows = math.ceil(len(num_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

for i, col in enumerate(num_cols):
    r, c = divmod(i, n_cols)
    ax = axes[r, c] if n_rows > 1 else axes[c]   # handle single row
    
    sns.scatterplot(x=train[col], y=train[target_col], alpha=0.5, color="orange", ax=ax)
    ax.set_title(f"{col} vs {target_col}", fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel(target_col)

for j in range(len(num_cols), n_rows * n_cols):
    r, c = divmod(j, n_cols)
    fig.delaxes(axes[r, c] if n_rows > 1 else axes[c])

plt.tight_layout()
plt.show()


# --- Categorical Features vs Numeric Target ---
exclude_cols = ["id"]   # contoh exclude
target_col = "diagnosed_diabetes"  # target numerik

cat_cols = train.select_dtypes(include=["object"]).columns
cat_cols = [c for c in cat_cols if c not in exclude_cols]

n_cols = 2
n_rows = math.ceil(len(cat_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

for i, col in enumerate(cat_cols):
    r, c = divmod(i, n_cols)
    ax = axes[r, c] if n_rows > 1 else axes[c]
    
    sns.boxplot(x=train[col], y=train[target_col], ax=ax)
    ax.set_title(f"{target_col} by {col}", fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel(target_col)
    ax.tick_params(axis="x", rotation=45)

# hapus subplot kosong
for j in range(len(cat_cols), n_rows * n_cols):
    r, c = divmod(j, n_cols)
    fig.delaxes(axes[r, c] if n_rows > 1 else axes[c])

plt.tight_layout()
plt.show()


def create_comprehensive_features(df, is_train=True):
    """
    Create comprehensive medical and lifestyle features
    Compatible with both train (with target) and test (without target) data
    """
    df = df.copy()
    
    # Define original columns WITHOUT target
    original_cols = [
        'id', 'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
        'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day',
        'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp',
        'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
        'triglycerides', 'gender', 'ethnicity', 'education_level', 'income_level',
        'smoking_status', 'employment_status', 'family_history_diabetes',
        'hypertension_history', 'cardiovascular_history'
    ]
    
    # Add target column if it's training data
    if is_train:
        original_cols.append('diagnosed_diabetes')
    
    # === 1. MEDICAL COMPOSITE INDICATORS ===
    
    # Blood Pressure Categories
    df['bp_category'] = pd.cut(df['systolic_bp'],
                               bins=[0, 90, 120, 130, 140, 180, 300],
                               labels=['hypo', 'normal', 'elevated', 'stage1', 'stage2', 'crisis'])
    
    # Cholesterol Ratio (important predictor)
    df['chol_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
    
    # Metabolic Syndrome Indicators
    df['has_high_bp'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 85)).astype(int)
    df['has_high_triglycerides'] = (df['triglycerides'] >= 150).astype(int)
    
    # Fix: Handle gender comparison properly
    df['has_low_hdl'] = ((df['gender'] == 'Male') & (df['hdl_cholesterol'] < 40)) | \
                        ((df['gender'] != 'Male') & (df['hdl_cholesterol'] < 50))
    df['has_low_hdl'] = df['has_low_hdl'].astype(int)
    
    df['has_high_waist'] = ((df['gender'] == 'Male') & (df['waist_to_hip_ratio'] > 0.9)) | \
                           ((df['gender'] != 'Male') & (df['waist_to_hip_ratio'] > 0.85))
    df['has_high_waist'] = df['has_high_waist'].astype(int)
    
    # Metabolic Syndrome Score
    df['metabolic_syndrome_score'] = (df['has_high_bp'] + 
                                      df['has_high_triglycerides'] + 
                                      df['has_low_hdl'] + 
                                      df['has_high_waist'])
    
    # === 2. LIFESTYLE COMPOSITE SCORES ===
    
    # Physical Activity Score (inverse relationship with diabetes risk)
    # Fix: Handle potential division by zero
    df['activity_score'] = np.log1p(df['physical_activity_minutes_per_week']) * 0.4 + \
                          df['diet_score'] * 0.3 + \
                          (8 - abs(7 - df['sleep_hours_per_day'])) * 0.3
    
    # Sedentary Lifestyle Score
    df['sedentary_score'] = df['screen_time_hours_per_day'] * 0.5 + \
                           (1 / (df['physical_activity_minutes_per_week'] + 1)) * 0.3 + \
                           df['alcohol_consumption_per_week'] * 0.2
    
    # Overall Lifestyle Risk Score
    # Fix: Add small epsilon to prevent division by zero
    df['lifestyle_risk_score'] = (df['sedentary_score'] * 0.4 + 
                                  (10 - df['diet_score']) * 0.3 + 
                                  (1 / (df['activity_score'] + 1e-10)) * 0.3)
    
    # === 3. PHYSIOLOGICAL INTERACTIONS ===
    
    # Age-adjusted metrics
    df['bmi_age_adjusted'] = df['bmi'] * (df['age'] / 50)  # Normalize to age 50
    df['bp_age_adjusted'] = df['systolic_bp'] * (df['age'] / 50)
    
    # BMI-Waist Ratio Interaction
    df['bmi_waist_interaction'] = df['bmi'] * df['waist_to_hip_ratio']
    
    # Cholesterol-BMI Interaction
    df['chol_bmi_risk'] = df['ldl_cholesterol'] * (df['bmi'] / 25)
    
    # === 4. CARDIOVASCULAR RISK INDICATORS ===
    
    # Fix: Handle missing smoking_status categories gracefully
    smoking_mapping = {'never': 0, 'former': 1, 'current': 2}
    # Use get method to handle unknown categories
    df['smoking_numeric'] = df['smoking_status'].map(lambda x: smoking_mapping.get(x, 0))
    
    # Framingham-like Risk Score (simplified)
    df['cv_risk_score'] = (df['age'] / 10 * 1.0 + 
                          (df['systolic_bp'] - 120) / 10 * 0.5 +
                          (df['cholesterol_total'] - 200) / 10 * 0.3 +
                          df['smoking_numeric'] * 1.5)
    
    # Pulse Pressure (indicator of arterial stiffness)
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    
    # Mean Arterial Pressure
    df['map'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
    
    # === 5. METABOLIC HEALTH INDICATORS ===
    
    # Insulin Resistance Proxy
    df['trig_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1)
    
    # Visceral Fat Index Proxy
    df['visceral_fat_index'] = df['waist_to_hip_ratio'] * df['bmi'] * (df['age'] / 50)
    
    # Metabolic Age
    df['metabolic_age'] = df['age'] * (1 + (df['bmi'] - 22) * 0.02 + (df['chol_hdl_ratio'] - 3.5) * 0.1)
    
    # === 6. BEHAVIORAL PATTERNS ===
    
    # Sleep Quality Score
    df['sleep_quality'] = np.where(
        (df['sleep_hours_per_day'] >= 7) & (df['sleep_hours_per_day'] <= 9), 3,
        np.where((df['sleep_hours_per_day'] >= 6) & (df['sleep_hours_per_day'] <= 10), 2, 1)
    )
    
    # Screen Time Impact Score
    df['screen_impact'] = df['screen_time_hours_per_day'] * (1 - (df['physical_activity_minutes_per_week'] / 3000))
    
    # Alcohol Risk Categories
    df['alcohol_risk'] = pd.cut(df['alcohol_consumption_per_week'],
                                bins=[-1, 0, 7, 14, 100],
                                labels=['none', 'low', 'moderate', 'high'])
    
    # === 7. SOCIO-DEMOGRAPHIC RISK FACTORS ===
    
    # Create risk encoding for categorical variables with default values
    risk_factors = {
        'education_level': {'Less than high school': 3, 'High school': 2, 
                           'Some college': 1, 'College': 0, 'Postgraduate': 0},
        'income_level': {'Low': 2, 'Lower middle': 1, 'Upper middle': 0, 'High': 0},
        'employment_status': {'Unemployed': 2, 'Part-time': 1, 'Full-time': 0, 'Retired': 1}
    }
    
    for col, mapping in risk_factors.items():
        df[f'{col}_risk'] = df[col].map(mapping).fillna(0).astype(int)  # Fill missing with 0
    
    # Combined Socioeconomic Risk
    df['socioeconomic_risk'] = df['education_level_risk'] + df['income_level_risk'] + df['employment_status_risk']
    
    # === 8. FAMILY & GENETIC RISK ===
    
    # Genetic Loading Score
    df['genetic_risk_score'] = df['family_history_diabetes'] * 2 + \
                              df['hypertension_history'] * 1 + \
                              df['cardiovascular_history'] * 1.5
    
    # Combined Hereditary Risk
    df['hereditary_burden'] = (df['family_history_diabetes'] * 0.4 + 
                              df['hypertension_history'] * 0.3 + 
                              df['cardiovascular_history'] * 0.3)
    
    # === 9. POLYNOMIAL & NONLINEAR FEATURES ===
    
    # Quadratic terms for key continuous variables
    for col in ['bmi', 'age', 'waist_to_hip_ratio', 'chol_hdl_ratio']:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_log'] = np.log1p(np.abs(df[col]))  # Use abs to handle negative values
    
    # Interaction between age and all risk factors
    df['age_risk_multiplier'] = df['age'] * df['genetic_risk_score'] * df['lifestyle_risk_score'] / 1000
    
    # === 10. CLINICAL RISK CATEGORIES ===
    
    # Diabetes Risk Stratification
    conditions = [
        (df['metabolic_syndrome_score'] >= 3) & (df['genetic_risk_score'] >= 2),
        (df['metabolic_syndrome_score'] >= 2) | (df['genetic_risk_score'] >= 2),
        (df['metabolic_syndrome_score'] >= 1) | (df['genetic_risk_score'] >= 1),
        (df['metabolic_syndrome_score'] == 0) & (df['genetic_risk_score'] == 0)
    ]
    choices = ['very_high', 'high', 'moderate', 'low']
    df['diabetes_risk_stratification'] = np.select(conditions, choices, default='low')
    
    # === 11. CREATE DUMMY VARIABLES ===
    
    # Categorical columns to encode
    categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
                       'smoking_status', 'employment_status', 'bp_category',
                       'alcohol_risk', 'diabetes_risk_stratification']
    
    # One-hot encode with consistent columns
    all_dummies = pd.DataFrame(index=df.index)
    
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            all_dummies = pd.concat([all_dummies, dummies], axis=1)
    
    # Drop original categorical columns
    df = df.drop(categorical_cols, axis=1, errors='ignore')
    
    # Add all dummies at once
    df = pd.concat([df, all_dummies], axis=1)
    
    # === 12. FINAL ENGINEERED FEATURES SUMMARY ===
    
    # Get engineered features by comparing with original columns
    current_cols = set(df.columns)
    original_set = set(original_cols)
    
    engineered_features = list(current_cols - original_set)
    
    if is_train:
        print(f"Training data - Total features created: {len(engineered_features)}")
        print(f"Training data - Total dataset features: {len(df.columns)}")
    else:
        print(f"Test data - Total features created: {len(engineered_features)}")
        print(f"Test data - Total dataset features: {len(df.columns)}")
    
    return df, engineered_features


# Apply feature engineering with different is_train parameters
df_engineered_train, engineered_features_train = create_comprehensive_features(train, is_train=True)
df_engineered_test, engineered_features_test = create_comprehensive_features(test, is_train=False)

# Ensure that the columns in the train and test are the same.
# Get the same columns in both datasets
common_columns = set(df_engineered_train.columns) & set(df_engineered_test.columns)

# Make sure we don't include targets in the test
common_columns = [col for col in common_columns if col != 'diagnosed_diabetes']

# Sort the columns so they are the same
df_engineered_train = df_engineered_train[['id', 'diagnosed_diabetes'] + sorted(common_columns)]
df_engineered_test = df_engineered_test[['id'] + sorted(common_columns)]

print(f"\nTraining shape: {df_engineered_train.shape}")
print(f"Test shape: {df_engineered_test.shape}")
print(f"Common features: {len(common_columns)}")


def build_diabetes_models(df, final_features, target_col='diagnosed_diabetes', 
                         test_size=0.2, random_state=101, use_shap=False):
    """
    Build and evaluate models for diabetes prediction
    """
    # --- Data Preparation ---
    X = df[final_features]
    y = df[target_col]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {len(final_features)}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    print(f"\nTraining: {X_train.shape}, Test: {X_test.shape}")
    
    # --- Models for Diabetes Prediction ---
    models = {
        "RandomForest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=random_state, eval_metric='logloss', n_jobs=-1),
        "LightGBM": LGBMClassifier(random_state=random_state, verbose=-1, n_jobs=-1),
        "LogisticRegression": LogisticRegression(random_state=random_state, max_iter=1000, n_jobs=-1)
    }
    
    # --- Optimized Parameter Grids for Medical Data ---
    param_grids = {
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced"]
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "gamma": [0, 0.1, 0.2],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [1, 1.5, 2],
            "scale_pos_weight": [1]  # Akan dihitung otomatis nanti
        },
        "LightGBM": {
            "n_estimators": [100, 200, 300],
            "num_leaves": [31, 50, 70],
            "learning_rate": [0.01, 0.05, 0.1],
            "feature_fraction": [0.8, 0.9, 1.0],
            "bagging_fraction": [0.8, 0.9, 1.0],
            "min_child_samples": [10, 20, 30]
        },
        "LogisticRegression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ['l1', 'l2'],
            "solver": ['liblinear', 'saga']
        }
    }
    
    # Handle imbalanced data
    class_ratio = len(y_train[y_train==0]) / len(y_train[y_train==1])
    param_grids["XGBoost"]["scale_pos_weight"] = [1, class_ratio]
    
    # --- Cross-validation ---
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    
    # --- Results storage ---
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"=== TRAINING {name} ===")
        print(f"{'='*60}")
        
        # --- Hyperparameter Tuning ---
        print(f"Hyperparameter tuning for {name}...")
        
        # Gunakan RandomizedSearchCV untuk lebih cepat
        grid = RandomizedSearchCV(
            model, param_grids[name],
            cv=kf, scoring='roc_auc',
            n_iter=10,  # Reduced for speed
            n_jobs=-1, verbose=0,
            random_state=random_state)
        
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        best_score = grid.best_score_
        
        print(f"Best CV Score (ROC-AUC): {best_score:.4f}")
        print(f"Best Parameters: {grid.best_params_}")
        
        # --- Predictions ---
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)
        
        # --- Evaluation Metrics ---
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        print(f"\nTest Performance:")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Avg Precision: {avg_precision:.4f}")
        
        # --- Store Results ---
        results[name] = {
            "model": best_model,
            "cv_score": best_score,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_precision": avg_precision,
            "best_params": grid.best_params_,
            "y_pred_proba": y_pred_proba,
            "y_pred": y_pred
        }
        
        # --- Visualization: ROC Curve ---
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # --- Feature Importance (for tree-based models) ---
        if hasattr(best_model, "feature_importances_"):
            importance = best_model.feature_importances_
            feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importance})
            feat_imp = feat_imp.sort_values("Importance", ascending=False)
            
            # Plot top 15 features
            plt.figure(figsize=(10, 6))
            top_features = feat_imp.head(15)
            plt.barh(range(len(top_features)), top_features['Importance'])
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Importance')
            plt.title(f'Top 15 Feature Importance - {name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            # Print top 10 features
            print(f"\nTop 10 Features for {name}:")
            for i, row in feat_imp.head(10).iterrows():
                print(f"  {row['Feature']}: {row['Importance']:.4f}")
        
        # --- SHAP Analysis (Optional) ---
        if use_shap and name in ["XGBoost", "RandomForest", "LightGBM"]:
            try:
                print(f"\nCalculating SHAP values for {name}...")
                
                # Sample data untuk efisiensi
                sample_size = min(500, len(X_test))
                X_test_sample = X_test.iloc[:sample_size]
                
                # Calculate SHAP values
                explainer = shap.TreeExplainer(best_model)
                shap_values = explainer.shap_values(X_test_sample)
                
                # Handle binary classification
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]  # Use class 1
                
                # Summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_test_sample, 
                                 feature_names=X.columns.tolist(), 
                                 show=False, max_display=20)
                plt.title(f'SHAP Summary - {name}')
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"SHAP analysis skipped: {str(e)}")
    
    # --- Model Comparison ---
    print(f"\n{'='*60}")
    print("=== MODEL COMPARISON ===")
    print(f"{'='*60}")
    
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'CV_Score': result['cv_score'],
            'ROC_AUC': result['roc_auc'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1_Score': result['f1'],
            'Avg_Precision': result['avg_precision']
        })
    
    comparison_df = pd.DataFrame(comparison_data).round(4)
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Visual comparison
    metrics = ['ROC_AUC', 'Accuracy', 'Precision', 'Recall', 'F1_Score']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics[:6]):
        if idx < len(axes):
            ax = axes[idx]
            values = comparison_df[metric]
            models_list = comparison_df['Model']
            
            bars = ax.bar(models_list, values)
            ax.set_title(metric.replace('_', ' '))
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return results, comparison_df


def predict_diabetes_probability(model_results, X_new, model_name=None):
    """
    Predict diabetes probability for new data
    
    Parameters:
    -----------
    model_results : dict from build_diabetes_models
    X_new : DataFrame with same features as training
    model_name : which model to use (None = best by ROC-AUC)
    
    Returns:
    --------
    probabilities : array of probabilities for diabetes (class 1)
    """
    
    # --- Select Model ---
    if model_name is None:
        # Find model with highest ROC-AUC
        best_model_name = max(model_results.keys(), 
                            key=lambda x: model_results[x]['roc_auc'])
        print(f"Using best model: {best_model_name} "
              f"(ROC-AUC: {model_results[best_model_name]['roc_auc']:.4f})")
        model = model_results[best_model_name]['model']
    else:
        if model_name not in model_results:
            available = list(model_results.keys())
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")
        model = model_results[model_name]['model']
        print(f"Using model: {model_name}")
    
    # --- Align Features ---
    # Get training features from model
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
    else:
        expected_features = list(X_new.columns)
        print("Note: Using X_new columns as features")
    
    # Check and align features
    missing = set(expected_features) - set(X_new.columns)
    if missing:
        print(f"Warning: {len(missing)} features missing. Adding with zeros.")
        for feat in missing:
            X_new[feat] = 0
    
    # Use only expected features in correct order
    X_aligned = X_new[expected_features].copy()
    
    # --- Predict ---
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_aligned)[:, 1]
    else:
        probabilities = model.predict(X_aligned)  # For models without predict_proba
    
    return probabilities


def create_submission_file(test_ids, probabilities, filename='submission.csv'):
    """
    Create Kaggle submission file
    """
    submission = pd.DataFrame({
        'id': test_ids,
        'diagnosed_diabetes': probabilities
    })
    
    submission.to_csv(filename, index=False)
    
    print(f"✅ Submission saved: {filename}")
    print(f"   Samples: {len(submission)}")
    print(f"   Probability range: {probabilities.min():.4f} - {probabilities.max():.4f}")
    print(f"   Mean probability: {probabilities.mean():.4f}")
    
    return submission


def complete_diabetes_pipeline(train_df, test_df, 
                              use_engineered_features=True, 
                              test_size=0.2, random_state=42):
    """
    Complete pipeline from feature engineering to submission
    """
    print("=" * 60)
    print("DIABETES PREDICTION PIPELINE")
    print("=" * 60)
    
    # --- 1. Feature Engineering ---
    print("\n1. Feature Engineering...")
    
    if use_engineered_features:
        # Gunakan fungsi feature engineering yang sudah dibuat
        df_train_engineered, _ = create_comprehensive_features(train_df, is_train=True)
        df_test_engineered, _ = create_comprehensive_features(test_df, is_train=False)
        
        # Pastikan kolom sama
        train_cols = [col for col in df_train_engineered.columns 
                     if col not in ['id', 'diagnosed_diabetes']]
        test_cols = [col for col in df_test_engineered.columns 
                    if col != 'id']
        
        common_cols = list(set(train_cols) & set(test_cols))
        
        X_train = df_train_engineered[common_cols]
        y_train = df_train_engineered['diagnosed_diabetes']
        X_test = df_test_engineered[common_cols]
        
        print(f"   Engineered features: {len(common_cols)}")
    else:
        # Gunakan features original saja
        X_train = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
        y_train = train_df['diagnosed_diabetes']
        X_test = test_df.drop('id', axis=1)
        
        print(f"   Original features: {len(X_train.columns)}")
    
    print(f"   Training data: {X_train.shape}")
    print(f"   Test data: {X_test.shape}")
    
    # --- 2. Prepare Data for Modeling ---
    print("\n2. Preparing data for modeling...")
    
    # Gabungkan X dan y untuk modeling function
    df_for_modeling = X_train.copy()
    df_for_modeling['diagnosed_diabetes'] = y_train
    
    final_features = list(X_train.columns)
    
    # --- 3. Train and Evaluate Models ---
    print("\n3. Training and evaluating models...")
    
    model_results, comparison_df = build_diabetes_models(
        df=df_for_modeling,
        final_features=final_features,
        target_col='diagnosed_diabetes',
        test_size=test_size,
        random_state=random_state,
        use_shap=False  # Set True jika ingin SHAP analysis
    )
    
    # --- 4. Predict on Test Data ---
    print("\n4. Predicting on test data...")
    
    # Pilih model terbaik
    best_model_name = comparison_df.loc[comparison_df['ROC_AUC'].idxmax(), 'Model']
    print(f"   Best model: {best_model_name}")
    
    # Predict probabilities
    test_probabilities = predict_diabetes_probability(
        model_results=model_results,
        X_new=X_test,
        model_name=best_model_name
    )
    
    # --- 5. Create Submission ---
    print("\n5. Creating submission file...")
    
    submission = create_submission_file(
        test_ids=test_df['id'],
        probabilities=test_probabilities,
        filename='diabetes_submission.csv'
    )
    
    # --- 6. Additional Analysis (Optional) ---
    print("\n6. Additional analysis...")
    
    # Probability distribution
    plt.figure(figsize=(10, 6))
    plt.hist(test_probabilities, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution on Test Data')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Top features from best model
    best_model = model_results[best_model_name]['model']
    if hasattr(best_model, 'feature_importances_'):
        feat_imp = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop 10 features from {best_model_name}:")
        for i, row in feat_imp.head(10).iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.4f}")
    
    return {
        'model_results': model_results,
        'comparison_df': comparison_df,
        'test_probabilities': test_probabilities,
        'submission': submission,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train
    }


# Run complete pipeline
results = complete_diabetes_pipeline(
    train_df=train,
    test_df=test,
    use_engineered_features=True,  # Gunakan feature engineering
    test_size=0.2,
    random_state=42
)

# Atau jika ingin step by step:
# 1. Feature engineering
df_train_engineered, _ = create_comprehensive_features(train, is_train=True)
df_test_engineered, _ = create_comprehensive_features(test, is_train=False)

# 2. Prepare features
train_features = [col for col in df_train_engineered.columns 
                 if col not in ['id', 'diagnosed_diabetes']]
test_features = [col for col in df_test_engineered.columns 
                if col != 'id']

common_features = list(set(train_features) & set(test_features))

X_train = df_train_engineered[common_features]
y_train = df_train_engineered['diagnosed_diabetes']
X_test = df_test_engineered[common_features]

# 3. Modeling
df_model = X_train.copy()
df_model['diagnosed_diabetes'] = y_train

model_results, comparison_df = build_diabetes_models(
    df=df_model,
    final_features=common_features,
    target_col='diagnosed_diabetes',
    test_size=0.2,
    random_state=42
)

# 4. Predict
probabilities = predict_diabetes_probability(model_results, X_test)

# 5. Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': probabilities
})
submission.to_csv('final_submission.csv', index=False)

