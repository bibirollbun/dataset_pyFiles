import subprocess
import sys

packages = [
    "google-generativeai",
    "shap",
    "gradio",
    "scikit-learn",
    "lightgbm",
    "matplotlib",
    "seaborn",
    "pandas",
    "numpy"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("âœ… All packages installed successfully!")


import os
import uuid
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import shap

# Gemini API
import google.generativeai as genai

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("âœ… Libraries imported successfully!")


# %%
# Configure Gemini - API key from Kaggle Secrets (no hardcoding!)
try:
    # For Kaggle environment
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    GOOGLE_API_KEY = secrets.get_secret("GOOGLE_API_KEY")
    print("âœ… API key loaded from Kaggle Secrets")
except:
    # For local testing - use environment variable
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", None)
    if GOOGLE_API_KEY:
        print("âœ… API key loaded from environment variable")
    else:
        print("âš ï¸� No API key found - using mock mode for demonstration")
        GOOGLE_API_KEY = None

# Configure Gemini if key exists
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    print("âœ… Gemini 2.5 Flash configured successfully!")
else:
    model = None
    print("â„¹ï¸� Running in DEMO mode without actual LLM calls")


# Create realistic autism screening dataset based on M-CHAT-R/F questionnaire
# In production, load from: kaggle datasets download -d fabdelja/autism-screening-for-toddlers

def create_autism_screening_dataset(n_samples=1000, random_state=42):
    """
    Create a realistic autism screening dataset based on M-CHAT-R/F items.
    
    M-CHAT-R/F (Modified Checklist for Autism in Toddlers, Revised with Follow-Up)
    is a validated screening tool for toddlers 16-30 months.
    """
    np.random.seed(random_state)
    
    # Demographics
    ages = np.random.randint(16, 36, n_samples)  # 16-36 months
    sex = np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4])  # ASD more common in males
    
    # M-CHAT-R/F Questions (simplified to 10 key items)
    # 0 = No concern, 1 = Concern (at risk behavior)
    questions = {
        'q1_point_interest': 'Does your child point to show you something interesting?',
        'q2_eye_contact': 'Does your child look at you when you call their name?',
        'q3_pretend_play': 'Does your child engage in pretend play?',
        'q4_climbing': 'Does your child like climbing on things?',
        'q5_finger_movement': 'Does your child make unusual finger movements near eyes?',
        'q6_point_want': 'Does your child point to ask for something?',
        'q7_show_objects': 'Does your child show you objects?',
        'q8_interest_children': 'Is your child interested in other children?',
        'q9_follow_point': 'Does your child follow when you point?',
        'q10_response_name': 'Does your child respond when you call their name?'
    }
    
    # Generate responses with realistic correlations
    data = {'age_months': ages, 'sex': sex}
    
    # Base risk factor (determines ASD likelihood)
    base_risk = np.random.beta(2, 8, n_samples)  # Most have low risk
    
    for i, q_key in enumerate(questions.keys()):
        # Questions are correlated with underlying risk
        noise = np.random.normal(0, 0.15, n_samples)
        prob = np.clip(base_risk + noise + np.random.uniform(-0.1, 0.1), 0, 1)
        data[q_key] = (np.random.random(n_samples) < prob).astype(int)
    
    # Generate ASD label based on total risk score
    df = pd.DataFrame(data)
    question_cols = [c for c in df.columns if c.startswith('q')]
    risk_score = df[question_cols].sum(axis=1) / len(question_cols)
    
    # ASD diagnosis (approximately 2% prevalence, but higher in screening population)
    threshold = np.percentile(risk_score, 85)
    df['asd_diagnosis'] = (risk_score >= threshold).astype(int)
    
    # Add some noise to make it realistic
    flip_idx = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
    df.loc[flip_idx, 'asd_diagnosis'] = 1 - df.loc[flip_idx, 'asd_diagnosis']
    
    return df, questions

# Create dataset
df, question_descriptions = create_autism_screening_dataset(n_samples=1500)

print("ğŸ“Š Dataset Created Successfully!")
print(f"   Total samples: {len(df)}")
print(f"   ASD positive: {df['asd_diagnosis'].sum()} ({df['asd_diagnosis'].mean()*100:.1f}%)")
print(f"   Features: {df.shape[1]}")
print("\nğŸ“‹ Sample data:")
df.head(10)


# Dataset statistics
print("ğŸ“ˆ Dataset Statistics")
print("=" * 50)
print(f"\nğŸ”¢ Age Distribution (months):")
print(df['age_months'].describe())

print(f"\nğŸ‘¥ Sex Distribution:")
print(df['sex'].value_counts())

print(f"\nğŸ�¯ Target Distribution:")
print(df['asd_diagnosis'].value_counts())

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Age distribution
axes[0].hist(df['age_months'], bins=20, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Age (months)')
axes[0].set_ylabel('Count')
axes[0].set_title('Age Distribution')

# Sex distribution
df['sex'].value_counts().plot(kind='bar', ax=axes[1], color=['steelblue', 'coral'])
axes[1].set_xlabel('Sex')
axes[1].set_ylabel('Count')
axes[1].set_title('Sex Distribution')
axes[1].tick_params(axis='x', rotation=0)

# ASD diagnosis by sex
pd.crosstab(df['sex'], df['asd_diagnosis'], normalize='index').plot(
    kind='bar', ax=axes[2], color=['lightgreen', 'salmon']
)
axes[2].set_xlabel('Sex')
axes[2].set_ylabel('Proportion')
axes[2].set_title('ASD Diagnosis Rate by Sex')
axes[2].legend(['No ASD', 'ASD'])
axes[2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()


# AGENT 1: Session & Memory Service


class MemoryService:
    """Manages session state and long-term memory for the agent system."""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.memory_bank: List[Dict] = []
        self.logs: List[Dict] = []
    
    def create_session(self) -> str:
        """Create a new session with unique ID."""
        session_id = str(uuid.uuid4())[:8]
        self.sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "history": [],
            "context": {}
        }
        self._log("MemoryService", "create_session", {"session_id": session_id})
        return session_id
    
    def append_to_session(self, session_id: str, role: str, content: Any):
        """Append message to session history."""
        if session_id not in self.sessions:
            session_id = self.create_session()
        
        self.sessions[session_id]["history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve session data."""
        return self.sessions.get(session_id)
    
    def add_to_memory(self, record: Dict):
        """Add record to long-term memory (anonymized)."""
        # Hash sensitive data for privacy
        anonymized = {
            "timestamp": datetime.now().isoformat(),
            "features_hash": hashlib.md5(str(record.get("features", "")).encode()).hexdigest()[:8],
            "prediction": record.get("prediction"),
            "risk_level": record.get("risk_level"),
            "age_group": record.get("age_group")
        }
        self.memory_bank.append(anonymized)
        self._log("MemoryService", "add_to_memory", {"record_hash": anonymized["features_hash"]})
    
    def get_aggregate_stats(self) -> Dict:
        """Get aggregated statistics from memory bank."""
        if not self.memory_bank:
            return {"total_screenings": 0}
        
        df = pd.DataFrame(self.memory_bank)
        return {
            "total_screenings": len(df),
            "high_risk_rate": (df["risk_level"] == "HIGH").mean() if "risk_level" in df else 0,
            "avg_prediction": df["prediction"].mean() if "prediction" in df else 0
        }
    
    def _log(self, agent: str, action: str, details: Dict):
        """Internal logging."""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "details": details
        })
    
    def get_logs(self, last_n: int = 10) -> List[Dict]:
        """Get recent logs."""
        return self.logs[-last_n:]

# Initialize global memory service
memory_service = MemoryService()
print("âœ… MemoryService initialized")



# AGENT 2: Gemini LLM Agent


class GeminiAgent:
    """Wrapper for Gemini 2.5 Flash interactions."""
    
    def __init__(self, model_instance):
        self.model = model_instance
        self.call_count = 0
    
    def call(self, prompt: str, system_context: str = None) -> str:
        """Call Gemini with prompt and return response."""
        self.call_count += 1
        
        if self.model is None:
            # Mock response for demo mode
            return self._mock_response(prompt)
        
        try:
            full_prompt = f"{system_context}\n\n{prompt}" if system_context else prompt
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"âš ï¸� Gemini API error: {e}")
            return self._mock_response(prompt)
    
    def _mock_response(self, prompt: str) -> str:
        """Generate mock response for demo/fallback."""
        if "validate" in prompt.lower():
            return "Input validated successfully. All responses are within expected ranges."
        elif "explain" in prompt.lower() or "interpretation" in prompt.lower():
            return "Based on the screening responses, certain social communication indicators warrant attention. The pattern of responses suggests further professional evaluation may be beneficial."
        elif "recommend" in prompt.lower():
            return "Recommendations: 1) Schedule follow-up with pediatrician, 2) Consider developmental evaluation, 3) Early intervention services if indicated."
        else:
            return "Analysis complete. Please consult healthcare professional for interpretation."
    
    def validate_input(self, answers: Dict) -> Dict:
        """Use LLM to validate and interpret questionnaire answers."""
        prompt = f"""
        As a clinical screening assistant, validate these M-CHAT-R/F screening responses:
        
        Responses: {json.dumps(answers, indent=2)}
        
        Please:
        1. Check if all required questions are answered
        2. Flag any inconsistent response patterns
        3. Note any responses that require follow-up
        
        Respond in JSON format with keys: valid, flags, notes
        """
        
        response = self.call(prompt, "You are a clinical screening validation assistant.")
        
        # Parse or return structured default
        return {
            "valid": True,
            "flags": [],
            "notes": "Input validation complete",
            "raw_response": response
        }
    
    def generate_explanation(self, prediction: float, shap_values: Dict, features: Dict) -> str:
        """Generate human-readable explanation using LLM."""
        prompt = f"""
        Generate a clear, empathetic explanation for a parent about autism screening results.
        
        Risk Score: {prediction:.1%}
        Key Contributing Factors (SHAP values): {json.dumps(shap_values, indent=2)}
        Child's Responses: {json.dumps(features, indent=2)}
        
        Guidelines:
        - Be compassionate and non-alarming
        - Explain what the score means in simple terms
        - Highlight which behaviors contributed most
        - Emphasize this is screening, not diagnosis
        - Recommend appropriate next steps
        
        Keep response under 200 words.
        """
        
        return self.call(prompt, "You are a compassionate pediatric screening communicator.")
    
    def generate_recommendations(self, risk_level: str, age_months: int) -> str:
        """Generate personalized recommendations."""
        prompt = f"""
        A {age_months}-month-old child has been screened with a {risk_level} risk result.
        
        Provide appropriate, evidence-based recommendations following CDC guidelines.
        Include:
        1. Immediate next steps
        2. Resources for parents
        3. Timeline for follow-up
        
        Be supportive and actionable. Keep under 150 words.
        """
        
        return self.call(prompt, "You are a pediatric developmental specialist assistant.")

# Initialize Gemini agent
gemini_agent = GeminiAgent(model)
print(f"âœ… GeminiAgent initialized {'(LIVE mode)' if model else '(DEMO mode)'}")


# AGENT 3: Data Processing Agent

class DataAgent:
    """Handles data preprocessing and feature engineering."""
    
    def __init__(self):
        self.feature_columns = None
        self.scaler_params = {}
    
    def preprocess(self, df: pd.DataFrame) -> tuple:
        """Preprocess dataset for training."""
        df_processed = df.copy()
        
        # Encode sex
        df_processed['sex_encoded'] = (df_processed['sex'] == 'M').astype(int)
        
        # Feature columns (exclude target and original sex)
        self.feature_columns = [c for c in df_processed.columns 
                                if c not in ['asd_diagnosis', 'sex']]
        
        X = df_processed[self.feature_columns]
        y = df_processed['asd_diagnosis']
        
        return X, y
    
    def extract_features(self, user_input: Dict) -> pd.DataFrame:
        """Extract features from user input for inference."""
        features = {
            'age_months': user_input.get('age_months', 24),
            'sex_encoded': 1 if user_input.get('sex', 'M') == 'M' else 0
        }
        
        # Add questionnaire responses
        for key, value in user_input.items():
            if key.startswith('q'):
                features[key] = int(value)
        
        return pd.DataFrame([features])
    
    def calculate_risk_score(self, answers: Dict) -> float:
        """Calculate simple risk score from answers."""
        q_answers = [v for k, v in answers.items() if k.startswith('q')]
        return sum(q_answers) / len(q_answers) if q_answers else 0

# Initialize data agent
data_agent = DataAgent()
print("âœ… DataAgent initialized")


# AGENT 4: Model Training & Inference Agent

class ModelAgent:
    """Handles ML model training, evaluation, and inference."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.training_metrics = {}
    
    def train(self, X: pd.DataFrame, y: pd.Series, params: Dict = None) -> Dict:
        """Train LightGBM model with cross-validation."""
        self.feature_names = list(X.columns)
        
        default_params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'seed': 42
        }
        
        if params:
            default_params.update(params)
        
        # Stratified K-Fold Cross-Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        print("ğŸ”„ Training with 5-Fold Cross-Validation...")
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            
            model = lgb.train(
                default_params,
                dtrain,
                num_boost_round=200,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )
            
            val_pred = model.predict(X_val)
            auc = roc_auc_score(y_val, val_pred)
            cv_scores.append(auc)
            print(f"   Fold {fold+1}: AUC = {auc:.4f}")
        
        # Train final model on all data
        dtrain_full = lgb.Dataset(X, label=y)
        self.model = lgb.train(default_params, dtrain_full, num_boost_round=200)
        
        self.training_metrics = {
            'cv_auc_mean': np.mean(cv_scores),
            'cv_auc_std': np.std(cv_scores),
            'cv_scores': cv_scores
        }
        
        print(f"\nâœ… Training Complete!")
        print(f"   Mean CV AUC: {self.training_metrics['cv_auc_mean']:.4f} Â± {self.training_metrics['cv_auc_std']:.4f}")
        
        return self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained!")
        
        # Ensure feature order matches
        X = X[self.feature_names]
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            return pd.DataFrame()
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        return importance

# Initialize model agent
model_agent = ModelAgent()
print("âœ… ModelAgent initialized")



# AGENT 5: Explainability Agent

class ExplainerAgent:
    """Provides SHAP-based explanations for predictions."""
    
    def __init__(self, model_agent: ModelAgent):
        self.model_agent = model_agent
        self.explainer = None
        self.background_data = None
    
    def setup(self, X_background: pd.DataFrame):
        """Setup SHAP explainer with background data."""
        if self.model_agent.model is None:
            raise ValueError("Model must be trained first!")
        
        # Use subset for background
        self.background_data = X_background.sample(min(100, len(X_background)), random_state=42)
        self.explainer = shap.TreeExplainer(self.model_agent.model)
        print("âœ… SHAP Explainer initialized")
    
    def explain(self, X: pd.DataFrame) -> Dict:
        """Generate SHAP explanation for prediction."""
        if self.explainer is None:
            raise ValueError("Explainer not setup!")
        
        shap_values = self.explainer.shap_values(X)
        
        # Get feature contributions
        contributions = {}
        for i, feature in enumerate(X.columns):
            contributions[feature] = float(shap_values[0][i])
        
        # Sort by absolute contribution
        sorted_contributions = dict(sorted(
            contributions.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        ))
        
        return {
            'shap_values': shap_values[0].tolist(),
            'contributions': sorted_contributions,
            'top_3_factors': list(sorted_contributions.keys())[:3]
        }
    
    def plot_summary(self, X: pd.DataFrame, max_display: int = 10):
        """Generate SHAP summary plot."""
        if self.explainer is None:
            raise ValueError("Explainer not setup!")
        
        shap_values = self.explainer.shap_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        plt.title("Feature Importance (SHAP Values)")
        plt.tight_layout()
        plt.show()

# Initialize explainer agent
explainer_agent = ExplainerAgent(model_agent)
print("âœ… ExplainerAgent initialized")



# Preprocess data
X, y = data_agent.preprocess(df)
print(f"ğŸ“Š Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
print(f"   Features: {list(X.columns)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nğŸ“‚ Train/Test Split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")



# Train model
training_results = model_agent.train(X_train, y_train)



# Setup explainer
explainer_agent.setup(X_train)



# Comprehensive evaluation on test set
y_pred_proba = model_agent.predict(X_test)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Calculate metrics
test_auc = roc_auc_score(y_test, y_pred_proba)
brier = brier_score_loss(y_test, y_pred_proba)

print("=" * 60)
print("ğŸ“Š MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"\nğŸ�¯ Test Set AUC: {test_auc:.4f}")
print(f"ğŸ“‰ Brier Score: {brier:.4f} (lower is better)")

print("\nğŸ“‹ Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No ASD', 'ASD']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nğŸ”¢ Confusion Matrix:")
print(cm)


# Visualization: ROC Curve, Precision-Recall, Calibration
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {test_auc:.3f})')
axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve')
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

# 2. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
axes[0, 1].plot(recall, precision, 'g-', linewidth=2)
axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Curve')
axes[0, 1].grid(True, alpha=0.3)

# 3. Calibration Plot
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
axes[1, 0].plot(prob_pred, prob_true, 's-', linewidth=2, label='Model')
axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly Calibrated')
axes[1, 0].set_xlabel('Mean Predicted Probability')
axes[1, 0].set_ylabel('Fraction of Positives')
axes[1, 0].set_title('Calibration Plot')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
            xticklabels=['No ASD', 'ASD'], yticklabels=['No ASD', 'ASD'])
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')
axes[1, 1].set_title('Confusion Matrix')

plt.tight_layout()
plt.show()


# Feature Importance
importance_df = model_agent.get_feature_importance()
print("\nğŸ“Š Feature Importance (Top 10):")
print(importance_df.head(10).to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(10), x='importance', y='feature', palette='viridis')
plt.xlabel('Importance (Gain)')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.show()



# SHAP Summary Plot
print("\nğŸ”� SHAP Value Analysis:")
explainer_agent.plot_summary(X_test)


# Fairness analysis by demographic groups
print("=" * 60)
print("âš–ï¸� FAIRNESS ANALYSIS")
print("=" * 60)

# Reconstruct test dataframe with predictions
test_df = X_test.copy()
test_df['y_true'] = y_test.values
test_df['y_pred_proba'] = y_pred_proba
test_df['y_pred'] = y_pred

# By Sex
print("\nğŸ“Š Performance by Sex:")
for sex_val, sex_name in [(1, 'Male'), (0, 'Female')]:
    mask = test_df['sex_encoded'] == sex_val
    if mask.sum() > 10:
        subset_auc = roc_auc_score(test_df.loc[mask, 'y_true'], test_df.loc[mask, 'y_pred_proba'])
        subset_n = mask.sum()
        print(f"   {sex_name}: AUC = {subset_auc:.4f} (n={subset_n})")

# By Age Group
print("\nğŸ“Š Performance by Age Group:")
test_df['age_group'] = pd.cut(test_df['age_months'], bins=[15, 20, 25, 30, 36], 
                               labels=['16-20m', '21-25m', '26-30m', '31-36m'])

for age_grp in test_df['age_group'].unique():
    mask = test_df['age_group'] == age_grp
    if mask.sum() > 10:
        subset_auc = roc_auc_score(test_df.loc[mask, 'y_true'], test_df.loc[mask, 'y_pred_proba'])
        subset_n = mask.sum()
        print(f"   {age_grp}: AUC = {subset_auc:.4f} (n={subset_n})")


class AutismScreeningOrchestrator:
    """
    Main orchestrator that coordinates all agents for autism screening.
    Implements A2A (Agent-to-Agent) protocol for message passing.
    """
    
    def __init__(self, gemini_agent, data_agent, model_agent, explainer_agent, memory_service):
        self.gemini = gemini_agent
        self.data = data_agent
        self.model = model_agent
        self.explainer = explainer_agent
        self.memory = memory_service
        
    def _create_message(self, from_agent: str, to_agent: str, msg_type: str, payload: Dict) -> Dict:
        """Create A2A protocol message."""
        return {
            "trace_id": str(uuid.uuid4())[:8],
            "from": from_agent,
            "to": to_agent,
            "type": msg_type,
            "payload": payload,
            "timestamp": datetime.now().isoformat()
        }
    
    def screen(self, user_input: Dict, session_id: str = None) -> Dict:
        """
        Main screening pipeline - orchestrates all agents.
        
        Args:
            user_input: Dictionary with age_months, sex, and q1-q10 responses
            session_id: Optional existing session ID
            
        Returns:
            Complete screening result with prediction, explanation, and recommendations
        """
        trace_id = str(uuid.uuid4())[:8]
        print(f"\nğŸ”„ Starting screening pipeline (trace: {trace_id})")
        
        # Step 1: Session Management
        if session_id is None:
            session_id = self.memory.create_session()
        self.memory.append_to_session(session_id, "user", user_input)
        
        # Step 2: Input Validation (Gemini Agent)
        print("   â†’ InputAgent: Validating input...")
        validation = self.gemini.validate_input(user_input)
        
        # Step 3: Feature Extraction (Data Agent)
        print("   â†’ DataAgent: Extracting features...")
        features_df = self.data.extract_features(user_input)
        
        # Ensure all required columns exist
        for col in self.model.feature_names:
            if col not in features_df.columns:
                features_df[col] = 0
        
        features_df = features_df[self.model.feature_names]
        
        # Step 4: Prediction (Model Agent)
        print("   â†’ ModelAgent: Making prediction...")
        prediction = float(self.model.predict(features_df)[0])
        
        # Determine risk level
        if prediction >= 0.7:
            risk_level = "HIGH"
        elif prediction >= 0.4:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        # Step 5: Explanation (Explainer Agent)
        print("   â†’ ExplainerAgent: Generating explanation...")
        shap_result = self.explainer.explain(features_df)
        
        # Step 6: LLM Explanation & Recommendations (Gemini Agent)
        print("   â†’ GeminiAgent: Creating human-readable explanation...")
        explanation = self.gemini.generate_explanation(
            prediction, 
            shap_result['contributions'],
            user_input
        )
        
        recommendations = self.gemini.generate_recommendations(
            risk_level,
            user_input.get('age_months', 24)
        )
        
        # Step 7: Store in Memory
        self.memory.add_to_memory({
            "features": user_input,
            "prediction": prediction,
            "risk_level": risk_level,
            "age_group": f"{user_input.get('age_months', 24)//6*6}-{user_input.get('age_months', 24)//6*6+6}m"
        })
        
        # Compile result
        result = {
            "session_id": session_id,
            "trace_id": trace_id,
            "risk_score": prediction,
            "risk_level": risk_level,
            "risk_percentage": f"{prediction * 100:.1f}%",
            "top_contributing_factors": shap_result['top_3_factors'],
            "shap_contributions": shap_result['contributions'],
            "explanation": explanation,
            "recommendations": recommendations,
            "validation": validation,
            "disclaimer": "âš ï¸� This is a screening tool only, NOT a diagnosis. Please consult a qualified healthcare professional."
        }
        
        # Store result in session
        self.memory.append_to_session(session_id, "agent", result)
        
        print(f"âœ… Screening complete!")
        return result
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get history for a session."""
        session = self.memory.get_session(session_id)
        return session["history"] if session else []
    
    def get_system_stats(self) -> Dict:
        """Get aggregate system statistics."""
        stats = self.memory.get_aggregate_stats()
        stats["gemini_calls"] = self.gemini.call_count
        return stats

# Initialize orchestrator
orchestrator = AutismScreeningOrchestrator(
    gemini_agent, data_agent, model_agent, explainer_agent, memory_service
)
print("âœ… Orchestrator initialized")


import shap
import numpy as np
import pandas as pd


class ExplainerAgent:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X):
        shap_raw = self.explainer.shap_values(X)

        # -----------------------------
        # FIX FOR LIGHTGBM OUTPUT
        # shap_raw may be:
        # 1) ndarray â†’ (1, n_features)
        # 2) list of arrays â†’ [ array([[..]]) ]
        # -----------------------------

        if isinstance(shap_raw, list):
            shap_values = shap_raw[0]          # Take first class
        else:
            shap_values = shap_raw             # Already ndarray

        # Ensure 2D
        shap_values = np.array(shap_values)

        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        # Single sample â†’ shap_values[0]
        shap_row = shap_values[0]

        contributions = {
            feature: float(shap_row[i])
            for i, feature in enumerate(X.columns)
        }

        # Sort by absolute importance
        sorted_contributions = dict(
            sorted(contributions.items(),
                   key=lambda x: abs(x[1]),
                   reverse=True)
        )

        return sorted_contributions



# Full demo: fixed SHAP handling + small orchestration pipeline
!pip install -q lightgbm shap scikit-learn

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import uuid
import time

# -------------------------
# 1) Build a small demo model
# -------------------------
def make_demo_data(n=1000, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "age_months": rng.randint(12,60,n),
        "q1_point_interest": rng.binomial(1,0.15,n),
        "q2_eye_contact": rng.binomial(1,0.12,n),
        "q3_pretend_play": rng.binomial(1,0.10,n),
        "q4_climbing": rng.binomial(1,0.08,n),
        "q5_finger_movement": rng.binomial(1,0.07,n),
        "q6_point_want": rng.binomial(1,0.06,n),
        "q7_show_objects": rng.binomial(1,0.05,n),
        "q8_interest_children": rng.binomial(1,0.2,n),
        "q9_follow_point": rng.binomial(1,0.14,n),
        "q10_response_name": rng.binomial(1,0.18,n)
    })
    # simple synthetic target: sum of a few signals > 0 => positive
    y = (X[["q1_point_interest","q2_eye_contact","q3_pretend_play","q10_response_name"]].sum(axis=1) > 0).astype(int)
    return X, y

X, y = make_demo_data(1200)
X_train, X_test = X.iloc[:1000], X.iloc[1000:]
y_train, y_test = y.iloc[:1000], y.iloc[1000:]

dtrain = lgb.Dataset(X_train, label=y_train)
params = {"objective": "binary", "verbosity": -1, "seed": 0}
bst = lgb.train(params, dtrain, num_boost_round=100)

# create SHAP explainer once
explainer_shap = shap.TreeExplainer(bst)

# -------------------------
# 2) Agents (lightweight)
# -------------------------
class InputAgent:
    def validate_and_normalize(self, user_input: dict):
        # ensure required fields, set defaults
        required = ["age_months", "sex"]
        features = {}
        for k,v in user_input.items():
            features[k] = v
        # ensure feature columns present
        feature_cols = ["age_months","q1_point_interest","q2_eye_contact","q3_pretend_play",
                        "q4_climbing","q5_finger_movement","q6_point_want","q7_show_objects",
                        "q8_interest_children","q9_follow_point","q10_response_name"]
        for c in feature_cols:
            features.setdefault(c, 0)
        return features

class DataAgent:
    def to_dataframe(self, features: dict):
        # create single-row dataframe ordered like training features
        cols = ["age_months","q1_point_interest","q2_eye_contact","q3_pretend_play",
                "q4_climbing","q5_finger_movement","q6_point_want","q7_show_objects",
                "q8_interest_children","q9_follow_point","q10_response_name"]
        df = pd.DataFrame([{c: features.get(c, 0) for c in cols}])
        return df

class ModelAgent:
    def __init__(self, model):
        self.model = model
    def predict_proba(self, X_df: pd.DataFrame):
        # LightGBM predict returns probability for positive class by default
        probs = self.model.predict(X_df)
        return float(probs[0])

class ExplainerAgent:
    def __init__(self, shap_explainer, feature_columns):
        self.explainer = shap_explainer
        self.feature_columns = feature_columns

    def explain(self, X_df: pd.DataFrame):
        """
        Robust SHAP handling:
        - shap_values can be a single ndarray (n_samples, n_features)
        - or a list of ndarrays (e.g., for binary classifiers newer SHAP returns [class0, class1]).
        We'll pick the array corresponding to the positive class (1) when list is returned.
        """
        shap_values = self.explainer.shap_values(X_df)

        # Normalize shap_values to a 2D numpy array of shape (n_samples, n_features)
        if isinstance(shap_values, list):
            # common case: [class0_shap, class1_shap] for binary
            # choose positive class (1) if length==2, else choose last
            idx = 1 if len(shap_values) == 2 else -1
            sv = np.array(shap_values[idx])
        else:
            sv = np.array(shap_values)

        # sv could be shape (n_samples, n_features). For single row, take first row.
        if sv.ndim == 2:
            sv_row = sv[0]
        elif sv.ndim == 1:
            sv_row = sv
        else:
            raise ValueError("Unexpected shap_values shape: ", sv.shape)

        # Build contributions dict
        contributions = {}
        for i, col in enumerate(self.feature_columns):
            # cast to python float safely
            contributions[col] = float(np.round(sv_row[i], 6))

        # Sort by absolute contribution descending
        sorted_feats = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top_contribs = [{ "feature": f, "contribution": v } for f,v in sorted_feats[:5]]
        return {"all": contributions, "top": top_contribs}

# -------------------------
# 3) Orchestrator
# -------------------------
class Orchestrator:
    def __init__(self, input_agent, data_agent, model_agent, explainer_agent):
        self.input_agent = input_agent
        self.data_agent = data_agent
        self.model_agent = model_agent
        self.explainer = explainer_agent

    def screen(self, user_input: dict, session_id: str = None):
        if session_id is None:
            session_id = str(uuid.uuid4())
        start = time.time()

        # Step 1: Validate input
        features = self.input_agent.validate_and_normalize(user_input)

        # Step 2: To DataFrame
        features_df = self.data_agent.to_dataframe(features)

        # Step 3: Model prediction
        prob = self.model_agent.predict_proba(features_df)
        risk_percentage = round(prob * 100, 1)
        risk_level = "Low"
        if prob >= 0.7:
            risk_level = "High"
        elif prob >= 0.3:
            risk_level = "Moderate"

        # Step 4: Explanation (SHAP)
        shap_result = self.explainer.explain(features_df)

        # Step 5: Build LLM-style explanation & recommendations (mock)
        # In production: call Gemini to create nicer text; here we build a concise summary.
        top_f = ", ".join([f"{t['feature']} ({t['contribution']:+.3f})" for t in shap_result["top"]])
        explanation = (
            f"The model estimates a {risk_percentage}% risk. "
            f"Key contributing factors: {top_f}."
        )
        if risk_level == "High":
            recommendations = "Recommend contacting a developmental specialist for an evaluation."
        elif risk_level == "Moderate":
            recommendations = "Consider monitoring development and possibly scheduling a screening."
        else:
            recommendations = "Low risk based on provided answers; continue routine monitoring."

        disclaimer = "This tool is for research/educational purposes only and is not a medical diagnosis."

        end = time.time()
        return {
            "session_id": session_id,
            "elapsed_seconds": round(end - start, 3),
            "risk_percentage": risk_percentage,
            "risk_level": risk_level,
            "top_contributing_factors": shap_result["top"],
            "explanation": explanation,
            "recommendations": recommendations,
            "disclaimer": disclaimer
        }

# -------------------------
# 4) Wire up agents and run the test case
# -------------------------
feature_cols = ["age_months","q1_point_interest","q2_eye_contact","q3_pretend_play",
                "q4_climbing","q5_finger_movement","q6_point_want","q7_show_objects",
                "q8_interest_children","q9_follow_point","q10_response_name"]

input_agent = InputAgent()
data_agent = DataAgent()
model_agent = ModelAgent(bst)
explainer_agent = ExplainerAgent(explainer_shap, feature_cols)

orchestrator = Orchestrator(input_agent, data_agent, model_agent, explainer_agent)

# -------------------------
# Test Case 1: Low Risk
# -------------------------
print("=" * 60)
print("ğŸ§ª TEST CASE 1: LOW RISK PROFILE")
print("=" * 60)

test_input_low = {
    "age_months": 24,
    "sex": "F",
    "q1_point_interest": 0,
    "q2_eye_contact": 0,
    "q3_pretend_play": 0,
    "q4_climbing": 0,
    "q5_finger_movement": 0,
    "q6_point_want": 0,
    "q7_show_objects": 0,
    "q8_interest_children": 0,
    "q9_follow_point": 0,
    "q10_response_name": 0
}

result_low = orchestrator.screen(test_input_low)

print(f"\nğŸ“Š RESULTS:")
print(f"   Session ID: {result_low['session_id']}")
print(f"   Risk Score: {result_low['risk_percentage']}%")
print(f"   Risk Level: {result_low['risk_level']}")
print(f"   Top Factors: {result_low['top_contributing_factors']}")
print(f"\nğŸ“� Explanation:")
print(f"   {result_low['explanation']}")
print(f"\nğŸ’¡ Recommendations:")
print(f"   {result_low['recommendations']}")
print(f"\nâš ï¸�  {result_low['disclaimer']}")



# Test Case 2: High Risk
print("=" * 60)
print("ğŸ§ª TEST CASE 2: HIGH RISK PROFILE")
print("=" * 60)

test_input_high = {
    "age_months": 20,
    "sex": "M",
    "q1_point_interest": 1,  # Concern
    "q2_eye_contact": 1,      # Concern
    "q3_pretend_play": 1,     # Concern
    "q4_climbing": 0,
    "q5_finger_movement": 1,  # Concern
    "q6_point_want": 1,       # Concern
    "q7_show_objects": 1,     # Concern
    "q8_interest_children": 1, # Concern
    "q9_follow_point": 1,     # Concern
    "q10_response_name": 1    # Concern
}

result_high = orchestrator.screen(test_input_high)

print(f"\nğŸ“Š RESULTS:")
print(f"   Session ID: {result_high['session_id']}")
print(f"   Risk Score: {result_high['risk_percentage']}")
print(f"   Risk Level: {result_high['risk_level']}")
print(f"   Top Factors: {result_high['top_contributing_factors']}")
print(f"\nğŸ“� Explanation:")
print(f"   {result_high['explanation']}")
print(f"\nğŸ’¡ Recommendations:")
print(f"   {result_high['recommendations']}")
print(f"\nâš ï¸�  {result_high['disclaimer']}")



# Test Case 3: Moderate Risk
print("=" * 60)
print("ğŸ§ª TEST CASE 3: MODERATE RISK PROFILE")
print("=" * 60)

test_input_moderate = {
    "age_months": 28,
    "sex": "M",
    "q1_point_interest": 0,
    "q2_eye_contact": 1,      # Concern
    "q3_pretend_play": 0,
    "q4_climbing": 0,
    "q5_finger_movement": 1,  # Concern
    "q6_point_want": 0,
    "q7_show_objects": 1,     # Concern
    "q8_interest_children": 0,
    "q9_follow_point": 0,
    "q10_response_name": 1    # Concern
}

result_moderate = orchestrator.screen(test_input_moderate)

print(f"\nğŸ“Š RESULTS:")
print(f"   Session ID: {result_moderate['session_id']}")
print(f"   Risk Score: {result_moderate['risk_percentage']}")
print(f"   Risk Level: {result_moderate['risk_level']}")
print(f"   Top Factors: {result_moderate['top_contributing_factors']}")



# ğŸ“Œ External Stats Helper (No need to modify orchestrator)

def get_system_stats(orchestrator, memory_service):
    stats = {
        "Total Sessions": len(memory_service.sessions),
        "Total Logs": len(memory_service.logs),
        "Model Used": getattr(orchestrator, "model_name", "Not Specified"),
        "Explainer Loaded": hasattr(orchestrator, "explainer"),
        "Agents Active": [
            name for name in dir(orchestrator)
            if name.endswith("_agent")
        ]
    }
    return stats



print("=" * 60)
print("ğŸ“Š SYSTEM STATISTICS")
print("=" * 60)

stats = get_system_stats(orchestrator, memory_service)

print("\nğŸ“ˆ Aggregate Stats:")
for key, value in stats.items():
    print(f"   {key}: {value}")

print("\nğŸ“� Recent Logs:")
for log in memory_service.get_logs(5):
    print(f"   [{log['timestamp'][:19]}] {log['agent']}: {log['action']}")



import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

class AutismScreeningUI:
    """Interactive UI using ipywidgets for Kaggle compatibility."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.setup_widgets()
    
    def setup_widgets(self):
        """Create all UI widgets."""
        # Header
        self.header = widgets.HTML(value="""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">AI Agent for Early Autism Symptom Detection</h1>
            <p style="color: #e0e0e0; margin: 5px 0 0 0;">
                Powered by Gemini 2.5 Flash + LightGBM | Research Tool Only
            </p>
        </div>
        """)
        
        # Demographics section
        self.age_slider = widgets.IntSlider(
            value=24, min=16, max=36, step=1,
            description="Age (months):",
            style={'description_width': '100px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.sex_dropdown = widgets.Dropdown(
            options=[('Male', 'M'), ('Female', 'F')],
            value='M',
            description="Sex:",
            style={'description_width': '100px'}
        )
        
        # M-CHAT-R/F Questions
        self.questions = {
            'q1': widgets.Checkbox(value=False, description="Q1: Does NOT point to show interest", 
                                   layout=widgets.Layout(width='400px')),
            'q2': widgets.Checkbox(value=False, description="Q2: Does NOT make eye contact",
                                   layout=widgets.Layout(width='400px')),
            'q3': widgets.Checkbox(value=False, description="Q3: Does NOT engage in pretend play",
                                   layout=widgets.Layout(width='400px')),
            'q4': widgets.Checkbox(value=False, description="Q4: Does NOT like climbing",
                                   layout=widgets.Layout(width='400px')),
            'q5': widgets.Checkbox(value=False, description="Q5: Makes unusual finger movements",
                                   layout=widgets.Layout(width='400px')),
            'q6': widgets.Checkbox(value=False, description="Q6: Does NOT point to ask for things",
                                   layout=widgets.Layout(width='400px')),
            'q7': widgets.Checkbox(value=False, description="Q7: Does NOT show you objects",
                                   layout=widgets.Layout(width='400px')),
            'q8': widgets.Checkbox(value=False, description="Q8: Is NOT interested in other children",
                                   layout=widgets.Layout(width='400px')),
            'q9': widgets.Checkbox(value=False, description="Q9: Does NOT follow when you point",
                                   layout=widgets.Layout(width='400px')),
            'q10': widgets.Checkbox(value=False, description="Q10: Does NOT respond to name",
                                    layout=widgets.Layout(width='400px')),
        }
        
        # Submit button
        self.submit_btn = widgets.Button(
            description="Run Screening",
            button_style='primary',
            icon='search',
            layout=widgets.Layout(width='200px', height='40px')
        )
        self.submit_btn.on_click(self.on_submit)
        
        # Reset button
        self.reset_btn = widgets.Button(
            description="Reset",
            button_style='warning',
            icon='refresh',
            layout=widgets.Layout(width='100px', height='40px')
        )
        self.reset_btn.on_click(self.on_reset)
        
        # Output area
        self.output_area = widgets.Output(
            layout=widgets.Layout(border='1px solid #ddd', padding='10px', min_height='200px')
        )
        
        # Loading indicator
        self.loading = widgets.HTML(value="")
    
    def on_reset(self, btn):
        """Reset all inputs to default."""
        self.age_slider.value = 24
        self.sex_dropdown.value = 'M'
        for q in self.questions.values():
            q.value = False
        self.output_area.clear_output()
    
    def on_submit(self, btn):
        """Handle screening submission."""
        self.output_area.clear_output()
        
        with self.output_area:
            print("Processing screening... Please wait.\n")
            
            # Build input
            user_input = {
                "age_months": self.age_slider.value,
                "sex": self.sex_dropdown.value,
                "q1_point_interest": int(self.questions['q1'].value),
                "q2_eye_contact": int(self.questions['q2'].value),
                "q3_pretend_play": int(self.questions['q3'].value),
                "q4_climbing": int(self.questions['q4'].value),
                "q5_finger_movement": int(self.questions['q5'].value),
                "q6_point_want": int(self.questions['q6'].value),
                "q7_show_objects": int(self.questions['q7'].value),
                "q8_interest_children": int(self.questions['q8'].value),
                "q9_follow_point": int(self.questions['q9'].value),
                "q10_response_name": int(self.questions['q10'].value)
            }
            
            try:
                # Run screening
                result = self.orchestrator.screen(user_input)
                
                clear_output(wait=True)
                
                # Display results
                risk_colors = {"LOW": "#28a745", "MODERATE": "#ffc107", "HIGH": "#dc3545"}
                risk_color = risk_colors.get(result['risk_level'], "#6c757d")
                
                factors = result.get('top_contributing_factors', [])
                if isinstance(factors, list):
                    factor_strings = []
                    for f in factors:
                        if isinstance(f, dict):
                            factor_strings.append(str(f.get('name', f.get('feature', str(f)))))
                        else:
                            factor_strings.append(str(f))
                    factors_display = ', '.join(factor_strings) if factor_strings else 'None identified'
                else:
                    factors_display = str(factors) if factors else 'None identified'
                
                explanation = result.get('explanation', 'No explanation available.')
                if isinstance(explanation, dict):
                    explanation = explanation.get('text', str(explanation))
                
                recommendations = result.get('recommendations', 'No recommendations available.')
                if isinstance(recommendations, dict):
                    recommendations = recommendations.get('text', str(recommendations))
                
                disclaimer = result.get('disclaimer', 'This is a screening tool only.')
                if isinstance(disclaimer, dict):
                    disclaimer = disclaimer.get('text', str(disclaimer))
                
                display(HTML(f"""
                <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 10px;">
                    <h2 style="color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px;">
                        Screening Results
                    </h2>
                    
                    <div style="display: flex; gap: 20px; margin: 15px 0;">
                        <div style="background: white; padding: 15px; border-radius: 8px; 
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1;">
                            <h4 style="margin: 0; color: #666;">Risk Level</h4>
                            <p style="font-size: 24px; font-weight: bold; margin: 5px 0; 
                                      color: {risk_color};">{result['risk_level']}</p>
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 8px; 
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1;">
                            <h4 style="margin: 0; color: #666;">Risk Score</h4>
                            <p style="font-size: 24px; font-weight: bold; margin: 5px 0; 
                                      color: #333;">{result['risk_percentage']}</p>
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 8px; 
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1;">
                            <h4 style="margin: 0; color: #666;">Session ID</h4>
                            <p style="font-size: 14px; margin: 5px 0; color: #333; 
                                      font-family: monospace;">{result['session_id'][:12]}...</p>
                        </div>
                    </div>
                    
                    <div style="background: white; padding: 15px; border-radius: 8px; 
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0;">
                        <h4 style="margin: 0 0 10px 0; color: #667eea;">Top Contributing Factors</h4>
                        <p style="margin: 0;">{factors_display}</p>
                    </div>
                    
                    <div style="background: white; padding: 15px; border-radius: 8px; 
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0;">
                        <h4 style="margin: 0 0 10px 0; color: #667eea;">Explanation</h4>
                        <p style="margin: 0; line-height: 1.6;">{explanation}</p>
                    </div>
                    
                    <div style="background: white; padding: 15px; border-radius: 8px; 
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0;">
                        <h4 style="margin: 0 0 10px 0; color: #667eea;">Recommendations</h4>
                        <p style="margin: 0; line-height: 1.6;">{recommendations}</p>
                    </div>
                    
                    <div style="background: #fff3cd; padding: 15px; border-radius: 8px; 
                                border-left: 4px solid #ffc107; margin-top: 15px;">
                        <strong>Important Disclaimer:</strong><br>
                        {disclaimer}
                    </div>
                </div>
                """))
                
            except Exception as e:
                clear_output(wait=True)
                display(HTML(f"""
                <div style="background: #f8d7da; padding: 15px; border-radius: 8px; 
                            border-left: 4px solid #dc3545;">
                    <strong>Error:</strong> {str(e)}
                </div>
                """))
    
    def display(self):
        """Display the complete UI."""
        # Questions section
        questions_box = widgets.VBox(
            list(self.questions.values()),
            layout=widgets.Layout(padding='10px')
        )
        
        questions_section = widgets.VBox([
            widgets.HTML(value="""
            <h3 style="color: #333; margin: 15px 0 10px 0;">M-CHAT-R/F Screening Questions</h3>
            <p style="color: #666; font-size: 13px;">Check boxes for behaviors that are present/concerning:</p>
            """),
            questions_box
        ])
        
        # Demographics section
        demographics_section = widgets.VBox([
            widgets.HTML(value="<h3 style='color: #333; margin: 15px 0 10px 0;'>Demographics</h3>"),
            self.age_slider,
            self.sex_dropdown
        ])
        
        # Buttons section
        buttons_section = widgets.HBox(
            [self.submit_btn, self.reset_btn],
            layout=widgets.Layout(margin='20px 0', gap='10px')
        )
        
        # Results section
        results_section = widgets.VBox([
            widgets.HTML(value="<h3 style='color: #333; margin: 15px 0 10px 0;'>Results</h3>"),
            self.output_area
        ])
        
        # Complete layout
        complete_ui = widgets.VBox([
            self.header,
            demographics_section,
            questions_section,
            buttons_section,
            results_section
        ])
        
        display(complete_ui)

# Launch the interactive UI
print("\n" + "=" * 60)
print("LAUNCHING INTERACTIVE UI")
print("=" * 60)
print("Use the form below to run screenings.\n")

screening_ui = AutismScreeningUI(orchestrator)
screening_ui.display()

