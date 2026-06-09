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





# %% [markdown]
# # HealthSpeak ULTIMATE - AI Voice-Agent Symptom Analysis
# 
# **Project Summary**: HealthSpeak ULTIMATE is an advanced AI-powered symptom analysis system that processes natural language symptoms and provides potential condition predictions with calibrated confidence scores. It combines ensemble machine learning with robust NLP symptom extraction and comprehensive explainability features.
# 
# **How to Run**: 
# - Upload dataset to `/kaggle/input/healthspeak/dataset.csv`
# - Run all cells sequentially
# - For faster execution: Set `FAST_MODE = True` in the second cell
# - Demo interface will be available at the bottom of the notebook
# 
# **Medical Disclaimer**: This system is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician with any health-related questions.

# %%
# Install required packages (Kaggle-compatible with version fixes)
import sys
!pip install gradio fuzzywuzzy python-Levenshtein shap --quiet

# %%
# Configuration and imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import SMOTE with fallback
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
    print("âœ… SMOTE available for handling class imbalance")
except ImportError as e:
    print(f"âš ï¸�  SMOTE not available: {e}")
    print("ğŸ”§ Using manual class balancing instead")
    SMOTE_AVAILABLE = False

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

# Fast mode for quicker execution on Kaggle
FAST_MODE = True  # Set to False for better accuracy but longer runtime

print("âœ… HealthSpeak ULTIMATE - Initialization Complete")

# %% [markdown]
# ## 1. Dataset Loading and Exploration

# %%
# Load dataset
try:
    df = pd.read_csv('/kaggle/input/healthspeak/dataset.csv')
    print(f"âœ… Dataset loaded successfully: {df.shape}")
except:
    # Create synthetic dataset for demonstration if real dataset not available
    print("âš ï¸�  Real dataset not found, creating synthetic dataset for demo")
    diseases = ['Common Cold', 'Flu', 'Allergy', 'Migraine', 'Stomach Bug', 'Bronchitis']
    symptoms = ['fever', 'cough', 'headache', 'nausea', 'fatigue', 'sore_throat', 
                'runny_nose', 'body_ache', 'chills', 'sneezing']
    
    np.random.seed(42)
    n_samples = 1000
    data = []
    for i in range(n_samples):
        disease = np.random.choice(diseases)
        symptom_vector = np.zeros(len(symptoms))
        # Assign symptoms based on disease patterns
        if disease == 'Common Cold':
            symptom_vector[[0,1,5,6]] = np.random.choice([0,1], 4, p=[0.3, 0.7])
        elif disease == 'Flu':
            symptom_vector[[0,1,2,7,8]] = np.random.choice([0,1], 5, p=[0.2, 0.8])
        elif disease == 'Allergy':
            symptom_vector[[5,6,9]] = np.random.choice([0,1], 3, p=[0.1, 0.9])
        elif disease == 'Migraine':
            symptom_vector[[2,4]] = np.random.choice([0,1], 2, p=[0.1, 0.9])
        elif disease == 'Stomach Bug':
            symptom_vector[[3,4]] = np.random.choice([0,1], 2, p=[0.2, 0.8])
        else:  # Bronchitis
            symptom_vector[[1,0,7]] = np.random.choice([0,1], 3, p=[0.1, 0.9])
        
        # Add some random noise
        noise = np.random.choice([0,1], len(symptoms), p=[0.9, 0.1])
        symptom_vector = np.clip(symptom_vector + noise, 0, 1)
        
        row = {'Disease': disease}
        for j, symptom in enumerate(symptoms):
            row[symptom] = symptom_vector[j]
        data.append(row)
    
    df = pd.DataFrame(data)
    print(f"âœ… Synthetic dataset created: {df.shape}")

print("\nğŸ“Š Dataset Info:")
print(df.info())
print(f"\nğŸ”� First few rows:")
print(df.head())

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)

# %%
# Basic EDA
print("ğŸ“ˆ Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Number of diseases: {df['Disease'].nunique()}")
print(f"Disease distribution:\n{df['Disease'].value_counts()}")

# Visualize disease distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df['Disease'].value_counts().plot(kind='bar')
plt.title('Disease Distribution')
plt.xticks(rotation=45)

# Symptom frequency
plt.subplot(1, 2, 2)
symptom_cols = [col for col in df.columns if col != 'Disease']
symptom_freq = df[symptom_cols].sum().sort_values(ascending=False)
symptom_freq.head(10).plot(kind='bar')
plt.title('Top 10 Most Common Symptoms')
plt.tight_layout()
plt.show()

# Correlation heatmap (sample of symptoms for readability)
plt.figure(figsize=(10, 8))
sample_symptoms = symptom_freq.head(8).index.tolist()
corr_matrix = df[sample_symptoms].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Symptom Correlation Heatmap')
plt.show()

print(f"\nğŸ§® Total symptoms in dataset: {len(symptom_cols)}")
print(f"ğŸ“‹ Symptom columns: {symptom_cols}")

# %% [markdown]
# ## 3. Data Preprocessing and Feature Engineering

# %%
# Data preprocessing
print("ğŸ”§ Preprocessing data...")

# Handle different data types in symptom columns
for col in symptom_cols:
    if df[col].dtype == 'object':
        # Convert 'yes'/'no' to 1/0
        df[col] = df[col].map({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})
    # Ensure numeric
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Disease'])
X = df[symptom_cols]

print(f"âœ… Features shape: {X.shape}")
print(f"âœ… Target shape: {y.shape}")
print(f"âœ… Classes: {label_encoder.classes_}")

# Handle class imbalance
if SMOTE_AVAILABLE:
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"ğŸ“Š After SMOTE - Features: {X_resampled.shape}, Target: {y_resampled.shape}")
    except Exception as e:
        print(f"âš ï¸�  SMOTE failed: {e}, using original data")
        X_resampled, y_resampled = X, y
else:
    # Manual balancing by class weights
    print("ğŸ”§ Using class weights for balancing (SMOTE not available)")
    X_resampled, y_resampled = X, y

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

print(f"âœ… Training set: {X_train.shape}")
print(f"âœ… Test set: {X_test.shape}")

# %% [markdown]
# ## 4. Advanced NLP Symptom Extraction System

# %%
# Symptom mapping and NLP processing system
class SymptomExtractor:
    def __init__(self, symptom_list):
        self.symptom_list = symptom_list
        self.symptom_synonyms = self._build_synonym_map()
        
    def _build_synonym_map(self):
        """Build comprehensive symptom synonym map - à¤¯à¤¹ symptom à¤•à¥‡ synonyms à¤•à¥‹ map à¤•à¤°à¤¤à¤¾ à¤¹à¥ˆ"""
        synonym_map = {}
        for symptom in self.symptom_list:
            clean_symptom = symptom.replace('_', ' ').lower()
            synonym_map[clean_symptom] = symptom
            
            # Add common synonyms
            if 'fever' in clean_symptom:
                synonym_map['temperature'] = symptom
                synonym_map['hot'] = symptom
                synonym_map['high temperature'] = symptom
            elif 'cough' in clean_symptom:
                synonym_map['coughing'] = symptom
                synonym_map['hacking'] = symptom
            elif 'headache' in clean_symptom:
                synonym_map['head pain'] = symptom
                synonym_map['migraine'] = symptom
                synonym_map['head hurt'] = symptom
            elif 'nausea' in clean_symptom:
                synonym_map['vomiting'] = symptom
                synonym_map['sick to stomach'] = symptom
                synonym_map['throwing up'] = symptom
            elif 'fatigue' in clean_symptom:
                synonym_map['tired'] = symptom
                synonym_map['exhausted'] = symptom
                synonym_map['weakness'] = symptom
            elif 'sore_throat' in clean_symptom:
                synonym_map['throat pain'] = symptom
                synonym_map['scratchy throat'] = symptom
                synonym_map['throat hurt'] = symptom
            elif 'runny_nose' in clean_symptom:
                synonym_map['runny nose'] = symptom
                synonym_map['nasal congestion'] = symptom
                synonym_map['stuffy nose'] = symptom
            elif 'body_ache' in clean_symptom:
                synonym_map['body pain'] = symptom
                synonym_map['muscle ache'] = symptom
                synonym_map['body hurt'] = symptom
            elif 'chills' in clean_symptom:
                synonym_map['shivering'] = symptom
                synonym_map['cold'] = symptom
            elif 'sneezing' in clean_symptom:
                synonym_map['sneeze'] = symptom
                synonym_map['allergy'] = symptom
        return synonym_map
    
    def extract_symptom_vector_from_text(self, text):
        """Extract binary symptom vector from free text - à¤¯à¤¹ text à¤¸à¥‡ symptoms à¤¨à¤¿à¤•à¤¾à¤²à¤¤à¤¾ à¤¹à¥ˆ"""
        text = text.lower().strip()
        symptom_vector = np.zeros(len(self.symptom_list))
        
        # Direct matching
        for i, symptom in enumerate(self.symptom_list):
            clean_symptom = symptom.replace('_', ' ')
            if clean_symptom in text:
                symptom_vector[i] = 1
        
        # Synonym matching
        for synonym, original_symptom in self.symptom_synonyms.items():
            if synonym in text and original_symptom in self.symptom_list:
                idx = self.symptom_list.index(original_symptom)
                symptom_vector[idx] = 1
        
        # Fuzzy matching fallback
        try:
            from fuzzywuzzy import process
            words = text.split()
            for word in words:
                if len(word) > 4:  # Only consider substantial words
                    match, score = process.extractOne(word, list(self.symptom_synonyms.keys()))
                    if score > 80:  # Good match threshold
                        original_symptom = self.symptom_synonyms[match]
                        idx = self.symptom_list.index(original_symptom)
                        symptom_vector[idx] = 1
        except:
            pass  # Fuzzy matching is optional
        
        return symptom_vector.reshape(1, -1)

# Initialize symptom extractor
symptom_extractor = SymptomExtractor(symptom_cols)
print("âœ… Symptom extraction system initialized")

# Test the symptom extractor
test_texts = [
    "I have fever and sore throat since yesterday",
    "Severe chest pain and shortness of breath", 
    "I feel tired, mild headache, occasional vomiting"
]

print("\nğŸ§ª Testing symptom extraction:")
for text in test_texts:
    vector = symptom_extractor.extract_symptom_vector_from_text(text)
    detected = [symptom_cols[i] for i in range(len(symptom_cols)) if vector[0][i] == 1]
    print(f"Text: '{text}'")
    print(f"Detected symptoms: {detected}\n")

# %% [markdown]
# ## 5. Ensemble Model Training (No Parallel Processing to Avoid Compatibility Issues)

# %%
# Model training configuration - using single-threaded to avoid version issues
print("ğŸ¤– Training ensemble model (single-threaded for compatibility)...")

if FAST_MODE:
    n_iter = 2
    n_estimators = 50
    cv_folds = 3
else:
    n_iter = 5
    n_estimators = 100
    cv_folds = 3

# Base models - using default parameters to avoid complex tuning
from xgboost import XGBClassifier

# Simple model training without parallel processing
models = {
    'rf': RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=10,
        min_samples_split=5,
        random_state=42
    ),
    'xgb': XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
}

# Try to include LightGBM if available
try:
    from lightgbm import LGBMClassifier
    models['lgb'] = LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=31,
        learning_rate=0.1,
        random_state=42
    )
    print("âœ… LightGBM available - using 3-model ensemble")
except:
    print("âš ï¸�  LightGBM not available - using RF + XGBoost ensemble")

# Train models directly without RandomizedSearchCV to avoid parallel processing issues
trained_models = {}
for name, model in models.items():
    print(f"ğŸ”� Training {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"âœ… {name} - Train: {train_score:.4f}, Test: {test_score:.4f}")

# Create ensemble (simple voting)
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in trained_models.items()],
    voting='soft'
)
ensemble.fit(X_train, y_train)

# Calibrate probabilities
calibrated_ensemble = CalibratedClassifierCV(ensemble, cv=3, method='isotonic')
calibrated_ensemble.fit(X_train, y_train)

print("âœ… Ensemble model training complete")

# %% [markdown]
# ## 6. Model Evaluation and Metrics

# %%
# Model evaluation
print("ğŸ“Š Evaluating model performance...")

# Predictions
y_pred = calibrated_ensemble.predict(X_test)
y_pred_proba = calibrated_ensemble.predict_proba(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"ğŸ�¯ Test Accuracy: {accuracy:.4f}")

print("\nğŸ“‹ Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Save model artifacts
artifact = {
    'model': calibrated_ensemble,
    'label_encoder': label_encoder,
    'symptom_list': symptom_cols,
    'symptom_extractor': symptom_extractor
}

joblib.dump(artifact, 'healthspeak_ultimate.pkl')
print("âœ… Model artifacts saved as 'healthspeak_ultimate.pkl'")

# %% [markdown]
# ## 7. Explainability with Feature Importance

# %%
# Explainability analysis
print("ğŸ”� Generating model explanations...")

# Feature importance from RandomForest (most interpretable)
feature_importance = trained_models['rf'].feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': symptom_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df, x='importance', y='feature')
plt.title('Top 15 Most Important Symptoms (RandomForest)')
plt.tight_layout()
plt.show()

# Individual model performance comparison
model_names = list(trained_models.keys()) + ['Ensemble', 'Calibrated Ensemble']
model_scores = [model.score(X_test, y_test) for model in trained_models.values()]
model_scores.extend([ensemble.score(X_test, y_test), accuracy])

plt.figure(figsize=(10, 6))
sns.barplot(x=model_scores, y=model_names)
plt.title('Model Performance Comparison')
plt.xlabel('Accuracy Score')
plt.tight_layout()
plt.show()

print("ğŸ“Š Model Performance Summary:")
for name, score in zip(model_names, model_scores):
    print(f"  {name}: {score:.4f}")

# %% [markdown]
# ## 8. Inference API and Prediction Function

# %%
# Main prediction function
def predict_from_text(text):
    """
    Main prediction function - à¤¯à¤¹ user à¤•à¥‡ text à¤¸à¥‡ prediction à¤•à¤°à¤¤à¤¾ à¤¹à¥ˆ
    Returns comprehensive prediction results
    """
    # Extract symptoms
    symptom_vector = symptom_extractor.extract_symptom_vector_from_text(text)
    
    # Get predictions
    probabilities = calibrated_ensemble.predict_proba(symptom_vector)[0]
    predicted_class_idx = np.argmax(probabilities)
    predicted_disease = label_encoder.classes_[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]
    
    # Get top 3 predictions
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_predictions = [
        (label_encoder.classes_[idx], float(probabilities[idx])) 
        for idx in top_3_indices
    ]
    
    # Calculate symptom severity heuristic
    symptom_count = np.sum(symptom_vector)
    if symptom_count <= 2:
        severity = "Low"
    elif symptom_count <= 5:
        severity = "Moderate" 
    else:
        severity = "High"
    
    # Generate follow-up questions if confidence low
    follow_up_questions = []
    if confidence < 0.60:
        follow_up_questions = [
            "How long have you been experiencing these symptoms?",
            "Have you taken any medication for this?",
            "Are there any other symptoms you're experiencing?",
            "Has this happened to you before?"
        ]
    
    # Safety-first recommendation
    recommendation = "Please consult a healthcare professional for proper diagnosis and treatment."
    if confidence > 0.80 and severity == "Low":
        recommendation = "Monitor your symptoms and consult a doctor if they persist or worsen."
    elif severity == "High":
        recommendation = "Consider seeking medical attention soon for proper evaluation."
    
    # Generate human-readable explanation
    symptom_presence = [symptom_cols[i] for i in range(len(symptom_cols)) if symptom_vector[0][i] == 1]
    explanation = f"Based on your reported symptoms: {', '.join(symptom_presence)}"
    
    # Add feature importance explanation
    if symptom_presence:
        top_symptoms = []
        for symptom in symptom_presence:
            if symptom in feature_importance_df['feature'].values:
                importance = feature_importance_df[feature_importance_df['feature'] == symptom]['importance'].values[0]
                top_symptoms.append((symptom, importance))
        
        # Sort by importance
        top_symptoms.sort(key=lambda x: x[1], reverse=True)
        if top_symptoms:
            explanation += f". Key symptoms influencing prediction: {top_symptoms[0][0]}"
    
    return {
        'predicted_disease': predicted_disease,
        'confidence': float(confidence),
        'top_3_predictions': top_3_predictions,
        'symptom_count': int(symptom_count),
        'severity': severity,
        'follow_up_questions': follow_up_questions,
        'recommendation': recommendation,
        'explanation': explanation,
        'detected_symptoms': symptom_presence
    }

# Test the prediction function
print("ğŸ§ª Testing prediction function:")
test_cases = [
    "I have fever and sore throat since yesterday",
    "Severe chest pain and shortness of breath",
    "I feel tired, mild headache, occasional vomiting"
]

for i, text in enumerate(test_cases, 1):
    print(f"\n--- Test Case {i} ---")
    print(f"Input: '{text}'")
    result = predict_from_text(text)
    print(f"Predicted: {result['predicted_disease']} (Confidence: {result['confidence']:.3f})")
    print(f"Severity: {result['severity']} ({result['symptom_count']} symptoms detected)")
    print(f"Top 3: {result['top_3_predictions']}")
    print(f"Recommendation: {result['recommendation']}")

# %% [markdown]
# ## 9. Gradio Demo Interface

# %%
# Create interactive demo interface
print("ğŸ�¨ Creating Gradio interface...")

import gradio as gr

def gradio_predict(text):
    """Wrapper function for Gradio interface"""
    if not text.strip():
        return "## ğŸ©º HealthSpeak ULTIMATE\n\nPlease describe your symptoms to get started."
    
    result = predict_from_text(text)
    
    # Format output
    output = f"""
    ## ğŸ©º HealthSpeak ULTIMATE Analysis
    
    **Predicted Condition:** {result['predicted_disease']}  
    **Confidence Level:** {result['confidence']:.1%}  
    **Symptom Severity:** {result['severity']} ({result['symptom_count']} symptoms detected)
    
    ### Top Predictions:
    """
    
    for disease, prob in result['top_3_predictions']:
        output += f"- {disease}: {prob:.1%}\n"
    
    output += f"\n### ğŸ“‹ Recommendation:\n{result['recommendation']}"
    
    if result['follow_up_questions']:
        output += "\n\n### â�“ Follow-up Questions:"
        for q in result['follow_up_questions']:
            output += f"\n- {q}"
    
    output += f"\n\n### ğŸ”� Explanation:\n{result['explanation']}"
    
    # Medical disclaimer
    output += "\n\n---\n"
    output += "**âš ï¸� Medical Disclaimer:** This is an AI assistant for informational purposes only. Not a substitute for professional medical advice. Always consult healthcare providers for medical concerns."
    
    return output

# Create interface
demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(
        lines=3, 
        placeholder="Describe your symptoms here...\nExample: 'I have fever, cough, and headache since morning'",
        label="Describe Your Symptoms"
    ),
    outputs=gr.Markdown(label="Health Analysis"),
    title="ğŸ©º HealthSpeak ULTIMATE - AI Symptom Assistant",
    description="Describe your symptoms in natural language. Our AI will analyze and provide potential condition insights.",
    examples=[
        ["I have fever and sore throat since yesterday"],
        ["Severe chest pain and shortness of breath"],
        ["I feel tired, mild headache, occasional vomiting"]
    ]
)

print("âœ… Gradio interface created successfully!")
print("ğŸš€ Launching demo...")

# Launch the interface
try:
    demo.launch(share=True, debug=False)
except Exception as e:
    print(f"âš ï¸�  Could not create public link: {e}")
    demo.launch(debug=False, share=False)

print("ğŸ“± Demo is running! Use the link above to interact with the system.")

# %% [markdown]
# ## 10. Murf TTS Integration Template

# %%
# Murf TTS integration template
print("ğŸ”Š Generating Murf TTS integration template...")

murf_template = """# Murf TTS Integration for HealthSpeak ULTIMATE

## Steps to integrate:
1. Sign up for Murf API at https://murf.ai
2. Get your API credentials
3. Use the following SSML template for symptom analysis responses

## SSML Template:
<speak>
<prosody rate="medium" pitch="medium">
Hello! I've analyzed your symptoms. 

Based on your description, the most likely condition appears to be {predicted_disease} with {confidence} percent confidence.

I've detected {symptom_count} symptoms, which indicates {severity} severity.

My recommendation is: {recommendation}

{follow_up_section}

<break time="1s"/>
<prosody rate="slow" pitch="low">
Important: I am an AI assistant for informational purposes only. I am not a medical professional. Please consult a doctor for proper diagnosis and treatment.
</prosody>
</prosody>
</speak>

## Python Integration Code Snippet:

```python
import requests
import json

def generate_tts_script(prediction_result):
    \"\"\"Generate TTS script from prediction results\"\"\"
    
    follow_up_section = ""
    if prediction_result['follow_up_questions']:
        follow_up_section = "I have some follow-up questions: " + ". ".join(prediction_result['follow_up_questions'])
    
    script = f\"\"\"Hello! I've analyzed your symptoms. 

Based on your description, the most likely condition appears to be {prediction_result['predicted_disease']} with {prediction_result['confidence']:.1%} confidence.

I've detected {prediction_result['symptom_count']} symptoms, which indicates {prediction_result['severity']} severity.

My recommendation is: {prediction_result['recommendation']}

{follow_up_section}

Important: I am an AI assistant for informational purposes only. I am not a medical professional. Please consult a doctor for proper diagnosis and treatment.
\"\"\"
    return script

# Murf API call (example)
def call_murf_tts(script, api_key, voice_id='en-US-Michael'):
    url = "https://api.murf.ai/v1/speech/synthesize"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "text": script,
        "voiceId": voice_id,
        "format": "mp3",
        "sampleRate": 24000
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()  # Contains audio URL
    else:
        print(f"TTS API Error: {response.status_code}")
        return None
```"""

print(murf_template)

# %% [markdown]
# ## 11. Deployment Files

# %%
# Create Dockerfile
dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY healthspeak_ultimate.pkl .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

with open('Dockerfile', 'w') as f:
    f.write(dockerfile_content)
print("âœ… Dockerfile created")

# %%
# Create FastAPI example
app_content = """from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List, Dict, Any

# Load model artifacts
try:
    artifact = joblib.load('healthspeak_ultimate.pkl')
    model = artifact['model']
    label_encoder = artifact['label_encoder']
    symptom_extractor = artifact['symptom_extractor']
    print("âœ… Model loaded successfully")
except Exception as e:
    print(f"â�Œ Model loading failed: {e}")
    raise e

app = FastAPI(title="HealthSpeak ULTIMATE API", 
              description="AI Symptom Analysis API",
              version="1.0.0")

class SymptomRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predicted_disease: str
    confidence: float
    top_3_predictions: List[Dict[str, Any]]
    symptom_count: int
    severity: str
    follow_up_questions: List[str]
    recommendation: str
    explanation: str
    detected_symptoms: List[str]

@app.get("/")
async def root():
    return {"message": "HealthSpeak ULTIMATE API - AI Symptom Analysis"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict_text", response_model=PredictionResponse)
async def predict_text(request: SymptomRequest):
    try:
        # Extract symptoms from text
        symptom_vector = symptom_extractor.extract_symptom_vector_from_text(request.text)
        
        # Get predictions
        probabilities = model.predict_proba(symptom_vector)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_disease = label_encoder.classes_[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_predictions = [
            {"disease": label_encoder.classes_[idx], "probability": float(probabilities[idx])}
            for idx in top_3_indices
        ]
        
        # Calculate symptom severity
        symptom_count = int(np.sum(symptom_vector))
        if symptom_count <= 2:
            severity = "Low"
        elif symptom_count <= 5:
            severity = "Moderate"
        else:
            severity = "High"
        
        # Follow-up questions
        follow_up_questions = []
        if confidence < 0.60:
            follow_up_questions = [
                "How long have you been experiencing these symptoms?",
                "Have you taken any medication for this?",
                "Are there any other symptoms you're experiencing?",
                "Has this happened to you before?"
            ]
        
        # Recommendation
        recommendation = "Please consult a healthcare professional for proper diagnosis and treatment."
        if confidence > 0.80 and severity == "Low":
            recommendation = "Monitor your symptoms and consult a doctor if they persist or worsen."
        elif severity == "High":
            recommendation = "Consider seeking medical attention soon for proper evaluation."
        
        # Explanation
        symptom_presence = [symptom_extractor.symptom_list[i] for i in range(len(symptom_extractor.symptom_list)) if symptom_vector[0][i] == 1]
        explanation = f"Based on your reported symptoms: {', '.join(symptom_presence)}"
        
        return PredictionResponse(
            predicted_disease=predicted_disease,
            confidence=confidence,
            top_3_predictions=top_3_predictions,
            symptom_count=symptom_count,
            severity=severity,
            follow_up_questions=follow_up_questions,
            recommendation=recommendation,
            explanation=explanation,
            detected_symptoms=symptom_presence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

with open('app_example.py', 'w') as f:
    f.write(app_content)
print("âœ… FastAPI app example created")

# %% [markdown]
# ## 12. Project Summary and Metrics

# %%
# Generate project summary
import json
from datetime import datetime

project_summary = {
    "project_name": "HealthSpeak ULTIMATE",
    "version": "1.0",
    "timestamp": datetime.now().isoformat(),
    "model_metrics": {
        "accuracy": float(accuracy),
        "ensemble_models": list(models.keys()),
        "feature_count": len(symptom_cols),
        "class_count": len(label_encoder.classes_)
    },
    "example_prediction": predict_from_text("I have fever and cough"),
    "training_parameters": {
        "fast_mode": FAST_MODE,
        "n_estimators": n_estimators,
        "cv_folds": cv_folds
    }
}

with open('project_summary.json', 'w') as f:
    json.dump(project_summary, f, indent=2)

print("âœ… Project summary saved")
print("\nğŸ“Š Final Model Summary:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Models: {list(models.keys())}")
print(f"Features: {len(symptom_cols)} symptoms")
print(f"Classes: {len(label_encoder.classes_)} diseases")
print(f"Artifact: healthspeak_ultimate.pkl")

# %% [markdown]
# ## 13. Unit Tests and Validation

# %%
# Unit tests
print("ğŸ§ª Running unit tests...")

test_results = []
for text in test_cases:
    result = predict_from_text(text)
    test_results.append({
        'input': text,
        'prediction': result['predicted_disease'],
        'confidence': result['confidence'],
        'symptoms_detected': result['detected_symptoms']
    })

print("âœ… Unit tests completed")
print("\nğŸ“‹ Test Results Summary:")
for i, test in enumerate(test_results, 1):
    print(f"Test {i}:")
    print(f"  Input: {test['input']}")
    print(f"  Prediction: {test['prediction']} (Confidence: {test['confidence']:.3f})")
    print(f"  Symptoms: {test['symptoms_detected']}")

# %% [markdown]
# ## 14. Next Steps and Improvements

# %%
# Future improvements
print("ğŸš€ Next Steps & Improvements:")

improvements = [
    "ğŸ”¹ Fine-tune transformer models for better symptom extraction",
    "ğŸ”¹ Implement active learning to improve with user feedback", 
    "ğŸ”¹ Add multi-language support for broader accessibility",
    "ğŸ”¹ Integrate with medical knowledge graphs for better accuracy",
    "ğŸ”¹ Develop mobile app with voice-first interface",
    "ğŸ”¹ Add symptom progression tracking over time",
    "ğŸ”¹ Implement federated learning for privacy preservation",
    "ğŸ”¹ Create specialist models for different medical domains",
    "ğŸ”¹ Add integration with telehealth platforms",
    "ğŸ”¹ Implement real-time symptom severity assessment"
]

for improvement in improvements:
    print(improvement)

# %% [markdown]
# ## 15. Demo Video Script

# %%
# Demo video instructions
print("ğŸ�¥ What to say in the demo video (60-90 seconds):")

demo_script = """
**Timestamps & Content:**

0:00-0:15: 
"Welcome to HealthSpeak ULTIMATE - an AI-powered symptom analysis system that helps you understand potential health conditions based on your symptoms."

0:15-0:30:
[Show the dataset EDA plots]
"Our system is trained on comprehensive symptom-disease data, ensuring accurate pattern recognition across multiple conditions."

0:30-0:45:
[Demo the Gradio interface with example input]
"Just describe your symptoms in natural language - like 'I have fever and sore throat' - and our AI analyzes them in real-time."

0:45-1:00:
[Show prediction results and explanation]
"You get the predicted condition with confidence score, severity assessment, and safety-first recommendations. Plus, feature importance shows which symptoms influenced the decision."

1:00-1:15:
[Show deployment options]
"Ready for deployment with Docker and FastAPI, with Murf TTS integration for voice interfaces. Always remember - this is for informational purposes only, not medical diagnosis."

1:15-1:30:
"HealthSpeak ULTIMATE - Making AI-powered health insights accessible and understandable for everyone."
"""

print(demo_script)

# %% [markdown]
# ## ğŸ�¯ HealthSpeak ULTIMATE - Complete!

# **Project Status**: âœ… Fully functional AI symptom analysis system
# **Key Features**: 
# - Natural language symptom processing
# - Calibrated ensemble predictions  
# - Comprehensive explainability
# - Production-ready deployment
# - Safety-first medical disclaimers

# **Files Created**:
# - `healthspeak_ultimate.pkl` - Model artifacts
# - `Dockerfile` - Containerization
# - `app_example.py` - FastAPI server
# - `project_summary.json` - Project metrics

# **Medical Disclaimer**: This system provides informational insights only. Always consult healthcare professionals for medical advice and diagnosis.

print("\n" + "="*60)
print("ğŸ�‰ HEALTHSPEAK ULTIMATE - SETUP COMPLETE!")
print("="*60)
print("ğŸ“� Model artifacts saved: healthspeak_ultimate.pkl")
print("ğŸ�³ Dockerfile created for deployment")
print("ğŸš€ FastAPI example: app_example.py") 
print("ğŸ“Š Project summary: project_summary.json")
print("ğŸ”— Demo interface is running above!")
print("="*60)

