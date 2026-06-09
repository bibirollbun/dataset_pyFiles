!pip install textstat


import os
import warnings
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm import tqdm
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
warnings.filterwarnings("ignore")


def enhanced_train_data_generator(data_dir, csv_path):
    """
    Enhanced generator that extracts additional metadata from text pairs
    """
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        folder_id = row["id"]
        real_text_id = row["real_text_id"]
        
        folder_path = os.path.join(data_dir, f"article_{folder_id:04d}")
        file1_path = os.path.join(folder_path, "file_1.txt")
        file2_path = os.path.join(folder_path, "file_2.txt")
        
        with open(file1_path, encoding="utf-8") as f1:
            text1 = f1.read().strip()
        with open(file2_path, encoding="utf-8") as f2:
            text2 = f2.read().strip()
            
        # Enhanced preprocessing
        text1_clean = preprocess_text(text1)
        text2_clean = preprocess_text(text2)
        
        label = 1 if real_text_id == 1 else 0
        
        yield {
            "id": folder_id,
            "text1": text1_clean,
            "text2": text2_clean,
            "text1_raw": text1,
            "text2_raw": text2,
            "label": label
        }

def enhanced_test_data_generator(data_dir):
    """
    Enhanced test generator with preprocessing
    """
    folders = sorted([
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f)) and re.match(r'article_\d+', f)
    ])
    
    for folder in folders:
        folder_id = int(folder.split('_')[1])
        folder_path = os.path.join(data_dir, folder)
        
        file1_path = os.path.join(folder_path, "file_1.txt")
        file2_path = os.path.join(folder_path, "file_2.txt")
        
        with open(file1_path, encoding="utf-8") as f1:
            text1 = f1.read().strip()
        with open(file2_path, encoding="utf-8") as f2:
            text2 = f2.read().strip()
            
        text1_clean = preprocess_text(text1)
        text2_clean = preprocess_text(text2)
        
        yield {
            "id": folder_id,
            "text1": text1_clean,
            "text2": text2_clean,
            "text1_raw": text1,
            "text2_raw": text2
        }

def preprocess_text(text):
    """
    Advanced text preprocessing while preserving important features
    """
    # Remove excessive whitespace but preserve structure
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Normalize quotes and dashes
    text = re.sub(r'[""''`]', '"', text)
    text = re.sub(r'[â€“â€”]', '-', text)
    
    return text

# Load the datasets
train_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
train_csv = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"
test_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"

train_dataset = Dataset.from_generator(lambda: enhanced_train_data_generator(train_dir, train_csv))
test_dataset = Dataset.from_generator(lambda: enhanced_test_data_generator(test_dir))

datasets = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

print(f"Training samples: {len(datasets['train'])}")
print(f"Test samples: {len(datasets['test'])}")


class EnhancedBERTFeatureExtractor:
    def __init__(self, model_name="bert-base-uncased", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def extract_multilayer_embeddings(self, text):
        """
        Extract embeddings from multiple BERT layers for richer representation
        """
        # Tokenize with sliding window for long texts
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=True,
            stride=256
        )
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(len(inputs["input_ids"])):
                chunk_inputs = {k: v[i:i+1].to(self.device) for k, v in inputs.items() 
                              if k in ["input_ids", "attention_mask"]}
                
                outputs = self.model(**chunk_inputs)
                hidden_states = outputs.hidden_states
                
                # Extract from last 4 layers
                layers_to_use = [-4, -3, -2, -1]
                layer_embeddings = []
                
                for layer_idx in layers_to_use:
                    layer_output = hidden_states[layer_idx]
                    # Apply attention mask and mean pool
                    mask = chunk_inputs["attention_mask"].unsqueeze(-1)
                    masked_output = layer_output * mask
                    pooled = masked_output.sum(dim=1) / mask.sum(dim=1)
                    layer_embeddings.append(pooled.cpu())
                
                # Concatenate layer embeddings
                chunk_embedding = torch.cat(layer_embeddings, dim=-1)
                all_embeddings.append(chunk_embedding)
        
        # Average across chunks
        final_embedding = torch.stack(all_embeddings).mean(dim=0).squeeze()
        return final_embedding
    
    def extract_statistical_features(self, text1, text2):
        """
        Extract statistical and linguistic features
        """
        features = []
        
        for text in [text1, text2]:
            # Basic statistics
            features.extend([
                len(text),
                len(text.split()),
                len(text.split('.')),
                text.count('!'),
                text.count('?'),
                text.count(','),
                text.count(';'),
                text.count(':')
            ])
            
            # Readability scores
            try:
                features.extend([
                    flesch_reading_ease(text),
                    flesch_kincaid_grade(text)
                ])
            except:
                features.extend([50.0, 10.0])  # Default values
                
            # Character-level features
            features.extend([
                text.count(' ') / len(text) if len(text) > 0 else 0,
                sum(c.isupper() for c in text) / len(text) if len(text) > 0 else 0,
                sum(c.isdigit() for c in text) / len(text) if len(text) > 0 else 0
            ])
        
        # Comparative features
        features.extend([
            abs(len(text1) - len(text2)),
            abs(len(text1.split()) - len(text2.split())),
            abs(text1.count('.') - text2.count('.')),
        ])
        
        return np.array(features)
    
    def extract_features(self, dataset):
        """
        Extract comprehensive features from the dataset
        """
        bert_features = []
        stat_features = []
        ids = []
        
        for sample in tqdm(dataset, desc="Extracting enhanced features"):
            # BERT embeddings
            emb1 = self.extract_multilayer_embeddings(sample['text1'])
            emb2 = self.extract_multilayer_embeddings(sample['text2'])
            
            # Create interaction features
            diff = emb1 - emb2
            product = emb1 * emb2
            concat = torch.cat([emb1, emb2, diff, product])
            
            bert_features.append(concat.numpy())
            
            # Statistical features
            stat_feat = self.extract_statistical_features(sample['text1'], sample['text2'])
            stat_features.append(stat_feat)
            
            ids.append(sample['id'])
        
        return np.array(bert_features), np.array(stat_features), ids

# Initialize the enhanced feature extractor
feature_extractor = EnhancedBERTFeatureExtractor()

# Extract features
print("ğŸ”� Extracting enhanced features from training data...")
X_bert_train, X_stat_train, train_ids = feature_extractor.extract_features(datasets["train"])
y_train = np.array([sample["label"] for sample in datasets["train"]])

print(f"BERT features shape: {X_bert_train.shape}")
print(f"Statistical features shape: {X_stat_train.shape}")
print(f"Training labels distribution: {np.bincount(y_train)}")


# Apply PCA to BERT features
n_components_bert = 50  # Increased from 20 for better representation
pca_bert = PCA(n_components=n_components_bert, random_state=42)
X_bert_train_pca = pca_bert.fit_transform(X_bert_train)

# Standardize statistical features
scaler_stat = StandardScaler()
X_stat_train_scaled = scaler_stat.fit_transform(X_stat_train)

# Combine features
X_train_combined = np.hstack([X_bert_train_pca, X_stat_train_scaled])

print(f"Combined features shape: {X_train_combined.shape}")
print(f"BERT PCA explained variance ratio: {pca_bert.explained_variance_ratio_.sum():.3f}")

# Visualize feature importance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(np.cumsum(pca_bert.explained_variance_ratio_))
plt.title('BERT PCA Cumulative Explained Variance')
plt.xlabel('Components')
plt.ylabel('Cumulative Variance Explained')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.hist(X_stat_train_scaled.flatten(), bins=50, alpha=0.7)
plt.title('Distribution of Scaled Statistical Features')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()
plt.show()


# Define enhanced models with optimized hyperparameters
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    ),
    'CatBoost': CatBoostClassifier(
        iterations=200,
        depth=8,
        learning_rate=0.1,
        random_state=42,
        verbose=False
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    ),
    'LogisticRegression': LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )
}

# Enhanced cross-validation with stratified folds
def enhanced_cross_validation(models, X, y, cv_folds=7):
    """
    Perform enhanced cross-validation with detailed metrics
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        print(f"ğŸ”„ Evaluating {name}...")
        
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        
        results[name] = {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"   {name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results

# Run enhanced cross-validation
cv_results = enhanced_cross_validation(models, X_train_combined, y_train)

# Display results
results_df = pd.DataFrame({
    name: [results['mean_accuracy'], results['std_accuracy']] 
    for name, results in cv_results.items()
}, index=['Mean_Accuracy', 'Std_Accuracy']).T

results_df = results_df.sort_values('Mean_Accuracy', ascending=False)
print("\nğŸ“Š Cross-Validation Results:")
print(results_df.round(4))


# Select top performing models for ensemble
top_models = results_df.head(3).index.tolist()
print(f"ğŸ�¯ Top models for ensemble: {top_models}")

# Create ensemble
ensemble_models = [(name, models[name]) for name in top_models]
ensemble = VotingClassifier(
    estimators=ensemble_models,
    voting='soft',  # Use probabilities
    n_jobs=-1
)

# Train ensemble on full training data
print("ğŸš€ Training ensemble model...")
ensemble.fit(X_train_combined, y_train)

# Get training accuracy
train_pred = ensemble.predict(X_train_combined)
train_accuracy = accuracy_score(y_train, train_pred)
print(f"ğŸ“ˆ Ensemble training accuracy: {train_accuracy:.4f}")

# Detailed classification report
print("\nğŸ“‹ Training Classification Report:")
print(classification_report(y_train, train_pred))


print("ğŸ”� Extracting features from test data...")
X_bert_test, X_stat_test, test_ids = feature_extractor.extract_features(datasets["test"])

# Apply same transformations
X_bert_test_pca = pca_bert.transform(X_bert_test)
X_stat_test_scaled = scaler_stat.transform(X_stat_test)
X_test_combined = np.hstack([X_bert_test_pca, X_stat_test_scaled])

print(f"Test features shape: {X_test_combined.shape}")

# Generate predictions
print("ğŸ�¯ Generating predictions...")
test_probabilities = ensemble.predict_proba(X_test_combined)[:, 1]

# Create submission
submission_data = []
for i, test_id in enumerate(test_ids):
    prob = test_probabilities[i]
    real_text_id = 1 if prob >= 0.5 else 2
    submission_data.append([test_id, real_text_id])

submission_df = pd.DataFrame(submission_data, columns=['id', 'real_text_id'])
submission_df = submission_df.sort_values('id').reset_index(drop=True)

print(f"ğŸ“¤ Submission shape: {submission_df.shape}")
print(f"ğŸ�² Prediction distribution: {submission_df['real_text_id'].value_counts().to_dict()}")
print("\nğŸ‘€ Sample predictions:")
print(submission_df.head(10))

# Save submission
submission_df.to_csv('enhanced_submission.csv', index=False)
print("âœ… Submission saved as 'enhanced_submission.csv'")


# Analyze prediction confidence
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(test_probabilities, bins=30, alpha=0.7, edgecolor='black')
plt.title('Distribution of Prediction Probabilities')
plt.xlabel('Probability (Text1 is Real)')
plt.ylabel('Frequency')
plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
confidence_scores = np.abs(test_probabilities - 0.5)
plt.hist(confidence_scores, bins=30, alpha=0.7, edgecolor='black', color='orange')
plt.title('Prediction Confidence Distribution')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
submission_counts = submission_df['real_text_id'].value_counts()
plt.pie(submission_counts.values, labels=[f'Text {i}' for i in submission_counts.index], 
        autopct='%1.1f%%', startangle=90)
plt.title('Final Prediction Distribution')

plt.tight_layout()
plt.show()

# High confidence vs low confidence predictions
high_confidence = confidence_scores > 0.3
low_confidence = confidence_scores <= 0.3

print(f"ğŸ�¯ High confidence predictions: {high_confidence.sum()} ({high_confidence.mean()*100:.1f}%)")
print(f"ğŸ¤” Low confidence predictions: {low_confidence.sum()} ({low_confidence.mean()*100:.1f}%)")

# Feature importance analysis (using the best single model)
best_model_name = results_df.index[0]
best_model = models[best_model_name]
best_model.fit(X_train_combined, y_train)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
    
    plt.figure(figsize=(12, 6))
    
    # BERT features importance
    bert_importance = feature_importance[:n_components_bert]
    plt.subplot(1, 2, 1)
    plt.plot(bert_importance)
    plt.title(f'BERT PCA Feature Importance ({best_model_name})')
    plt.xlabel('PCA Component')
    plt.ylabel('Importance')
    plt.grid(True, alpha=0.3)
    
    # Statistical features importance
    stat_importance = feature_importance[n_components_bert:]
    plt.subplot(1, 2, 2)
    plt.bar(range(len(stat_importance)), stat_importance)
    plt.title('Statistical Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"ğŸ“Š Top 5 most important features (overall):")
    top_features = np.argsort(feature_importance)[-5:][::-1]
    for i, idx in enumerate(top_features):
        print(f"   {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")




