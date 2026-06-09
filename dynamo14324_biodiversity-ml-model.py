# Import essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
import random

# Scikit-learn for modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Set visualization styles
plt.style.use('seaborn-whitegrid')
sns.set_palette('viridis')
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# Create the simulated metadata dataframe
# In a real scenario, you would load this from the competition data

# Generate sample data parameters
species = ['Bird_Species_A', 'Bird_Species_B', 'Bird_Species_C', 'Amphibian_Species_A', 
           'Amphibian_Species_B', 'Mammal_Species_A', 'Mammal_Species_B', 'Insect_Species_A']
taxonomic_groups = ['Bird', 'Bird', 'Bird', 'Amphibian', 'Amphibian', 'Mammal', 'Mammal', 'Insect']
conservation_status = ['Least Concern', 'Vulnerable', 'Endangered', 'Near Threatened', 
                       'Vulnerable', 'Least Concern', 'Endangered', 'Data Deficient']

# Create sample metadata
np.random.seed(RANDOM_SEED)
n_samples = 500
metadata = {
    'recording_id': [f'rec_{i:04d}' for i in range(n_samples)],
    'species': np.random.choice(species, n_samples),
    'duration': np.random.uniform(3, 15, n_samples).round(2),
    'time_of_day': np.random.choice(['Dawn', 'Day', 'Dusk', 'Night'], n_samples),
    'habitat_type': np.random.choice(['Forest', 'Wetland', 'Grassland', 'Restoration_Area'], n_samples),
    'recording_date': pd.date_range(start='2024-01-01', periods=n_samples, freq='H'),
    'recording_quality': np.random.choice(['High', 'Medium', 'Low'], n_samples, p=[0.6, 0.3, 0.1])
}

# Create DataFrame
df = pd.DataFrame(metadata)

# Add taxonomic group based on species
species_to_group = dict(zip(species, taxonomic_groups))
df['taxonomic_group'] = df['species'].map(species_to_group)

# Add conservation status based on species
species_to_status = dict(zip(species, conservation_status))
df['conservation_status'] = df['species'].map(species_to_status)

# Display the first few rows
print("Simulated Metadata:")
df.head()


# Function to extract MFCC features (Simulated)
# In a real scenario, this function would load an audio file and compute MFCCs
def extract_mfcc_simulated(recording_id, duration, n_mfcc=20):
    # Simulate feature extraction based on duration
    # The number of frames depends on the duration, sample rate, hop length
    # Let's assume a sample rate of 22050 Hz and hop length of 512
    sr = 22050
    hop_length = 512
    n_frames = int(np.ceil(duration * sr / hop_length))
    
    # Generate random MFCCs of shape (n_mfcc, n_frames)
    # We add some noise based on recording_id hash for variability
    np.random.seed(int(hash(recording_id) % (2**32 - 1)))
    mfccs = np.random.randn(n_mfcc, n_frames)
    
    # We need a fixed-size feature vector for standard ML models.
    # A common approach is to aggregate over the time dimension (frames).
    # We'll compute the mean and standard deviation of MFCCs across time.
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    # Concatenate mean and std dev
    features = np.concatenate((mfcc_mean, mfcc_std))
    
    return features

# Apply the feature extraction function to each recording
print("Extracting Features...")
features_list = []
for idx, row in df.iterrows():
    if idx % 100 == 0:
        print(f"Processed {idx} of {len(df)} recordings")
    features_list.append(extract_mfcc_simulated(row['recording_id'], row['duration']))
df['features'] = features_list

# Display the DataFrame with features
print("\nDataFrame with extracted features:")
df.head()


# Convert the list of features into separate columns for visualization
feature_df = pd.DataFrame(df['features'].tolist(), index=df.index)
n_mfcc = 20 # Must match the value used in extraction
feature_df.columns = [f'mfcc_mean_{i}' for i in range(n_mfcc)] + [f'mfcc_std_{i}' for i in range(n_mfcc)]

# Combine with the original DataFrame
df_combined = pd.concat([df, feature_df], axis=1)

# Plot the distribution of the first few mean MFCCs by species
plt.figure(figsize=(15, 10))
for i in range(4): # Plot first 4 mean MFCCs
    plt.subplot(2, 2, i+1)
    sns.kdeplot(data=df_combined, x=f'mfcc_mean_{i}', hue='species', fill=True, common_norm=False)
    plt.title(f'Distribution of MFCC Mean {i}')
plt.tight_layout()
plt.show()


# 1. Encode Labels
X = np.array(df['features'].tolist())
y_labels = df['species'].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)

print(f"Shape of feature matrix X: {X.shape}")
print(f"Shape of label vector y: {y.shape}")
print(f"Encoded classes: {label_encoder.classes_}")

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# 3. Scale Features
# We fit the scaler only on the training data and transform both train and test sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures scaled successfully.")


# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
    'Support Vector Machine': SVC(random_state=RANDOM_SEED, probability=True), # Probability=True for consistency, can be slow
    'Random Forest': RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100)
}

# Train and evaluate each model
results = {}

for model_name, model in models.items():
    print(f"--- Training {model_name} ---")
    
    # Create a pipeline with scaling and the model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Store results
    results[model_name] = {'accuracy': accuracy, 'report': report, 'confusion_matrix': cm, 'pipeline': pipeline}
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix:")
    print(cm)
    print("------------------------------\n")


# Find the best model based on accuracy
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_cm = results[best_model_name]['confusion_matrix']

print(f"Best performing model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")

# Plot confusion matrix for the best model
plt.figure(figsize=(10, 8))
sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix for {best_model_name}', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right' if len(label_encoder.classes_) > 5 else 'center')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

