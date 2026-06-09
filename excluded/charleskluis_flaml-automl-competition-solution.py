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


!pip install flaml pandas numpy scikit-learn matplotlib seaborn joblib


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import flaml
import time
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory for results
output_dir = '/kaggle/working'
os.makedirs(output_dir, exist_ok=True)

# Load the data
print("Loading data...")
train_data = pd.read_csv('/kaggle/input/playground-series-s3e22/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s3e22/test.csv')

print("\nTrain data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Function to preprocess data
def preprocess_data(df, label_encoders=None, is_train=True):
    # Make a copy of the dataset
    data = df.copy()
    
    # Fill missing values
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna('missing')
        else:
            if col not in ['id', 'lesion_1', 'lesion_2', 'lesion_3']:
                data[col] = data[col].fillna(data[col].median())
    
    # Separate features and target
    if is_train:
        X = data.drop(['id', 'outcome'], axis=1)
        y = data['outcome']
    else:
        X = data.drop(['id'], axis=1)
        y = None
    
    # Initialize label encoders dictionary if not provided
    if label_encoders is None:
        label_encoders = {}
    
    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        if is_train:
            label_encoders[col] = LabelEncoder()
            X[col] = label_encoders[col].fit_transform(X[col])
        else:
            # Handle unseen categories in test set
            le = label_encoders[col]
            X[col] = X[col].apply(lambda x: x if x in le.classes_ else 'missing')
            X[col] = le.transform(X[col])
    
    return X, y, label_encoders

# Preprocess train data
print("\nPreprocessing data...")
X_train, y_train, label_encoders = preprocess_data(train_data)

# Preprocess test data using the same label encoders
X_test, _, _ = preprocess_data(test_data, label_encoders, is_train=False)

# Split training data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Train model using FLAML with minimal settings
print("\nTraining model with FLAML AutoML...")
start_time = time.time()

# Configure FLAML with minimal settings
automl = flaml.AutoML()
automl.fit(
    X_train_split, 
    y_train_split,
    task='classification',
    time_budget=600,  # 10 minutes
    metric='micro_f1'  # Using built-in micro F1 score
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# Evaluate model
print("\nBest model:", automl.best_estimator)
print("Best configuration:", automl.best_config)
print(f"Best validation score: {automl.best_loss * -1:.4f}")

# Evaluate on validation set
y_pred = automl.predict(X_val)
val_f1 = f1_score(y_val, y_pred, average='micro')
print(f"\nValidation F1 Score: {val_f1:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_train), 
            yticklabels=np.unique(y_train))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

# Generate predictions for test set
print("\nGenerating predictions for test set...")
test_predictions = automl.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'outcome': test_predictions
})

submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
print("\nSubmission file created:", os.path.join(output_dir, 'submission.csv'))

# Save the model for future use
joblib.dump(automl, os.path.join(output_dir, 'flaml_model.pkl'))
print("\nModel saved as:", os.path.join(output_dir, 'flaml_model.pkl'))

# Save label encoders for future use
joblib.dump(label_encoders, os.path.join(output_dir, 'label_encoders.pkl'))
print("\nLabel encoders saved as:", os.path.join(output_dir, 'label_encoders.pkl'))

print("\nEnd of script")

