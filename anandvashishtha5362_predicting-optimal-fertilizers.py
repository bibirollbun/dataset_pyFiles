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


# libraries to download
import pandas as pd
# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization 
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Machine learning models
from sklearn.ensemble import RandomForestClassifier

# Model selection and evaluation
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

# Utility for saving/loading models and encoders
import joblib

import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df.head()


import pandas as pd

# Load the datasets using Kaggle input paths
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# --- TRAINING DATA INSPECTION ---
print("\n=== TRAINING DATA ===\n")
print("--- Info ---")
train_df.info()
print("\n--- Description (Numerical Columns) ---")
print(train_df.describe())
print("\n--- Missing Values ---")
print(train_df.isnull().sum())
print("\n--- Duplicate Rows ---")
print(train_df.duplicated().sum())

# --- TEST DATA INSPECTION ---
print("\n=== TEST DATA ===\n")
print("--- Info ---")
test_df.info()
print("\n--- Description (Numerical Columns) ---")
print(test_df.describe())
print("\n--- Missing Values ---")
print(test_df.isnull().sum())
print("\n--- Duplicate Rows ---")
print(test_df.duplicated().sum())

# --- SAMPLE SUBMISSION INSPECTION ---
print("\n=== SAMPLE SUBMISSION ===\n")
print("--- Info ---")
sample_submission_df.info()
print("\n--- Head ---")
print(sample_submission_df.head())



from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Separate features and target
X_train = train_df.drop(['Fertilizer Name', 'id'], axis=1)
y_train = train_df['Fertilizer Name']
X_test = test_df.drop('id', axis=1)

# Concatenate for consistent preprocessing
combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

# Label encode categorical features
le_soil = LabelEncoder()
le_crop = LabelEncoder()
combined['Soil Type'] = le_soil.fit_transform(combined['Soil Type'])
combined['Crop Type'] = le_crop.fit_transform(combined['Crop Type'])

# Scale numerical features
num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
scaler = StandardScaler()
combined[num_cols] = scaler.fit_transform(combined[num_cols])

# Split back into train and test
X_train_processed = combined.iloc[:len(X_train), :].reset_index(drop=True)
X_test_processed = combined.iloc[len(X_train):, :].reset_index(drop=True)

# Encode target
le_fert = LabelEncoder()
y_train_encoded = le_fert.fit_transform(y_train)



import joblib
joblib.dump(le_soil, 'le_soil.pkl')
joblib.dump(le_crop, 'le_crop.pkl')
joblib.dump(le_fert, 'le_fert.pkl')
joblib.dump(scaler, 'scaler.pkl')



print("Processed training features shape:", X_train_processed.shape)
print("Processed test features shape:", X_test_processed.shape)
print("Encoded target shape:", y_train_encoded.shape)
print("Unique classes in target:", np.unique(y_train_encoded))



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np



def map_at_3(y_true, y_pred_proba):
    # Get indices of top 3 predictions for each sample
    top_3 = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
    y_true = np.array(y_true)
    score = 0.0
    for i in range(len(y_true)):
        if y_true[i] in top_3[i]:
            rank = np.where(top_3[i] == y_true[i])[0][0] + 1  # rank (1-based)
            score += 1.0 / rank
    return score / len(y_true)



# Set up cross-validation
n_splits = 5  # You can increase this for more robust results
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_accuracies = []
fold_map3_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train_processed, y_train_encoded)):
    X_tr, X_val = X_train_processed.iloc[train_idx], X_train_processed.iloc[valid_idx]
    y_tr, y_val = y_train_encoded[train_idx], y_train_encoded[valid_idx]

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)

    # Predict probabilities for validation set
    y_pred_proba = model.predict_proba(X_val)

    # Calculate accuracy and MAP@3
    accuracy = model.score(X_val, y_val)
    map3 = map_at_3(y_val, y_pred_proba)

    fold_accuracies.append(accuracy)
    fold_map3_scores.append(map3)

    print(f"Fold {fold+1}: Accuracy = {accuracy:.4f}, MAP@3 = {map3:.4f}")

print(f"Average Accuracy: {np.mean(fold_accuracies):.4f}")
print(f"Average MAP@3: {np.mean(fold_map3_scores):.4f}")



final_model = RandomForestClassifier(n_estimators=100,max_depth=10, random_state=42, n_jobs=-1)
final_model.fit(X_train_processed, y_train_encoded)



# Predict class probabilities for the test set
test_pred_proba = final_model.predict_proba(X_test_processed)



# Get indices of top 3 predictions for each test sample
top_3_indices = np.argsort(test_pred_proba, axis=1)[:, -3:][:, ::-1]

# Convert indices to fertilizer names row by row
predicted_fertilizers = []
for row in top_3_indices:
    fert_names = le_fert.inverse_transform(row)
    predicted_fertilizers.append(' '.join(fert_names))

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': predicted_fertilizers
})

# Save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")



!ls
from IPython.display import FileLink
FileLink('submission.csv')

# to download submission.csv do run this notebook


num_cols = ['Temparature', 'Humidity', 'Moisture',
            'Nitrogen', 'Potassium', 'Phosphorous']

plt.figure(figsize=(10, 4))
sns.countplot(
    data=df,
    x='Fertilizer Name',
    order=df['Fertilizer Name'].value_counts().index,
    palette='viridis'
)
plt.title('Training-Set Fertilizer Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df[num_cols].hist(
    bins=30,
    figsize=(12, 8),
    color='#3FA7D6',
    edgecolor='black'
)
plt.suptitle('Numeric Feature Distributions', y=1.02)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
corr = df[num_cols].corr()
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt='.2f'
)
plt.title('Feature Correlation Matrix')
plt.show()


