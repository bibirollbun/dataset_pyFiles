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


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss


# Load the training data
train_data = pd.read_csv('/kaggle/input/spaceship-titanic-in-all-probability/train.csv')

# Feature selection
features = ['CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Split the data into features and target variable
X = train_data[features]
y = train_data['Transported']

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_data.head()


# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)


# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}


# Fit and calibrate models
calibrated_models = {}

for name, model in models.items():
    # Fit the model
    model.fit(X_train_imputed, y_train)

   
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X_val_imputed, y_val)

    calibrated_models[name] = calibrated_model


# Load the test data (replace 'test.csv' with the actual file path)
test_data = pd.read_csv('/kaggle/input/spaceship-titanic-in-all-probability/test.csv')

# Extract features from the test data
X_test = test_data[features]


# Impute missing values in the test set
X_test_imputed = imputer.transform(X_test)


# Make probability predictions on the test set for each model
probabilities = {}

for name, model in calibrated_models.items():
    probabilities[name] = model.predict_proba(X_test_imputed)[:, 1]  # Probability of being transported


pd.DataFrame(probabilities).head()


import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss
)

# Dictionary to store validation scores
validation_scores = {}

for name, model in calibrated_models.items():
    # Predict probabilities on validation set
    val_probabilities = model.predict_proba(X_val_imputed)[:, 1]
    
    # Convert probabilities to binary predictions (threshold = 0.5)
    val_predictions = (val_probabilities > 0.5).astype(int)

    # Calculate metrics
    brier_score = brier_score_loss(y_val, val_probabilities)
    roc_auc = roc_auc_score(y_val, val_probabilities)
    accuracy = accuracy_score(y_val, val_predictions)
    precision = precision_score(y_val, val_predictions)
    recall = recall_score(y_val, val_predictions)
    f1 = f1_score(y_val, val_predictions)

    validation_scores[name] = {
        'Brier Score': brier_score,
        'ROC AUC': roc_auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Convert to DataFrame for easy viewing
scores_df = pd.DataFrame(validation_scores).T  # Transpose to get models as rows
print(scores_df)



# Create a submission file for each model
for name, probs in probabilities.items():
    submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': probs})
    submission.to_csv(f'submission_{name.replace(" ", "_")}.csv', index=False)


submission

