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
import numpy as np
import warnings
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output


print("ğŸ“¥ Loading dataset...")

# Load train & test datasets
train_df = pd.read_csv("/kaggle/input/thapar-kaggle-hack-v02/train.csv")
test_df = pd.read_csv("/kaggle/input/thapar-kaggle-hack-v02/test.csv")

# ğŸ”¹ Reduce Data Size for Faster Training (Optional)
train_df = train_df.sample(frac=0.5, random_state=42)  # Using only 50% of the dataset

# -------------------------------
# âœ… EDA (EXPLORATORY DATA ANALYSIS)
# -------------------------------
print("ğŸ“Š Performing Exploratory Data Analysis...")

# ğŸ”¹ Check for missing values
print("Missing Values:", train_df.isnull().sum().sum())

# ğŸ”¹ Check target class distribution
sns.countplot(x=train_df["target"])
plt.title("Target Class Distribution")
plt.show()


print("ğŸ”„ Preprocessing data...")

# ğŸ”¹ Identify the Target & Features
TARGET_COLUMN = "target"
FEATURE_COLUMNS = [col for col in train_df.columns if col not in ["id", TARGET_COLUMN]]

# ğŸ”¹ Feature Engineering (Fewer New Features for Speed)
train_df["feature_sum"] = train_df[FEATURE_COLUMNS].sum(axis=1)
test_df["feature_sum"] = test_df[FEATURE_COLUMNS].sum(axis=1)

# ğŸ”¹ Update Feature List
FEATURE_COLUMNS += ["feature_sum"]

# ğŸ”¹ Extract Features & Target
X = train_df[FEATURE_COLUMNS]
y = train_df[TARGET_COLUMN]

# ğŸ”¹ Split Data (80% Train, 20% Validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# âœ… Fix SMOTE for Multiclass Classification
print("âš–ï¸� Checking Class Balance...")

# Count class distribution before SMOTE
class_counts = Counter(y_train)
print("Before SMOTE:", class_counts)

# Define SMOTE strategy (Oversample each class to 90% of the majority class)
# Ensure SMOTE does not reduce the original sample size
sampling_strategy = {cls: max(class_counts[cls], int(0.9 * max(class_counts.values()))) for cls in class_counts}

# Apply SMOTE with corrected sampling strategy
smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Count class distribution after SMOTE
print("After SMOTE:", Counter(y_train))

# âœ… Feature Scaling
print("ğŸ“� Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


print("ğŸš€ Training Base Models...")

# ğŸ”¹ Train RandomForest (Reduced Estimators for Speed)
rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# ğŸ”¹ Train Neural Network (MLP)
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=300, random_state=42)
mlp_model.fit(X_train, y_train)


print("ğŸ“Š Evaluating Models...")

# ğŸ”¹ Evaluate RandomForest
y_pred_rf = rf_model.predict(X_val)
accuracy_rf = accuracy_score(y_val, y_pred_rf)

# ğŸ”¹ Evaluate Neural Network
y_pred_mlp = mlp_model.predict(X_val)
accuracy_mlp = accuracy_score(y_val, y_pred_mlp)

# âœ… Faster Hyperparameter Tuning for XGBoost
print("ğŸ�¯ Hyperparameter Tuning for XGBoost...")

xgb_params = {
    "n_estimators": [100],  # Reduced to 100 trees for speed
    "max_depth": [6],
    "learning_rate": [0.1]
}

xgb_grid = GridSearchCV(XGBClassifier(tree_method='hist'), param_grid=xgb_params, cv=3, scoring="accuracy", n_jobs=-1)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_

# ğŸ”¹ Evaluate Optimized XGBoost
y_pred_xgb = best_xgb.predict(X_val)
accuracy_xgb = accuracy_score(y_val, y_pred_xgb)

# âœ… Save Results
results_df = pd.DataFrame({
    "Model": ["RandomForest", "Neural Network (MLP)", "Optimized XGBoost"],
    "Accuracy": [accuracy_rf, accuracy_mlp, accuracy_xgb]
})
results_df.to_csv("final_model_results.csv", index=False)

print(results_df)
print("âœ… Model training completed. Results saved in 'final_model_results.csv'.")




print("ğŸ“¡ Generating Kaggle submission file...")

# ğŸ”¹ Prepare Test Data
X_test = test_df[FEATURE_COLUMNS]
X_test = scaler.transform(X_test)  # Apply the same scaling

# ğŸ”¹ Use the best model (Optimized XGBoost) for final predictions
test_predictions = best_xgb.predict(X_test)

# ğŸ”¹ Create Submission File
submission = pd.DataFrame({"id": test_df["id"], "target": test_predictions})
submission.to_csv("submission.csv", index=False)

print("ğŸ�¯ Submission file 'submission.csv' created! ğŸš€ Upload it to Kaggle.")


