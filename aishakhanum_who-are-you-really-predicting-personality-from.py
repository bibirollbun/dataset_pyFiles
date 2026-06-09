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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier



# Load CSVs
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')



train.head()  # shows first 5 rows of training data



# Add a dummy column to test so we can combine
test['Personality'] = np.nan

# Combine for uniform preprocessing
data = pd.concat([train, test], ignore_index=True)



 # Define columns
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']
cat_cols = ['Stage_fear', 'Drained_after_socializing']

# Fill numeric columns with median
for col in num_cols:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].median())

# Fill categorical columns with mode (safe)
for col in cat_cols:
    if col in data.columns:
        mode_val = data[col].mode()
        if not mode_val.empty:
            data[col] = data[col].fillna(mode_val[0])
        data[col] = data[col].map({'No': 0, 'Yes': 1})  # Encode




# Combine datasets
test['Personality'] = np.nan
data = pd.concat([train, test], ignore_index=True)

print("âœ… Combined train & test shape:", data.shape)
print("ğŸ§¾ Columns:", list(data.columns))



# Define columns
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']
cat_cols = ['Stage_fear', 'Drained_after_socializing']

# Fill numeric columns
for col in num_cols:
    if col in data.columns:
        before = data[col].isna().sum()
        data[col] = data[col].fillna(data[col].median())
        after = data[col].isna().sum()
        print(f"ğŸ§® {col}: filled {before} â†’ {after} missing values")

# Fill categorical columns
for col in cat_cols:
    if col in data.columns:
        mode_val = data[col].mode()
        if not mode_val.empty:
            before = data[col].isna().sum()
            data[col] = data[col].fillna(mode_val[0])
            after = data[col].isna().sum()
            print(f"ğŸ“� {col}: filled {before} â†’ {after} missing")
        data[col] = data[col].map({'No': 0, 'Yes': 1})
        print(f"ğŸ”� {col}: converted Yes/No to 1/0")



from sklearn.preprocessing import LabelEncoder

# Split back into cleaned train and test sets
train_clean = data[~data['Personality'].isna()].copy()
test_clean = data[data['Personality'].isna()].copy()

print("ğŸ”¹ Cleaned Train Shape:", train_clean.shape)
print("ğŸ”¹ Cleaned Test Shape:", test_clean.shape)

# Encode the target labels: 'Introvert' â†’ 1, 'Extrovert' â†’ 0
le = LabelEncoder()
train_clean['Personality'] = le.fit_transform(train_clean['Personality'].astype(str))

# Print label encoding result
print("âœ… Label Encoding Classes:", list(le.classes_))  # Should be ['Extrovert', 'Introvert']
print(train_clean['Personality'].value_counts())  # Show counts per class



# Drop ID and target from train, and ID from test
X = train_clean.drop(['id', 'Personality'], axis=1)
y = train_clean['Personality']
X_test = test_clean.drop(['id', 'Personality'], axis=1)

print("âœ… Feature matrix X shape:", X.shape)
print("âœ… Target y shape:", y.shape)



from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("ğŸŒ² Starting 5-Fold Cross-Validation with RandomForest...")

cv_scores = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_fold, y_train_fold)
    
    preds = model.predict(X_val_fold)
    acc = accuracy_score(y_val_fold, preds)
    cv_scores.append(acc)
    
    print(f"ğŸ“Š Fold {fold + 1} Accuracy: {acc:.4f}")

print("âœ… Average Accuracy:", np.mean(cv_scores).round(4))



from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

print("ğŸ”§ Tuning XGBoost with GridSearchCV...")

# Define model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}

# Grid search with 3-fold cross-validation
grid = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='accuracy', cv=3)
grid.fit(X, y)

# Best model
xgb_best = grid.best_estimator_

# Output best params
print("âœ… Best XGBoost Parameters:", grid.best_params_)



from sklearn.ensemble import VotingClassifier

print("ğŸ¤� Training Ensemble Model (RandomForest + XGBoost)...")

# Refit Random Forest with full data
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_clf.fit(X, y)

# Use best XGBoost model from previous step (already tuned)
ensemble = VotingClassifier(estimators=[
    ('rf', rf_clf),
    ('xgb', xgb_best)
], voting='soft')

ensemble.fit(X, y)
print("âœ… Ensemble model trained successfully!")



# Optional: Evaluate ensemble on a hold-out fold (e.g., last fold from earlier)
preds_ensemble = ensemble.predict(X_val_fold)
acc_ensemble = accuracy_score(y_val_fold, preds_ensemble)
print(f"ğŸ“Š Ensemble Accuracy on Validation Fold: {acc_ensemble:.4f}")



import seaborn as sns
import matplotlib.pyplot as plt

# Set dark theme and font style
sns.set_theme(style="darkgrid", palette="coolwarm", font_scale=1.3)
plt.rcParams['figure.figsize'] = (10, 6)



plt.figure()
sns.boxplot(x=y, y=X['Time_spent_Alone'])
plt.title("Time Spent Alone by Personality", fontsize=16, weight='bold')
plt.xlabel("Personality (0=Extrovert, 1=Introvert)")
plt.ylabel("Time Spent Alone")
plt.show()



plt.figure()
corr = X.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=16, weight='bold')
plt.show()



from sklearn.metrics import confusion_matrix

# Use previous val fold
y_val_pred = ensemble.predict(X_val_fold)
cm = confusion_matrix(y_val_fold, y_val_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False)
plt.title("Confusion Matrix on Validation Fold", fontsize=16, weight='bold')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



selected = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size', 'Post_frequency']
sns.pairplot(pd.concat([X[selected], y.rename('Personality')], axis=1),
             hue='Personality', palette='coolwarm', diag_kind='kde', corner=True)
plt.suptitle("Pairplot of Selected Features by Personality", fontsize=20, weight='bold', y=1.02)
plt.show()



# Make predictions on test set
test_preds = ensemble.predict(X_test)

# Convert back to 'Extrovert' / 'Introvert'
submission['Personality'] = le.inverse_transform(test_preds)

# Save CSV
submission.to_csv("my_submission.csv", index=False)
print("ğŸ“� Submission file saved as: my_submission.csv")


