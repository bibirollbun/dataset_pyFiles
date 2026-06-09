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
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt



train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
test['Personality'] = np.nan



data = pd.concat([train, test], ignore_index=True)


num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']
cat_cols = ['Stage_fear', 'Drained_after_socializing']



for col in num_cols:
    data[col] = data[col].fillna(data[col].median())

for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])
    # Map categorical strings to numeric explicitly
    data[col] = data[col].map({'No': 0, 'Yes': 1})


# 3. Split back BEFORE label encoding
train_clean = data[~data['Personality'].isna()].copy()
test_clean = data[data['Personality'].isna()].copy()



# 4. Label encode target only in train
le = LabelEncoder()
train_clean['Personality'] = le.fit_transform(train_clean['Personality'].astype(str))

X = train_clean.drop(['id', 'Personality'], axis=1)
y = train_clean['Personality']
X_test = test_clean.drop(['id', 'Personality'], axis=1)


print("ğŸ“Š Cross-validating RandomForest...")
cv_scores = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_fold, y_train_fold)
    preds = rf.predict(X_val_fold)
    acc = accuracy_score(y_val_fold, preds)
    cv_scores.append(acc)
    print(f"Fold {fold+1} Accuracy: {acc:.4f}")
print(f"âœ… Avg Accuracy: {np.mean(cv_scores):.4f}\n")



print("ğŸ”§ GridSearchCV on XGBoost...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}
grid_search = GridSearchCV(xgb, param_grid, scoring='accuracy', cv=3)
grid_search.fit(X, y)
xgb_best = grid_search.best_estimator_
print("âœ… Best XGBoost Params:", grid_search.best_params_)



print("\nğŸ¤� Training Ensemble...")
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_clf.fit(X, y)
ensemble = VotingClassifier(estimators=[('rf', rf_clf), ('xgb', xgb_best)], voting='soft')
ensemble.fit(X, y)



import seaborn as sns
import matplotlib.pyplot as plt

# Set dark theme with cool palette and larger fonts for clarity
sns.set_theme(style="darkgrid", palette="coolwarm", font_scale=1.3)
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Feature Distribution by Personality Class (Boxplot)
plt.figure()
sns.boxplot(x=y, y=X['Time_spent_Alone'], palette="coolwarm")
plt.title("Time Spent Alone Distribution by Personality Class", fontsize=18, weight='bold')
plt.xlabel("Personality Class")
plt.ylabel("Time Spent Alone")
plt.show()

# 2. Correlation Heatmap of Features
plt.figure()
corr = X.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=18, weight='bold')
plt.show()

# 3. Confusion Matrix on Validation Fold (Example)
from sklearn.metrics import confusion_matrix

# Use last fold validation predictions from your CV loop or final model
# Example assumes rf and X_val_fold, y_val_fold from last fold
y_val_pred = rf.predict(X_val_fold)  # or ensemble.predict(X_val_fold)
cm = confusion_matrix(y_val_fold, y_val_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False)
plt.title("Confusion Matrix on Validation Set", fontsize=18, weight='bold')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 4. Pairplot for Selected Features Colored by Personality
selected_features = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size', 'Post_frequency']
sns.pairplot(pd.concat([X[selected_features], y.rename('Personality')], axis=1),
             hue='Personality', palette='coolwarm', diag_kind='kde', corner=True)
plt.suptitle("Pairplot of Selected Features by Personality Class", fontsize=20, weight='bold', y=1.02)
plt.show()



test_preds = ensemble.predict(X_test)
submission['Personality'] = le.inverse_transform(test_preds)
submission.to_csv("my_submission.csv", index=False)
print("ğŸ“� Submission saved as my_submission.csv")


