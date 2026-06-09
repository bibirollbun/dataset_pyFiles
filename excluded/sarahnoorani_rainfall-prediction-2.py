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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


# Check and fill missing values
print(train_df.isnull().sum())
print(test_df.isnull().sum())

# Fill missing winddirection in test with median
test_df['winddirection'].fillna(test_df['winddirection'].median(), inplace=True)



# Rainfall class distribution
rainfall_distribution = train_df['rainfall'].value_counts(normalize=True)

# Correlation heatmap
correlation = train_df.drop(columns='id').corr()

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.barplot(x=rainfall_distribution.index, y=rainfall_distribution.values)
plt.title("Rainfall Class Distribution")
plt.xticks([0, 1], ['No Rain', 'Rain'])
plt.ylabel("Proportion")

plt.subplot(1, 2, 2)
sns.heatmap(correlation[['rainfall']].sort_values(by='rainfall', ascending=False), annot=True, cmap='coolwarm')
plt.title("Feature Correlation with Rainfall")

plt.tight_layout()
plt.show()



# Separate features and target
X = train_df.drop(columns=['id', 'rainfall'])
y = train_df['rainfall']
X_test_final = test_df.drop(columns=['id'])

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_final)

# Train model with class_weight='balanced'
model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Validate
y_val_pred = model.predict(X_val_scaled)

# Evaluate
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))



# Predict on test data
y_test_pred = model.predict(X_test_scaled)

# Prepare submission
submission = sample_submission.copy()
submission['rainfall'] = y_test_pred

# Save to CSV
submission.to_csv("final_submission.csv", index=False)
print("✅ Submission file saved as 'final_submission.csv'")



from sklearn.linear_model import LogisticRegression

# Train logistic regression
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Evaluate
y_val_pred_lr = log_reg.predict(X_val_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_val, y_val_pred_lr))
print("Classification Report:\n", classification_report(y_val, y_val_pred_lr))

# Predict test and save
y_test_pred_lr = log_reg.predict(X_test_scaled)
submission_lr = sample_submission.copy()
submission_lr['rainfall'] = y_test_pred_lr
submission_lr.to_csv("submission_logistic_regression.csv", index=False)



from xgboost import XGBClassifier

# Train XGBoost
xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5,
                          scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                          use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Evaluate
y_val_pred_xgb = xgb_model.predict(X_val_scaled)
print("XGBoost Accuracy:", accuracy_score(y_val, y_val_pred_xgb))
print("Classification Report:\n", classification_report(y_val, y_val_pred_xgb))

# Predict test and save
y_test_pred_xgb = xgb_model.predict(X_test_scaled)
submission_xgb = sample_submission.copy()
submission_xgb['rainfall'] = y_test_pred_xgb
submission_xgb.to_csv("submission_xgboost.csv", index=False)



from sklearn.ensemble import VotingClassifier

# Combine models
voting = VotingClassifier(estimators=[
    ('rf', model),
    ('lr', log_reg),
    ('xgb', xgb_model)
], voting='hard')

voting.fit(X_train_scaled, y_train)

# Evaluate
y_val_pred_vote = voting.predict(X_val_scaled)
print("Voting Classifier Accuracy:", accuracy_score(y_val, y_val_pred_vote))
print("Classification Report:\n", classification_report(y_val, y_val_pred_vote))

# Predict test and save
y_test_pred_vote = voting.predict(X_test_scaled)
submission_vote = sample_submission.copy()
submission_vote['rainfall'] = y_test_pred_vote
submission_vote.to_csv("submission_voting_ensemble.csv", index=False)



from sklearn.feature_selection import SelectKBest, f_classif

# Keep top 8 features
selector = SelectKBest(score_func=f_classif, k=8)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Train with reduced features using XGBoost again
xgb_fs = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5,
                       scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                       use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_fs.fit(X_train_selected, y_train)

# Predict test and save
y_test_pred_fs = xgb_fs.predict(X_test_selected)
submission_fs = sample_submission.copy()
submission_fs['rainfall'] = y_test_pred_fs
submission_fs.to_csv("submission_feature_selected_xgboost.csv", index=False)



from sklearn.ensemble import StackingClassifier

# Define base learners
base_learners = [
    ('rf', model),
    ('lr', log_reg),
    ('xgb', xgb_model)
]

# Meta learner
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression(), cv=5, n_jobs=-1)
stacking_model.fit(X_train_scaled, y_train)

# Evaluation
y_val_pred_stack = stacking_model.predict(X_val_scaled)
print("Stacking Classifier Accuracy:", accuracy_score(y_val, y_val_pred_stack))
print("Classification Report:\n", classification_report(y_val, y_val_pred_stack))

# Prediction and submission
y_test_pred_stack = stacking_model.predict(X_test_scaled)
submission_stack = sample_submission.copy()
submission_stack['rainfall'] = y_test_pred_stack
submission_stack.to_csv("submission_stacking.csv", index=False)



from sklearn.neural_network import MLPClassifier

# Train a basic Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Evaluation
y_val_pred_mlp = mlp.predict(X_val_scaled)
print("Neural Network Accuracy:", accuracy_score(y_val, y_val_pred_mlp))
print("Classification Report:\n", classification_report(y_val, y_val_pred_mlp))

# Prediction and submission
y_test_pred_mlp = mlp.predict(X_test_scaled)
submission_mlp = sample_submission.copy()
submission_mlp['rainfall'] = y_test_pred_mlp
submission_mlp.to_csv("submission_neural_network.csv", index=False)





