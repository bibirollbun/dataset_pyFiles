# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


#Encode target and categorical features
le_target = LabelEncoder()
train_df['Personality_encoded'] = le_target.fit_transform(train_df['Personality'])

for col in ['Stage_fear', 'Drained_after_socializing']:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))


#Define features and targets
feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Drained_after_socializing',
                'Social_event_attendance', 'Friends_circle_size', 'Post_frequency']

X_train = train_df[feature_cols]
y_train = train_df['Personality_encoded']
X_test = test_df[feature_cols]

# ðŸ”§ Fill missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())



rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


def create_rule_based_baseline(test_df):
    predictions = []
    
    for _, row in test_df.iterrows():
        introvert_score = 0
        extrovert_score = 0
        
        # Time spent alone
        if pd.notna(row.get('Time_spent_Alone', np.nan)):
            if row['Time_spent_Alone'] >= 4:
                introvert_score += 3
            elif row['Time_spent_Alone'] <= 2:
                extrovert_score += 3
        
        # Stage fear
        if pd.notna(row.get('Stage_fear', np.nan)):
            if row['Stage_fear'] in ['Yes', 1, True]:
                introvert_score += 3
            elif row['Stage_fear'] in ['No', 0, False]:
                extrovert_score += 3
        
        # Drained after socializing
        if pd.notna(row.get('Drained_after_socializing', np.nan)):
            if row['Drained_after_socializing'] in ['Yes', 1, True]:
                introvert_score += 3
            elif row['Drained_after_socializing'] in ['No', 0, False]:
                extrovert_score += 3
        
        # Social events
        if pd.notna(row.get('Social_event_attendance', np.nan)):
            if row['Social_event_attendance'] <= 2:
                introvert_score += 1
            elif row['Social_event_attendance'] >= 6:
                extrovert_score += 1
        
        # Friends circle
        if pd.notna(row.get('Friends_circle_size', np.nan)):
            if row['Friends_circle_size'] <= 3:
                introvert_score += 1
            elif row['Friends_circle_size'] >= 8:
                extrovert_score += 1
        
        # Post frequency
        if pd.notna(row.get('Post_frequency', np.nan)):
            if row['Post_frequency'] <= 2:
                introvert_score += 1
            elif row['Post_frequency'] >= 6:
                extrovert_score += 1
        
        # Final prediction
        if introvert_score > extrovert_score:
            predictions.append('Introvert')
        else:
            predictions.append('Extrovert')
    
    return predictions

rule_predictions = create_rule_based_baseline(test_df)


rule_preds_train = create_rule_based_baseline(train_df)
rule_preds_train_encoded = le_target.transform(rule_preds_train)

rf_preds_train = rf.predict(X_train)


ensemble_preds_train = []
for r_rule, r_rf in zip(rule_preds_train_encoded, rf_preds_train):
    # If both agree, use their prediction
    if r_rule == r_rf:
        ensemble_preds_train.append(r_rule)
    else:
        # Else prefer Random Forest (or you can choose rule-based)
        ensemble_preds_train.append(r_rf)


accuracy = accuracy_score(y_train, ensemble_preds_train)
print(f"Weighted Ensemble Training Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_train, ensemble_preds_train, target_names=le_target.classes_))

# Confusion matrix
cm = confusion_matrix(y_train, ensemble_preds_train)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_)
plt.title('Confusion Matrix - Ensemble (Rule-Based + Random Forest)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


rule_preds_test = create_rule_based_baseline(test_df)
rule_preds_test_encoded = le_target.transform(rule_preds_test)

rf_preds_test = rf.predict(X_test)

ensemble_preds_test = []
for r_rule, r_rf in zip(rule_preds_test_encoded, rf_preds_test):
    if r_rule == r_rf:
        ensemble_preds_test.append(r_rule)
    else:
        ensemble_preds_test.append(r_rf)

# Decode final predictions
ensemble_preds_test_labels = le_target.inverse_transform(ensemble_preds_test)


submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': ensemble_preds_test_labels
})

submission.to_csv('submission.csv', index=False)




