# -----------------------------
# KNN Personality Prediction Model - Kaggle Notebook
# -----------------------------

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import os

# Optional: check files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

# Fill missing values
train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)

# Encode categorical features
le_stage = LabelEncoder()
le_drain = LabelEncoder()
le_personality = LabelEncoder()

train['Stage_fear'] = le_stage.fit_transform(train['Stage_fear'].astype(str))
train['Drained_after_socializing'] = le_drain.fit_transform(train['Drained_after_socializing'].astype(str))
train['Personality'] = le_personality.fit_transform(train['Personality'].astype(str))  # 0=Extrovert, 1=Introvert

test['Stage_fear'] = le_stage.transform(test['Stage_fear'].astype(str))
test['Drained_after_socializing'] = le_drain.transform(test['Drained_after_socializing'].astype(str))

# Features used for prediction
features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
            'Going_outside', 'Drained_after_socializing',
            'Friends_circle_size', 'Post_frequency']

X_train = train[features]
y_train = train['Personality']
X_test = test[features]

# Train the KNN model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Predict test data
test_preds = knn.predict(X_test)
test_preds_labels = le_personality.inverse_transform(test_preds)

# Prepare submission
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': test_preds_labels
})

# Save submission
submission.to_csv("/kaggle/working/submission.csv", index=False)

# Final message
print("✅ Submission file saved as 'submission.csv'")


