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


train_df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train_df.head()


print(train_df.shape)
print(test_df.shape)


train_df.isna().sum()


test_df.isna().sum()


filtered_df = train_df.dropna(subset=['Drained_after_socializing', 'Personality'])

# Group by 'Drained_after_socializing' and 'Personality', then count
group_counts = filtered_df.groupby(['Drained_after_socializing', 'Personality']).size().unstack(fill_value=0)

# Calculate percentage for each group
percentages = group_counts.div(group_counts.sum(axis=1), axis=0) * 100

print("Percentage of personalities based on Drained_after_socializing:")
print(percentages)


columns_to_check = [col for col in train_df.columns if col not in ['id', 'Personality']]

# For categorical and discrete values, convert to string for grouping
train_df[columns_to_check] = train_df[columns_to_check].astype(str)

# Function to calculate percentage breakdown
def calculate_personality_percentages(df, column):
    filtered = df.dropna(subset=[column, 'Personality'])
    group_counts = filtered.groupby([column, 'Personality']).size().unstack(fill_value=0)
    percentages = group_counts.div(group_counts.sum(axis=1), axis=0) * 100
    return percentages

# Calculate and print percentages for all columns
for col in columns_to_check:
    print(f"\nðŸ“Š Personality percentages grouped by '{col}':")
    print(calculate_personality_percentages(train_df, col))


def categorize_features(df):
    df = df.copy()

    # Ensure numeric types (safely handles strings and NaNs)
    for col in ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Time_spent_Alone
    df['Time_spent_Alone_Level'] = pd.cut(
        df['Time_spent_Alone'],
        bins=[-float('inf'), 3, 3.00001, float('inf')],
        labels=['high', 'ok', 'low']
    )

    # Social_event_attendance
    df['Social_event_attendance_Level'] = pd.cut(
        df['Social_event_attendance'],
        bins=[-float('inf'), 2, 3, float('inf')],
        labels=['low', 'ok', 'high']
    )

    # Going_outside
    df['Going_outside_Level'] = pd.cut(
        df['Going_outside'],
        bins=[-float('inf'), 2, 3, float('inf')],
        labels=['low', 'ok', 'high']
    )

    # Friends_circle_size
    df['Friends_circle_size_Level'] = pd.cut(
        df['Friends_circle_size'],
        bins=[-float('inf'), 2, 5, float('inf')],
        labels=['low', 'ok', 'high']
    )

    # Post_frequency
    df['Post_frequency_Level'] = pd.cut(
        df['Post_frequency'],
        bins=[-float('inf'), 2, 3, float('inf')],
        labels=['low', 'ok', 'high']
    )

    return df



train_df = categorize_features(train_df)
test_df = categorize_features(test_df)



train_df.head()


def assign_scores(df):
    df = df.copy()

    # Map values to scores
    scoring = {
        'Time_spent_Alone_Level': {'high': (2, 0), 'ok': (1, 1), 'low': (0, 2)},
        'Social_event_attendance_Level': {'low': (2, 0), 'ok': (1, 1), 'high': (0, 2)},
        'Going_outside_Level': {'low': (2, 0), 'ok': (1, 1), 'high': (0, 2)},
        'Friends_circle_size_Level': {'low': (2, 0), 'ok': (1, 1), 'high': (0, 2)},
        'Post_frequency_Level': {'low': (2, 0), 'ok': (1, 1), 'high': (0, 2)},
        'Stage_fear': {'Yes': (2, 0), 'No': (0, 2)},
        'Drained_after_socializing': {'Yes': (2, 0), 'No': (0, 2)}
    }

    # Initialize scores
    df['Introvert_score'] = 0
    df['Extrovert_score'] = 0

    # Apply scoring
    for col, mapping in scoring.items():
        intro_scores = df[col].map(lambda x: mapping.get(x, (0, 0))[0])
        extro_scores = df[col].map(lambda x: mapping.get(x, (0, 0))[1])
        df['Introvert_score'] += intro_scores.fillna(0)
        df['Extrovert_score'] += extro_scores.fillna(0)

    return df



train_df = assign_scores(train_df)
test_df = assign_scores(test_df)



train_df.head()


train_df['Predicted_Personality'] = train_df.apply(lambda row: 'Introvert' if row['Introvert_score'] > row['Extrovert_score'] else 'Extrovert', axis=1)


correct_predictions = (train_df['Predicted_Personality'] == train_df['Personality']).sum()
total_predictions = train_df['Personality'].notna().sum()  # exclude NaNs

accuracy = correct_predictions / total_predictions * 100

print(f"âœ… Accuracy of rule-based personality prediction: {accuracy:.2f}%")


# Step 1: Predict Personality based on higher score
test_df['Predicted_Personality'] = test_df.apply(
    lambda row: 'Introvert' if row['Introvert_score'] > row['Extrovert_score'] else 'Extrovert',
    axis=1
)

# Step 2: Prepare submission DataFrame
submission_df = test_df[['id', 'Predicted_Personality']].rename(columns={
    'Predicted_Personality': 'Personality'
})

# Step 3: Save to CSV
submission_df.to_csv('submission.csv', index=False)

print("âœ… Submission file 'submission.csv' created successfully!")



train_df.head()


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Label encode the relevant columns
categorical_cols = [
    'Time_spent_Alone_Level',
    'Social_event_attendance_Level',
    'Going_outside_Level',
    'Friends_circle_size_Level',
    'Post_frequency_Level',
    'Stage_fear',
    'Drained_after_socializing'
]

# Encode labels for train and test using same encoders
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(combined)
    train_df[col + '_enc'] = le.transform(train_df[col].astype(str))
    test_df[col + '_enc'] = le.transform(test_df[col].astype(str))
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
train_df['Personality_enc'] = target_encoder.fit_transform(train_df['Personality'])

# 2. Prepare features and target
feature_cols = [col + '_enc' for col in categorical_cols]
X_train = train_df[feature_cols]
y_train = train_df['Personality_enc']
X_test = test_df[feature_cols]

# 3. Train model
clf = RandomForestClassifier(n_estimators=1000, random_state=42)
clf.fit(X_train, y_train)

# 4. Predict
train_preds = clf.predict(X_train)
test_preds = clf.predict(X_test)

# 5. Evaluate accuracy on training set
train_accuracy = accuracy_score(y_train, train_preds)
print(f"âœ… Model accuracy on training data: {train_accuracy * 100:.2f}%")

# 6. Decode predictions
test_df['Personality'] = target_encoder.inverse_transform(test_preds)

# 7. Prepare submission
submission_df = test_df[['id', 'Personality']]
submission_df.to_csv('submission_rf.csv', index=False)
print("âœ… Enhanced submission file 'submission.csv' saved successfully.")



import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# === Step 1: Encode categorical features ===
categorical_cols = [
    'Time_spent_Alone_Level',
    'Social_event_attendance_Level',
    'Going_outside_Level',
    'Friends_circle_size_Level',
    'Post_frequency_Level',
    'Stage_fear',
    'Drained_after_socializing'
]

# Encode labels (train + test jointly)
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    all_vals = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(all_vals)
    train_df[col + '_enc'] = le.transform(train_df[col].astype(str))
    test_df[col + '_enc'] = le.transform(test_df[col].astype(str))
    encoders[col] = le

# Target encoding
target_encoder = LabelEncoder()
train_df['Personality_enc'] = target_encoder.fit_transform(train_df['Personality'])

# === Step 2: Prepare full feature set ===
numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                'Friends_circle_size', 'Post_frequency', 'Introvert_score', 'Extrovert_score']
encoded_cols = [col + '_enc' for col in categorical_cols]

all_features = numeric_cols + encoded_cols

# Fill NaNs with median for numeric, -1 for encoded
train_df[numeric_cols] = train_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
test_df[numeric_cols] = test_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

for col in numeric_cols:
    median_val = train_df[col].median()
    train_df[col].fillna(median_val, inplace=True)
    test_df[col].fillna(median_val, inplace=True)

for col in encoded_cols:
    train_df[col].fillna(-1, inplace=True)
    test_df[col].fillna(-1, inplace=True)

# Normalize numeric features
scaler = StandardScaler()
train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

# Final data
X_train = train_df[all_features]
y_train = train_df['Personality_enc']
X_test = test_df[all_features]

# === Step 3: Train Gradient Boosting model ===
model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# === Step 4: Predictions ===
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Evaluate training accuracy
train_accuracy = accuracy_score(y_train, train_preds)
print(f"ðŸ“Š GradientBoosting training accuracy: {train_accuracy * 100:.2f}%")

# === Step 5: Create submission ===
test_df['Personality'] = target_encoder.inverse_transform(test_preds)
submission_df = test_df[['id', 'Personality']]
submission_df.to_csv('submission_min.csv', index=False)
print("âœ… Submission file 'submission.csv' created with boosted predictions.")





