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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb


# Custom MAP@3 metric
def map_at_3(y_true, y_pred, n=3):
    map_score = 0.0
    for i in range(len(y_true)):
        score = 0.0
        used_labels = set()
        for k in range(min(n, len(y_pred[i]))):
            if y_pred[i][k] == y_true[i] and y_pred[i][k] not in used_labels:
                score += 1.0 / (k + 1)
                used_labels.add(y_pred[i][k])
        map_score += score / min(n, len(y_pred[i]))
    return map_score / len(y_true)


# Load data (replace with actual file paths)
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


print('train data')
print(train_df.head())

print('test data')
print(test_df.head())


print(train_df['Fertilizer Name'].value_counts(normalize=True))


print(train_df.columns)


missing_values = train_df.isna().sum()
print("Missing values in each column:")
print(missing_values)



print("Train Data Columns:", train_df.columns.tolist())


# Preprocessing
feature_cols = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 
                'Nitrogen', 'Potassium', 'Phosphorous']  # Initialize feature_cols
categorical_cols = ['Soil Type', 'Crop Type']
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])


# Feature engineering (ensure unique feature names)
if 'temp_humidity_interaction' not in train_df.columns:
    train_df['temp_humidity_interaction'] = train_df['Temparature'] * train_df['Humidity']
    test_df['temp_humidity_interaction'] = test_df['Temparature'] * test_df['Humidity']
    feature_cols.append('temp_humidity_interaction')
if 'nitrogen_phosphorous_interaction' not in train_df.columns:
    train_df['nitrogen_phosphorous_interaction'] = train_df['Nitrogen'] * train_df['Phosphorous']
    test_df['nitrogen_phosphorous_interaction'] = test_df['Nitrogen'] * test_df['Phosphorous']
    feature_cols.append('nitrogen_phosphorous_interaction')




# Encode target
le_target = LabelEncoder()
train_df['Fertilizer Name'] = le_target.fit_transform(train_df['Fertilizer Name'])


# Split data
X = train_df[feature_cols]
y = train_df['Fertilizer Name']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Train model
model = XGBClassifier(random_state=42, n_estimators=500, learning_rate=1.0, max_depth=5)
model.fit(X_train, y_train)


# Predict
val_probs = model.predict_proba(X_val)
def get_top_n_predictions(probs, label_encoder, n=3):
    top_n_indices = np.argsort(probs, axis=1)[:, -n:][:, ::-1]
    top_n_labels = label_encoder.inverse_transform(top_n_indices.flatten()).reshape(top_n_indices.shape)
    return top_n_labels


val_pred_labels = get_top_n_predictions(val_probs, le_target, n=3)
y_val_labels = le_target.inverse_transform(y_val)
print(f"Validation MAP@3: {map_at_3(y_val_labels, val_pred_labels):.4f}")


# Visualization 1: Class Distribution
plt.figure(figsize=(10, 6))
class_dist = train_df['Fertilizer Name'].value_counts(normalize=True).sort_index()
sns.barplot(x=class_dist.index, y=class_dist.values)
plt.title('Fertilizer Class Distribution')
plt.xlabel('Fertilizer Class (Encoded)')
plt.ylabel('Proportion')
plt.xticks(ticks=range(len(le_target.classes_)), labels=le_target.classes_, rotation=45)
plt.show()


# Visualization 2: Feature Importance
plt.figure(figsize=(10, 6))
feature_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title('Feature Importance (XGBoost)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


#Visualization 3: Prediction Errors (Correct Position in Top 3)
correct_positions = []
for i in range(len(y_val_labels)):
    true_label = y_val_labels[i]
    pred_labels = val_pred_labels[i]
    position = next((k + 1 for k, pred in enumerate(pred_labels) if pred == true_label), 0)
    correct_positions.append(position)

plt.figure(figsize=(10, 6))
sns.countplot(x=correct_positions)
plt.title('Position of Correct Fertilizer in Top 3 Predictions')
plt.xlabel('Position (0 = Not in Top3, 1 = First, 2 = Second, 3 = Third)')
plt.ylabel('Count')
plt.show()




# Visualization 4: Feature Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = train_df[feature_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.show()


# Submission
test_probs = model.predict_proba(test_df[feature_cols])
test_pred_labels = get_top_n_predictions(test_probs, le_target, n=3)
submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': [' '.join(pred) for pred in test_pred_labels]
})
submission.to_csv('submission.csv', index=False)
print(submission.head())

