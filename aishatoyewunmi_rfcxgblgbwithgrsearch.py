import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load data
train = pd.read_csv ("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv ("/kaggle/input/playground-series-s5e7/test.csv")

print (train.info())            # Overview of column names, data types, and non-null counts
print (train.describe())        # Basic statistical summary (only for numeric columns)
print (train.isnull().sum())    # Check for missing values
print (train['Personality'].value_counts())    # How many Introverts vs Extroverts
print (train['Personality'].value_counts(normalize=True) * 100) # Percentage breakdown


# VISUALIZE THE TARGET

sns.countplot (x= 'Personality', data=train)
plt.title('Introvert vs Extrovert Count')
plt.show()


# Stage Fear vs Personality

# Frequency of Yes/No
print(train['Stage_fear'].value_counts())
print(train['Drained_after_socializing'].value_counts())

# Cross-tab with target
pd.crosstab(train['Stage_fear'], train['Personality'], normalize='index')

sns.countplot(x='Stage_fear', hue='Personality', data=train, palette= 'Set2')
plt.title('Stage Fear vs Personality')
plt.show()


# Cross-tab with target
pd.crosstab(train['Drained_after_socializing'], train['Personality'], normalize='index')

sns.countplot(x='Drained_after_socializing', hue='Personality', data=train, palette= 'Set2')
plt.title('Drained_after_socializing vs Personality')
plt.show()


# Distribution
sns.kdeplot(data=train, x= 'Time_spent_Alone', hue= 'Personality', fill= True, alpha=0.6, linewidth=1.5)
plt.title('Time Spent Alone Distribution')
plt.show()

# Boxplot by Personality
sns.boxplot(x='Personality', y='Time_spent_Alone', hue='Personality', data=train, color= 'green', palette= 'Set2')
plt.title('Time Spent Alone by Personality Type')
plt.show()


# Distribution
sns.kdeplot(data=train, x= 'Social_event_attendance',  hue= 'Personality', fill= True, alpha=0.6, linewidth=1.5)
plt.title('Social_event_attendance Distribution')
plt.show()

# Boxplot by Personality
sns.boxplot(x='Personality', y='Social_event_attendance', hue='Personality', data=train, color= 'green', palette= 'Set2')
plt.title('Social_event_attendance by Personality Type')
plt.show()


# Distribution
sns.kdeplot(data=train, x= 'Friends_circle_size',  hue= 'Personality', fill= True, alpha=0.6, linewidth=1.5)
plt.title('Friends_circle_size Distribution')
plt.show()

# Boxplot by Personality
sns.boxplot(x='Personality', y='Friends_circle_size', hue='Personality', data=train, color= 'green', palette= 'Set2')
plt.title('Friends_circle_size by Personality Type')
plt.show()


# Distribution
sns.kdeplot(data=train, x= 'Post_frequency',  hue= 'Personality', fill= True, alpha=0.6, linewidth=1.5)
plt.title('Post_frequency Distribution')
plt.show()

# Boxplot by Personality
sns.boxplot(x='Personality', y='Post_frequency', hue='Personality', data=train, color= 'green', palette= 'Set2')
plt.title('Post_frequency by Personality Type')
plt.show()


# Temporarily encode Personality to numeric for correlation
train['Personality_encoded'] = train['Personality'].map({'Introvert': 1, 'Extrovert': 0})

# Compute correlation
corr = train.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

print(train.isnull().sum())


# Handle Missing Values

train['Time_spent_Alone'] = train['Time_spent_Alone'].fillna(train['Time_spent_Alone'].median())
train['Post_frequency'] = train['Post_frequency'].fillna(train['Post_frequency'].median())
train['Friends_circle_size'] = train['Friends_circle_size'].fillna(train['Friends_circle_size'].median())
train['Social_event_attendance'] = train['Social_event_attendance'].fillna(train['Social_event_attendance'].median())
train['Going_outside'] = train['Going_outside'].fillna(train['Going_outside'].median())


train['Stage_fear'] = train['Stage_fear'].fillna(train['Stage_fear'].mode()[0])
train['Drained_after_socializing'] = train['Drained_after_socializing'].fillna(train['Drained_after_socializing'].mode()[0])


train['Stage_fear'] = train['Stage_fear'].map({'Yes': 1, 'No': 0})
train['Drained_after_socializing'] = train['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
train['Personality'] = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})

print(train.dtypes)


print(train.dtypes)
print(train.head())
print(train.isnull().sum())
print(train.describe())


test['Time_spent_Alone'] = test['Time_spent_Alone'].fillna(test['Time_spent_Alone'].median())
test['Post_frequency'] = test['Post_frequency'].fillna(test['Post_frequency'].median())
test['Friends_circle_size'] = test['Friends_circle_size'].fillna(test['Friends_circle_size'].median())
test['Social_event_attendance'] = test['Social_event_attendance'].fillna(test['Social_event_attendance'].median())
test['Going_outside'] = test['Going_outside'].fillna(test['Going_outside'].median())


test['Stage_fear'] = test['Stage_fear'].fillna(test['Stage_fear'].mode()[0])
test['Drained_after_socializing'] = test['Drained_after_socializing'].fillna(test['Drained_after_socializing'].mode()[0])


test['Stage_fear'] = test['Stage_fear'].map({'Yes': 1, 'No': 0})
test['Drained_after_socializing'] = test['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

test.dtypes


test.isnull().sum()

print(test.head())
print(test.isnull().sum())
print(test.describe())


# duplicate rows
print(f"Duplicate rows in training: {train.duplicated().sum()}")

# Drop duplicates (keep first occurrence)
train_clean = train.drop_duplicates()

# Verify
print(f"Original shape: {train.shape}, Cleaned shape: {train_clean.shape}")
print(f"Duplicates remaining: {train_clean.duplicated().sum()}")



# duplicate rows
print(f"Duplicate rows in testing: {test.duplicated().sum()}")

# Drop duplicates (keep first occurrence)
test_clean = test.drop_duplicates()

# Verify
print(f"Original shape: {test.shape}, Cleaned shape: {test_clean.shape}")
print(f"Duplicates remaining: {test_clean.duplicated().sum()}")


# Drop 'id' and keep others
X = train.drop(['id', 'Personality_encoded', 'Personality'], axis=1)  # Features
y = train['Personality']                      # Target


X


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size= 0.2, random_state= 42)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Evaluate on original (imbalanced) test set
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names =['Introvert', 'Extrovert']))


from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
print("Average CV Accuracy:", scores.mean())


import pandas as pd

# Generate predictions (replace with your model)

predictions = rf_model.predict(test.drop("id", axis=1))  # Predict

# Define mapping
label_map = {0: "Extrovert", 1: "Introvert"}

# Convert predictions
submission = pd.DataFrame({
    "id": test["id"],
    "Personality": pd.Series(predictions).map(label_map)
})


# Save to CSV
submission.to_csv("submission.csv", index=False)


pip install xgboost



import xgboost as xgb
print(xgb.__version__)  # Should print the version (e.g., '2.0.3')


from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Calculate class weight ratio (Introvert=0, Extrovert=1)
# scale_pos_weight = number of Introverts / number of Extroverts
scale_pos_weight = 2740 / 965  # ≈ 2.84 (use your exact counts)

# Create and train the model
xg_model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
xg_model.fit(X_train, y_train)

# Make predictions and print report - USE xg_model (your trained model), not xgb!
y_pred = xg_model.predict(X_test)
print("XGBoost:\n", classification_report(y_test, y_pred, target_names=["Introvert","Extrovert"]))



# Generate predictions (replace with your model)

predictions = xg_model.predict(test.drop("id", axis=1))  # Predict

# Define mapping
label_map = {0: "Extrovert", 1: "Introvert"}

# Convert predictions
xgsubmission = pd.DataFrame({
    "id": test["id"],
    "Personality": pd.Series(predictions).map(label_map)
})


# Save to CSV
xgsubmission.to_csv("xgsubmission.csv", index=False)


pip install lightgbm


import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score

# Calculate class weight ratio (Introvert=0, Extrovert=1)
# For LightGBM, use class_weight parameter
class_weight = {0: 1, 1: 965/2740}  # Adjust if your labels are reversed

lgb_model = lgb.LGBMClassifier(
    class_weight=class_weight,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)


# Define parameter grid
param_grid = {
    'num_leaves': [31, 63],          # Control tree complexity
    'learning_rate': [0.01, 0.05],   # Step size shrinkage
    'n_estimators': [100, 200],      # Number of boosting rounds
    'min_child_samples': [20, 50],   # Minimal samples per leaf
    'reg_alpha': [0, 0.1],           # L1 regularization
    'reg_lambda': [0, 0.1]           # L2 regularization
}

# Initialize GridSearch
grid_search = GridSearchCV(
    estimator=lgb_model,
    param_grid=param_grid,
    scoring='f1_weighted',           # Optimize for F1-score
    cv=5,                            # 5-fold cross-validation
    verbose=2                        # Print progress
)

# Run grid search
grid_search.fit(X_train, y_train)


# Get best parameters
print("Best Parameters:", grid_search.best_params_)

# Predict with best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred, target_names=["Introvert","Extrovert"]))

