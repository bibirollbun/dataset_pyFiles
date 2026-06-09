import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, RobustScaler
%matplotlib inline

import warnings 
warnings.filterwarnings('ignore')


#reading data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
org_train = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

#droping id column
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

# Combine both datasets
train = pd.concat([train, org_train], ignore_index=True)

# Returns a boolean Series: True = duplicate row (after first occurrence)
duplicates = train.duplicated()

# Count total duplicate rows
duplicate_count = duplicates.sum()
print(f"Number of duplicate rows being dropped: {duplicate_count}")

# Remove duplicates (keep first occurrence)
train = train.drop_duplicates(keep='first')


train.head()


train.describe()


train.info()


train.nunique()


mapping = {'Yes': 1, 'No': 0}

Drained_after_socializing_Train = train['Drained_after_socializing'].copy()
Drained_after_socializing_Train = Drained_after_socializing_Train.map(mapping)
Drained_after_socializing_Test = test['Drained_after_socializing'].copy()
Drained_after_socializing_Test = Drained_after_socializing_Test.map(mapping)

Stage_fear_Train = train['Stage_fear'].copy()
Stage_fear_Train = Stage_fear_Train.map(mapping)
Stage_fear_Test = test['Stage_fear'].copy()
Stage_fear_Test = Stage_fear_Test.map(mapping)


# Train features 
train['Extrovert_Score'] = train['Social_event_attendance'] + train['Going_outside'] + train['Friends_circle_size'] + train['Post_frequency']
train['Introvert_Score'] = train['Time_spent_Alone'] + (train['Time_spent_Alone'] * Drained_after_socializing_Train)
train['Social_Score'] = train['Social_event_attendance'] + (train['Social_event_attendance'] * Stage_fear_Train)
train['Time_spent_Alone_Squared'] = train['Time_spent_Alone'] * train['Time_spent_Alone']
train['Social_event_attendance_Squared'] = train['Social_event_attendance'] * train['Social_event_attendance']
train['Going_outside_Squared'] = train['Going_outside'] * train['Going_outside']

# Test features
test['Extrovert_Score'] = test['Social_event_attendance'] + test['Going_outside'] + test['Friends_circle_size'] + test['Post_frequency']
test['Introvert_Score'] = test['Time_spent_Alone'] + (test['Time_spent_Alone'] * Drained_after_socializing_Test)
test['Social_Score'] = test['Social_event_attendance'] + (test['Social_event_attendance'] * Stage_fear_Test)
test['Time_spent_Alone_Squared'] = test['Time_spent_Alone'] * test['Time_spent_Alone']
test['Social_event_attendance_Squared'] = test['Social_event_attendance'] * test['Social_event_attendance']
test['Going_outside_Squared'] = test['Going_outside'] * test['Going_outside']


# Numerical features
train_num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
            'Friends_circle_size', 'Post_frequency', 'Personality', 'Extrovert_Score','Introvert_Score','Social_Score']

test_num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
            'Friends_circle_size', 'Post_frequency', 'Extrovert_Score','Introvert_Score','Social_Score']


# Initialize the scaler
scaler = RobustScaler()

# Fit on the integer columns of training data
train[test_num_cols] = scaler.fit_transform(train[test_num_cols])

# Transform test data using the SAME scaler
test[test_num_cols] = scaler.transform(test[test_num_cols])


# Binary features
bin_cols = ['Stage_fear', 'Drained_after_socializing']

# enc = OrdinalEncoder()
# train[bin_cols] = enc.fit_transform(train[bin_cols])
# test[bin_cols] = enc.transform(test[bin_cols])

# Replace NaNs with 'Missing' and create missing indicator columns
for col in bin_cols:
    # Create missingness indicator
    train[f'{col}_missing'] = train[col].isna().astype(int)
    test[f'{col}_missing'] = test[col].isna().astype(int)

    # Fill NA with 'Missing'
    train[col] = train[col].fillna('Missing')
    test[col] = test[col].fillna('Missing')

# Now apply One-Hot Encoding to the original columns
train = pd.get_dummies(train, columns=bin_cols, drop_first=False)
test = pd.get_dummies(test, columns=bin_cols, drop_first=False)

# Convert all boolean columns to integers
bool_cols = train.select_dtypes(include='bool').columns
train[bool_cols] = train[bool_cols].astype(int)

# Do the same for test if needed
bool_cols_test = test.select_dtypes(include='bool').columns
test[bool_cols_test] = test[bool_cols_test].astype(int)

#Drop extra missing column
train = train.drop('Stage_fear_missing', axis=1)  # Drop
train = train.drop('Drained_after_socializing_missing', axis=1)  # Drop
test = test.drop('Stage_fear_missing', axis=1)  # Drop
test = test.drop('Drained_after_socializing_missing', axis=1)  # Drop

# # Binary feature columns can be updated. We are using the ones with Yes
# This will not effect our visualization. As the pattern of 1 and 0 will be
#same in Yes columns as we had before One Hot Encoding
bin_cols = ['Stage_fear_Yes', 'Drained_after_socializing_Yes']


# Encode target
le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality']) 


train.head()


train_cols = train.columns
test_cols = test.columns

# Histogram for train column
train[train_cols].hist(figsize=(12, 8), bins=25)
plt.suptitle("Histogram of Train Features")
plt.tight_layout()
plt.show()


# Histogram for test column
test[test_cols].hist(figsize=(12, 8), bins=25)
plt.suptitle("Histogram of Test Features")
plt.tight_layout()
plt.show()


# Get value counts for both datasets
train_counts = train['Time_spent_Alone'].value_counts()
test_counts = test['Time_spent_Alone'].value_counts()

# Calculate percentages
train_percentages = (train_counts / train_counts.sum()) * 100
test_percentages = (test_counts / test_counts.sum()) * 100

# Create combined labels with count and percentage
train_labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(train_counts.index, train_counts.values, train_percentages)]
test_labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(test_counts.index, test_counts.values, test_percentages)]

# Set up subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 8))


# Train pie chart
axes[0].pie(train_counts, labels=train_labels, startangle=160, counterclock=False,
            wedgeprops={'edgecolor': 'black'})
axes[0].set_title('Distribution of Time_spent_Alone - Train Data')
axes[0].axis('equal')

# Test pie chart
axes[1].pie(test_counts, labels=test_labels, startangle=160, counterclock=False,
            wedgeprops={'edgecolor': 'black'})
axes[1].set_title('Distribution of Time_spent_Alone - Test Data')
axes[1].axis('equal')

plt.tight_layout()
plt.show()


# Get value counts for both datasets
train_counts = train['Stage_fear_Yes'].value_counts()
test_counts = test['Stage_fear_Yes'].value_counts()

# Calculate percentages
train_percentages = (train_counts / train_counts.sum()) * 100
test_percentages = (test_counts / test_counts.sum()) * 100

# Create combined labels with count and percentage
train_labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(train_counts.index, train_counts.values, train_percentages)]
test_labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(test_counts.index, test_counts.values, test_percentages)]

# Set up subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 8))


# Train pie chart
axes[0].pie(train_counts, labels=train_labels, startangle=160, counterclock=False,
            wedgeprops={'edgecolor': 'black'})
axes[0].set_title('Distribution of Stage_fear - Train Data')
axes[0].axis('equal')

# Test pie chart
axes[1].pie(test_counts, labels=test_labels, startangle=160, counterclock=False,
            wedgeprops={'edgecolor': 'black'})
axes[1].set_title('Distribution of Stage_fear - Test Data')
axes[1].axis('equal')

plt.tight_layout()
plt.show()


# Get value counts for both datasets
train_counts = train['Social_event_attendance'].value_counts()
test_counts = test['Social_event_attendance'].value_counts()

# Calculate percentages
train_percentages = (train_counts / train_counts.sum()) * 100
test_percentages = (test_counts / test_counts.sum()) * 100

# Create combined labels with count and percentage
train_labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(train_counts.index, train_counts.values, train_percentages)]
test_labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(test_counts.index, test_counts.values, test_percentages)]

# Set up subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 8))


# Train pie chart
axes[0].pie(train_counts, labels=train_labels, startangle=160, counterclock=False,
            wedgeprops={'edgecolor': 'black'})
axes[0].set_title('Distribution of Social_event_attendance - Train Data')
axes[0].axis('equal')

# Test pie chart
axes[1].pie(test_counts, labels=test_labels, startangle=160, counterclock=False,
            wedgeprops={'edgecolor': 'black'})
axes[1].set_title('Distribution of Social_event_attendance - Test Data')
axes[1].axis('equal')

plt.tight_layout()
plt.show()


# Get value counts for both datasets
train_counts = train['Going_outside'].value_counts()
test_counts = test['Going_outside'].value_counts()

# Calculate percentages
train_percentages = (train_counts / train_counts.sum()) * 100
test_percentages = (test_counts / test_counts.sum()) * 100

# Create combined labels with count and percentage
train_labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(train_counts.index, train_counts.values, train_percentages)]
test_labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(test_counts.index, test_counts.values, test_percentages)]

# Set up subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 8))


# Train pie chart
axes[0].pie(train_counts, labels=train_labels, startangle=160, counterclock=False,
            wedgeprops={'edgecolor': 'black'})
axes[0].set_title('Distribution of Going_outside - Train Data')
axes[0].axis('equal')

# Test pie chart
axes[1].pie(test_counts, labels=test_labels, startangle=160, counterclock=False,
            wedgeprops={'edgecolor': 'black'})
axes[1].set_title('Distribution of Going_outside - Test Data')
axes[1].axis('equal')

plt.tight_layout()
plt.show()


# Get value counts for both datasets
train_counts = train['Drained_after_socializing_Yes'].value_counts()
test_counts = test['Drained_after_socializing_Yes'].value_counts()

# Calculate percentages
train_percentages = (train_counts / train_counts.sum()) * 100
test_percentages = (test_counts / test_counts.sum()) * 100

# Create combined labels with count and percentage
train_labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(train_counts.index, train_counts.values, train_percentages)]
test_labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(test_counts.index, test_counts.values, test_percentages)]

# Set up subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 8))


# Train pie chart
axes[0].pie(train_counts, labels=train_labels, startangle=160, counterclock=False,
            wedgeprops={'edgecolor': 'black'})
axes[0].set_title('Distribution of Drained_after_socializing - Train Data')
axes[0].axis('equal')

# Test pie chart
axes[1].pie(test_counts, labels=test_labels, startangle=160, counterclock=False,
            wedgeprops={'edgecolor': 'black'})
axes[1].set_title('Distribution of Drained_after_socializing - Test Data')
axes[1].axis('equal')

plt.tight_layout()
plt.show()


# Get value counts for both datasets
train_counts = train['Friends_circle_size'].value_counts()
test_counts = test['Friends_circle_size'].value_counts()

# Calculate percentages
train_percentages = (train_counts / train_counts.sum()) * 100
test_percentages = (test_counts / test_counts.sum()) * 100

# Create combined labels with count and percentage
train_labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(train_counts.index, train_counts.values, train_percentages)]
test_labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(test_counts.index, test_counts.values, test_percentages)]

# Set up subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 8))


# Train pie chart
axes[0].pie(train_counts, labels=train_labels, startangle=160, counterclock=False,
            wedgeprops={'edgecolor': 'black'})
axes[0].set_title('Distribution of Friends_circle_size - Train Data')
axes[0].axis('equal')

# Test pie chart
axes[1].pie(test_counts, labels=test_labels, startangle=160, counterclock=False,
            wedgeprops={'edgecolor': 'black'})
axes[1].set_title('Distribution of Friends_circle_size - Test Data')
axes[1].axis('equal')

plt.tight_layout()
plt.show()


# Get value counts for both datasets
train_counts = train['Post_frequency'].value_counts()
test_counts = test['Post_frequency'].value_counts()

# Calculate percentages
train_percentages = (train_counts / train_counts.sum()) * 100
test_percentages = (test_counts / test_counts.sum()) * 100

# Create combined labels with count and percentage
train_labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(train_counts.index, train_counts.values, train_percentages)]
test_labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(test_counts.index, test_counts.values, test_percentages)]

# Set up subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 8))


# Train pie chart
axes[0].pie(train_counts, labels=train_labels, startangle=160, counterclock=False,
            wedgeprops={'edgecolor': 'black'})
axes[0].set_title('Distribution of Post_frequency - Train Data')
axes[0].axis('equal')

# Test pie chart
axes[1].pie(test_counts, labels=test_labels, startangle=160, counterclock=False,
            wedgeprops={'edgecolor': 'black'})
axes[1].set_title('Distribution of Post_frequency - Test Data')
axes[1].axis('equal')

plt.tight_layout()
plt.show()


Personality_counts = train['Personality'].value_counts()

# Calculate percentages
percentages = (Personality_counts / Personality_counts.sum()) * 100

# Create labels with both value counts and percentages
labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(Personality_counts.index, Personality_counts.values, percentages)]

# Create the pie chart
plt.figure(figsize=(6, 4)) # Set figure size for better readability
plt.pie(Personality_counts, labels=labels, startangle=160, counterclock=False, wedgeprops={'edgecolor': 'black'})
plt.title('Distribution of Personality - Train Data')
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()



corr_train = train.corr()
corr_test = test.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_train, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
plt.title("Heatmap - Train Data")
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(corr_test, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
plt.title("Heatmap - Test Data")
plt.show()


# Histograms for numerical features
for col in train_num_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(train[col], bins=10, kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Countplots for binary features
for col in bin_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=train, x=col, palette='Set2')
    plt.title(f'Count of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()


# Histograms for numerical features
for col in test_num_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(test[col], bins=10, kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Countplots for binary features
for col in bin_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=test, x=col, palette='Set2')
    plt.title(f'Count of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()


# Boxplots for numerical features vs Personality
for col in train_num_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=train, x='Personality', y=col, palette='Set3')
    plt.title(f'{col} vs Personality')
    plt.xlabel('Personality (0=Introvert, 1=Extrovert)')
    plt.ylabel(col)
    plt.show()

# Barplots for binary features vs Personality
for col in bin_cols:
    plt.figure(figsize=(6, 4))
    sns.barplot(data=train, x='Personality', y=col, errorbar='sd', palette='Set1')
    plt.title(f'{col} vs Personality')
    plt.xlabel('Personality')
    plt.ylabel(f'Proportion of {col}')
    plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(data=train, x='Personality', palette='Set2')
plt.title('Distribution of Personality Types')
plt.xlabel('Personality (0=Introvert, 1=Extrovert)')
plt.ylabel('Count')
plt.show()


# Pairplot with hue='Personality'
sns.pairplot(data=train, vars=train_num_cols, hue='Personality', palette='viridis', diag_kind='hist', markers=["o", "s"])
plt.suptitle('Pairplot of Numerical Features by Personality', y=1.02)
plt.show()


# train = train.drop('Friends_circle_size', axis=1)  # Drop
# train = train.drop('Post_frequency', axis=1)  # Drop

# test = test.drop('Friends_circle_size', axis=1)  # Drop
# test = test.drop('Post_frequency', axis=1)  # Drop

# Define features and target
X = train.drop('Personality', axis=1)  # Features
y = train['Personality']               # Target

# Test features
X_test = test.copy()


# XGB model parameters
xgb_params = {
    'objective': 'binary:logistic',  # Binary classification
    'eval_metric': 'logloss',        # Evaluation metric
    'max_depth': 4,                  # Approximate num_leaves=31 (2^4 â‰ˆ 16 leaves)
    'colsample_bytree': 0.8,         # Subsample features per iteration
    'subsample': 0.8,                # Subsample data per iteration
    'verbosity': 0,                  # Suppress XGBoost warnings
    'eta': 0.01,                     # Learning Rate
    'seed': 42                       # Random state
}

lgb_params = {
    'objective': 'binary',  # Change to 'regression' if needed
    'metric': 'binary_logloss',
    'max_depth': 4,
    'num_leaves': 16,       # Approximate num_leaves for max_depth=4
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'verbosity': -1,
    'random_state': 42,
    'force_col_wise': True  # Faster on small datasets
}


# Define number of splits and repetitions for RepeatedStratifiedKFold
SPLITS = 5      # Number of folds per repeat
REPEATS = 2     # How many times to shuffle and run stratified k-fold

# Initialize the cross-validator
# This ensures class distribution is preserved across folds and repeats
skf = RepeatedStratifiedKFold(n_splits=SPLITS, n_repeats=REPEATS, random_state=42)

# Preallocate arrays for:
# - Out-of-Fold predictions (used to evaluate model performance)
# - Final Test predictions (to be averaged over all folds and repeats)
# OOF and Test predictions for each model
oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
y_pred_xgb = np.zeros(len(X_test))
y_pred_lgb = np.zeros(len(X_test))

# Start cross-validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    ### Train XGBoost ###
    dtrain_xgb = xgb.DMatrix(X_train, label=y_train)
    dval_xgb = xgb.DMatrix(X_val, label=y_val)
    dtest_xgb = xgb.DMatrix(X_test)

    model_xgb = xgb.train(
        xgb_params,
        dtrain_xgb,
        num_boost_round=1000,
        evals=[(dval_xgb, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    pred_xgb_val = model_xgb.predict(dval_xgb)
    pred_xgb_test = model_xgb.predict(dtest_xgb)

    ### Train LightGBM ###
    model_lgb = LGBMClassifier(**lgb_params).fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]
    )

    pred_lgb_val = model_lgb.predict_proba(X_val)[:, 1]  # For binary classification
    pred_lgb_test = model_lgb.predict_proba(X_test)[:, 1]

    ### Aggregate Predictions ###
    oof_xgb[val_idx] += pred_xgb_val / REPEATS
    oof_lgb[val_idx] += pred_lgb_val / REPEATS

    y_pred_xgb += pred_xgb_test / (REPEATS * SPLITS)
    y_pred_lgb += pred_lgb_test / (REPEATS * SPLITS)

### Final OOF and Test predictions from both models
oof_preds = (oof_xgb + oof_lgb) / 2
y_pred = (y_pred_xgb + y_pred_lgb) / 2


# Predict on validation set
y_pred_class = (oof_preds > 0.5).astype(int)

# Accuracy
print("Accuracy:", accuracy_score(y, y_pred_class))

#CV Logloss
print("CV Logloss:", log_loss(y, y_pred_class))

# Compute F1 score
f1 = f1_score(y, y_pred_class)
print("F1 Score:", f1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y, y_pred_class))

# Confusion Matrix
cm = confusion_matrix(y, y_pred_class)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(model_xgb, max_num_features=7, ax=ax)
ax.set_title('Feature Importance')
plt.tight_layout()
plt.show()


# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y, oof_preds)  # Use predicted probabilities
roc_auc = roc_auc_score(y, oof_preds)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random guessing')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# Find best threshold Youdenâ€™s J statistic
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Youdenâ€™s J statistic Threshold:", optimal_threshold)


# Predict probabilities (for submission)
test_preds = (y_pred > 0.66).astype(int)
test_preds_class = le.inverse_transform(test_preds)

# Prepare submission.csv
sub['Personality'] = test_preds_class

# Save to CSV
sub.to_csv('submission.csv', index=False)
sub.head()


df1 = pd.read_csv('submission.csv')
df2 = pd.read_csv('/kaggle/input/ps-s5e7-don-t-look-at-me/submission.csv')
df3 = pd.read_csv('/kaggle/input/top-4-solution-0-976518-easy-is-all-you-need/submission.csv')
df4 = pd.read_csv('/kaggle/input/0-976518-random-forest-boost/submission.csv')


all_df=df1.merge(df2, on='id').merge(df3, on='id').merge(df4, on='id').query('Personality_x != Personality_y or Personality_x != Personality or Personality_x != personality')
print("Different = ",len(all_df))
all_df


(df1.merge(df2, on='id').query('Personality_x != Personality_y'))


(df1.merge(df3, on='id').query('Personality_x != Personality_y'))


(df1.merge(df4, on='id').query('Personality != personality'))

