import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
import missingno as msno
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


import gc

# Force garbage collection
gc.collect()


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
test_ids = test['id']


train.isnull().sum()


# Visualize missing values
msno.matrix(train)


for i in train :
    print (i) 


# convert categorical variable to numerical labes
def preprocess(data):
    # Convert categorical features
    cat_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in cat_cols:
        if col in data.columns:
            data[col] = data[col].str.strip().str.lower().map({'yes': 1, 'no': 0})
    
    # Convert target if exists
    if 'Personality' in data.columns:
        data['Personality'] = data['Personality'].str.strip().str.lower().map({
            'introvert': 1, 'extrovert': 0
        })
    return data


train = preprocess(train)
test = preprocess(test)


print("Missing Values in Train:")
print(train.isnull().sum())

print("\nMissing Values in Test:")
print(test.isnull().sum())

msno.matrix(train)
plt.title('Missing Values in Training Data')
plt.show()


plt.figure(figsize=(14, 12))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title('Feature Correlation Matrix')
plt.show()


def create_features(df):
    # Create new features
    df['Social_Engagement'] = (df['Social_event_attendance'] + df['Post_frequency'])/ (df['Time_spent_Alone'].replace(0, 1))
    df['Recovery_Need'] = (df['Drained_after_socializing']) * df['Social_event_attendance']
    df['Alone_Outside_Ratio'] = df['Time_spent_Alone'] / (df['Going_outside'].replace(0, 1))
    df['Social_Recovery_Balance'] = df['Friends_circle_size'] / (df['Drained_after_socializing'] + 1)
    df['Solitude'] = df['Stage_fear']+ df['Drained_after_socializing'] * df['Time_spent_Alone']
    df['hidden_extrovert'] = df['Stage_fear']- (df['Time_spent_Alone'] +df['Drained_after_socializing'])

    # Contradiction indicator features
    df['Potential_Introvert_Contradiction'] = ((
        (df['Time_spent_Alone'] < 2) &
        (df['Stage_fear'] == 0) &
        (df['Social_event_attendance'] > 5) &
        (df['Drained_after_socializing'] == 0)
    ).astype(int))+6
    
    df['Potential_Extrovert_Contradiction'] = ((
        (df['Time_spent_Alone'] > 8) &
        (df['Stage_fear'] == 1) &
        (df['Social_event_attendance'] < 3) &
        (df['Going_outside'] < 2) &
        (df['Drained_after_socializing'] == 1)
    ).astype(int))+6
    
    return df


train = create_features(train)
test = create_features(test)


plt.figure(figsize=(14, 12))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title('Feature Correlation Matrix')
plt.show()


# separate target before imputation
y = train['Personality']
train.drop(['id', 'Personality'], axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

#Impute missing values
imputer = KNNImputer(n_neighbors=7)
train_imputed = imputer.fit_transform(train)
test_imputed = imputer.transform(test)

# Convert back to DataFrames
train = pd.DataFrame(train_imputed, columns=train.columns)
test = pd.DataFrame(test_imputed, columns=test.columns)


train['Personality'] = y  # Reattach target for correlation
plt.figure(figsize=(14, 12))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title('Feature Correlation Matrix')
plt.show()


train.drop('Personality', axis=1, inplace=True)  # Remove target for training


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
confidence_weights = np.zeros(len(train))

for train_idx, val_idx in kf.split(train, y):
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    lr = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs')
    lr.fit(X_train, y_train)
    
    # Get confidence for true labels
    val_probs = lr.predict_proba(X_val)
    confidence_weights[val_idx] = val_probs[np.arange(len(val_idx)), y_val.values]


iso = IsolationForest(contamination=0.03, random_state=42)
outliers = iso.fit_predict(train)
outlier_weights = np.where(outliers == -1, 0.5, 1.0)

# Combine weights
final_weights = confidence_weights * outlier_weights

# Visualize weight distribution
plt.figure(figsize=(10, 6))
sns.histplot(final_weights, bins=50)
plt.title('Sample Weight Distribution')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.show()


# Robust Scaling 
scaler = RobustScaler()
X_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)


# Cross-validation setup
N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
test_preds = []
oof_preds = np.zeros(len(train))
scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    sample_weights = final_weights[train_idx]
    
    # Handle class imbalance
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    model = LGBMClassifier(
        n_estimators=1700,
        learning_rate=0.03,
        scale_pos_weight=scale_pos_weight,
        subsample=0.7,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.2,
        random_state=fold,
        early_stopping_rounds=200,
        min_child_samples=20,
        #silent=False, # replacement for verbosity
        verbose=100,
        force_row_wise=True  # Prevent warning
    )
    
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        #early_stopping_rounds=100,
        #verbose=200
    )
    
    # Validation predictions
    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds
    
    # Track fold accuracy
    fold_acc = accuracy_score(y_val, val_preds)
    scores.append(fold_acc)
    print(f"\nFold {fold+1} | Accuracy: {fold_acc:.4f}")
    
    # Test predictions
    test_preds.append(model.predict_proba(test_scaled)[:, 1])  # Use probability for ensembling



print("\nOverall CV Results:")
print(f"Mean Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
print("\nClassification Report:")
print(classification_report(y, oof_preds))


# Feature importance
importance = pd.DataFrame({
    'Feature': train.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance.head(15))
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.show()


# Test Predictions & Submission 
# Ensemble predictions from all folds
test_prob = np.mean(test_preds, axis=0)
final_preds = (test_prob > 0.5).astype(int)  # Threshold at 0.5

# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'Personality': np.where(final_preds == 1, 'Introvert', 'Extrovert')
})

# Verify distribution
print("\nPrediction Distribution:")
print(submission['Personality'].value_counts(normalize=True))

# Save submission
submission.to_csv('submission.csv', index=False)
print("\nSubmission saved successfully!")




