import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt
import seaborn as sns


PTH_TRAIN = '/kaggle/input/d/nirmalsankalana/march-machine-learning-mania-2025/train.csv'


df_train = pd.read_csv(PTH_TRAIN)
df_train.tail()


# Load your dataset
df = df_train.copy()  # Assuming `dataset` contains the training data

# Drop ID column if present
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

if 'Team1' in df.columns:
    df = df.drop(columns=['Team1'])

if 'Team2' in df.columns:
    df = df.drop(columns=['Team2'])

if 'Season' in df.columns:
    df = df.drop(columns=['Season'])


label_encoders = {}
for col in ['Type', 'Gender']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders if needed later


correlation_matrix = df.corr()
# Extract correlation values for each feature with the target variable 'Team1Winning'
target_correlation = correlation_matrix['Team1Winning'].sort_values(ascending=False)
print(target_correlation)


plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix, 
    annot=True,  # Show correlation values
    fmt=".2f",   # Limit decimal places
    cmap="coolwarm", 
    linewidths=0.5, 
    vmin=-1, vmax=1  # Set range for color scale
)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.show()


# high_correlation_threshold = 0.85  

# # Identify features that are highly correlated
# correlated_features = set()
# for i in range(len(correlation_matrix.columns)):
#     for j in range(i):
#         if abs(correlation_matrix.iloc[i, j]) > high_correlation_threshold:
#             feature_name = correlation_matrix.columns[i]
#             correlated_features.add(feature_name)

# # Remove highly correlated features
# filtered_df = df.drop(columns=correlated_features)

# # Print removed features
# print(f"Removed {len(correlated_features)} highly correlated features: {correlated_features}")

# # Print remaining features
# print(f"Remaining features: {filtered_df.columns.tolist()}")


# Set a threshold value (e.g., 0.1 or 0.2) to filter out low correlated features
threshold = 0.1

selected_features = target_correlation[abs(target_correlation) > threshold].index.tolist()
selected_features.remove('Team1Winning')
X_filtered = df[selected_features]
print(f"Selected features with correlation above {threshold}:")
print(selected_features)


scaler = MinMaxScaler()
cols_to_normalize = [
    'team1ScoreWithTeam2PrevMatch', 'team2ScoreWithTeam1PrevMatch',
    'prevMatchScoreDiff',
    'Team1WonRegMatchesPrev', 'Team2WonRegMatchesPrev', 'Team1WonTourMatchesPrev', 'Team2WonTourMatchesPrev',	
    'Team1TotalScoresPrevReg', 'Team2TotalScoresPrevReg', 'Team1TotalScoresPrevTour', 'Team2TotalScoresPrevTour',
    'PrevRegDiff', 'PrevTourDiff', 'PrevRegScoreDiff', 'PrevTourScoreDiff'
]

df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])


X = df.drop(columns=['Team1Winning'])  # Features
X = X[selected_features]
y = df['Team1Winning']  # Target

# Convert to numpy arrays
X = X.values
y = y.values

# Initialize K-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Store log loss scores
log_losses = []


param_dist = {
    'n_estimators': [25, 50, 100],  # Number of boosting rounds
    'learning_rate': [0.05, 0.1, 0.5],  # Shrinks feature weights
    'max_depth': [1, 2, 4],  # Tree depth
    'subsample': [0.8, 0.9, 0.95],  # Fraction of samples per tree
    'colsample_bytree': [0.6, 0.7, 0.8]  # Fraction of features per tree
}

# Initialize XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# Set up RandomizedSearchCV with 5-fold cross-validation
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,  # Correct argument
    n_iter=200,  # Reduce iterations to 20 for speed
    scoring='neg_log_loss',  # Optimize for log loss
    cv=5,  # 5-Fold Cross-Validation
    verbose=1,
    n_jobs=-1,  # Use all CPU cores
    random_state=42
)

# Fit RandomizedSearchCV on training data
random_search.fit(X, y)

# Get the best hyperparameters
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)


log_losses = []
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Train and Evaluate Model using K-Fold Cross Validation
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Train XGBoost Model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        **best_params,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predict probabilities
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Calculate Log Loss
    loss = log_loss(y_val, y_pred_prob)
    log_losses.append(loss)

    # Calculate Accuracy, Precision, Recall, and F1-Score
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    # Print metrics for the fold
    print(f'Fold Log Loss: {loss:.4f}')
    print(f'Fold Accuracy: {accuracy:.4f}')
    print(f'Fold Precision: {precision:.4f}')
    print(f'Fold Recall: {recall:.4f}')
    print(f'Fold F1 Score: {f1:.4f}')
    print('-' * 50)

# Print Average metrics
print(f'\nAverage Log Loss: {np.mean(log_losses):.4f}')
print(f'Average Accuracy: {np.mean(accuracies):.4f}')
print(f'Average Precision: {np.mean(precisions):.4f}')
print(f'Average Recall: {np.mean(recalls):.4f}')
print(f'Average F1 Score: {np.mean(f1_scores):.4f}')


test_df = pd.read_csv('/kaggle/input/d/nirmalsankalana/march-machine-learning-mania-2025/test.csv')
test_df.head()


# Load your dataset
df = test_df.copy()  # Assuming `dataset` contains the training data

# Drop ID column if present
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# Encode categorical features
label_encoders = {}
for col in ['Type', 'Gender']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders if needed later


scaler = MinMaxScaler()
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])


df = df[selected_features]


df.head()


y_pred_prob_test = model.predict_proba(df)[:, 1]

# Add predictions to df_test
df['Pred'] = y_pred_prob_test

df.head()


df_merged = test_df.join(df[['Pred']], how='left')
df_merged = df_merged[['ID', 'Pred']]
df_merged.head()
df_merged.to_csv('xgb_1.csv', index=False)


df_submission = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')
df_submission.shape

