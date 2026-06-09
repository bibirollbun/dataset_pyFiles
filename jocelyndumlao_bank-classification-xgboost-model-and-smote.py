import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
try:
    from imblearn.over_sampling import SMOTE  # for handling imbalanced data
except ModuleNotFoundError:
    print("imblearn not found. Please install it using: pip install imbalanced-learn")
    SMOTE = None  # Set SMOTE to None if not installed
import optuna # for hyperparameter optimisation



# Load data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


def preprocess_data(df, is_train=True):
    df = df.copy() # Avoid modifying original DataFrame

    # Impute missing values (using 'unknown' as missing category)
    for col in ['job', 'marital', 'education', 'contact', 'poutcome']:
      df[col] = df[col].fillna(df[col].mode()[0])

    # Feature Engineering
    df['balance_age_ratio'] = df['balance'] / (df['age'] + 1e-6) # Avoid division by zero
    df['campaign_duration_ratio'] = df['campaign'] / (df['duration'] + 1e-6)
    df['pdays_success'] = (df['pdays'] > 0).astype(int)

    # Convert 'default', 'housing', 'loan' to numerical (0,1)
    for col in ['default', 'housing', 'loan']:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # Age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 40, 60, 100], labels=['Young', 'Adult', 'Middle-Aged', 'Senior'])

    # Interaction between job and education
    df['job_education'] = df['job'].astype(str) + "_" + df['education'].astype(str)

    if is_train:
        y = df['y']
        X = df.drop(['id', 'y'], axis=1)   
        return X, y
    else:
        return df.drop('id', axis=1)

train_df_processed, y = preprocess_data(train_df)
test_df_processed = preprocess_data(test_df, is_train=False)


categorical_features = train_df_processed.select_dtypes(include='object').columns
numerical_features = train_df_processed.select_dtypes(include=np.number).columns

# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler()),  # Scale numerical features
    #('poly', PolynomialFeatures(degree=2, include_bias=False))  # Add polynomial features (optional)
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')), # Handle missing categorical values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])



def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 700),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'gamma': trial.suggest_float('gamma', 0, 0.8),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.4), # L1 Regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 0.4), # L2 Regularization
        'random_state': 42,
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 5) # Handle class imbalance
    }
    model = XGBClassifier(**xgb_params, use_label_encoder=False, eval_metric='logloss')

    # Preprocessor and SMOTE within cross-validation loop
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('smote', SMOTE(random_state=42)) if SMOTE is not None else ('passthrough', 'passthrough'),  # Conditional SMOTE
                               ('classifier', model)])

    return cross_val_score(pipeline, train_df_processed, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='roc_auc', n_jobs=-1).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25)

best_params = study.best_params
print("Best Optuna parameters:", best_params)

best_xgb = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)



# Create the final pipeline with the best XGBoost model and SMOTE

final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('smote', SMOTE(random_state=42)) if SMOTE is not None else ('passthrough', 'passthrough'), # Conditional SMOTE
                                 ('classifier', best_xgb)])


# Train the final pipeline on the entire training dataset
final_pipeline.fit(train_df_processed, y)



# Ensemble with VotingClassifier
# Simplified Logistic Regression and RandomForest models for the ensemble

# Define other models with hyperparameter setting
logreg = LogisticRegression(solver='liblinear', penalty='l1', C=0.07, random_state=42) # Adjusted C
rf = RandomForestClassifier(n_estimators=150, max_depth=9, min_samples_leaf=4, random_state=42) # Adjusted hyperparameters
gb = GradientBoostingClassifier(n_estimators=120, learning_rate=0.09, max_depth=4, random_state=42) # added Gb

# Create pipelines for other models
pipeline_logreg = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', logreg)])
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf)])
pipeline_gb = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', gb)])

# Train individual models
pipeline_logreg.fit(train_df_processed, y)
pipeline_rf.fit(train_df_processed, y)
pipeline_gb.fit(train_df_processed, y)



# Create VotingClassifier with optimized weights

X_train, X_val, y_train, y_val = train_test_split(train_df_processed, y, test_size=0.2, random_state=42, stratify=y)

# Get predicted probabilities on the validation set
proba_xgb = final_pipeline.predict_proba(X_val)[:, 1] # Use the optimized XGB pipeline
proba_lr = pipeline_logreg.predict_proba(X_val)[:, 1]
proba_rf = pipeline_rf.predict_proba(X_val)[:, 1]
proba_gb = pipeline_gb.predict_proba(X_val)[:, 1]


# Function to calculate the ROC AUC score for different weights
def evaluate_weights(weights):
    y_pred_proba = (weights[0] * proba_xgb +
                    weights[1] * proba_lr +
                    weights[2] * proba_rf +
                    weights[3] * proba_gb)
    return roc_auc_score(y_val, y_pred_proba)

# Perform a simple grid search for the weights
best_score = 0
best_weights = None
for w1 in np.arange(0, 1.1, 0.1):
    for w2 in np.arange(0, 1.1 - w1, 0.1):
        for w3 in np.arange(0, 1.1 - w1 - w2, 0.1):
            w4 = 1 - w1 - w2 - w3
            weights = [w1, w2, w3, w4]
            score = evaluate_weights(weights)
            if score > best_score:
                best_score = score
                best_weights = weights

print("Best Weights:", best_weights)
print("Best Validation ROC AUC:", best_score)



# Create VotingClassifier with optimized weights
voting_clf = VotingClassifier(estimators=[('xgb', final_pipeline), ('lr', pipeline_logreg), ('rf', pipeline_rf), ('gb', pipeline_gb)], voting='soft', weights=best_weights)
voting_clf.fit(train_df_processed, y) # Retrain on full training set



# a. Predict probabilities on the test set
y_pred_proba = voting_clf.predict_proba(test_df_processed)[:, 1]

# b. Create the submission DataFrame with id + predictions
submission = pd.DataFrame({
    'id': test_df['id'],     # keep original test ids
    'y':  y_pred_proba      # predicted probabilities
})


# c. Save the submission file
submission.to_csv('submission.csv', index=False)

# d. Display the head of the submission file
submission.head()


# Visualization (ROC Curve)

y_pred_proba_train = final_pipeline.predict_proba(train_df_processed)[:, 1]  # Using final_pipeline
fpr, tpr, thresholds = roc_curve(y, y_pred_proba_train)
roc_auc = roc_auc_score(y, y_pred_proba_train)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()




