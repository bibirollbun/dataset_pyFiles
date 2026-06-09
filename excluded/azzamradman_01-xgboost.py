import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
train = train.drop('id', axis=1)
train


test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
test = test.drop('id', axis=1)
test





# Label encode categorical features
cat_features = ['gender', 'marital_status', 'education_level', 'employment_status',
                'loan_purpose', 'grade_subgrade']
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

# Separate features and target
X = train.drop(columns='loan_paid_back')
y = train['loan_paid_back']

n_splits = 5

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store ROC AUC scores
roc_auc_scores = []

# Perform cross-validation
for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set parameters for GPU training and ROC AUC optimization
    
    xgb_params = {
        'objective': 'binary:logistic',       # Binary classification
        'eval_metric': 'auc',                 # Optimize for ROC AUC
        'tree_method': 'gpu_hist',            # Use GPU acceleration
        'predictor': 'gpu_predictor',         # GPU-based prediction
        'max_depth': 6,                       # Controls tree depth
        'learning_rate': 0.1,                 # Step size shrinkage
        'n_estimators': 10000,                  # Number of boosting rounds
        'subsample': 0.8,                     # Row sampling per tree
        'colsample_bytree': 0.8,              # Feature sampling per tree
        'gamma': 0,                           # Minimum loss reduction
        'scale_pos_weight': 1,                # Class imbalance handling
        'min_child_weight': 1,                # Minimum sum of instance weight
    }


    # Train model
    evals = [(dtrain, 'train'), (dtest, 'valid')]
    model = xgb.train(xgb_params, dtrain, num_boost_round=1000, 
                      evals=evals, early_stopping_rounds=100)


    # Predict probabilities
    y_pred_proba = model.predict(dtest)

    y_pred_proba_test =  model.predict(xgb.DMatrix(test))
    
    if idx == 0:
        place_holder = y_pred_proba_test / n_splits
    else:
        place_holder += y_pred_proba_test / n_splits
    

    # Calculate ROC AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    roc_auc_scores.append(auc)

# Print average ROC AUC score
print(f"Average ROC AUC score across 5 Stratified K-Fold CV: {np.mean(roc_auc_scores):.4f}")


sample = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
sample


sample.iloc[:, -1] = place_holder


sample.to_csv('submission.csv', index=False)







