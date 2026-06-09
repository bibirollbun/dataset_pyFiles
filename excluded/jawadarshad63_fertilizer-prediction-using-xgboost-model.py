# Importing pandas for data manipulation and analysis
import pandas as pd

# Importing numpy for numerical operations
import numpy as np

# Importing LabelEncoder to convert categorical labels into numeric format
from sklearn.preprocessing import LabelEncoder

# Importing train_test_split to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Importing XGBoost, a powerful gradient boosting framework
import xgboost as xgb



# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


test.head()


train.head()


train.info()


cat_columns = [i for i in train.columns if train[i].dtype == np.object_]
num_columns = [i for i in train.columns if i not in cat_columns]


train.duplicated().sum()


train.describe()


test.describe()


# Encode categorical variables using LabelEncoder
# Machine learning models like XGBoost require all input features to be numeric.
# Therefore, we use LabelEncoder to convert categorical string labels into numeric values.

le_soil = LabelEncoder()  # Encoder for soil type
le_crop = LabelEncoder()  # Encoder for crop type
le_fert = LabelEncoder()  # Encoder for fertilizer type



train['Soil Type'] = le_soil.fit_transform(train['Soil Type'])
test['Soil Type'] = le_soil.transform(test['Soil Type'])

train['Crop Type'] = le_crop.fit_transform(train['Crop Type'])
test['Crop Type'] = le_crop.transform(test['Crop Type'])

train['Fertilizer Name'] = le_fert.fit_transform(train['Fertilizer Name'])


# Feature selection
# Selecting relevant input variables (features) for training the model

features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
# This list defines which columns will be used as input features for the model

X = train[features]
# Extracting the feature columns from the training dataset and storing them in X (input matrix)

y = train['Fertilizer Name']
# Extracting the target variable (label) from the training dataset â€” what we want to predict

X_test = test[features]
# Extracting the same feature columns from the test dataset to use for making predictions later



# Optional: reduce size for faster training
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=50000, stratify=y, random_state=42)


# Importing required libraries
from sklearn.metrics import log_loss  # For evaluating model performance with multi-class log loss
from sklearn.model_selection import StratifiedKFold  # For stratified k-fold cross-validation


# XGBoost model hyperparameters (with GPU support in v2.0+)
xgb_params = {
    'tree_method': 'hist',        # Efficient histogram-based algorithm (required for GPU in newer versions)
    'device': 'cuda',             # Use GPU for training to speed up computation
    'n_estimators': 500,          # Number of trees to build (updated from 100 to 500)
    'max_depth': 10,              # Maximum depth of each tree (controls model complexity)
    'learning_rate': 0.05,        # Learning rate (step size shrinkage)
    'subsample': 0.8,             # Fraction of training samples to use for each tree
    'colsample_bytree': 0.8,      # Fraction of features to use for each tree
    'use_label_encoder': False,   # Avoid using the deprecated label encoder inside XGBoost
    'eval_metric': 'mlogloss',    # Multi-class log loss for evaluation
    'random_state': 42            # Seed for reproducibility
}
# Cross-validation
# Setting up 5-fold stratified cross-validation
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store validation log loss scores and trained models
scores = []
models = []

# Loop through each fold
for fold, (train_idx, val_idx) in enumerate(folds.split(X, y), 1):
    print(f"\nâœ… Fold {fold}")
    
    # Splitting data into training and validation sets for this fold
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Initialize XGBoost classifier with specified parameters
    model = xgb.XGBClassifier(**xgb_params)
    
    # Train the model using training set and evaluate on validation set
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Predict class probabilities on validation set
    y_pred = model.predict_proba(X_val)
    
    # Calculate log loss for the validation predictions
    score = log_loss(y_val, y_pred)
    print(f"ðŸ“‰ Fold {fold} Log Loss: {score:.5f}")
    
    # Save the score and trained model
    scores.append(score)
    models.append(model)

# Print average log loss across all folds
print(f"\nðŸ“Š Average Log Loss across folds: {np.mean(scores):.5f}")


# Use the last trained model to make predictions on the test set
final_model = models[-1]
probs = final_model.predict_proba(X_test)  # Get predicted probabilities for each class


# Get top 3 predictions
top_3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
top_3_labels = le_fert.inverse_transform(top_3.ravel()).reshape(top_3.shape)
submission_preds = [' '.join(row) for row in top_3_labels]



# Get top 3 predictions
top_3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
top_3_labels = le_fert.inverse_transform(top_3.ravel()).reshape(top_3.shape)
submission_preds = [' '.join(row) for row in top_3_labels]


# Submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': submission_preds
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")




