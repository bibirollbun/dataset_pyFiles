import numpy as np
import pandas as pd
import os

# List files in the dataset directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def get_feats(mode='TRAIN'):
    """
    Load data for the specified mode (TRAIN or TEST).
    """
    # Load quantitative metadata
    feats = pd.read_excel(f"/kaggle/input/widsdatathon2025/{mode}/{mode}_QUANTITATIVE_METADATA.xlsx")  # Use Excel format
    
    # Load categorical metadata
    if mode == 'TRAIN':
        cate = pd.read_excel(f"/kaggle/input/widsdatathon2025/{mode}/{mode}_CATEGORICAL_METADATA.xlsx")  # Use Excel format
    else:
        cate = pd.read_excel(f"/kaggle/input/widsdatathon2025/{mode}/{mode}_CATEGORICAL.xlsx")  # Use Excel format
    
    # Merge quantitative and categorical data
    feats = pd.merge(feats, cate, on='participant_id', how='left')
    
    # Load functional connectome matrices
    func = pd.read_csv(f"/kaggle/input/widsdatathon2025/{mode}/{mode}_FUNCTIONAL_CONNECTOME_MATRICES.csv")
    feats = pd.merge(feats, func, on='participant_id', how='left')
    
    # Load training solutions (only for TRAIN mode)
    if mode == 'TRAIN':
        solution = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx")  # Use Excel format
        feats = pd.merge(feats, solution, on='participant_id', how='left')
    
    return feats

# Load training and test data
print("Loading data...")
train = get_feats(mode='TRAIN')
test = get_feats(mode='TEST')

# Display the first few rows of the training data
train.head()


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Separate features and target variables
X = train.drop(['participant_id', 'ADHD_Outcome', 'Sex_F'], axis=1, errors='ignore')
y_adhd = train['ADHD_Outcome']
y_sex = train['Sex_F']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))  # Use sparse matrices for efficiency
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Apply preprocessing to training data
print("Preprocessing data...")
X_preprocessed = preprocessor.fit_transform(X)


# Example: Create interaction features
X_preprocessed = pd.DataFrame(X_preprocessed)  # Convert NumPy array to DataFrame
X_preprocessed['interaction_feature'] = X_preprocessed[0] * X_preprocessed[1]  # Example interaction

# Display the first few rows of the preprocessed data
X_preprocessed.head()


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Split data into training and validation sets
X_train, X_val, y_train_adhd, y_val_adhd, y_train_sex, y_val_sex = train_test_split(
    X_preprocessed, y_adhd, y_sex, test_size=0.2, random_state=42, stratify=y_sex
)

# Calculate class weights for Sex_F
sex_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_sex), y=y_sex)
sex_scale_pos_weight = sex_weights[1] / sex_weights[0]  # Ratio of negative to positive class

# Define LightGBM models
adhd_model = lgb.LGBMClassifier(
    objective='binary',
    num_leaves=63,  # Reduced complexity for faster training
    learning_rate=0.05,  # Increased learning rate for faster convergence
    n_estimators=500,  # Reduced number of trees
    scale_pos_weight=1.0,  # No imbalance for ADHD
    early_stopping_rounds=None,  # Disable early stopping for now
    verbose=-1
)

sex_model = lgb.LGBMClassifier(
    objective='binary',
    num_leaves=127,  # Reduced complexity for faster training
    learning_rate=0.05,  # Increased learning rate for faster convergence
    n_estimators=500,  # Reduced number of trees
    scale_pos_weight=sex_scale_pos_weight,  # Apply class weight
    early_stopping_rounds=None,  # Disable early stopping for now
    verbose=-1
)

# Train ADHD model
print("Training ADHD model...")
adhd_model.fit(X_train, y_train_adhd)

# Train Sex model
print("Training Sex model...")
sex_model.fit(X_train, y_train_sex)


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define hyperparameter grid for ADHD model
adhd_param_dist = {
    'num_leaves': randint(31, 127),  # Reduced range
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': randint(100, 500)  # Reduced range
}

# Perform randomized search for ADHD model
adhd_random_search = RandomizedSearchCV(adhd_model, adhd_param_dist, n_iter=5, cv=3, scoring='f1', n_jobs=-1, random_state=42)
adhd_random_search.fit(X_train, y_train_adhd)

# Update ADHD model with best parameters
adhd_model = adhd_random_search.best_estimator_

# Define hyperparameter grid for Sex model
sex_param_dist = {
    'num_leaves': randint(63, 255),  # Reduced range
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': randint(100, 500)  # Reduced range
}

# Perform randomized search for Sex model
sex_random_search = RandomizedSearchCV(sex_model, sex_param_dist, n_iter=5, cv=3, scoring='f1', n_jobs=-1, random_state=42)
sex_random_search.fit(X_train, y_train_sex)

# Update Sex model with best parameters
sex_model = sex_random_search.best_estimator_


from sklearn.metrics import f1_score

# Make predictions on the validation set
adhd_pred = adhd_model.predict(X_val)
sex_pred = sex_model.predict(X_val)

# Calculate F1 scores
adhd_f1 = f1_score(y_val_adhd, adhd_pred)
sex_f1 = f1_score(y_val_sex, sex_pred)
combined_f1 = (adhd_f1 + sex_f1) / 2

print(f"ADHD F1 Score: {adhd_f1:.4f}")
print(f"Sex F1 Score: {sex_f1:.4f}")
print(f"Combined F1 Score: {combined_f1:.4f}")


# Preprocess test data
test_preprocessed = preprocessor.transform(test.drop('participant_id', axis=1, errors='ignore'))

# Make predictions
test_adhd_pred = adhd_model.predict(test_preprocessed)
test_sex_pred = sex_model.predict(test_preprocessed)

# Create submission file
submission = pd.DataFrame({
    'participant_id': test['participant_id'],
    'ADHD_Outcome': test_adhd_pred,
    'Sex_F': test_sex_pred
})

# Save submission file
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file saved!")

