import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# models
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# 1. Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


train.head()


train.info()


# 2. Separate Target and ID
target = 'diagnosed_diabetes'

# IDs are usually not useful for prediction
X = train.drop(columns=['id', target])
y = train[target]
X_test = test.drop(columns=['id'])


# 3. Preprocessing
# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Create transformers
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# 4. Define Model
ratio = float(np.sum(y == 0)) / np.sum(y == 1) 

model = xgb.XGBClassifier(
    n_estimators=1000,           # Reduced slightly to prevent overfitting
    learning_rate=0.03,          # Lower LR requires more trees, usually generalizes better
    max_depth=4,                 # 6 might be too deep for this data. Try 4 or 5.
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=ratio,
    
    # Regularization
    min_child_weight=3,          # Increase to 3 or 5 to stop learning "noise"
    gamma=0.1,                   # Slight increase to make splitting harder
    subsample=0.7,
    colsample_bytree=0.7,        # Lowering this forces diverse trees
    reg_alpha=0.5,               # L1 reg (good for feature selection)
    reg_lambda=1.5,              # L2 reg
    
    n_jobs=-1,
    random_state=42
)



# Create Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])


# Fit
# We use a simple hold-out here, but Cross-Validation (CV) is better
from sklearn.model_selection import train_test_split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train_split, y_train_split)


# Predict

val_preds = pipeline.predict_proba(X_val_split)[:, 1] # Get probability for class 1

print(f"Validation AUC: {roc_auc_score(y_val_split, val_preds)}")


# 6. Train on Full Data and Predict
pipeline.fit(X, y)
test_preds = pipeline.predict_proba(X_test)[:, 1]


# 7. Create Submission File
submission[target] = test_preds
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")

