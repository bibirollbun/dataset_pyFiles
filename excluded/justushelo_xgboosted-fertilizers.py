import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns



# Insert csv files into dataframe
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
train= pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
# Set Index
test = test.set_index('id')
train = train.set_index('id')

# Check categorical values .unique()
#train['Soil Type'].unique()
train['Crop Type'].unique()


# Categorical columns 
categorical_cols = ['Soil Type', 'Crop Type']
target_col = 'Fertilizer Name'

# Prepare X and y
X = train.drop(columns=[target_col]).copy()
y = train[target_col].copy()
X_test = test.copy()

# OneHotEncoder
ohe = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols)
    ],
    remainder='passthrough'
)

# LabelEncoder for target
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)
num_classes = len(np.unique(y_encoded))


# K-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X), num_classes))
models = []

# Building the metric for competition
def mapk(y_true, y_proba, k=3):
    top_k_preds = np.argsort(y_proba, axis=1)[:, ::-1][:, :k]
    return np.mean([1 if y in preds else 0 for y, preds in zip(y_true, top_k_preds)])

# Building the pipeline inside K-fold Cross-validation
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    print(f"Fold {fold + 1}")
    X_train, y_train = X.iloc[train_idx], y_encoded[train_idx]
    X_val, y_val = X.iloc[val_idx], y_encoded[val_idx]

    # Pipeline 
    pipeline = Pipeline([
        ('preprocessing', ohe),
        ('model', XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss',
            learning_rate=0.05,
            max_depth=6,
            n_estimators=100,
            tree_method='hist',
            verbosity=0,
            reg_alpha=0.2,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
        ))
    ])

    pipeline.fit(X_train, y_train)

    val_preds = pipeline.predict_proba(X_val)
    oof_preds[val_idx] = val_preds
    fold_score = mapk(y_val, val_preds, k=3)
    print(f"Fold {fold + 1} MAP@3: {fold_score:.4f}")

    models.append(pipeline)

# Overall MAP@3
map3_score = mapk(y_encoded, oof_preds, k=3)
print(f"\nOverall MAP@3: {map3_score:.4f}")


# Evaluating overfitting
full_train_preds = pipeline.predict_proba(X)
full_train_map3 = mapk(y_encoded, full_train_preds, k=3)
print(f"Full Train MAP@3: {full_train_map3:.4f}")


# Train a model on the full dataset for overfitting comparison
full_pipeline = Pipeline([
        ('preprocessing', ohe),
        ('model', XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss',
            learning_rate=0.05,
            max_depth=6,
            n_estimators=100,
            tree_method='hist',
            verbosity=0,
            reg_alpha=0.2,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
        ))
    ])
full_pipeline.fit(X, y_encoded)
full_train_preds = full_pipeline.predict_proba(X)
full_train_map3 = mapk(y_encoded, full_train_preds, k=3)
print(full_train_map3)




# Test data perdictions
test_preds_proba = full_pipeline.predict_proba(X_test)

# Top-3 preds
top3_preds = np.argsort(test_preds_proba, axis=1)[:, ::-1][:, :3]

# Decoding the classes
top3_labels = target_le.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)

# Formatting the labels into submission form
top3_str = [' '.join(row) for row in top3_labels]



fullpipeline_submission_df = pd.DataFrame({
    'id': X_test.index,
    'Fertilizer Name': top3_str
})
fullpipeline_submission_df.to_csv('fullpipeline_submission.csv', index=False)
fullpipeline_submission_df.head()


fullpipeline_submission_df.shape


# Ensemble preds
test_pred_probas = np.zeros((len(X_test), num_classes))

for model in models:
    test_pred_probas += model.predict_proba(X_test)

# Mean
test_pred_probas /= len(models)

# Predicted classes
top3_preds = np.argsort(test_pred_probas, axis=1)[:, ::-1][:, :3]

# Inverse LabelEncode predicted classes back
top3_labels = target_le.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)

# Format submission
top3_str = [' '.join(row) for row in top3_labels]

# Submission dataframe
ensemble_submission_df = pd.DataFrame({
    'id': X_test.index,
    'Fertilizer Name': top3_str
})

# Save submission
ensemble_submission_df.to_csv('ensemble_submission.csv', index=False)



ensemble_submission_df.shape


ensemble_submission_df.head()

