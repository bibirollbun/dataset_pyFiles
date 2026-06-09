#imports
import pandas as pd
import numpy as np
import random
from lightgbm import LGBMClassifier
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns


# Load main data
train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
demographics_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')

# Merge on 'subject'
merged_df = train_df.merge(demographics_df, on='subject', how='left')

# Check the result
print("Merged shape:", merged_df.shape)
print(merged_df[['subject', 'age', 'sex', 'gesture']].head())


# Identify sensor columns
imu_cols = [col for col in merged_df.columns if col.startswith('acc_') or col.startswith('rot_')]
thm_cols = [col for col in merged_df.columns if col.startswith('thm_')]
tof_cols = [col for col in merged_df.columns if col.startswith('tof_')]

# Create a helper column: does this row have ToF data?
merged_df['has_tof_data'] = (merged_df[tof_cols] != -1).any(axis=1)

# Group by sequence_id
grouped = merged_df.groupby('sequence_id')


# Function to extract stats
def extract_features(group):
    features = {}
    for col in imu_cols + thm_cols + tof_cols:
        if col in group:
            features[f'{col}_mean'] = group[col].mean()
            features[f'{col}_std'] = group[col].std()
            features[f'{col}_min'] = group[col].min()
            features[f'{col}_max'] = group[col].max()
    # Add gesture and subject-level info (demographics)
    features['gesture'] = group['gesture'].iloc[0]
    features['subject'] = group['subject'].iloc[0]
    features['has_tof_data'] = group['has_tof_data'].any()
    for demo_col in ['age', 'sex', 'handedness', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm', 'adult_child']:
        features[demo_col] = group[demo_col].iloc[0]
    return pd.Series(features)

# Apply the feature extraction
sequence_features_df = grouped.apply(extract_features).reset_index()

print("Final feature table shape:", sequence_features_df.shape)
print(sequence_features_df[['sequence_id', 'gesture', 'has_tof_data']].head())


# 1. Missing value check
missing_counts = sequence_features_df.isnull().sum()
print("Missing values per column:\n", missing_counts[missing_counts > 0])

# 2. Class distribution
print("\nGesture distribution:")
print(sequence_features_df['gesture'].value_counts())


# Set seed for reproducibility
random.seed(42)

# Copy full dataset
sequence_features_simulated = sequence_features_df.copy()

# Identify all ToF feature columns
tof_feature_cols = [col for col in sequence_features_df.columns if col.startswith('tof_')]

# Randomly select 50% of sequence_ids
sequence_ids = sequence_features_simulated['sequence_id'].unique().tolist()
num_to_mask = len(sequence_ids) // 2
masked_sequence_ids = set(random.sample(sequence_ids, num_to_mask))

# Mask ToF features by setting to -1
sequence_features_simulated.loc[
    sequence_features_simulated['sequence_id'].isin(masked_sequence_ids),
    tof_feature_cols
] = -1

# Update has_tof_data flag
sequence_features_simulated['has_tof_data'] = ~sequence_features_simulated['sequence_id'].isin(masked_sequence_ids)

# First, reserve holdout test set (e.g., 15%) from the full simulated data
train_val_df, test_holdout_df = train_test_split(
    sequence_features_simulated,
    test_size=0.15,
    stratify=sequence_features_simulated['gesture'],
    random_state=42
)


# Create two sets
with_tof_df = train_val_df[train_val_df['has_tof_data']]
without_tof_df = train_val_df[~train_val_df['has_tof_data']]


# For With-ToF data
train_with_tof, val_with_tof = train_test_split(
    with_tof_df,
    test_size=0.2,
    stratify=with_tof_df['gesture'],
    random_state=42
)

# For Without-ToF data
train_without_tof, val_without_tof = train_test_split(
    without_tof_df,
    test_size=0.2,
    stratify=without_tof_df['gesture'],
    random_state=42
)

print(f"Train with ToF: {len(train_with_tof)}, Val with ToF: {len(val_with_tof)}")
print(f"Train without ToF: {len(train_without_tof)}, Val without ToF: {len(val_without_tof)}")



print("With ToF Gesture Distribution (Train):")
print(train_with_tof['gesture'].value_counts(normalize=True))

print("\nWithout ToF Gesture Distribution (Train):")
print(train_without_tof['gesture'].value_counts(normalize=True))



imu_base = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
thermo_base = [f'thm_{i}' for i in range(1, 6)]
tof_base = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
stats = ['mean', 'std', 'min', 'max']
imu_cols = [f"{col}_{stat}" for col in imu_base for stat in stats]
thermo_cols = [f"{col}_{stat}" for col in thermo_base for stat in stats]
tof_cols = [f"{col}_{stat}" for col in tof_base for stat in stats]
demo_cols = [
    'age', 'sex', 'handedness', 'height_cm',
    'shoulder_to_wrist_cm', 'elbow_to_wrist_cm', 'adult_child'
]


# Combine all columns per model type
features_with_tof = imu_cols + thermo_cols + tof_cols + demo_cols
features_without_tof = imu_cols + thermo_cols + demo_cols


le = LabelEncoder()
train_with_tof['gesture_encoded'] = le.fit_transform(train_with_tof['gesture'])
val_with_tof['gesture_encoded'] = le.transform(val_with_tof['gesture'])

train_without_tof['gesture_encoded'] = le.transform(train_without_tof['gesture'])
val_without_tof['gesture_encoded'] = le.transform(val_without_tof['gesture'])


def clean_tof(df, tof_columns):
    for col in tof_columns:
        if(df[col].isnull().any() or (df[col]==-1).any()):
            mean = df[df[col]!=-1][col].mean()
            df[col] = df[col].replace(-1,mean)
    return df
    
# Already aggregated â€” no cleaning needed here
# train_with_tof = clean_tof(train_with_tof, tof_cols)
# val_with_tof = clean_tof(val_with_tof, tof_cols)


# ML models like LightGBM don't require scaling â€” but it's helpful if you later use models like SVM, MLP
# We'll add it for completeness, especially for demographic values with different scales
scaler_with_tof = StandardScaler()
scaler_without_tof = StandardScaler()

X_train_with_tof = scaler_with_tof.fit_transform(train_with_tof[features_with_tof])
X_val_with_tof = scaler_with_tof.transform(val_with_tof[features_with_tof])

X_train_without_tof = scaler_without_tof.fit_transform(train_without_tof[features_without_tof])
X_val_without_tof = scaler_without_tof.transform(val_without_tof[features_without_tof])

Y_train_with_tof = train_with_tof['gesture_encoded']
Y_val_with_tof = val_with_tof['gesture_encoded']

Y_train_without_tof = train_without_tof['gesture_encoded']
Y_val_without_tof = val_without_tof['gesture_encoded']


model_with_tof = LGBMClassifier(
    objective='multiclass',
    num_class=len(le.classes_),
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

model_without_tof = LGBMClassifier(
    objective='multiclass',
    num_class=len(le.classes_),
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)


print("Training model WITH ToF...")
model_with_tof.fit(X_train_with_tof, Y_train_with_tof)

print("Training model WITHOUT ToF...")
model_without_tof.fit(X_train_without_tof, Y_train_without_tof)


print("\nğŸ”� Evaluation - Model WITH ToF:")
preds_with_tof = model_with_tof.predict(X_val_with_tof)
print(classification_report(Y_val_with_tof, preds_with_tof, target_names=le.classes_))

print("\nğŸ”� Evaluation - Model WITHOUT ToF:")
preds_without_tof = model_without_tof.predict(X_val_without_tof)
print(classification_report(Y_val_without_tof, preds_without_tof, target_names=le.classes_))


# Create a directory to store saved models
os.makedirs("saved_models/baseline/lgbm", exist_ok=True)

# Save the model WITH ToF
joblib.dump(model_with_tof, "saved_models/baseline/lgbm/model_with_tof.pkl")
joblib.dump(scaler_with_tof, "saved_models/baseline/lgbm/scaler_with_tof.pkl")
joblib.dump(features_with_tof, "saved_models/baseline/lgbm/features_with_tof_cols.pkl")

# Save the model WITHOUT ToF
joblib.dump(model_without_tof, "saved_models/baseline/lgbm/model_without_tof.pkl")
joblib.dump(scaler_without_tof, "saved_models/baseline/lgbm/scaler_without_tof.pkl")
joblib.dump(features_without_tof, "saved_models/baseline/lgbm/features_without_tof_cols.pkl")

print("âœ… All models and pipeline components saved to 'saved_models/' directory.")



def plot_feature_importance(model, feature_names, title, top_n=25):
    importance = model.feature_importances_
    indices = np.argsort(importance[-top_n:])
    
    plt.figure(figsize=(10,8))
    plt.title(title)
    plt.barh(range(top_n), importance[indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel("Importance score")
    plt.tight_layout()
    plt.show()

# Plot for model WITH ToF
plot_feature_importance(model_with_tof, features_with_tof, "Top Features - Model WITH ToF")

# Plot for model WITHOUT ToF
plot_feature_importance(model_without_tof, features_without_tof, "Top Features - Model WITHOUT ToF")


def plot_confusion(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Use this on your validation predictions
plot_confusion(Y_val_with_tof, preds_with_tof, le.classes_, "Confusion Matrix - With tof-")
plot_confusion(Y_val_without_tof, preds_without_tof, le.classes_, "Confusion Matrix - Without tof-")


rf_params = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10,20, None],
    'classifier__min_samples_split': [2,5,10],
    'classifier__min_samples_leaf': [1,2,4]
}

"""for quick execution of notebook I suggest you not to use Gradient Boosting, 
the performance metrics of random forest and gradient boosting were similar."""
gb_params = {
    'classifier__n_estimators': [100,200],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__max_depth': [3,5,7]
}

xgb_params = {
    'classifier__n_estimators': [100,200],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__max_depth': [3,5,7],
    'classifier__sub_sample': [0.8,1.0],
}


def tune_model(model, param_grid ,x_train, y_train, x_val, y_val, model_name):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', model)
    ])

    search = RandomizedSearchCV(pipe, param_grid, n_iter=10, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(x_val)

    print(f"\nğŸ“Š {model_name} Report:\n")
    print(classification_report(y_val, y_pred))

    os.makedirs('saved_models/optimized', exist_ok=True)
    joblib.dump(best_model, f"saved_models/optimized/{model_name}_model.joblib")
    return best_model


rf_model = tune_model(RandomForestClassifier(random_state=42), rf_params, X_train_with_tof, Y_train_with_tof, X_val_with_tof, Y_val_with_tof, "RandomForest")


gb_model = tune_model(GradientBoostingClassifier(random_state=42), gb_params, X_train_with_tof, Y_train_with_tof, X_val_with_tof, Y_val_with_tof, "GradientBoosting")


xgb_model = tune_model(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), xgb_params, X_train_with_tof, Y_train_with_tof, X_val_with_tof, Y_val_with_tof, "XGBoost")


rf_model = joblib.load("saved_models/optimized/RandomForest_model.joblib")
gb_model = joblib.load("saved_models/optimized/GradientBoosting_model.joblib")
xgb_model = joblib.load("saved_models/optimized/XGBoost_model.joblib")


# Use soft voting (uses predicted probabilities)
voting_clf = VotingClassifier(
    estimators = [
        ('rf', rf_model.named_steps['classifier']),
        ('gb', gb_model.named_steps['classifier']),
        ('xgb', xgb_model.named_steps['classifier']),
    ],
    voting = 'soft'
)

final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('classifier', voting_clf)
])


final_pipeline.fit(X_train_with_tof, Y_train_with_tof)


y_pred_ensemble = final_pipeline.predict(X_val_with_tof)
print("\nğŸ”® Ensemble Model Report:\n")
print(classification_report(Y_val_with_tof, y_pred_ensemble))

joblib.dump(final_pipeline, "saved_models/optimized/Ensemble_model.joblib")

