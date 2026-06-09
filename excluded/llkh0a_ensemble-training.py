import numpy as np
import pandas as pd

# Read data
train = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')

print('Train shape:', train.shape)
print('Train demographics shape:', train_demo.shape)
print('Train columns:', train.columns.tolist())
print(train.head())


train.columns


# Check missing values
missing = train.isnull().sum()
print('Missing values per column:')
print(missing[missing > 0])

# Replace -1 in ToF columns with NaN for easier statistics
tof_cols = [col for col in train.columns if col.startswith('tof_')]
train[tof_cols] = train[tof_cols].replace(-1, np.nan)


def extract_features(df):
    feats = []
    # Only use numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove columns that should not be aggregated
    exclude_cols = ['row_id', 'sequence_id', 'sequence_counter', 'subject']
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    for seq_id, group in df.groupby('sequence_id'):
        feat = {'sequence_id': seq_id}
        for col in numeric_cols:
            feat[col + '_mean'] = group[col].mean()
            feat[col + '_std'] = group[col].std()
            feat[col + '_min'] = group[col].min()
            feat[col + '_max'] = group[col].max()
            feat[col + '_skew'] = group[col].skew()
            feat[col + '_kurt'] = group[col].kurt()
            feat[col + '_missing'] = group[col].isnull().sum()
        # New feature: total missing values in sequence
        feat['total_missing'] = group.isnull().sum().sum()
        # New feature: duration of sequence
        feat['duration'] = group['sequence_counter'].max() - group['sequence_counter'].min()
        # New feature: IMU signal energy
        for axis in ['acc_x', 'acc_y', 'acc_z']:
            if axis in group.columns:
                feat[axis + '_energy'] = (group[axis] ** 2).sum()
        feat['subject'] = group['subject'].iloc[0]
        feats.append(feat)
    return pd.DataFrame(feats)

X = extract_features(train)
# Merge demographic features (excluding subject key)
demographic_features = [col for col in train_demo.columns if col != 'subject']
X = X.merge(train_demo, on='subject', how='left')
# Fill any remaining missing values in features with column mean (or 0 as fallback)
X = X.fillna(X.mean(numeric_only=True)).fillna(0)
# Add demographic features to training set
feature_cols = [col for col in X.columns if col not in ['sequence_id', 'subject']]


target_gestures = [
    'Above ear - pull hair',
    'Cheek - pinch skin',
    'Eyebrow - pull hair',
    'Eyelash - pull hair',
    'Forehead - pull hairline',
    'Forehead - scratch',
    'Neck - pinch skin',
    'Neck - scratch',
]
non_target_gestures = [
    'Write name on leg',
    'Wave hello',
    'Glasses on/off',
    'Text on phone',
    'Write name in air',
    'Feel around in tray and pull out an object',
    'Scratch knee/leg skin',
    'Pull air toward your face',
    'Drink from bottle/cup',
    'Pinch knee/leg skin'
]

# Map all gestures not in target_gestures to 'non_target'
train['gesture_mapped'] = train['gesture'].apply(lambda x: x if x in target_gestures else 'non_target')
y = train.groupby('sequence_id')['gesture_mapped'].first().values


import joblib
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(y)
joblib.dump(le, 'label_encoder.joblib')
# Print label map for reference
label_map = {i: label for i, label in enumerate(le.classes_)}
print('Label map:', label_map)


from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
import joblib
import numpy as np

# Helper for binary F1
def to_binary(y, target_gestures, label_encoder):
    y_labels = label_encoder.inverse_transform(y)
    return [1 if gesture in target_gestures else 0 for gesture in y_labels]

# Custom scorer for competition metric
def competition_metric(y_true, y_pred):
    binary_f1 = f1_score(to_binary(y_true, target_gestures, le), to_binary(y_pred, target_gestures, le))
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return (binary_f1 + macro_f1) / 2

competition_scorer = make_scorer(competition_metric, greater_is_better=True)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X[feature_cols], y_enc, test_size=0.2, random_state=42, stratify=y_enc)




# Hyperparameter tuning for LightGBM
lgbm_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [15, 31, 63],
    'max_depth': [3, 5, 7, -1],
}
lgbm_search = RandomizedSearchCV(
    LGBMClassifier(),
    param_distributions=lgbm_params,
    n_iter=10,
    scoring=competition_scorer,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)
lgbm_search.fit(X_train, y_train)
print('Best LightGBM params:', lgbm_search.best_params_)
print('Best LightGBM score:', lgbm_search.best_score_)
lgbm_best = lgbm_search.best_estimator_




# Hyperparameter tuning for XGBoost
xgb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.85, 1.0],
    'colsample_bytree': [0.7, 0.85, 1.0],
}
xgb_search = RandomizedSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    param_distributions=xgb_params,
    n_iter=10,
    scoring=competition_scorer,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)
xgb_search.fit(X_train, y_train)
print('Best XGBoost params:', xgb_search.best_params_)
print('Best XGBoost score:', xgb_search.best_score_)
xgb_best = xgb_search.best_estimator_




# Hyperparameter tuning for CatBoost
cat_params = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [3, 5, 7],
    'l2_leaf_reg': [1, 3, 5, 7],
}
cat_search = RandomizedSearchCV(
    CatBoostClassifier(verbose=0),
    param_distributions=cat_params,
    n_iter=10,
    scoring=competition_scorer,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)
cat_search.fit(X_train, y_train)
print('Best CatBoost params:', cat_search.best_params_)
print('Best CatBoost score:', cat_search.best_score_)
cat_best = cat_search.best_estimator_




# Ensemble with best estimators
ensemble = VotingClassifier(estimators=[
    ('lgbm', lgbm_best),
    ('xgb', xgb_best),
    ('cat', cat_best)
], voting='soft')

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_val)
print('Validation accuracy:', accuracy_score(y_val, y_pred))

# Save the ensemble model to a file
joblib.dump(ensemble, 'ensemble_model.joblib')
print('Model saved as ensemble_model.joblib')


from sklearn.metrics import f1_score
# Calculate competition metrics
y_true = y_val
binary_f1 = f1_score(to_binary(y_true, target_gestures, le), to_binary(y_pred, target_gestures, le))
macro_f1 = f1_score(y_true, y_pred, average='macro')
print('Binary F1 score:', binary_f1)
print('Macro F1 score:', macro_f1)
competition_metric_score = competition_metric(y_val, y_pred)

print('Competition metric score on validation set:', competition_metric_score)

