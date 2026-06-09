import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report
from catboost import CatBoostClassifier
from xgboost import  XGBClassifier


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
train


print(f"Train shape : {train.shape}")
print(f"Test shape : {test.shape}")


train.columns.tolist()
train.info()


numerical_int_feat = ['age',]
numerical_float_feat = [
     'sleep_hours_per_day','screen_time_hours_per_day', 'waist_to_hip_ratio','bmi',
    'alcohol_consumption_per_week','physical_activity_minutes_per_week',
            'systolic_bp','diastolic_bp','cholesterol_total','hdl_cholesterol',
                      'ldl_cholesterol', 'family_history_diabetes',
]

categorical_label_feat = [
    'smoking_status','income_level','employment_status','education_level'
]
categorical_oneHot_feat = ['ethnicity']
# 'gender',''hypertension_history','cardiovascular_history','diet_score',
target = 'diagnosed_diabetes'


stand_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
ordinal_encode = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
oneHot_encode = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers = [
        ('num_standard', stand_scaler, numerical_int_feat),
        ('num_minmax', minmax_scaler, numerical_float_feat),
        ('cat_ordinal', ordinal_encode,categorical_label_feat),
        ('cat_onehot',oneHot_encode,categorical_oneHot_feat)
    ]
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])



transformed_data = pipeline.fit_transform(train)
X = transformed_data
Y = train['diagnosed_diabetes']
print(f"X shape after preprocessing : {X.shape}")
print(f"Y shape after preprocessing : {Y.shape}")


# feature_cols = ['age','alcohol_consumption_per_week','physical_activity_minutes_per_week',
#  'diet_score','sleep_hours_per_day','screen_time_hours_per_day','bmi',
#  'waist_to_hip_ratio','systolic_bp','diastolic_bp','heart_rate','cholesterol_total','hdl_cholesterol',
#  'ldl_cholesterol','triglycerides','family_history_diabetes',
#  'hypertension_history','cardiovascular_history',]

# X = train[feature_cols].copy()
# scaler = MinMaxScaler()

# X_scaled = scaler.fit_transform(X)
# Y = train['diagnosed_diabetes']
# X_scaled


from imblearn.over_sampling import SMOTE


print(f"Original : {np.bincount(Y)}")

smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X,Y)




print(f"SMOTE : {np.bincount(Y_resampled)}")
print(f"shape of X_resampled: {X_resampled.shape}")
print(f"shape of Y_resampled: {Y_resampled.shape}")


X_train, X_test, Y_train, Y_test = train_test_split(
    X_resampled, Y_resampled, test_size=0.20, random_state=42,
)
print(f" Train split: {X_train.shape}")
print(f" Test split: {X_test.shape}")


model  = XGBClassifier(
    objective = "binary:logistic",
    n_estimators = 5000,
    learning_rate = 0.01,
    max_depth = 8,
    random_state = 42
)

model.fit(
    X_train, Y_train
)

train_auc = roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1])
test_auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])


# model  = CatBoostClassifier(
#     iterations=5000,           # Max trees
#     learning_rate=0.01,        # Conservative LR
#     depth=8,                   # Tree depth
#     l2_leaf_reg=4,             # Regularization
#     bagging_temperature=0.1,   # Randomness
#     random_strength=1.0,       # Bootstrap
#     border_count=250,          # Feature splits
#     eval_metric='AUC',         # Track AUC
#     early_stopping_rounds=100, # Stop if no improvement
#     verbose=100,
#     random_seed=42
# )

# model.fit(
#     X_train, Y_train, 
#     eval_set =(X_test, Y_test),
#     use_best_model=True,       # Use best iteration
# )

# train_auc = roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1])
# test_auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])


# from sklearn.svm import SVC
# model = SVC(
#     kernel='linear',      # linear Function (default) for binary classification
#     C=1.0,             # Regularization
#     gamma='scale',     # Kernel coefficient
#     probability=True,  # For predict_proba
#     random_state=42
# )

# model.fit(X_train, Y_train)
# train_auc = roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1])
# test_auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])


print(f"\nRESULTS:")
print(f"Train AUC: {train_auc:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print("\n Classification Report:")
print(classification_report(Y_test, model.predict(X_test)))


test_transformed = pipeline.fit_transform(test)
test_pred = model.predict_proba(test_transformed)[:, 1]


submission = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': test_pred
})

print(f"Submission shape: {submission.shape}")
print(f"\nFirst 5 rows:")
display(submission.head())

print(f"\nPrediction statistics:")
print(submission['diagnosed_diabetes'].describe())

# Save
submission.to_csv('submission.csv', index=False)
print(f"\nSubmission saved to submission.csv")

# Verify format
print(f"\n Format verification:")
print(f"  Columns: {submission.columns.tolist()}")
print(f"  Shape matches sample_submission: {submission.shape == sample_submission.shape}")
print(f"  IDs match: {(submission['id'] == sample_submission['id']).all()}")




