# Imports
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42



# 1) Load data
TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'

train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

print('train shape:', train_df.shape)
print('test shape :', test_df.shape)
display(train_df.head())
display(test_df.head())



# 2) Define features/target
TARGET_COL = 'diagnosed_diabetes'
ID_COL = 'id'

X = train_df.drop(columns=[TARGET_COL])
y = train_df[TARGET_COL].astype(int)

print('Target positive rate:', y.mean())



# 3) Identify numeric vs categorical columns
# NOTE: Some datasets store categoricals as strings; others store them as integer codes.
# We handle both by explicitly listing the known categorical feature names.
known_categoricals = [
    'gender',
    'ethnicity',
    'education_level',
    'income_level',
    'smoking_status',
    'employment_status',
]

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
for c in known_categoricals:
    if c in X.columns and c not in categorical_cols:
        categorical_cols.append(c)

numeric_cols = [c for c in X.columns if c not in categorical_cols]

print('Categorical columns:', categorical_cols)
print('Numeric columns     :', numeric_cols)



# 4) Build a baseline pipeline
# - Impute missing values
# - One-hot encode categoricals (handle unseen categories in test)
# - Scale numerics (with_mean=False so it stays compatible with sparse matrices)
# - Train Logistic Regression (fast, strong baseline for tabular problems)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler(with_mean=False)),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ],
    remainder='drop',
)

model = LogisticRegression(
    solver='saga',
    max_iter=200,
    n_jobs=-1,
)

clf = Pipeline(steps=[('preprocess', preprocess), ('model', model)])



# 5) Quick local validation (train/valid split)
# This gives you a rough idea of model quality before you train on all data.

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

clf.fit(X_train, y_train)
valid_pred = clf.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, valid_pred)
print(f'Validation AUC: {auc:.5f}')



# 6) Train on full training data and write submission.csv
clf.fit(X, y)
test_pred = clf.predict_proba(test_df)[:, 1]

submission = pd.DataFrame({
    ID_COL: test_df[ID_COL],
    TARGET_COL: test_pred,
})

# Sanity checks
assert submission.columns.tolist() == [ID_COL, TARGET_COL]
assert submission[TARGET_COL].between(0, 1).all()

SUBMISSION_PATH = 'submission.csv'
submission.to_csv(SUBMISSION_PATH, index=False)
print('Wrote', SUBMISSION_PATH, 'with shape', submission.shape)
display(submission.head())


