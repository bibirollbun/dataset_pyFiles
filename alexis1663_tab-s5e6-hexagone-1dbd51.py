import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer

from lightgbm import LGBMClassifier

# Custom MAP@3 scorer
def apk(actual, predicted, k=3):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score

def map3(y_true, y_pred_prob):
    top_3 = np.argsort(y_pred_prob, axis=1)[:, -3:][:, ::-1]
    return np.mean([apk(a, list(p), 3) for a, p in zip(y_true, top_3)])

map3_scorer = make_scorer(map3, needs_proba=True)

# Load data
base = "/kaggle/input/playground-series-s5e6/"
train = pd.read_csv(os.path.join(base, "train.csv"))
test = pd.read_csv(os.path.join(base, "test.csv"))
sub = pd.read_csv(os.path.join(base, "sample_submission.csv"))

# Features and target
target = "Fertilizer Name"
cat_cols = ["Soil Type", "Crop Type"]
num_cols = [c for c in train.columns if c not in cat_cols + ["id", target]]

X = train.drop(columns=["id", target])
y = train[target]
X_test = test.drop(columns=["id"])

# Encode target labels
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
y_enc = lbl.fit_transform(y)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.1, random_state=42, stratify=y_enc
)

# Preprocessing pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
])

# Build full pipeline
def build_pipeline(params=None):
    # dans build_pipeline ou directement dans LGBMClassifier :
    clf = LGBMClassifier(
        objective='multiclass',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        **(params or {}),
        n_estimators=1000,                    # nombre max d’arbres
        early_stopping_rounds=50,             # stop après 50 itérations sans amélioration
        verbose=-1
    )
    return Pipeline([
        ('pre', preprocessor),
        ('clf', clf)
    ])

# Hyperparameter search space
param_dist = {
    'clf__num_leaves': [31, 50, 100],
    'clf__max_depth': [-1, 10, 20, 30],
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__n_estimators': [100, 300, 500],
    'clf__subsample': [0.6, 0.8, 1.0],
    'clf__colsample_bytree': [0.6, 0.8, 1.0]
}

search = RandomizedSearchCV(
    build_pipeline(),
    param_distributions=param_dist,
    n_iter=20,
    scoring=map3_scorer,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Run hyperparameter tuning
search.fit(X_train, y_train)
best = search.best_estimator_
print("Best parameters:", search.best_params_)

# Evaluate on validation set
val_pred = best.predict_proba(X_val)
print("Validation MAP@3:", map3(y_val, val_pred))

# Retrain on full train set with best params
final_pipe = build_pipeline(search.best_params_)
final_pipe.fit(X, y_enc)

# Predict on test set and prepare submission
test_pred = final_pipe.predict_proba(X_test)
top_3 = np.argsort(test_pred, axis=1)[:, -3:][:, ::-1]
top_3_labels = lbl.inverse_transform(top_3.ravel()).reshape(top_3.shape)

submission = pd.DataFrame({
    'id': sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv("submission.csv", index=False)
print("Done. Submission saved.")





















"""
model = AdaBoostClassifier()
model.fit(X_train, y_train)

pred_val = model.predict(X_val)
pred = model.predict(X_test)
"""


"""
param_grid_reg =  {'num_leaves': [31], 'max_depth': [10], 'n_estimators':[1000], 'learning_rate':[0.3, 0.1, 0.03, 0.01]}
clf = LGBMClassifier(random_state=0, class_weight='balanced', force_col_wise=True)

gscv = GridSearchCV(estimator=clf, scoring="accuracy", cv=5, param_grid=param_grid_reg, verbose=5)
gscv.fit(X,y)

print("Score", gscv.best_score_)
print("Params", gscv.best_params_)
"""
















